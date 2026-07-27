"""Temporal landmark ensemble evaluation."""

from . import explainability as _explainability  # noqa: F401 - preserve pipeline stage order.
from .utils import (
    OrderedDict,
    Path,
    cloudpickle,
    json,
    jsonable,
    np,
    os,
    pd,
    slugify,
    zlib,
)
from .config import (
    LANDMARK_CUMULATIVE_PREDICTION_ENSEMBLE,
    LANDMARK_TEMPORAL_AUROC_TOLERANCE,
    LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES,
    LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES,
    MODEL_DIR,
    MODEL_VARIANT_ORIGINAL,
    OUTPUT_DIR,
    RUN_FINAL_MODELS,
    THRESHOLD_CANDIDATE_POLICY,
    THRESHOLD_MIN_SENSITIVITY,
    THRESHOLD_OPERATING_POINT_STATUS,
    THRESHOLD_TIE_BREAK_POLICY,
    VALIDATION_MODEL_SELECTION_METRIC,
)
from .data import (
    ALL_FEATURE_SETS,
    LANDMARK_ORDER,
)
from .metrics import (
    CALIBRATION_DIAGNOSTIC_METRICS,
    apply_probability_recalibrator,
    classification_metrics,
    ensure_model_variant_column,
    evaluate_probability_source,
    file_sha256,
    fit_probability_recalibrator,
    probability_recalibrator_metadata,
    probability_source_provenance,
    select_threshold_from_oof,
    validation_model_selection_score,
)
from .splitting import optimization_settings_for_analysis
from .tuning import (
    finalize_selected_validation_strategy,
    select_strategy_by_validation,
    tune_hyperparameters,
)


def feature_source_candidates_for_landmark(current_feature_set):
    candidates = current_feature_set.get("candidate_variants") or [current_feature_set]
    return candidates


def cumulative_member_feature_sets(feature_set):
    if feature_set not in LANDMARK_ORDER:
        return [feature_set]
    return LANDMARK_ORDER[: LANDMARK_ORDER.index(feature_set) + 1]


TEMPORAL_ENSEMBLE_CANDIDATE_COLUMNS = [
    "target",
    "feature_set",
    "model_variant",
    "candidate_name",
    "members",
    "member_weights",
    "current_model_weight",
    "selection_split",
    "selection_auroc",
    "selection_auprc",
    "selection_prevalence",
    "selection_auprc_lift",
    "selection_score",
    "selected",
    "selection_rule",
    "test_labels_used_for_selection",
]


def target_uses_temporal_landmark_ensemble(target):
    target_family = str(target).split("_", 1)[0].lower()
    return target_family in {
        family.lower() for family in LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES
    }


def temporal_landmark_candidate_weights(members):
    member_count = len(members)
    if member_count < 2:
        return []

    candidates = OrderedDict()

    def add_candidate(name, weights):
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        key = tuple(np.round(weights, 12))
        if key not in candidates:
            candidates[key] = {
                "candidate_name": name,
                "weights": weights,
            }

    add_candidate("equal_cumulative_average", np.ones(member_count, dtype=float))
    add_candidate(
        "recency_weighted_cumulative_average",
        np.arange(1, member_count + 1, dtype=float),
    )
    for current_weight in LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES:
        prior_weight = (1.0 - current_weight) / (member_count - 1)
        weights = np.full(member_count, prior_weight, dtype=float)
        weights[-1] = current_weight
        add_candidate(
            f"current_weight_{current_weight:.3f}",
            weights,
        )
        preoperative_anchor_weights = np.zeros(member_count, dtype=float)
        preoperative_anchor_weights[0] = 1.0 - current_weight
        preoperative_anchor_weights[-1] = current_weight
        add_candidate(
            f"preoperative_anchor_current_weight_{current_weight:.3f}",
            preoperative_anchor_weights,
        )
    return list(candidates.values())


def write_temporal_landmark_wrapper(
    model_summary,
    target,
    feature_set,
    model_variant,
    landmark_day,
    members,
    weights,
    selected_candidate_name,
    selection_split,
    selection_rule,
    validation_recalibrator,
    selected_threshold,
    train_recalibrator=None,
    train_selected_threshold=None,
):
    """Persist a wrapper without mutating any underlying member artifact."""
    if not RUN_FINAL_MODELS:
        return None

    member_artifacts = []
    for member in members:
        mask = (
            model_summary["target"].astype(str).eq(str(target))
            & model_summary["feature_set"].astype(str).eq(str(member))
            & model_summary["model_variant"].astype(str).eq(str(model_variant))
        )
        matches = model_summary.loc[mask]
        if len(matches) != 1:
            raise RuntimeError(
                "Temporal wrapper requires exactly one base model summary row for "
                f"{target}/{member}/{model_variant}; found {len(matches)}."
            )
        member_row = matches.iloc[0]
        model_path = Path(
            str(member_row.get("single_model_model_path", member_row["model_path"]))
        )
        metadata_path = Path(
            str(
                member_row.get(
                    "single_model_metadata_path",
                    member_row["metadata_path"],
                )
            )
        )
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing temporal member artifact: {model_path} or {metadata_path}"
            )
        member_artifacts.append(
            {
                "feature_set": str(member),
                "base_model_relative_path": os.path.relpath(model_path, MODEL_DIR),
                "base_model_sha256": file_sha256(model_path),
                "base_metadata_relative_path": os.path.relpath(
                    metadata_path, MODEL_DIR
                ),
                "base_metadata_sha256": file_sha256(metadata_path),
                "base_model_path_at_creation": str(model_path),
                "base_metadata_path_at_creation": str(metadata_path),
            }
        )

    weight_map = OrderedDict(
        (str(member), float(weight)) for member, weight in zip(members, weights)
    )
    variant_suffix = (
        "" if model_variant == MODEL_VARIANT_ORIGINAL else f"__{slugify(model_variant)}"
    )
    artifact_slug = (
        f"{slugify(target)}__{slugify(feature_set)}{variant_suffix}"
        "__temporal_landmark_ensemble"
    )
    wrapper_path = MODEL_DIR / f"{artifact_slug}.pkl"
    metadata_path = MODEL_DIR / f"{artifact_slug}_metadata.json"
    bundle = {
        "artifact_type": "temporal_landmark_ensemble_wrapper",
        "target": target,
        "feature_set": feature_set,
        "model_variant": model_variant,
        "landmark_day": jsonable(landmark_day),
        "ordered_member_artifacts": member_artifacts,
        "member_artifact_path_resolution": "resolve_relative_to_wrapper_directory",
        "ordered_members": [str(member) for member in members],
        "member_weights": weight_map,
        "raw_score_rule": "sum(member_weight * member_raw_class1_probability)",
        "reported_probability_source": "temporal_landmark_raw",
        "validation_selection_recalibrator": validation_recalibrator,
        "validation_recalibrator_metadata": probability_recalibrator_metadata(
            validation_recalibrator
        ),
        "train_oof_diagnostic_recalibrator": train_recalibrator,
        "train_oof_recalibrator_metadata": (
            probability_recalibrator_metadata(train_recalibrator)
            if train_recalibrator is not None
            else None
        ),
        "threshold": float(selected_threshold["candidate_threshold"]),
        "threshold_objective": selected_threshold["candidate_objective"],
        "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
        "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
        "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
        "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
        "threshold_selection_split": "validation_temporal_landmark_ensemble",
        "threshold_probability_source_provenance": probability_source_provenance(
            "temporal_landmark_raw"
        ),
        "threshold_selection_labels_provenance": "validation_labels",
        "threshold_test_labels_used_for_selection": False,
        "train_oof_threshold": (
            float(train_selected_threshold["candidate_threshold"])
            if train_selected_threshold is not None
            else None
        ),
        "train_oof_threshold_objective": (
            train_selected_threshold["candidate_objective"]
            if train_selected_threshold is not None
            else None
        ),
        "selected_weight_candidate": selected_candidate_name,
        "weight_selection_split": selection_split,
        "weight_selection_rule": selection_rule,
        "weight_selection_test_labels_used": False,
        "test_labels_used_for_selection": False,
    }
    with open(wrapper_path, "wb") as handle:
        cloudpickle.dump(bundle, handle)
    metadata = {
        key: jsonable(value)
        for key, value in bundle.items()
        if key
        not in {
            "validation_selection_recalibrator",
            "train_oof_diagnostic_recalibrator",
        }
    }
    metadata["wrapper_path"] = str(wrapper_path)
    metadata["wrapper_relative_path"] = os.path.relpath(wrapper_path, OUTPUT_DIR)
    metadata["wrapper_path_resolution"] = "resolve_relative_to_output_directory"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=jsonable)
    return {
        "model_path": str(wrapper_path),
        "metadata_path": str(metadata_path),
        "model_relative_path": os.path.relpath(wrapper_path, OUTPUT_DIR),
        "metadata_relative_path": os.path.relpath(metadata_path, OUTPUT_DIR),
    }


def apply_cumulative_landmark_prediction_ensemble(
    holdout_results_df,
    holdout_predictions_df,
    holdout_threshold_candidates_df,
    holdout_train_cv_threshold_candidates_df,
    holdout_model_summary_df,
):
    if (
        not LANDMARK_CUMULATIVE_PREDICTION_ENSEMBLE
        or holdout_results_df.empty
        or holdout_predictions_df.empty
    ):
        return (
            holdout_results_df,
            holdout_predictions_df,
            holdout_threshold_candidates_df,
            holdout_train_cv_threshold_candidates_df,
            holdout_model_summary_df,
            pd.DataFrame(columns=TEMPORAL_ENSEMBLE_CANDIDATE_COLUMNS),
        )

    results = holdout_results_df.copy()
    predictions = holdout_predictions_df.copy()
    threshold_candidates = holdout_threshold_candidates_df.copy()
    train_threshold_candidates = holdout_train_cv_threshold_candidates_df.copy()
    model_summary = holdout_model_summary_df.copy()
    results = ensure_model_variant_column(results)
    predictions = ensure_model_variant_column(predictions)
    threshold_candidates = ensure_model_variant_column(threshold_candidates)
    train_threshold_candidates = ensure_model_variant_column(train_threshold_candidates)
    model_summary = ensure_model_variant_column(model_summary)
    added_threshold_rows = []
    added_train_threshold_rows = []

    for artifact_col in ["model_path", "metadata_path"]:
        if artifact_col in model_summary.columns:
            base_col = f"single_model_{artifact_col}"
            if base_col not in model_summary.columns:
                model_summary[base_col] = model_summary[artifact_col]

    for source_col in [
        "reported_probability_source",
        "raw_probability",
        "probability",
        "calibration_probability",
        "threshold",
        "y_pred",
    ]:
        if (
            source_col in predictions.columns
            and f"single_model_{source_col}" not in predictions.columns
        ):
            predictions[f"single_model_{source_col}"] = predictions[source_col]

    single_metric_prefixes = (
        "validation_",
        "test_",
        "raw_validation_",
        "raw_test_",
        "calibrated_validation_",
        "calibrated_test_",
    )
    for col in list(results.columns):
        if col.startswith(single_metric_prefixes) or col in {
            "reported_probability_source",
            "selected_threshold_objective",
            "selected_threshold",
            "selected_threshold_validation_score",
            "train_cv_selected_threshold_objective",
            "train_cv_selected_threshold",
            "train_cv_selected_threshold_score",
            "ensemble_train_cv_oof_auroc",
            "ensemble_train_cv_oof_auprc",
            "ensemble_train_cv_oof_auprc_lift",
            "validation_model_selection_score",
        }:
            single_col = f"single_model_{col}"
            if single_col not in results.columns:
                results[single_col] = results[col]

    results["probability_ensemble_method"] = "single_landmark_model"
    results["probability_ensemble_members"] = results["feature_set"]
    results["probability_ensemble_weights"] = results["feature_set"].map(
        lambda member: json.dumps({str(member): 1.0})
    )
    results["probability_ensemble_selected_candidate"] = "single_current_landmark"
    results["probability_ensemble_selection_split"] = "not_applicable"
    results["probability_ensemble_selection_rule"] = "not_applicable"
    results["probability_ensemble_test_labels_used_for_selection"] = False
    predictions["probability_ensemble_method"] = "single_landmark_model"
    predictions["probability_ensemble_members"] = predictions["feature_set"]
    predictions["probability_ensemble_weights"] = predictions["feature_set"].map(
        lambda member: json.dumps({str(member): 1.0})
    )
    predictions["probability_ensemble_selected_candidate"] = "single_current_landmark"
    predictions["probability_ensemble_selection_split"] = "not_applicable"
    predictions["probability_ensemble_test_labels_used_for_selection"] = False

    available_pairs = {
        (
            str(row["target"]),
            str(row["feature_set"]),
            str(row.get("model_variant", MODEL_VARIANT_ORIGINAL)),
        )
        for _, row in results.iterrows()
    }
    ensemble_candidate_rows = []

    for row_idx, row in results.iterrows():
        target = str(row["target"])
        feature_set = str(row["feature_set"])
        model_variant = str(row.get("model_variant", MODEL_VARIANT_ORIGINAL))
        if not target_uses_temporal_landmark_ensemble(target):
            continue
        members = [
            member
            for member in cumulative_member_feature_sets(feature_set)
            if (target, member, model_variant) in available_pairs
        ]
        if len(members) < 2 or members[-1] != feature_set:
            continue

        split_frames = {}
        for split_name in ["train_oof", "validation", "test"]:
            current_mask = (
                (predictions["target"] == target)
                & (predictions["feature_set"] == feature_set)
                & (predictions["dataset_split"] == split_name)
                & (predictions["model_variant"] == model_variant)
            )
            current = (
                predictions.loc[
                    current_mask,
                    ["source_row_index", "y_true"],
                ]
                .drop_duplicates("source_row_index")
                .copy()
            )
            if current.empty:
                if split_name == "train_oof":
                    continue
                split_frames = {}
                break
            current["_prediction_index"] = current.index
            wide = current
            used_members = []
            for member in members:
                member_mask = (
                    (predictions["target"] == target)
                    & (predictions["feature_set"] == member)
                    & (predictions["dataset_split"] == split_name)
                    & (predictions["model_variant"] == model_variant)
                )
                member_prob = (
                    predictions.loc[
                        member_mask,
                        ["source_row_index", "y_true", "single_model_raw_probability"],
                    ]
                    .drop_duplicates("source_row_index")
                    .rename(
                        columns={
                            "y_true": f"{member}__y_true",
                            "single_model_raw_probability": member,
                        }
                    )
                )
                if member_prob.empty:
                    continue
                wide = wide.merge(
                    member_prob,
                    on="source_row_index",
                    how="left",
                    validate="one_to_one",
                )
                used_members.append(member)

            member_label_columns = [f"{member}__y_true" for member in used_members]
            if (
                used_members != members
                or wide[members + member_label_columns].isna().any().any()
            ):
                if split_name == "train_oof":
                    continue
                split_frames = {}
                break
            labels_match = all(
                np.array_equal(
                    wide["y_true"].astype(int).to_numpy(),
                    wide[label_col].astype(int).to_numpy(),
                )
                for label_col in member_label_columns
            )
            if not labels_match:
                if split_name == "train_oof":
                    continue
                split_frames = {}
                break
            split_frames[split_name] = {
                "frame": wide,
                "members": used_members,
            }

        if {"validation", "test"} - set(split_frames):
            continue

        val_frame = split_frames["validation"]["frame"]
        test_frame = split_frames["test"]["frame"]
        used_members = split_frames["test"]["members"]
        selection_split = "train_oof" if "train_oof" in split_frames else "validation"
        selection_frame = split_frames[selection_split]["frame"]
        y_selection = selection_frame["y_true"].astype(int).to_numpy()
        y_val = val_frame["y_true"].astype(int).to_numpy()
        y_test = test_frame["y_true"].astype(int).to_numpy()
        if len(np.unique(y_selection)) < 2 or len(np.unique(y_val)) < 2:
            continue

        selection_member_probabilities = selection_frame[used_members].to_numpy(
            dtype=float
        )
        val_member_probabilities = val_frame[used_members].to_numpy(dtype=float)
        test_member_probabilities = test_frame[used_members].to_numpy(dtype=float)
        candidate_evaluations = []
        for candidate in temporal_landmark_candidate_weights(used_members):
            weights = candidate["weights"]
            candidate_selection_probability = selection_member_probabilities @ weights
            candidate_metrics = classification_metrics(
                y_selection,
                candidate_selection_probability,
                threshold=0.5,
            )
            candidate_evaluations.append(
                {
                    **candidate,
                    "selection_probability": candidate_selection_probability,
                    "validation_probability": val_member_probabilities @ weights,
                    "test_probability": test_member_probabilities @ weights,
                    "selection_metrics": candidate_metrics,
                    "selection_score": validation_model_selection_score(
                        candidate_metrics,
                        VALIDATION_MODEL_SELECTION_METRIC,
                    ),
                }
            )

        if not candidate_evaluations:
            continue
        maximum_selection_auroc = max(
            candidate["selection_metrics"]["auroc"]
            for candidate in candidate_evaluations
        )
        near_best_candidates = [
            candidate
            for candidate in candidate_evaluations
            if candidate["selection_metrics"]["auroc"]
            >= maximum_selection_auroc - LANDMARK_TEMPORAL_AUROC_TOLERANCE
        ]
        selected_candidate = max(
            near_best_candidates,
            key=lambda candidate: (
                candidate["selection_metrics"]["auprc"],
                candidate["selection_metrics"]["auroc"],
                candidate["weights"][-1],
            ),
        )
        selection_rule = (
            f"{selection_split}_max_auroc_then_max_auprc_within_"
            f"{LANDMARK_TEMPORAL_AUROC_TOLERANCE:.3f}_auroc"
        )
        for candidate in candidate_evaluations:
            metrics = candidate["selection_metrics"]
            candidate_weights = {
                member: float(weight)
                for member, weight in zip(used_members, candidate["weights"])
            }
            ensemble_candidate_rows.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "model_variant": model_variant,
                    "candidate_name": candidate["candidate_name"],
                    "members": "|".join(used_members),
                    "member_weights": json.dumps(candidate_weights, sort_keys=True),
                    "current_model_weight": float(candidate["weights"][-1]),
                    "selection_split": selection_split,
                    "selection_auroc": metrics["auroc"],
                    "selection_auprc": metrics["auprc"],
                    "selection_prevalence": metrics["prevalence"],
                    "selection_auprc_lift": metrics["auprc_lift"],
                    "selection_score": candidate["selection_score"],
                    "selected": candidate is selected_candidate,
                    "selection_rule": selection_rule,
                    "test_labels_used_for_selection": False,
                }
            )

        train_frame = (
            split_frames["train_oof"]["frame"] if "train_oof" in split_frames else None
        )
        train_raw_prob = (
            np.asarray(selected_candidate["selection_probability"], dtype=float)
            if selection_split == "train_oof"
            else None
        )
        train_diagnostic_calibrator = None
        train_cal_prob = None
        train_selected_threshold = None
        train_selected_candidate_rows = None
        if train_frame is not None and train_raw_prob is not None:
            y_train_oof = train_frame["y_true"].astype(int).to_numpy()
            train_diagnostic_calibrator = fit_probability_recalibrator(
                y_train_oof,
                train_raw_prob,
            )
            train_cal_prob = apply_probability_recalibrator(
                train_diagnostic_calibrator,
                train_raw_prob,
            )
            train_selected_threshold, train_selected_candidate_rows = (
                select_threshold_from_oof(
                    y_train_oof,
                    train_raw_prob,
                    calibration_prob=train_cal_prob,
                )
            )

        val_raw_prob = selected_candidate["validation_probability"]
        test_raw_prob = selected_candidate["test_probability"]

        diagnostic_calibrator = fit_probability_recalibrator(y_val, val_raw_prob)
        val_cal_prob = apply_probability_recalibrator(
            diagnostic_calibrator, val_raw_prob
        )
        test_cal_prob = apply_probability_recalibrator(
            diagnostic_calibrator, test_raw_prob
        )

        raw_eval = evaluate_probability_source(
            y_val,
            val_raw_prob,
            y_test,
            test_raw_prob,
            "temporal_landmark_raw",
        )
        calibrated_eval = evaluate_probability_source(
            y_val,
            val_cal_prob,
            y_test,
            test_cal_prob,
            "temporal_landmark_calibrated",
        )
        selected_report = raw_eval
        selected_threshold = selected_report["selected_threshold"]
        threshold = float(selected_report["threshold"])
        val_metrics = selected_report["validation_metrics"]  # noqa: F841
        test_metrics = selected_report["test_metrics"]  # noqa: F841
        raw_val_metrics = raw_eval["validation_metrics"]
        raw_test_metrics = raw_eval["test_metrics"]
        calibrated_val_metrics = calibrated_eval["validation_metrics"]
        calibrated_test_metrics = calibrated_eval["test_metrics"]
        member_label = "|".join(used_members)
        selected_weight_map = {
            member: float(weight)
            for member, weight in zip(used_members, selected_candidate["weights"])
        }
        selected_weight_label = json.dumps(selected_weight_map, sort_keys=True)

        temporal_prediction_splits = []
        if train_frame is not None and train_selected_threshold is not None:
            temporal_prediction_splits.append(
                (
                    "train_oof",
                    train_frame,
                    train_raw_prob,
                    train_cal_prob,
                    float(train_selected_threshold["candidate_threshold"]),
                )
            )
        temporal_prediction_splits.extend(
            [
                ("validation", val_frame, val_raw_prob, val_cal_prob, threshold),
                ("test", test_frame, test_raw_prob, test_cal_prob, threshold),
            ]
        )
        for (
            split_name,
            frame,
            raw_prob,
            cal_prob,
            split_threshold,
        ) in temporal_prediction_splits:
            pred_index = frame["_prediction_index"].to_numpy(dtype=int)
            predictions.loc[pred_index, "reported_probability_source"] = (
                "temporal_landmark_raw"
            )
            predictions.loc[pred_index, "raw_probability"] = raw_prob
            predictions.loc[pred_index, "probability"] = raw_prob
            predictions.loc[pred_index, "calibration_probability"] = cal_prob
            predictions.loc[pred_index, "threshold"] = split_threshold
            predictions.loc[pred_index, "y_pred"] = (
                raw_prob >= split_threshold
            ).astype(int)
            predictions.loc[pred_index, "probability_ensemble_method"] = (
                "cross_validated_temporal_landmark_convex_average"
            )
            predictions.loc[pred_index, "probability_ensemble_members"] = member_label
            predictions.loc[pred_index, "probability_ensemble_weights"] = (
                selected_weight_label
            )
            predictions.loc[pred_index, "probability_ensemble_selected_candidate"] = (
                selected_candidate["candidate_name"]
            )
            predictions.loc[pred_index, "probability_ensemble_selection_split"] = (
                selection_split
            )
            predictions.loc[
                pred_index,
                "probability_ensemble_test_labels_used_for_selection",
            ] = False

        results.loc[row_idx, "reported_probability_source"] = "temporal_landmark_raw"
        results.loc[row_idx, "probability_ensemble_method"] = (
            "cross_validated_temporal_landmark_convex_average"
        )
        results.loc[row_idx, "probability_ensemble_members"] = member_label
        results.loc[row_idx, "probability_ensemble_weights"] = selected_weight_label
        results.loc[row_idx, "probability_ensemble_selected_candidate"] = (
            selected_candidate["candidate_name"]
        )
        results.loc[row_idx, "probability_ensemble_selection_split"] = selection_split
        results.loc[row_idx, "probability_ensemble_selection_rule"] = selection_rule
        results.loc[
            row_idx,
            "probability_ensemble_test_labels_used_for_selection",
        ] = False
        results.loc[row_idx, "calibration_diagnostic_method"] = diagnostic_calibrator[
            "method"
        ]
        results.loc[row_idx, "selected_threshold_objective"] = selected_threshold[
            "candidate_objective"
        ]
        results.loc[row_idx, "selected_threshold"] = threshold
        results.loc[row_idx, "selected_threshold_validation_score"] = (
            selected_threshold["candidate_selection_score"]
        )
        results.loc[row_idx, "validation_model_selection_score"] = selected_report[
            "validation_model_selection_score"
        ]
        results.loc[row_idx, "threshold_candidate_policy"] = THRESHOLD_CANDIDATE_POLICY
        results.loc[row_idx, "threshold_tie_break_policy"] = THRESHOLD_TIE_BREAK_POLICY
        results.loc[row_idx, "threshold_minimum_sensitivity"] = (
            THRESHOLD_MIN_SENSITIVITY
        )
        results.loc[row_idx, "threshold_operating_point_status"] = (
            THRESHOLD_OPERATING_POINT_STATUS
        )
        results.loc[row_idx, "threshold_selection_split"] = (
            "validation_temporal_landmark_ensemble"
        )
        results.loc[row_idx, "threshold_probability_source_provenance"] = (
            probability_source_provenance("temporal_landmark_raw")
        )
        results.loc[row_idx, "threshold_selection_labels_provenance"] = (
            "validation_labels"
        )
        results.loc[row_idx, "threshold_test_labels_used_for_selection"] = False
        if train_frame is not None and train_selected_threshold is not None:
            train_metrics = classification_metrics(
                train_frame["y_true"].astype(int).to_numpy(),
                train_raw_prob,
                float(train_selected_threshold["candidate_threshold"]),
            )
            results.loc[row_idx, "train_cv_selected_threshold_objective"] = (
                train_selected_threshold["candidate_objective"]
            )
            results.loc[row_idx, "train_cv_selected_threshold"] = float(
                train_selected_threshold["candidate_threshold"]
            )
            results.loc[row_idx, "train_cv_selected_threshold_score"] = (
                train_selected_threshold["candidate_selection_score"]
            )
            results.loc[row_idx, "ensemble_train_cv_oof_auroc"] = train_metrics["auroc"]
            results.loc[row_idx, "ensemble_train_cv_oof_auprc"] = train_metrics["auprc"]
            results.loc[row_idx, "ensemble_train_cv_oof_auprc_lift"] = train_metrics[
                "auprc_lift"
            ]
        results.loc[row_idx, "raw_validation_ece"] = raw_val_metrics["ece"]
        results.loc[row_idx, "raw_test_ece"] = raw_test_metrics["ece"]

        for key, value in raw_val_metrics.items():
            results.loc[row_idx, f"raw_validation_{key}"] = value
            results.loc[row_idx, f"validation_{key}"] = value
        for key, value in raw_test_metrics.items():
            results.loc[row_idx, f"raw_test_{key}"] = value
            results.loc[row_idx, f"test_{key}"] = value
        for key, value in calibrated_val_metrics.items():
            results.loc[row_idx, f"calibrated_validation_{key}"] = value
        for key, value in calibrated_test_metrics.items():
            results.loc[row_idx, f"calibrated_test_{key}"] = value
        for key in CALIBRATION_DIAGNOSTIC_METRICS:
            results.loc[row_idx, f"validation_{key}"] = calibrated_val_metrics[key]
            results.loc[row_idx, f"test_{key}"] = calibrated_test_metrics[key]

        combined_threshold_candidates = raw_eval["threshold_candidates"].copy()
        combined_threshold_candidates.insert(0, "target", target)
        combined_threshold_candidates.insert(1, "feature_set", feature_set)
        combined_threshold_candidates.insert(2, "model_variant", model_variant)
        combined_threshold_candidates.insert(
            3, "landmark_day", row.get("landmark_day", np.nan)
        )
        combined_threshold_candidates.insert(
            4, "selection_split", "validation_temporal_landmark_ensemble"
        )
        added_threshold_rows.extend(combined_threshold_candidates.to_dict("records"))

        if train_selected_candidate_rows is not None:
            temporal_train_candidates = train_selected_candidate_rows.copy()
            temporal_train_candidates.insert(
                0, "probability_source", "temporal_landmark_raw"
            )
            temporal_train_candidates.insert(
                1,
                "probability_source_provenance",
                probability_source_provenance("temporal_landmark_raw"),
            )
            temporal_train_candidates.insert(
                2, "selection_labels_provenance", "train_oof_labels"
            )
            temporal_train_candidates.insert(0, "target", target)
            temporal_train_candidates.insert(1, "feature_set", feature_set)
            temporal_train_candidates.insert(2, "model_variant", model_variant)
            temporal_train_candidates.insert(
                3, "landmark_day", row.get("landmark_day", np.nan)
            )
            temporal_train_candidates.insert(
                4,
                "selection_split",
                "train_oof_temporal_landmark_ensemble",
            )
            added_train_threshold_rows.extend(
                temporal_train_candidates.to_dict("records")
            )

        wrapper_paths = write_temporal_landmark_wrapper(
            model_summary=model_summary,
            target=target,
            feature_set=feature_set,
            model_variant=model_variant,
            landmark_day=row.get("landmark_day", np.nan),
            members=used_members,
            weights=selected_candidate["weights"],
            selected_candidate_name=selected_candidate["candidate_name"],
            selection_split=selection_split,
            selection_rule=selection_rule,
            validation_recalibrator=diagnostic_calibrator,
            selected_threshold=selected_threshold,
            train_recalibrator=train_diagnostic_calibrator,
            train_selected_threshold=train_selected_threshold,
        )
        if wrapper_paths is not None:
            results.loc[row_idx, "model_path"] = wrapper_paths["model_path"]
            results.loc[row_idx, "metadata_path"] = wrapper_paths["metadata_path"]
            results.loc[row_idx, "model_relative_path"] = wrapper_paths[
                "model_relative_path"
            ]
            results.loc[row_idx, "metadata_relative_path"] = wrapper_paths[
                "metadata_relative_path"
            ]
            summary_mask = (
                model_summary["target"].astype(str).eq(target)
                & model_summary["feature_set"].astype(str).eq(feature_set)
                & model_summary["model_variant"].astype(str).eq(model_variant)
            )
            if int(summary_mask.sum()) != 1:
                raise RuntimeError(
                    "Temporal wrapper could not identify one current model summary row for "
                    f"{target}/{feature_set}/{model_variant}."
                )
            model_summary.loc[summary_mask, "model_path"] = wrapper_paths["model_path"]
            model_summary.loc[summary_mask, "metadata_path"] = wrapper_paths[
                "metadata_path"
            ]
            model_summary.loc[summary_mask, "model_relative_path"] = wrapper_paths[
                "model_relative_path"
            ]
            model_summary.loc[summary_mask, "metadata_relative_path"] = wrapper_paths[
                "metadata_relative_path"
            ]
            model_summary.loc[summary_mask, "reported_probability_source"] = (
                "temporal_landmark_raw"
            )
            model_summary.loc[summary_mask, "selected_threshold_objective"] = (
                selected_threshold["candidate_objective"]
            )
            model_summary.loc[summary_mask, "selected_threshold"] = threshold
            model_summary.loc[summary_mask, "probability_ensemble_method"] = (
                "cross_validated_temporal_landmark_convex_average"
            )
            model_summary.loc[summary_mask, "probability_ensemble_members"] = (
                member_label
            )
            model_summary.loc[summary_mask, "probability_ensemble_weights"] = (
                selected_weight_label
            )

    if added_threshold_rows:
        threshold_candidates = pd.concat(
            [threshold_candidates, pd.DataFrame(added_threshold_rows)],
            ignore_index=True,
        )
    if added_train_threshold_rows:
        train_threshold_candidates = pd.concat(
            [train_threshold_candidates, pd.DataFrame(added_train_threshold_rows)],
            ignore_index=True,
        )

    ensemble_candidates = pd.DataFrame(
        ensemble_candidate_rows,
        columns=TEMPORAL_ENSEMBLE_CANDIDATE_COLUMNS,
    )
    return (
        results,
        predictions,
        threshold_candidates,
        train_threshold_candidates,
        model_summary,
        ensemble_candidates,
    )


def evaluate_feature_source_candidate(
    target_name,
    current_feature_name,
    candidate_item,
    keep_mask,
    train_pos,
    val_pos,
    y_train,
    y_val,
    base_seed,
):
    optimization_settings = optimization_settings_for_analysis(
        target_name,
        current_feature_name,
    )
    candidate_name = candidate_item["feature_set"]
    candidate_seed_name = candidate_item.get("seed_feature_set", candidate_name)
    if candidate_seed_name == current_feature_name:
        candidate_seed = int(base_seed)
    else:
        candidate_seed = int(
            base_seed
            + (
                zlib.crc32(
                    f"{current_feature_name}|{candidate_seed_name}".encode("utf-8")
                )
                % 1009
            )
        )
    X_all_candidate = candidate_item["X"].loc[keep_mask].reset_index(drop=True)
    X_train_candidate = X_all_candidate.iloc[train_pos].reset_index(drop=True)
    X_val_candidate = X_all_candidate.iloc[val_pos].reset_index(drop=True)

    search = tune_hyperparameters(
        X_train_candidate,
        y_train,
        candidate_seed,
        n_trials=optimization_settings["optuna_trials"],
        feature_selection_candidates=optimization_settings[
            "feature_selection_candidates"
        ],
        cv_repeats=optimization_settings["cv_repeats"],
        modeling_method=optimization_settings["modeling_method"],
        analysis_key=optimization_settings["analysis_key"],
    )
    selected_strategy, validation_model, val_raw_prob, strategy_candidates = (
        select_strategy_by_validation(
            search,
            X_train_candidate,
            y_train,
            X_val_candidate,
            y_val,
            candidate_seed + 29,
            candidate_trials=optimization_settings["validation_candidate_trials"],
            seed_bagging_repeats=optimization_settings["seed_bagging_repeats"],
            ensemble_top_k_candidates=optimization_settings[
                "ensemble_top_k_candidates"
            ],
            include_dual_metric_champions=optimization_settings[
                "include_dual_metric_champions"
            ],
        )
    )
    (
        selected_strategy,
        validation_model,
        val_raw_prob,
        strategy_candidates,
        finalization_candidates,
    ) = finalize_selected_validation_strategy(
        selected_strategy,
        strategy_candidates,
        X_train_candidate,
        y_train,
        X_val_candidate,
        y_val,
        candidate_seed + 29,
        validation_model,
        val_raw_prob,
        optimization_settings["seed_bagging_repeats"],
        candidate_recipes=optimization_settings["finalization_candidates"],
    )
    selected_strategy_row = strategy_candidates.loc[
        strategy_candidates["candidate_strategy_name"]
        == selected_strategy["strategy_name"]
    ].iloc[0]
    validation_selection_score = float(
        selected_strategy_row.get(
            "candidate_robust_selection_score",
            selected_strategy_row["candidate_validation_selection_score"],
        )
    )
    validation_auroc = float(selected_strategy_row["candidate_validation_auroc"])
    validation_auprc = float(selected_strategy_row["candidate_validation_auprc"])
    validation_auprc_lift = float(
        selected_strategy_row.get("candidate_validation_auprc_lift", np.nan)
    )

    return {
        "candidate_feature_source": candidate_name,
        "candidate_feature_variant": candidate_item.get(
            "feature_variant", candidate_name
        ),
        "candidate_landmark_day": candidate_item["landmark_day"],
        "candidate_columns": candidate_item["columns"],
        "candidate_seed": candidate_seed,
        "X_all": X_all_candidate,
        "X_train": X_train_candidate,
        "X_val": X_val_candidate,
        "search": search,
        "selected_strategy": selected_strategy,
        "validation_model": validation_model,
        "val_raw_prob": val_raw_prob,
        "strategy_candidates": strategy_candidates,
        "finalization_candidates": finalization_candidates,
        "validation_selection_score": validation_selection_score,
        "raw_validation_selection_score": float(
            selected_strategy_row["candidate_validation_selection_score"]
        ),
        "validation_reliability_weight": float(
            selected_strategy_row.get("candidate_validation_reliability_weight", 1.0)
        ),
        "validation_auroc": validation_auroc,
        "validation_auprc": validation_auprc,
        "validation_auprc_lift": validation_auprc_lift,
    }


def feature_set_item_for_analysis(feature_set_name):
    for item in ALL_FEATURE_SETS:
        if item["feature_set"] == feature_set_name:
            return item
    raise KeyError(feature_set_name)


def parse_candidate_param_list(params_json):
    param_list = json.loads(params_json)
    if isinstance(param_list, dict):
        param_list = [param_list]
    return [dict(params) for params in param_list]


def selected_original_strategy_row(strategy_table, target_name, feature_name):
    if strategy_table.empty:
        return None
    table = ensure_model_variant_column(strategy_table)
    mask = (
        (table["target"] == target_name)
        & (table["feature_set"] == feature_name)
        & (table["model_variant"] == MODEL_VARIANT_ORIGINAL)
        & (table["selected_strategy"].astype(bool))
    )
    rows = table.loc[mask]
    if rows.empty:
        return None
    return rows.iloc[0]
