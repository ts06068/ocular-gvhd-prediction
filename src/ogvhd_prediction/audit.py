"""Optimization and split-seed audit helpers."""

from .utils import (
    Path,
    json,
    np,
    pd,
    train_test_split,
)
from .config import (
    AGVHD_XGB_REPRODUCTION_PROFILE,
    AUROC_NEAR_BEST_TOLERANCE,
    DEVELOPMENT_SPLIT_SEED_OVERRIDES,
    DUAL_DISCRIMINATION_SELECTION,
    LANDMARK_TEMPORAL_AUROC_TOLERANCE,
    OPTUNA_OBJECTIVE_METRIC,
    OUTPUT_DIR,
    RUN_FINAL_MODELS,
    SPLIT_SEED_SEARCH_CANDIDATES,
    SPLIT_SEED_SEARCH_N_ESTIMATORS,
    TARGETS_TO_RUN,
    THRESHOLD_ACCURACY_SENSITIVITY_OBJECTIVE,
    THRESHOLD_CANDIDATE_POLICY,
    THRESHOLD_MIN_SENSITIVITY,
    THRESHOLD_OBJECTIVE,
    THRESHOLD_OPERATING_POINT_STATUS,
    THRESHOLD_TIE_BREAK_POLICY,
    TRAIN_SIZE,
    VAL_SIZE,
)
from .data import (
    FEATURE_SETS,
    TARGET_DEFINITIONS,
)
from .targets import (
    make_target,
    min_class_count,
    target_landmark_day_for_policy,
)
from .metrics import (
    classification_metrics,
    ensure_model_variant_column,
    file_sha256,
    optimize_threshold,
    probability_source_provenance,
    threshold_metrics,
    validation_model_selection_score,
)
from .preprocessing import make_xgb_pipeline
from .splitting import (
    SELECTED_DEVELOPMENT_SPLIT_SEEDS,
    SOURCE_SPLIT_CACHE,
    fixed_split_seed,
    frozen_complete_source_split,
    frozen_test_source_indices,
    make_train_val_test_split,
)
from .tuning import preset_xgb_trials


def _audit_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _audit_close(left, right, atol=1e-10):
    left = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    return bool(
        np.isfinite(left)
        and np.isfinite(right)
        and np.isclose(left, right, atol=atol, rtol=0.0)
    )


# Audit rows verify that each reported model followed the prespecified selection path.
def build_model_optimization_audit(
    results,
    strategy_candidates,
    threshold_candidates,
    optuna_trials,
    temporal_ensemble_candidates=None,
    predictions=None,
    train_threshold_candidates=None,
    model_summary=None,
):
    """Verify per-model tuning and cross-validated temporal ensembling."""
    if results.empty:
        return pd.DataFrame()

    predictions = (
        ensure_model_variant_column(predictions)
        if predictions is not None and not predictions.empty
        else pd.DataFrame()
    )
    train_threshold_candidates = (
        ensure_model_variant_column(train_threshold_candidates)
        if train_threshold_candidates is not None
        and not train_threshold_candidates.empty
        else pd.DataFrame()
    )
    model_summary = (
        ensure_model_variant_column(model_summary)
        if model_summary is not None and not model_summary.empty
        else pd.DataFrame()
    )

    key_columns = ["target", "feature_set", "model_variant"]
    rows = []
    for _, result in results.iterrows():
        is_temporal_ensemble = (
            str(result.get("probability_ensemble_method", ""))
            == "cross_validated_temporal_landmark_convex_average"
        )
        key_mask = np.ones(len(strategy_candidates), dtype=bool)
        for key in key_columns:
            key_mask &= (
                strategy_candidates[key].astype(str).eq(str(result[key])).to_numpy()
            )
        model_strategies = strategy_candidates.loc[key_mask].copy()
        score_column = (
            "candidate_robust_selection_score"
            if "candidate_robust_selection_score" in model_strategies.columns
            else "candidate_validation_selection_score"
        )
        strategy_scores = pd.to_numeric(
            model_strategies.get(score_column, pd.Series(dtype=float)),
            errors="coerce",
        )
        selected_mask = model_strategies.get(
            "selected_strategy",
            pd.Series(False, index=model_strategies.index),
        ).map(_audit_bool)
        selected_rows = model_strategies.loc[selected_mask]
        selected_strategy_count = int(len(selected_rows))
        selected_strategy_score = (
            float(pd.to_numeric(selected_rows.iloc[0][score_column], errors="coerce"))
            if selected_strategy_count == 1
            else np.nan
        )
        maximum_strategy_score = (
            float(strategy_scores.max()) if strategy_scores.notna().any() else np.nan
        )
        robust_aurocs = pd.to_numeric(
            model_strategies.get("candidate_robust_auroc", pd.Series(dtype=float)),
            errors="coerce",
        )
        robust_auprcs = pd.to_numeric(
            model_strategies.get("candidate_robust_auprc", pd.Series(dtype=float)),
            errors="coerce",
        )
        maximum_robust_auroc = (
            float(robust_aurocs.max()) if robust_aurocs.notna().any() else np.nan
        )
        maximum_robust_auprc = (
            float(robust_auprcs.max()) if robust_auprcs.notna().any() else np.nan
        )
        selected_robust_auroc = (
            float(
                pd.to_numeric(
                    selected_rows.iloc[0].get("candidate_robust_auroc"), errors="coerce"
                )
            )
            if selected_strategy_count == 1
            else np.nan
        )
        selected_robust_auprc = (
            float(
                pd.to_numeric(
                    selected_rows.iloc[0].get("candidate_robust_auprc"), errors="coerce"
                )
            )
            if selected_strategy_count == 1
            else np.nan
        )
        if DUAL_DISCRIMINATION_SELECTION:
            pareto_mask = model_strategies["candidate_pareto_efficient"].map(
                _audit_bool
            )
            near_best_mask = model_strategies["candidate_near_best_auroc"].map(
                _audit_bool
            )
            dual_eligible = pareto_mask & near_best_mask
            if not dual_eligible.any():
                dual_eligible = near_best_mask
            maximum_eligible_auprc = float(robust_auprcs.loc[dual_eligible].max())
            selected_is_pareto = selected_strategy_count == 1 and _audit_bool(
                selected_rows.iloc[0]["candidate_pareto_efficient"]
            )
            selected_auroc_is_near_best = (
                selected_strategy_count == 1
                and selected_robust_auroc
                >= maximum_robust_auroc - AUROC_NEAR_BEST_TOLERANCE - 1e-10
            )
            selected_auprc_is_best_eligible = (
                selected_strategy_count == 1
                and _audit_close(selected_robust_auprc, maximum_eligible_auprc)
            )
            strategy_is_optimal = (
                selected_strategy_count == 1
                and float(selected_rows.iloc[0]["candidate_dual_metric_rank"]) == 1.0
                and selected_is_pareto
                and selected_auroc_is_near_best
                and selected_auprc_is_best_eligible
            )
            selected_auroc_is_absolute_best = _audit_close(
                selected_robust_auroc,
                maximum_robust_auroc,
            )
            selected_auprc_is_absolute_best = _audit_close(
                selected_robust_auprc,
                maximum_robust_auprc,
            )
            simultaneous_champion_exists = bool(
                np.any(
                    np.isclose(
                        robust_aurocs, maximum_robust_auroc, atol=1e-10, rtol=0.0
                    )
                    & np.isclose(
                        robust_auprcs, maximum_robust_auprc, atol=1e-10, rtol=0.0
                    )
                )
            )
        else:
            maximum_eligible_auprc = np.nan
            selected_is_pareto = True
            selected_auroc_is_near_best = True
            selected_auprc_is_best_eligible = True
            strategy_is_optimal = selected_strategy_count == 1 and _audit_close(
                selected_strategy_score, maximum_strategy_score
            )
            selected_auroc_is_absolute_best = np.nan
            selected_auprc_is_absolute_best = np.nan
            simultaneous_champion_exists = np.nan
        strategy_name_matches = selected_strategy_count == 1 and str(
            selected_rows.iloc[0]["candidate_strategy_name"]
        ) == str(result["selected_strategy_name"])

        trial_mask = np.ones(len(optuna_trials), dtype=bool)
        for key in key_columns:
            trial_mask &= optuna_trials[key].astype(str).eq(str(result[key])).to_numpy()
        model_trials = optuna_trials.loc[trial_mask].copy()
        completed_trial_count = (
            int(model_trials["trial_number"].nunique()) if not model_trials.empty else 0
        )
        expected_trial_count = int(result.get("optuna_trials", 0))
        trial_count_matches = completed_trial_count == expected_trial_count
        trial_values = pd.to_numeric(
            model_trials.get("optuna_objective_value", pd.Series(dtype=float)),
            errors="coerce",
        )
        maximum_trial_value = (
            float(trial_values.max()) if trial_values.notna().any() else np.nan
        )
        best_trial_matches = _audit_close(
            result.get("best_optuna_objective_value", np.nan),
            maximum_trial_value,
        )

        temporal_selection_count = 0
        temporal_selection_is_optimal = True
        temporal_selection_uses_oof = True
        temporal_selection_avoids_test = True
        if is_temporal_ensemble:
            temporal_candidates = (
                temporal_ensemble_candidates.copy()
                if temporal_ensemble_candidates is not None
                else pd.DataFrame()
            )
            if not temporal_candidates.empty:
                temporal_mask = np.ones(len(temporal_candidates), dtype=bool)
                for key in key_columns:
                    temporal_mask &= (
                        temporal_candidates[key]
                        .astype(str)
                        .eq(str(result[key]))
                        .to_numpy()
                    )
                temporal_candidates = temporal_candidates.loc[temporal_mask].copy()
            selected_temporal = temporal_candidates.loc[
                temporal_candidates.get(
                    "selected",
                    pd.Series(False, index=temporal_candidates.index),
                ).map(_audit_bool)
            ]
            temporal_selection_count = int(len(selected_temporal))
            temporal_aurocs = pd.to_numeric(
                temporal_candidates.get("selection_auroc", pd.Series(dtype=float)),
                errors="coerce",
            )
            maximum_temporal_auroc = (
                float(temporal_aurocs.max())
                if temporal_aurocs.notna().any()
                else np.nan
            )
            temporal_near_best = temporal_candidates.loc[
                temporal_aurocs
                >= maximum_temporal_auroc - LANDMARK_TEMPORAL_AUROC_TOLERANCE - 1e-10
            ]
            maximum_temporal_auprc = pd.to_numeric(
                temporal_near_best.get("selection_auprc", pd.Series(dtype=float)),
                errors="coerce",
            ).max()
            expected_temporal_candidate_name = None
            if not temporal_near_best.empty:
                expected_temporal_candidate = max(
                    temporal_near_best.to_dict("records"),
                    key=lambda candidate: (
                        float(candidate["selection_auprc"]),
                        float(candidate["selection_auroc"]),
                        float(candidate["current_model_weight"]),
                    ),
                )
                expected_temporal_candidate_name = str(
                    expected_temporal_candidate["candidate_name"]
                )
            temporal_selection_is_optimal = bool(
                temporal_selection_count == 1
                and str(selected_temporal.iloc[0]["candidate_name"])
                == str(result.get("probability_ensemble_selected_candidate", ""))
                and str(selected_temporal.iloc[0]["candidate_name"])
                == expected_temporal_candidate_name
                and _audit_close(
                    selected_temporal.iloc[0]["selection_auprc"],
                    maximum_temporal_auprc,
                )
            )
            temporal_selection_uses_oof = bool(
                temporal_selection_count == 1
                and str(selected_temporal.iloc[0]["selection_split"]) == "train_oof"
            )
            temporal_selection_avoids_test = bool(
                not temporal_candidates.empty
                and not temporal_candidates["test_labels_used_for_selection"]
                .map(_audit_bool)
                .any()
            )

        threshold_mask = np.ones(len(threshold_candidates), dtype=bool)
        for key in key_columns:
            threshold_mask &= (
                threshold_candidates[key].astype(str).eq(str(result[key])).to_numpy()
            )
        model_thresholds = threshold_candidates.loc[threshold_mask].copy()
        if "selection_split" in model_thresholds.columns:
            selection_split = model_thresholds["selection_split"].astype(str)
            if is_temporal_ensemble:
                model_thresholds = model_thresholds.loc[
                    selection_split.eq("validation_temporal_landmark_ensemble")
                ]
            else:
                model_thresholds = model_thresholds.loc[
                    selection_split.str.startswith("validation")
                    & ~selection_split.eq("validation_temporal_landmark_ensemble")
                ]

        source_scores = {}
        source_metric_rows = {}
        for source, source_rows in model_thresholds.groupby(
            "probability_source", dropna=False
        ):
            first = source_rows.iloc[0]
            source_metrics = {
                metric: pd.to_numeric(
                    first.get(f"threshold_cv_{metric}", np.nan), errors="coerce"
                )
                for metric in ("auroc", "auprc", "prevalence", "ece")
            }
            source_scores[str(source)] = validation_model_selection_score(
                source_metrics,
                str(result["validation_model_selection_metric"]),
            )
            source_metric_rows[str(source)] = source_metrics
        reported_source = str(result["reported_probability_source"])
        reported_source_score = source_scores.get(reported_source, np.nan)
        maximum_source_score = max(source_scores.values()) if source_scores else np.nan
        if DUAL_DISCRIMINATION_SELECTION and source_metric_rows:
            maximum_source_auroc = max(
                row["auroc"] for row in source_metric_rows.values()
            )
            eligible_sources = {
                source: row
                for source, row in source_metric_rows.items()
                if row["auroc"] >= maximum_source_auroc - AUROC_NEAR_BEST_TOLERANCE
            }
            maximum_source_auprc = max(
                row["auprc"] for row in eligible_sources.values()
            )
            probability_source_is_optimal = (
                reported_source in eligible_sources
                and _audit_close(
                    eligible_sources[reported_source]["auprc"],
                    maximum_source_auprc,
                )
            )
        else:
            maximum_source_auroc = np.nan
            maximum_source_auprc = np.nan
            probability_source_is_optimal = _audit_close(
                reported_source_score,
                maximum_source_score,
            )

        selected_objective = str(result["selected_threshold_objective"])
        selected_threshold = float(result["selected_threshold"])
        matching_threshold = model_thresholds.loc[
            model_thresholds["probability_source"].astype(str).eq(reported_source)
            & model_thresholds["candidate_objective"].astype(str).eq(selected_objective)
            & np.isclose(
                pd.to_numeric(model_thresholds["candidate_threshold"], errors="coerce"),
                selected_threshold,
                atol=1e-10,
                rtol=0.0,
            )
        ]
        threshold_candidate_found = len(matching_threshold) == 1
        if THRESHOLD_OBJECTIVE == "auto":
            source_thresholds = model_thresholds.loc[
                model_thresholds["probability_source"].astype(str).eq(reported_source)
            ]
            maximum_threshold_score = pd.to_numeric(
                source_thresholds["candidate_selection_score"],
                errors="coerce",
            ).max()
            threshold_is_optimal = threshold_candidate_found and _audit_close(
                matching_threshold.iloc[0]["candidate_selection_score"],
                maximum_threshold_score,
            )
        else:
            threshold_is_optimal = (
                threshold_candidate_found and selected_objective == THRESHOLD_OBJECTIVE
            )
        threshold_sensitivity_constraint_satisfied = (
            True
            if selected_objective != THRESHOLD_ACCURACY_SENSITIVITY_OBJECTIVE
            else (
                threshold_candidate_found
                and bool(
                    matching_threshold.iloc[0].get(
                        "candidate_sensitivity_constraint_satisfied",
                        False,
                    )
                )
            )
        )
        threshold_selection_avoids_test_labels = (
            threshold_candidate_found
            and not bool(
                matching_threshold.iloc[0].get(
                    "candidate_test_labels_used_for_selection",
                    True,
                )
            )
            and str(
                matching_threshold.iloc[0].get(
                    "selection_labels_provenance",
                    "",
                )
            )
            == "validation_labels"
            and str(
                matching_threshold.iloc[0].get(
                    "probability_source_provenance",
                    "",
                )
            )
            == probability_source_provenance(reported_source)
        )

        prediction_threshold_consistency = True
        probability_source_provenance_consistent = True
        validation_threshold_recomputed_optimal = threshold_is_optimal
        validation_sensitivity_recomputed_satisfied = (
            threshold_sensitivity_constraint_satisfied
        )
        model_predictions = pd.DataFrame()
        if not predictions.empty:
            prediction_mask = np.ones(len(predictions), dtype=bool)
            for key in key_columns:
                prediction_mask &= (
                    predictions[key].astype(str).eq(str(result[key])).to_numpy()
                )
            model_predictions = predictions.loc[prediction_mask].copy()
            probability_source_provenance_consistent = bool(
                not model_predictions.empty
                and model_predictions["reported_probability_source"]
                .astype(str)
                .eq(reported_source)
                .all()
            )
            if probability_source_provenance_consistent:
                reported_values = pd.to_numeric(
                    model_predictions["probability"], errors="coerce"
                ).to_numpy(dtype=float)
                raw_values = pd.to_numeric(
                    model_predictions["raw_probability"], errors="coerce"
                ).to_numpy(dtype=float)
                expected_reported_values = (
                    1.0 - raw_values
                    if reported_source == "raw_inverted"
                    else raw_values
                )
                probability_source_provenance_consistent = bool(
                    np.allclose(
                        reported_values,
                        expected_reported_values,
                        # XGBoost probabilities are commonly float32; saving
                        # both p and 1-p can differ by a few 1e-8 after casting.
                        atol=1e-7,
                        rtol=0.0,
                        equal_nan=False,
                    )
                )
            prediction_threshold_consistency = bool(
                not model_predictions.empty
                and np.array_equal(
                    (
                        pd.to_numeric(model_predictions["probability"], errors="coerce")
                        >= pd.to_numeric(
                            model_predictions["threshold"], errors="coerce"
                        )
                    )
                    .astype(int)
                    .to_numpy(),
                    pd.to_numeric(model_predictions["y_pred"], errors="coerce")
                    .astype(int)
                    .to_numpy(),
                )
            )
            validation_predictions = model_predictions.loc[
                model_predictions["dataset_split"].astype(str).eq("validation")
            ]
            if validation_predictions.empty:
                validation_threshold_recomputed_optimal = False
                validation_sensitivity_recomputed_satisfied = False
            else:
                validation_labels = (
                    pd.to_numeric(validation_predictions["y_true"], errors="coerce")
                    .astype(int)
                    .to_numpy()
                )
                validation_probability = pd.to_numeric(
                    validation_predictions["probability"], errors="coerce"
                ).to_numpy(dtype=float)
                recomputed_threshold, _ = optimize_threshold(
                    validation_labels,
                    validation_probability,
                    selected_objective,
                )
                validation_threshold_recomputed_optimal = _audit_close(
                    selected_threshold,
                    recomputed_threshold,
                )
                recomputed_metrics = threshold_metrics(
                    validation_labels,
                    validation_probability,
                    selected_threshold,
                )
                validation_sensitivity_recomputed_satisfied = bool(
                    selected_objective != THRESHOLD_ACCURACY_SENSITIVITY_OBJECTIVE
                    or recomputed_metrics["sensitivity"] + 1e-12
                    >= THRESHOLD_MIN_SENSITIVITY
                )
                threshold_is_optimal = validation_threshold_recomputed_optimal
                threshold_sensitivity_constraint_satisfied = (
                    validation_sensitivity_recomputed_satisfied
                )

        temporal_train_oof_probability_consistent = True
        temporal_train_oof_threshold_recomputed_optimal = True
        temporal_train_threshold_candidate_found = True
        temporal_wrapper_artifact_valid = True
        if is_temporal_ensemble:
            temporal_train = model_predictions.loc[
                model_predictions.get(
                    "dataset_split",
                    pd.Series(dtype=str),
                )
                .astype(str)
                .eq("train_oof")
            ].copy()
            members = str(result.get("probability_ensemble_members", "")).split("|")
            weights = json.loads(str(result.get("probability_ensemble_weights", "{}")))
            if (
                temporal_train.empty
                or not members
                or any(member not in weights for member in members)
            ):
                temporal_train_oof_probability_consistent = False
                temporal_train_oof_threshold_recomputed_optimal = False
                temporal_train_threshold_candidate_found = False
            else:
                expected_train = temporal_train.loc[
                    :, ["source_row_index", "raw_probability"]
                ].drop_duplicates("source_row_index")
                expected_train = expected_train.rename(
                    columns={"raw_probability": "reported_temporal_raw_probability"}
                )
                for member in members:
                    member_mask = (
                        predictions["target"].astype(str).eq(str(result["target"]))
                        & predictions["feature_set"].astype(str).eq(member)
                        & predictions["model_variant"]
                        .astype(str)
                        .eq(str(result["model_variant"]))
                        & predictions["dataset_split"].astype(str).eq("train_oof")
                    )
                    member_rows = (
                        predictions.loc[
                            member_mask,
                            ["source_row_index", "single_model_raw_probability"],
                        ]
                        .drop_duplicates("source_row_index")
                        .rename(columns={"single_model_raw_probability": member})
                    )
                    expected_train = expected_train.merge(
                        member_rows,
                        on="source_row_index",
                        how="left",
                        validate="one_to_one",
                    )
                expected_values = sum(
                    float(weights[member])
                    * pd.to_numeric(expected_train[member], errors="coerce").to_numpy(
                        dtype=float
                    )
                    for member in members
                )
                reported_values = pd.to_numeric(
                    expected_train["reported_temporal_raw_probability"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                temporal_train_oof_probability_consistent = bool(
                    np.isfinite(expected_values).all()
                    and np.allclose(
                        expected_values, reported_values, atol=1e-12, rtol=0.0
                    )
                )
                temporal_train_labels = (
                    pd.to_numeric(temporal_train["y_true"], errors="coerce")
                    .astype(int)
                    .to_numpy()
                )
                temporal_train_probability = pd.to_numeric(
                    temporal_train["probability"], errors="coerce"
                ).to_numpy(dtype=float)
                temporal_train_thresholds = (
                    pd.to_numeric(temporal_train["threshold"], errors="coerce")
                    .dropna()
                    .unique()
                )
                if len(temporal_train_thresholds) != 1:
                    temporal_train_oof_threshold_recomputed_optimal = False
                    temporal_train_threshold_candidate_found = False
                else:
                    temporal_train_objective = str(
                        result.get(
                            "train_cv_selected_threshold_objective", THRESHOLD_OBJECTIVE
                        )
                    )
                    expected_train_threshold, _ = optimize_threshold(
                        temporal_train_labels,
                        temporal_train_probability,
                        temporal_train_objective,
                    )
                    temporal_train_oof_threshold_recomputed_optimal = _audit_close(
                        temporal_train_thresholds[0],
                        expected_train_threshold,
                    )
                    if train_threshold_candidates.empty:
                        temporal_train_threshold_candidate_found = False
                    else:
                        train_candidate_mask = np.ones(
                            len(train_threshold_candidates), dtype=bool
                        )
                        for key in key_columns:
                            train_candidate_mask &= (
                                train_threshold_candidates[key]
                                .astype(str)
                                .eq(str(result[key]))
                                .to_numpy()
                            )
                        train_candidate_mask &= (
                            train_threshold_candidates["selection_split"]
                            .astype(str)
                            .eq("train_oof_temporal_landmark_ensemble")
                            .to_numpy()
                        )
                        train_candidate_mask &= (
                            train_threshold_candidates["probability_source"]
                            .astype(str)
                            .eq("temporal_landmark_raw")
                            .to_numpy()
                        )
                        train_candidate_mask &= (
                            train_threshold_candidates["candidate_objective"]
                            .astype(str)
                            .eq(temporal_train_objective)
                            .to_numpy()
                        )
                        train_candidate_mask &= np.isclose(
                            pd.to_numeric(
                                train_threshold_candidates["candidate_threshold"],
                                errors="coerce",
                            ),
                            temporal_train_thresholds[0],
                            atol=1e-10,
                            rtol=0.0,
                        )
                        temporal_train_threshold_candidate_found = bool(
                            int(train_candidate_mask.sum()) == 1
                        )

            if RUN_FINAL_MODELS:
                wrapper_path = Path(str(result.get("model_path", "")))
                wrapper_metadata_path = Path(str(result.get("metadata_path", "")))
                temporal_wrapper_artifact_valid = bool(
                    wrapper_path.is_file() and wrapper_metadata_path.is_file()
                )
                if temporal_wrapper_artifact_valid:
                    try:
                        with open(
                            wrapper_metadata_path, "r", encoding="utf-8"
                        ) as handle:
                            wrapper_metadata = json.load(handle)
                        member_artifacts = wrapper_metadata.get(
                            "ordered_member_artifacts", []
                        )
                        temporal_wrapper_artifact_valid = bool(
                            wrapper_metadata.get("artifact_type")
                            == "temporal_landmark_ensemble_wrapper"
                            and wrapper_metadata.get(
                                "threshold_test_labels_used_for_selection"
                            )
                            is False
                            and wrapper_metadata.get(
                                "weight_selection_test_labels_used"
                            )
                            is False
                            and _audit_close(
                                wrapper_metadata.get("threshold"), selected_threshold
                            )
                            and len(member_artifacts) == len(members)
                            and all(
                                (
                                    wrapper_metadata_path.parent
                                    / item["base_model_relative_path"]
                                ).is_file()
                                and file_sha256(
                                    wrapper_metadata_path.parent
                                    / item["base_model_relative_path"]
                                )
                                == item["base_model_sha256"]
                                and (
                                    wrapper_metadata_path.parent
                                    / item["base_metadata_relative_path"]
                                ).is_file()
                                and file_sha256(
                                    wrapper_metadata_path.parent
                                    / item["base_metadata_relative_path"]
                                )
                                == item["base_metadata_sha256"]
                                for item in member_artifacts
                            )
                        )
                    except (
                        OSError,
                        KeyError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ):
                        temporal_wrapper_artifact_valid = False

        checks = {
            "one_selected_strategy": selected_strategy_count == 1,
            "selected_strategy_is_validation_optimal": strategy_is_optimal,
            "selected_strategy_is_pareto_efficient": selected_is_pareto,
            "selected_strategy_auroc_is_near_best": selected_auroc_is_near_best,
            "selected_strategy_auprc_is_best_within_auroc_tolerance": selected_auprc_is_best_eligible,
            "selected_strategy_matches_result": strategy_name_matches,
            "independent_optuna_trial_count_matches": trial_count_matches,
            "best_optuna_trial_matches": best_trial_matches,
            "temporal_ensemble_selection_is_oof_optimal": temporal_selection_is_optimal,
            "temporal_ensemble_selection_uses_train_oof": temporal_selection_uses_oof,
            "temporal_ensemble_selection_avoids_test_labels": temporal_selection_avoids_test,
            "reported_probability_source_is_validation_optimal": probability_source_is_optimal,
            "selected_threshold_is_validation_optimal": threshold_is_optimal,
            "selected_threshold_recomputed_from_validation_predictions": (
                validation_threshold_recomputed_optimal
            ),
            "selected_threshold_sensitivity_constraint_satisfied": (
                threshold_sensitivity_constraint_satisfied
            ),
            "threshold_selection_avoids_test_labels": threshold_selection_avoids_test_labels,
            "prediction_threshold_consistency": prediction_threshold_consistency,
            "probability_source_provenance_consistent": (
                probability_source_provenance_consistent
            ),
            "temporal_train_oof_probability_consistent": (
                temporal_train_oof_probability_consistent
            ),
            "temporal_train_oof_threshold_recomputed_optimal": (
                temporal_train_oof_threshold_recomputed_optimal
            ),
            "temporal_train_threshold_candidate_found": (
                temporal_train_threshold_candidate_found
            ),
            "temporal_wrapper_artifact_valid": temporal_wrapper_artifact_valid,
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                **{key: result[key] for key in key_columns},
                "optimization_audit_pass": not failed_checks,
                "failed_checks": "|".join(failed_checks),
                "test_used_for_model_selection": False,
                "optuna_objective_metric": OPTUNA_OBJECTIVE_METRIC,
                "validation_model_selection_metric": result[
                    "validation_model_selection_metric"
                ],
                "completed_optuna_trials": completed_trial_count,
                "expected_optuna_trials": expected_trial_count,
                "maximum_optuna_objective_value": maximum_trial_value,
                "selected_strategy_name": result["selected_strategy_name"],
                "selected_strategy_count": selected_strategy_count,
                "selected_temporal_ensemble_candidate_count": temporal_selection_count,
                "selected_strategy_score": selected_strategy_score,
                "maximum_strategy_score": maximum_strategy_score,
                "selected_strategy_robust_auroc": selected_robust_auroc,
                "maximum_strategy_robust_auroc": maximum_robust_auroc,
                "selected_strategy_robust_auprc": selected_robust_auprc,
                "maximum_strategy_robust_auprc": maximum_robust_auprc,
                "maximum_eligible_strategy_robust_auprc": maximum_eligible_auprc,
                "selected_strategy_auroc_is_absolute_best": selected_auroc_is_absolute_best,
                "selected_strategy_auprc_is_absolute_best": selected_auprc_is_absolute_best,
                "simultaneous_absolute_metric_champion_exists": simultaneous_champion_exists,
                "auroc_auprc_tradeoff_present": (
                    not simultaneous_champion_exists
                    if isinstance(simultaneous_champion_exists, (bool, np.bool_))
                    else np.nan
                ),
                "auroc_near_best_tolerance": AUROC_NEAR_BEST_TOLERANCE,
                "reported_probability_source": reported_source,
                "reported_probability_source_score": reported_source_score,
                "maximum_probability_source_score": maximum_source_score,
                "maximum_probability_source_auroc": maximum_source_auroc,
                "maximum_eligible_probability_source_auprc": maximum_source_auprc,
                "selected_threshold_objective": selected_objective,
                "selected_threshold": selected_threshold,
                "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
                "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
                "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
                "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
                "threshold_test_labels_used_for_selection": False,
                "test_positive_n": result.get("test_positive_n", np.nan),
                "test_min_class_count": result.get("test_min_class_count", np.nan),
                "test_auroc": result.get("test_auroc", np.nan),
                "test_auroc_reliable": result.get("test_auroc_reliable", False),
                **checks,
            }
        )
    return pd.DataFrame(rows)


def evaluate_development_split_seed(target_name, spec, candidate_seed):
    keep_reference, y_reference = make_target(
        spec["stage_column"],
        spec["positive_rule"],
        spec.get("exclude_control_death_3m", False),
        spec.get("event_date_column"),
        landmark_day=target_landmark_day_for_policy(0),
    )
    reference_source_indices = np.flatnonzero(keep_reference)
    frozen_complete_split = frozen_complete_source_split(
        target_name,
        fixed_split_seed(target_name),
        int(candidate_seed),
        reference_source_indices,
        y_reference,
    )
    if frozen_complete_split is not None:
        train_source_indices = set(map(int, frozen_complete_split["train"]))
        validation_source_indices = set(map(int, frozen_complete_split["validation"]))
    else:
        frozen_test_sources = frozen_test_source_indices(
            target_name,
            fixed_split_seed(target_name),
            reference_source_indices,
            y_reference,
        )
        if frozen_test_sources is None:
            base_train_pos, base_val_pos, _ = make_train_val_test_split(
                y_reference,
                fixed_split_seed(target_name),
            )
            development_pos = np.concatenate([base_train_pos, base_val_pos])
        else:
            frozen_test_source_set = set(map(int, frozen_test_sources))
            development_pos = np.asarray(
                [
                    position
                    for position, source_index in enumerate(reference_source_indices)
                    if int(source_index) not in frozen_test_source_set
                ],
                dtype=int,
            )
        development_source_indices = reference_source_indices[development_pos]
        development_y = y_reference.iloc[development_pos].reset_index(drop=True)
        train_source_indices, validation_source_indices = train_test_split(
            development_source_indices,
            train_size=TRAIN_SIZE / (TRAIN_SIZE + VAL_SIZE),
            stratify=development_y,
            random_state=int(candidate_seed),
            shuffle=True,
        )
        train_source_indices = set(map(int, train_source_indices))
        validation_source_indices = set(map(int, validation_source_indices))
    rows = []

    for feature_set in FEATURE_SETS:
        feature_name = feature_set["feature_set"]
        landmark_day = feature_set["landmark_day"]
        keep_mask, y_target = make_target(
            spec["stage_column"],
            spec["positive_rule"],
            spec.get("exclude_control_death_3m", False),
            spec.get("event_date_column"),
            target_landmark_day_for_policy(landmark_day),
        )
        source_indices = np.flatnonzero(keep_mask)
        train_pos = np.asarray(
            [
                idx
                for idx, source_idx in enumerate(source_indices)
                if int(source_idx) in train_source_indices
            ],
            dtype=int,
        )
        val_pos = np.asarray(
            [
                idx
                for idx, source_idx in enumerate(source_indices)
                if int(source_idx) in validation_source_indices
            ],
            dtype=int,
        )
        y_train = y_target.iloc[train_pos].reset_index(drop=True)
        y_val = y_target.iloc[val_pos].reset_index(drop=True)
        row = {
            "target": target_name,
            "candidate_development_split_seed": int(candidate_seed),
            "fixed_test_split_seed": int(fixed_split_seed(target_name)),
            "feature_set": feature_name,
            "landmark_day": int(landmark_day),
            "train_n": int(len(train_pos)),
            "validation_n": int(len(val_pos)),
            "train_positive_n": int(y_train.sum()) if len(y_train) else 0,
            "validation_positive_n": int(y_val.sum()) if len(y_val) else 0,
            "test_labels_used_for_seed_selection": False,
        }
        if min_class_count(y_train) < 2 or min_class_count(y_val) < 1:
            row.update(
                {
                    "validation_auroc": np.nan,
                    "validation_auprc": np.nan,
                    "validation_auprc_lift": np.nan,
                    "seed_candidate_status": "skipped_sparse_landmark",
                }
            )
            rows.append(row)
            continue

        X_target = feature_set["X"].loc[keep_mask].reset_index(drop=True)
        model_seed = int(candidate_seed + 1009 * (landmark_day + 1))
        model = make_xgb_pipeline(model_seed)
        params = preset_xgb_trials(y_train, [1])[0]
        params["preprocess__max_iter"] = 1
        params["model__n_estimators"] = SPLIT_SEED_SEARCH_N_ESTIMATORS
        model.set_params(**params)
        model.fit(X_target.iloc[train_pos], y_train)
        val_prob = model.predict_proba(X_target.iloc[val_pos])[:, 1]
        metrics = classification_metrics(y_val, val_prob, 0.5)
        row.update(
            {
                "validation_auroc": metrics["auroc"],
                "validation_auprc": metrics["auprc"],
                "validation_auprc_lift": metrics["auprc_lift"],
                "seed_candidate_status": "evaluated",
            }
        )
        rows.append(row)

    evaluated = [row for row in rows if np.isfinite(row["validation_auroc"])]
    aurocs = np.asarray([row["validation_auroc"] for row in evaluated], dtype=float)
    auprcs = np.asarray([row["validation_auprc"] for row in evaluated], dtype=float)
    lifts = np.asarray([row["validation_auprc_lift"] for row in evaluated], dtype=float)
    aggregate = {
        "target": target_name,
        "candidate_development_split_seed": int(candidate_seed),
        "fixed_test_split_seed": int(fixed_split_seed(target_name)),
        "valid_landmark_count": int(len(evaluated)),
        "worst_landmark_validation_auroc": float(np.min(aurocs))
        if len(aurocs)
        else np.nan,
        "mean_validation_auroc": float(np.mean(aurocs)) if len(aurocs) else np.nan,
        "std_validation_auroc": float(np.std(aurocs)) if len(aurocs) else np.nan,
        "mean_validation_auprc": float(np.mean(auprcs)) if len(auprcs) else np.nan,
        "mean_validation_auprc_lift": float(np.mean(lifts)) if len(lifts) else np.nan,
    }
    return aggregate, rows


def run_development_split_seed_search():
    global SPLIT_SEED_SEARCH_ROWS
    search_rows = []
    SELECTED_DEVELOPMENT_SPLIT_SEEDS.clear()
    SOURCE_SPLIT_CACHE.clear()
    for target_name in TARGETS_TO_RUN:
        spec = TARGET_DEFINITIONS[target_name]
        if AGVHD_XGB_REPRODUCTION_PROFILE and target_name.startswith("aGVHD_"):
            # Preserve the archived development partition for exact XGBoost
            # reproduction; current test labels are not searched.
            candidate_seeds = [int(fixed_split_seed(target_name)) + 17]
        elif target_name in DEVELOPMENT_SPLIT_SEED_OVERRIDES:
            candidate_seeds = [int(DEVELOPMENT_SPLIT_SEED_OVERRIDES[target_name])]
        else:
            candidate_seeds = sorted(
                set(SPLIT_SEED_SEARCH_CANDIDATES + [fixed_split_seed(target_name)])
            )
        aggregates = []
        detail_rows = []
        for candidate_seed in candidate_seeds:
            aggregate, rows = evaluate_development_split_seed(
                target_name,
                spec,
                candidate_seed,
            )
            aggregates.append(aggregate)
            detail_rows.extend(rows)
        valid = [
            row
            for row in aggregates
            if row["valid_landmark_count"] > 0
            and np.isfinite(row["worst_landmark_validation_auroc"])
        ]
        if not valid:
            selected_seed = int(fixed_split_seed(target_name))
        else:
            selected = sorted(
                valid,
                key=lambda row: (
                    row["valid_landmark_count"],
                    row["worst_landmark_validation_auroc"],
                    row["mean_validation_auroc"],
                    row["mean_validation_auprc_lift"],
                    -row["std_validation_auroc"],
                    -row["candidate_development_split_seed"],
                ),
                reverse=True,
            )[0]
            selected_seed = int(selected["candidate_development_split_seed"])
        SELECTED_DEVELOPMENT_SPLIT_SEEDS[target_name] = selected_seed
        for aggregate in aggregates:
            aggregate["selected_development_split_seed"] = selected_seed
            aggregate["selected_seed"] = (
                int(aggregate["candidate_development_split_seed"]) == selected_seed
            )
        for row in detail_rows:
            row["selected_development_split_seed"] = selected_seed
            row["selected_seed"] = (
                int(row["candidate_development_split_seed"]) == selected_seed
            )
        search_rows.extend(detail_rows)
        search_rows.extend(
            {**row, "feature_set": "__aggregate__"} for row in aggregates
        )
        print(
            f"Split-seed selection | {target_name}: development_seed={selected_seed} | "
            f"fixed_test_seed={fixed_split_seed(target_name)} | "
            f"candidates={len(candidate_seeds)} | test_labels_used=False"
        )
    SPLIT_SEED_SEARCH_ROWS = search_rows
    SOURCE_SPLIT_CACHE.clear()
    table = pd.DataFrame(search_rows)
    table.to_csv(OUTPUT_DIR / "development_split_seed_search.csv", index=False)
    return table
