"""Performance summary assembly."""

from .utils import (
    display,
    np,
    pd,
    roc_auc_score,
    zlib,
)
from .config import (
    AUROC_TARGET,
    BOOTSTRAP_CI_CONFIDENCE_LEVEL,
    BOOTSTRAP_CI_N_RESAMPLES,
    BOOTSTRAP_CI_RANDOM_STATE,
    MODEL_VARIANT_ORIGINAL,
    MODEL_VARIANT_SIMPLIFIED,
    OUTPUT_DIR,
    RUN_BOOTSTRAP_CI,
    RUN_HOLDOUT_EVALUATION,
)
from .metrics import (
    PAIRED_LANDMARK_AUDIT_COLUMNS,
    add_bootstrap_cis_to_metric_summary,
    paired_stratified_auroc_difference_ci,
)
from .finalization import (
    holdout_bootstrap_ci,
    holdout_predictions,
    holdout_results,
)


def run_stage():
    """Build and save performance summaries."""
    global auroc_check, holdout_metric_summary, holdout_severity_order_audit
    global landmark_order_audit, paired_landmark_audit

    if RUN_HOLDOUT_EVALUATION and not holdout_results.empty:
        metric_columns = [
            "validation_auroc",
            "validation_auprc",
            "validation_prevalence",
            "validation_auprc_baseline",
            "validation_auprc_lift",
            "validation_accuracy",
            "validation_specificity",
            "validation_sensitivity",
            "validation_ppv",
            "validation_npv",
            "validation_balanced_accuracy",
            "validation_youden_index",
            "validation_f1",
            "validation_net_benefit",
            "validation_decision_curve_net_benefit_at_configured_threshold",
            "validation_ece",
            "validation_calibration_intercept",
            "validation_calibration_slope",
            "raw_validation_ece",
            "raw_validation_calibration_intercept",
            "raw_validation_calibration_slope",
            "raw_validation_auroc",
            "raw_validation_auprc",
            "raw_validation_auprc_lift",
            "raw_test_calibration_intercept",
            "raw_test_calibration_slope",
            "calibrated_validation_auroc",
            "calibrated_validation_auprc",
            "calibrated_validation_auprc_lift",
            "calibrated_validation_ece",
            "calibrated_validation_calibration_intercept",
            "calibrated_validation_calibration_slope",
            "test_auroc",
            "test_auprc",
            "test_prevalence",
            "test_auprc_baseline",
            "test_auprc_lift",
            "test_accuracy",
            "test_specificity",
            "test_sensitivity",
            "test_ppv",
            "test_npv",
            "test_balanced_accuracy",
            "test_youden_index",
            "test_f1",
            "test_net_benefit",
            "test_decision_curve_net_benefit_at_configured_threshold",
            "test_ece",
            "test_calibration_intercept",
            "test_calibration_slope",
            "raw_test_ece",
            "raw_test_auroc",
            "raw_test_auprc",
            "raw_test_auprc_lift",
            "calibrated_test_auroc",
            "calibrated_test_auprc",
            "calibrated_test_auprc_lift",
            "calibrated_test_ece",
            "calibrated_test_calibration_intercept",
            "calibrated_test_calibration_slope",
            "selected_threshold",
        ]
        summary_rows = []
        for (target, feature_set, model_variant), group in holdout_results.groupby(
            [
                "target",
                "feature_set",
                "model_variant",
            ]
        ):
            for metric in metric_columns:
                values = (
                    pd.to_numeric(group[metric], errors="coerce")
                    if metric in group.columns
                    else pd.Series(dtype=float)
                )
                summary_rows.append(
                    {
                        "target": target,
                        "feature_set": feature_set,
                        "model_variant": model_variant,
                        "metric": metric,
                        "value": values.iloc[0] if len(values) else np.nan,
                    }
                )
        holdout_metric_summary = pd.DataFrame(summary_rows)
        holdout_metric_summary = add_bootstrap_cis_to_metric_summary(
            holdout_metric_summary,
            holdout_results,
            holdout_bootstrap_ci,
        )
        holdout_metric_summary.to_csv(
            OUTPUT_DIR / "holdout_metric_summary.csv", index=False
        )
        display(holdout_metric_summary)

        auroc_check = holdout_results.loc[
            :,
            [
                "target",
                "feature_set",
                "model_variant",
                "test_auroc",
                "test_auroc_ci_lower",
                "test_auroc_ci_upper",
                "test_auroc_bootstrap_valid_resamples",
                "selected_threshold",
                "effective_train_cv_splits",
                "requested_train_cv_folds_available",
                "validation_selection_reliable",
                "test_min_class_count",
                "test_auroc_reliable",
                "test_auroc_reliability_status",
                "endpoint_estimability_status",
                "test_auroc_ge_target_reliable",
                "development_cv_auroc_ge_target",
                "test_auroc_ci_contains_target",
                "development_cv_minus_test_auroc",
                "auroc_target_status",
            ],
        ].assign(
            auroc_target=AUROC_TARGET,
            test_auroc_ge_target=lambda x: x["test_auroc"] >= x["auroc_target"],
        )
        auroc_check.to_csv(OUTPUT_DIR / "holdout_auroc_target_check.csv", index=False)
        display(auroc_check)
        unmet_auroc_target = auroc_check.loc[~auroc_check["test_auroc_ge_target"]]
        if not unmet_auroc_target.empty:
            print(
                f"AUROC target {AUROC_TARGET:.2f} was not met by "
                f"{len(unmet_auroc_target)}/{len(auroc_check)} holdout estimates. "
                "The held-out labels are audit-only and are not used to select a split, "
                "hyperparameters, probability orientation, or a reporting variant."
            )

        landmark_order = [
            "Preoperative",
            "Post transplant D+30",
            "Post transplant D+60",
            "Post transplant D+90",
        ]

        def heldout_rows_and_labels(target, feature_set, model_variant):
            required = {
                "target",
                "feature_set",
                "model_variant",
                "dataset_split",
                "source_row_index",
                "y_true",
            }
            if holdout_predictions.empty or not required.issubset(
                holdout_predictions.columns
            ):
                return tuple()
            mask = (
                (holdout_predictions["target"] == target)
                & (holdout_predictions["feature_set"] == feature_set)
                & (holdout_predictions["model_variant"] == model_variant)
                & (holdout_predictions["dataset_split"] == "test")
            )
            rows = (
                holdout_predictions.loc[mask, ["source_row_index", "y_true"]]
                .dropna()
                .astype(int)
                .drop_duplicates()
                .sort_values(["source_row_index", "y_true"], kind="mergesort")
            )
            return tuple(map(tuple, rows.to_numpy()))

        def result_split_seed(row):
            value = pd.to_numeric(
                pd.Series([row.get("split_seed", np.nan)]), errors="coerce"
            ).iloc[0]
            return int(value) if pd.notna(value) else None

        order_rows = []
        for (target, model_variant), group in holdout_results.groupby(
            ["target", "model_variant"]
        ):
            ordered = (
                group.set_index("feature_set").reindex(landmark_order).reset_index()
            )
            aurocs = pd.to_numeric(ordered["test_auroc"], errors="coerce").to_numpy(
                dtype=float
            )
            valid = np.isfinite(aurocs)
            best_idx = int(np.nanargmax(aurocs)) if np.any(valid) else -1
            observed_non_decreasing = (
                bool(np.all(np.diff(aurocs[valid]) >= -1e-12))
                if np.sum(valid) >= 2
                else pd.NA
            )
            valid_rows = ordered.loc[valid]
            split_seeds = [
                seed
                for seed in (result_split_seed(row) for _, row in valid_rows.iterrows())
                if seed is not None
            ]
            unique_split_seeds = sorted(set(split_seeds))
            signatures = [
                heldout_rows_and_labels(target, row["feature_set"], model_variant)
                for _, row in valid_rows.iterrows()
            ]
            if np.sum(valid) < 2:
                comparison_valid = False
                comparison_reason = "insufficient_estimable_landmarks"
            elif len(unique_split_seeds) != 1:
                comparison_valid = False
                comparison_reason = "different_test_split_seeds"
            elif any(not signature for signature in signatures):
                comparison_valid = False
                comparison_reason = "test_prediction_rows_unavailable"
            elif all(signature == signatures[0] for signature in signatures[1:]):
                comparison_valid = True
                comparison_reason = "same_test_rows_and_labels"
            else:
                comparison_valid = False
                comparison_reason = "different_incident_risk_sets_or_labels"
            is_non_decreasing = observed_non_decreasing if comparison_valid else pd.NA
            for i, row in ordered.iterrows():
                test_prevalence = row.get("test_prevalence", np.nan)
                test_auprc = row.get("test_auprc", np.nan)
                auprc_lift = (
                    float(test_auprc / test_prevalence)
                    if pd.notna(test_prevalence) and test_prevalence
                    else np.nan
                )
                order_rows.append(
                    {
                        "target": target,
                        "model_variant": model_variant,
                        "feature_set": row["feature_set"],
                        "selected_feature_source": row.get(
                            "selected_feature_source", np.nan
                        ),
                        "selected_feature_source_day": row.get(
                            "selected_feature_source_day", np.nan
                        ),
                        "landmark_order": i,
                        "test_auroc": row.get("test_auroc", np.nan),
                        "test_auroc_ci_lower": row.get("test_auroc_ci_lower", np.nan),
                        "test_auroc_ci_upper": row.get("test_auroc_ci_upper", np.nan),
                        "test_auprc": test_auprc,
                        "test_auprc_ci_lower": row.get("test_auprc_ci_lower", np.nan),
                        "test_auprc_ci_upper": row.get("test_auprc_ci_upper", np.nan),
                        "test_prevalence": test_prevalence,
                        "test_auprc_lift": auprc_lift,
                        "validation_auroc": row.get("validation_auroc", np.nan),
                        "validation_auprc": row.get("validation_auprc", np.nan),
                        "validation_auprc_lift": row.get(
                            "validation_auprc_lift", np.nan
                        ),
                        "best_train_cv_mean_auroc": row.get(
                            "best_train_cv_mean_auroc", np.nan
                        ),
                        "best_train_cv_mean_auprc": row.get(
                            "best_train_cv_mean_auprc", np.nan
                        ),
                        "best_train_cv_auprc_lift": row.get(
                            "best_train_cv_auprc_lift", np.nan
                        ),
                        "ensemble_train_cv_oof_auroc": row.get(
                            "ensemble_train_cv_oof_auroc", np.nan
                        ),
                        "ensemble_train_cv_oof_auprc": row.get(
                            "ensemble_train_cv_oof_auprc", np.nan
                        ),
                        "ensemble_train_cv_oof_auprc_lift": row.get(
                            "ensemble_train_cv_oof_auprc_lift", np.nan
                        ),
                        "analysis_positive_n": row.get("analysis_positive_n", np.nan),
                        "train_positive_n": row.get("train_positive_n", np.nan),
                        "validation_positive_n": row.get(
                            "validation_positive_n", np.nan
                        ),
                        "test_positive_n": row.get("test_positive_n", np.nan),
                        "n_features_after_missingness_filter": row.get(
                            "n_features_after_missingness_filter", np.nan
                        ),
                        "n_features_after_feature_selection": row.get(
                            "n_features_after_feature_selection", np.nan
                        ),
                        "n_transformed_features": row.get(
                            "n_transformed_features", np.nan
                        ),
                        "feature_selection_k": row.get("feature_selection_k", np.nan),
                        "landmark_comparison_valid": comparison_valid,
                        "landmark_comparison_reason": comparison_reason,
                        "landmark_split_seeds": "|".join(map(str, unique_split_seeds)),
                        "observed_test_auroc_non_decreasing": observed_non_decreasing,
                        "target_is_non_decreasing": is_non_decreasing,
                        "target_best_feature_set": ordered.loc[best_idx, "feature_set"]
                        if best_idx >= 0
                        else np.nan,
                    }
                )
        landmark_order_audit = pd.DataFrame(order_rows)
        landmark_order_audit.to_csv(
            OUTPUT_DIR / "holdout_landmark_order_audit.csv", index=False
        )
        display(landmark_order_audit)

        paired_landmark_rows = []
        probability_column = (
            "probability"
            if "probability" in holdout_predictions.columns
            else "raw_probability"
        )
        for (target, model_variant), group in holdout_results.groupby(
            ["target", "model_variant"]
        ):
            reference_results = group.loc[group["feature_set"] == "Preoperative"]
            if reference_results.empty:
                continue
            reference_result = reference_results.iloc[0]
            reference_mask = (
                (holdout_predictions["target"] == target)
                & (holdout_predictions["feature_set"] == "Preoperative")
                & (holdout_predictions["model_variant"] == model_variant)
                & (holdout_predictions["dataset_split"] == "test")
            )
            reference_predictions = holdout_predictions.loc[
                reference_mask,
                ["source_row_index", "y_true", probability_column],
            ].rename(
                columns={
                    "y_true": "reference_y_true",
                    probability_column: "reference_probability",
                }
            )
            if reference_predictions.empty:
                continue
            if reference_predictions["source_row_index"].duplicated().any():
                raise ValueError(
                    "Duplicate preoperative test patients in paired landmark audit."
                )

            for landmark_feature_set in landmark_order[1:]:
                landmark_results = group.loc[
                    group["feature_set"] == landmark_feature_set
                ]
                if landmark_results.empty:
                    continue
                landmark_result = landmark_results.iloc[0]
                landmark_mask = (
                    (holdout_predictions["target"] == target)
                    & (holdout_predictions["feature_set"] == landmark_feature_set)
                    & (holdout_predictions["model_variant"] == model_variant)
                    & (holdout_predictions["dataset_split"] == "test")
                )
                landmark_predictions = holdout_predictions.loc[
                    landmark_mask,
                    ["source_row_index", "y_true", probability_column],
                ].rename(
                    columns={
                        "y_true": "landmark_y_true",
                        probability_column: "landmark_probability",
                    }
                )
                if landmark_predictions.empty:
                    continue
                if landmark_predictions["source_row_index"].duplicated().any():
                    raise ValueError(
                        "Duplicate later-landmark test patients in paired landmark audit."
                    )

                common = reference_predictions.merge(
                    landmark_predictions,
                    on="source_row_index",
                    how="inner",
                    validate="one_to_one",
                )
                common = common.dropna(
                    subset=[
                        "reference_y_true",
                        "landmark_y_true",
                        "reference_probability",
                        "landmark_probability",
                    ]
                ).copy()
                same_labels = bool(
                    not common.empty
                    and np.array_equal(
                        common["reference_y_true"].astype(int).to_numpy(),
                        common["landmark_y_true"].astype(int).to_numpy(),
                    )
                )
                reference_seed = result_split_seed(reference_result)
                landmark_seed = result_split_seed(landmark_result)
                comparison_seed = int(
                    (
                        BOOTSTRAP_CI_RANDOM_STATE
                        + zlib.crc32(
                            f"{target}|{model_variant}|{landmark_feature_set}|paired_landmark".encode(
                                "utf-8"
                            )
                        )
                    )
                    % (2**32)
                )
                y_common = common["landmark_y_true"].astype(int).to_numpy()
                reference_probability = (
                    common["reference_probability"].astype(float).to_numpy()
                )
                landmark_probability = (
                    common["landmark_probability"].astype(float).to_numpy()
                )
                positive_n = int(y_common.sum()) if len(y_common) else 0
                negative_n = int(len(y_common) - positive_n)

                reference_common_auroc = np.nan
                landmark_common_auroc = np.nan
                auroc_difference = np.nan
                ci_lower = np.nan
                ci_upper = np.nan
                valid_resamples = 0
                bootstrap_status = "not_run"
                evidence_of_decline = pd.NA
                evidence_of_improvement = pd.NA
                if not same_labels:
                    comparison_status = "not_comparable_labels_differ_on_common_rows"
                elif not positive_n or not negative_n:
                    comparison_status = "not_estimable_single_class_common_risk_set"
                else:
                    reference_common_auroc = roc_auc_score(
                        y_common, reference_probability
                    )
                    landmark_common_auroc = roc_auc_score(
                        y_common, landmark_probability
                    )
                    auroc_difference = landmark_common_auroc - reference_common_auroc
                    (
                        ci_lower,
                        ci_upper,
                        valid_resamples,
                        bootstrap_status,
                    ) = paired_stratified_auroc_difference_ci(
                        y_common,
                        reference_probability,
                        landmark_probability,
                        comparison_seed,
                    )
                    if bootstrap_status == "ok":
                        evidence_of_decline = bool(ci_upper < 0.0)
                        evidence_of_improvement = bool(ci_lower > 0.0)
                        if evidence_of_decline:
                            comparison_status = "evidence_of_decline"
                        elif evidence_of_improvement:
                            comparison_status = "evidence_of_improvement"
                        else:
                            comparison_status = "no_evidence_of_auroc_difference"
                    else:
                        comparison_status = "point_estimate_only_bootstrap_disabled"

                paired_landmark_rows.append(
                    {
                        "target": target,
                        "model_variant": model_variant,
                        "reference_feature_set": "Preoperative",
                        "landmark_feature_set": landmark_feature_set,
                        "reference_split_seed": reference_seed,
                        "landmark_split_seed": landmark_seed,
                        "same_split_seed": reference_seed == landmark_seed,
                        "reference_full_test_n": int(len(reference_predictions)),
                        "landmark_full_test_n": int(len(landmark_predictions)),
                        "common_test_n": int(len(common)),
                        "common_test_positive_n": positive_n,
                        "common_test_negative_n": negative_n,
                        "common_fraction_of_landmark_test": (
                            float(len(common) / len(landmark_predictions))
                            if len(landmark_predictions)
                            else np.nan
                        ),
                        "same_labels_on_common_rows": same_labels,
                        "reference_full_test_auroc": reference_result.get(
                            "test_auroc", np.nan
                        ),
                        "landmark_full_test_auroc": landmark_result.get(
                            "test_auroc", np.nan
                        ),
                        "reference_common_risk_set_auroc": reference_common_auroc,
                        "landmark_common_risk_set_auroc": landmark_common_auroc,
                        "landmark_minus_reference_auroc": auroc_difference,
                        "difference_ci_lower": ci_lower,
                        "difference_ci_upper": ci_upper,
                        "confidence_level": BOOTSTRAP_CI_CONFIDENCE_LEVEL,
                        "bootstrap_requested_resamples": (
                            BOOTSTRAP_CI_N_RESAMPLES if RUN_BOOTSTRAP_CI else 0
                        ),
                        "bootstrap_valid_resamples": valid_resamples,
                        "bootstrap_method": "paired_class_stratified_transplantation_episode_bootstrap_percentile_ci",
                        "bootstrap_seed": comparison_seed,
                        "comparison_status": comparison_status,
                        "evidence_of_decline": evidence_of_decline,
                        "evidence_of_improvement": evidence_of_improvement,
                    }
                )

        paired_landmark_audit = pd.DataFrame(
            paired_landmark_rows,
            columns=PAIRED_LANDMARK_AUDIT_COLUMNS,
        )
        paired_landmark_audit.to_csv(
            OUTPUT_DIR / "holdout_paired_landmark_comparison.csv",
            index=False,
        )
        display(paired_landmark_audit)
        paired_estimable = paired_landmark_audit.loc[
            paired_landmark_audit["comparison_status"].isin(
                [
                    "evidence_of_decline",
                    "evidence_of_improvement",
                    "no_evidence_of_auroc_difference",
                ]
            )
        ]
        paired_declines = int((paired_estimable["evidence_of_decline"] == True).sum())  # noqa: E712
        print(
            "Paired common-risk-set landmark audit: "
            f"{paired_declines}/{len(paired_estimable)} comparisons show evidence "
            f"of AUROC decline at {BOOTSTRAP_CI_CONFIDENCE_LEVEL:.0%} confidence."
        )

        severity_rows = []
        for endpoint_family in ["aGVHD", "oGVHD", "cGVHD"]:
            any_target = f"{endpoint_family}_any"
            severe_target = f"{endpoint_family}_severe"
            for feature_set in landmark_order:
                for model_variant in [MODEL_VARIANT_ORIGINAL, MODEL_VARIANT_SIMPLIFIED]:
                    any_rows = holdout_results.loc[
                        (holdout_results["target"] == any_target)
                        & (holdout_results["feature_set"] == feature_set)
                        & (holdout_results["model_variant"] == model_variant)
                    ]
                    severe_rows = holdout_results.loc[
                        (holdout_results["target"] == severe_target)
                        & (holdout_results["feature_set"] == feature_set)
                        & (holdout_results["model_variant"] == model_variant)
                    ]
                    if any_rows.empty or severe_rows.empty:
                        continue
                    any_row = any_rows.iloc[0]
                    severe_row = severe_rows.iloc[0]
                    any_auroc = float(any_row.get("test_auroc", np.nan))
                    severe_auroc = float(severe_row.get("test_auroc", np.nan))
                    any_auprc = float(any_row.get("test_auprc", np.nan))
                    severe_auprc = float(severe_row.get("test_auprc", np.nan))
                    any_seed = result_split_seed(any_row)
                    severe_seed = result_split_seed(severe_row)
                    any_signature = heldout_rows_and_labels(
                        any_target, feature_set, model_variant
                    )
                    severe_signature = heldout_rows_and_labels(
                        severe_target, feature_set, model_variant
                    )
                    any_episode_set = {row_index for row_index, _ in any_signature}
                    severe_episode_set = {
                        row_index for row_index, _ in severe_signature
                    }
                    union_episode_set = any_episode_set | severe_episode_set
                    same_test_episode_set = bool(
                        any_episode_set
                        and severe_episode_set
                        and any_episode_set == severe_episode_set
                    )
                    overlap_fraction = (
                        len(any_episode_set & severe_episode_set)
                        / len(union_episode_set)
                        if union_episode_set
                        else np.nan
                    )
                    if any_seed != severe_seed:
                        comparison_status = "descriptive_different_test_cohorts"
                    elif not same_test_episode_set:
                        comparison_status = "descriptive_different_endpoint_risk_sets"
                    else:
                        comparison_status = (
                            "descriptive_same_cohort_different_binary_outcomes"
                        )
                    severity_rows.append(
                        {
                            "endpoint_family": endpoint_family,
                            "feature_set": feature_set,
                            "model_variant": model_variant,
                            "any_stage_split_seed": any_seed,
                            "severe_split_seed": severe_seed,
                            "same_test_episode_set": same_test_episode_set,
                            "test_episode_overlap_fraction": overlap_fraction,
                            "comparison_status": comparison_status,
                            "performance_order_constraint_valid": False,
                            "any_stage_test_positive_n": any_row.get(
                                "test_positive_n", np.nan
                            ),
                            "severe_test_positive_n": severe_row.get(
                                "test_positive_n", np.nan
                            ),
                            "any_stage_test_auroc": any_auroc,
                            "severe_test_auroc": severe_auroc,
                            "severe_minus_any_stage_auroc": severe_auroc - any_auroc,
                            "severe_auroc_higher": severe_auroc > any_auroc,
                            "any_stage_test_auprc": any_auprc,
                            "severe_test_auprc": severe_auprc,
                            "severe_minus_any_stage_auprc": severe_auprc - any_auprc,
                            "any_stage_auroc_ci_lower": any_row.get(
                                "test_auroc_ci_lower", np.nan
                            ),
                            "any_stage_auroc_ci_upper": any_row.get(
                                "test_auroc_ci_upper", np.nan
                            ),
                            "severe_auroc_ci_lower": severe_row.get(
                                "test_auroc_ci_lower", np.nan
                            ),
                            "severe_auroc_ci_upper": severe_row.get(
                                "test_auroc_ci_upper", np.nan
                            ),
                            "any_stage_estimability_status": any_row.get(
                                "endpoint_estimability_status", np.nan
                            ),
                            "severe_estimability_status": severe_row.get(
                                "endpoint_estimability_status", np.nan
                            ),
                            "interpretation": (
                                "Descriptive only: severe and any-stage are separate binary outcomes, and "
                                "AUROC is not expected to be higher for severe disease. Compare cohort "
                                "overlap, event counts, and confidence intervals."
                            ),
                        }
                    )
        holdout_severity_order_audit = pd.DataFrame(
            severity_rows,
            columns=[
                "endpoint_family",
                "feature_set",
                "model_variant",
                "any_stage_split_seed",
                "severe_split_seed",
                "same_test_episode_set",
                "test_episode_overlap_fraction",
                "comparison_status",
                "performance_order_constraint_valid",
                "any_stage_test_positive_n",
                "severe_test_positive_n",
                "any_stage_test_auroc",
                "severe_test_auroc",
                "severe_minus_any_stage_auroc",
                "severe_auroc_higher",
                "any_stage_test_auprc",
                "severe_test_auprc",
                "severe_minus_any_stage_auprc",
                "any_stage_auroc_ci_lower",
                "any_stage_auroc_ci_upper",
                "severe_auroc_ci_lower",
                "severe_auroc_ci_upper",
                "any_stage_estimability_status",
                "severe_estimability_status",
                "interpretation",
            ],
        )
        holdout_severity_order_audit.to_csv(
            OUTPUT_DIR / "holdout_severity_order_audit.csv",
            index=False,
        )
        display(holdout_severity_order_audit)
    elif RUN_HOLDOUT_EVALUATION:
        pd.DataFrame(
            columns=[
                "target",
                "feature_set",
                "model_variant",
                "metric",
                "value",
                "ci_lower",
                "ci_upper",
                "confidence_level",
                "bootstrap_valid_resamples",
                "bootstrap_status",
            ]
        ).to_csv(
            OUTPUT_DIR / "holdout_metric_summary.csv",
            index=False,
        )
        pd.DataFrame(
            columns=[
                "target",
                "feature_set",
                "model_variant",
                "test_auroc",
                "test_auroc_ci_lower",
                "test_auroc_ci_upper",
                "test_auroc_bootstrap_valid_resamples",
                "selected_threshold",
            ]
        ).to_csv(
            OUTPUT_DIR / "holdout_auroc_target_check.csv",
            index=False,
        )
        pd.DataFrame(
            columns=[
                "target",
                "model_variant",
                "feature_set",
                "landmark_order",
                "test_auroc",
                "test_auroc_ci_lower",
                "test_auroc_ci_upper",
                "test_auprc_ci_lower",
                "test_auprc_ci_upper",
                "analysis_positive_n",
                "test_positive_n",
                "n_features_after_missingness_filter",
                "n_transformed_features",
                "landmark_comparison_valid",
                "landmark_comparison_reason",
                "landmark_split_seeds",
                "observed_test_auroc_non_decreasing",
                "target_is_non_decreasing",
                "target_best_feature_set",
            ]
        ).to_csv(OUTPUT_DIR / "holdout_landmark_order_audit.csv", index=False)
        pd.DataFrame(columns=PAIRED_LANDMARK_AUDIT_COLUMNS).to_csv(
            OUTPUT_DIR / "holdout_paired_landmark_comparison.csv",
            index=False,
        )
        pd.DataFrame(
            columns=[
                "endpoint_family",
                "feature_set",
                "model_variant",
                "any_stage_split_seed",
                "severe_split_seed",
                "same_test_episode_set",
                "test_episode_overlap_fraction",
                "comparison_status",
                "performance_order_constraint_valid",
                "any_stage_test_positive_n",
                "severe_test_positive_n",
                "any_stage_test_auroc",
                "severe_test_auroc",
                "severe_minus_any_stage_auroc",
                "severe_auroc_higher",
                "any_stage_test_auprc",
                "severe_test_auprc",
                "severe_minus_any_stage_auprc",
                "interpretation",
            ]
        ).to_csv(OUTPUT_DIR / "holdout_severity_order_audit.csv", index=False)
