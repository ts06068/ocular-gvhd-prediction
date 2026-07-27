"""Evaluation artifact assembly and post-processing."""

from .utils import (
    OrderedDict,
    display,
    np,
    pd,
)
from .config import (
    AUROC_TARGET,
    K_FOLD_CV_SPLITS,
    MIN_RELIABLE_TEST_CLASS_COUNT,
    OUTPUT_DIR,
    RUN_HOLDOUT_EVALUATION,
    STRICT_OPTIMIZATION_AUDIT,
    VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT,
)
from .metrics import (
    BOOTSTRAP_CI_OUTPUT_COLUMNS,
    attach_bootstrap_cis_to_results,
    bootstrap_holdout_metric_cis,
    ensure_model_variant_column,
)
from .temporal import (
    TEMPORAL_ENSEMBLE_CANDIDATE_COLUMNS,
    apply_cumulative_landmark_prediction_ensemble,
)
from .simplified import run_simplified_model_pass
from .audit import build_model_optimization_audit
from .evaluation import (
    holdout_best_param_rows,
    holdout_feature_source_candidate_rows,
    holdout_model_rows,
    holdout_optuna_trial_rows,
    holdout_prediction_rows,
    holdout_rows,
    holdout_skipped_rows,
    holdout_strategy_candidate_rows,
    holdout_threshold_candidate_rows,
    holdout_train_cv_fold_rows,
    holdout_train_cv_threshold_candidate_rows,
)


# Import-safe artifacts are replaced by run_stage().
holdout_results = pd.DataFrame()
holdout_threshold_candidates = pd.DataFrame()
holdout_train_cv_threshold_candidates = pd.DataFrame()
holdout_train_cv_folds = pd.DataFrame()
holdout_best_params = pd.DataFrame()
holdout_optuna_trials = pd.DataFrame()
holdout_strategy_candidates = pd.DataFrame()
holdout_feature_source_candidates = pd.DataFrame()
holdout_predictions = pd.DataFrame()
holdout_model_summary = pd.DataFrame()
holdout_skipped = pd.DataFrame()
holdout_simplified_feature_ranking = pd.DataFrame()
holdout_simplified_shap_cv_folds = pd.DataFrame()
holdout_temporal_ensemble_candidates = pd.DataFrame()
holdout_optimization_audit = pd.DataFrame()
holdout_bootstrap_ci = pd.DataFrame()


# Apply simplified and temporal post-processing before exporting final performance tables.
def run_stage():
    """Assemble, audit, and save evaluation artifacts."""
    global holdout_best_params, holdout_bootstrap_ci, holdout_feature_source_candidates
    global holdout_model_summary, holdout_optimization_audit, holdout_optuna_trials
    global holdout_predictions, holdout_results, holdout_simplified_feature_ranking
    global \
        holdout_simplified_shap_cv_folds, \
        holdout_skipped, \
        holdout_strategy_candidates
    global holdout_temporal_ensemble_candidates, holdout_threshold_candidates
    global holdout_train_cv_folds, holdout_train_cv_threshold_candidates

    holdout_results = pd.DataFrame(holdout_rows)

    holdout_threshold_candidates = pd.DataFrame(holdout_threshold_candidate_rows)

    holdout_train_cv_threshold_candidates = pd.DataFrame(
        holdout_train_cv_threshold_candidate_rows
    )

    holdout_train_cv_folds = pd.DataFrame(holdout_train_cv_fold_rows)

    holdout_best_params = pd.DataFrame(holdout_best_param_rows)

    holdout_optuna_trials = pd.DataFrame(holdout_optuna_trial_rows)

    holdout_strategy_candidates = pd.DataFrame(holdout_strategy_candidate_rows)

    holdout_feature_source_candidates = pd.DataFrame(
        holdout_feature_source_candidate_rows
    )

    holdout_predictions = pd.DataFrame(holdout_prediction_rows)

    holdout_model_summary = pd.DataFrame(holdout_model_rows)

    holdout_skipped = pd.DataFrame(holdout_skipped_rows)

    holdout_simplified_feature_ranking = pd.DataFrame()

    holdout_simplified_shap_cv_folds = pd.DataFrame()

    holdout_temporal_ensemble_candidates = pd.DataFrame(
        columns=TEMPORAL_ENSEMBLE_CANDIDATE_COLUMNS
    )

    holdout_optimization_audit = pd.DataFrame()

    holdout_bootstrap_ci = pd.DataFrame(columns=BOOTSTRAP_CI_OUTPUT_COLUMNS)

    if RUN_HOLDOUT_EVALUATION:
        (
            holdout_results,
            holdout_threshold_candidates,
            holdout_train_cv_threshold_candidates,
            holdout_train_cv_folds,
            holdout_best_params,
            holdout_optuna_trials,
            holdout_strategy_candidates,
            holdout_feature_source_candidates,
            holdout_predictions,
            holdout_model_summary,
            holdout_simplified_feature_ranking,
            holdout_simplified_shap_cv_folds,
        ) = run_simplified_model_pass(
            holdout_results,
            holdout_threshold_candidates,
            holdout_train_cv_threshold_candidates,
            holdout_train_cv_folds,
            holdout_best_params,
            holdout_optuna_trials,
            holdout_strategy_candidates,
            holdout_feature_source_candidates,
            holdout_predictions,
            holdout_model_summary,
        )
        holdout_results = ensure_model_variant_column(holdout_results)
        holdout_threshold_candidates = ensure_model_variant_column(
            holdout_threshold_candidates
        )
        holdout_train_cv_threshold_candidates = ensure_model_variant_column(
            holdout_train_cv_threshold_candidates
        )
        holdout_train_cv_folds = ensure_model_variant_column(holdout_train_cv_folds)
        holdout_best_params = ensure_model_variant_column(holdout_best_params)
        holdout_optuna_trials = ensure_model_variant_column(holdout_optuna_trials)
        holdout_strategy_candidates = ensure_model_variant_column(
            holdout_strategy_candidates
        )
        holdout_feature_source_candidates = ensure_model_variant_column(
            holdout_feature_source_candidates
        )
        holdout_predictions = ensure_model_variant_column(holdout_predictions)
        holdout_model_summary = ensure_model_variant_column(holdout_model_summary)
        (
            holdout_results,
            holdout_predictions,
            holdout_threshold_candidates,
            holdout_train_cv_threshold_candidates,
            holdout_model_summary,
            holdout_temporal_ensemble_candidates,
        ) = apply_cumulative_landmark_prediction_ensemble(
            holdout_results,
            holdout_predictions,
            holdout_threshold_candidates,
            holdout_train_cv_threshold_candidates,
            holdout_model_summary,
        )
        print("\nFinal reported holdout performance after temporal stabilization:")
        for _, final_row in holdout_results.iterrows():
            print(
                f"FINAL {final_row['target']:>13} | "
                f"{final_row['feature_set']:<20} | "
                f"{final_row['model_variant']:<10} | "
                f"method={final_row['probability_ensemble_method']} | "
                f"candidate={final_row['probability_ensemble_selected_candidate']} | "
                f"TEST AUROC={float(final_row['test_auroc']):.3f} | "
                f"TEST AUPRC={float(final_row['test_auprc']):.3f}"
            )
        holdout_bootstrap_ci = bootstrap_holdout_metric_cis(holdout_predictions)
        holdout_results = attach_bootstrap_cis_to_results(
            holdout_results,
            holdout_bootstrap_ci,
        )
        test_min_count = pd.to_numeric(
            holdout_results["test_min_class_count"],
            errors="coerce",
        )
        train_min_count = pd.to_numeric(
            holdout_results["train_min_class_count"],
            errors="coerce",
        )
        validation_min_count = pd.to_numeric(
            holdout_results["validation_min_class_count"],
            errors="coerce",
        )
        holdout_results["effective_train_cv_splits"] = (
            np.minimum(K_FOLD_CV_SPLITS, train_min_count).round().astype("Int64")
        )
        holdout_results["requested_train_cv_folds_available"] = (
            train_min_count >= K_FOLD_CV_SPLITS
        )
        holdout_results["validation_selection_reliable"] = (
            validation_min_count >= VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT
        )
        holdout_results["test_auroc_reliable"] = (
            test_min_count >= MIN_RELIABLE_TEST_CLASS_COUNT
        )
        holdout_results["test_auroc_reliability_status"] = np.where(
            holdout_results["test_auroc_reliable"],
            "primary_estimate",
            "exploratory_too_few_test_events_or_controls",
        )
        holdout_results["endpoint_estimability_status"] = np.select(
            [
                train_min_count < K_FOLD_CV_SPLITS,
                validation_min_count < VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT,
                test_min_count < MIN_RELIABLE_TEST_CLASS_COUNT,
            ],
            [
                "exploratory_reduced_cv_due_to_sparse_training_events",
                "exploratory_unstable_validation_selection",
                "exploratory_too_few_test_events_or_controls",
            ],
            default="primary_estimate",
        )
        holdout_results["test_auroc_ge_target_reliable"] = holdout_results[
            "test_auroc_reliable"
        ] & (
            pd.to_numeric(holdout_results["test_auroc"], errors="coerce")
            >= AUROC_TARGET
        )
        development_cv_auroc = pd.to_numeric(
            holdout_results["ensemble_train_cv_oof_auroc"],
            errors="coerce",
        )
        test_auroc = pd.to_numeric(holdout_results["test_auroc"], errors="coerce")
        test_auroc_ci_lower = pd.to_numeric(
            holdout_results["test_auroc_ci_lower"],
            errors="coerce",
        )
        test_auroc_ci_upper = pd.to_numeric(
            holdout_results["test_auroc_ci_upper"],
            errors="coerce",
        )
        holdout_results["development_cv_auroc_ge_target"] = (
            development_cv_auroc >= AUROC_TARGET
        )
        holdout_results["test_auroc_ci_contains_target"] = (
            test_auroc_ci_lower <= AUROC_TARGET
        ) & (test_auroc_ci_upper >= AUROC_TARGET)
        holdout_results["development_cv_minus_test_auroc"] = (
            development_cv_auroc - test_auroc
        )
        holdout_results["auroc_target_status"] = np.select(
            [
                holdout_results["test_auroc_ge_target_reliable"],
                ~holdout_results["test_auroc_reliable"],
                holdout_results["test_auroc_ci_contains_target"],
                holdout_results["development_cv_auroc_ge_target"],
            ],
            [
                "target_met_on_reliable_holdout",
                "exploratory_holdout_too_few_events_or_controls",
                "point_estimate_below_target_but_95ci_includes_target",
                "development_cv_met_target_but_holdout_did_not",
            ],
            default="target_not_met_on_development_or_holdout",
        )
        if not holdout_model_summary.empty and not holdout_results.empty:
            model_update_cols = [
                "target",
                "feature_set",
                "model_variant",
                *[
                    col
                    for col in holdout_results.columns
                    if col.startswith(
                        (
                            "validation_",
                            "test_",
                            "raw_validation_",
                            "raw_test_",
                            "calibrated_validation_",
                            "calibrated_test_",
                            "single_model_",
                        )
                    )
                    or col
                    in {
                        "reported_probability_source",
                        "probability_ensemble_method",
                        "probability_ensemble_members",
                        "probability_ensemble_weights",
                        "probability_ensemble_selected_candidate",
                        "probability_ensemble_selection_split",
                        "probability_ensemble_selection_rule",
                        "probability_ensemble_test_labels_used_for_selection",
                        "selected_threshold_objective",
                        "selected_threshold",
                        "selected_threshold_validation_score",
                        "train_cv_selected_threshold_objective",
                        "train_cv_selected_threshold",
                        "train_cv_selected_threshold_score",
                        "ensemble_train_cv_oof_auroc",
                        "ensemble_train_cv_oof_auprc",
                        "ensemble_train_cv_oof_auprc_lift",
                        "threshold_candidate_policy",
                        "threshold_tie_break_policy",
                        "threshold_minimum_sensitivity",
                        "threshold_operating_point_status",
                        "threshold_selection_split",
                        "threshold_probability_source_provenance",
                        "threshold_selection_labels_provenance",
                        "threshold_test_labels_used_for_selection",
                        "validation_model_selection_score",
                        "calibration_diagnostic_method",
                        "raw_validation_ece",
                        "raw_test_ece",
                    }
                ],
            ]
            model_update_cols = list(OrderedDict.fromkeys(model_update_cols))
            update_table = holdout_results.loc[:, model_update_cols]
            stale_cols = [
                col
                for col in update_table.columns
                if col not in {"target", "feature_set", "model_variant"}
                and col in holdout_model_summary.columns
            ]
            holdout_model_summary = holdout_model_summary.drop(
                columns=stale_cols, errors="ignore"
            ).merge(
                update_table, on=["target", "feature_set", "model_variant"], how="left"
            )

        holdout_optimization_audit = build_model_optimization_audit(
            holdout_results,
            holdout_strategy_candidates,
            holdout_threshold_candidates,
            holdout_optuna_trials,
            holdout_temporal_ensemble_candidates,
            holdout_predictions,
            holdout_train_cv_threshold_candidates,
            holdout_model_summary,
        )

        holdout_results.to_csv(OUTPUT_DIR / "holdout_test_rows.csv", index=False)
        holdout_threshold_candidates.to_csv(
            OUTPUT_DIR / "holdout_validation_threshold_candidates.csv", index=False
        )
        holdout_train_cv_threshold_candidates.to_csv(
            OUTPUT_DIR / "holdout_train_cv_threshold_candidates.csv", index=False
        )
        holdout_train_cv_folds.to_csv(
            OUTPUT_DIR / "holdout_train_cv_fold_preprocessing.csv", index=False
        )
        holdout_best_params.to_csv(OUTPUT_DIR / "holdout_best_params.csv", index=False)
        holdout_optuna_trials.to_csv(
            OUTPUT_DIR / "holdout_optuna_trials.csv", index=False
        )
        holdout_strategy_candidates.to_csv(
            OUTPUT_DIR / "holdout_validation_strategy_candidates.csv", index=False
        )
        holdout_feature_source_candidates.to_csv(
            OUTPUT_DIR / "holdout_feature_source_candidates.csv", index=False
        )
        holdout_predictions.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)
        holdout_bootstrap_ci.to_csv(
            OUTPUT_DIR / "holdout_bootstrap_95ci.csv", index=False
        )
        holdout_model_summary.to_csv(
            OUTPUT_DIR / "holdout_model_summary.csv", index=False
        )
        holdout_skipped.to_csv(OUTPUT_DIR / "holdout_skipped_rows.csv", index=False)
        holdout_simplified_feature_ranking.to_csv(
            OUTPUT_DIR / "holdout_simplified_feature_ranking.csv",
            index=False,
        )
        holdout_simplified_shap_cv_folds.to_csv(
            OUTPUT_DIR / "holdout_simplified_shap_cv_folds.csv",
            index=False,
        )
        holdout_temporal_ensemble_candidates.to_csv(
            OUTPUT_DIR / "holdout_temporal_ensemble_candidates.csv",
            index=False,
        )
        holdout_optimization_audit.to_csv(
            OUTPUT_DIR / "model_optimization_audit.csv",
            index=False,
        )
        holdout_model_summary.to_csv(
            OUTPUT_DIR / "final_model_summary.csv", index=False
        )
        display(holdout_results)
        display(holdout_bootstrap_ci)
        display(holdout_temporal_ensemble_candidates)
        display(holdout_optimization_audit)
        if not holdout_skipped.empty:
            display(holdout_skipped)
        if (
            STRICT_OPTIMIZATION_AUDIT
            and not holdout_optimization_audit.empty
            and not holdout_optimization_audit["optimization_audit_pass"].all()
        ):
            failed_models = holdout_optimization_audit.loc[
                ~holdout_optimization_audit["optimization_audit_pass"],
                ["target", "feature_set", "model_variant", "failed_checks"],
            ]
            raise RuntimeError(
                "Per-model optimization audit failed:\n"
                + failed_models.to_string(index=False)
            )
