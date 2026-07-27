"""Primary model evaluation workflow."""

from .utils import (
    average_precision_score,
    cloudpickle,
    json,
    jsonable,
    np,
    pd,
    roc_auc_score,
    slugify,
)
from .config import (
    ALLOW_PROBABILITY_ORIENTATION_CORRECTION,
    DUAL_DISCRIMINATION_SELECTION,
    K_FOLD_CV_SPLITS,
    LANDMARK_FEATURE_SOURCE_POLICY,
    MIN_TOTAL_POSITIVES,
    MODEL_DIR,
    OPTUNA_OBJECTIVE_METRIC,
    ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC,
    ORIENTATION_CORRECTION_TARGETS,
    OUTPUT_DIR,
    ROBUST_VALIDATION_SELECTION,
    RUN_FINAL_MODELS,
    RUN_HOLDOUT_EVALUATION,
    SPLIT_POLICY,
    SPLIT_SEED_SEARCH_ENABLED,
    TARGETS_TO_RUN,
    TARGET_TIMING_POLICY,
    THRESHOLD_CANDIDATE_POLICY,
    THRESHOLD_MIN_SENSITIVITY,
    THRESHOLD_OPERATING_POINT_STATUS,
    THRESHOLD_TIE_BREAK_POLICY,
    VALIDATION_MODEL_SELECTION_METRIC,
)
from .data import (
    FEATURE_SETS,
    FORBIDDEN_FEATURES,
    TARGET_DEFINITIONS,
    identifier_columns,
)
from .targets import (
    make_target,
    min_class_count,
    target_landmark_day_for_policy,
)
from .metrics import (
    CALIBRATION_DIAGNOSTIC_METRICS,
    analysis_supports_orientation_correction,
    apply_probability_recalibrator,
    choose_probability_source,
    classification_metrics,
    evaluate_probability_source,
    fit_probability_recalibrator,
    probability_source_provenance,
    select_threshold_from_oof,
)
from .splitting import (
    analysis_seed,
    effective_cv_repeats_for_settings,
    final_refit_scope_for_analysis,
    model_family_label,
    optimization_settings_for_analysis,
    split_random_state_base,
    train_val_test_positions_for_analysis,
)
from .tuning import (
    cross_validated_probabilities,
    fit_with_finalization_recipe,
)
from .explainability import extract_fit_info
from .temporal import (
    evaluate_feature_source_candidate,
    feature_source_candidates_for_landmark,
)
from .audit import run_development_split_seed_search


# Import-safe result containers are reset by run_stage().
development_split_seed_search = pd.DataFrame()
holdout_rows = []
holdout_threshold_candidate_rows = []
holdout_train_cv_threshold_candidate_rows = []
holdout_train_cv_fold_rows = []
holdout_best_param_rows = []
holdout_optuna_trial_rows = []
holdout_strategy_candidate_rows = []
holdout_feature_source_candidate_rows = []
holdout_prediction_rows = []
holdout_model_rows = []
holdout_skipped_rows = []


def run_stage():
    """Run the prespecified model-evaluation loop."""
    global development_split_seed_search
    global holdout_best_param_rows, holdout_feature_source_candidate_rows
    global holdout_model_rows, holdout_optuna_trial_rows, holdout_prediction_rows
    global holdout_rows, holdout_skipped_rows, holdout_strategy_candidate_rows
    global holdout_threshold_candidate_rows, holdout_train_cv_fold_rows
    global holdout_train_cv_threshold_candidate_rows

    if (
        RUN_HOLDOUT_EVALUATION
        and SPLIT_POLICY == "fixed_by_target"
        and SPLIT_SEED_SEARCH_ENABLED
    ):
        development_split_seed_search = run_development_split_seed_search()
    else:
        development_split_seed_search = pd.DataFrame()
        development_split_seed_search.to_csv(
            OUTPUT_DIR / "development_split_seed_search.csv",
            index=False,
        )

    holdout_rows = []

    holdout_threshold_candidate_rows = []

    holdout_train_cv_threshold_candidate_rows = []

    holdout_train_cv_fold_rows = []

    holdout_best_param_rows = []

    holdout_optuna_trial_rows = []

    holdout_strategy_candidate_rows = []

    holdout_feature_source_candidate_rows = []

    holdout_prediction_rows = []

    holdout_model_rows = []

    holdout_skipped_rows = []

    if RUN_HOLDOUT_EVALUATION:
        for target_name in TARGETS_TO_RUN:
            spec = TARGET_DEFINITIONS[target_name]

            for feature_set in FEATURE_SETS:
                feature_name = feature_set["feature_set"]
                landmark_day = feature_set["landmark_day"]
                split_seed_base = split_random_state_base(target_name, feature_name)
                final_refit_scope = final_refit_scope_for_analysis(
                    target_name, feature_name
                )
                seed = analysis_seed(target_name, feature_name)
                optimization_settings = optimization_settings_for_analysis(
                    target_name, feature_name
                )
                keep_mask, y_target = make_target(
                    spec["stage_column"],
                    spec["positive_rule"],
                    spec.get("exclude_control_death_3m", False),
                    spec.get("event_date_column"),
                    target_landmark_day_for_policy(landmark_day),
                )
                source_indices = np.flatnonzero(keep_mask)
                class_min_n = min_class_count(y_target)
                if class_min_n < 4 or int(y_target.sum()) < MIN_TOTAL_POSITIVES:
                    skipped_row = {
                        "target": target_name,
                        "feature_set": feature_name,
                        "landmark_day": landmark_day,
                        "event_date_column": spec.get("event_date_column"),
                        "n": int(len(y_target)),
                        "positive_n": int(y_target.sum()) if len(y_target) else 0,
                        "control_n": int(len(y_target) - y_target.sum())
                        if len(y_target)
                        else 0,
                        "min_class_count": int(class_min_n),
                        "reason": (
                            "Too few positives/controls for stable stratified 6:2:2 evaluation "
                            f"(min class < 4 or positives < {MIN_TOTAL_POSITIVES})."
                        ),
                    }
                    holdout_skipped_rows.append(skipped_row)
                    print(
                        f"SKIP {target_name:>13} | {feature_name:<20} | "
                        f"landmark_day={landmark_day} | n={skipped_row['n']} | "
                        f"positive={skipped_row['positive_n']} | reason={skipped_row['reason']}"
                    )
                    continue
                train_pos, val_pos, test_pos, split_info = (
                    train_val_test_positions_for_analysis(
                        target_name,
                        spec,
                        feature_name,
                        source_indices,
                        y_target,
                    )
                )
                split_seed = split_info["split_seed"]
                y_train = y_target.iloc[train_pos].reset_index(drop=True)
                y_val = y_target.iloc[val_pos].reset_index(drop=True)
                y_test = y_target.iloc[test_pos].reset_index(drop=True)

                split_class_counts = {
                    "train_min_class_count": min_class_count(y_train),
                    "validation_min_class_count": min_class_count(y_val),
                    "test_min_class_count": min_class_count(y_test),
                }
                if (
                    split_class_counts["train_min_class_count"] < 2
                    or split_class_counts["validation_min_class_count"] < 1
                    or split_class_counts["test_min_class_count"] < 1
                ):
                    skipped_row = {
                        "target": target_name,
                        "feature_set": feature_name,
                        "landmark_day": landmark_day,
                        "event_date_column": spec.get("event_date_column"),
                        "n": int(len(y_target)),
                        "positive_n": int(y_target.sum()) if len(y_target) else 0,
                        "control_n": int(len(y_target) - y_target.sum())
                        if len(y_target)
                        else 0,
                        "min_class_count": int(class_min_n),
                        "split_policy": split_info["split_policy"],
                        "split_seed": int(split_seed),
                        **split_class_counts,
                        "reason": "Too few positive/control samples in at least one split after landmark censoring.",
                    }
                    holdout_skipped_rows.append(skipped_row)
                    print(
                        f"SKIP {target_name:>13} | {feature_name:<20} | "
                        f"landmark_day={landmark_day} | split_seed={split_seed} | "
                        f"reason={skipped_row['reason']}"
                    )
                    continue

                feature_source_results = []
                for candidate_item in feature_source_candidates_for_landmark(
                    feature_set
                ):
                    candidate_result = evaluate_feature_source_candidate(
                        target_name=target_name,
                        current_feature_name=feature_name,
                        candidate_item=candidate_item,
                        keep_mask=keep_mask,
                        train_pos=train_pos,
                        val_pos=val_pos,
                        y_train=y_train,
                        y_val=y_val,
                        base_seed=seed,
                    )
                    feature_source_results.append(candidate_result)

                selected_feature_source_result = sorted(
                    feature_source_results,
                    key=lambda item: (
                        item["validation_selection_score"],
                        item["validation_auroc"],
                        item["candidate_landmark_day"],
                    ),
                    reverse=True,
                )[0]
                selected_feature_source = selected_feature_source_result[
                    "candidate_feature_source"
                ]
                selected_feature_variant = selected_feature_source_result[
                    "candidate_feature_variant"
                ]
                selected_feature_source_day = selected_feature_source_result[
                    "candidate_landmark_day"
                ]

                for candidate_result in feature_source_results:
                    holdout_feature_source_candidate_rows.append(
                        {
                            "target": target_name,
                            "feature_set": feature_name,
                            "landmark_day": landmark_day,
                            "candidate_feature_source": candidate_result[
                                "candidate_feature_source"
                            ],
                            "candidate_feature_variant": candidate_result[
                                "candidate_feature_variant"
                            ],
                            "candidate_landmark_day": candidate_result[
                                "candidate_landmark_day"
                            ],
                            "selected_feature_source": (
                                candidate_result["candidate_feature_source"]
                                == selected_feature_source
                            ),
                            "candidate_seed": int(candidate_result["candidate_seed"]),
                            "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
                            "candidate_validation_selection_score": candidate_result[
                                "validation_selection_score"
                            ],
                            "candidate_raw_validation_selection_score": candidate_result[
                                "raw_validation_selection_score"
                            ],
                            "candidate_validation_reliability_weight": candidate_result[
                                "validation_reliability_weight"
                            ],
                            "candidate_validation_auroc": candidate_result[
                                "validation_auroc"
                            ],
                            "candidate_validation_auprc": candidate_result[
                                "validation_auprc"
                            ],
                            "candidate_validation_auprc_lift": candidate_result[
                                "validation_auprc_lift"
                            ],
                            "candidate_selected_strategy_name": candidate_result[
                                "selected_strategy"
                            ]["strategy_name"],
                            "candidate_selected_strategy_type": candidate_result[
                                "selected_strategy"
                            ]["strategy_type"],
                            "candidate_ensemble_top_k": int(
                                len(candidate_result["selected_strategy"]["param_list"])
                            ),
                            "candidate_optuna_trials": int(
                                len(candidate_result["search"].study_.trials)
                            ),
                            "candidate_best_optuna_objective_value": float(
                                candidate_result["search"].best_score_
                            ),
                        }
                    )

                X_all = selected_feature_source_result["X_all"]
                X_train = selected_feature_source_result["X_train"]
                X_val = selected_feature_source_result["X_val"]
                X_test = X_all.iloc[test_pos].reset_index(drop=True)
                search = selected_feature_source_result["search"]
                best_params = search.best_params_
                best_trial = search.study_.best_trial
                best_cv_mean_auroc = best_trial.user_attrs.get(
                    "cv_mean_auroc", search.best_score_
                )
                best_cv_std_auroc = best_trial.user_attrs.get("cv_std_auroc", np.nan)
                best_cv_mean_auprc = best_trial.user_attrs.get("cv_mean_auprc", np.nan)
                best_cv_std_auprc = best_trial.user_attrs.get("cv_std_auprc", np.nan)
                best_cv_auprc_lift = best_trial.user_attrs.get("cv_auprc_lift", np.nan)
                selected_strategy = selected_feature_source_result["selected_strategy"]
                validation_model = selected_feature_source_result["validation_model"]
                val_raw_prob = selected_feature_source_result["val_raw_prob"]
                strategy_candidates = selected_feature_source_result[
                    "strategy_candidates"
                ]
                ensemble_param_list = selected_strategy["param_list"]
                ensemble_top_k_actual = len(ensemble_param_list)
                selected_model_family = model_family_label(ensemble_param_list)
                train_cv_raw_prob, train_cv_fold_info = cross_validated_probabilities(
                    X_train,
                    y_train,
                    ensemble_param_list,
                    seed + 17,
                    cv_role="train_threshold_cv",
                    seed_bagging_repeats=optimization_settings["seed_bagging_repeats"],
                )
                diagnostic_calibrator = fit_probability_recalibrator(
                    y_train, train_cv_raw_prob
                )
                train_cv_calibration_prob = apply_probability_recalibrator(
                    diagnostic_calibrator,
                    train_cv_raw_prob,
                )
                val_calibration_prob = apply_probability_recalibrator(
                    diagnostic_calibrator, val_raw_prob
                )

                final_refit_cv_fold_info = pd.DataFrame()
                if final_refit_scope == "train_validation":
                    X_final_fit = pd.concat([X_train, X_val], ignore_index=True)
                    y_final_fit = pd.concat([y_train, y_val], ignore_index=True)
                    holdout_model = fit_with_finalization_recipe(
                        X_final_fit,
                        y_final_fit,
                        ensemble_param_list,
                        seed + 104729,
                        selected_strategy["finalization_strategy"],
                    )
                    test_raw_prob = holdout_model.predict_proba(X_test)[:, 1]
                    final_cv_raw_prob, final_refit_cv_fold_info = (
                        cross_validated_probabilities(
                            X_final_fit,
                            y_final_fit,
                            ensemble_param_list,
                            seed + 131071,
                            cv_role="train_validation_calibration_cv",
                            seed_bagging_repeats=optimization_settings[
                                "seed_bagging_repeats"
                            ],
                        )
                    )
                    test_diagnostic_calibrator = fit_probability_recalibrator(
                        y_final_fit, final_cv_raw_prob
                    )
                    final_fit_info_input = X_final_fit
                    final_fit_n = len(y_final_fit)
                    final_fit_positive_n = int(y_final_fit.sum())
                    test_calibration_fit_scope = "train_validation_cv"
                else:
                    holdout_model = validation_model
                    test_raw_prob = holdout_model.predict_proba(X_test)[:, 1]
                    test_diagnostic_calibrator = diagnostic_calibrator
                    final_fit_info_input = X_train
                    final_fit_n = len(y_train)
                    final_fit_positive_n = int(y_train.sum())
                    test_calibration_fit_scope = "train_cv"

                test_calibration_prob = apply_probability_recalibrator(
                    test_diagnostic_calibrator, test_raw_prob
                )
                if not final_refit_cv_fold_info.empty:
                    train_cv_fold_info = pd.concat(
                        [train_cv_fold_info, final_refit_cv_fold_info],
                        ignore_index=True,
                    )

                # Classification and thresholding use raw scores; recalibration is
                # reserved for calibration diagnostics and plots.
                probability_source_arrays = {
                    "raw": (train_cv_raw_prob, val_raw_prob, test_raw_prob),
                }
                raw_eval = evaluate_probability_source(
                    y_val, val_raw_prob, y_test, test_raw_prob, "raw"
                )
                calibrated_eval = evaluate_probability_source(
                    y_val,
                    val_calibration_prob,
                    y_test,
                    test_calibration_prob,
                    "calibrated",
                )
                probability_source_evaluations = [raw_eval]
                inverted_diagnostic_calibrator = None
                inverted_test_diagnostic_calibrator = None
                if (
                    ALLOW_PROBABILITY_ORIENTATION_CORRECTION
                    and target_name in ORIENTATION_CORRECTION_TARGETS
                    and analysis_supports_orientation_correction(
                        target_name,
                        feature_name,
                        y_train,
                        train_cv_raw_prob,
                    )
                    and np.isfinite(raw_eval["validation_metrics"].get("auroc", np.nan))
                    and raw_eval["validation_metrics"]["auroc"]
                    < ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC
                ):
                    inverted_diagnostic_calibrator = fit_probability_recalibrator(
                        y_train,
                        1.0 - train_cv_raw_prob,
                    )
                    probability_source_arrays["raw_inverted"] = (
                        1.0 - train_cv_raw_prob,
                        1.0 - val_raw_prob,
                        1.0 - test_raw_prob,
                    )
                    probability_source_evaluations.append(
                        evaluate_probability_source(
                            y_val,
                            probability_source_arrays["raw_inverted"][1],
                            y_test,
                            probability_source_arrays["raw_inverted"][2],
                            "raw_inverted",
                        )
                    )
                selected_report = choose_probability_source(
                    probability_source_evaluations
                )
                selected_threshold = selected_report["selected_threshold"]
                threshold = selected_report["threshold"]
                val_metrics = selected_report["validation_metrics"]
                test_metrics = selected_report["test_metrics"]
                threshold_candidates = pd.concat(
                    [
                        item["threshold_candidates"]
                        for item in probability_source_evaluations
                    ],
                    ignore_index=True,
                )
                raw_val_metrics = raw_eval["validation_metrics"]
                raw_test_metrics = raw_eval["test_metrics"]
                calibrated_val_metrics = calibrated_eval["validation_metrics"]
                calibrated_test_metrics = calibrated_eval["test_metrics"]
                raw_val_ece = raw_val_metrics["ece"]
                raw_test_ece = raw_test_metrics["ece"]
                selected_train_prob, selected_val_prob, selected_test_prob = (
                    probability_source_arrays[selected_report["probability_source"]]
                )
                if selected_report["probability_source"].endswith("_inverted"):
                    selected_train_calibration_prob = apply_probability_recalibrator(
                        inverted_diagnostic_calibrator,
                        selected_train_prob,
                    )
                    selected_val_calibration_prob = apply_probability_recalibrator(
                        inverted_diagnostic_calibrator,
                        selected_val_prob,
                    )
                    if final_refit_scope == "train_validation":
                        inverted_test_diagnostic_calibrator = (
                            fit_probability_recalibrator(
                                y_final_fit,
                                1.0 - final_cv_raw_prob,
                            )
                        )
                    else:
                        inverted_test_diagnostic_calibrator = (
                            inverted_diagnostic_calibrator
                        )
                    selected_test_calibration_prob = apply_probability_recalibrator(
                        inverted_test_diagnostic_calibrator,
                        selected_test_prob,
                    )
                    selected_validation_diagnostic_calibrator = (
                        inverted_diagnostic_calibrator
                    )
                    selected_test_diagnostic_calibrator = (
                        inverted_test_diagnostic_calibrator
                    )
                else:
                    selected_train_calibration_prob = train_cv_calibration_prob
                    selected_val_calibration_prob = val_calibration_prob
                    selected_test_calibration_prob = test_calibration_prob
                    selected_validation_diagnostic_calibrator = diagnostic_calibrator
                    selected_test_diagnostic_calibrator = test_diagnostic_calibrator
                if selected_report["probability_source"].endswith("_inverted"):
                    # Keep stored diagnostics aligned with the inverted
                    # calibration_probability emitted for validation and test.
                    calibrated_val_metrics = classification_metrics(
                        y_val,
                        selected_val_calibration_prob,
                        0.5,
                    )
                    calibrated_test_metrics = classification_metrics(
                        y_test,
                        selected_test_calibration_prob,
                        0.5,
                    )
                train_cv_selected, train_cv_candidates = select_threshold_from_oof(
                    y_train,
                    selected_train_prob,
                    calibration_prob=selected_train_calibration_prob,
                )
                train_cv_candidates = train_cv_candidates.copy()
                train_cv_candidates.insert(
                    0, "probability_source", selected_report["probability_source"]
                )
                train_cv_candidates.insert(
                    1,
                    "probability_source_provenance",
                    probability_source_provenance(
                        selected_report["probability_source"]
                    ),
                )
                train_cv_candidates.insert(
                    2, "selection_labels_provenance", "train_oof_labels"
                )
                ensemble_train_cv_auroc = (
                    roc_auc_score(y_train, selected_train_prob)
                    if len(np.unique(y_train)) == 2
                    else np.nan
                )
                ensemble_train_cv_auprc = average_precision_score(
                    y_train, selected_train_prob
                )
                ensemble_train_cv_prevalence = (
                    float(y_train.mean()) if len(y_train) else np.nan
                )
                ensemble_train_cv_auprc_lift = (
                    float(ensemble_train_cv_auprc / ensemble_train_cv_prevalence)
                    if ensemble_train_cv_prevalence
                    else np.nan
                )
                fit_info = extract_fit_info(holdout_model, final_fit_info_input)

                row = {
                    "target": target_name,
                    "feature_set": feature_name,
                    "modeling_method": optimization_settings["modeling_method"],
                    "selected_model_family": selected_model_family,
                    "landmark_day": landmark_day,
                    "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
                    "selected_feature_source": selected_feature_source,
                    "selected_feature_variant": selected_feature_variant,
                    "selected_feature_source_day": selected_feature_source_day,
                    "event_date_column": spec.get("event_date_column"),
                    "target_timing_policy": TARGET_TIMING_POLICY,
                    "analysis_n": int(len(y_target)),
                    "analysis_positive_n": int(y_target.sum()),
                    "analysis_control_n": int(len(y_target) - y_target.sum()),
                    "split_random_state_base": int(split_seed_base),
                    "split_policy": split_info["split_policy"],
                    "split_seed": int(split_seed),
                    "split_reference_n": int(split_info["split_reference_n"]),
                    "split_reference_positive_n": int(
                        split_info["split_reference_positive_n"]
                    ),
                    "analysis_seed": int(seed),
                    "train_n": int(len(train_pos)),
                    "validation_n": int(len(val_pos)),
                    "test_n": int(len(test_pos)),
                    "train_positive_n": int(y_train.sum()),
                    "validation_positive_n": int(y_val.sum()),
                    "test_positive_n": int(y_test.sum()),
                    **split_class_counts,
                    "best_optuna_objective_value": float(search.best_score_),
                    "best_train_cv_auroc": float(best_cv_mean_auroc),
                    "best_train_cv_mean_auroc": float(best_cv_mean_auroc),
                    "best_train_cv_std_auroc": float(best_cv_std_auroc),
                    "best_train_cv_mean_auprc": float(best_cv_mean_auprc),
                    "best_train_cv_std_auprc": float(best_cv_std_auprc),
                    "best_train_cv_auprc_lift": float(best_cv_auprc_lift),
                    "ensemble_train_cv_oof_auroc": float(ensemble_train_cv_auroc),
                    "ensemble_train_cv_oof_auprc": float(ensemble_train_cv_auprc),
                    "ensemble_train_cv_oof_auprc_lift": float(
                        ensemble_train_cv_auprc_lift
                    ),
                    "optuna_trials": int(len(search.study_.trials)),
                    "selected_strategy_name": selected_strategy["strategy_name"],
                    "selected_strategy_type": selected_strategy["strategy_type"],
                    "selected_strategy_validation_auroc": float(
                        strategy_candidates.loc[
                            strategy_candidates["candidate_strategy_name"]
                            == selected_strategy["strategy_name"],
                            "candidate_validation_auroc",
                        ].iloc[0]
                    ),
                    "selected_strategy_validation_selection_metric": (
                        "pareto_near_best_auroc_then_auprc"
                        if DUAL_DISCRIMINATION_SELECTION
                        else f"cv_validation_shrinkage_{VALIDATION_MODEL_SELECTION_METRIC}"
                        if ROBUST_VALIDATION_SELECTION
                        else VALIDATION_MODEL_SELECTION_METRIC
                    ),
                    "selected_strategy_validation_selection_score": float(
                        strategy_candidates.loc[
                            strategy_candidates["candidate_strategy_name"]
                            == selected_strategy["strategy_name"],
                            "candidate_robust_selection_score",
                        ].iloc[0]
                    ),
                    "selected_strategy_raw_validation_selection_score": float(
                        strategy_candidates.loc[
                            strategy_candidates["candidate_strategy_name"]
                            == selected_strategy["strategy_name"],
                            "candidate_validation_selection_score",
                        ].iloc[0]
                    ),
                    "selected_strategy_validation_reliability_weight": float(
                        strategy_candidates.loc[
                            strategy_candidates["candidate_strategy_name"]
                            == selected_strategy["strategy_name"],
                            "candidate_validation_reliability_weight",
                        ].iloc[0]
                    ),
                    "ensemble_top_k": int(ensemble_top_k_actual),
                    "performance_rescue": bool(
                        optimization_settings["performance_rescue"]
                    ),
                    "configured_optuna_trials": int(
                        optimization_settings["optuna_trials"]
                    ),
                    "optuna_cv_repeats": effective_cv_repeats_for_settings(
                        y_train, optimization_settings
                    ),
                    "configured_feature_selection_candidates": "|".join(
                        optimization_settings["feature_selection_candidates"]
                    ),
                    "configured_finalization_candidates": "|".join(
                        optimization_settings["finalization_candidates"]
                    ),
                    "validation_candidate_trials": int(
                        optimization_settings["validation_candidate_trials"]
                    ),
                    "seed_bagging_repeats": int(
                        optimization_settings["seed_bagging_repeats"]
                    ),
                    "ensemble_model_members": int(
                        ensemble_top_k_actual
                        * optimization_settings["seed_bagging_repeats"]
                    ),
                    "selected_finalization_strategy": selected_strategy[
                        "finalization_strategy"
                    ],
                    "finalization_validation_auroc": selected_strategy[
                        "finalization_validation_auroc"
                    ],
                    "finalization_validation_auprc": selected_strategy[
                        "finalization_validation_auprc"
                    ],
                    "finalization_selection_policy": selected_strategy[
                        "finalization_selection_policy"
                    ],
                    "finalization_model_members": int(
                        selected_strategy["finalization_model_members"]
                    ),
                    "finalization_validation_candidates_json": selected_strategy[
                        "finalization_candidates_json"
                    ],
                    "final_refit_scope": final_refit_scope,
                    "final_fit_n": int(final_fit_n),
                    "final_fit_positive_n": int(final_fit_positive_n),
                    "test_calibration_fit_scope": test_calibration_fit_scope,
                    "reported_probability_source": selected_report[
                        "probability_source"
                    ],
                    "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
                    "validation_model_selection_score": selected_report[
                        "validation_model_selection_score"
                    ],
                    "selected_threshold_objective": selected_threshold[
                        "candidate_objective"
                    ],
                    "selected_threshold": float(threshold),
                    "selected_threshold_validation_score": selected_threshold[
                        "candidate_selection_score"
                    ],
                    "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
                    "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
                    "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
                    "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
                    "threshold_selection_split": "validation",
                    "threshold_probability_source_provenance": probability_source_provenance(
                        selected_report["probability_source"]
                    ),
                    "threshold_selection_labels_provenance": "validation_labels",
                    "threshold_test_labels_used_for_selection": False,
                    "train_cv_selected_threshold_objective": train_cv_selected[
                        "candidate_objective"
                    ],
                    "train_cv_selected_threshold": float(
                        train_cv_selected["candidate_threshold"]
                    ),
                    "train_cv_selected_threshold_score": train_cv_selected[
                        "candidate_selection_score"
                    ],
                    "calibration_diagnostic_method": selected_test_diagnostic_calibrator[
                        "method"
                    ],
                    "n_features_before_missingness_filter": fit_info[
                        "n_features_before_missingness_filter"
                    ],
                    "n_features_after_missingness_filter": fit_info[
                        "n_features_after_missingness_filter"
                    ],
                    "n_features_after_mice_preprocessing": fit_info[
                        "n_features_after_mice_preprocessing"
                    ],
                    "feature_selection_k": fit_info["feature_selection_k"],
                    "n_features_after_feature_selection": fit_info[
                        "n_features_after_feature_selection"
                    ],
                    "n_transformed_features": fit_info["n_transformed_features"],
                    "dropped_missingness_columns": "|".join(
                        fit_info["dropped_missingness_columns"]
                    ),
                    "dropped_unusable_columns": "|".join(
                        fit_info["dropped_unusable_columns"]
                    ),
                    "raw_validation_ece": raw_val_ece,
                    "raw_test_ece": raw_test_ece,
                }
                row.update(
                    {
                        f"raw_validation_{key}": value
                        for key, value in raw_val_metrics.items()
                    }
                )
                row.update(
                    {
                        f"raw_test_{key}": value
                        for key, value in raw_test_metrics.items()
                    }
                )
                row.update(
                    {
                        f"calibrated_validation_{key}": value
                        for key, value in calibrated_val_metrics.items()
                    }
                )
                row.update(
                    {
                        f"calibrated_test_{key}": value
                        for key, value in calibrated_test_metrics.items()
                    }
                )
                row.update(
                    {f"validation_{key}": value for key, value in val_metrics.items()}
                )
                row.update(
                    {f"test_{key}": value for key, value in test_metrics.items()}
                )
                holdout_rows.append(row)

                threshold_candidates = threshold_candidates.copy()
                threshold_candidates.insert(0, "target", target_name)
                threshold_candidates.insert(1, "feature_set", feature_name)
                threshold_candidates.insert(2, "landmark_day", landmark_day)
                threshold_candidates.insert(3, "selection_split", "validation")
                holdout_threshold_candidate_rows.extend(
                    threshold_candidates.to_dict("records")
                )

                train_cv_candidates = train_cv_candidates.copy()
                train_cv_candidates.insert(0, "target", target_name)
                train_cv_candidates.insert(1, "feature_set", feature_name)
                train_cv_candidates.insert(2, "landmark_day", landmark_day)
                train_cv_candidates.insert(3, "selection_split", "train_cv")
                holdout_train_cv_threshold_candidate_rows.extend(
                    train_cv_candidates.to_dict("records")
                )

                train_cv_fold_info = train_cv_fold_info.copy()
                train_cv_fold_info.insert(0, "target", target_name)
                train_cv_fold_info.insert(1, "feature_set", feature_name)
                train_cv_fold_info.insert(2, "landmark_day", landmark_day)
                holdout_train_cv_fold_rows.extend(train_cv_fold_info.to_dict("records"))

                strategy_candidates = strategy_candidates.copy()
                strategy_candidates.insert(0, "target", target_name)
                strategy_candidates.insert(1, "feature_set", feature_name)
                strategy_candidates.insert(2, "landmark_day", landmark_day)
                strategy_candidates.insert(
                    3, "selected_feature_source", selected_feature_source
                )
                strategy_candidates.insert(
                    4,
                    "selected_strategy",
                    strategy_candidates["candidate_strategy_name"]
                    == selected_strategy["strategy_name"],
                )
                holdout_strategy_candidate_rows.extend(
                    strategy_candidates.to_dict("records")
                )

                best_param_row = {
                    "target": target_name,
                    "feature_set": feature_name,
                    "landmark_day": landmark_day,
                    "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
                    "selected_feature_source": selected_feature_source,
                    "selected_feature_source_day": selected_feature_source_day,
                    "best_optuna_objective_value": float(search.best_score_),
                    "best_train_cv_auroc": float(best_cv_mean_auroc),
                    "best_train_cv_mean_auroc": float(best_cv_mean_auroc),
                    "best_train_cv_std_auroc": float(best_cv_std_auroc),
                    "best_train_cv_mean_auprc": float(best_cv_mean_auprc),
                    "best_train_cv_std_auprc": float(best_cv_std_auprc),
                    "best_train_cv_auprc_lift": float(best_cv_auprc_lift),
                    "ensemble_train_cv_oof_auroc": float(ensemble_train_cv_auroc),
                    "ensemble_train_cv_oof_auprc": float(ensemble_train_cv_auprc),
                    "ensemble_train_cv_oof_auprc_lift": float(
                        ensemble_train_cv_auprc_lift
                    ),
                    "optuna_trials": int(len(search.study_.trials)),
                    "selected_strategy_name": selected_strategy["strategy_name"],
                    "selected_strategy_type": selected_strategy["strategy_type"],
                    "selected_strategy_validation_auroc": float(
                        strategy_candidates.loc[
                            strategy_candidates["candidate_strategy_name"]
                            == selected_strategy["strategy_name"],
                            "candidate_validation_auroc",
                        ].iloc[0]
                    ),
                    "ensemble_top_k": int(ensemble_top_k_actual),
                    "final_refit_scope": final_refit_scope,
                    "calibration_diagnostic_method": selected_test_diagnostic_calibrator[
                        "method"
                    ],
                    **best_params,
                }
                holdout_best_param_rows.append(best_param_row)

                for trial in search.study_.trials:
                    if trial.value is None:
                        continue
                    trial_row = {
                        "target": target_name,
                        "feature_set": feature_name,
                        "landmark_day": landmark_day,
                        "selected_feature_source": selected_feature_source,
                        "trial_number": int(trial.number),
                        "trial_state": str(trial.state),
                        "optuna_objective_value": float(trial.value),
                        "cv_mean_auroc": trial.user_attrs.get("cv_mean_auroc", np.nan),
                        "cv_std_auroc": trial.user_attrs.get("cv_std_auroc", np.nan),
                        "cv_mean_auprc": trial.user_attrs.get("cv_mean_auprc", np.nan),
                        "cv_std_auprc": trial.user_attrs.get("cv_std_auprc", np.nan),
                        "cv_prevalence": trial.user_attrs.get("cv_prevalence", np.nan),
                        "cv_auprc_lift": trial.user_attrs.get("cv_auprc_lift", np.nan),
                        "cv_selection_metric": trial.user_attrs.get(
                            "cv_selection_metric", OPTUNA_OBJECTIVE_METRIC
                        ),
                    }
                    trial_row.update(trial.params)
                    holdout_optuna_trial_rows.append(trial_row)

                for (
                    split_name,
                    split_pos,
                    split_y,
                    raw_prob,
                    cal_prob,
                    report_prob,
                    split_threshold,
                ) in [
                    (
                        "train_oof",
                        train_pos,
                        y_train,
                        train_cv_raw_prob,
                        selected_train_calibration_prob,
                        selected_train_prob,
                        float(train_cv_selected["candidate_threshold"]),
                    ),
                    (
                        "validation",
                        val_pos,
                        y_val,
                        val_raw_prob,
                        selected_val_calibration_prob,
                        selected_val_prob,
                        threshold,
                    ),
                    (
                        "test",
                        test_pos,
                        y_test,
                        test_raw_prob,
                        selected_test_calibration_prob,
                        selected_test_prob,
                        threshold,
                    ),
                ]:
                    y_pred = (report_prob >= split_threshold).astype(int)
                    for local_i, row_pos in enumerate(split_pos):
                        holdout_prediction_rows.append(
                            {
                                "target": target_name,
                                "feature_set": feature_name,
                                "selected_feature_source": selected_feature_source,
                                "landmark_day": landmark_day,
                                "dataset_split": split_name,
                                "analysis_row_index": int(row_pos),
                                "source_row_index": int(source_indices[row_pos]),
                                "y_true": int(split_y.iloc[local_i]),
                                "reported_probability_source": selected_report[
                                    "probability_source"
                                ],
                                "raw_probability": float(raw_prob[local_i]),
                                "probability": float(report_prob[local_i]),
                                "calibration_probability": float(cal_prob[local_i]),
                                "threshold": float(split_threshold),
                                "y_pred": int(y_pred[local_i]),
                            }
                        )

                if RUN_FINAL_MODELS:
                    model_slug = f"{slugify(target_name)}__{slugify(feature_name)}"
                    model_path = MODEL_DIR / f"{model_slug}.pkl"
                    metadata_path = MODEL_DIR / f"{model_slug}_metadata.json"

                    bundle = {
                        "model": holdout_model,
                        "modeling_method": optimization_settings["modeling_method"],
                        "selected_model_family": selected_model_family,
                        "calibration_diagnostic_recalibrator": selected_test_diagnostic_calibrator,
                        "validation_selection_recalibrator": selected_validation_diagnostic_calibrator,
                        "calibration_diagnostic_method": selected_test_diagnostic_calibrator[
                            "method"
                        ],
                        "target": target_name,
                        "target_definition": spec,
                        "target_timing_policy": TARGET_TIMING_POLICY,
                        "feature_set": feature_name,
                        "landmark_day": landmark_day,
                        "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
                        "selected_feature_source": selected_feature_source,
                        "selected_feature_source_day": selected_feature_source_day,
                        "feature_columns_before_filter": selected_feature_source_result[
                            "candidate_columns"
                        ],
                        "analysis_source_indices": source_indices.tolist(),
                        "split_random_state_base": int(split_seed_base),
                        "split_policy": split_info["split_policy"],
                        "split_seed": int(split_seed),
                        "split_reference_n": int(split_info["split_reference_n"]),
                        "split_reference_positive_n": int(
                            split_info["split_reference_positive_n"]
                        ),
                        "analysis_seed": int(seed),
                        "train_indices": train_pos.tolist(),
                        "validation_indices": val_pos.tolist(),
                        "test_indices": test_pos.tolist(),
                        "threshold": float(threshold),
                        "threshold_objective": selected_threshold[
                            "candidate_objective"
                        ],
                        "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
                        "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
                        "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
                        "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
                        "threshold_selection_split": "validation",
                        "threshold_probability_source_provenance": probability_source_provenance(
                            selected_report["probability_source"]
                        ),
                        "threshold_selection_labels_provenance": "validation_labels",
                        "threshold_test_labels_used_for_selection": False,
                        "train_oof_threshold": float(
                            train_cv_selected["candidate_threshold"]
                        ),
                        "train_oof_threshold_objective": train_cv_selected[
                            "candidate_objective"
                        ],
                        "reported_probability_source": selected_report[
                            "probability_source"
                        ],
                        "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
                        "probability_output": "reported_probability",
                        "raw_probability_output": "model_class1_probability",
                        "classification_probability_provenance": probability_source_provenance(
                            selected_report["probability_source"]
                        ),
                        "calibration_probability": "post_hoc_recalibrated_for_calibration_metrics_and_plots",
                        "best_params": best_params,
                        "ensemble_params": ensemble_param_list,
                        "optuna_objective_metric": OPTUNA_OBJECTIVE_METRIC,
                        "selected_strategy_name": selected_strategy["strategy_name"],
                        "selected_strategy_type": selected_strategy["strategy_type"],
                        "ensemble_top_k": int(ensemble_top_k_actual),
                        "selected_finalization_strategy": selected_strategy[
                            "finalization_strategy"
                        ],
                        "finalization_model_members": int(
                            selected_strategy["finalization_model_members"]
                        ),
                        "finalization_validation_candidates": json.loads(
                            selected_strategy["finalization_candidates_json"]
                        ),
                        "final_refit_scope": final_refit_scope,
                        "final_fit_n": int(final_fit_n),
                        "final_fit_positive_n": int(final_fit_positive_n),
                        "test_calibration_fit_scope": test_calibration_fit_scope,
                        "fit_info": fit_info,
                        "forbidden_features": sorted(FORBIDDEN_FEATURES),
                        "identifier_columns_removed": identifier_columns,
                    }
                    with open(model_path, "wb") as f:
                        cloudpickle.dump(bundle, f)

                    metadata = {
                        "target": target_name,
                        "feature_set": feature_name,
                        "modeling_method": optimization_settings["modeling_method"],
                        "selected_model_family": selected_model_family,
                        "landmark_day": landmark_day,
                        "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
                        "selected_feature_source": selected_feature_source,
                        "selected_feature_source_day": selected_feature_source_day,
                        "event_date_column": spec.get("event_date_column"),
                        "target_timing_policy": TARGET_TIMING_POLICY,
                        "model_path": str(model_path),
                        "split_random_state_base": int(split_seed_base),
                        "split_policy": split_info["split_policy"],
                        "split_seed": int(split_seed),
                        "split_reference_n": int(split_info["split_reference_n"]),
                        "split_reference_positive_n": int(
                            split_info["split_reference_positive_n"]
                        ),
                        "analysis_seed": int(seed),
                        "threshold": float(threshold),
                        "threshold_objective": selected_threshold[
                            "candidate_objective"
                        ],
                        "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
                        "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
                        "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
                        "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
                        "threshold_selection_split": "validation",
                        "threshold_probability_source_provenance": probability_source_provenance(
                            selected_report["probability_source"]
                        ),
                        "threshold_selection_labels_provenance": "validation_labels",
                        "threshold_test_labels_used_for_selection": False,
                        "train_oof_threshold": float(
                            train_cv_selected["candidate_threshold"]
                        ),
                        "train_oof_threshold_objective": train_cv_selected[
                            "candidate_objective"
                        ],
                        "calibration_diagnostic_method": selected_test_diagnostic_calibrator[
                            "method"
                        ],
                        "reported_probability_source": selected_report[
                            "probability_source"
                        ],
                        "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
                        "validation_model_selection_score": selected_report[
                            "validation_model_selection_score"
                        ],
                        "probability_output": "reported_probability",
                        "raw_probability_output": "model_class1_probability",
                        "classification_probability_provenance": probability_source_provenance(
                            selected_report["probability_source"]
                        ),
                        "calibration_probability": "post_hoc_recalibrated_for_calibration_metrics_and_plots",
                        "best_optuna_objective_value": float(search.best_score_),
                        "best_train_cv_auroc_for_hyperparameter_tuning": float(
                            best_cv_mean_auroc
                        ),
                        "best_train_cv_mean_auroc_for_hyperparameter_tuning": float(
                            best_cv_mean_auroc
                        ),
                        "best_train_cv_std_auroc_for_hyperparameter_tuning": float(
                            best_cv_std_auroc
                        ),
                        "best_train_cv_mean_auprc_for_hyperparameter_tuning": float(
                            best_cv_mean_auprc
                        ),
                        "best_train_cv_std_auprc_for_hyperparameter_tuning": float(
                            best_cv_std_auprc
                        ),
                        "best_train_cv_auprc_lift_for_hyperparameter_tuning": float(
                            best_cv_auprc_lift
                        ),
                        "optuna_objective_metric": OPTUNA_OBJECTIVE_METRIC,
                        "ensemble_train_cv_oof_auroc": float(ensemble_train_cv_auroc),
                        "ensemble_train_cv_oof_auprc": float(ensemble_train_cv_auprc),
                        "ensemble_train_cv_oof_auprc_lift": float(
                            ensemble_train_cv_auprc_lift
                        ),
                        "optuna_trials": int(len(search.study_.trials)),
                        "selected_strategy_name": selected_strategy["strategy_name"],
                        "selected_strategy_type": selected_strategy["strategy_type"],
                        "ensemble_top_k": int(ensemble_top_k_actual),
                        "selected_finalization_strategy": selected_strategy[
                            "finalization_strategy"
                        ],
                        "finalization_model_members": int(
                            selected_strategy["finalization_model_members"]
                        ),
                        "final_refit_scope": final_refit_scope,
                        "final_fit_n": int(final_fit_n),
                        "final_fit_positive_n": int(final_fit_positive_n),
                        "test_calibration_fit_scope": test_calibration_fit_scope,
                        "validation_metrics": {
                            key: jsonable(value) for key, value in val_metrics.items()
                        },
                        "test_metrics": {
                            key: jsonable(value) for key, value in test_metrics.items()
                        },
                        "raw_validation_metrics": {
                            key: jsonable(value)
                            for key, value in raw_val_metrics.items()
                        },
                        "raw_test_metrics": {
                            key: jsonable(value)
                            for key, value in raw_test_metrics.items()
                        },
                        "calibrated_validation_metrics": {
                            key: jsonable(value)
                            for key, value in calibrated_val_metrics.items()
                        },
                        "calibrated_test_metrics": {
                            key: jsonable(value)
                            for key, value in calibrated_test_metrics.items()
                        },
                        "raw_validation_ece": jsonable(raw_val_ece),
                        "raw_test_ece": jsonable(raw_test_ece),
                        "best_params": {
                            key: jsonable(value) for key, value in best_params.items()
                        },
                        "ensemble_params": [
                            {key: jsonable(value) for key, value in params.items()}
                            for params in ensemble_param_list
                        ],
                        "fit_info": {
                            key: jsonable(value)
                            for key, value in fit_info.items()
                            if key != "transformed_feature_names"
                        },
                    }
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)

                    model_row = {
                        "target": target_name,
                        "feature_set": feature_name,
                        "modeling_method": optimization_settings["modeling_method"],
                        "selected_model_family": selected_model_family,
                        "landmark_day": landmark_day,
                        "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
                        "selected_feature_source": selected_feature_source,
                        "selected_feature_source_day": selected_feature_source_day,
                        "event_date_column": spec.get("event_date_column"),
                        "target_timing_policy": TARGET_TIMING_POLICY,
                        "model_path": str(model_path),
                        "metadata_path": str(metadata_path),
                        "split_random_state_base": int(split_seed_base),
                        "split_policy": split_info["split_policy"],
                        "split_seed": int(split_seed),
                        "split_reference_n": int(split_info["split_reference_n"]),
                        "split_reference_positive_n": int(
                            split_info["split_reference_positive_n"]
                        ),
                        "analysis_seed": int(seed),
                        "best_optuna_objective_value": float(search.best_score_),
                        "best_train_cv_auroc_for_hyperparameter_tuning": float(
                            best_cv_mean_auroc
                        ),
                        "best_train_cv_mean_auroc_for_hyperparameter_tuning": float(
                            best_cv_mean_auroc
                        ),
                        "best_train_cv_std_auroc_for_hyperparameter_tuning": float(
                            best_cv_std_auroc
                        ),
                        "best_train_cv_mean_auprc_for_hyperparameter_tuning": float(
                            best_cv_mean_auprc
                        ),
                        "best_train_cv_std_auprc_for_hyperparameter_tuning": float(
                            best_cv_std_auprc
                        ),
                        "best_train_cv_auprc_lift_for_hyperparameter_tuning": float(
                            best_cv_auprc_lift
                        ),
                        "optuna_objective_metric": OPTUNA_OBJECTIVE_METRIC,
                        "ensemble_train_cv_oof_auroc": float(ensemble_train_cv_auroc),
                        "ensemble_train_cv_oof_auprc": float(ensemble_train_cv_auprc),
                        "ensemble_train_cv_oof_auprc_lift": float(
                            ensemble_train_cv_auprc_lift
                        ),
                        "optuna_trials": int(len(search.study_.trials)),
                        "selected_strategy_name": selected_strategy["strategy_name"],
                        "selected_strategy_type": selected_strategy["strategy_type"],
                        "ensemble_top_k": int(ensemble_top_k_actual),
                        "selected_finalization_strategy": selected_strategy[
                            "finalization_strategy"
                        ],
                        "finalization_model_members": int(
                            selected_strategy["finalization_model_members"]
                        ),
                        "final_refit_scope": final_refit_scope,
                        "final_fit_n": int(final_fit_n),
                        "final_fit_positive_n": int(final_fit_positive_n),
                        "test_calibration_fit_scope": test_calibration_fit_scope,
                        "reported_probability_source": selected_report[
                            "probability_source"
                        ],
                        "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
                        "validation_model_selection_score": selected_report[
                            "validation_model_selection_score"
                        ],
                        "calibration_diagnostic_method": selected_test_diagnostic_calibrator[
                            "method"
                        ],
                        "selected_threshold_objective": selected_threshold[
                            "candidate_objective"
                        ],
                        "selected_threshold": float(threshold),
                        "selected_threshold_validation_score": selected_threshold[
                            "candidate_selection_score"
                        ],
                        "n_features_before_missingness_filter": fit_info[
                            "n_features_before_missingness_filter"
                        ],
                        "n_features_after_missingness_filter": fit_info[
                            "n_features_after_missingness_filter"
                        ],
                        "n_features_after_mice_preprocessing": fit_info[
                            "n_features_after_mice_preprocessing"
                        ],
                        "feature_selection_k": fit_info["feature_selection_k"],
                        "n_features_after_feature_selection": fit_info[
                            "n_features_after_feature_selection"
                        ],
                        "n_transformed_features": fit_info["n_transformed_features"],
                        "dropped_missingness_columns": "|".join(
                            fit_info["dropped_missingness_columns"]
                        ),
                        "dropped_unusable_columns": "|".join(
                            fit_info["dropped_unusable_columns"]
                        ),
                        "raw_validation_ece": raw_val_ece,
                        "raw_test_ece": raw_test_ece,
                    }
                    model_row.update(
                        {
                            f"raw_validation_{key}": value
                            for key, value in raw_val_metrics.items()
                        }
                    )
                    model_row.update(
                        {
                            f"raw_test_{key}": value
                            for key, value in raw_test_metrics.items()
                        }
                    )
                    model_row.update(
                        {
                            f"calibrated_validation_{key}": value
                            for key, value in calibrated_val_metrics.items()
                        }
                    )
                    model_row.update(
                        {
                            f"calibrated_test_{key}": value
                            for key, value in calibrated_test_metrics.items()
                        }
                    )
                    model_row.update(
                        {
                            f"validation_{key}": value
                            for key, value in val_metrics.items()
                        }
                    )
                    model_row.update(
                        {f"test_{key}": value for key, value in test_metrics.items()}
                    )
                    for key in CALIBRATION_DIAGNOSTIC_METRICS:
                        model_row[f"validation_{key}"] = calibrated_val_metrics[key]
                        model_row[f"test_{key}"] = calibrated_test_metrics[key]
                    holdout_model_rows.append(model_row)

                print(
                    f"{target_name:>13} | {feature_name:<20} | "
                    f"day={landmark_day:>3} | "
                    f"split={split_info['split_policy']} | "
                    f"development_seed={split_info['development_split_seed']} | "
                    f"fixed_test_seed={split_info['test_split_seed']} | "
                    f"source={selected_feature_source} | "
                    f"train/val/test={len(train_pos)}/{len(val_pos)}/{len(test_pos)} | "
                    f"events={int(y_train.sum())}/{int(y_val.sum())}/{int(y_test.sum())} | "
                    f"CV={min(K_FOLD_CV_SPLITS, min_class_count(y_train))}-fold"
                    f"x{effective_cv_repeats_for_settings(y_train, optimization_settings)} | "
                    f"tuning={optimization_settings['optuna_trials']} trials/"
                    f"{optimization_settings['seed_bagging_repeats']} bag | "
                    f"train CV AUROC={best_cv_mean_auroc:.3f} | "
                    f"train ensemble OOF AUROC={ensemble_train_cv_auroc:.3f} | "
                    f"model={selected_model_family} | "
                    f"strategy={selected_strategy['strategy_name']} | "
                    f"ensemble={ensemble_top_k_actual} | final_fit={final_refit_scope} | "
                    f"report={selected_report['probability_source']} | "
                    f"SINGLE-MODEL TEST AUROC={test_metrics['auroc']:.3f} | "
                    f"SINGLE-MODEL TEST AUPRC={test_metrics['auprc']:.3f}"
                )
