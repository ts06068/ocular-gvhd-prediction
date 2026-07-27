"""Simplified-model selection and retuning."""

from . import temporal as _temporal  # noqa: F401 - preserve pipeline stage order.
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
    MODEL_DIR,
    MODEL_VARIANT_ORIGINAL,
    MODEL_VARIANT_SIMPLIFIED,
    OPTUNA_OBJECTIVE_METRIC,
    ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC,
    ORIENTATION_CORRECTION_TARGETS,
    ROBUST_VALIDATION_SELECTION,
    RUN_FINAL_MODELS,
    RUN_SIMPLIFIED_MODELS,
    SIMPLIFIED_MICE_ITER_CANDIDATES,
    SIMPLIFIED_RETUNE,
    SIMPLIFIED_TOP_N_FEATURES,
    TARGET_TIMING_POLICY,
    THRESHOLD_CANDIDATE_POLICY,
    THRESHOLD_MIN_SENSITIVITY,
    THRESHOLD_OPERATING_POINT_STATUS,
    THRESHOLD_TIE_BREAK_POLICY,
    VALIDATION_MODEL_SELECTION_METRIC,
)
from .data import (
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
    ensure_model_variant_column,
    evaluate_probability_source,
    fit_probability_recalibrator,
    probability_source_provenance,
    select_threshold_from_oof,
    validation_model_selection_score,
)
from .preprocessing import prettify_base_feature_name
from .splitting import (
    analysis_seed,
    effective_cv_repeats_for_settings,
    final_refit_scope_for_analysis,
    model_family_label,
    optimization_settings_for_analysis,
    simplified_shap_ranking_mode_for_analysis,
    split_random_state_base,
    train_val_test_positions_for_analysis,
)
from .tuning import (
    cross_validated_probabilities,
    finalize_selected_validation_strategy,
    fit_ensemble_with_params,
    fit_with_finalization_recipe,
    preset_xgb_trials,
    robust_strategy_selection_score,
    select_strategy_by_validation,
    strategy_cv_selection_score,
    tune_hyperparameters,
)
from .explainability import (
    cv_shap_top_base_features,
    extract_fit_info,
)
from .temporal import (
    feature_set_item_for_analysis,
    parse_candidate_param_list,
    selected_original_strategy_row,
)


def evaluate_simplified_model_variant(original_row, original_strategy_row):
    target_name = original_row["target"]
    feature_name = original_row["feature_set"]
    feature_item = feature_set_item_for_analysis(feature_name)
    landmark_day = feature_item["landmark_day"]
    selected_feature_source = original_row.get("selected_feature_source", feature_name)
    if pd.isna(selected_feature_source):
        selected_feature_source = feature_name
    source_item = feature_set_item_for_analysis(selected_feature_source)
    selected_feature_source_day = source_item["landmark_day"]
    spec = TARGET_DEFINITIONS[target_name]
    split_seed_base = split_random_state_base(target_name, feature_name)
    final_refit_scope = final_refit_scope_for_analysis(target_name, feature_name)
    seed = analysis_seed(target_name, feature_name) + 700001
    optimization_settings = optimization_settings_for_analysis(
        target_name, feature_name
    )
    simplified_validation_candidate_trials = optimization_settings[
        "simplified_validation_candidate_trials"
    ]
    simplified_seed_bagging_repeats = optimization_settings[
        "simplified_seed_bagging_repeats"
    ]
    simplified_shap_ranking_mode = simplified_shap_ranking_mode_for_analysis(
        target_name,
        feature_name,
    )
    keep_mask, y_target = make_target(
        spec["stage_column"],
        spec["positive_rule"],
        spec.get("exclude_control_death_3m", False),
        spec.get("event_date_column"),
        target_landmark_day_for_policy(landmark_day),
    )
    source_indices = np.flatnonzero(keep_mask)
    train_pos, val_pos, test_pos, split_info = train_val_test_positions_for_analysis(
        target_name,
        spec,
        feature_name,
        source_indices,
        y_target,
    )
    y_train = y_target.iloc[train_pos].reset_index(drop=True)
    y_val = y_target.iloc[val_pos].reset_index(drop=True)
    y_test = y_target.iloc[test_pos].reset_index(drop=True)
    if simplified_shap_ranking_mode == "fixed_depth1":
        ranking_param_list = [
            preset_xgb_trials(y_train, SIMPLIFIED_MICE_ITER_CANDIDATES)[1]
        ]
        simplified_source_strategy_name = "prespecified_regularized_depth1_cv_reference"
    else:
        ranking_param_list = parse_candidate_param_list(
            original_strategy_row["candidate_params_json"]
        )
        simplified_source_strategy_name = (
            f"original_selected:{original_strategy_row['candidate_strategy_name']}"
        )
    X_all_full = source_item["X"].loc[keep_mask].reset_index(drop=True)
    X_train_full = X_all_full.iloc[train_pos].reset_index(drop=True)
    top_features, shap_ranking, shap_fold_info = cv_shap_top_base_features(
        X_train_full,
        y_train,
        ranking_param_list,
        seed + 101,
        SIMPLIFIED_TOP_N_FEATURES,
        seed_bagging_repeats=simplified_seed_bagging_repeats,
    )
    X_all = X_all_full.loc[:, top_features].reset_index(drop=True)
    X_train = X_all.iloc[train_pos].reset_index(drop=True)
    X_val = X_all.iloc[val_pos].reset_index(drop=True)
    X_test = X_all.iloc[test_pos].reset_index(drop=True)

    search = None
    if SIMPLIFIED_RETUNE:
        search = tune_hyperparameters(
            X_train,
            y_train,
            seed + 1009,
            n_trials=optimization_settings["simplified_optuna_trials"],
            mice_iter_candidates=SIMPLIFIED_MICE_ITER_CANDIDATES,
            feature_selection_candidates=optimization_settings[
                "simplified_feature_selection_candidates"
            ],
            cv_repeats=optimization_settings["cv_repeats"],
            modeling_method=optimization_settings["modeling_method"],
            analysis_key=optimization_settings["analysis_key"],
        )
        selected_strategy, validation_model, val_raw_prob, strategy_candidates = (
            select_strategy_by_validation(
                search,
                X_train,
                y_train,
                X_val,
                y_val,
                seed + 1031,
                candidate_trials=simplified_validation_candidate_trials,
                seed_bagging_repeats=simplified_seed_bagging_repeats,
                ensemble_top_k_candidates=optimization_settings[
                    "ensemble_top_k_candidates"
                ],
                include_dual_metric_champions=optimization_settings[
                    "include_dual_metric_champions"
                ],
            )
        )
        selected_strategy_row = strategy_candidates.loc[
            strategy_candidates["candidate_strategy_name"]
            == selected_strategy["strategy_name"]
        ].iloc[0]
        strategy_validation_metrics = classification_metrics(y_val, val_raw_prob, 0.5)
        raw_strategy_selection_score = float(
            selected_strategy_row["candidate_validation_selection_score"]
        )
        strategy_selection_score = float(
            selected_strategy_row.get(
                "candidate_robust_selection_score",
                raw_strategy_selection_score,
            )
        )
        strategy_validation_weight = float(
            selected_strategy_row.get("candidate_validation_reliability_weight", 1.0)
        )
        best_params = dict(search.best_params_)
        best_trial = search.study_.best_trial
        best_cv_mean_auroc = best_trial.user_attrs.get(
            "cv_mean_auroc", search.best_score_
        )
        best_cv_std_auroc = best_trial.user_attrs.get("cv_std_auroc", np.nan)
        best_cv_mean_auprc = best_trial.user_attrs.get("cv_mean_auprc", np.nan)
        best_cv_std_auprc = best_trial.user_attrs.get("cv_std_auprc", np.nan)
        best_cv_auprc_lift = best_trial.user_attrs.get("cv_auprc_lift", np.nan)
    else:
        ensemble_param_list = ranking_param_list
        validation_model = fit_ensemble_with_params(
            X_train,
            y_train,
            ensemble_param_list,
            seed + 29,
            seed_bagging_repeats=simplified_seed_bagging_repeats,
        )
        val_raw_prob = validation_model.predict_proba(X_val)[:, 1]
        strategy_validation_metrics = classification_metrics(y_val, val_raw_prob, 0.5)
        raw_strategy_selection_score = validation_model_selection_score(
            strategy_validation_metrics,
            VALIDATION_MODEL_SELECTION_METRIC,
        )
        strategy_selection_score = float(raw_strategy_selection_score)
        strategy_validation_weight = 1.0
        selected_strategy_name = (
            f"{simplified_source_strategy_name}_cv_shap_top{SIMPLIFIED_TOP_N_FEATURES}"
        )
        selected_strategy_type = (
            f"{simplified_shap_ranking_mode}_reference_after_train_cv_shap_top_features"
        )
        strategy_candidates = pd.DataFrame(
            [
                {
                    "candidate_strategy_name": selected_strategy_name,
                    "candidate_strategy_type": selected_strategy_type,
                    "candidate_requested_top_k": int(len(ensemble_param_list)),
                    "candidate_actual_top_k": int(len(ensemble_param_list)),
                    "candidate_optuna_rank": np.nan,
                    "candidate_optuna_trial_number": np.nan,
                    "candidate_optuna_objective_value": np.nan,
                    "candidate_cv_mean_auroc": np.nan,
                    "candidate_cv_std_auroc": np.nan,
                    "candidate_cv_mean_auprc": np.nan,
                    "candidate_cv_auprc_lift": np.nan,
                    "candidate_validation_selection_metric": f"raw_validation_{VALIDATION_MODEL_SELECTION_METRIC}",
                    "candidate_validation_selection_score": raw_strategy_selection_score,
                    "candidate_cv_selection_score": np.nan,
                    "candidate_validation_reliability_weight": 1.0,
                    "candidate_robust_selection_score": strategy_selection_score,
                    "candidate_selection_metric": f"raw_validation_{VALIDATION_MODEL_SELECTION_METRIC}",
                    "candidate_params_json": json.dumps(
                        [
                            {key: jsonable(value) for key, value in params.items()}
                            for params in ensemble_param_list
                        ],
                        ensure_ascii=False,
                    ),
                    **{
                        f"candidate_validation_{key}": value
                        for key, value in strategy_validation_metrics.items()
                    },
                }
            ]
        )
        selected_strategy = {
            "strategy_name": selected_strategy_name,
            "strategy_type": selected_strategy_type,
            "param_list": ensemble_param_list,
        }
        best_params = dict(ensemble_param_list[0])
        best_cv_mean_auroc = np.nan
        best_cv_std_auroc = np.nan
        best_cv_mean_auprc = np.nan
        best_cv_std_auprc = np.nan
        best_cv_auprc_lift = np.nan

    ensemble_param_list = selected_strategy["param_list"]
    ensemble_top_k_actual = len(ensemble_param_list)
    selected_model_family = model_family_label(ensemble_param_list)
    simplified_validation_fit_seed = seed + (1031 if search is not None else 29)
    (
        selected_strategy,
        validation_model,
        val_raw_prob,
        strategy_candidates,
        finalization_candidates,
    ) = finalize_selected_validation_strategy(
        selected_strategy,
        strategy_candidates,
        X_train,
        y_train,
        X_val,
        y_val,
        simplified_validation_fit_seed,
        validation_model,
        val_raw_prob,
        simplified_seed_bagging_repeats,
        candidate_recipes=optimization_settings["finalization_candidates"],
    )
    strategy_validation_metrics = classification_metrics(y_val, val_raw_prob, 0.5)
    raw_strategy_selection_score = validation_model_selection_score(
        strategy_validation_metrics,
        VALIDATION_MODEL_SELECTION_METRIC,
    )
    cv_strategy_selection_score = strategy_cv_selection_score(
        selected_strategy,
        VALIDATION_MODEL_SELECTION_METRIC,
    )
    strategy_selection_score, strategy_validation_weight = (
        robust_strategy_selection_score(
            raw_strategy_selection_score,
            cv_strategy_selection_score,
            y_val,
        )
    )

    train_cv_raw_prob, train_cv_fold_info = cross_validated_probabilities(
        X_train,
        y_train,
        ensemble_param_list,
        seed + 17,
        cv_role="train_threshold_cv",
        seed_bagging_repeats=simplified_seed_bagging_repeats,
    )
    diagnostic_calibrator = fit_probability_recalibrator(y_train, train_cv_raw_prob)
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
        final_cv_raw_prob, final_refit_cv_fold_info = cross_validated_probabilities(
            X_final_fit,
            y_final_fit,
            ensemble_param_list,
            seed + 131071,
            cv_role="train_validation_calibration_cv",
            seed_bagging_repeats=simplified_seed_bagging_repeats,
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

    # Classification and thresholding use raw scores. Recalibrated scores are
    # diagnostics only, so a threshold can never cross incompatible calibration
    # maps between validation and the final refit.
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
    selected_report = choose_probability_source(probability_source_evaluations)
    selected_threshold = selected_report["selected_threshold"]
    threshold = selected_report["threshold"]
    val_metrics = selected_report["validation_metrics"]
    test_metrics = selected_report["test_metrics"]
    threshold_candidates = pd.concat(
        [item["threshold_candidates"] for item in probability_source_evaluations],
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
            inverted_test_diagnostic_calibrator = fit_probability_recalibrator(
                y_final_fit,
                1.0 - final_cv_raw_prob,
            )
        else:
            inverted_test_diagnostic_calibrator = inverted_diagnostic_calibrator
        selected_test_calibration_prob = apply_probability_recalibrator(
            inverted_test_diagnostic_calibrator,
            selected_test_prob,
        )
        selected_validation_diagnostic_calibrator = inverted_diagnostic_calibrator
        selected_test_diagnostic_calibrator = inverted_test_diagnostic_calibrator
    else:
        selected_train_calibration_prob = train_cv_calibration_prob
        selected_val_calibration_prob = val_calibration_prob
        selected_test_calibration_prob = test_calibration_prob
        selected_validation_diagnostic_calibrator = diagnostic_calibrator
        selected_test_diagnostic_calibrator = test_diagnostic_calibrator

    if selected_report["probability_source"].endswith("_inverted"):
        # Stored calibration diagnostics must be calculated from the same
        # orientation-corrected calibration_probability written to predictions.
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
        probability_source_provenance(selected_report["probability_source"]),
    )
    train_cv_candidates.insert(2, "selection_labels_provenance", "train_oof_labels")
    ensemble_train_cv_auroc = (
        roc_auc_score(y_train, selected_train_prob)
        if len(np.unique(y_train)) == 2
        else np.nan
    )
    ensemble_train_cv_auprc = average_precision_score(y_train, selected_train_prob)
    ensemble_train_cv_prevalence = float(y_train.mean()) if len(y_train) else np.nan
    ensemble_train_cv_auprc_lift = (
        float(ensemble_train_cv_auprc / ensemble_train_cv_prevalence)
        if ensemble_train_cv_prevalence
        else np.nan
    )
    if search is None:
        best_cv_mean_auroc = ensemble_train_cv_auroc
        best_cv_mean_auprc = ensemble_train_cv_auprc
        best_cv_auprc_lift = ensemble_train_cv_auprc_lift
    fit_info = extract_fit_info(holdout_model, final_fit_info_input)

    split_class_counts = {
        "train_min_class_count": min_class_count(y_train),
        "validation_min_class_count": min_class_count(y_val),
        "test_min_class_count": min_class_count(y_test),
    }
    common = {
        "target": target_name,
        "feature_set": feature_name,
        "model_variant": MODEL_VARIANT_SIMPLIFIED,
        "modeling_method": optimization_settings["modeling_method"],
        "selected_model_family": selected_model_family,
        "landmark_day": landmark_day,
        "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
        "selected_feature_source": selected_feature_source,
        "selected_feature_variant": MODEL_VARIANT_SIMPLIFIED,
        "selected_feature_source_day": selected_feature_source_day,
        "event_date_column": spec.get("event_date_column"),
        "target_timing_policy": TARGET_TIMING_POLICY,
        "analysis_n": int(len(y_target)),
        "analysis_positive_n": int(y_target.sum()),
        "analysis_control_n": int(len(y_target) - y_target.sum()),
        "split_random_state_base": int(split_seed_base),
        "split_policy": split_info["split_policy"],
        "split_seed": int(split_info["split_seed"]),
        "split_reference_n": int(split_info["split_reference_n"]),
        "split_reference_positive_n": int(split_info["split_reference_positive_n"]),
        "analysis_seed": int(seed),
        "train_n": int(len(train_pos)),
        "validation_n": int(len(val_pos)),
        "test_n": int(len(test_pos)),
        "train_positive_n": int(y_train.sum()),
        "validation_positive_n": int(y_val.sum()),
        "test_positive_n": int(y_test.sum()),
        **split_class_counts,
        "best_optuna_objective_value": float(
            search.best_score_ if search is not None else best_cv_mean_auroc
        ),
        "best_train_cv_auroc": float(best_cv_mean_auroc),
        "best_train_cv_mean_auroc": float(best_cv_mean_auroc),
        "best_train_cv_std_auroc": float(best_cv_std_auroc),
        "best_train_cv_mean_auprc": float(best_cv_mean_auprc),
        "best_train_cv_std_auprc": float(best_cv_std_auprc),
        "best_train_cv_auprc_lift": float(best_cv_auprc_lift),
        "ensemble_train_cv_oof_auroc": float(ensemble_train_cv_auroc),
        "ensemble_train_cv_oof_auprc": float(ensemble_train_cv_auprc),
        "ensemble_train_cv_oof_auprc_lift": float(ensemble_train_cv_auprc_lift),
        "optuna_trials": int(len(search.study_.trials)) if search is not None else 0,
        "selected_strategy_name": selected_strategy["strategy_name"],
        "selected_strategy_type": selected_strategy["strategy_type"],
        "selected_strategy_validation_auroc": float(
            strategy_validation_metrics["auroc"]
        ),
        "selected_strategy_validation_selection_metric": (
            "pareto_near_best_auroc_then_auprc"
            if DUAL_DISCRIMINATION_SELECTION
            else f"cv_validation_shrinkage_{VALIDATION_MODEL_SELECTION_METRIC}"
            if ROBUST_VALIDATION_SELECTION and search is not None
            else VALIDATION_MODEL_SELECTION_METRIC
        ),
        "selected_strategy_validation_selection_score": float(strategy_selection_score),
        "selected_strategy_raw_validation_selection_score": float(
            raw_strategy_selection_score
        ),
        "selected_strategy_validation_reliability_weight": float(
            strategy_validation_weight
        ),
        "ensemble_top_k": int(ensemble_top_k_actual),
        "performance_rescue": bool(optimization_settings["performance_rescue"]),
        "configured_optuna_trials": int(
            optimization_settings["simplified_optuna_trials"]
        ),
        "optuna_cv_repeats": effective_cv_repeats_for_settings(
            y_train, optimization_settings
        ),
        "configured_feature_selection_candidates": "|".join(
            optimization_settings["simplified_feature_selection_candidates"]
        ),
        "configured_finalization_candidates": "|".join(
            optimization_settings["finalization_candidates"]
        ),
        "validation_candidate_trials": int(simplified_validation_candidate_trials),
        "seed_bagging_repeats": int(simplified_seed_bagging_repeats),
        "ensemble_model_members": int(
            ensemble_top_k_actual * simplified_seed_bagging_repeats
        ),
        "selected_finalization_strategy": selected_strategy["finalization_strategy"],
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
        "reported_probability_source": selected_report["probability_source"],
        "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
        "validation_model_selection_score": selected_report[
            "validation_model_selection_score"
        ],
        "selected_threshold_objective": selected_threshold["candidate_objective"],
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
        "train_cv_selected_threshold": float(train_cv_selected["candidate_threshold"]),
        "train_cv_selected_threshold_score": train_cv_selected[
            "candidate_selection_score"
        ],
        "calibration_diagnostic_method": selected_test_diagnostic_calibrator["method"],
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
        "dropped_unusable_columns": "|".join(fit_info["dropped_unusable_columns"]),
        "raw_validation_ece": raw_val_ece,
        "raw_test_ece": raw_test_ece,
        "simplified_feature_selection_method": (
            f"train_cv_oof_shap_stability_top_n_with_{simplified_shap_ranking_mode}_reference"
        ),
        "simplified_top_n_features": int(SIMPLIFIED_TOP_N_FEATURES),
        "simplified_selected_features": "|".join(top_features),
        "simplified_shap_ranking_mode": simplified_shap_ranking_mode,
        "simplified_source_strategy_name": simplified_source_strategy_name,
        "simplified_hyperparameter_method": (
            "independent_train_5fold_optuna"
            if search is not None
            else "reuse_fixed_regularized_depth1_reference"
        ),
    }
    result_row = dict(common)
    result_row.update(
        {f"raw_validation_{key}": value for key, value in raw_val_metrics.items()}
    )
    result_row.update(
        {f"raw_test_{key}": value for key, value in raw_test_metrics.items()}
    )
    result_row.update(
        {
            f"calibrated_validation_{key}": value
            for key, value in calibrated_val_metrics.items()
        }
    )
    result_row.update(
        {
            f"calibrated_test_{key}": value
            for key, value in calibrated_test_metrics.items()
        }
    )
    result_row.update(
        {f"validation_{key}": value for key, value in val_metrics.items()}
    )
    result_row.update({f"test_{key}": value for key, value in test_metrics.items()})
    for key in CALIBRATION_DIAGNOSTIC_METRICS:
        result_row[f"validation_{key}"] = calibrated_val_metrics[key]
        result_row[f"test_{key}"] = calibrated_test_metrics[key]

    threshold_candidates = threshold_candidates.copy()
    threshold_candidates.insert(0, "target", target_name)
    threshold_candidates.insert(1, "feature_set", feature_name)
    threshold_candidates.insert(2, "model_variant", MODEL_VARIANT_SIMPLIFIED)
    threshold_candidates.insert(3, "landmark_day", landmark_day)
    threshold_candidates.insert(4, "selection_split", "validation")

    train_cv_candidates = train_cv_candidates.copy()
    train_cv_candidates.insert(0, "target", target_name)
    train_cv_candidates.insert(1, "feature_set", feature_name)
    train_cv_candidates.insert(2, "model_variant", MODEL_VARIANT_SIMPLIFIED)
    train_cv_candidates.insert(3, "landmark_day", landmark_day)
    train_cv_candidates.insert(4, "selection_split", "train_cv")

    train_cv_fold_info = train_cv_fold_info.copy()
    train_cv_fold_info.insert(0, "target", target_name)
    train_cv_fold_info.insert(1, "feature_set", feature_name)
    train_cv_fold_info.insert(2, "model_variant", MODEL_VARIANT_SIMPLIFIED)
    train_cv_fold_info.insert(3, "landmark_day", landmark_day)

    strategy_candidates = strategy_candidates.copy()
    strategy_candidates.insert(0, "target", target_name)
    strategy_candidates.insert(1, "feature_set", feature_name)
    strategy_candidates.insert(2, "model_variant", MODEL_VARIANT_SIMPLIFIED)
    strategy_candidates.insert(3, "landmark_day", landmark_day)
    strategy_candidates.insert(4, "selected_feature_source", selected_feature_source)
    strategy_candidates.insert(
        5,
        "selected_strategy",
        strategy_candidates["candidate_strategy_name"]
        == selected_strategy["strategy_name"],
    )

    best_param_row = {
        "target": target_name,
        "feature_set": feature_name,
        "model_variant": MODEL_VARIANT_SIMPLIFIED,
        "landmark_day": landmark_day,
        "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
        "selected_feature_source": selected_feature_source,
        "selected_feature_source_day": selected_feature_source_day,
        "best_optuna_objective_value": float(
            search.best_score_ if search is not None else best_cv_mean_auroc
        ),
        "best_train_cv_auroc": float(best_cv_mean_auroc),
        "best_train_cv_mean_auroc": float(best_cv_mean_auroc),
        "best_train_cv_std_auroc": float(best_cv_std_auroc),
        "best_train_cv_mean_auprc": float(best_cv_mean_auprc),
        "best_train_cv_std_auprc": float(best_cv_std_auprc),
        "best_train_cv_auprc_lift": float(best_cv_auprc_lift),
        "ensemble_train_cv_oof_auroc": float(ensemble_train_cv_auroc),
        "ensemble_train_cv_oof_auprc": float(ensemble_train_cv_auprc),
        "ensemble_train_cv_oof_auprc_lift": float(ensemble_train_cv_auprc_lift),
        "optuna_trials": int(len(search.study_.trials)) if search is not None else 0,
        "selected_strategy_name": selected_strategy["strategy_name"],
        "selected_strategy_type": selected_strategy["strategy_type"],
        "selected_strategy_validation_auroc": float(
            strategy_validation_metrics["auroc"]
        ),
        "ensemble_top_k": int(ensemble_top_k_actual),
        "final_refit_scope": final_refit_scope,
        "calibration_diagnostic_method": selected_test_diagnostic_calibrator["method"],
        **best_params,
    }

    prediction_rows = []
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
            prediction_rows.append(
                {
                    "target": target_name,
                    "feature_set": feature_name,
                    "model_variant": MODEL_VARIANT_SIMPLIFIED,
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

    ranking_rows = []
    for _, ranking_row in shap_ranking.iterrows():
        ranking_rows.append(
            {
                "target": target_name,
                "feature_set": feature_name,
                "model_variant": MODEL_VARIANT_SIMPLIFIED,
                "selected_feature_source": selected_feature_source,
                "landmark_day": landmark_day,
                "feature": ranking_row["feature"],
                "pretty_feature": prettify_base_feature_name(ranking_row["feature"]),
                "mean_abs_shap": float(ranking_row["mean_abs_shap"]),
                "mean_normalized_shap": float(ranking_row["mean_normalized_shap"]),
                "mean_shap_percentile_rank": float(
                    ranking_row["mean_shap_percentile_rank"]
                ),
                "top_n_fold_frequency": float(ranking_row["top_n_fold_frequency"]),
                "shap_stability_score": float(ranking_row["shap_stability_score"]),
                "rank": int(ranking_row["rank"]),
                "selected_top_n": bool(ranking_row["feature"] in set(top_features)),
            }
        )

    model_rows = []
    if RUN_FINAL_MODELS:
        model_slug = f"{slugify(target_name)}__{slugify(feature_name)}__simplified"
        model_path = MODEL_DIR / f"{model_slug}.pkl"
        metadata_path = MODEL_DIR / f"{model_slug}_metadata.json"
        bundle = {
            "model": holdout_model,
            "model_variant": MODEL_VARIANT_SIMPLIFIED,
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
            "feature_columns_before_filter": top_features,
            "simplified_feature_selection_method": common[
                "simplified_feature_selection_method"
            ],
            "simplified_top_n_features": int(SIMPLIFIED_TOP_N_FEATURES),
            "simplified_selected_features": top_features,
            "simplified_shap_ranking": ranking_rows,
            "analysis_source_indices": source_indices.tolist(),
            "split_random_state_base": int(split_seed_base),
            "split_policy": split_info["split_policy"],
            "split_seed": int(split_info["split_seed"]),
            "split_reference_n": int(split_info["split_reference_n"]),
            "split_reference_positive_n": int(split_info["split_reference_positive_n"]),
            "analysis_seed": int(seed),
            "train_indices": train_pos.tolist(),
            "validation_indices": val_pos.tolist(),
            "test_indices": test_pos.tolist(),
            "threshold": float(threshold),
            "threshold_objective": selected_threshold["candidate_objective"],
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
            "train_oof_threshold": float(train_cv_selected["candidate_threshold"]),
            "train_oof_threshold_objective": train_cv_selected["candidate_objective"],
            "reported_probability_source": selected_report["probability_source"],
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
            key: jsonable(value)
            for key, value in bundle.items()
            if key
            not in {
                "model",
                "calibration_diagnostic_recalibrator",
                "validation_selection_recalibrator",
            }
        }
        metadata.update(
            {
                "model_path": str(model_path),
                "validation_metrics": {
                    key: jsonable(value) for key, value in val_metrics.items()
                },
                "test_metrics": {
                    key: jsonable(value) for key, value in test_metrics.items()
                },
                "raw_validation_metrics": {
                    key: jsonable(value) for key, value in raw_val_metrics.items()
                },
                "raw_test_metrics": {
                    key: jsonable(value) for key, value in raw_test_metrics.items()
                },
                "calibrated_validation_metrics": {
                    key: jsonable(value)
                    for key, value in calibrated_val_metrics.items()
                },
                "calibrated_test_metrics": {
                    key: jsonable(value)
                    for key, value in calibrated_test_metrics.items()
                },
            }
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=jsonable)
        model_row = dict(common)
        model_row.update(
            {
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
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
            }
        )
        model_row.update(
            {f"raw_validation_{key}": value for key, value in raw_val_metrics.items()}
        )
        model_row.update(
            {f"raw_test_{key}": value for key, value in raw_test_metrics.items()}
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
            {f"validation_{key}": value for key, value in val_metrics.items()}
        )
        model_row.update({f"test_{key}": value for key, value in test_metrics.items()})
        for key in CALIBRATION_DIAGNOSTIC_METRICS:
            model_row[f"validation_{key}"] = calibrated_val_metrics[key]
            model_row[f"test_{key}"] = calibrated_test_metrics[key]
        model_rows.append(model_row)

    feature_source_candidate_row = {
        "target": target_name,
        "feature_set": feature_name,
        "model_variant": MODEL_VARIANT_SIMPLIFIED,
        "landmark_day": landmark_day,
        "candidate_feature_source": selected_feature_source,
        "candidate_feature_variant": MODEL_VARIANT_SIMPLIFIED,
        "candidate_landmark_day": selected_feature_source_day,
        "selected_feature_source": True,
        "candidate_seed": int(seed),
        "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
        "candidate_validation_selection_score": float(strategy_selection_score),
        "candidate_validation_auroc": float(strategy_validation_metrics["auroc"]),
        "candidate_validation_auprc": float(strategy_validation_metrics["auprc"]),
        "candidate_validation_auprc_lift": float(
            strategy_validation_metrics["auprc_lift"]
        ),
        "candidate_selected_strategy_name": selected_strategy["strategy_name"],
        "candidate_selected_strategy_type": selected_strategy["strategy_type"],
        "candidate_ensemble_top_k": int(ensemble_top_k_actual),
        "candidate_optuna_trials": int(len(search.study_.trials))
        if search is not None
        else 0,
        "candidate_best_optuna_objective_value": float(
            search.best_score_ if search is not None else best_cv_mean_auroc
        ),
    }

    optuna_trial_rows = []
    if search is not None:
        for trial in search.study_.trials:
            if trial.value is None:
                continue
            trial_row = {
                "target": target_name,
                "feature_set": feature_name,
                "model_variant": MODEL_VARIANT_SIMPLIFIED,
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
                    "cv_selection_metric",
                    OPTUNA_OBJECTIVE_METRIC,
                ),
            }
            trial_row.update(trial.params)
            optuna_trial_rows.append(trial_row)

    print(
        f"{target_name:>13} | {feature_name:<20} | {MODEL_VARIANT_SIMPLIFIED:<10} | "
        f"top{SIMPLIFIED_TOP_N_FEATURES}={', '.join(top_features[:3])}"
        f"{'...' if len(top_features) > 3 else ''} | "
        f"events={int(y_train.sum())}/{int(y_val.sum())}/{int(y_test.sum())} | "
        f"CV={min(K_FOLD_CV_SPLITS, min_class_count(y_train))}-fold"
        f"x{effective_cv_repeats_for_settings(y_train, optimization_settings)} | "
        f"tuning={optimization_settings['simplified_optuna_trials']} trials/"
        f"{simplified_seed_bagging_repeats} bag | "
        f"train CV AUROC={best_cv_mean_auroc:.3f} | "
        f"SINGLE-MODEL TEST AUROC={test_metrics['auroc']:.3f} | "
        f"SINGLE-MODEL TEST AUPRC={test_metrics['auprc']:.3f}"
    )

    return {
        "results": pd.DataFrame([result_row]),
        "threshold_candidates": threshold_candidates,
        "train_cv_threshold_candidates": train_cv_candidates,
        "train_cv_folds": train_cv_fold_info,
        "best_params": pd.DataFrame([best_param_row]),
        "optuna_trials": pd.DataFrame(optuna_trial_rows),
        "strategy_candidates": strategy_candidates,
        "feature_source_candidates": pd.DataFrame([feature_source_candidate_row]),
        "predictions": pd.DataFrame(prediction_rows),
        "model_summary": pd.DataFrame(model_rows),
        "simplified_feature_ranking": pd.DataFrame(ranking_rows),
        "simplified_shap_cv_folds": shap_fold_info.assign(
            target=target_name,
            feature_set=feature_name,
            model_variant=MODEL_VARIANT_SIMPLIFIED,
            landmark_day=landmark_day,
        ),
    }


def run_simplified_model_pass(
    holdout_results_df,
    holdout_threshold_candidates_df,
    holdout_train_cv_threshold_candidates_df,
    holdout_train_cv_folds_df,
    holdout_best_params_df,
    holdout_optuna_trials_df,
    holdout_strategy_candidates_df,
    holdout_feature_source_candidates_df,
    holdout_predictions_df,
    holdout_model_summary_df,
):
    if not RUN_SIMPLIFIED_MODELS or holdout_results_df.empty:
        return (
            holdout_results_df,
            holdout_threshold_candidates_df,
            holdout_train_cv_threshold_candidates_df,
            holdout_train_cv_folds_df,
            holdout_best_params_df,
            holdout_optuna_trials_df,
            holdout_strategy_candidates_df,
            holdout_feature_source_candidates_df,
            holdout_predictions_df,
            holdout_model_summary_df,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    holdout_results_df = ensure_model_variant_column(holdout_results_df)
    holdout_threshold_candidates_df = ensure_model_variant_column(
        holdout_threshold_candidates_df
    )
    holdout_train_cv_threshold_candidates_df = ensure_model_variant_column(
        holdout_train_cv_threshold_candidates_df
    )
    holdout_train_cv_folds_df = ensure_model_variant_column(holdout_train_cv_folds_df)
    holdout_best_params_df = ensure_model_variant_column(holdout_best_params_df)
    holdout_optuna_trials_df = ensure_model_variant_column(holdout_optuna_trials_df)
    holdout_strategy_candidates_df = ensure_model_variant_column(
        holdout_strategy_candidates_df
    )
    holdout_feature_source_candidates_df = ensure_model_variant_column(
        holdout_feature_source_candidates_df
    )
    holdout_predictions_df = ensure_model_variant_column(holdout_predictions_df)
    holdout_model_summary_df = ensure_model_variant_column(holdout_model_summary_df)

    simplified_outputs = []
    original_rows = holdout_results_df.loc[
        holdout_results_df["model_variant"] == MODEL_VARIANT_ORIGINAL
    ]
    for _, original_row in original_rows.iterrows():
        strategy_row = selected_original_strategy_row(
            holdout_strategy_candidates_df,
            original_row["target"],
            original_row["feature_set"],
        )
        if strategy_row is None:
            continue
        simplified_outputs.append(
            evaluate_simplified_model_variant(original_row, strategy_row)
        )

    def append_frames(base, key):
        frames = [base]
        frames.extend(
            output[key] for output in simplified_outputs if not output[key].empty
        )
        return (
            pd.concat(frames, ignore_index=True, sort=False)
            if len(frames) > 1
            else base
        )

    simplified_feature_ranking = (
        pd.concat(
            [
                output["simplified_feature_ranking"]
                for output in simplified_outputs
                if not output["simplified_feature_ranking"].empty
            ],
            ignore_index=True,
            sort=False,
        )
        if simplified_outputs
        else pd.DataFrame()
    )
    simplified_shap_cv_folds = (
        pd.concat(
            [
                output["simplified_shap_cv_folds"]
                for output in simplified_outputs
                if not output["simplified_shap_cv_folds"].empty
            ],
            ignore_index=True,
            sort=False,
        )
        if simplified_outputs
        else pd.DataFrame()
    )

    return (
        append_frames(holdout_results_df, "results"),
        append_frames(holdout_threshold_candidates_df, "threshold_candidates"),
        append_frames(
            holdout_train_cv_threshold_candidates_df, "train_cv_threshold_candidates"
        ),
        append_frames(holdout_train_cv_folds_df, "train_cv_folds"),
        append_frames(holdout_best_params_df, "best_params"),
        append_frames(holdout_optuna_trials_df, "optuna_trials"),
        append_frames(holdout_strategy_candidates_df, "strategy_candidates"),
        append_frames(
            holdout_feature_source_candidates_df, "feature_source_candidates"
        ),
        append_frames(holdout_predictions_df, "predictions"),
        append_frames(holdout_model_summary_df, "model_summary"),
        simplified_feature_ranking,
        simplified_shap_cv_folds,
    )
