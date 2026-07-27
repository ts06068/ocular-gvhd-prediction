"""Hyperparameter search and model-selection helpers."""

from .utils import (
    OrderedDict,
    Parallel,
    cross_validate,
    delayed,
    json,
    jsonable,
    np,
    optuna,
    parallel_backend,
    pd,
    time,
)
from .config import (
    AGVHD_XGB_REPRODUCTION_PRESETS,
    AGVHD_XGB_REPRODUCTION_PROFILE,
    AUPRC_LIFT_WEIGHT,
    AUROC_NEAR_BEST_TOLERANCE,
    DUAL_DISCRIMINATION_SELECTION,
    ENSEMBLE_TOP_K_CANDIDATES,
    FEATURE_SELECTION_CANDIDATES,
    FINALIZATION_CANDIDATES,
    FINAL_CV_BAGGING,
    FINAL_CV_BAGGING_INCLUDE_FULL_REFIT,
    HIGH_MISSING_CUTOFF,
    K_FOLD_CV_SPLITS,
    MICE_ITER_CANDIDATES,
    OPTUNA_CV_REPEATS,
    OPTUNA_OBJECTIVE_METRIC,
    OPTUNA_PROGRESS_INTERVAL,
    OPTUNA_STARTUP_TRIALS,
    OPTUNA_STD_PENALTY,
    OPTUNA_TIMEOUT,
    OPTUNA_TRIALS,
    PIPELINE_CACHE_ENABLED,
    PIPELINE_CACHE_VERSION,
    PIPELINE_MEMORY,
    RANDOM_STATE,
    ROBUST_VALIDATION_SELECTION,
    SEARCH_N_JOBS,
    SEED_BAGGING_REPEATS,
    USE_PRESET_TRIALS,
    VALIDATION_CANDIDATE_TRIALS,
    VALIDATION_MODEL_SELECTION_METRIC,
    VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT,
    XGB_BOOSTER_CANDIDATES,
    XGB_N_JOBS,
)
from .targets import min_class_count
from .metrics import (
    classification_metrics,
    combined_discrimination_score,
    validation_model_selection_score,
)
from .preprocessing import make_pipeline_for_params
from .splitting import (
    effective_tuning_cv_repeats,
    make_cv,
    make_tuning_cv,
)


def suggest_xgb_params(
    trial,
    y,
    mice_iter_candidates=None,
    feature_selection_candidates=None,
):
    y = pd.Series(y).astype(int)
    mice_iter_candidates = list(mice_iter_candidates or MICE_ITER_CANDIDATES)
    feature_selection_candidates = list(
        feature_selection_candidates or FEATURE_SELECTION_CANDIDATES
    )
    pos = int(y.sum())
    neg = int(len(y) - pos)
    pos_weight = neg / max(pos, 1)
    if len(XGB_BOOSTER_CANDIDATES) == 1:
        booster = XGB_BOOSTER_CANDIDATES[0]
    else:
        booster = trial.suggest_categorical(
            "model__booster",
            XGB_BOOSTER_CANDIDATES,
        )
    params = {
        "preprocess__max_iter": trial.suggest_categorical(
            "preprocess__max_iter",
            mice_iter_candidates,
        ),
        "model__n_estimators": trial.suggest_int(
            "model__n_estimators", 100, 2000, step=50
        ),
        "model__learning_rate": trial.suggest_float(
            "model__learning_rate", 0.001, 0.20, log=True
        ),
        "model__reg_alpha": trial.suggest_float(
            "model__reg_alpha", 1e-8, 20.0, log=True
        ),
        "model__reg_lambda": trial.suggest_float(
            "model__reg_lambda", 0.01, 100.0, log=True
        ),
        "model__scale_pos_weight": trial.suggest_float(
            "model__scale_pos_weight",
            0.25,
            float(min(max(pos_weight * 3.0, 2.0), 200.0)),
            log=True,
        ),
    }
    if len(XGB_BOOSTER_CANDIDATES) > 1 or booster != "gbtree":
        params["model__booster"] = booster
    if booster == "gbtree":
        params.update(
            {
                "model__max_depth": trial.suggest_int("model__max_depth", 1, 6),
                "model__min_child_weight": trial.suggest_float(
                    "model__min_child_weight", 0.1, 60.0, log=True
                ),
                "model__subsample": trial.suggest_float("model__subsample", 0.50, 1.00),
                "model__colsample_bytree": trial.suggest_float(
                    "model__colsample_bytree", 0.40, 1.00
                ),
                "model__colsample_bylevel": trial.suggest_float(
                    "model__colsample_bylevel", 0.50, 1.00
                ),
                "model__colsample_bynode": trial.suggest_float(
                    "model__colsample_bynode", 0.50, 1.00
                ),
                "model__gamma": trial.suggest_float("model__gamma", 0.0, 10.0),
                "model__max_delta_step": trial.suggest_int(
                    "model__max_delta_step", 0, 10
                ),
                "model__max_bin": trial.suggest_categorical(
                    "model__max_bin", [64, 128, 256, 512]
                ),
            }
        )
    else:
        params.update(
            {
                "model__updater": "coord_descent",
                "model__feature_selector": "cyclic",
            }
        )
    if len(feature_selection_candidates) == 1:
        params["select__k"] = feature_selection_candidates[0]
    else:
        params["select__k"] = trial.suggest_categorical(
            "select__k",
            feature_selection_candidates,
        )
    return params


def preset_xgb_trials(
    y,
    mice_iter_candidates=None,
    feature_selection_candidates=None,
):
    y = pd.Series(y).astype(int)
    mice_iter_candidates = list(mice_iter_candidates or MICE_ITER_CANDIDATES)
    feature_selection_candidates = list(
        feature_selection_candidates or FEATURE_SELECTION_CANDIDATES
    )
    pos = int(y.sum())
    neg = int(len(y) - pos)
    pos_weight = neg / max(pos, 1)
    moderate_weight = float(np.sqrt(max(pos_weight, 1.0)))
    full_weight = float(max(pos_weight, 1.0))
    select_k = (
        feature_selection_candidates[0]
        if len(feature_selection_candidates) == 1
        else "all"
    )

    tree_presets = [
        {
            "preprocess__max_iter": 5
            if 5 in mice_iter_candidates
            else mice_iter_candidates[0],
            "model__n_estimators": 350,
            "model__learning_rate": 0.03,
            "model__max_depth": 2,
            "model__min_child_weight": 8.0,
            "model__subsample": 0.85,
            "model__colsample_bytree": 0.75,
            "model__colsample_bylevel": 0.85,
            "model__colsample_bynode": 0.85,
            "model__gamma": 1.0,
            "model__reg_alpha": 0.1,
            "model__reg_lambda": 8.0,
            "model__max_delta_step": 1,
            "model__max_bin": 256,
            "model__scale_pos_weight": moderate_weight,
            "select__k": select_k,
        },
        {
            "preprocess__max_iter": 10
            if 10 in mice_iter_candidates
            else mice_iter_candidates[-1],
            "model__n_estimators": 550,
            "model__learning_rate": 0.015,
            "model__max_depth": 1,
            "model__min_child_weight": 4.0,
            "model__subsample": 0.80,
            "model__colsample_bytree": 0.70,
            "model__colsample_bylevel": 0.80,
            "model__colsample_bynode": 0.80,
            "model__gamma": 0.5,
            "model__reg_alpha": 0.5,
            "model__reg_lambda": 12.0,
            "model__max_delta_step": 2,
            "model__max_bin": 128,
            "model__scale_pos_weight": full_weight,
            "select__k": select_k,
        },
        {
            "preprocess__max_iter": 1
            if 1 in mice_iter_candidates
            else mice_iter_candidates[0],
            "model__n_estimators": 250,
            "model__learning_rate": 0.05,
            "model__max_depth": 2,
            "model__min_child_weight": 2.0,
            "model__subsample": 0.90,
            "model__colsample_bytree": 0.85,
            "model__colsample_bylevel": 0.90,
            "model__colsample_bynode": 0.90,
            "model__gamma": 0.0,
            "model__reg_alpha": 0.01,
            "model__reg_lambda": 4.0,
            "model__max_delta_step": 1,
            "model__max_bin": 256,
            "model__scale_pos_weight": moderate_weight,
            "select__k": select_k,
        },
        {
            "preprocess__max_iter": 1
            if 1 in mice_iter_candidates
            else mice_iter_candidates[0],
            "model__n_estimators": 400,
            "model__learning_rate": 0.025,
            "model__max_depth": 1,
            "model__min_child_weight": 8.0,
            "model__subsample": 1.0,
            "model__colsample_bytree": 0.80,
            "model__colsample_bylevel": 0.90,
            "model__colsample_bynode": 0.90,
            "model__gamma": 0.0,
            "model__reg_alpha": 1e-8,
            "model__reg_lambda": 15.0,
            "model__max_delta_step": 1,
            "model__max_bin": 128,
            "model__scale_pos_weight": 1.0,
            "select__k": select_k,
        },
        {
            "preprocess__max_iter": mice_iter_candidates[-1],
            "model__n_estimators": 600,
            "model__learning_rate": 0.0125,
            "model__max_depth": 2,
            "model__min_child_weight": 15.0,
            "model__subsample": 0.80,
            "model__colsample_bytree": 0.65,
            "model__colsample_bylevel": 0.80,
            "model__colsample_bynode": 0.80,
            "model__gamma": 2.0,
            "model__reg_alpha": 0.25,
            "model__reg_lambda": 25.0,
            "model__max_delta_step": 2,
            "model__max_bin": 128,
            "model__scale_pos_weight": moderate_weight,
            "select__k": select_k,
        },
        {
            "preprocess__max_iter": mice_iter_candidates[-1],
            "model__n_estimators": 300,
            "model__learning_rate": 0.02,
            "model__max_depth": 3,
            "model__min_child_weight": 20.0,
            "model__subsample": 0.75,
            "model__colsample_bytree": 0.60,
            "model__colsample_bylevel": 0.75,
            "model__colsample_bynode": 0.75,
            "model__gamma": 4.0,
            "model__reg_alpha": 1.0,
            "model__reg_lambda": 30.0,
            "model__max_delta_step": 3,
            "model__max_bin": 64,
            "model__scale_pos_weight": full_weight,
            "select__k": select_k,
        },
    ]
    for preset in tree_presets:
        preset["model__booster"] = "gbtree"

    linear_presets = [
        {
            "preprocess__max_iter": 1
            if 1 in mice_iter_candidates
            else mice_iter_candidates[0],
            "model__booster": "gblinear",
            "model__n_estimators": 300,
            "model__learning_rate": 0.03,
            "model__reg_alpha": 0.1,
            "model__reg_lambda": 10.0,
            "model__scale_pos_weight": moderate_weight,
            "model__updater": "coord_descent",
            "model__feature_selector": "cyclic",
            "select__k": select_k,
        },
        {
            "preprocess__max_iter": mice_iter_candidates[-1],
            "model__booster": "gblinear",
            "model__n_estimators": 600,
            "model__learning_rate": 0.01,
            "model__reg_alpha": 0.5,
            "model__reg_lambda": 25.0,
            "model__scale_pos_weight": 1.0,
            "model__updater": "coord_descent",
            "model__feature_selector": "cyclic",
            "select__k": select_k,
        },
        {
            "preprocess__max_iter": mice_iter_candidates[-1],
            "model__booster": "gblinear",
            "model__n_estimators": 1000,
            "model__learning_rate": 0.005,
            "model__reg_alpha": 0.01,
            "model__reg_lambda": 50.0,
            "model__scale_pos_weight": full_weight,
            "model__updater": "coord_descent",
            "model__feature_selector": "cyclic",
            "select__k": select_k,
        },
    ]
    presets = [
        preset
        for preset in tree_presets + linear_presets
        if preset["model__booster"] in XGB_BOOSTER_CANDIDATES
    ]
    if XGB_BOOSTER_CANDIDATES == ["gbtree"]:
        for preset in presets:
            preset.pop("model__booster", None)

    if len(feature_selection_candidates) > 1:
        allowed = set(feature_selection_candidates)
        for preset in presets:
            if preset["select__k"] not in allowed:
                preset["select__k"] = feature_selection_candidates[0]

    return presets


class OptunaSearchResult:
    def __init__(self, study):
        self.study_ = study
        self.best_params_ = dict(study.best_params)
        self.best_score_ = float(study.best_value)


class ProbabilityAveragingEnsemble:
    def __init__(self, models):
        self.models_ = list(models)

    def predict_proba(self, X):
        probabilities = [model.predict_proba(X) for model in self.models_]
        return np.mean(np.stack(probabilities, axis=0), axis=0)


def representative_pipeline(model):
    while hasattr(model, "models_"):
        model = model.models_[0]
    return model


def completed_trials_by_objective(search):
    completed_trials = [
        trial
        for trial in search.study_.trials
        if trial.value is not None and np.isfinite(trial.value)
    ]
    return sorted(completed_trials, key=lambda trial: trial.value, reverse=True)


def top_trial_param_list(search, top_k):
    completed_trials = completed_trials_by_objective(search)
    param_list = [dict(trial.params) for trial in completed_trials[: int(top_k)]]
    if not param_list:
        param_list = [dict(search.best_params_)]
    return param_list


def completed_trials_by_metric(search, user_attr):
    trials = completed_trials_by_objective(search)
    return sorted(
        trials,
        key=lambda trial: (
            float(trial.user_attrs.get(user_attr, -np.inf))
            if np.isfinite(trial.user_attrs.get(user_attr, np.nan))
            else -np.inf,
            float(trial.value),
        ),
        reverse=True,
    )


def validation_trial_pool(search, candidate_trials=VALIDATION_CANDIDATE_TRIALS):
    objective_trials = completed_trials_by_objective(search)
    ranked_groups = [
        ("optuna_objective", objective_trials),
        ("cv_auroc", completed_trials_by_metric(search, "cv_mean_auroc")),
        ("cv_auprc", completed_trials_by_metric(search, "cv_mean_auprc")),
    ]
    pool = OrderedDict()
    for source, trials in ranked_groups:
        for trial in trials[: int(candidate_trials)]:
            record = pool.setdefault(
                int(trial.number),
                {"trial": trial, "candidate_origins": []},
            )
            record["candidate_origins"].append(source)
    return list(pool.values())


def validation_strategy_candidates(
    search,
    candidate_trials=VALIDATION_CANDIDATE_TRIALS,
    ensemble_top_k_candidates=None,
    include_dual_metric_champions=True,
):
    if ensemble_top_k_candidates is None:
        ensemble_top_k_candidates = ENSEMBLE_TOP_K_CANDIDATES
    completed_trials = completed_trials_by_objective(search)
    strategies = []
    objective_rank = {
        int(trial.number): rank for rank, trial in enumerate(completed_trials, start=1)
    }
    strategy_fingerprints = set()

    for record in validation_trial_pool(search, candidate_trials):
        trial = record["trial"]
        fingerprint = (int(trial.number),)
        strategy_fingerprints.add(fingerprint)
        strategies.append(
            {
                "strategy_name": f"single_trial_{int(trial.number)}",
                "strategy_type": "single_trial",
                "candidate_origin": "|".join(record["candidate_origins"]),
                "trial_numbers": [int(trial.number)],
                "optuna_rank": objective_rank[int(trial.number)],
                "optuna_trial_number": int(trial.number),
                "optuna_objective_value": float(trial.value),
                "cv_mean_auroc": trial.user_attrs.get("cv_mean_auroc", np.nan),
                "cv_std_auroc": trial.user_attrs.get("cv_std_auroc", np.nan),
                "cv_mean_auprc": trial.user_attrs.get("cv_mean_auprc", np.nan),
                "cv_auprc_lift": trial.user_attrs.get("cv_auprc_lift", np.nan),
                "requested_top_k": 1,
                "param_list": [dict(trial.params)],
            }
        )

    def append_ensemble(strategy_name, strategy_type, trial_subset, candidate_origin):
        fingerprint = tuple(int(trial.number) for trial in trial_subset)
        if len(fingerprint) < 2 or fingerprint in strategy_fingerprints:
            return
        strategy_fingerprints.add(fingerprint)
        strategies.append(
            {
                "strategy_name": strategy_name,
                "strategy_type": strategy_type,
                "candidate_origin": candidate_origin,
                "trial_numbers": list(fingerprint),
                "optuna_rank": np.nan,
                "optuna_trial_number": np.nan,
                "optuna_objective_value": float(
                    np.nanmean([trial.value for trial in trial_subset])
                ),
                "cv_mean_auroc": float(
                    np.nanmean(
                        [
                            trial.user_attrs.get("cv_mean_auroc", np.nan)
                            for trial in trial_subset
                        ]
                    )
                ),
                "cv_std_auroc": float(
                    np.nanmean(
                        [
                            trial.user_attrs.get("cv_std_auroc", np.nan)
                            for trial in trial_subset
                        ]
                    )
                ),
                "cv_mean_auprc": float(
                    np.nanmean(
                        [
                            trial.user_attrs.get("cv_mean_auprc", np.nan)
                            for trial in trial_subset
                        ]
                    )
                ),
                "cv_auprc_lift": float(
                    np.nanmean(
                        [
                            trial.user_attrs.get("cv_auprc_lift", np.nan)
                            for trial in trial_subset
                        ]
                    )
                ),
                "requested_top_k": int(len(trial_subset)),
                "param_list": [dict(trial.params) for trial in trial_subset],
            }
        )

    for top_k in ensemble_top_k_candidates:
        # Skip top-1: it duplicates rank-1 and tests the same parameters twice
        # against validation.
        if int(top_k) == 1:
            continue
        trial_subset = completed_trials[: int(top_k)]
        if len(trial_subset) < int(top_k):
            continue
        append_ensemble(
            f"top_{int(top_k)}_objective_ensemble",
            "top_k_objective_ensemble",
            trial_subset,
            "optuna_objective",
        )

    if include_dual_metric_champions:
        auroc_trials = completed_trials_by_metric(search, "cv_mean_auroc")
        auprc_trials = completed_trials_by_metric(search, "cv_mean_auprc")
        for champion_n in (1, 2):
            champion_trials = []
            seen_trial_numbers = set()
            for trial in auroc_trials[:champion_n] + auprc_trials[:champion_n]:
                if int(trial.number) in seen_trial_numbers:
                    continue
                seen_trial_numbers.add(int(trial.number))
                champion_trials.append(trial)
            append_ensemble(
                f"auroc_auprc_top_{champion_n}_champions_ensemble",
                "dual_metric_champions_ensemble",
                champion_trials,
                "cv_auroc|cv_auprc",
            )

    if not strategies:
        strategies.append(
            {
                "strategy_name": "fallback_best_trial",
                "strategy_type": "single_trial",
                "candidate_origin": "fallback",
                "trial_numbers": [],
                "optuna_rank": 1,
                "optuna_trial_number": np.nan,
                "optuna_objective_value": float(search.best_score_),
                "cv_mean_auroc": np.nan,
                "cv_std_auroc": np.nan,
                "cv_mean_auprc": np.nan,
                "cv_auprc_lift": np.nan,
                "requested_top_k": 1,
                "param_list": [dict(search.best_params_)],
            }
        )

    return strategies


def strategy_cv_selection_score(strategy, metric_name):
    auroc = strategy.get("cv_mean_auroc", np.nan)
    auprc = strategy.get("cv_mean_auprc", np.nan)
    lift = strategy.get("cv_auprc_lift", np.nan)
    if metric_name == "auroc":
        return float(auroc) if np.isfinite(auroc) else np.nan
    if metric_name == "auprc":
        return float(auprc) if np.isfinite(auprc) else np.nan
    if metric_name == "auprc_lift":
        return float(lift) if np.isfinite(lift) else np.nan
    if metric_name == "auroc_auprc":
        return (
            float(auroc + auprc)
            if np.isfinite(auroc) and np.isfinite(auprc)
            else np.nan
        )
    if metric_name == "auroc_auprc_lift":
        if not np.isfinite(auroc):
            return np.nan
        lift_bonus = np.log1p(max(float(lift), 0.0)) if np.isfinite(lift) else 0.0
        return float(auroc + AUPRC_LIFT_WEIGHT * lift_bonus)
    return np.nan


def robust_strategy_selection_score(validation_score, cv_score, y_val):
    if (
        not ROBUST_VALIDATION_SELECTION
        or not np.isfinite(cv_score)
        or not np.isfinite(validation_score)
    ):
        return float(validation_score), 1.0
    min_count = float(min_class_count(y_val))
    denominator = min_count + VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT
    validation_weight = min_count / denominator if denominator > 0 else 1.0
    score = validation_weight * float(validation_score) + (
        1.0 - validation_weight
    ) * float(cv_score)
    return float(score), float(validation_weight)


def pareto_efficient_mask(auroc_values, auprc_values):
    auroc_values = np.asarray(auroc_values, dtype=float)
    auprc_values = np.asarray(auprc_values, dtype=float)
    efficient = np.ones(len(auroc_values), dtype=bool)
    for idx in range(len(auroc_values)):
        if not np.isfinite(auroc_values[idx]) or not np.isfinite(auprc_values[idx]):
            efficient[idx] = False
            continue
        dominates = (
            (auroc_values >= auroc_values[idx])
            & (auprc_values >= auprc_values[idx])
            & ((auroc_values > auroc_values[idx]) | (auprc_values > auprc_values[idx]))
        )
        if np.any(dominates):
            efficient[idx] = False
    return efficient


def select_strategy_by_validation(
    search,
    X_train,
    y_train,
    X_val,
    y_val,
    seed,
    candidate_trials=VALIDATION_CANDIDATE_TRIALS,
    seed_bagging_repeats=SEED_BAGGING_REPEATS,
    ensemble_top_k_candidates=None,
    include_dual_metric_champions=True,
):
    candidate_rows = []
    candidate_artifacts = {}
    component_cache = {}

    for strategy in validation_strategy_candidates(
        search,
        candidate_trials,
        ensemble_top_k_candidates=ensemble_top_k_candidates,
        include_dual_metric_champions=include_dual_metric_champions,
    ):
        model = fit_ensemble_with_params(
            X_train,
            y_train,
            strategy["param_list"],
            seed,
            seed_bagging_repeats=seed_bagging_repeats,
            component_cache=component_cache,
            data_cache_key="validation_strategy_train",
        )
        val_prob = model.predict_proba(X_val)[:, 1]
        val_metrics = classification_metrics(y_val, val_prob, 0.5)
        validation_selection_score = validation_model_selection_score(
            val_metrics,
            VALIDATION_MODEL_SELECTION_METRIC,
        )
        cv_selection_score = strategy_cv_selection_score(
            strategy,
            VALIDATION_MODEL_SELECTION_METRIC,
        )
        selection_score, validation_weight = robust_strategy_selection_score(
            validation_selection_score,
            cv_selection_score,
            y_val,
        )
        robust_auroc, _ = robust_strategy_selection_score(
            val_metrics["auroc"],
            strategy.get("cv_mean_auroc", np.nan),
            y_val,
        )
        robust_auprc, _ = robust_strategy_selection_score(
            val_metrics["auprc"],
            strategy.get("cv_mean_auprc", np.nan),
            y_val,
        )
        candidate_row = {
            "candidate_strategy_name": strategy["strategy_name"],
            "candidate_strategy_type": strategy["strategy_type"],
            "candidate_origin": strategy.get("candidate_origin", ""),
            "candidate_trial_numbers_json": json.dumps(
                strategy.get("trial_numbers", [])
            ),
            "candidate_requested_top_k": int(strategy["requested_top_k"]),
            "candidate_actual_top_k": int(len(strategy["param_list"])),
            "candidate_optuna_rank": strategy["optuna_rank"],
            "candidate_optuna_trial_number": strategy["optuna_trial_number"],
            "candidate_optuna_objective_value": strategy["optuna_objective_value"],
            "candidate_cv_mean_auroc": strategy["cv_mean_auroc"],
            "candidate_cv_std_auroc": strategy["cv_std_auroc"],
            "candidate_cv_mean_auprc": strategy["cv_mean_auprc"],
            "candidate_cv_auprc_lift": strategy["cv_auprc_lift"],
            "candidate_validation_selection_metric": f"raw_validation_{VALIDATION_MODEL_SELECTION_METRIC}",
            "candidate_validation_selection_score": validation_selection_score,
            "candidate_cv_selection_score": cv_selection_score,
            "candidate_validation_reliability_weight": validation_weight,
            "candidate_robust_selection_score": selection_score,
            "candidate_robust_auroc": robust_auroc,
            "candidate_robust_auprc": robust_auprc,
            "candidate_selection_metric": (
                "pareto_near_best_auroc_then_auprc"
                if DUAL_DISCRIMINATION_SELECTION
                else f"cv_validation_shrinkage_{VALIDATION_MODEL_SELECTION_METRIC}"
                if ROBUST_VALIDATION_SELECTION
                else f"raw_validation_{VALIDATION_MODEL_SELECTION_METRIC}"
            ),
            "candidate_params_json": json.dumps(
                [
                    {key: jsonable(value) for key, value in params.items()}
                    for params in strategy["param_list"]
                ],
                ensure_ascii=False,
            ),
        }
        candidate_row.update(
            {f"candidate_validation_{key}": value for key, value in val_metrics.items()}
        )
        candidate_rows.append(candidate_row)
        candidate_artifacts[strategy["strategy_name"]] = (strategy, model, val_prob)

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        raise RuntimeError("No valid validation strategy candidates were generated.")

    if DUAL_DISCRIMINATION_SELECTION:
        robust_auroc = pd.to_numeric(
            candidates["candidate_robust_auroc"], errors="coerce"
        )
        robust_auprc = pd.to_numeric(
            candidates["candidate_robust_auprc"], errors="coerce"
        )
        candidates["candidate_pareto_efficient"] = pareto_efficient_mask(
            robust_auroc,
            robust_auprc,
        )
        maximum_auroc = float(robust_auroc.max())
        candidates["candidate_near_best_auroc"] = (
            robust_auroc >= maximum_auroc - AUROC_NEAR_BEST_TOLERANCE
        )
        eligible = (
            candidates["candidate_near_best_auroc"]
            & candidates["candidate_pareto_efficient"]
        )
        if not eligible.any():
            eligible = candidates["candidate_near_best_auroc"]
        ranking = candidates.assign(_eligible=eligible).sort_values(
            [
                "_eligible",
                "candidate_robust_auprc",
                "candidate_robust_auroc",
                "candidate_robust_selection_score",
                "candidate_validation_ece",
                "candidate_strategy_name",
            ],
            ascending=[False, False, False, False, True, True],
            kind="mergesort",
        )
    else:
        candidates["candidate_pareto_efficient"] = np.nan
        candidates["candidate_near_best_auroc"] = np.nan
        ranking = candidates.sort_values(
            ["candidate_robust_selection_score", "candidate_strategy_name"],
            ascending=[False, True],
            kind="mergesort",
        )

    candidates["candidate_dual_metric_rank"] = np.nan
    candidates.loc[ranking.index, "candidate_dual_metric_rank"] = np.arange(
        1, len(ranking) + 1
    )
    selected_name = str(ranking.iloc[0]["candidate_strategy_name"])
    best_strategy, best_model, best_val_prob = candidate_artifacts[selected_name]

    return best_strategy, best_model, best_val_prob, candidates


def model_params_fingerprint(params):
    normalized = {
        str(key): jsonable(value)
        for key, value in sorted(params.items(), key=lambda item: str(item[0]))
    }
    return json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def fit_ensemble_with_params(
    X,
    y,
    param_list,
    seed,
    seed_bagging_repeats=SEED_BAGGING_REPEATS,
    component_cache=None,
    data_cache_key=None,
):
    models = []
    for rank, params in enumerate(param_list):
        for repeat_idx in range(int(seed_bagging_repeats)):
            component_seed = seed + 104729 * rank + 8191 * repeat_idx
            cache_key = (
                data_cache_key,
                int(component_seed),
                model_params_fingerprint(params),
            )
            model = (
                component_cache.get(cache_key) if component_cache is not None else None
            )
            if model is None:
                model = make_pipeline_for_params(component_seed, params)
                model.set_params(**params)
                model.fit(X, y)
                if component_cache is not None:
                    component_cache[cache_key] = model
            models.append(model)
    if len(models) == 1:
        return models[0]
    return ProbabilityAveragingEnsemble(models)


def fit_final_cv_bagged_model(
    X,
    y,
    param_list,
    seed,
    seed_bagging_repeats=SEED_BAGGING_REPEATS,
    use_cv_bagging=None,
    include_full_refit=None,
    component_cache=None,
    data_cache_key=None,
):
    if use_cv_bagging is None:
        use_cv_bagging = FINAL_CV_BAGGING
    if include_full_refit is None:
        include_full_refit = FINAL_CV_BAGGING_INCLUDE_FULL_REFIT
    if not use_cv_bagging:
        return fit_ensemble_with_params(
            X,
            y,
            param_list,
            seed,
            seed_bagging_repeats=seed_bagging_repeats,
            component_cache=component_cache,
            data_cache_key=data_cache_key,
        )

    models = []
    if include_full_refit:
        models.append(
            fit_ensemble_with_params(
                X,
                y,
                param_list,
                seed,
                seed_bagging_repeats=seed_bagging_repeats,
                component_cache=component_cache,
                data_cache_key=data_cache_key,
            )
        )

    cv = make_cv(y, K_FOLD_CV_SPLITS, seed + 3571)
    splits = list(enumerate(cv.split(X, y)))

    def fit_fold(fold_idx, fit_idx):
        return fit_ensemble_with_params(
            X.iloc[fit_idx],
            y.iloc[fit_idx],
            param_list,
            seed + 65537 * (fold_idx + 1),
            seed_bagging_repeats=seed_bagging_repeats,
            component_cache=component_cache,
            data_cache_key=(
                data_cache_key,
                "cv_fold",
                tuple(int(value) for value in fit_idx),
            ),
        )

    with parallel_backend("threading", n_jobs=SEARCH_N_JOBS):
        fold_models = Parallel(n_jobs=SEARCH_N_JOBS, prefer="threads")(
            delayed(fit_fold)(fold_idx, fit_idx) for fold_idx, (fit_idx, _) in splits
        )
    models.extend(fold_models)

    if len(models) == 1:
        return models[0]
    return ProbabilityAveragingEnsemble(models)


def fit_repeated_cv_bagged_model(
    X,
    y,
    param_list,
    seed,
    repeats=OPTUNA_CV_REPEATS,
    component_cache=None,
    data_cache_key=None,
):
    models = [
        fit_ensemble_with_params(
            X,
            y,
            param_list,
            seed,
            seed_bagging_repeats=1,
            component_cache=component_cache,
            data_cache_key=data_cache_key,
        )
    ]
    cv = make_tuning_cv(y, K_FOLD_CV_SPLITS, repeats, seed + 3571)
    splits = list(enumerate(cv.split(X, y)))

    def fit_fold(fold_idx, fit_idx):
        return fit_ensemble_with_params(
            X.iloc[fit_idx],
            y.iloc[fit_idx],
            param_list,
            seed + 65537 * (fold_idx + 1),
            seed_bagging_repeats=1,
            component_cache=component_cache,
            data_cache_key=(
                data_cache_key,
                "repeated_cv_fold",
                tuple(int(value) for value in fit_idx),
            ),
        )

    with parallel_backend("threading", n_jobs=SEARCH_N_JOBS):
        fold_models = Parallel(n_jobs=SEARCH_N_JOBS, prefer="threads")(
            delayed(fit_fold)(fold_idx, fit_idx) for fold_idx, (fit_idx, _) in splits
        )
    models.extend(fold_models)
    return ProbabilityAveragingEnsemble(models)


def finalization_model_member_count(recipe, param_list):
    param_count = max(1, len(param_list))
    if recipe == "full_fit_seed1":
        return param_count
    if recipe == "full_fit_seed3":
        return 3 * param_count
    if recipe == "full_fit_seed5":
        return 5 * param_count
    if recipe == "cv_bagging_seed1":
        return (K_FOLD_CV_SPLITS + 1) * param_count
    if recipe == "repeated_cv_bagging_seed1":
        return (K_FOLD_CV_SPLITS * OPTUNA_CV_REPEATS + 1) * param_count
    raise KeyError(recipe)


def fit_with_finalization_recipe(
    X,
    y,
    param_list,
    seed,
    recipe,
    component_cache=None,
    data_cache_key=None,
):
    if recipe == "full_fit_seed1":
        return fit_ensemble_with_params(
            X,
            y,
            param_list,
            seed,
            seed_bagging_repeats=1,
            component_cache=component_cache,
            data_cache_key=data_cache_key,
        )
    if recipe == "full_fit_seed3":
        return fit_ensemble_with_params(
            X,
            y,
            param_list,
            seed,
            seed_bagging_repeats=3,
            component_cache=component_cache,
            data_cache_key=data_cache_key,
        )
    if recipe == "full_fit_seed5":
        return fit_ensemble_with_params(
            X,
            y,
            param_list,
            seed,
            seed_bagging_repeats=5,
            component_cache=component_cache,
            data_cache_key=data_cache_key,
        )
    if recipe == "cv_bagging_seed1":
        return fit_final_cv_bagged_model(
            X,
            y,
            param_list,
            seed,
            seed_bagging_repeats=1,
            use_cv_bagging=True,
            include_full_refit=True,
            component_cache=component_cache,
            data_cache_key=data_cache_key,
        )
    if recipe == "repeated_cv_bagging_seed1":
        return fit_repeated_cv_bagged_model(
            X,
            y,
            param_list,
            seed,
            repeats=OPTUNA_CV_REPEATS,
            component_cache=component_cache,
            data_cache_key=data_cache_key,
        )
    raise KeyError(recipe)


def select_finalization_recipe_by_validation(
    X_train,
    y_train,
    X_val,
    y_val,
    param_list,
    seed,
    existing_model=None,
    existing_probability=None,
    existing_seed_bagging_repeats=None,
    candidate_recipes=None,
):
    candidate_recipes = list(candidate_recipes or FINALIZATION_CANDIDATES)
    rows = []
    artifacts = {}
    component_cache = {}
    for recipe in candidate_recipes:
        recipe_seed_repeats = {
            "full_fit_seed3": 3,
            "full_fit_seed5": 5,
        }.get(recipe, 1)
        can_reuse = (
            recipe.startswith("full_fit_")
            and existing_model is not None
            and existing_probability is not None
            and int(existing_seed_bagging_repeats or 0) == recipe_seed_repeats
        )
        if can_reuse:
            model = existing_model
            val_prob = np.asarray(existing_probability, dtype=float)
        else:
            model = fit_with_finalization_recipe(
                X_train,
                y_train,
                param_list,
                seed,
                recipe,
                component_cache=component_cache,
                data_cache_key="finalization_train",
            )
            val_prob = model.predict_proba(X_val)[:, 1]
        metrics = classification_metrics(y_val, val_prob, 0.5)
        rows.append(
            {
                "finalization_strategy": recipe,
                "validation_auroc": metrics["auroc"],
                "validation_auprc": metrics["auprc"],
                "validation_auprc_lift": metrics["auprc_lift"],
                "validation_selection_score": validation_model_selection_score(
                    metrics,
                    VALIDATION_MODEL_SELECTION_METRIC,
                ),
                "model_members": finalization_model_member_count(recipe, param_list),
            }
        )
        artifacts[recipe] = (model, val_prob)

    candidates = pd.DataFrame(rows)
    if min_class_count(y_val) < 5:
        preferred = "full_fit_seed3"
        if preferred not in artifacts:
            preferred = str(candidates.iloc[0]["finalization_strategy"])
        selected_name = preferred
        selection_policy = "prespecified_stable_recipe_sparse_validation"
    else:
        auroc_values = pd.to_numeric(candidates["validation_auroc"], errors="coerce")
        auprc_values = pd.to_numeric(candidates["validation_auprc"], errors="coerce")
        candidates["pareto_efficient"] = pareto_efficient_mask(
            auroc_values,
            auprc_values,
        )
        maximum_auroc = float(auroc_values.max())
        candidates["near_best_auroc"] = (
            auroc_values >= maximum_auroc - AUROC_NEAR_BEST_TOLERANCE
        )
        eligible = candidates["pareto_efficient"] & candidates["near_best_auroc"]
        if not eligible.any():
            eligible = candidates["near_best_auroc"]
        ranking = candidates.assign(_eligible=eligible).sort_values(
            [
                "_eligible",
                "validation_auprc",
                "validation_auroc",
                "validation_selection_score",
                "model_members",
                "finalization_strategy",
            ],
            ascending=[False, False, False, False, True, True],
            kind="mergesort",
        )
        selected_name = str(ranking.iloc[0]["finalization_strategy"])
        selection_policy = "validation_pareto_near_best_auroc_then_auprc"

    candidates["selected_finalization"] = (
        candidates["finalization_strategy"] == selected_name
    )
    candidates["selection_policy"] = selection_policy
    selected_model, selected_probability = artifacts[selected_name]
    return selected_name, selected_model, selected_probability, candidates


def finalize_selected_validation_strategy(
    selected_strategy,
    strategy_candidates,
    X_train,
    y_train,
    X_val,
    y_val,
    seed,
    existing_model,
    existing_probability,
    existing_seed_bagging_repeats,
    candidate_recipes=None,
):
    finalization_name, model, val_prob, finalization_candidates = (
        select_finalization_recipe_by_validation(
            X_train,
            y_train,
            X_val,
            y_val,
            selected_strategy["param_list"],
            seed,
            existing_model=existing_model,
            existing_probability=existing_probability,
            existing_seed_bagging_repeats=existing_seed_bagging_repeats,
            candidate_recipes=candidate_recipes,
        )
    )
    selected_strategy = dict(selected_strategy)
    selected_strategy["finalization_strategy"] = finalization_name
    selected_strategy["finalization_model_members"] = finalization_model_member_count(
        finalization_name,
        selected_strategy["param_list"],
    )
    selected_finalization_row = finalization_candidates.loc[
        finalization_candidates["finalization_strategy"] == finalization_name
    ].iloc[0]
    selected_strategy["finalization_validation_auroc"] = float(
        selected_finalization_row["validation_auroc"]
    )
    selected_strategy["finalization_validation_auprc"] = float(
        selected_finalization_row["validation_auprc"]
    )
    selected_strategy["finalization_selection_policy"] = str(
        selected_finalization_row["selection_policy"]
    )
    selected_strategy["finalization_candidates_json"] = json.dumps(
        [
            {key: jsonable(value) for key, value in row.items()}
            for row in finalization_candidates.to_dict("records")
        ],
        ensure_ascii=False,
    )

    metrics = classification_metrics(y_val, val_prob, 0.5)
    raw_score = validation_model_selection_score(
        metrics,
        VALIDATION_MODEL_SELECTION_METRIC,
    )
    cv_score = strategy_cv_selection_score(
        selected_strategy,
        VALIDATION_MODEL_SELECTION_METRIC,
    )
    robust_score, validation_weight = robust_strategy_selection_score(
        raw_score,
        cv_score,
        y_val,
    )
    robust_auroc, _ = robust_strategy_selection_score(
        metrics["auroc"],
        selected_strategy.get("cv_mean_auroc", np.nan),
        y_val,
    )
    robust_auprc, _ = robust_strategy_selection_score(
        metrics["auprc"],
        selected_strategy.get("cv_mean_auprc", np.nan),
        y_val,
    )

    strategy_candidates = strategy_candidates.copy()
    selected_mask = (
        strategy_candidates["candidate_strategy_name"]
        == selected_strategy["strategy_name"]
    )
    strategy_candidates.loc[selected_mask, "candidate_finalization_strategy"] = (
        finalization_name
    )
    strategy_candidates.loc[
        selected_mask, "candidate_finalization_validation_auroc"
    ] = metrics["auroc"]
    strategy_candidates.loc[
        selected_mask, "candidate_finalization_validation_auprc"
    ] = metrics["auprc"]
    strategy_candidates.loc[
        selected_mask,
        "candidate_finalization_validation_selection_score",
    ] = raw_score
    strategy_candidates.loc[
        selected_mask,
        "candidate_finalization_robust_selection_score",
    ] = robust_score
    strategy_candidates.loc[
        selected_mask,
        "candidate_finalization_validation_reliability_weight",
    ] = validation_weight
    strategy_candidates.loc[selected_mask, "candidate_finalization_robust_auroc"] = (
        robust_auroc
    )
    strategy_candidates.loc[selected_mask, "candidate_finalization_robust_auprc"] = (
        robust_auprc
    )
    strategy_candidates.loc[
        selected_mask,
        "candidate_finalization_model_members",
    ] = selected_strategy["finalization_model_members"]
    strategy_candidates.loc[
        selected_mask,
        "candidate_finalization_candidates_json",
    ] = selected_strategy["finalization_candidates_json"]
    return (
        selected_strategy,
        model,
        val_prob,
        strategy_candidates,
        finalization_candidates,
    )


def compute_tuning_cv_scores(
    X,
    y,
    params,
    model_seed,
    cv_splits,
    search_n_jobs,
    cache_signature,
):
    # Keep this argument: joblib hashes it before the function runs.
    del cache_signature
    model = make_pipeline_for_params(model_seed, params)
    model.set_params(**params)
    with parallel_backend("threading", n_jobs=search_n_jobs):
        scores = cross_validate(
            model,
            X,
            y,
            scoring={
                "auroc": "roc_auc",
                "auprc": "average_precision",
            },
            cv=cv_splits,
            n_jobs=search_n_jobs,
            error_score=np.nan,
        )
    return scores["test_auroc"], scores["test_auprc"]


CACHED_TUNING_CV_SCORES = (
    PIPELINE_MEMORY.cache(compute_tuning_cv_scores)
    if PIPELINE_CACHE_ENABLED and PIPELINE_MEMORY is not None
    else compute_tuning_cv_scores
)


# Give Optuna training-fold scores only; reserve validation for strategy selection.
def tune_hyperparameters(
    X,
    y,
    seed,
    n_trials=OPTUNA_TRIALS,
    mice_iter_candidates=None,
    feature_selection_candidates=None,
    cv_repeats=None,
    modeling_method="xgboost",
    analysis_key=None,
):
    mice_iter_candidates = list(mice_iter_candidates or MICE_ITER_CANDIDATES)
    feature_selection_candidates = list(
        feature_selection_candidates or FEATURE_SELECTION_CANDIDATES
    )
    effective_cv_repeats = effective_tuning_cv_repeats(y, cv_repeats)
    tuning_objective_metric = OPTUNA_OBJECTIVE_METRIC
    cv = make_tuning_cv(y, K_FOLD_CV_SPLITS, effective_cv_repeats, seed)
    cv_splits = tuple(
        (
            np.asarray(train_idx, dtype=int),
            np.asarray(heldout_idx, dtype=int),
        )
        for train_idx, heldout_idx in cv.split(X, y)
    )
    reproduction_presets = (
        AGVHD_XGB_REPRODUCTION_PRESETS.get(str(analysis_key), [])
        if AGVHD_XGB_REPRODUCTION_PROFILE
        else []
    )
    if reproduction_presets:
        n_trials = len(reproduction_presets)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=int(seed),
            n_startup_trials=min(int(OPTUNA_STARTUP_TRIALS), int(n_trials)),
            multivariate=True,
            group=True,
        ),
        study_name=f"{modeling_method}_auc_{int(seed)}",
    )
    if reproduction_presets:
        for preset in reproduction_presets:
            study.enqueue_trial(dict(preset))
    elif USE_PRESET_TRIALS:
        presets = preset_xgb_trials(
            y,
            mice_iter_candidates,
            feature_selection_candidates,
        )
        for preset in presets[: int(n_trials)]:
            study.enqueue_trial(preset)

    def objective(trial):
        params = suggest_xgb_params(
            trial,
            y,
            mice_iter_candidates,
            feature_selection_candidates,
        )
        try:
            auroc_scores, auprc_scores = CACHED_TUNING_CV_SCORES(
                X,
                y,
                params,
                int(seed + trial.number),
                cv_splits,
                SEARCH_N_JOBS,
                {
                    "version": PIPELINE_CACHE_VERSION,
                    "high_missing_cutoff": HIGH_MISSING_CUTOFF,
                    "preprocess_random_state": RANDOM_STATE,
                    "xgb_n_jobs": XGB_N_JOBS,
                },
            )
            mean_auroc = float(np.nanmean(auroc_scores))
            std_auroc = float(np.nanstd(auroc_scores))
            mean_auprc = float(np.nanmean(auprc_scores))
            std_auprc = float(np.nanstd(auprc_scores))
            prevalence = float(pd.Series(y).astype(int).mean())
            lift = float(mean_auprc / prevalence) if prevalence else np.nan
            raw_score = combined_discrimination_score(
                mean_auroc,
                mean_auprc,
                prevalence,
                tuning_objective_metric,
            )
            trial.set_user_attr("cv_mean_auroc", mean_auroc)
            trial.set_user_attr("cv_std_auroc", std_auroc)
            trial.set_user_attr("cv_mean_auprc", mean_auprc)
            trial.set_user_attr("cv_std_auprc", std_auprc)
            trial.set_user_attr("cv_prevalence", prevalence)
            trial.set_user_attr("cv_auprc_lift", lift)
            trial.set_user_attr("cv_repeats", int(effective_cv_repeats))
            trial.set_user_attr("cv_selection_metric", tuning_objective_metric)
            score = raw_score - OPTUNA_STD_PENALTY * std_auroc
        except Exception:
            score = np.nan
        if not np.isfinite(score):
            return 0.5
        return score

    tuning_started_at = time.perf_counter()

    def report_progress(study, trial):
        completed = int(trial.number) + 1
        if (
            OPTUNA_PROGRESS_INTERVAL > 0
            and int(n_trials) >= OPTUNA_PROGRESS_INTERVAL
            and (
                completed % OPTUNA_PROGRESS_INTERVAL == 0 or completed == int(n_trials)
            )
        ):
            elapsed = time.perf_counter() - tuning_started_at
            print(
                f"Optuna progress | {analysis_key or modeling_method} | "
                f"{completed}/{int(n_trials)} trials | "
                f"elapsed={elapsed:.1f}s | best={study.best_value:.6f}"
            )

    study.optimize(
        objective,
        n_trials=int(n_trials),
        timeout=OPTUNA_TIMEOUT,
        n_jobs=1,
        show_progress_bar=False,
        callbacks=[report_progress],
    )
    return OptunaSearchResult(study)


def cross_validated_probabilities(
    X,
    y,
    param_list,
    seed,
    cv_role="train_cv",
    seed_bagging_repeats=SEED_BAGGING_REPEATS,
):
    cv = make_cv(y, K_FOLD_CV_SPLITS, seed)
    prob = np.full(len(y), np.nan, dtype=float)
    fold_rows = []

    def fit_fold(fold_idx, train_idx, heldout_idx):
        fold_seed = int(seed + 7919 * (fold_idx + 1))
        model = fit_ensemble_with_params(
            X.iloc[train_idx],
            y.iloc[train_idx],
            param_list,
            fold_seed,
            seed_bagging_repeats=seed_bagging_repeats,
        )
        fold_probability = model.predict_proba(X.iloc[heldout_idx])[:, 1]
        fit_info_model = representative_pipeline(model)
        missing_step = fit_info_model.named_steps["missingness"]
        preprocess_step = fit_info_model.named_steps["preprocess"]
        select_step = fit_info_model.named_steps["select"]
        row = {
            "cv_role": cv_role,
            "threshold_cv_fold": fold_idx,
            "fit_n": int(len(train_idx)),
            "heldout_n": int(len(heldout_idx)),
            "fit_positive_n": int(y.iloc[train_idx].sum()),
            "heldout_positive_n": int(y.iloc[heldout_idx].sum()),
            "ensemble_top_k": int(len(param_list)),
            "n_features_before_missingness_filter": int(X.shape[1]),
            "n_features_after_missingness_filter": int(len(missing_step.keep_columns_)),
            "n_features_after_mice_preprocessing": int(
                len(preprocess_step.encoded_columns_)
            ),
            "feature_selection_k": select_step.k,
            "n_features_after_feature_selection": int(
                len(select_step.selected_indices_)
            ),
            "dropped_missingness_columns": "|".join(missing_step.dropped_columns_),
            "dropped_unusable_columns": "|".join(preprocess_step.unusable_columns_),
            "numeric_columns_after_filter": "|".join(preprocess_step.numeric_columns_),
            "categorical_columns_after_filter": "|".join(
                preprocess_step.categorical_columns_
            ),
        }
        return heldout_idx, fold_probability, row

    splits = list(enumerate(cv.split(X, y)))
    with parallel_backend("threading", n_jobs=SEARCH_N_JOBS):
        fold_results = Parallel(n_jobs=SEARCH_N_JOBS, prefer="threads")(
            delayed(fit_fold)(fold_idx, train_idx, heldout_idx)
            for fold_idx, (train_idx, heldout_idx) in splits
        )
    for heldout_idx, fold_probability, row in fold_results:
        prob[heldout_idx] = fold_probability
        fold_rows.append(row)
    if np.isnan(prob).any():
        raise RuntimeError("Cross-validated probabilities contain missing values.")
    return prob, pd.DataFrame(fold_rows)
