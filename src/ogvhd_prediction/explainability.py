"""Final fitting and SHAP feature attribution."""

from .utils import (
    DMatrix,
    Parallel,
    delayed,
    np,
    parallel_backend,
    pd,
    shap,
)
from .config import (
    K_FOLD_CV_SPLITS,
    SEARCH_N_JOBS,
    SEED_BAGGING_REPEATS,
)
from .splitting import make_cv
from .tuning import (
    fit_ensemble_with_params,
    representative_pipeline,
)


def fit_final_with_params(X, y, best_params, seed):
    if isinstance(best_params, dict):
        param_list = [best_params]
    else:
        param_list = list(best_params)
    return fit_ensemble_with_params(X, y, param_list, seed)


def extract_fit_info(model, X):
    model = representative_pipeline(model)
    missing_step = model.named_steps["missingness"]
    preprocess_step = model.named_steps["preprocess"]
    select_step = model.named_steps["select"]
    try:
        transformed_feature_names = preprocess_step.get_feature_names_out().tolist()
    except Exception:
        transformed_feature_names = []
    selected_indices = getattr(
        select_step, "selected_indices_", np.arange(len(transformed_feature_names))
    )
    selected_feature_names = [
        transformed_feature_names[idx]
        for idx in selected_indices
        if idx < len(transformed_feature_names)
    ]
    return {
        "n_features_before_missingness_filter": int(X.shape[1]),
        "n_features_after_missingness_filter": int(len(missing_step.keep_columns_)),
        "n_features_after_mice_preprocessing": int(
            len(preprocess_step.encoded_columns_)
        ),
        "feature_selection_k": select_step.k,
        "n_features_after_feature_selection": int(len(selected_indices)),
        "dropped_missingness_columns": missing_step.dropped_columns_,
        "dropped_unusable_columns": preprocess_step.unusable_columns_,
        "kept_columns_after_missingness_filter": missing_step.keep_columns_,
        "numeric_columns_after_filter": preprocess_step.numeric_columns_,
        "categorical_columns_after_filter": preprocess_step.categorical_columns_,
        "n_transformed_features": int(len(transformed_feature_names)),
        "transformed_feature_names": transformed_feature_names,
        "selected_transformed_feature_names": selected_feature_names,
    }


def transformed_feature_to_base_feature(name, categorical_columns=None):
    raw = str(name)
    if raw.startswith("numeric__"):
        return raw.removeprefix("numeric__")
    if raw.startswith("categorical__"):
        encoded = raw.removeprefix("categorical__")
        for col in sorted(categorical_columns or [], key=len, reverse=True):
            prefix = f"{col}_"
            if encoded.startswith(prefix):
                return col
        if "_" in encoded:
            return encoded.split("_", 1)[0]
        return encoded
    return raw


def xgb_shap_contributions(model, X_selected):
    X_selected = np.asarray(X_selected, dtype=float)
    try:
        contributions = model.get_booster().predict(
            DMatrix(X_selected),
            pred_contribs=True,
            validate_features=False,
        )
        contributions = np.asarray(contributions, dtype=float)
        if contributions.ndim == 3:
            contributions = contributions[:, -1, :]
        if (
            contributions.ndim == 2
            and contributions.shape[1] == X_selected.shape[1] + 1
        ):
            return contributions[:, :-1]
    except Exception:
        pass

    explainer = shap.TreeExplainer(model)
    contributions = explainer.shap_values(X_selected)
    if isinstance(contributions, list):
        contributions = contributions[-1]
    return np.asarray(contributions, dtype=float)


def shap_base_importance_for_pipeline(pipeline, X):
    missing_step = pipeline.named_steps["missingness"]
    preprocess_step = pipeline.named_steps["preprocess"]
    select_step = pipeline.named_steps["select"]
    X_missing = missing_step.transform(X)
    X_preprocessed = preprocess_step.transform(X_missing)
    X_selected = select_step.transform(X_preprocessed)
    feature_names = preprocess_step.get_feature_names_out().tolist()
    selected_indices = getattr(
        select_step, "selected_indices_", np.arange(len(feature_names))
    )
    selected_feature_names = [
        feature_names[idx] for idx in selected_indices if idx < len(feature_names)
    ]
    if X_selected.size == 0 or not selected_feature_names:
        return {}
    shap_values = xgb_shap_contributions(
        pipeline.named_steps["model"],
        X_selected,
    )
    mean_abs = np.nanmean(np.abs(shap_values), axis=0)
    importance = {}
    for transformed_name, value in zip(selected_feature_names, mean_abs):
        base_name = transformed_feature_to_base_feature(
            transformed_name,
            preprocess_step.categorical_columns_,
        )
        importance[base_name] = importance.get(base_name, 0.0) + float(value)
    return importance


# Rank features with training-fold SHAP only; never expose the holdout cohort.
def cv_shap_top_base_features(
    X,
    y,
    param_list,
    seed,
    top_n,
    seed_bagging_repeats=SEED_BAGGING_REPEATS,
):
    cv = make_cv(y, K_FOLD_CV_SPLITS, seed)
    weighted_importance = {}
    normalized_importance = {feature: [] for feature in X.columns}
    percentile_rank_score = {feature: [] for feature in X.columns}
    top_n_frequency = {feature: [] for feature in X.columns}
    fold_rows = []

    def fit_fold(fold_idx, fit_idx, heldout_idx):
        fold_seed = int(seed + 524287 * (fold_idx + 1))
        model = fit_ensemble_with_params(
            X.iloc[fit_idx],
            y.iloc[fit_idx],
            param_list,
            fold_seed,
            seed_bagging_repeats=seed_bagging_repeats,
        )
        pipelines = model.models_ if hasattr(model, "models_") else [model]
        member_importances = []
        for pipeline in pipelines:
            member_importances.append(
                shap_base_importance_for_pipeline(pipeline, X.iloc[heldout_idx])
            )
        fold_importance = {}
        for member_importance in member_importances:
            for feature, value in member_importance.items():
                fold_importance[feature] = fold_importance.get(feature, 0.0) + float(
                    value
                )
        if member_importances:
            fold_importance = {
                feature: value / len(member_importances)
                for feature, value in fold_importance.items()
            }
        row = {
            "simplified_cv_fold": fold_idx,
            "fit_n": int(len(fit_idx)),
            "heldout_n": int(len(heldout_idx)),
            "fit_positive_n": int(y.iloc[fit_idx].sum()),
            "heldout_positive_n": int(y.iloc[heldout_idx].sum()),
            "n_ranked_features": int(len(fold_importance)),
        }
        return heldout_idx, fold_importance, row

    splits = list(enumerate(cv.split(X, y)))
    with parallel_backend("threading", n_jobs=SEARCH_N_JOBS):
        fold_results = Parallel(n_jobs=SEARCH_N_JOBS, prefer="threads")(
            delayed(fit_fold)(fold_idx, fit_idx, heldout_idx)
            for fold_idx, (fit_idx, heldout_idx) in splits
        )

    for heldout_idx, fold_importance, row in fold_results:
        ranked_features = sorted(
            (
                (feature, float(value))
                for feature, value in fold_importance.items()
                if feature in X.columns and np.isfinite(value)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        rank_by_feature = {
            feature: rank for rank, (feature, _) in enumerate(ranked_features, start=1)
        }
        total_importance = sum(max(value, 0.0) for _, value in ranked_features)
        rank_denominator = max(1, len(ranked_features) - 1)
        for feature in X.columns:
            value = float(fold_importance.get(feature, 0.0))
            normalized_importance[feature].append(
                max(value, 0.0) / total_importance if total_importance > 0 else 0.0
            )
            rank = rank_by_feature.get(feature)
            percentile_rank_score[feature].append(
                1.0 - ((rank - 1) / rank_denominator) if rank is not None else 0.0
            )
            top_n_frequency[feature].append(
                1.0 if rank is not None and rank <= int(top_n) else 0.0
            )
        for feature, value in fold_importance.items():
            weighted_importance[feature] = weighted_importance.get(feature, 0.0) + (
                float(value) * len(heldout_idx)
            )
        fold_rows.append(row)

    ranking = pd.DataFrame(
        [
            {
                "feature": feature,
                "mean_abs_shap": importance / max(1, len(y)),
                "mean_normalized_shap": float(np.mean(normalized_importance[feature])),
                "mean_shap_percentile_rank": float(
                    np.mean(percentile_rank_score[feature])
                ),
                "top_n_fold_frequency": float(np.mean(top_n_frequency[feature])),
            }
            for feature, importance in weighted_importance.items()
            if feature in X.columns
        ]
    )
    if ranking.empty:
        ranking = pd.DataFrame(
            {
                "feature": list(X.columns),
                "mean_abs_shap": np.zeros(X.shape[1], dtype=float),
                "mean_normalized_shap": np.zeros(X.shape[1], dtype=float),
                "mean_shap_percentile_rank": np.zeros(X.shape[1], dtype=float),
                "top_n_fold_frequency": np.zeros(X.shape[1], dtype=float),
            }
        )
    max_normalized = float(ranking["mean_normalized_shap"].max())
    ranking["normalized_shap_magnitude_score"] = (
        ranking["mean_normalized_shap"] / max_normalized if max_normalized > 0 else 0.0
    )
    ranking["shap_stability_score"] = (
        0.50 * ranking["mean_shap_percentile_rank"]
        + 0.30 * ranking["top_n_fold_frequency"]
        + 0.20 * ranking["normalized_shap_magnitude_score"]
    )
    ranking = ranking.sort_values(
        ["mean_abs_shap", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    selected_features = ranking.head(int(top_n))["feature"].tolist()
    if len(selected_features) < int(top_n):
        selected = set(selected_features)
        selected_features.extend(
            [col for col in X.columns if col not in selected][
                : int(top_n) - len(selected_features)
            ]
        )
    fold_info = pd.DataFrame(fold_rows)
    return selected_features, ranking, fold_info
