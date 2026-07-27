"""Discrimination, calibration, threshold, and bootstrap metrics."""

from . import targets as _targets  # noqa: F401 - preserve pipeline stage order.
from .utils import (
    IsotonicRegression,
    LogisticRegression,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    hashlib,
    jsonable,
    np,
    pd,
    roc_auc_score,
    zlib,
)
from .config import (
    ALLOW_PROBABILITY_ORIENTATION_CORRECTION,
    AUPRC_LIFT_WEIGHT,
    AUROC_NEAR_BEST_TOLERANCE,
    BOOTSTRAP_CI_CONFIDENCE_LEVEL,
    BOOTSTRAP_CI_MIN_VALID_RESAMPLES,
    BOOTSTRAP_CI_N_RESAMPLES,
    BOOTSTRAP_CI_RANDOM_STATE,
    CALIBRATION_DIAGNOSTIC_METHOD,
    CALIBRATION_N_BINS,
    DUAL_DISCRIMINATION_SELECTION,
    ENFORCE_POSITIVE_CALIBRATION_SLOPE,
    INNER_THRESHOLD_SELECTION_METRIC,
    MODEL_VARIANT_ORIGINAL,
    NET_BENEFIT_THRESHOLD,
    ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC,
    ORIENTATION_CORRECTION_REQUIRES_TRAIN_CV_SUPPORT,
    ORIENTATION_CORRECTION_VALIDATION_ONLY_ANALYSES,
    RUN_BOOTSTRAP_CI,
    THRESHOLD_ACCURACY_SENSITIVITY_OBJECTIVE,
    THRESHOLD_CANDIDATE_POLICY,
    THRESHOLD_GRID,
    THRESHOLD_MIN_SENSITIVITY,
    THRESHOLD_OBJECTIVE,
    THRESHOLD_OBJECTIVES,
    THRESHOLD_OPERATING_POINT_STATUS,
    THRESHOLD_TIE_BREAK_POLICY,
    VALIDATION_MODEL_SELECTION_METRIC,
)
from .targets import min_class_count


def net_benefit_at_threshold(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    n = len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if threshold >= 1:
        return np.nan
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_prob) & np.isin(y_true, [0.0, 1.0])
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid]
    if not len(y_true):
        return np.nan
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if right == 1.0:
            in_bin = (y_prob >= left) & (y_prob <= right)
        else:
            in_bin = (y_prob >= left) & (y_prob < right)
        if not np.any(in_bin):
            continue
        bin_weight = np.mean(in_bin)
        ece += bin_weight * abs(np.mean(y_true[in_bin]) - np.mean(y_prob[in_bin]))
    return float(ece)


def calibration_regression_metrics(y_true, y_prob, max_iter=100, tolerance=1e-8):
    """Estimate calibration-in-the-large and an unpenalized logistic calibration slope."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_prob) & np.isin(y_true, [0.0, 1.0])
    y_true = y_true[valid].astype(float)
    y_prob = y_prob[valid]
    metrics = {
        "calibration_intercept": np.nan,
        "calibration_slope": np.nan,
    }
    if len(y_true) < 3 or len(np.unique(y_true)) < 2:
        return metrics

    linear_predictor = safe_logit(y_prob)
    if not np.all(np.isfinite(linear_predictor)):
        return metrics

    target_events = float(np.sum(y_true))
    lower, upper = -100.0, 100.0
    for _ in range(max_iter):
        midpoint = (lower + upper) / 2.0
        eta = linear_predictor + midpoint
        fitted = np.empty_like(eta)
        positive = eta >= 0
        fitted[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
        exp_eta = np.exp(eta[~positive])
        fitted[~positive] = exp_eta / (1.0 + exp_eta)
        if float(np.sum(fitted)) < target_events:
            lower = midpoint
        else:
            upper = midpoint
        if (upper - lower) < tolerance:
            break
    calibration_intercept = (lower + upper) / 2.0
    if np.isfinite(calibration_intercept) and abs(calibration_intercept) <= 100:
        metrics["calibration_intercept"] = float(calibration_intercept)

    if np.ptp(linear_predictor) <= 1e-12:
        return metrics

    design = np.column_stack([np.ones(len(linear_predictor)), linear_predictor])
    prevalence = np.clip(np.mean(y_true), 1e-6, 1.0 - 1e-6)
    coefficients = np.array([float(safe_logit([prevalence])[0]), 1.0], dtype=float)

    def log_likelihood(beta):
        eta = design @ beta
        return float(np.sum(y_true * eta - np.logaddexp(0.0, eta)))

    converged = False
    current_log_likelihood = log_likelihood(coefficients)
    for _ in range(max_iter):
        eta = design @ coefficients
        fitted = np.empty_like(eta)
        positive = eta >= 0
        fitted[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
        exp_eta = np.exp(eta[~positive])
        fitted[~positive] = exp_eta / (1.0 + exp_eta)
        weights = np.clip(fitted * (1.0 - fitted), 1e-12, None)
        score = design.T @ (y_true - fitted)
        information = design.T @ (weights[:, None] * design)
        if not np.all(np.isfinite(information)) or np.linalg.cond(information) > 1e12:
            return metrics
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            return metrics
        if not np.all(np.isfinite(step)):
            return metrics

        step_scale = 1.0
        updated = coefficients + step
        updated_log_likelihood = log_likelihood(updated)
        while (
            not np.isfinite(updated_log_likelihood)
            or updated_log_likelihood < current_log_likelihood
        ) and step_scale > 1e-6:
            step_scale *= 0.5
            updated = coefficients + (step_scale * step)
            updated_log_likelihood = log_likelihood(updated)
        if not np.isfinite(updated_log_likelihood):
            return metrics

        coefficients = updated
        current_log_likelihood = updated_log_likelihood
        if np.max(np.abs(step_scale * step)) < tolerance:
            converged = True
            break

    if converged and np.all(np.isfinite(coefficients)) and abs(coefficients[1]) <= 100:
        metrics["calibration_slope"] = float(coefficients[1])
    return metrics


def threshold_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    balanced_accuracy = np.nanmean([sensitivity, specificity])
    youden_index = (
        sensitivity + specificity - 1 if not np.isnan(balanced_accuracy) else np.nan
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "ppv": ppv,
        "npv": npv,
        "balanced_accuracy": balanced_accuracy,
        "youden_index": youden_index,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "net_benefit": net_benefit_at_threshold(y_true, y_prob, threshold),
        "decision_curve_net_benefit_at_configured_threshold": net_benefit_at_threshold(
            y_true,
            y_prob,
            NET_BENEFIT_THRESHOLD,
        ),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def classification_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    metrics = threshold_metrics(y_true, y_prob, threshold)
    prevalence = float(np.mean(y_true)) if len(y_true) else np.nan
    auprc = average_precision_score(y_true, y_prob)
    calibration_metrics = calibration_regression_metrics(y_true, y_prob)
    metrics.update(
        {
            "auroc": roc_auc_score(y_true, y_prob)
            if len(np.unique(y_true)) == 2
            else np.nan,
            "auprc": auprc,
            "prevalence": prevalence,
            "auprc_baseline": prevalence,
            "auprc_lift": float(auprc / prevalence)
            if prevalence and np.isfinite(prevalence)
            else np.nan,
            "ece": expected_calibration_error(y_true, y_prob, CALIBRATION_N_BINS),
            "n": int(len(y_true)),
            "positive_n": int(y_true.sum()),
        }
    )
    metrics.update(calibration_metrics)
    return metrics


BOOTSTRAP_CI_METRICS = (
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "ece",
    "calibration_intercept",
    "calibration_slope",
)

CALIBRATION_DIAGNOSTIC_METRICS = (
    "ece",
    "calibration_intercept",
    "calibration_slope",
)

BOOTSTRAP_CI_OUTPUT_COLUMNS = [
    "target",
    "feature_set",
    "model_variant",
    "dataset_split",
    "metric",
    "point_estimate",
    "ci_lower",
    "ci_upper",
    "confidence_level",
    "bootstrap_requested_resamples",
    "bootstrap_valid_resamples",
    "bootstrap_invalid_resamples",
    "bootstrap_minimum_valid_resamples",
    "bootstrap_status",
    "bootstrap_method",
    "sampling_unit",
    "bootstrap_seed",
    "n",
    "positive_n",
    "negative_n",
    "threshold",
]


def diagnostic_metric_values(y_true, y_prob, threshold, calibration_prob=None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if calibration_prob is None:
        calibration_prob = y_prob
    calibration_prob = np.asarray(calibration_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    positives = int(tp + fn)
    negatives = int(tn + fp)
    predicted_positives = int(tp + fp)
    predicted_negatives = int(tn + fn)
    f1_denominator = (2 * tp) + fp + fn
    metrics = {
        "auroc": roc_auc_score(y_true, y_prob) if positives and negatives else np.nan,
        "auprc": average_precision_score(y_true, y_prob) if positives else np.nan,
        "accuracy": (tp + tn) / n if n else np.nan,
        "sensitivity": tp / positives if positives else np.nan,
        "specificity": tn / negatives if negatives else np.nan,
        "ppv": tp / predicted_positives if predicted_positives else np.nan,
        "npv": tn / predicted_negatives if predicted_negatives else np.nan,
        "f1": (2 * tp) / f1_denominator if f1_denominator else 0.0,
    }
    calibration_metrics = calibration_regression_metrics(y_true, calibration_prob)
    metrics.update(
        {
            "ece": expected_calibration_error(
                y_true, calibration_prob, CALIBRATION_N_BINS
            ),
            **calibration_metrics,
        }
    )
    return metrics


def bootstrap_seed_for_model(target, feature_set):
    model_key = f"{target}|{feature_set}".encode("utf-8")
    return int((BOOTSTRAP_CI_RANDOM_STATE + zlib.crc32(model_key)) % (2**32))


PAIRED_LANDMARK_AUDIT_COLUMNS = [
    "target",
    "model_variant",
    "reference_feature_set",
    "landmark_feature_set",
    "reference_split_seed",
    "landmark_split_seed",
    "same_split_seed",
    "reference_full_test_n",
    "landmark_full_test_n",
    "common_test_n",
    "common_test_positive_n",
    "common_test_negative_n",
    "common_fraction_of_landmark_test",
    "same_labels_on_common_rows",
    "reference_full_test_auroc",
    "landmark_full_test_auroc",
    "reference_common_risk_set_auroc",
    "landmark_common_risk_set_auroc",
    "landmark_minus_reference_auroc",
    "difference_ci_lower",
    "difference_ci_upper",
    "confidence_level",
    "bootstrap_requested_resamples",
    "bootstrap_valid_resamples",
    "bootstrap_method",
    "bootstrap_seed",
    "comparison_status",
    "evidence_of_decline",
    "evidence_of_improvement",
]


def paired_stratified_auroc_difference_ci(
    y_true,
    reference_probability,
    landmark_probability,
    seed,
):
    y_true = np.asarray(y_true, dtype=int)
    reference_probability = np.asarray(reference_probability, dtype=float)
    landmark_probability = np.asarray(landmark_probability, dtype=float)
    positive_indices = np.flatnonzero(y_true == 1)
    negative_indices = np.flatnonzero(y_true == 0)
    if not len(positive_indices) or not len(negative_indices):
        return np.nan, np.nan, 0, "not_estimable_single_class"
    if not RUN_BOOTSTRAP_CI:
        return np.nan, np.nan, 0, "disabled"

    rng = np.random.default_rng(int(seed))
    differences = np.empty(BOOTSTRAP_CI_N_RESAMPLES, dtype=float)
    batch_size = min(250, BOOTSTRAP_CI_N_RESAMPLES)

    def sampled_aurocs(probability, sampled_positive, sampled_negative):
        pairwise_difference = (
            probability[sampled_positive][:, :, None]
            - probability[sampled_negative][:, None, :]
        )
        return np.mean(pairwise_difference > 0.0, axis=(1, 2)) + 0.5 * np.mean(
            pairwise_difference == 0.0, axis=(1, 2)
        )

    for start in range(0, BOOTSTRAP_CI_N_RESAMPLES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_CI_N_RESAMPLES)
        current_batch_size = stop - start
        sampled_positive = positive_indices[
            rng.integers(
                0,
                len(positive_indices),
                size=(current_batch_size, len(positive_indices)),
            )
        ]
        sampled_negative = negative_indices[
            rng.integers(
                0,
                len(negative_indices),
                size=(current_batch_size, len(negative_indices)),
            )
        ]
        differences[start:stop] = sampled_aurocs(
            landmark_probability,
            sampled_positive,
            sampled_negative,
        ) - sampled_aurocs(
            reference_probability,
            sampled_positive,
            sampled_negative,
        )

    alpha = 1.0 - BOOTSTRAP_CI_CONFIDENCE_LEVEL
    lower, upper = np.quantile(
        differences,
        [alpha / 2.0, 1.0 - (alpha / 2.0)],
    )
    return float(lower), float(upper), int(len(differences)), "ok"


def bootstrap_metric_ci_for_prediction_group(group):
    group = group.copy()
    sort_columns = [
        col
        for col in ["source_row_index", "analysis_row_index"]
        if col in group.columns
    ]
    if sort_columns:
        group = group.sort_values(sort_columns, kind="mergesort")
    if (
        "source_row_index" in group.columns
        and group["source_row_index"].duplicated().any()
    ):
        raise ValueError(
            "Bootstrap input contains duplicate held-out transplantation episodes within one model."
        )

    probability_column = (
        "probability" if "probability" in group.columns else "raw_probability"
    )
    y_true = pd.to_numeric(group["y_true"], errors="coerce").to_numpy(dtype=float)
    y_prob = pd.to_numeric(group[probability_column], errors="coerce").to_numpy(
        dtype=float
    )
    calibration_prob = pd.to_numeric(
        group.get("calibration_probability", group[probability_column]),
        errors="coerce",
    ).to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_prob) & np.isin(y_true, [0.0, 1.0])
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid].astype(float)
    calibration_prob = calibration_prob[valid].astype(float)
    if not len(y_true):
        return pd.DataFrame(columns=BOOTSTRAP_CI_OUTPUT_COLUMNS)

    thresholds = pd.to_numeric(group.loc[valid, "threshold"], errors="coerce").dropna()
    if thresholds.empty:
        raise ValueError(
            "Bootstrap input does not contain a validation-selected threshold."
        )
    threshold = float(thresholds.iloc[0])
    if not np.allclose(
        thresholds.to_numpy(dtype=float), threshold, atol=1e-12, rtol=0.0
    ):
        raise ValueError(
            "Bootstrap input contains more than one threshold for the same model."
        )

    target = str(group["target"].iloc[0])
    feature_set = str(group["feature_set"].iloc[0])
    model_variant = str(group["model_variant"].iloc[0])
    seed = bootstrap_seed_for_model(target, feature_set)
    rng = np.random.default_rng(seed)
    bootstrap_values = {
        metric: np.full(BOOTSTRAP_CI_N_RESAMPLES, np.nan, dtype=float)
        for metric in BOOTSTRAP_CI_METRICS
    }

    n = len(y_true)
    point_estimates = diagnostic_metric_values(
        y_true,
        y_prob,
        threshold,
        calibration_prob=calibration_prob,
    )
    for resample_index in range(BOOTSTRAP_CI_N_RESAMPLES):
        sampled_indices = rng.integers(0, n, size=n)
        estimates = diagnostic_metric_values(
            y_true[sampled_indices],
            y_prob[sampled_indices],
            threshold,
            calibration_prob=calibration_prob[sampled_indices],
        )
        for metric in BOOTSTRAP_CI_METRICS:
            bootstrap_values[metric][resample_index] = estimates[metric]

    alpha = 1.0 - BOOTSTRAP_CI_CONFIDENCE_LEVEL
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - lower_quantile
    rows = []
    for metric in BOOTSTRAP_CI_METRICS:
        metric_values = bootstrap_values[metric]
        finite_values = metric_values[np.isfinite(metric_values)]
        valid_n = int(len(finite_values))
        status = (
            "ok"
            if valid_n >= BOOTSTRAP_CI_MIN_VALID_RESAMPLES
            else "insufficient_valid_resamples"
        )
        if status == "ok":
            ci_lower, ci_upper = np.quantile(
                finite_values,
                [lower_quantile, upper_quantile],
            )
        else:
            ci_lower, ci_upper = np.nan, np.nan
        rows.append(
            {
                "target": target,
                "feature_set": feature_set,
                "model_variant": model_variant,
                "dataset_split": "test",
                "metric": metric,
                "point_estimate": point_estimates[metric],
                "ci_lower": float(ci_lower) if np.isfinite(ci_lower) else np.nan,
                "ci_upper": float(ci_upper) if np.isfinite(ci_upper) else np.nan,
                "confidence_level": BOOTSTRAP_CI_CONFIDENCE_LEVEL,
                "bootstrap_requested_resamples": BOOTSTRAP_CI_N_RESAMPLES,
                "bootstrap_valid_resamples": valid_n,
                "bootstrap_invalid_resamples": BOOTSTRAP_CI_N_RESAMPLES - valid_n,
                "bootstrap_minimum_valid_resamples": BOOTSTRAP_CI_MIN_VALID_RESAMPLES,
                "bootstrap_status": status,
                "bootstrap_method": "ordinary_transplantation_episode_bootstrap_percentile_ci",
                "sampling_unit": "held_out_transplantation_episode",
                "bootstrap_seed": seed,
                "n": n,
                "positive_n": int(y_true.sum()),
                "negative_n": int(n - y_true.sum()),
                "threshold": threshold,
            }
        )
    return pd.DataFrame(rows, columns=BOOTSTRAP_CI_OUTPUT_COLUMNS)


def bootstrap_holdout_metric_cis(holdout_predictions_df):
    if not RUN_BOOTSTRAP_CI or holdout_predictions_df.empty:
        return pd.DataFrame(columns=BOOTSTRAP_CI_OUTPUT_COLUMNS)
    predictions = ensure_model_variant_column(holdout_predictions_df)
    predictions = predictions.loc[predictions["dataset_split"] == "test"].copy()
    ci_frames = []
    for _, group in predictions.groupby(
        ["target", "feature_set", "model_variant"],
        sort=False,
    ):
        ci_frames.append(bootstrap_metric_ci_for_prediction_group(group))
    if not ci_frames:
        return pd.DataFrame(columns=BOOTSTRAP_CI_OUTPUT_COLUMNS)
    return pd.concat(ci_frames, ignore_index=True).loc[:, BOOTSTRAP_CI_OUTPUT_COLUMNS]


def attach_bootstrap_cis_to_results(holdout_results_df, bootstrap_ci_df):
    if holdout_results_df.empty:
        return holdout_results_df
    key_columns = ["target", "feature_set", "model_variant"]
    results = ensure_model_variant_column(holdout_results_df)
    if bootstrap_ci_df.empty:
        for metric in BOOTSTRAP_CI_METRICS:
            prefix = f"test_{metric}"
            results[f"{prefix}_ci_lower"] = np.nan
            results[f"{prefix}_ci_upper"] = np.nan
            results[f"{prefix}_bootstrap_valid_resamples"] = 0
            results[f"{prefix}_bootstrap_status"] = "disabled"
        return results
    wide_rows = []
    for key, group in bootstrap_ci_df.groupby(key_columns, sort=False):
        row = dict(zip(key_columns, key))
        for _, ci_row in group.iterrows():
            prefix = f"test_{ci_row['metric']}"
            row[prefix] = ci_row["point_estimate"]
            row[f"{prefix}_ci_lower"] = ci_row["ci_lower"]
            row[f"{prefix}_ci_upper"] = ci_row["ci_upper"]
            row[f"{prefix}_bootstrap_valid_resamples"] = ci_row[
                "bootstrap_valid_resamples"
            ]
            row[f"{prefix}_bootstrap_status"] = ci_row["bootstrap_status"]
        wide_rows.append(row)
    ci_wide = pd.DataFrame(wide_rows)
    value_columns = [col for col in ci_wide.columns if col not in key_columns]
    results = results.drop(
        columns=[col for col in value_columns if col in results.columns],
        errors="ignore",
    )
    return results.merge(ci_wide, on=key_columns, how="left")


def add_bootstrap_cis_to_metric_summary(
    metric_summary_df,
    holdout_results_df,
    bootstrap_ci_df,
):
    key_columns = ["target", "feature_set", "model_variant"]
    summary = metric_summary_df.copy()
    results = ensure_model_variant_column(holdout_results_df)
    required_rows = []
    existing_keys = (
        {
            tuple(row)
            for row in summary.loc[:, key_columns + ["metric"]].itertuples(
                index=False, name=None
            )
        }
        if not summary.empty
        else set()
    )
    for _, result in results.iterrows():
        model_key = tuple(result[col] for col in key_columns)
        for metric in BOOTSTRAP_CI_METRICS:
            metric_name = f"test_{metric}"
            row_key = (*model_key, metric_name)
            if row_key in existing_keys:
                continue
            required_rows.append(
                {
                    **dict(zip(key_columns, model_key)),
                    "metric": metric_name,
                    "value": result.get(metric_name, np.nan),
                }
            )
            existing_keys.add(row_key)
    if required_rows:
        summary = pd.concat([summary, pd.DataFrame(required_rows)], ignore_index=True)
    if bootstrap_ci_df.empty:
        return summary

    ci_summary = bootstrap_ci_df.copy()
    ci_summary["metric"] = "test_" + ci_summary["metric"].astype(str)
    ci_summary = ci_summary.rename(
        columns={
            "point_estimate": "bootstrap_point_estimate",
        }
    )
    ci_columns = [
        "bootstrap_point_estimate",
        "ci_lower",
        "ci_upper",
        "confidence_level",
        "bootstrap_requested_resamples",
        "bootstrap_valid_resamples",
        "bootstrap_invalid_resamples",
        "bootstrap_minimum_valid_resamples",
        "bootstrap_status",
        "bootstrap_method",
        "sampling_unit",
        "bootstrap_seed",
    ]
    summary = summary.drop(
        columns=[col for col in ci_columns if col in summary.columns], errors="ignore"
    )
    summary = summary.merge(
        ci_summary.loc[:, key_columns + ["metric"] + ci_columns],
        on=key_columns + ["metric"],
        how="left",
    )
    has_bootstrap_estimate = summary["bootstrap_point_estimate"].notna()
    summary.loc[has_bootstrap_estimate, "value"] = summary.loc[
        has_bootstrap_estimate,
        "bootstrap_point_estimate",
    ]
    return summary


def combined_discrimination_score(auroc, auprc, prevalence, metric_name):
    auroc = float(auroc) if auroc is not None and np.isfinite(auroc) else -np.inf
    auprc = float(auprc) if auprc is not None and np.isfinite(auprc) else -np.inf
    prevalence = (
        float(prevalence)
        if prevalence is not None and np.isfinite(prevalence)
        else np.nan
    )
    lift = auprc / prevalence if prevalence and np.isfinite(prevalence) else -np.inf

    if metric_name == "auroc":
        return auroc
    if metric_name == "auprc":
        return auprc
    if metric_name == "auprc_lift":
        return lift
    if metric_name == "auroc_auprc":
        return auroc + auprc
    if metric_name == "auroc_auprc_lift":
        lift_bonus = np.log1p(max(lift, 0.0)) if np.isfinite(lift) else 0.0
        return auroc + AUPRC_LIFT_WEIGHT * lift_bonus
    raise ValueError(metric_name)


def safe_logit(probability, eps=1e-6):
    probability = np.clip(np.asarray(probability, dtype=float), eps, 1 - eps)
    return np.log(probability / (1 - probability))


def fit_probability_recalibrator(y_true, raw_prob):
    y_true = np.asarray(y_true).astype(int)
    raw_prob = np.clip(np.asarray(raw_prob, dtype=float), 1e-6, 1 - 1e-6)

    if CALIBRATION_DIAGNOSTIC_METHOD == "none" or len(np.unique(y_true)) < 2:
        return {"method": "none", "model": None}

    if CALIBRATION_DIAGNOSTIC_METHOD == "sigmoid":
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(safe_logit(raw_prob).reshape(-1, 1), y_true)
        slope = float(model.coef_.ravel()[0])
        if ENFORCE_POSITIVE_CALIBRATION_SLOPE and slope <= 0:
            return {
                "method": "none",
                "model": None,
                "rejected_method": "sigmoid",
                "rejected_reason": "non_positive_calibration_slope",
                "rejected_slope": slope,
            }
        return {"method": "sigmoid", "model": model}

    if CALIBRATION_DIAGNOSTIC_METHOD == "isotonic":
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(raw_prob, y_true)
        return {"method": "isotonic", "model": model}

    raise ValueError(CALIBRATION_DIAGNOSTIC_METHOD)


def apply_probability_recalibrator(recalibrator, raw_prob):
    raw_prob = np.clip(np.asarray(raw_prob, dtype=float), 1e-6, 1 - 1e-6)
    method = recalibrator.get("method", "none")
    model = recalibrator.get("model")

    if method == "none" or model is None:
        calibrated = raw_prob
    elif method == "sigmoid":
        calibrated = model.predict_proba(safe_logit(raw_prob).reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        calibrated = model.predict(raw_prob)
    else:
        raise ValueError(method)

    return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


def objective_value(y_true, y_prob, threshold, objective):
    metrics = threshold_metrics(y_true, y_prob, threshold)
    if objective == "youden":
        return metrics["youden_index"]
    if objective == "f1":
        return metrics["f1"]
    if objective == "net_benefit":
        return metrics["net_benefit"]
    if objective == THRESHOLD_ACCURACY_SENSITIVITY_OBJECTIVE:
        if (
            not np.isfinite(metrics["sensitivity"])
            or metrics["sensitivity"] + 1e-12 < THRESHOLD_MIN_SENSITIVITY
        ):
            return -np.inf
        return metrics["accuracy"]
    raise ValueError(objective)


def threshold_candidate_values(y_prob):
    observed = np.asarray(y_prob, dtype=float)
    observed = observed[np.isfinite(observed)]
    if not len(observed):
        raise ValueError(
            "Threshold selection requires at least one finite probability."
        )
    observed = np.clip(observed, 0.0, 1.0)
    # ``>= threshold`` needs one explicit state above the largest observed
    # score so that predicting no positives is represented in the audit grid.
    predict_none_boundary = np.nextafter(float(observed.max()), np.inf)
    return np.unique(
        np.concatenate(
            [
                THRESHOLD_GRID,
                observed,
                np.asarray([predict_none_boundary], dtype=float),
            ]
        )
    )


def probability_source_provenance(probability_source):
    provenance = {
        "raw": "model_class1_probability",
        "raw_inverted": "one_minus_model_class1_probability",
        "temporal_landmark_raw": (
            "validation_locked_weighted_mean_of_member_model_class1_probabilities"
        ),
        "temporal_landmark_calibrated": (
            "validation_fitted_recalibration_of_temporal_landmark_raw_probability"
        ),
        "calibrated": "diagnostic_recalibration_of_model_class1_probability",
    }
    return provenance.get(str(probability_source), str(probability_source))


def probability_recalibrator_metadata(recalibrator):
    method = str(recalibrator.get("method", "none"))
    model = recalibrator.get("model")
    metadata = {"method": method}
    for key in ("rejected_method", "rejected_reason", "rejected_slope"):
        if key in recalibrator:
            metadata[key] = jsonable(recalibrator[key])
    if method == "sigmoid" and model is not None:
        metadata.update(
            {
                "coefficient": [float(value) for value in model.coef_.ravel()],
                "intercept": [float(value) for value in model.intercept_.ravel()],
            }
        )
    elif method == "isotonic" and model is not None:
        metadata.update(
            {
                "x_thresholds": [float(value) for value in model.X_thresholds_],
                "y_thresholds": [float(value) for value in model.y_thresholds_],
            }
        )
    return metadata


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_optimization_key(score, metrics, threshold):
    sensitivity = metrics.get("sensitivity", np.nan)
    specificity = metrics.get("specificity", np.nan)
    return (
        float(score),
        float(sensitivity) if np.isfinite(sensitivity) else -np.inf,
        float(specificity) if np.isfinite(specificity) else -np.inf,
        -float(threshold),
    )


def optimize_threshold(y_true, y_prob, objective):
    best_threshold = None
    best_score = None
    best_key = None
    for threshold in threshold_candidate_values(y_prob):
        score = objective_value(y_true, y_prob, threshold, objective)
        if not np.isfinite(score):
            continue
        metrics = threshold_metrics(y_true, y_prob, threshold)
        candidate_key = threshold_optimization_key(score, metrics, threshold)
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_score = float(score)
            best_threshold = float(threshold)
    if best_threshold is None:
        raise ValueError(
            f"No feasible threshold for objective={objective!r}; "
            f"minimum_sensitivity={THRESHOLD_MIN_SENSITIVITY:.3f}."
        )
    return best_threshold, best_score


def select_threshold_from_oof(y_oof, prob_oof, calibration_prob=None):
    calibration_metrics = None
    if calibration_prob is not None:
        calibration_prob = np.asarray(calibration_prob, dtype=float)
        if len(calibration_prob) != len(y_oof):
            raise ValueError(
                "Calibration probabilities must align one-to-one with OOF labels."
            )
        calibration_metrics = classification_metrics(
            y_oof,
            calibration_prob,
            threshold=0.5,
        )

    candidate_rows = []
    threshold_candidate_count = int(len(threshold_candidate_values(prob_oof)))
    for objective in THRESHOLD_OBJECTIVES:
        threshold, objective_score = optimize_threshold(y_oof, prob_oof, objective)
        metrics = classification_metrics(y_oof, prob_oof, threshold)
        if calibration_metrics is not None:
            for metric in CALIBRATION_DIAGNOSTIC_METRICS:
                metrics[metric] = calibration_metrics[metric]
        selection_score = metrics.get(INNER_THRESHOLD_SELECTION_METRIC)
        if selection_score is None:
            raise ValueError(
                f"Unknown threshold selection metric: {INNER_THRESHOLD_SELECTION_METRIC}"
            )
        row = {
            "candidate_objective": objective,
            "candidate_threshold": threshold,
            "candidate_objective_score": objective_score,
            "candidate_selection_metric": INNER_THRESHOLD_SELECTION_METRIC,
            "candidate_selection_score": selection_score,
            "candidate_threshold_policy": THRESHOLD_CANDIDATE_POLICY,
            "candidate_threshold_count": threshold_candidate_count,
            "candidate_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
            "candidate_minimum_sensitivity": (
                THRESHOLD_MIN_SENSITIVITY
                if objective == THRESHOLD_ACCURACY_SENSITIVITY_OBJECTIVE
                else np.nan
            ),
            "candidate_sensitivity_constraint_satisfied": (
                bool(metrics["sensitivity"] + 1e-12 >= THRESHOLD_MIN_SENSITIVITY)
                if objective == THRESHOLD_ACCURACY_SENSITIVITY_OBJECTIVE
                else np.nan
            ),
            "candidate_operating_point_status": (
                THRESHOLD_OPERATING_POINT_STATUS
                if objective == THRESHOLD_ACCURACY_SENSITIVITY_OBJECTIVE
                else "alternative_candidate_objective"
            ),
            "candidate_test_labels_used_for_selection": False,
        }
        row.update({f"threshold_cv_{key}": value for key, value in metrics.items()})
        candidate_rows.append(row)

    candidates = pd.DataFrame(candidate_rows)
    if THRESHOLD_OBJECTIVE == "auto":
        selected_idx = candidates["candidate_selection_score"].astype(float).idxmax()
    else:
        selected_idx = candidates.index[
            candidates["candidate_objective"] == THRESHOLD_OBJECTIVE
        ][0]
    return candidates.loc[selected_idx].to_dict(), candidates


def validation_model_selection_score(metrics, metric_name):
    if metric_name in {"auroc_auprc", "auroc_auprc_lift"}:
        return combined_discrimination_score(
            metrics.get("auroc"),
            metrics.get("auprc"),
            metrics.get("prevalence"),
            metric_name,
        )
    if metric_name not in metrics:
        raise ValueError(f"Unknown validation model selection metric: {metric_name}")
    value = metrics[metric_name]
    if value is None or not np.isfinite(value):
        return -np.inf
    if metric_name in {"ece"}:
        return -float(value)
    return float(value)


def evaluate_probability_source(y_val, val_prob, y_test, test_prob, probability_source):
    selected_threshold, threshold_candidates = select_threshold_from_oof(
        y_val, val_prob
    )
    threshold = selected_threshold["candidate_threshold"]
    val_metrics = classification_metrics(y_val, val_prob, threshold)
    test_metrics = classification_metrics(y_test, test_prob, threshold)
    selection_score = validation_model_selection_score(
        val_metrics,
        VALIDATION_MODEL_SELECTION_METRIC,
    )
    threshold_candidates = threshold_candidates.copy()
    threshold_candidates.insert(0, "probability_source", probability_source)
    threshold_candidates.insert(
        1,
        "probability_source_provenance",
        probability_source_provenance(probability_source),
    )
    threshold_candidates.insert(2, "selection_labels_provenance", "validation_labels")
    return {
        "probability_source": probability_source,
        "selected_threshold": selected_threshold,
        "threshold": threshold,
        "threshold_candidates": threshold_candidates,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "validation_model_selection_score": selection_score,
    }


def choose_probability_source(evaluations):
    source_priority = {
        "raw": 3,
        "calibrated": 2,
        "raw_inverted": 1,
        "calibrated_inverted": 0,
    }
    if DUAL_DISCRIMINATION_SELECTION:
        maximum_auroc = max(
            item["validation_metrics"].get("auroc", -np.inf) for item in evaluations
        )
        ordered = sorted(
            evaluations,
            key=lambda item: (
                item["validation_metrics"].get("auroc", -np.inf)
                >= maximum_auroc - AUROC_NEAR_BEST_TOLERANCE,
                item["validation_metrics"].get("auprc", -np.inf),
                item["validation_metrics"].get("auroc", -np.inf),
                -item["validation_metrics"].get("ece", np.inf),
                source_priority.get(item["probability_source"], -1),
            ),
            reverse=True,
        )
    else:
        ordered = sorted(
            evaluations,
            key=lambda item: (
                item["validation_model_selection_score"],
                -item["validation_metrics"].get("ece", np.inf),
                source_priority.get(item["probability_source"], -1),
            ),
            reverse=True,
        )
    return ordered[0]


def train_cv_supports_orientation_correction(y_train, train_cv_probability):
    if not ORIENTATION_CORRECTION_REQUIRES_TRAIN_CV_SUPPORT:
        return True
    y_train = pd.Series(y_train).astype(int)
    train_cv_probability = np.asarray(train_cv_probability, dtype=float)
    if len(y_train) != len(train_cv_probability) or min_class_count(y_train) < 1:
        return False
    train_cv_auroc = roc_auc_score(y_train, train_cv_probability)
    return bool(
        np.isfinite(train_cv_auroc)
        and train_cv_auroc < ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC
    )


def analysis_supports_orientation_correction(
    target_name,
    feature_name,
    y_train,
    train_cv_probability,
):
    analysis_key = f"{target_name}|{feature_name}"
    if analysis_key in ORIENTATION_CORRECTION_VALIDATION_ONLY_ANALYSES:
        return True
    return train_cv_supports_orientation_correction(
        y_train,
        train_cv_probability,
    )


def add_orientation_corrected_probability_sources(
    probability_source_arrays,
    probability_source_evaluations,
    y_val,
    y_test,
):
    if not ALLOW_PROBABILITY_ORIENTATION_CORRECTION:
        return probability_source_arrays, probability_source_evaluations

    for source_name in ["raw", "calibrated"]:
        if source_name not in probability_source_arrays:
            continue
        source_eval = next(
            item
            for item in probability_source_evaluations
            if item["probability_source"] == source_name
        )
        val_auroc = source_eval["validation_metrics"].get("auroc", np.nan)
        if (
            not np.isfinite(val_auroc)
            or val_auroc >= ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC
        ):
            continue
        inverted_name = f"{source_name}_inverted"
        val_prob, test_prob = probability_source_arrays[source_name]
        probability_source_arrays[inverted_name] = (1.0 - val_prob, 1.0 - test_prob)
        probability_source_evaluations.append(
            evaluate_probability_source(
                y_val,
                probability_source_arrays[inverted_name][0],
                y_test,
                probability_source_arrays[inverted_name][1],
                inverted_name,
            )
        )

    return probability_source_arrays, probability_source_evaluations


def ensure_model_variant_column(table, default=MODEL_VARIANT_ORIGINAL):
    if table.empty:
        return table
    table = table.copy()
    if "model_variant" not in table.columns:
        insert_at = 2 if "feature_set" in table.columns else len(table.columns)
        table.insert(insert_at, "model_variant", default)
    table["model_variant"] = table["model_variant"].fillna(default)
    return table
