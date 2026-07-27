"""Prespecified subgroup analyses."""

from . import reporting as _reporting  # noqa: F401 - preserve pipeline stage order.
from .utils import (
    average_precision_score,
    display,
    env_list,
    np,
    pd,
    roc_auc_score,
    zlib,
)
from .config import (
    BOOTSTRAP_CI_CONFIDENCE_LEVEL,
    BOOTSTRAP_CI_MIN_VALID_RESAMPLES,
    BOOTSTRAP_CI_N_RESAMPLES,
    BOOTSTRAP_CI_RANDOM_STATE,
    CALIBRATION_N_BINS,
    MIN_RELIABLE_TEST_CLASS_COUNT,
    MODEL_VARIANT_SIMPLIFIED,
    OUTPUT_DIR,
    RUN_BOOTSTRAP_CI,
    RUN_HOLDOUT_EVALUATION,
)
from .data import (
    HLA_FEATURES,
    df,
)
from .metrics import (
    calibration_regression_metrics,
    diagnostic_metric_values,
    expected_calibration_error,
    threshold_metrics,
)
from .finalization import holdout_predictions

SUBGROUP_SOURCE_COLUMNS = {
    "Age": "Age",
    "Sex": "Male Sex",
    "Diagnosis": "Diagnosis",
    "Graft source": "Graft source",
    "Conditioning regimen": "Myeloablating regimen MAC vs RIC",
    "Relatedness": "Related",
    "Prophylaxis": "Prophylaxis regimen",
}
HLA_FULL_MATCH_COLUMNS = [col for col in HLA_FEATURES if "antigen" in col.lower()]
SUBGROUP_DEFINITIONS = {
    "Age": "Age <65 versus >=65 years; missing age is reported separately.",
    "Sex": "Male Sex=Yes as Male and Male Sex=No as Female.",
    "Diagnosis": "Observed diagnosis category in the input dataset.",
    "Graft source": "Observed graft-source category in the input dataset.",
    "Conditioning regimen": "Myeloablating regimen MAC versus RIC.",
    "Relatedness": "Observed Related versus Unrelated category.",
    "HLA match": (
        "Fully matched when HLA-A/B/C/DR antigen-level matching indicators "
        "are all Yes; any explicit No is Not fully matched; otherwise "
        "Missing/indeterminate."
    ),
    "Prophylaxis": "Observed prophylaxis-regimen category; missing is separate.",
}
SUBGROUP_ORDER = list(SUBGROUP_DEFINITIONS)
SUBGROUP_METRIC_COLUMNS = [
    "n",
    "positive_n",
    "negative_n",
    "prevalence",
    "auroc",
    "auprc",
    "auprc_baseline",
    "auprc_lift",
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "balanced_accuracy",
    "youden_index",
    "f1",
    "net_benefit",
    "decision_curve_net_benefit_at_configured_threshold",
    "reported_probability_ece",
    "recalibrated_probability_ece",
    "ece",
    "calibration_intercept",
    "calibration_slope",
    "threshold",
    "tn",
    "fp",
    "fn",
    "tp",
]
SUBGROUP_CI_METRICS = (
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
SUBGROUP_BOOTSTRAP_TARGETS = set(
    env_list(
        "XGB_V2_SUBGROUP_BOOTSTRAP_TARGETS",
        "oGVHD_any,oGVHD_severe,cGVHD_any,cGVHD_severe",
    )
)
SUBGROUP_BOOTSTRAP_FEATURE_SETS = set(
    env_list(
        "XGB_V2_SUBGROUP_BOOTSTRAP_FEATURE_SETS",
        "Preoperative",
    )
)
SUBGROUP_BOOTSTRAP_MODEL_VARIANTS = set(
    env_list(
        "XGB_V2_SUBGROUP_BOOTSTRAP_MODEL_VARIANTS",
        MODEL_VARIANT_SIMPLIFIED,
    )
)
SUBGROUP_CI_RESULT_COLUMNS = [
    column
    for metric in SUBGROUP_CI_METRICS
    for column in (
        f"{metric}_ci_lower",
        f"{metric}_ci_upper",
        f"{metric}_bootstrap_valid_resamples",
        f"{metric}_bootstrap_status",
    )
]
SUBGROUP_BOOTSTRAP_METADATA_COLUMNS = [
    "bootstrap_requested_resamples",
    "bootstrap_confidence_level",
    "bootstrap_method",
    "bootstrap_sampling_unit",
    "bootstrap_seed",
]
SUBGROUP_RESULT_ID_COLUMNS = [
    "target",
    "feature_set",
    "model_variant",
    "landmark_day",
    "dataset_split",
    "subgroup_name",
    "subgroup_level",
    "subgroup_definition",
    "reported_probability_source",
    "estimability_status",
    "auroc_reliable",
]


def clean_subgroup_category(series):
    values = series.astype("string").str.strip()
    values = values.mask(values.eq(""), pd.NA)
    return values.fillna("Missing")


def make_subgroup_lookup(data):
    required = set(SUBGROUP_SOURCE_COLUMNS.values()) | set(HLA_FULL_MATCH_COLUMNS)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing subgroup columns: {missing}")
    if len(HLA_FULL_MATCH_COLUMNS) != 4:
        raise ValueError(
            "Expected four HLA antigen-level matching columns (A/B/C/DR), got "
            f"{HLA_FULL_MATCH_COLUMNS}."
        )

    source_index = pd.Series(np.arange(len(data), dtype=int), name="source_row_index")
    subgroup_frames = []

    age = pd.to_numeric(data[SUBGROUP_SOURCE_COLUMNS["Age"]], errors="coerce")
    age_level = pd.Series("Missing", index=data.index, dtype="string")
    age_level.loc[age.notna() & (age < 65)] = "<65"
    age_level.loc[age.notna() & (age >= 65)] = ">=65"

    sex_raw = clean_subgroup_category(data[SUBGROUP_SOURCE_COLUMNS["Sex"]])
    sex_level = sex_raw.replace({"Yes": "Male", "No": "Female"})

    levels = {
        "Age": age_level,
        "Sex": sex_level,
        "Diagnosis": clean_subgroup_category(
            data[SUBGROUP_SOURCE_COLUMNS["Diagnosis"]]
        ),
        "Graft source": clean_subgroup_category(
            data[SUBGROUP_SOURCE_COLUMNS["Graft source"]]
        ),
        "Conditioning regimen": clean_subgroup_category(
            data[SUBGROUP_SOURCE_COLUMNS["Conditioning regimen"]]
        ),
        "Relatedness": clean_subgroup_category(
            data[SUBGROUP_SOURCE_COLUMNS["Relatedness"]]
        ),
        "Prophylaxis": clean_subgroup_category(
            data[SUBGROUP_SOURCE_COLUMNS["Prophylaxis"]]
        ),
    }

    hla = data[HLA_FULL_MATCH_COLUMNS].apply(
        lambda column: column.astype("string").str.strip().str.casefold()
    )
    all_hla_yes = hla.eq("yes").all(axis=1)
    any_hla_no = hla.eq("no").any(axis=1)
    hla_level = pd.Series(
        "Missing/indeterminate",
        index=data.index,
        dtype="string",
    )
    hla_level.loc[any_hla_no] = "Not fully matched"
    hla_level.loc[all_hla_yes] = "Fully matched"
    levels["HLA match"] = hla_level

    for subgroup_name in SUBGROUP_ORDER:
        subgroup_frames.append(
            pd.DataFrame(
                {
                    "source_row_index": source_index,
                    "subgroup_name": subgroup_name,
                    "subgroup_level": levels[subgroup_name].astype("string"),
                    "subgroup_definition": SUBGROUP_DEFINITIONS[subgroup_name],
                }
            )
        )
    return pd.concat(subgroup_frames, ignore_index=True)


def subgroup_estimability_status(positive_n, negative_n):
    if positive_n == 0 or negative_n == 0:
        return "single_class_not_estimable"
    if min(positive_n, negative_n) < MIN_RELIABLE_TEST_CLASS_COUNT:
        return "exploratory_min_class_count_lt_reliability_threshold"
    return "reliable"


def subgroup_bootstrap_seed(
    target, feature_set, model_variant, subgroup_name, subgroup_level
):
    key = "|".join(
        map(
            str,
            [
                target,
                feature_set,
                model_variant,
                subgroup_name,
                subgroup_level,
            ],
        )
    ).encode("utf-8")
    return int((BOOTSTRAP_CI_RANDOM_STATE + zlib.crc32(key)) % (2**32))


def subgroup_bootstrap_in_scope(target, feature_set, model_variant):
    return (
        target in SUBGROUP_BOOTSTRAP_TARGETS
        and feature_set in SUBGROUP_BOOTSTRAP_FEATURE_SETS
        and model_variant in SUBGROUP_BOOTSTRAP_MODEL_VARIANTS
    )


def bootstrap_subgroup_metric_cis(
    y_true,
    y_prob,
    calibration_prob,
    threshold,
    seed,
    enabled,
):
    output = {}
    if not RUN_BOOTSTRAP_CI:
        status = "disabled"
    elif not enabled:
        status = "not_requested_outside_manuscript_scope"
    else:
        status = None

    if status is not None:
        for metric in SUBGROUP_CI_METRICS:
            output[f"{metric}_ci_lower"] = np.nan
            output[f"{metric}_ci_upper"] = np.nan
            output[f"{metric}_bootstrap_valid_resamples"] = 0
            output[f"{metric}_bootstrap_status"] = status
        output.update(
            {
                "bootstrap_requested_resamples": 0,
                "bootstrap_confidence_level": BOOTSTRAP_CI_CONFIDENCE_LEVEL,
                "bootstrap_method": "not_run",
                "bootstrap_sampling_unit": "held_out_transplantation_episode",
                "bootstrap_seed": int(seed),
            }
        )
        return output

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    calibration_prob = np.asarray(calibration_prob, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(int(seed))
    bootstrap_values = {
        metric: np.full(BOOTSTRAP_CI_N_RESAMPLES, np.nan, dtype=float)
        for metric in SUBGROUP_CI_METRICS
    }
    for resample_index in range(BOOTSTRAP_CI_N_RESAMPLES):
        sampled_indices = rng.integers(0, n, size=n)
        estimates = diagnostic_metric_values(
            y_true[sampled_indices],
            y_prob[sampled_indices],
            threshold,
            calibration_prob=calibration_prob[sampled_indices],
        )
        for metric in SUBGROUP_CI_METRICS:
            bootstrap_values[metric][resample_index] = estimates[metric]

    alpha = 1.0 - BOOTSTRAP_CI_CONFIDENCE_LEVEL
    quantiles = [alpha / 2.0, 1.0 - (alpha / 2.0)]
    for metric, values in bootstrap_values.items():
        finite_values = values[np.isfinite(values)]
        valid_n = int(len(finite_values))
        metric_status = (
            "ok"
            if valid_n >= BOOTSTRAP_CI_MIN_VALID_RESAMPLES
            else "insufficient_valid_resamples"
        )
        if metric_status == "ok":
            lower, upper = np.quantile(finite_values, quantiles)
        else:
            lower, upper = np.nan, np.nan
        output[f"{metric}_ci_lower"] = float(lower) if np.isfinite(lower) else np.nan
        output[f"{metric}_ci_upper"] = float(upper) if np.isfinite(upper) else np.nan
        output[f"{metric}_bootstrap_valid_resamples"] = valid_n
        output[f"{metric}_bootstrap_status"] = metric_status
    output.update(
        {
            "bootstrap_requested_resamples": BOOTSTRAP_CI_N_RESAMPLES,
            "bootstrap_confidence_level": BOOTSTRAP_CI_CONFIDENCE_LEVEL,
            "bootstrap_method": "ordinary_transplantation_episode_bootstrap_percentile_ci",
            "bootstrap_sampling_unit": "held_out_transplantation_episode",
            "bootstrap_seed": int(seed),
        }
    )
    return output


def calculate_test_subgroup_metrics(predictions, source_data):
    if predictions.empty:
        return pd.DataFrame(
            columns=SUBGROUP_RESULT_ID_COLUMNS + SUBGROUP_METRIC_COLUMNS
        )
    test_predictions = predictions.loc[predictions["dataset_split"].eq("test")].copy()
    if test_predictions.empty:
        return pd.DataFrame(
            columns=SUBGROUP_RESULT_ID_COLUMNS + SUBGROUP_METRIC_COLUMNS
        )

    subgroup_lookup = make_subgroup_lookup(source_data)
    merged = test_predictions.merge(
        subgroup_lookup,
        on="source_row_index",
        how="left",
        validate="many_to_many",
    )
    if merged["subgroup_name"].isna().any():
        raise ValueError("Some test predictions could not be linked to subgroup data.")

    group_columns = [
        "target",
        "feature_set",
        "model_variant",
        "landmark_day",
        "dataset_split",
        "subgroup_name",
        "subgroup_level",
        "subgroup_definition",
    ]
    rows = []
    for keys, group in merged.groupby(
        group_columns,
        sort=False,
        observed=True,
        dropna=False,
    ):
        y_true = pd.to_numeric(group["y_true"], errors="coerce").to_numpy(dtype=float)
        y_prob = pd.to_numeric(group["probability"], errors="coerce").to_numpy(
            dtype=float
        )
        calibration_prob = pd.to_numeric(
            group["calibration_probability"],
            errors="coerce",
        ).to_numpy(dtype=float)
        threshold_values = pd.to_numeric(group["threshold"], errors="coerce").dropna()
        valid = np.isfinite(y_true) & np.isfinite(y_prob) & np.isin(y_true, [0.0, 1.0])
        y_true = y_true[valid].astype(int)
        y_prob = y_prob[valid]
        calibration_prob = calibration_prob[valid]
        if not len(y_true) or threshold_values.empty:
            continue
        threshold = float(threshold_values.iloc[0])
        if not np.allclose(
            threshold_values.to_numpy(dtype=float),
            threshold,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "A subgroup model contains more than one selected threshold."
            )

        positive_n = int(y_true.sum())
        negative_n = int(len(y_true) - positive_n)
        status = subgroup_estimability_status(positive_n, negative_n)
        threshold_values_dict = threshold_metrics(y_true, y_prob, threshold)
        prevalence = float(np.mean(y_true))
        auprc = average_precision_score(y_true, y_prob) if positive_n else np.nan
        recalibrated_metrics = calibration_regression_metrics(y_true, calibration_prob)
        probability_sources = (
            group["reported_probability_source"].dropna().astype(str).unique()
        )
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "reported_probability_source": (
                    probability_sources[0] if len(probability_sources) == 1 else "mixed"
                ),
                "estimability_status": status,
                "auroc_reliable": status == "reliable",
                "n": int(len(y_true)),
                "positive_n": positive_n,
                "negative_n": negative_n,
                "prevalence": prevalence,
                "auroc": (
                    roc_auc_score(y_true, y_prob)
                    if positive_n and negative_n
                    else np.nan
                ),
                "auprc": auprc,
                "auprc_baseline": prevalence,
                "auprc_lift": (
                    float(auprc / prevalence)
                    if positive_n and prevalence > 0 and np.isfinite(auprc)
                    else np.nan
                ),
                "reported_probability_ece": expected_calibration_error(
                    y_true,
                    y_prob,
                    CALIBRATION_N_BINS,
                ),
                "recalibrated_probability_ece": (
                    expected_calibration_error(
                        y_true[np.isfinite(calibration_prob)],
                        calibration_prob[np.isfinite(calibration_prob)],
                        CALIBRATION_N_BINS,
                    )
                    if np.isfinite(calibration_prob).any()
                    else np.nan
                ),
                "ece": expected_calibration_error(
                    y_true,
                    calibration_prob,
                    CALIBRATION_N_BINS,
                ),
                **recalibrated_metrics,
            }
        )
        row.update(threshold_values_dict)
        subgroup_seed = subgroup_bootstrap_seed(
            row["target"],
            row["feature_set"],
            row["model_variant"],
            row["subgroup_name"],
            row["subgroup_level"],
        )
        row.update(
            bootstrap_subgroup_metric_cis(
                y_true,
                y_prob,
                calibration_prob,
                threshold,
                subgroup_seed,
                enabled=subgroup_bootstrap_in_scope(
                    row["target"],
                    row["feature_set"],
                    row["model_variant"],
                ),
            )
        )
        rows.append(row)

    result = pd.DataFrame(
        rows,
        columns=(
            SUBGROUP_RESULT_ID_COLUMNS
            + SUBGROUP_METRIC_COLUMNS
            + SUBGROUP_CI_RESULT_COLUMNS
            + SUBGROUP_BOOTSTRAP_METADATA_COLUMNS
        ),
    )
    if result.empty:
        return result
    result["subgroup_order"] = result["subgroup_name"].map(
        {name: index for index, name in enumerate(SUBGROUP_ORDER)}
    )
    result = (
        result.sort_values(
            [
                "target",
                "model_variant",
                "landmark_day",
                "subgroup_order",
                "subgroup_level",
            ],
            kind="mergesort",
        )
        .drop(columns="subgroup_order")
        .reset_index(drop=True)
    )
    return result


holdout_subgroup_metrics = calculate_test_subgroup_metrics(
    holdout_predictions if RUN_HOLDOUT_EVALUATION else pd.DataFrame(),
    df,
)
holdout_subgroup_metric_summary = (
    holdout_subgroup_metrics.melt(
        id_vars=SUBGROUP_RESULT_ID_COLUMNS,
        value_vars=SUBGROUP_METRIC_COLUMNS,
        var_name="metric",
        value_name="value",
    )
    if not holdout_subgroup_metrics.empty
    else pd.DataFrame(columns=SUBGROUP_RESULT_ID_COLUMNS + ["metric", "value"])
)
if not holdout_subgroup_metric_summary.empty:
    for bound in [
        "ci_lower",
        "ci_upper",
        "bootstrap_valid_resamples",
        "bootstrap_status",
    ]:
        bound_columns = [f"{metric}_{bound}" for metric in SUBGROUP_CI_METRICS]
        bound_summary = holdout_subgroup_metrics.melt(
            id_vars=SUBGROUP_RESULT_ID_COLUMNS,
            value_vars=bound_columns,
            var_name="metric",
            value_name=bound,
        )
        bound_summary["metric"] = bound_summary["metric"].str.removesuffix(f"_{bound}")
        holdout_subgroup_metric_summary = holdout_subgroup_metric_summary.merge(
            bound_summary,
            on=SUBGROUP_RESULT_ID_COLUMNS + ["metric"],
            how="left",
        )
holdout_subgroup_definitions = pd.DataFrame(
    {
        "subgroup_name": SUBGROUP_ORDER,
        "definition": [SUBGROUP_DEFINITIONS[name] for name in SUBGROUP_ORDER],
    }
)

holdout_subgroup_metrics.to_csv(
    OUTPUT_DIR / "holdout_subgroup_metrics.csv",
    index=False,
)
holdout_subgroup_metric_summary.to_csv(
    OUTPUT_DIR / "holdout_subgroup_metric_summary.csv",
    index=False,
)
holdout_subgroup_definitions.to_csv(
    OUTPUT_DIR / "holdout_subgroup_definitions.csv",
    index=False,
)
print(
    "Saved test subgroup results: "
    f"{len(holdout_subgroup_metrics):,} strata rows and "
    f"{len(holdout_subgroup_metric_summary):,} long-format metric rows."
)
display(holdout_subgroup_metrics)
