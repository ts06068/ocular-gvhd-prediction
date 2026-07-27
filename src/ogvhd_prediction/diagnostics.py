"""Endpoint diagnostic plots."""

from .utils import (
    Line2D,
    OrderedDict,
    Path,
    average_precision_score,
    cloudpickle,
    gridspec,
    np,
    os,
    pd,
    plt,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    shap,
    show_plot,
    slugify,
)
from .config import (
    CALIBRATION_N_BINS,
    DCA_PROBABILITY_SOURCE,
    DCA_THRESHOLD_MAX,
    DCA_THRESHOLD_MIN,
    DCA_Y_MAX,
    DCA_Y_MIN,
    MODEL_VARIANT_ORDER,
    MODEL_VARIANT_ORIGINAL,
    MODEL_VARIANT_SIMPLIFIED,
    OUTPUT_DIR,
    PLOT_DIR,
    RUN_FINAL_MODELS,
    RUN_HOLDOUT_EVALUATION,
    TARGETS_TO_RUN,
)
from .data import (
    ALL_FEATURE_SETS,
    TARGET_DEFINITIONS,
)
from .targets import (
    make_target,
    target_landmark_day_for_policy,
)
from .metrics import (
    ensure_model_variant_column,
    expected_calibration_error,
    file_sha256,
    net_benefit_at_threshold,
)
from .preprocessing import (
    hla_locus_display_name,
    is_hla_matching_column,
)
from .explainability import xgb_shap_contributions
from .finalization import (
    holdout_model_summary,
    holdout_predictions,
)

FIG_DPI = int(os.getenv("XGB_V2_FIG_DPI", os.getenv("XGB_FIG_DPI", "600")))
PLOT_CALIBRATION_N_BINS = int(
    os.getenv(
        "XGB_V2_PLOT_CALIBRATION_N_BINS", os.getenv("XGB_PLOT_CALIBRATION_N_BINS", "5")
    )
)
PLOT_FONT_FAMILY = os.getenv("XGB_V2_PLOT_FONT_FAMILY", "Calibri")
PANEL_FIG_DPI = int(os.getenv("XGB_V2_PANEL_FIG_DPI", "300"))
PANEL_SHAP_MAX_DISPLAY = int(os.getenv("XGB_V2_PANEL_SHAP_MAX_DISPLAY", "8"))
PANEL_TITLE_FONTSIZE = int(os.getenv("XGB_V2_PANEL_TITLE_FONTSIZE", "22"))
PANEL_TITLE_PAD = float(os.getenv("XGB_V2_PANEL_TITLE_PAD", "4"))
PANEL_LEGEND_FONTSIZE = int(os.getenv("XGB_V2_PANEL_LEGEND_FONTSIZE", "18"))
PANEL_AXIS_LABEL_FONTSIZE = int(os.getenv("XGB_V2_PANEL_AXIS_LABEL_FONTSIZE", "15"))
PANEL_TICK_FONTSIZE = int(os.getenv("XGB_V2_PANEL_TICK_FONTSIZE", "13"))
PANEL_SHAP_AXIS_LABEL_FONTSIZE = int(
    os.getenv("XGB_V2_PANEL_SHAP_AXIS_LABEL_FONTSIZE", "18")
)
PANEL_SHAP_TICK_FONTSIZE = int(os.getenv("XGB_V2_PANEL_SHAP_TICK_FONTSIZE", "16"))
PANEL_SHAP_FEATURE_FONTSIZE = int(os.getenv("XGB_V2_PANEL_SHAP_FEATURE_FONTSIZE", "16"))
PANEL_SHAP_LABEL_WRAP = int(os.getenv("XGB_V2_PANEL_SHAP_LABEL_WRAP", "26"))
PANEL_SHAP_CMAP = shap.plots.colors.red_blue


TARGET_DISPLAY_NAMES = {
    "aGVHD_any": "any-stage aGVHD",
    "aGVHD_severe": "severe aGVHD",
    "oGVHD_any": "any-stage oGVHD",
    "oGVHD_severe": "severe oGVHD",
    "cGVHD_any": "any-stage cGVHD",
    "cGVHD_severe": "severe cGVHD",
}

FEATURE_SET_DISPLAY_NAMES = {
    "Preoperative": "Pretransplant",
    "Post transplant D+30": "D+30 Post-transplant",
    "Post transplant D+60": "D+60 Post-transplant",
    "Post transplant D+90": "D+90 Post-transplant",
}

FEATURE_SET_COLORS = {
    "Preoperative": "#1F4E79",
    "Post transplant D+30": "#8C1D18",
    "Post transplant D+60": "#1F6F3E",
    "Post transplant D+90": "#6A3D9A",
}
MODEL_VARIANT_LINESTYLES = {
    MODEL_VARIANT_ORIGINAL: ":",
    MODEL_VARIANT_SIMPLIFIED: "-",
}
MODEL_VARIANT_MARKERS = {
    MODEL_VARIANT_ORIGINAL: "o",
    MODEL_VARIANT_SIMPLIFIED: "o",
}

GRAY_TREAT = "#6E6E6E"
PANEL_FEATURE_SETS = OrderedDict(
    {
        "Pretransplant": "Preoperative",
        "D+30 Post-transplant": "Post transplant D+30",
        "D+60 Post-transplant": "Post transplant D+60",
        "D+90 Post-transplant": "Post transplant D+90",
    }
)

FEATURE_NAME_OVERRIDES = {
    "Age": "Age",
    "Male Sex": "Male sex",
    "CD34": "CD34 dose",
    "Donor T cell number": "Donor T-cell number",
    "ABO matching": "ABO matching",
    "Related": "Related donor",
    "Graft source": "Graft source",
    "Diagnosis": "Diagnosis",
    "Myeloablating regimen MAC vs RIC": "Conditioning intensity",
    "Prophylaxis regimen": "Prophylaxis regimen",
}


def target_display_name(target):
    return TARGET_DISPLAY_NAMES.get(target, target.replace("_", " "))


def feature_set_display_name(feature_set):
    return FEATURE_SET_DISPLAY_NAMES.get(feature_set, feature_set)


def model_variant_display_name(model_variant):
    if str(model_variant) == MODEL_VARIANT_SIMPLIFIED:
        return "Simplified"
    return ""


def curve_label(panel_title, model_variant, metric_name, score):
    suffix = model_variant_display_name(model_variant)
    title = f"{panel_title} {suffix}".strip()
    return f"{title}\n{metric_name}={score:.3f}"


def endpoint_file_slug(target):
    return slugify(target_display_name(target)).lower()


def make_faceted_square_figure(suptitle, figsize=(14.0, 13.8)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    fig.subplots_adjust(
        top=0.91, bottom=0.20, left=0.07, right=0.98, hspace=0.22, wspace=0.14
    )
    fig.suptitle(suptitle, y=0.965, fontsize=20)
    for ax in axes:
        ax.set_box_aspect(1)
    return fig, axes


def make_faceted_square_figure_with_legend_row(
    title,
    panel_size=6,
    legend_row_height=1.35,
    left=0.07,
    right=0.98,
    top=0.92,
    bottom=0.06,
    wspace=0,
    hspace=0.4,
):
    fig_width = panel_size * 2 + 2.5
    fig_height = panel_size * 2 + 2.5 + legend_row_height
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1, 1, legend_row_height / panel_size],
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=wspace,
        hspace=hspace,
    )
    axes = np.array(
        [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
        ],
        dtype=object,
    )
    for ax in axes:
        ax.set_box_aspect(1)
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis("off")
    fig.suptitle(title, y=0.965, fontsize=20)
    return fig, axes, legend_ax


def style_axis(ax):
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=5,
        width=1.1,
        color="#222222",
        labelsize=14,
    )
    ax.grid(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#222222")
        ax.spines[spine].set_linewidth(1.0)


def add_combined_legend_below(fig, axes, ncol=4, y=0.05):
    legend_items = OrderedDict()
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in legend_items:
                legend_items[label] = handle
    if not legend_items:
        return
    fig.legend(
        legend_items.values(),
        legend_items.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.3,
        borderaxespad=0.0,
    )


def add_ordered_legend_on_axis(legend_ax, axes, desired_rows, fontsize=15):
    handle_map = {}
    for ax in np.ravel(axes):
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in handle_map:
                handle_map[label] = handle

    nrows = len(desired_rows)
    ncols = max(len(row) for row in desired_rows)
    padded_rows = [row + [""] * (ncols - len(row)) for row in desired_rows]
    blank = Line2D([], [], linestyle="none", linewidth=0, alpha=0)
    ordered_handles = []
    ordered_labels = []

    for col in range(ncols):
        for row in range(nrows):
            label = padded_rows[row][col]
            if label == "":
                ordered_handles.append(blank)
                ordered_labels.append("")
            else:
                ordered_handles.append(handle_map.get(label, blank))
                ordered_labels.append(label if label in handle_map else "")

    legend = legend_ax.legend(
        ordered_handles,
        ordered_labels,
        loc="center",
        ncol=ncols,
        frameon=False,
        fontsize=fontsize,
        handlelength=3.0,
        columnspacing=1.6,
        handletextpad=0.8,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        if text.get_text() == "":
            text.set_alpha(0)
    return legend


def save_plot_figure(fig, filename):
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")


def prettify_base_feature_name(name):
    name = str(name).replace("numeric__", "").replace("categorical__", "")
    if name in FEATURE_NAME_OVERRIDES:
        return FEATURE_NAME_OVERRIDES[name]
    if is_hla_matching_column(name):
        return f"HLA-{hla_locus_display_name(name)} Antigen Matched: Yes"
    replacements = {
        "_baseline": " (baseline)",
        "+30": " (D+30)",
        "+60": " (D+60)",
        "+90": " (D+90)",
        "_": " ",
    }
    pretty = name
    for old, new in replacements.items():
        pretty = pretty.replace(old, new)
    pretty = " ".join(pretty.split())
    return pretty[:1].upper() + pretty[1:]


def prettify_transformed_feature_name(name, categorical_columns=None):
    raw = str(name)
    if raw.startswith("numeric__"):
        return prettify_base_feature_name(raw.removeprefix("numeric__"))
    if raw.startswith("categorical__"):
        value = raw.removeprefix("categorical__")
        categorical_columns = sorted(categorical_columns or [], key=len, reverse=True)
        for col in categorical_columns:
            prefix = f"{col}_"
            if value.startswith(prefix):
                category = value[len(prefix) :]
                if is_hla_matching_column(col):
                    return (
                        f"HLA-{hla_locus_display_name(col)} Antigen Matched: {category}"
                    )
                return f"{prettify_base_feature_name(col)}: {category}"
        if "_" in value:
            col, category = value.split("_", 1)
            return f"{prettify_base_feature_name(col)}: {category}"
        return prettify_base_feature_name(value)
    return prettify_base_feature_name(raw)


def ordered_feature_sets_present(table):
    present = set(table["feature_set"].dropna())
    return [
        feature_set
        for feature_set in PANEL_FEATURE_SETS.values()
        if feature_set in present
    ]


def ordered_model_variants_present(table):
    if "model_variant" not in table.columns:
        return [MODEL_VARIANT_ORIGINAL]
    present = [str(value) for value in table["model_variant"].dropna().unique()]
    ordered = [variant for variant in MODEL_VARIANT_ORDER if variant in present]
    ordered.extend([variant for variant in present if variant not in ordered])
    return ordered


def test_prediction_group(target, feature_set, model_variant=None):
    if holdout_predictions.empty:
        return pd.DataFrame()
    mask = (
        (holdout_predictions["dataset_split"] == "test")
        & (holdout_predictions["target"] == target)
        & (holdout_predictions["feature_set"] == feature_set)
    )
    if model_variant is not None and "model_variant" in holdout_predictions.columns:
        mask &= holdout_predictions["model_variant"] == model_variant
    return holdout_predictions.loc[mask].copy()


def curve_probability(group):
    if "probability" in group.columns:
        return pd.to_numeric(group["probability"], errors="coerce").to_numpy(
            dtype=float
        )
    return pd.to_numeric(group["raw_probability"], errors="coerce").to_numpy(
        dtype=float
    )


def decision_curve_probability(group):
    """Return development-fitted recalibrated probabilities for test-set DCA."""
    required_columns = {
        "target",
        "feature_set",
        "model_variant",
        "dataset_split",
        "calibration_probability",
    }
    missing_columns = sorted(required_columns - set(group.columns))
    if missing_columns:
        raise KeyError(
            "Decision-curve input lacks recalibrated-probability provenance columns: "
            f"{missing_columns}."
        )
    if group.empty:
        raise ValueError("Decision-curve input group is empty.")

    identity = {}
    for column in ("target", "feature_set", "model_variant", "dataset_split"):
        if group[column].isna().any():
            raise ValueError(
                f"Decision-curve input has missing model/split provenance in {column}."
            )
        values = group[column].dropna().astype(str).unique()
        if len(values) != 1:
            raise ValueError(
                "Decision-curve input must represent exactly one model and split; "
                f"{column} has values {values.tolist()}."
            )
        identity[column] = values[0]
    if identity["dataset_split"] != "test":
        raise ValueError(
            "Decision-curve probabilities must come from the held-out test split, got "
            f"{identity['dataset_split']!r}."
        )

    probabilities = pd.to_numeric(
        group["calibration_probability"],
        errors="coerce",
    ).to_numpy(dtype=float)
    invalid = ~np.isfinite(probabilities)
    if invalid.any():
        raise ValueError(
            "Decision-curve calibration_probability contains "
            f"{int(invalid.sum())} missing or non-finite value(s) for "
            f"{identity['target']}/{identity['feature_set']}/{identity['model_variant']}."
        )
    outside_unit_interval = (probabilities < 0.0) | (probabilities > 1.0)
    if outside_unit_interval.any():
        raise ValueError(
            "Decision-curve calibration_probability must be within [0, 1] for "
            f"{identity['target']}/{identity['feature_set']}/{identity['model_variant']}."
        )
    return probabilities


def save_endpoint_curve_grid(target, curve_kind):
    target_predictions = holdout_predictions.loc[
        (holdout_predictions["dataset_split"] == "test")
        & (holdout_predictions["target"] == target)
    ]
    if target_predictions.empty:
        return

    present_feature_sets = set(ordered_feature_sets_present(target_predictions))
    curve_title = {
        "roc": "Receiver-operating characteristic curve",
        "pr": "Precision-recall curve",
    }[curve_kind]
    fig, axes = make_faceted_square_figure(
        f"{curve_title} of models predicting {target_display_name(target)}"
    )

    for ax, (panel_title, feature_set) in zip(axes, PANEL_FEATURE_SETS.items()):
        if feature_set not in present_feature_sets:
            ax.axis("off")
            continue
        feature_predictions = test_prediction_group(target, feature_set)
        if feature_predictions.empty:
            ax.axis("off")
            continue
        color = FEATURE_SET_COLORS.get(feature_set, "#333333")

        plotted = False
        for model_variant in ordered_model_variants_present(feature_predictions):
            group = test_prediction_group(target, feature_set, model_variant)
            if group.empty or group["y_true"].nunique() < 2:
                continue
            y_true = group["y_true"].astype(int).to_numpy()
            y_prob = curve_probability(group)
            linestyle = MODEL_VARIANT_LINESTYLES.get(model_variant, "-")
            if curve_kind == "roc":
                x, y, _ = roc_curve(y_true, y_prob)
                score = roc_auc_score(y_true, y_prob)
                ax.plot(
                    x,
                    y,
                    linestyle=linestyle,
                    linewidth=2,
                    color=color,
                    label=curve_label(panel_title, model_variant, "AUROC", score),
                )
                plotted = True
            elif curve_kind == "pr":
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                score = average_precision_score(y_true, y_prob)
                ax.plot(
                    recall,
                    precision,
                    linestyle=linestyle,
                    linewidth=2,
                    color=color,
                    label=curve_label(panel_title, model_variant, "AUPRC", score),
                )
                plotted = True
            else:
                raise ValueError(curve_kind)

        if not plotted:
            ax.axis("off")
            continue
        if curve_kind == "roc":
            ax.plot([0, 1], [0, 1], linestyle="--", color="#B0B0B0", linewidth=1.4)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
        elif curve_kind == "pr":
            baseline = float(np.mean(feature_predictions["y_true"].astype(int)))
            ax.axhline(baseline, linestyle="--", color="#B0B0B0", linewidth=1.4)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")

        ax.set_title(panel_title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        style_axis(ax)

    add_combined_legend_below(fig, axes, ncol=4, y=0.05)
    save_plot_figure(
        fig,
        f"holdout_{curve_kind}_{endpoint_file_slug(target)}.png",
    )
    show_plot()


def binomial_wilson_ci(successes, n, confidence=0.95):
    if n <= 0:
        return np.nan, np.nan
    z = 1.959963984540054 if np.isclose(confidence, 0.95) else 1.959963984540054
    p = successes / n
    denominator = 1.0 + (z**2 / n)
    center = (p + (z**2 / (2 * n))) / denominator
    margin = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denominator
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def calibration_points(y_true, y_prob, n_bins=5, strategy="quantile"):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    valid = np.isfinite(y_prob) & np.isfinite(y_true)
    y_true = y_true[valid]
    y_prob = np.clip(y_prob[valid], 0.0, 1.0)
    if len(y_true) == 0:
        return pd.DataFrame(
            columns=[
                "mean_predicted_probability",
                "observed_event_rate",
                "observed_event_rate_lower",
                "observed_event_rate_upper",
                "event_n",
                "n",
                "ci_level",
                "ci_method",
            ]
        )

    if strategy == "quantile":
        bin_edges = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(bin_edges) <= 1:
            bin_edges = np.array([0.0, 1.0])
        else:
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
    elif strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(strategy)

    rows = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if right == 1.0:
            in_bin = (y_prob >= left) & (y_prob <= right)
        else:
            in_bin = (y_prob >= left) & (y_prob < right)
        if not np.any(in_bin):
            continue
        n = int(np.sum(in_bin))
        event_n = int(np.sum(y_true[in_bin]))
        observed = float(event_n / n) if n else np.nan
        lower, upper = binomial_wilson_ci(event_n, n, confidence=0.95)
        rows.append(
            {
                "mean_predicted_probability": float(np.mean(y_prob[in_bin])),
                "observed_event_rate": observed,
                "observed_event_rate_lower": lower,
                "observed_event_rate_upper": upper,
                "event_n": event_n,
                "n": n,
                "ci_level": 0.95,
                "ci_method": "Wilson",
            }
        )
    return pd.DataFrame(rows)


def save_endpoint_calibration_grid(target, calibration_table):
    target_cal = calibration_table.loc[calibration_table["target"] == target]
    if target_cal.empty:
        return
    present_feature_sets = set(ordered_feature_sets_present(target_cal))
    fig, axes = make_faceted_square_figure(
        f"Calibration of models predicting {target_display_name(target)}"
    )

    for ax, (panel_title, feature_set) in zip(axes, PANEL_FEATURE_SETS.items()):
        if feature_set not in present_feature_sets:
            ax.axis("off")
            continue
        feature_group = target_cal.loc[
            (target_cal["feature_set"] == feature_set)
            & (target_cal["model_variant"] == MODEL_VARIANT_SIMPLIFIED)
        ]
        if feature_group.empty:
            ax.axis("off")
            continue
        color = FEATURE_SET_COLORS.get(feature_set, "#333333")
        plotted = False
        for model_variant in [MODEL_VARIANT_SIMPLIFIED]:
            group = feature_group.loc[feature_group["model_variant"] == model_variant]
            if group.empty:
                continue
            y = group["observed_event_rate"].astype(float)
            yerr = np.vstack(
                [
                    y - group["observed_event_rate_lower"].astype(float),
                    group["observed_event_rate_upper"].astype(float) - y,
                ]
            )
            yerr = np.nan_to_num(np.clip(yerr, 0.0, None), nan=0.0)
            ece_values = pd.to_numeric(
                group.get("calibration_ece", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            ece = float(ece_values.iloc[0]) if len(ece_values) else np.nan
            label_title = (
                f"{panel_title} {model_variant_display_name(model_variant)}".strip()
            )
            label = f"{label_title}\nECE={ece:.3f}" if np.isfinite(ece) else label_title
            ax.errorbar(
                group["mean_predicted_probability"].astype(float),
                y,
                yerr=yerr,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                markersize=5,
                linestyle="none",
                label=label,
            )
            plotted = True
        if not plotted:
            ax.axis("off")
            continue
        ax.plot([0, 1], [0, 1], linestyle="--", color="#B0B0B0", linewidth=1.4)
        ax.set_title(panel_title)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        style_axis(ax)

    add_combined_legend_below(fig, axes, ncol=4, y=0.05)
    save_plot_figure(
        fig,
        f"holdout_calibration_{endpoint_file_slug(target)}.png",
    )
    show_plot()


def decision_curve_points(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    prevalence = float(np.mean(y_true))
    rows = []
    for threshold in thresholds:
        model_nb = net_benefit_at_threshold(y_true, y_prob, threshold)
        all_nb = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "model_net_benefit": float(model_nb),
                "treat_all_net_benefit": float(all_nb),
                "treat_none_net_benefit": 0.0,
                "model_standardized_net_benefit": float(model_nb / prevalence)
                if prevalence
                else np.nan,
                "treat_all_standardized_net_benefit": float(all_nb / prevalence)
                if prevalence
                else np.nan,
                "treat_none_standardized_net_benefit": 0.0,
                "prevalence": prevalence,
            }
        )
    return pd.DataFrame(rows)


def save_endpoint_decision_curve_grid(target):
    target_predictions = holdout_predictions.loc[
        (holdout_predictions["dataset_split"] == "test")
        & (holdout_predictions["target"] == target)
    ]
    if target_predictions.empty:
        return
    present_feature_sets = set(ordered_feature_sets_present(target_predictions))
    thresholds = np.linspace(DCA_THRESHOLD_MIN, DCA_THRESHOLD_MAX, 80)
    fig, axes, legend_ax = make_faceted_square_figure_with_legend_row(
        f"Decision curve analysis of models predicting {target_display_name(target)}"
    )

    dca_rows = []
    panel_dca = []
    y_limit_values = [np.array([0.0])]
    for panel_title, feature_set in PANEL_FEATURE_SETS.items():
        if feature_set not in present_feature_sets:
            panel_dca.append((panel_title, feature_set, pd.DataFrame()))
            continue
        feature_predictions = test_prediction_group(target, feature_set)
        if feature_predictions.empty:
            panel_dca.append((panel_title, feature_set, pd.DataFrame()))
            continue
        dca_frames = []
        for model_variant in ordered_model_variants_present(feature_predictions):
            group = test_prediction_group(target, feature_set, model_variant)
            if group.empty:
                continue
            y_true = group["y_true"].astype(int).to_numpy()
            y_prob = decision_curve_probability(group)
            dca = decision_curve_points(y_true, y_prob, thresholds)
            dca.insert(0, "target", target)
            dca.insert(1, "feature_set", feature_set)
            dca.insert(2, "model_variant", model_variant)
            dca.insert(3, "probability_source", DCA_PROBABILITY_SOURCE)
            dca_frames.append(dca)
        if not dca_frames:
            panel_dca.append((panel_title, feature_set, pd.DataFrame()))
            continue
        dca = pd.concat(dca_frames, ignore_index=True)
        panel_dca.append((panel_title, feature_set, dca))
        y_limit_values.extend(
            [
                dca["model_net_benefit"].to_numpy(dtype=float),
                np.clip(
                    dca["treat_all_net_benefit"].to_numpy(dtype=float), DCA_Y_MIN, None
                ),
                dca["treat_none_net_benefit"].to_numpy(dtype=float),
            ]
        )

    y_values = np.concatenate(
        [values[np.isfinite(values)] for values in y_limit_values if len(values)]
    )
    if DCA_Y_MAX is not None:
        y_upper = DCA_Y_MAX
    elif len(y_values):
        y_upper = max(0.03, float(np.nanpercentile(y_values, 98)) * 1.15 + 0.005)
    else:
        y_upper = 0.05
    if y_upper <= DCA_Y_MIN:
        y_upper = DCA_Y_MIN + 0.05

    for ax, (panel_title, feature_set, dca) in zip(axes, panel_dca):
        if feature_set not in present_feature_sets:
            ax.axis("off")
            continue
        if dca.empty:
            ax.axis("off")
            continue
        dca_rows.extend(dca.to_dict("records"))

        color = FEATURE_SET_COLORS.get(feature_set, "#333333")
        for model_variant in ordered_model_variants_present(dca):
            variant_dca = dca.loc[dca["model_variant"] == model_variant]
            if variant_dca.empty:
                continue
            ax.plot(
                variant_dca["threshold"],
                variant_dca["model_net_benefit"],
                color=color,
                linestyle=MODEL_VARIANT_LINESTYLES.get(model_variant, "-"),
                linewidth=2,
                label=f"{panel_title} {model_variant_display_name(model_variant)}".strip(),
            )
        reference_dca = dca.loc[
            dca["model_variant"] == ordered_model_variants_present(dca)[0]
        ]
        ax.plot(
            reference_dca["threshold"],
            reference_dca["treat_all_net_benefit"],
            color=GRAY_TREAT,
            linestyle="--",
            linewidth=2,
            label="Treat all",
        )
        ax.plot(
            reference_dca["threshold"],
            reference_dca["treat_none_net_benefit"],
            color=GRAY_TREAT,
            linestyle="-.",
            linewidth=2,
            label="Treat none",
        )
        ax.set_ylim(DCA_Y_MIN, y_upper)
        ax.set_xlim(thresholds.min(), thresholds.max())
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title(panel_title)
        style_axis(ax)

    desired_dca_rows = [
        [f"{panel_title} Simplified" for panel_title in PANEL_FEATURE_SETS.keys()],
        list(PANEL_FEATURE_SETS.keys()),
        ["Treat all", "Treat none", "", ""],
    ]
    add_ordered_legend_on_axis(legend_ax, axes, desired_dca_rows, fontsize=15)
    save_plot_figure(
        fig,
        f"holdout_decision_curve_{endpoint_file_slug(target)}.png",
    )
    show_plot()
    if dca_rows:
        dca_table = pd.DataFrame(dca_rows)
        dca_table.to_csv(
            PLOT_DIR / f"holdout_decision_curve_{endpoint_file_slug(target)}.csv",
            index=False,
        )


def feature_set_item(feature_set_name):
    for item in ALL_FEATURE_SETS:
        if item["feature_set"] == feature_set_name:
            return item
    raise KeyError(feature_set_name)


def model_bundle_for(target, feature_set, model_variant=MODEL_VARIANT_ORIGINAL):
    if holdout_model_summary.empty:
        return None
    table = ensure_model_variant_column(holdout_model_summary)
    rows = table.loc[
        (table["target"] == target)
        & (table["feature_set"] == feature_set)
        & (table["model_variant"] == model_variant)
    ]
    if rows.empty:
        return None
    model_path = Path(rows.iloc[0]["model_path"])
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        bundle = cloudpickle.load(f)
    if bundle.get("artifact_type") == "temporal_landmark_ensemble_wrapper":
        # Explain the landmark member, not its orchestration-only wrapper.
        member_artifacts = bundle.get("ordered_member_artifacts", [])
        current_member = next(
            (
                item
                for item in reversed(member_artifacts)
                if str(item.get("feature_set")) == str(feature_set)
            ),
            None,
        )
        if current_member is None:
            return None
        base_model_path = model_path.parent / current_member["base_model_relative_path"]
        if not base_model_path.is_file():
            return None
        if file_sha256(base_model_path) != current_member.get("base_model_sha256"):
            raise RuntimeError(
                f"Temporal member artifact hash mismatch: {base_model_path}"
            )
        with open(base_model_path, "rb") as f:
            return cloudpickle.load(f)
    return bundle


def shap_matrix_for_pipeline(pipeline, X):
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
    pretty_names = [
        prettify_transformed_feature_name(name, preprocess_step.categorical_columns_)
        for name in selected_feature_names
    ]
    return X_selected, selected_feature_names, pretty_names


def shap_values_for_pipeline(pipeline, X):
    X_selected, feature_names, pretty_names = shap_matrix_for_pipeline(pipeline, X)
    model = pipeline.named_steps["model"]
    shap_values = xgb_shap_contributions(model, X_selected)
    return (
        pd.DataFrame(shap_values, columns=feature_names),
        pd.DataFrame(X_selected, columns=feature_names),
        dict(zip(feature_names, pretty_names)),
    )


def shap_frames_for_fitted_model(model, X):
    if not hasattr(model, "models_"):
        return shap_values_for_pipeline(model, X)

    components = [shap_frames_for_fitted_model(member, X) for member in model.models_]
    shap_frames = [component[0] for component in components]
    value_frames = [component[1] for component in components]
    pretty_name_map = {}
    for _, _, local_pretty_names in components:
        pretty_name_map.update(local_pretty_names)

    all_features = sorted(
        {feature for frame in shap_frames for feature in frame.columns}
    )
    shap_stack = np.stack(
        [
            frame.reindex(columns=all_features, fill_value=0.0).to_numpy(dtype=float)
            for frame in shap_frames
        ]
    )
    value_stack = np.stack(
        [
            frame.reindex(columns=all_features).to_numpy(dtype=float)
            for frame in value_frames
        ]
    )
    mean_shap_values = np.mean(shap_stack, axis=0)
    with np.errstate(invalid="ignore"):
        mean_feature_values = np.nanmean(value_stack, axis=0)
    mean_feature_values = np.nan_to_num(mean_feature_values, nan=0.0)
    return (
        pd.DataFrame(mean_shap_values, columns=all_features),
        pd.DataFrame(mean_feature_values, columns=all_features),
        pretty_name_map,
    )


def shap_values_for_model(model, X):
    shap_frame, value_frame, pretty_name_map = shap_frames_for_fitted_model(model, X)
    all_features = shap_frame.columns.tolist()
    pretty_names = [
        pretty_name_map.get(feature, prettify_transformed_feature_name(feature))
        for feature in all_features
    ]
    return (
        shap_frame.to_numpy(dtype=float),
        value_frame.to_numpy(dtype=float),
        pretty_names,
    )


def save_shap_feature_importance_plot(
    target, feature_set, model_variant=MODEL_VARIANT_ORIGINAL
):
    bundle = model_bundle_for(target, feature_set, model_variant)
    if bundle is None:
        return
    selected_feature_source = bundle.get("selected_feature_source", feature_set)
    feature_item = feature_set_item(selected_feature_source)
    spec = TARGET_DEFINITIONS[target]
    keep_mask, _ = make_target(
        spec["stage_column"],
        spec["positive_rule"],
        spec.get("exclude_control_death_3m", False),
        spec.get("event_date_column"),
        target_landmark_day_for_policy(feature_set_item(feature_set)["landmark_day"]),
    )
    X_all = feature_item["X"].loc[keep_mask].reset_index(drop=True)
    test_indices = bundle.get("test_indices", [])
    if not test_indices:
        return
    X_test = X_all.iloc[test_indices].reset_index(drop=True)
    shap_values, feature_values, pretty_names = shap_values_for_model(
        bundle["model"], X_test
    )
    if feature_values.size == 0:
        return

    plt.figure(figsize=(10.8, 7.6))
    shap.summary_plot(
        shap_values,
        feature_values,
        feature_names=pretty_names,
        max_display=20,
        show=False,
        plot_size=None,
    )
    variant_suffix = model_variant_display_name(model_variant)
    variant_title = f" {variant_suffix}" if variant_suffix else ""
    plt.title(
        f"Feature importance of predicting {target_display_name(target)}\n({feature_set_display_name(feature_set)}{variant_title} model)",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout()
    filename_suffix = (
        f"__{slugify(model_variant).lower()}"
        if model_variant != MODEL_VARIANT_ORIGINAL
        else ""
    )
    save_plot_figure(
        plt.gcf(),
        f"holdout_shap_{endpoint_file_slug(target)}__{slugify(feature_set_display_name(feature_set)).lower()}{filename_suffix}.png",
    )
    show_plot()


# Generate diagnostics after selection; plots must not feed back into the model.
def run_stage():
    """Configure plotting and generate endpoint diagnostics."""

    plt.rcParams["font.family"] = PLOT_FONT_FAMILY

    plt.rcParams["font.sans-serif"] = [PLOT_FONT_FAMILY]

    plt.rcParams["font.size"] = 15

    plt.rcParams["axes.titlesize"] = 18

    plt.rcParams["axes.labelsize"] = 16

    plt.rcParams["xtick.labelsize"] = 14

    plt.rcParams["ytick.labelsize"] = 14

    plt.rcParams["legend.fontsize"] = 13

    plt.rcParams["figure.titlesize"] = 18

    plt.rcParams["legend.title_fontsize"] = 13

    plt.rcParams["axes.titleweight"] = "semibold"

    plt.rcParams["axes.labelweight"] = "normal"

    plt.rcParams["axes.spines.top"] = False

    plt.rcParams["axes.spines.right"] = False

    plt.rcParams["axes.linewidth"] = 1.0

    plt.rcParams["axes.edgecolor"] = "#222222"

    plt.rcParams["axes.facecolor"] = "white"

    plt.rcParams["figure.facecolor"] = "white"

    plt.rcParams["savefig.facecolor"] = "white"

    plt.rcParams["savefig.edgecolor"] = "white"

    plt.rcParams["grid.color"] = "#D0D0D0"

    plt.rcParams["grid.alpha"] = 0.5

    plt.rcParams["grid.linewidth"] = 0.6

    plt.rcParams["lines.linewidth"] = 2.2

    if RUN_HOLDOUT_EVALUATION and not holdout_predictions.empty:
        for target in TARGETS_TO_RUN:
            save_endpoint_curve_grid(target, "roc")
            save_endpoint_curve_grid(target, "pr")
            save_endpoint_decision_curve_grid(target)

    if RUN_HOLDOUT_EVALUATION and not holdout_predictions.empty:
        calibration_rows = []
        test_predictions = holdout_predictions.loc[
            holdout_predictions["dataset_split"] == "test"
        ]
        test_predictions = ensure_model_variant_column(test_predictions)
        for (target, feature_set, model_variant), group in test_predictions.groupby(
            [
                "target",
                "feature_set",
                "model_variant",
            ]
        ):
            cal = calibration_points(
                group["y_true"],
                group["calibration_probability"],
                PLOT_CALIBRATION_N_BINS,
            )
            calibration_ece = expected_calibration_error(
                group["y_true"],
                group["calibration_probability"],
                CALIBRATION_N_BINS,
            )
            cal.insert(0, "target", target)
            cal.insert(1, "feature_set", feature_set)
            cal.insert(2, "model_variant", model_variant)
            cal.insert(3, "calibration_ece", calibration_ece)
            calibration_rows.extend(cal.to_dict("records"))
        calibration_table = pd.DataFrame(calibration_rows)
        calibration_table.to_csv(
            OUTPUT_DIR / "holdout_calibration_points.csv", index=False
        )

        for target in TARGETS_TO_RUN:
            save_endpoint_calibration_grid(target, calibration_table)

        if RUN_FINAL_MODELS and not holdout_model_summary.empty:
            for _, model_row in holdout_model_summary.iterrows():
                save_shap_feature_importance_plot(
                    model_row["target"],
                    model_row["feature_set"],
                    model_row.get("model_variant", MODEL_VARIANT_ORIGINAL),
                )
    elif RUN_HOLDOUT_EVALUATION:
        pd.DataFrame(
            columns=[
                "target",
                "feature_set",
                "model_variant",
                "mean_predicted_probability",
                "observed_event_rate",
                "observed_event_rate_lower",
                "observed_event_rate_upper",
                "event_n",
                "n",
                "ci_level",
                "ci_method",
                "calibration_ece",
            ]
        ).to_csv(OUTPUT_DIR / "holdout_calibration_points.csv", index=False)
