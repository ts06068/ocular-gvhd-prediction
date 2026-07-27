"""Cross-endpoint panel plots."""

from .utils import (
    Line2D,
    Normalize,
    OrderedDict,
    average_precision_score,
    gridspec,
    np,
    pd,
    plt,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    show_plot,
    slugify,
    textwrap,
    zlib,
)
from .config import (
    CALIBRATION_N_BINS,
    DCA_THRESHOLD_MAX,
    DCA_THRESHOLD_MIN,
    DCA_Y_MAX,
    DCA_Y_MIN,
    MODEL_VARIANT_ORIGINAL,
    MODEL_VARIANT_SIMPLIFIED,
    PLOT_DIR,
    RUN_FINAL_MODELS,
    RUN_HOLDOUT_EVALUATION,
)
from .data import TARGET_DEFINITIONS
from .targets import (
    make_target,
    target_landmark_day_for_policy,
)
from .metrics import expected_calibration_error
from .finalization import (
    holdout_model_summary,
    holdout_predictions,
)
from .diagnostics import (
    FEATURE_SET_COLORS,
    GRAY_TREAT,
    MODEL_VARIANT_LINESTYLES,
    PANEL_AXIS_LABEL_FONTSIZE,
    PANEL_FEATURE_SETS,
    PANEL_FIG_DPI,
    PANEL_LEGEND_FONTSIZE,
    PANEL_SHAP_AXIS_LABEL_FONTSIZE,
    PANEL_SHAP_CMAP,
    PANEL_SHAP_FEATURE_FONTSIZE,
    PANEL_SHAP_LABEL_WRAP,
    PANEL_SHAP_MAX_DISPLAY,
    PANEL_SHAP_TICK_FONTSIZE,
    PANEL_TICK_FONTSIZE,
    PANEL_TITLE_FONTSIZE,
    PANEL_TITLE_PAD,
    PLOT_CALIBRATION_N_BINS,
    calibration_points,
    curve_probability,
    decision_curve_points,
    decision_curve_probability,
    feature_set_item,
    model_bundle_for,
    shap_values_for_model,
    style_axis,
    target_display_name,
    test_prediction_group,
)

PANEL_GVHD_ENDPOINTS = OrderedDict(
    {
        "aGVHD": ("aGVHD_any", "aGVHD_severe"),
        "cGVHD": ("cGVHD_any", "cGVHD_severe"),
        "oGVHD": ("oGVHD_any", "oGVHD_severe"),
    }
)


def panel_variant_name(model_variant):
    return (
        "Simplified" if str(model_variant) == MODEL_VARIANT_SIMPLIFIED else "Original"
    )


def panel_endpoint_title(target):
    title = target_display_name(target)
    return title[:1].upper() + title[1:]


def make_dual_endpoint_panel_figure(metric_title, targets, shap_panel=False):
    figsize = (36.0, 19.0) if shap_panel else (30.0, 17.0)
    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        left=0.06 if not shap_panel else 0.085,
        right=0.97,
        top=0.86,
        bottom=0.13,
        wspace=0.16,
    )
    axes_by_target = OrderedDict()
    all_axes = []
    for column, target in enumerate(targets):
        inner = gridspec.GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=outer[column],
            wspace=0.48 if shap_panel else 0.30,
            hspace=0.34 if shap_panel else 0.30,
        )
        target_axes = []
        for panel_index in range(4):
            ax = fig.add_subplot(inner[panel_index // 2, panel_index % 2])
            if not shap_panel:
                ax.set_box_aspect(1)
            target_axes.append(ax)
            all_axes.append(ax)
        axes_by_target[target] = target_axes

    fig.canvas.draw()
    for target, target_axes in axes_by_target.items():
        positions = [ax.get_position() for ax in target_axes]
        center_x = (
            min(position.x0 for position in positions)
            + max(position.x1 for position in positions)
        ) / 2
        fig.text(
            center_x,
            0.90,
            panel_endpoint_title(target),
            ha="center",
            va="center",
            fontsize=PANEL_TITLE_FONTSIZE,
            fontweight="semibold",
        )
    fig.suptitle(
        metric_title, y=0.95, fontsize=PANEL_TITLE_FONTSIZE, fontweight="semibold"
    )
    return fig, axes_by_target, all_axes


def panel_empty_axis(ax, panel_title, message="Not estimable"):
    ax.set_title(
        panel_title,
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="semibold",
        pad=PANEL_TITLE_PAD,
    )
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#666666",
        fontsize=PANEL_AXIS_LABEL_FONTSIZE,
    )
    ax.axis("off")


def save_panel_figure(fig, filename):
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=PANEL_FIG_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {pdf_path}")
    show_plot()


def plot_panel_discrimination_axis(ax, target, feature_set, panel_title, curve_kind):
    feature_predictions = test_prediction_group(target, feature_set)
    if feature_predictions.empty:
        panel_empty_axis(ax, panel_title)
        return
    color = FEATURE_SET_COLORS.get(feature_set, "#333333")
    plotted = False
    for model_variant in [MODEL_VARIANT_SIMPLIFIED, MODEL_VARIANT_ORIGINAL]:
        group = test_prediction_group(target, feature_set, model_variant)
        if group.empty or group["y_true"].nunique() < 2:
            continue
        y_true = group["y_true"].astype(int).to_numpy()
        y_prob = curve_probability(group)
        linestyle = MODEL_VARIANT_LINESTYLES[model_variant]
        if curve_kind == "roc":
            x_values, y_values, _ = roc_curve(y_true, y_prob)
            score = roc_auc_score(y_true, y_prob)
            metric_label = "AUROC"
        elif curve_kind == "pr":
            y_values, x_values, _ = precision_recall_curve(y_true, y_prob)
            score = average_precision_score(y_true, y_prob)
            metric_label = "AUPRC"
        else:
            raise ValueError(curve_kind)
        ax.plot(
            x_values,
            y_values,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"{panel_variant_name(model_variant)}: {metric_label}={score:.3f}",
        )
        plotted = True

    if not plotted:
        panel_empty_axis(ax, panel_title)
        return
    if curve_kind == "roc":
        ax.plot([0, 1], [0, 1], color="#B0B0B0", linestyle="--", linewidth=1.2)
        ax.set_xlabel("False positive rate", fontsize=PANEL_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("True positive rate", fontsize=PANEL_AXIS_LABEL_FONTSIZE)
        legend_location = "lower right"
    else:
        baseline = float(feature_predictions["y_true"].astype(int).mean())
        ax.axhline(baseline, color="#B0B0B0", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Recall", fontsize=PANEL_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Precision", fontsize=PANEL_AXIS_LABEL_FONTSIZE)
        legend_location = "upper right"
    ax.set_title(
        panel_title,
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="semibold",
        pad=PANEL_TITLE_PAD,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(
        loc=legend_location,
        frameon=False,
        fontsize=PANEL_LEGEND_FONTSIZE,
        handlelength=2.4,
    )
    style_axis(ax)
    ax.tick_params(labelsize=PANEL_TICK_FONTSIZE)


def save_gvhd_discrimination_panel(gvhd_label, targets, curve_kind):
    metric_title = (
        f"Receiver operating characteristic curves for {gvhd_label} models"
        if curve_kind == "roc"
        else f"Precision–recall curves for {gvhd_label} models"
    )
    fig, axes_by_target, _ = make_dual_endpoint_panel_figure(metric_title, targets)
    for target, axes in axes_by_target.items():
        for ax, (panel_title, feature_set) in zip(axes, PANEL_FEATURE_SETS.items()):
            plot_panel_discrimination_axis(
                ax, target, feature_set, panel_title, curve_kind
            )
    save_panel_figure(fig, f"panel_{curve_kind}_{slugify(gvhd_label).lower()}.png")


def plot_panel_calibration_axis(ax, target, feature_set, panel_title):
    feature_predictions = test_prediction_group(target, feature_set)
    if feature_predictions.empty:
        panel_empty_axis(ax, panel_title)
        return
    color = FEATURE_SET_COLORS.get(feature_set, "#333333")
    plotted = False
    for model_variant in [MODEL_VARIANT_SIMPLIFIED]:
        group = test_prediction_group(target, feature_set, model_variant)
        if group.empty:
            continue
        y_true = group["y_true"].astype(int).to_numpy()
        y_prob = pd.to_numeric(
            group["calibration_probability"], errors="coerce"
        ).to_numpy(dtype=float)
        calibration = calibration_points(y_true, y_prob, PLOT_CALIBRATION_N_BINS)
        if calibration.empty:
            continue
        observed = calibration["observed_event_rate"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                observed
                - calibration["observed_event_rate_lower"].to_numpy(dtype=float),
                calibration["observed_event_rate_upper"].to_numpy(dtype=float)
                - observed,
            ]
        )
        yerr = np.nan_to_num(np.clip(yerr, 0.0, None), nan=0.0)
        ece = expected_calibration_error(y_true, y_prob, CALIBRATION_N_BINS)
        ax.errorbar(
            calibration["mean_predicted_probability"].to_numpy(dtype=float),
            observed,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=4.0,
            markersize=7.0,
            markerfacecolor=(
                color if model_variant == MODEL_VARIANT_SIMPLIFIED else "white"
            ),
            markeredgecolor=color,
            linestyle="none",
            label=f"{panel_variant_name(model_variant)}: ECE={ece:.3f}",
        )
        plotted = True
    if not plotted:
        panel_empty_axis(ax, panel_title)
        return
    ax.plot([0, 1], [0, 1], color="#B0B0B0", linestyle="--", linewidth=1.2)
    ax.set_title(
        panel_title,
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="semibold",
        pad=PANEL_TITLE_PAD,
    )
    ax.set_xlabel("Mean predicted probability", fontsize=PANEL_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Observed event rate", fontsize=PANEL_AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=PANEL_LEGEND_FONTSIZE,
        handlelength=2.4,
    )
    style_axis(ax)
    ax.tick_params(labelsize=PANEL_TICK_FONTSIZE)


def save_gvhd_calibration_panel(gvhd_label, targets):
    title = f"Calibration plots for simplified {gvhd_label} models"
    fig, axes_by_target, _ = make_dual_endpoint_panel_figure(title, targets)
    for target, axes in axes_by_target.items():
        for ax, (panel_title, feature_set) in zip(axes, PANEL_FEATURE_SETS.items()):
            plot_panel_calibration_axis(ax, target, feature_set, panel_title)
    save_panel_figure(fig, f"panel_calibration_{slugify(gvhd_label).lower()}.png")


def panel_dca_payload(targets):
    thresholds = np.linspace(DCA_THRESHOLD_MIN, DCA_THRESHOLD_MAX, 80)
    payload = {}
    y_values = [np.array([0.0])]
    for target in targets:
        for feature_set in PANEL_FEATURE_SETS.values():
            variant_payload = OrderedDict()
            for model_variant in [MODEL_VARIANT_SIMPLIFIED, MODEL_VARIANT_ORIGINAL]:
                group = test_prediction_group(target, feature_set, model_variant)
                if group.empty:
                    continue
                dca = decision_curve_points(
                    group["y_true"].astype(int).to_numpy(),
                    decision_curve_probability(group),
                    thresholds,
                )
                variant_payload[model_variant] = dca
                y_values.append(dca["model_net_benefit"].to_numpy(dtype=float))
            payload[(target, feature_set)] = variant_payload
            if variant_payload:
                reference = next(iter(variant_payload.values()))
                y_values.extend(
                    [
                        np.clip(
                            reference["treat_all_net_benefit"].to_numpy(dtype=float),
                            DCA_Y_MIN,
                            None,
                        ),
                        reference["treat_none_net_benefit"].to_numpy(dtype=float),
                    ]
                )
    finite_values = np.concatenate(
        [values[np.isfinite(values)] for values in y_values if len(values)]
    )
    if DCA_Y_MAX is not None:
        y_upper = DCA_Y_MAX
    elif len(finite_values):
        y_upper = max(0.03, float(np.nanpercentile(finite_values, 98)) * 1.15 + 0.005)
    else:
        y_upper = 0.05
    return payload, thresholds, max(y_upper, DCA_Y_MIN + 0.05)


def save_gvhd_net_benefit_panel(gvhd_label, targets):
    payload, thresholds, y_upper = panel_dca_payload(targets)
    title = f"Decision curve analysis for {gvhd_label} models"
    fig, axes_by_target, _ = make_dual_endpoint_panel_figure(title, targets)
    for target, axes in axes_by_target.items():
        for ax, (panel_title, feature_set) in zip(axes, PANEL_FEATURE_SETS.items()):
            variant_payload = payload.get((target, feature_set), {})
            if not variant_payload:
                panel_empty_axis(ax, panel_title)
                continue
            color = FEATURE_SET_COLORS.get(feature_set, "#333333")
            for model_variant in [MODEL_VARIANT_SIMPLIFIED, MODEL_VARIANT_ORIGINAL]:
                dca = variant_payload.get(model_variant)
                if dca is None:
                    continue
                ax.plot(
                    dca["threshold"],
                    dca["model_net_benefit"],
                    color=color,
                    linestyle=MODEL_VARIANT_LINESTYLES[model_variant],
                    linewidth=1.8,
                )
            reference = next(iter(variant_payload.values()))
            ax.plot(
                reference["threshold"],
                reference["treat_all_net_benefit"],
                color=GRAY_TREAT,
                linestyle="--",
                linewidth=1.4,
            )
            ax.plot(
                reference["threshold"],
                reference["treat_none_net_benefit"],
                color=GRAY_TREAT,
                linestyle="-.",
                linewidth=1.4,
            )
            ax.set_title(
                panel_title,
                fontsize=PANEL_TITLE_FONTSIZE,
                fontweight="semibold",
                pad=PANEL_TITLE_PAD,
            )
            ax.set_xlabel("Threshold probability", fontsize=PANEL_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("Net benefit", fontsize=PANEL_AXIS_LABEL_FONTSIZE)
            ax.set_xlim(thresholds.min(), thresholds.max())
            ax.set_ylim(DCA_Y_MIN, y_upper)
            style_axis(ax)
            ax.tick_params(labelsize=PANEL_TICK_FONTSIZE)
    legend_handles = [
        Line2D([], [], color="#222222", linestyle="-", linewidth=2, label="Simplified"),
        Line2D([], [], color="#222222", linestyle=":", linewidth=2, label="Original"),
        Line2D(
            [], [], color=GRAY_TREAT, linestyle="--", linewidth=1.6, label="Treat all"
        ),
        Line2D(
            [], [], color=GRAY_TREAT, linestyle="-.", linewidth=1.6, label="Treat none"
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=PANEL_LEGEND_FONTSIZE,
        handlelength=3.0,
    )
    save_panel_figure(fig, f"panel_net_benefit_{slugify(gvhd_label).lower()}.png")


def simplified_shap_payload(target, feature_set):
    bundle = model_bundle_for(target, feature_set, MODEL_VARIANT_SIMPLIFIED)
    if bundle is None:
        return None
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
        return None
    X_test = X_all.iloc[test_indices].reset_index(drop=True)
    return shap_values_for_model(bundle["model"], X_test)


def panel_shap_feature_label(feature_name):
    return "\n".join(
        textwrap.wrap(
            str(feature_name),
            width=PANEL_SHAP_LABEL_WRAP,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def plot_compact_shap_axis(ax, target, feature_set, panel_title):
    payload = simplified_shap_payload(target, feature_set)
    if payload is None:
        panel_empty_axis(ax, panel_title)
        return False
    shap_values, feature_values, feature_names = payload
    if shap_values.size == 0:
        panel_empty_axis(ax, panel_title)
        return False
    importance = np.nanmean(np.abs(shap_values), axis=0)
    display_n = min(PANEL_SHAP_MAX_DISPLAY, len(importance))
    selected_indices = np.argsort(-importance, kind="mergesort")[:display_n]
    row_spacing = 1.55
    cmap = PANEL_SHAP_CMAP
    for y_position, feature_index in enumerate(selected_indices):
        shap_feature = shap_values[:, feature_index].astype(float)
        value_feature = feature_values[:, feature_index].astype(float)
        finite_values = value_feature[np.isfinite(value_feature)]
        if len(finite_values):
            lower, upper = np.nanpercentile(finite_values, [5, 95])
            if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
                normalized_value = np.clip(
                    (value_feature - lower) / (upper - lower), 0.0, 1.0
                )
            else:
                normalized_value = np.full(len(value_feature), 0.5)
        else:
            normalized_value = np.full(len(value_feature), 0.5)
        rng_seed = zlib.crc32(
            f"{target}|{feature_set}|{feature_names[feature_index]}".encode("utf-8")
        )
        rng = np.random.default_rng(rng_seed)
        jitter = rng.normal(0.0, 0.085, size=len(shap_feature))
        ax.scatter(
            shap_feature,
            y_position * row_spacing + jitter,
            c=normalized_value,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=16,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
        )
    ax.axvline(0.0, color="#888888", linewidth=1.0)
    ax.set_yticks(np.arange(display_n) * row_spacing)
    ax.set_yticklabels(
        [panel_shap_feature_label(feature_names[index]) for index in selected_indices],
        fontsize=PANEL_SHAP_FEATURE_FONTSIZE,
    )
    ax.set_ylim((display_n - 0.5) * row_spacing, -0.5 * row_spacing)
    ax.set_title(
        panel_title,
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="semibold",
        pad=PANEL_TITLE_PAD,
    )
    ax.set_xlabel("SHAP value", fontsize=PANEL_SHAP_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=PANEL_SHAP_TICK_FONTSIZE)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#444444")
        ax.spines[spine].set_linewidth(0.8)
    return True


def save_gvhd_shap_panel(gvhd_label, targets):
    title = f"SHAP summary plots for simplified {gvhd_label} models"
    fig, axes_by_target, all_axes = make_dual_endpoint_panel_figure(
        title, targets, shap_panel=True
    )
    plotted = False
    for target, axes in axes_by_target.items():
        for ax, (panel_title, feature_set) in zip(axes, PANEL_FEATURE_SETS.items()):
            plotted = (
                plot_compact_shap_axis(ax, target, feature_set, panel_title) or plotted
            )
    if plotted:
        scalar_mappable = plt.cm.ScalarMappable(
            norm=Normalize(vmin=0.0, vmax=1.0), cmap=PANEL_SHAP_CMAP
        )
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=all_axes, fraction=0.012, pad=0.018)
        colorbar.set_ticks([0.0, 1.0])
        colorbar.set_ticklabels(["Low", "High"])
        colorbar.set_label("Relative feature value", fontsize=PANEL_LEGEND_FONTSIZE)
        colorbar.ax.tick_params(labelsize=PANEL_LEGEND_FONTSIZE)
    save_panel_figure(fig, f"panel_shap_{slugify(gvhd_label).lower()}.png")


# Build cross-endpoint panels only after every endpoint diagnostic is final.
def save_all_panel_plots():
    for gvhd_label, targets in PANEL_GVHD_ENDPOINTS.items():
        save_gvhd_discrimination_panel(gvhd_label, targets, "roc")
        save_gvhd_discrimination_panel(gvhd_label, targets, "pr")
        save_gvhd_calibration_panel(gvhd_label, targets)
        save_gvhd_net_benefit_panel(gvhd_label, targets)
        if RUN_FINAL_MODELS and not holdout_model_summary.empty:
            save_gvhd_shap_panel(gvhd_label, targets)


def run_stage():
    """Generate the cross-endpoint panel plots."""

    if RUN_HOLDOUT_EVALUATION and not holdout_predictions.empty:
        save_all_panel_plots()
