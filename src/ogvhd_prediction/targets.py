"""Landmark outcome and risk-set definitions."""

from .utils import (
    display,
    np,
    pd,
)
from .config import (
    OUTPUT_DIR,
    TARGETS_TO_RUN,
    TARGET_TIMING_POLICY,
)
from .data import (
    FEATURE_SETS,
    TARGET_DEFINITIONS,
    df,
)


def outcome_event_days(event_date_column):
    if event_date_column is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    if event_date_column not in df.columns:
        raise ValueError(f"Missing event date column: {event_date_column}")
    transplant_date = pd.to_datetime(df["Transplant Date"], errors="coerce")
    event_date = pd.to_datetime(df[event_date_column], errors="coerce")
    return (event_date - transplant_date).dt.days


def make_target_masks(
    stage_column,
    positive_rule,
    exclude_control_death_3m=False,
    event_date_column=None,
    landmark_day=None,
):
    stage = pd.to_numeric(df[stage_column], errors="coerce")
    base_keep = stage.notna()
    if positive_rule == ">0":
        y_full = (stage > 0).astype("Int64")
    elif positive_rule == ">1":
        y_full = (stage > 1).astype("Int64")
    else:
        raise ValueError(positive_rule)
    y_full = y_full.where(base_keep)

    keep = base_keep.copy()
    prior_or_current_target_event = pd.Series(False, index=df.index)
    positive_with_unknown_timing = pd.Series(False, index=df.index)
    event_days = pd.Series(np.nan, index=df.index, dtype=float)

    if landmark_day is not None and event_date_column is not None:
        event_days = outcome_event_days(event_date_column)
        target_positive = base_keep & (y_full.fillna(-1) == 1)
        prior_or_current_target_event = (
            target_positive & event_days.notna() & (event_days <= float(landmark_day))
        )
        positive_with_unknown_timing = target_positive & event_days.isna()
        keep = keep & ~prior_or_current_target_event & ~positive_with_unknown_timing

    excluded_control_death_3m = pd.Series(False, index=df.index)
    if exclude_control_death_3m:
        if "Death_3M" not in df.columns:
            raise ValueError("Death_3M column is required for cGVHD control exclusion.")
        death_3m = pd.to_numeric(df["Death_3M"], errors="coerce")
        excluded_control_death_3m = (
            base_keep & (y_full.fillna(-1) == 0) & (death_3m == 1)
        ).fillna(False)
        keep = keep & ~excluded_control_death_3m

    return {
        "base_keep": base_keep,
        "y_full": y_full,
        "keep": keep,
        "event_days": event_days,
        "prior_or_current_target_event": prior_or_current_target_event,
        "positive_with_unknown_timing": positive_with_unknown_timing,
        "excluded_control_death_3m": excluded_control_death_3m,
    }


def make_target(
    stage_column,
    positive_rule,
    exclude_control_death_3m=False,
    event_date_column=None,
    landmark_day=None,
):
    masks = make_target_masks(
        stage_column=stage_column,
        positive_rule=positive_rule,
        exclude_control_death_3m=exclude_control_death_3m,
        event_date_column=event_date_column,
        landmark_day=landmark_day,
    )
    keep = masks["keep"]
    y = masks["y_full"].loc[keep].astype(int)
    return keep.to_numpy(), y.reset_index(drop=True)


def target_landmark_day_for_policy(landmark_day):
    if TARGET_TIMING_POLICY == "fixed_outcome":
        return None
    return landmark_day


def min_class_count(y):
    y = pd.Series(y).astype(int)
    counts = y.value_counts()
    if len(counts) < 2:
        return 0
    return int(counts.min())


def run_stage():
    """Build and save the target summary."""
    global target_summary

    target_summary = []

    for target_name, spec in TARGET_DEFINITIONS.items():
        if target_name not in TARGETS_TO_RUN:
            continue
        for feature_set in FEATURE_SETS:
            masks = make_target_masks(
                stage_column=spec["stage_column"],
                positive_rule=spec["positive_rule"],
                exclude_control_death_3m=spec.get("exclude_control_death_3m", False),
                event_date_column=spec.get("event_date_column"),
                landmark_day=target_landmark_day_for_policy(
                    feature_set["landmark_day"]
                ),
            )
            y_target = (
                masks["y_full"].loc[masks["keep"]].astype(int).reset_index(drop=True)
            )
            target_summary.append(
                {
                    "target": target_name,
                    "feature_set": feature_set["feature_set"],
                    "landmark_day": feature_set["landmark_day"],
                    "stage_column": spec["stage_column"],
                    "positive_rule": spec["positive_rule"],
                    "event_date_column": spec.get("event_date_column"),
                    "target_timing_policy": TARGET_TIMING_POLICY,
                    "exclude_control_death_3m": bool(
                        spec.get("exclude_control_death_3m", False)
                    ),
                    "stage_nonmissing_n": int(masks["base_keep"].sum()),
                    "pre_landmark_or_current_target_event_excluded_n": int(
                        masks["prior_or_current_target_event"].sum()
                    ),
                    "positive_unknown_timing_excluded_n": int(
                        masks["positive_with_unknown_timing"].sum()
                    ),
                    "excluded_control_death_3m_n": int(
                        masks["excluded_control_death_3m"].sum()
                    ),
                    "n": len(y_target),
                    "positive_n": int(y_target.sum()),
                    "control_n": int(len(y_target) - y_target.sum()),
                    "positive_rate": float(y_target.mean())
                    if len(y_target)
                    else np.nan,
                    "min_class_count": min_class_count(y_target)
                    if len(y_target)
                    else 0,
                }
            )

    target_summary = pd.DataFrame(target_summary)

    display(target_summary)

    target_summary.to_csv(OUTPUT_DIR / "target_summary.csv", index=False)
