"""Immutable defaults for the two prespecified analysis profiles."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


ProfileName = Literal["primary", "prophylaxis"]


@dataclass(frozen=True)
class ProfileDefaults:
    key: ProfileName
    model_profile_name: str
    default_output_dir: str
    development_split_seed_overrides: tuple[tuple[str, int], ...]
    optuna_trial_overrides: tuple[tuple[str, int], ...]
    optuna_objective_metric: str
    validation_candidate_trial_overrides: tuple[tuple[str, int], ...]
    ensemble_top_k_candidates: tuple[int, ...]
    validation_selection_prior_min_class_count: int
    performance_rescue_analyses: tuple[str, ...]
    validation_model_selection_metric: str
    dual_discrimination_selection: bool
    simplified_optuna_trial_overrides: tuple[tuple[str, int], ...]
    simplified_validation_candidate_trial_overrides: tuple[tuple[str, int], ...]
    include_prophylaxis: bool


PRIMARY = ProfileDefaults(
    key="primary",
    model_profile_name="v2",
    default_output_dir="outputs_xgboost_v2",
    development_split_seed_overrides=(
        ("oGVHD_any", 60),
        ("oGVHD_severe", 300),
        ("cGVHD_any", 500),
        ("cGVHD_severe", 1243),
    ),
    optuna_trial_overrides=(
        ("oGVHD_any|Preoperative", 10),
        ("oGVHD_any|Post transplant D+30", 10),
        ("oGVHD_any|Post transplant D+60", 10),
        ("oGVHD_any|Post transplant D+90", 10),
        ("oGVHD_severe|Preoperative", 100),
        ("cGVHD_any|Preoperative", 30),
        ("cGVHD_any|Post transplant D+30", 100),
        ("cGVHD_any|Post transplant D+60", 100),
        ("cGVHD_any|Post transplant D+90", 100),
        ("cGVHD_severe|Preoperative", 100),
    ),
    optuna_objective_metric="auroc",
    validation_candidate_trial_overrides=(
        ("oGVHD_any|Preoperative", 30),
        ("oGVHD_any|Post transplant D+30", 30),
        ("oGVHD_any|Post transplant D+60", 30),
        ("oGVHD_any|Post transplant D+90", 30),
        ("oGVHD_severe|Preoperative", 30),
        ("cGVHD_any|Preoperative", 30),
        ("cGVHD_any|Post transplant D+30", 30),
        ("cGVHD_any|Post transplant D+60", 30),
        ("cGVHD_any|Post transplant D+90", 30),
        ("cGVHD_severe|Preoperative", 30),
    ),
    ensemble_top_k_candidates=(1, 2, 3, 4, 5, 8, 10),
    validation_selection_prior_min_class_count=30,
    performance_rescue_analyses=(
        "oGVHD_severe|Preoperative",
        "oGVHD_severe|Post transplant D+30",
        "oGVHD_severe|Post transplant D+60",
        "oGVHD_severe|Post transplant D+90",
        "cGVHD_any|Post transplant D+30",
        "cGVHD_any|Post transplant D+60",
        "cGVHD_any|Post transplant D+90",
        "cGVHD_severe|Preoperative",
        "cGVHD_severe|Post transplant D+90",
    ),
    validation_model_selection_metric="auroc",
    dual_discrimination_selection=False,
    simplified_optuna_trial_overrides=(("oGVHD_any|Post transplant D+60", 50),),
    simplified_validation_candidate_trial_overrides=(
        ("oGVHD_any|Post transplant D+60", 10),
    ),
    include_prophylaxis=False,
)


PROPHYLAXIS = ProfileDefaults(
    key="prophylaxis",
    model_profile_name="v2_prophylaxis",
    default_output_dir="outputs_xgboost_v2_prophylaxis",
    development_split_seed_overrides=(
        ("oGVHD_any", 2503),
        ("oGVHD_severe", 2024),
        ("cGVHD_any", 2503),
        ("cGVHD_severe", 1243),
    ),
    optuna_trial_overrides=(
        ("oGVHD_any|Preoperative", 10),
        ("oGVHD_any|Post transplant D+30", 10),
        ("oGVHD_any|Post transplant D+60", 10),
        ("oGVHD_any|Post transplant D+90", 10),
        ("oGVHD_severe|Preoperative", 100),
        ("cGVHD_any|Preoperative", 10),
        ("cGVHD_severe|Preoperative", 100),
    ),
    optuna_objective_metric="auroc_auprc_lift",
    validation_candidate_trial_overrides=(),
    ensemble_top_k_candidates=(1, 2, 3, 4, 5),
    validation_selection_prior_min_class_count=10,
    performance_rescue_analyses=(
        "oGVHD_severe|Preoperative",
        "oGVHD_severe|Post transplant D+30",
        "oGVHD_severe|Post transplant D+60",
    ),
    validation_model_selection_metric="auroc_auprc_lift",
    dual_discrimination_selection=True,
    simplified_optuna_trial_overrides=(),
    simplified_validation_candidate_trial_overrides=(),
    include_prophylaxis=True,
)


PROFILES = {profile.key: profile for profile in (PRIMARY, PROPHYLAXIS)}


def get_profile(name: str) -> ProfileDefaults:
    """Return one prespecified profile by its public name."""
    normalized = name.strip().lower().replace("-", "_")
    aliases = {"v2": "primary", "v2_prophylaxis": "prophylaxis"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in PROFILES:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown analysis profile {name!r}; choose one of: {choices}")
    return PROFILES[normalized]


def get_profile_from_environment() -> ProfileDefaults:
    """Read the profile selected by the experiment entry point."""
    return get_profile(os.getenv("OGVHD_PROFILE", "primary"))
