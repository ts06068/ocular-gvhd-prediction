"""Frozen holdout and development split construction."""

from . import preprocessing as _preprocessing  # noqa: F401 - preserve pipeline stage order.
from .utils import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    np,
    pd,
    train_test_split,
    warnings,
    zlib,
)
from .config import (
    AGVHD_XGB_REPRODUCTION_PRESETS,
    AGVHD_XGB_REPRODUCTION_PROFILE,
    DATA_SHA256,
    DEVELOPMENT_SPLIT_SEED_OVERRIDES,
    ENSEMBLE_TOP_K_CANDIDATES,
    FEATURE_SELECTION_CANDIDATES,
    FINALIZATION_CANDIDATES,
    FINAL_REFIT_SCOPE,
    FINAL_REFIT_SCOPE_OVERRIDES,
    FIXED_SPLIT_SEED_OVERRIDES,
    FROZEN_TEST_SPLIT_MANIFEST_PATH,
    MODEL_PROFILE_NAME,
    OPTUNA_CV_REPEATS,
    OPTUNA_TRIALS,
    OPTUNA_TRIAL_OVERRIDES,
    PERFORMANCE_RESCUE_ANALYSES,
    PERFORMANCE_RESCUE_CV_REPEATS,
    PERFORMANCE_RESCUE_FEATURE_SELECTION_CANDIDATES,
    PERFORMANCE_RESCUE_FINALIZATION_CANDIDATES,
    PERFORMANCE_RESCUE_OPTUNA_TRIALS,
    PERFORMANCE_RESCUE_SIMPLIFIED_FEATURE_SELECTION_CANDIDATES,
    PERFORMANCE_RESCUE_VALIDATION_CANDIDATE_TRIALS,
    RANDOM_STATE,
    SEED_BAGGING_REPEATS,
    SEED_BAGGING_REPEAT_OVERRIDES,
    SIMPLIFIED_OPTUNA_TRIALS,
    SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES,
    SIMPLIFIED_SEED_BAGGING_REPEATS,
    SIMPLIFIED_SEED_BAGGING_REPEAT_OVERRIDES,
    SIMPLIFIED_SHAP_RANKING_MODE,
    SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES,
    SIMPLIFIED_VALIDATION_CANDIDATE_TRIALS,
    SIMPLIFIED_VALIDATION_CANDIDATE_TRIAL_OVERRIDES,
    SPLIT_POLICY,
    SPLIT_RANDOM_STATE_OVERRIDES,
    STRICT_FROZEN_TEST_SPLIT_MANIFEST,
    TEST_SIZE,
    TRAIN_SIZE,
    USE_FROZEN_TEST_SPLIT_MANIFEST,
    VALIDATION_CANDIDATE_TRIALS,
    VALIDATION_CANDIDATE_TRIAL_OVERRIDES,
    VAL_SIZE,
)
from .data import (
    FROZEN_TEST_SPLIT_MANIFEST_SHA256,
    SPLIT_EPISODE_HASHES,
    frozen_test_split_manifest,
)
from .targets import (
    make_target,
    min_class_count,
    target_landmark_day_for_policy,
)


def model_family_label(param_list):
    return "xgboost"


def make_cv(y, requested_splits, seed):
    n_splits = min(int(requested_splits), min_class_count(y))
    if n_splits < 2:
        raise ValueError("Not enough positive/negative samples for stratified CV.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))


def make_tuning_cv(y, requested_splits, repeats, seed):
    n_splits = min(int(requested_splits), min_class_count(y))
    if n_splits < 2:
        raise ValueError("Not enough positive/negative samples for stratified CV.")
    return RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=int(repeats),
        random_state=int(seed),
    )


def effective_tuning_cv_repeats(y, requested_repeats=None):
    repeats = OPTUNA_CV_REPEATS if requested_repeats is None else requested_repeats
    return int(repeats if min_class_count(y) >= 10 else 1)


def effective_cv_repeats_for_settings(y, settings):
    return effective_tuning_cv_repeats(y, settings["cv_repeats"])


def make_train_val_test_split(y, seed):
    y = pd.Series(y).astype(int).reset_index(drop=True)
    if min_class_count(y) < 4:
        raise ValueError(
            "At least 4 positives and 4 controls are required for stratified 6:2:2 splitting."
        )

    all_idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        all_idx,
        train_size=TRAIN_SIZE,
        stratify=y,
        random_state=int(seed),
        shuffle=True,
    )
    relative_val_size = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_val_size,
        stratify=y.iloc[temp_idx],
        random_state=int(seed) + 17,
        shuffle=True,
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def split_random_state_base(target_name, feature_name):
    feature_key = f"{target_name}|{feature_name}"
    if feature_key in SPLIT_RANDOM_STATE_OVERRIDES:
        return int(SPLIT_RANDOM_STATE_OVERRIDES[feature_key])
    if target_name in SPLIT_RANDOM_STATE_OVERRIDES:
        return int(SPLIT_RANDOM_STATE_OVERRIDES[target_name])
    return int(RANDOM_STATE)


def target_split_seed(target_name):
    return int(split_random_state_base(target_name, "__target_split__")) + (
        zlib.crc32(target_name.encode("utf-8")) % 1000
    )


def fixed_split_seed(target_name, feature_name=None):
    if feature_name is not None:
        feature_key = f"{target_name}|{feature_name}"
        if feature_key in FIXED_SPLIT_SEED_OVERRIDES:
            return int(FIXED_SPLIT_SEED_OVERRIDES[feature_key])
    if target_name in FIXED_SPLIT_SEED_OVERRIDES:
        return int(FIXED_SPLIT_SEED_OVERRIDES[target_name])
    return int(analysis_seed(target_name, "Preoperative"))


def split_seed_for_analysis(target_name, feature_name):
    if SPLIT_POLICY == "fixed_by_target":
        return fixed_split_seed(target_name, feature_name)
    if SPLIT_POLICY == "target_hash":
        return target_split_seed(target_name)
    return analysis_seed(target_name, feature_name)


def has_feature_fixed_split_override(target_name, feature_name):
    return f"{target_name}|{feature_name}" in FIXED_SPLIT_SEED_OVERRIDES


def final_refit_scope_for_analysis(target_name, feature_name):
    feature_key = f"{target_name}|{feature_name}"
    if feature_key in FINAL_REFIT_SCOPE_OVERRIDES:
        return FINAL_REFIT_SCOPE_OVERRIDES[feature_key]
    if target_name in FINAL_REFIT_SCOPE_OVERRIDES:
        return FINAL_REFIT_SCOPE_OVERRIDES[target_name]
    return FINAL_REFIT_SCOPE


def simplified_shap_ranking_mode_for_analysis(target_name, feature_name):
    feature_key = f"{target_name}|{feature_name}"
    if feature_key in SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES:
        return SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES[feature_key]
    if target_name in SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES:
        return SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES[target_name]
    return SIMPLIFIED_SHAP_RANKING_MODE


def integer_override_for_analysis(overrides, target_name, feature_name, default):
    feature_key = f"{target_name}|{feature_name}"
    if feature_key in overrides:
        return int(overrides[feature_key])
    if target_name in overrides:
        return int(overrides[target_name])
    return int(default)


def modeling_method_for_target(target_name):
    # Every endpoint in this notebook is intentionally trained with XGBoost.
    return "xgboost"


def optimization_settings_for_analysis(target_name, feature_name):
    analysis_key = f"{target_name}|{feature_name}"
    rescue_enabled = analysis_key in PERFORMANCE_RESCUE_ANALYSES
    modeling_method = modeling_method_for_target(target_name)
    agvhd_reproduction = (
        AGVHD_XGB_REPRODUCTION_PROFILE
        and analysis_key in AGVHD_XGB_REPRODUCTION_PRESETS
    )
    optuna_trials = integer_override_for_analysis(
        OPTUNA_TRIAL_OVERRIDES,
        target_name,
        feature_name,
        OPTUNA_TRIALS,
    )
    simplified_optuna_trials = integer_override_for_analysis(
        SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES,
        target_name,
        feature_name,
        SIMPLIFIED_OPTUNA_TRIALS,
    )
    validation_candidate_trials = integer_override_for_analysis(
        VALIDATION_CANDIDATE_TRIAL_OVERRIDES,
        target_name,
        feature_name,
        VALIDATION_CANDIDATE_TRIALS,
    )
    simplified_validation_candidate_trials = integer_override_for_analysis(
        SIMPLIFIED_VALIDATION_CANDIDATE_TRIAL_OVERRIDES,
        target_name,
        feature_name,
        SIMPLIFIED_VALIDATION_CANDIDATE_TRIALS,
    )
    if rescue_enabled:
        optuna_trials = max(optuna_trials, PERFORMANCE_RESCUE_OPTUNA_TRIALS)
        simplified_optuna_trials = max(
            simplified_optuna_trials,
            PERFORMANCE_RESCUE_OPTUNA_TRIALS,
        )
        validation_candidate_trials = max(
            validation_candidate_trials,
            PERFORMANCE_RESCUE_VALIDATION_CANDIDATE_TRIALS,
        )
        simplified_validation_candidate_trials = max(
            simplified_validation_candidate_trials,
            PERFORMANCE_RESCUE_VALIDATION_CANDIDATE_TRIALS,
        )
    if agvhd_reproduction:
        optuna_trials = len(AGVHD_XGB_REPRODUCTION_PRESETS[analysis_key])
        simplified_optuna_trials = optuna_trials
        validation_candidate_trials = 0
        simplified_validation_candidate_trials = 0

    return {
        "modeling_method": modeling_method,
        "analysis_key": analysis_key,
        "agvhd_xgb_reproduction_profile": agvhd_reproduction,
        "performance_rescue": rescue_enabled,
        "optuna_trials": optuna_trials,
        "simplified_optuna_trials": simplified_optuna_trials,
        "validation_candidate_trials": validation_candidate_trials,
        "seed_bagging_repeats": integer_override_for_analysis(
            SEED_BAGGING_REPEAT_OVERRIDES,
            target_name,
            feature_name,
            SEED_BAGGING_REPEATS,
        ),
        "simplified_validation_candidate_trials": (
            simplified_validation_candidate_trials
        ),
        "simplified_seed_bagging_repeats": integer_override_for_analysis(
            SIMPLIFIED_SEED_BAGGING_REPEAT_OVERRIDES,
            target_name,
            feature_name,
            SIMPLIFIED_SEED_BAGGING_REPEATS,
        ),
        "cv_repeats": (
            max(OPTUNA_CV_REPEATS, PERFORMANCE_RESCUE_CV_REPEATS)
            if rescue_enabled
            else OPTUNA_CV_REPEATS
        ),
        "feature_selection_candidates": (
            PERFORMANCE_RESCUE_FEATURE_SELECTION_CANDIDATES
            if rescue_enabled
            else FEATURE_SELECTION_CANDIDATES
        ),
        "simplified_feature_selection_candidates": (
            PERFORMANCE_RESCUE_SIMPLIFIED_FEATURE_SELECTION_CANDIDATES
            if rescue_enabled
            else FEATURE_SELECTION_CANDIDATES
        ),
        "finalization_candidates": (
            ["full_fit_seed1"]
            if agvhd_reproduction
            else PERFORMANCE_RESCUE_FINALIZATION_CANDIDATES
            if rescue_enabled
            else FINALIZATION_CANDIDATES
        ),
        "ensemble_top_k_candidates": (
            [2] if agvhd_reproduction else ENSEMBLE_TOP_K_CANDIDATES
        ),
        "include_dual_metric_champions": not agvhd_reproduction,
    }


def analysis_seed(target_name, feature_name):
    return (
        split_random_state_base(target_name, feature_name)
        + (zlib.crc32(target_name.encode("utf-8")) % 1000)
        + (zlib.crc32(feature_name.encode("utf-8")) % 1000)
    )


SOURCE_SPLIT_CACHE = {}
SELECTED_DEVELOPMENT_SPLIT_SEEDS = {}
SPLIT_SEED_SEARCH_ROWS = []
FROZEN_TEST_SPLIT_AUDIT = {}


def frozen_test_source_indices(
    target_name,
    test_seed,
    reference_source_indices,
    y_reference,
):
    if not USE_FROZEN_TEST_SPLIT_MANIFEST:
        return None

    manifest_group = frozen_test_split_manifest.loc[
        frozen_test_split_manifest["model_profile"].astype(str).eq(MODEL_PROFILE_NAME)
        & frozen_test_split_manifest["target"].astype(str).eq(str(target_name))
        & frozen_test_split_manifest["fixed_test_seed"].eq(int(test_seed))
        & frozen_test_split_manifest["dataset_split"].astype(str).eq("test")
    ].copy()
    if manifest_group.empty:
        message = (
            "No frozen test split was found for "
            f"target={target_name}, seed={test_seed}."
        )
        if STRICT_FROZEN_TEST_SPLIT_MANIFEST:
            raise ValueError(message)
        warnings.warn(message + " Falling back to positional stratified splitting.")
        return None

    expected_hashes = set(manifest_group["episode_hash"].astype(str))
    reference_source_indices = np.asarray(reference_source_indices, dtype=int)
    reference_hashes = (
        SPLIT_EPISODE_HASHES.iloc[reference_source_indices].astype(str).to_numpy()
    )
    matched_positions = np.flatnonzero(
        np.asarray([value in expected_hashes for value in reference_hashes])
    )
    matched_hashes = set(reference_hashes[matched_positions])
    missing_hashes = expected_hashes - matched_hashes
    expected_test_n = int(manifest_group["expected_test_n"].iloc[0])
    maximum_allowed_missing = max(2, int(np.ceil(expected_test_n * 0.02)))
    if len(missing_hashes) > maximum_allowed_missing:
        message = (
            f"Frozen test split mismatch for {target_name}, seed={test_seed}: "
            f"matched={len(matched_positions)}/{expected_test_n}, "
            f"missing={len(missing_hashes)}."
        )
        if STRICT_FROZEN_TEST_SPLIT_MANIFEST:
            raise ValueError(message)
        warnings.warn(message + " Falling back to positional stratified splitting.")
        return None

    test_y = pd.Series(y_reference).iloc[matched_positions].astype(int)
    if len(test_y) == 0 or test_y.nunique() < 2:
        raise ValueError(
            f"Frozen test split for {target_name}, seed={test_seed} lacks both classes."
        )

    audit_key = (str(target_name), int(test_seed))
    FROZEN_TEST_SPLIT_AUDIT[audit_key] = {
        "target": str(target_name),
        "fixed_test_seed": int(test_seed),
        "manifest_path": str(FROZEN_TEST_SPLIT_MANIFEST_PATH),
        "manifest_sha256": FROZEN_TEST_SPLIT_MANIFEST_SHA256,
        "manifest_source_data_sha256": str(
            manifest_group["source_data_sha256"].iloc[0]
        ),
        "current_data_sha256": DATA_SHA256,
        "reference_n": int(len(reference_source_indices)),
        "reference_positive_n": int(pd.Series(y_reference).sum()),
        "expected_test_n": expected_test_n,
        "matched_test_n": int(len(matched_positions)),
        "expected_test_positive_n": int(
            manifest_group["expected_test_positive_n"].iloc[0]
        ),
        "matched_test_positive_n": int(test_y.sum()),
        "missing_manifest_episode_n": int(len(missing_hashes)),
        "test_labels_used_for_current_split_selection": False,
        "status": "matched" if not missing_hashes else "matched_after_dataset_deletion",
    }
    return reference_source_indices[matched_positions]


def frozen_complete_source_split(
    target_name,
    test_seed,
    development_seed,
    reference_source_indices,
    y_reference,
):
    if not USE_FROZEN_TEST_SPLIT_MANIFEST:
        return None
    manifest_group = frozen_test_split_manifest.loc[
        frozen_test_split_manifest["model_profile"].astype(str).eq(MODEL_PROFILE_NAME)
        & frozen_test_split_manifest["target"].astype(str).eq(str(target_name))
        & frozen_test_split_manifest["fixed_test_seed"].eq(int(test_seed))
        & frozen_test_split_manifest["development_seed"].eq(int(development_seed))
    ].copy()
    if manifest_group.empty:
        return None

    reference_source_indices = np.asarray(reference_source_indices, dtype=int)
    source_by_hash = {
        str(episode_hash): int(source_index)
        for episode_hash, source_index in zip(
            SPLIT_EPISODE_HASHES.iloc[reference_source_indices].astype(str),
            reference_source_indices,
        )
    }
    expected_hashes = set(manifest_group["episode_hash"].astype(str))
    missing_hashes = expected_hashes - set(source_by_hash)
    maximum_allowed_missing = max(2, int(np.ceil(len(expected_hashes) * 0.02)))
    if len(missing_hashes) > maximum_allowed_missing:
        message = (
            f"Frozen full split mismatch for {target_name}, test_seed={test_seed}, "
            f"development_seed={development_seed}: missing={len(missing_hashes)}."
        )
        if STRICT_FROZEN_TEST_SPLIT_MANIFEST:
            raise ValueError(message)
        warnings.warn(
            message + " Reconstructing development split from the frozen test set."
        )
        return None

    assignment = {}
    for split_name in ["train", "validation", "test"]:
        split_rows = manifest_group.loc[
            manifest_group["dataset_split"].astype(str).eq(split_name)
        ].sort_values("split_order")
        hashes = split_rows["episode_hash"].astype(str)
        assignment[split_name] = np.asarray(
            [source_by_hash[value] for value in hashes if value in source_by_hash],
            dtype=int,
        )

    test_source_set = set(map(int, assignment["test"]))
    source_position = {
        int(source_index): position
        for position, source_index in enumerate(reference_source_indices)
    }
    test_positions = np.asarray(
        [source_position[source_index] for source_index in test_source_set],
        dtype=int,
    )
    test_y = pd.Series(y_reference).iloc[test_positions].astype(int)
    if len(test_y) == 0 or test_y.nunique() < 2:
        raise ValueError(
            f"Frozen full split for {target_name}, seed={test_seed} lacks both test classes."
        )

    audit_key = (str(target_name), int(test_seed))
    FROZEN_TEST_SPLIT_AUDIT[audit_key] = {
        "model_profile": MODEL_PROFILE_NAME,
        "target": str(target_name),
        "fixed_test_seed": int(test_seed),
        "development_seed": int(development_seed),
        "manifest_path": str(FROZEN_TEST_SPLIT_MANIFEST_PATH),
        "manifest_sha256": FROZEN_TEST_SPLIT_MANIFEST_SHA256,
        "manifest_source_data_sha256": str(
            manifest_group["source_data_sha256"].iloc[0]
        ),
        "current_data_sha256": DATA_SHA256,
        "reference_n": int(len(reference_source_indices)),
        "reference_positive_n": int(pd.Series(y_reference).sum()),
        "expected_train_n": int(manifest_group["expected_train_n"].iloc[0]),
        "matched_train_n": int(len(assignment["train"])),
        "expected_validation_n": int(manifest_group["expected_validation_n"].iloc[0]),
        "matched_validation_n": int(len(assignment["validation"])),
        "expected_test_n": int(manifest_group["expected_test_n"].iloc[0]),
        "matched_test_n": int(len(assignment["test"])),
        "expected_test_positive_n": int(
            manifest_group["expected_test_positive_n"].iloc[0]
        ),
        "matched_test_positive_n": int(test_y.sum()),
        "missing_manifest_episode_n": int(len(missing_hashes)),
        "test_labels_used_for_current_split_selection": False,
        "status": "matched" if not missing_hashes else "matched_after_dataset_deletion",
    }
    return assignment


def source_split_for_target(target_name, spec, seed=None):
    if seed is None:
        seed = fixed_split_seed(target_name)
    test_seed = int(seed)
    development_seed = SELECTED_DEVELOPMENT_SPLIT_SEEDS.get(
        target_name,
        DEVELOPMENT_SPLIT_SEED_OVERRIDES.get(target_name),
    )
    effective_development_seed = (
        test_seed + 17 if development_seed is None else int(development_seed)
    )
    cache_key = (
        target_name,
        test_seed,
        effective_development_seed,
        FROZEN_TEST_SPLIT_MANIFEST_SHA256,
    )
    if cache_key in SOURCE_SPLIT_CACHE:
        return SOURCE_SPLIT_CACHE[cache_key]

    keep_mask, y_reference = make_target(
        spec["stage_column"],
        spec["positive_rule"],
        spec.get("exclude_control_death_3m", False),
        spec.get("event_date_column"),
        landmark_day=target_landmark_day_for_policy(0),
    )
    reference_source_indices = np.flatnonzero(keep_mask)
    frozen_complete_split = frozen_complete_source_split(
        target_name,
        test_seed,
        effective_development_seed,
        reference_source_indices,
        y_reference,
    )
    if frozen_complete_split is not None:
        split_info = {
            "split_policy": "frozen_episode_manifest",
            "split_seed": test_seed,
            "test_split_seed": test_seed,
            "development_split_seed": effective_development_seed,
            "reference_n": int(len(y_reference)),
            "reference_positive_n": int(y_reference.sum()),
            "train_source_indices": frozen_complete_split["train"],
            "validation_source_indices": frozen_complete_split["validation"],
            "test_source_indices": frozen_complete_split["test"],
        }
        SOURCE_SPLIT_CACHE[cache_key] = split_info
        return split_info

    frozen_test_sources = frozen_test_source_indices(
        target_name,
        test_seed,
        reference_source_indices,
        y_reference,
    )
    if frozen_test_sources is None:
        base_train_pos, base_val_pos, test_pos = make_train_val_test_split(
            y_reference,
            test_seed,
        )
        development_pos = np.concatenate([base_train_pos, base_val_pos])
        split_policy = "fixed_by_target"
    else:
        frozen_test_source_set = set(map(int, frozen_test_sources))
        test_pos = np.asarray(
            [
                position
                for position, source_index in enumerate(reference_source_indices)
                if int(source_index) in frozen_test_source_set
            ],
            dtype=int,
        )
        development_pos = np.asarray(
            [
                position
                for position, source_index in enumerate(reference_source_indices)
                if int(source_index) not in frozen_test_source_set
            ],
            dtype=int,
        )
        split_policy = "frozen_episode_manifest"

    train_pos, val_pos = train_test_split(
        development_pos,
        train_size=TRAIN_SIZE / (TRAIN_SIZE + VAL_SIZE),
        stratify=y_reference.iloc[development_pos],
        random_state=effective_development_seed,
        shuffle=True,
    )
    train_pos = np.asarray(train_pos)
    val_pos = np.asarray(val_pos)
    split_info = {
        "split_policy": split_policy,
        "split_seed": test_seed,
        "test_split_seed": test_seed,
        "development_split_seed": effective_development_seed,
        "reference_n": int(len(y_reference)),
        "reference_positive_n": int(y_reference.sum()),
        "train_source_indices": reference_source_indices[train_pos],
        "validation_source_indices": reference_source_indices[val_pos],
        "test_source_indices": reference_source_indices[test_pos],
    }
    SOURCE_SPLIT_CACHE[cache_key] = split_info
    return split_info


def train_val_test_positions_for_analysis(
    target_name, spec, feature_name, source_indices, y_target
):
    if SPLIT_POLICY != "fixed_by_target":
        seed = split_seed_for_analysis(target_name, feature_name)
        train_pos, val_pos, test_pos = make_train_val_test_split(y_target, seed)
        return (
            train_pos,
            val_pos,
            test_pos,
            {
                "split_policy": SPLIT_POLICY,
                "split_seed": int(seed),
                "test_split_seed": int(seed),
                "development_split_seed": int(seed) + 17,
                "split_reference_n": int(len(y_target)),
                "split_reference_positive_n": int(pd.Series(y_target).sum()),
            },
        )

    use_feature_fixed_override = has_feature_fixed_split_override(
        target_name, feature_name
    )
    seed = split_seed_for_analysis(target_name, feature_name)
    split_info = source_split_for_target(target_name, spec, seed)
    local_position_by_source = {
        int(source_index): int(local_pos)
        for local_pos, source_index in enumerate(source_indices)
    }

    def local_positions(split_source_indices):
        return np.asarray(
            [
                local_position_by_source[int(source_index)]
                for source_index in split_source_indices
                if int(source_index) in local_position_by_source
            ],
            dtype=int,
        )

    train_pos = local_positions(split_info["train_source_indices"])
    val_pos = local_positions(split_info["validation_source_indices"])
    test_pos = local_positions(split_info["test_source_indices"])
    return (
        train_pos,
        val_pos,
        test_pos,
        {
            "split_policy": (
                f"{split_info['split_policy']}_feature_override"
                if use_feature_fixed_override
                else split_info["split_policy"]
            ),
            "split_seed": int(split_info["split_seed"]),
            "test_split_seed": int(split_info["test_split_seed"]),
            "development_split_seed": int(split_info["development_split_seed"]),
            "split_reference_n": int(split_info["reference_n"]),
            "split_reference_positive_n": int(split_info["reference_positive_n"]),
        },
    )
