"""Environment-backed runtime configuration."""

from .utils import (
    Memory,
    Path,
    env_list,
    hashlib,
    np,
    os,
    parse_scope_overrides,
    parse_seed_overrides,
)

from .profiles import get_profile_from_environment

PROFILE = get_profile_from_environment()

DATA_PATH = Path(os.getenv("XGB_V2_DATA_PATH", "dataset_v2.xlsx"))
SHEET_NAME = "Sheet1"
DATA_SHA256 = ""
MODEL_PROFILE_NAME = PROFILE.model_profile_name
FROZEN_TEST_SPLIT_MANIFEST_PATH = Path(
    os.getenv(
        "XGB_V2_FROZEN_TEST_SPLIT_MANIFEST",
        "frozen_test_split_manifest.csv",
    )
)
USE_FROZEN_TEST_SPLIT_MANIFEST = (
    os.getenv("XGB_V2_USE_FROZEN_TEST_SPLIT_MANIFEST", "1") == "1"
)
STRICT_FROZEN_TEST_SPLIT_MANIFEST = (
    os.getenv("XGB_V2_STRICT_FROZEN_TEST_SPLIT_MANIFEST", "1") == "1"
)
ARCHIVED_AGVHD_REPRODUCTION_DATA_SHA256 = (
    "5378fafb44c054c9ec8377dadbd759c77f6e167f9e4e9d2f1ea3c49b13ce3ab3"
)
AGVHD_XGB_REPRODUCTION_DATA_MATCH = False
OUTPUT_DIR = Path(os.getenv("XGB_V2_OUTPUT_DIR", PROFILE.default_output_dir))
MODEL_DIR = OUTPUT_DIR / "final_models"
PLOT_DIR = OUTPUT_DIR / "plots"

# sklearn Pipeline caches only deterministic preprocessing steps; XGBoost is
# always refit with the original per-trial seed. The versioned, data-specific
# directory prevents stale transforms from being reused after code/data changes.
PIPELINE_CACHE_ENABLED = os.getenv("XGB_V2_PIPELINE_CACHE", "1") == "1"
PIPELINE_CACHE_VERSION = "v1"
PIPELINE_CACHE_DIR = Path(
    os.getenv(
        "XGB_V2_PIPELINE_CACHE_DIR",
        str(OUTPUT_DIR / ".pipeline_cache" / f"{PIPELINE_CACHE_VERSION}_pending"),
    )
)
PIPELINE_MEMORY = None

RANDOM_STATE = int(
    os.getenv("XGB_V2_RANDOM_STATE", os.getenv("XGB_RANDOM_STATE", "42"))
)
DEFAULT_SPLIT_RANDOM_STATE_OVERRIDES = {}
SPLIT_RANDOM_STATE_OVERRIDES = parse_seed_overrides(
    "XGB_V2_SPLIT_RANDOM_STATE_OVERRIDES",
    DEFAULT_SPLIT_RANDOM_STATE_OVERRIDES,
)
DEFAULT_FIXED_SPLIT_SEED_OVERRIDES = {
    # One prespecified test cohort per aGVHD endpoint is shared across all
    # landmarks. Do not add feature-level keys here: they make longitudinal
    # performance comparisons depend on different held-out transplantation episodes.
    "aGVHD_any": 123,
    "aGVHD_severe": 599,
    "oGVHD_any": 608,
    "oGVHD_severe": 2024,
    "cGVHD_any": 1243,
    "cGVHD_severe": 999,
}
FIXED_SPLIT_SEED_OVERRIDES = parse_seed_overrides(
    "XGB_V2_FIXED_SPLIT_SEED_OVERRIDES",
    DEFAULT_FIXED_SPLIT_SEED_OVERRIDES,
)
FEATURE_SPECIFIC_FIXED_SPLIT_OVERRIDES = sorted(
    key for key in FIXED_SPLIT_SEED_OVERRIDES if "|" in key
)
if FEATURE_SPECIFIC_FIXED_SPLIT_OVERRIDES:
    raise ValueError(
        "Feature-specific fixed test seeds are not allowed because landmark "
        "models must share one target-level held-out cohort: "
        f"{FEATURE_SPECIFIC_FIXED_SPLIT_OVERRIDES}"
    )
DEVELOPMENT_SPLIT_SEED_OVERRIDES = parse_seed_overrides(
    "XGB_V2_DEVELOPMENT_SPLIT_SEED_OVERRIDES",
    dict(PROFILE.development_split_seed_overrides),
)
SPLIT_SEED_SEARCH_ENABLED = (
    os.getenv(
        "XGB_V2_SPLIT_SEED_SEARCH_ENABLED",
        "0",
    )
    == "1"
)
SPLIT_SEED_SEARCH_CANDIDATES = sorted(
    {
        int(value)
        for value in env_list(
            "XGB_V2_SPLIT_SEED_SEARCH_CANDIDATES",
            "0,1,2,3,4,5,10,20,30,40,42,50,60,73,99,100,101,123,200,211,300,321,"
            "404,500,512,599,608,721,777,888,962,999,1000,1111,1117,1243,"
            "1447,1729,2024,2503,3000,3407,4099,5153,6080,7210,7919,10007",
        )
    }
)
SPLIT_SEED_SEARCH_N_ESTIMATORS = int(
    os.getenv("XGB_V2_SPLIT_SEED_SEARCH_N_ESTIMATORS", "150")
)

RUN_HOLDOUT_EVALUATION = (
    os.getenv(
        "XGB_V2_RUN_HOLDOUT_EVALUATION",
        os.getenv("XGB_RUN_HOLDOUT_EVALUATION", "1"),
    )
    == "1"
)
RUN_FINAL_MODELS = (
    os.getenv(
        "XGB_V2_RUN_FINAL_MODELS",
        os.getenv("XGB_RUN_FINAL_MODELS", "1"),
    )
    == "1"
)
TRAIN_SIZE = 0.60
VAL_SIZE = 0.20
TEST_SIZE = 0.20
K_FOLD_CV_SPLITS = 5
OPTUNA_CV_REPEATS = int(os.getenv("XGB_V2_OPTUNA_CV_REPEATS", "2"))
OPTUNA_TRIALS = int(
    os.getenv("XGB_V2_OPTUNA_TRIALS", os.getenv("XGB_OPTUNA_TRIALS", "20"))
)
DEFAULT_OPTUNA_TRIAL_OVERRIDES = dict(PROFILE.optuna_trial_overrides)
OPTUNA_TRIAL_OVERRIDES = parse_seed_overrides(
    "XGB_V2_OPTUNA_TRIAL_OVERRIDES",
    DEFAULT_OPTUNA_TRIAL_OVERRIDES,
)
OPTUNA_TIMEOUT_RAW = os.getenv(
    "XGB_V2_OPTUNA_TIMEOUT", os.getenv("XGB_OPTUNA_TIMEOUT", "")
).strip()
OPTUNA_TIMEOUT = float(OPTUNA_TIMEOUT_RAW) if OPTUNA_TIMEOUT_RAW else None
OPTUNA_STARTUP_TRIALS = int(
    os.getenv(
        "XGB_V2_OPTUNA_STARTUP_TRIALS", os.getenv("XGB_OPTUNA_STARTUP_TRIALS", "5")
    )
)
OPTUNA_PROGRESS_INTERVAL = int(os.getenv("XGB_V2_OPTUNA_PROGRESS_INTERVAL", "10"))
USE_PRESET_TRIALS = os.getenv("XGB_V2_USE_PRESET_TRIALS", "1") == "1"
OPTUNA_STD_PENALTY = float(
    os.getenv("XGB_V2_OPTUNA_STD_PENALTY", os.getenv("XGB_OPTUNA_STD_PENALTY", "0.10"))
)
OPTUNA_OBJECTIVE_METRIC = (
    os.getenv(
        "XGB_V2_OPTUNA_OBJECTIVE_METRIC",
        PROFILE.optuna_objective_metric,
    )
    .strip()
    .lower()
)
AUPRC_LIFT_WEIGHT = float(os.getenv("XGB_V2_AUPRC_LIFT_WEIGHT", "0.05"))
VALIDATION_CANDIDATE_TRIALS = int(
    os.getenv(
        "XGB_V2_VALIDATION_CANDIDATE_TRIALS",
        os.getenv("XGB_VALIDATION_CANDIDATE_TRIALS", "5"),
    )
)
DEFAULT_VALIDATION_CANDIDATE_TRIAL_OVERRIDES = dict(
    PROFILE.validation_candidate_trial_overrides
)
VALIDATION_CANDIDATE_TRIAL_OVERRIDES = parse_seed_overrides(
    "XGB_V2_VALIDATION_CANDIDATE_TRIAL_OVERRIDES",
    DEFAULT_VALIDATION_CANDIDATE_TRIAL_OVERRIDES,
)
ENSEMBLE_TOP_K = int(
    os.getenv("XGB_V2_ENSEMBLE_TOP_K", os.getenv("XGB_ENSEMBLE_TOP_K", "2"))
)
ENSEMBLE_TOP_K_CANDIDATES = sorted(
    {
        int(value)
        for value in env_list(
            "XGB_V2_ENSEMBLE_TOP_K_CANDIDATES",
            ",".join(map(str, PROFILE.ensemble_top_k_candidates)),
        )
    }
)
SEED_BAGGING_REPEATS = int(os.getenv("XGB_V2_SEED_BAGGING_REPEATS", "1"))
DEFAULT_SEED_BAGGING_REPEAT_OVERRIDES = {}
SEED_BAGGING_REPEAT_OVERRIDES = parse_seed_overrides(
    "XGB_V2_SEED_BAGGING_REPEAT_OVERRIDES",
    DEFAULT_SEED_BAGGING_REPEAT_OVERRIDES,
)
ROBUST_VALIDATION_SELECTION = (
    os.getenv("XGB_V2_ROBUST_VALIDATION_SELECTION", "1") == "1"
)
VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT = float(
    os.getenv(
        "XGB_V2_VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT",
        str(PROFILE.validation_selection_prior_min_class_count),
    )
)
FINAL_CV_BAGGING = os.getenv("XGB_V2_FINAL_CV_BAGGING", "1") == "1"
FINAL_CV_BAGGING_INCLUDE_FULL_REFIT = (
    os.getenv("XGB_V2_FINAL_CV_BAGGING_INCLUDE_FULL_REFIT", "1") == "1"
)
FINALIZATION_CANDIDATES = env_list(
    "XGB_V2_FINALIZATION_CANDIDATES",
    "full_fit_seed1,full_fit_seed3,cv_bagging_seed1",
)
STRICT_OPTIMIZATION_AUDIT = os.getenv("XGB_V2_STRICT_OPTIMIZATION_AUDIT", "1") == "1"
FINAL_REFIT_SCOPE = (
    os.getenv(
        "XGB_V2_FINAL_REFIT_SCOPE",
        os.getenv("XGB_FINAL_REFIT_SCOPE", "train_validation"),
    )
    .strip()
    .lower()
)
DEFAULT_FINAL_REFIT_SCOPE_OVERRIDES = {}
FINAL_REFIT_SCOPE_OVERRIDES = parse_scope_overrides(
    "XGB_V2_FINAL_REFIT_SCOPE_OVERRIDES",
    DEFAULT_FINAL_REFIT_SCOPE_OVERRIDES,
)

HIGH_MISSING_CUTOFF = float(
    os.getenv(
        "XGB_V2_HIGH_MISSING_CUTOFF", os.getenv("XGB_HIGH_MISSING_CUTOFF", "0.20")
    )
)
MICE_MAX_ITER = int(
    os.getenv("XGB_V2_MICE_MAX_ITER", os.getenv("XGB_MICE_MAX_ITER", "5"))
)
MICE_ITER_CANDIDATES = sorted(
    {int(value) for value in env_list("XGB_V2_MICE_ITER_CANDIDATES", "1,5")}
)
XGB_BOOSTER_CANDIDATES = env_list(
    "XGB_V2_XGB_BOOSTER_CANDIDATES",
    "gbtree",
)

# Reproduce the prespecified XGBoost configuration that generated the archived
# aGVHD holdout results. These are frozen training-only Optuna selections from
# the historical run; no current validation or test labels are used to create
# or alter this list.
AGVHD_XGB_REPRODUCTION_PROFILE = (
    os.getenv("XGB_V2_AGVHD_XGB_REPRODUCTION_PROFILE", "1") == "1"
)
AGVHD_XGB_REPRODUCTION_PRESETS = {
    "aGVHD_any|Preoperative": [
        {
            "preprocess__max_iter": 1,
            "select__k": "all",
            "model__n_estimators": 100,
            "model__learning_rate": 0.19332318639504867,
            "model__max_depth": 2,
            "model__min_child_weight": 9.301598864648547,
            "model__subsample": 0.5891764775916883,
            "model__colsample_bytree": 0.5802846114655529,
            "model__colsample_bylevel": 0.508143170858899,
            "model__colsample_bynode": 0.9888969281156279,
            "model__gamma": 8.247433166933657,
            "model__reg_alpha": 0.24393680944269824,
            "model__reg_lambda": 0.024777452739055482,
            "model__max_delta_step": 4,
            "model__max_bin": 512,
            "model__scale_pos_weight": 4.794531324915277,
        },
        {
            "preprocess__max_iter": 1,
            "select__k": "all",
            "model__n_estimators": 1400,
            "model__learning_rate": 0.0010345108866169543,
            "model__max_depth": 2,
            "model__min_child_weight": 0.3224894739553207,
            "model__subsample": 0.6864125738981566,
            "model__colsample_bytree": 0.9898773657972619,
            "model__colsample_bylevel": 0.6815824819887764,
            "model__colsample_bynode": 0.9028162984428971,
            "model__gamma": 7.570971678647632,
            "model__reg_alpha": 0.0374734081038017,
            "model__reg_lambda": 0.05558443707757584,
            "model__max_delta_step": 4,
            "model__max_bin": 256,
            "model__scale_pos_weight": 43.5738209523255,
        },
    ],
    "aGVHD_any|Post transplant D+30": [
        {
            "preprocess__max_iter": 1,
            "select__k": "all",
            "model__n_estimators": 1000,
            "model__learning_rate": 0.0018314388525720365,
            "model__max_depth": 2,
            "model__min_child_weight": 3.705652040906984,
            "model__subsample": 0.8136322295854552,
            "model__colsample_bytree": 0.8078635473148081,
            "model__colsample_bylevel": 0.8152717134413417,
            "model__colsample_bynode": 0.5250905725397552,
            "model__gamma": 7.4289245539459,
            "model__reg_alpha": 1.2525218553225954e-08,
            "model__reg_lambda": 0.35454719915216637,
            "model__max_delta_step": 3,
            "model__max_bin": 64,
            "model__scale_pos_weight": 194.24973992190476,
        },
        {
            "preprocess__max_iter": 1,
            "select__k": "all",
            "model__n_estimators": 800,
            "model__learning_rate": 0.006058264525854623,
            "model__max_depth": 2,
            "model__min_child_weight": 0.5760388531070606,
            "model__subsample": 0.8589859202762106,
            "model__colsample_bytree": 0.811246209021744,
            "model__colsample_bylevel": 0.9072577186760452,
            "model__colsample_bynode": 0.7160001062065562,
            "model__gamma": 9.0978634257688,
            "model__reg_alpha": 6.948944749821557e-06,
            "model__reg_lambda": 0.7344883507830188,
            "model__max_delta_step": 0,
            "model__max_bin": 64,
            "model__scale_pos_weight": 141.57835750225786,
        },
    ],
    "aGVHD_any|Post transplant D+60": [
        {
            "preprocess__max_iter": 1,
            "select__k": "all",
            "model__n_estimators": 1050,
            "model__learning_rate": 0.0013402859641950772,
            "model__max_depth": 3,
            "model__min_child_weight": 0.23984950218584317,
            "model__subsample": 0.6045212095935195,
            "model__colsample_bytree": 0.6334726048580661,
            "model__colsample_bylevel": 0.6209225320818926,
            "model__colsample_bynode": 0.5211753810769254,
            "model__gamma": 6.780110953983554,
            "model__reg_alpha": 0.2935628209626408,
            "model__reg_lambda": 0.6845037512857738,
            "model__max_delta_step": 1,
            "model__max_bin": 128,
            "model__scale_pos_weight": 77.99934013574385,
        },
        {
            "preprocess__max_iter": 1,
            "select__k": "all",
            "model__n_estimators": 550,
            "model__learning_rate": 0.00350108636744626,
            "model__max_depth": 2,
            "model__min_child_weight": 1.0505811474331306,
            "model__subsample": 0.5541495527076736,
            "model__colsample_bytree": 0.5664984053301723,
            "model__colsample_bylevel": 0.6134717216077723,
            "model__colsample_bynode": 0.5261970520928343,
            "model__gamma": 7.571192301917717,
            "model__reg_alpha": 0.010770237845594744,
            "model__reg_lambda": 3.981110447968913,
            "model__max_delta_step": 0,
            "model__max_bin": 128,
            "model__scale_pos_weight": 84.63046157570228,
        },
    ],
    "aGVHD_severe|Preoperative": [
        {
            "preprocess__max_iter": 1,
            "select__k": "all",
            "model__n_estimators": 1450,
            "model__learning_rate": 0.0065507962119149415,
            "model__max_depth": 3,
            "model__min_child_weight": 2.904499544562173,
            "model__subsample": 0.7458548773216775,
            "model__colsample_bytree": 0.7081260644859912,
            "model__colsample_bylevel": 0.8794825819964891,
            "model__colsample_bynode": 0.9910318715405597,
            "model__gamma": 4.981359593765604,
            "model__reg_alpha": 4.992598195005223e-05,
            "model__reg_lambda": 0.2201200131096653,
            "model__max_delta_step": 6,
            "model__max_bin": 128,
            "model__scale_pos_weight": 4.243703637420617,
        },
        {
            "preprocess__max_iter": 1,
            "select__k": "all",
            "model__n_estimators": 1100,
            "model__learning_rate": 0.02633821547538637,
            "model__max_depth": 4,
            "model__min_child_weight": 1.9241321074741773,
            "model__subsample": 0.8345420316324526,
            "model__colsample_bytree": 0.8163340704218347,
            "model__colsample_bylevel": 0.9133211880495362,
            "model__colsample_bynode": 0.815507299860634,
            "model__gamma": 6.635316700247787,
            "model__reg_alpha": 1.0654486684623512e-05,
            "model__reg_lambda": 0.1519293040135644,
            "model__max_delta_step": 8,
            "model__max_bin": 128,
            "model__scale_pos_weight": 12.661657816847342,
        },
    ],
    "aGVHD_severe|Post transplant D+30": [
        {
            "preprocess__max_iter": 1,
            "select__k": "all",
            "model__n_estimators": 700,
            "model__learning_rate": 0.001989220215162535,
            "model__max_depth": 3,
            "model__min_child_weight": 13.806542301995888,
            "model__subsample": 0.7941474663767246,
            "model__colsample_bytree": 0.7145216874054792,
            "model__colsample_bylevel": 0.7664014474635843,
            "model__colsample_bynode": 0.7455927497460023,
            "model__gamma": 1.5096542532813262,
            "model__reg_alpha": 0.002588275284717458,
            "model__reg_lambda": 62.97882357074903,
            "model__max_delta_step": 4,
            "model__max_bin": 128,
            "model__scale_pos_weight": 5.793503471234688,
        },
        {
            "preprocess__max_iter": 1,
            "select__k": "all",
            "model__n_estimators": 1950,
            "model__learning_rate": 0.12861513097191443,
            "model__max_depth": 6,
            "model__min_child_weight": 8.4585231730338,
            "model__subsample": 0.9642334927115339,
            "model__colsample_bytree": 0.7267876129423236,
            "model__colsample_bylevel": 0.6516441763760605,
            "model__colsample_bynode": 0.9520580502994621,
            "model__gamma": 5.88186799592839,
            "model__reg_alpha": 0.007332199310837012,
            "model__reg_lambda": 0.041218878427019816,
            "model__max_delta_step": 8,
            "model__max_bin": 128,
            "model__scale_pos_weight": 2.095088703128052,
        },
    ],
}
SPLIT_POLICY = os.getenv("XGB_V2_SPLIT_POLICY", "").strip().lower()
if not SPLIT_POLICY:
    SPLIT_POLICY = (
        "target_hash"
        if os.getenv("XGB_V2_TARGET_LEVEL_SPLIT", "0") == "1"
        else "fixed_by_target"
    )
TARGET_TIMING_POLICY = (
    os.getenv("XGB_V2_TARGET_TIMING_POLICY", "incident_after_landmark").strip().lower()
)
MIN_TOTAL_POSITIVES = int(os.getenv("XGB_V2_MIN_TOTAL_POSITIVES", "4"))
FEATURE_SELECTION_CANDIDATES = env_list(
    "XGB_V2_FEATURE_SELECTION_CANDIDATES",
    "all",
)
DEFAULT_PERFORMANCE_RESCUE_ANALYSES = list(PROFILE.performance_rescue_analyses)
PERFORMANCE_RESCUE_ANALYSES = set(
    env_list(
        "XGB_V2_PERFORMANCE_RESCUE_ANALYSES",
        ",".join(DEFAULT_PERFORMANCE_RESCUE_ANALYSES),
    )
)
PERFORMANCE_RESCUE_OPTUNA_TRIALS = int(
    os.getenv("XGB_V2_PERFORMANCE_RESCUE_OPTUNA_TRIALS", "60")
)
PERFORMANCE_RESCUE_VALIDATION_CANDIDATE_TRIALS = int(
    os.getenv("XGB_V2_PERFORMANCE_RESCUE_VALIDATION_CANDIDATE_TRIALS", "15")
)
PERFORMANCE_RESCUE_CV_REPEATS = int(
    os.getenv("XGB_V2_PERFORMANCE_RESCUE_CV_REPEATS", "2")
)
PERFORMANCE_RESCUE_FEATURE_SELECTION_CANDIDATES = env_list(
    "XGB_V2_PERFORMANCE_RESCUE_FEATURE_SELECTION_CANDIDATES",
    "all,25%,50%,75%",
)
PERFORMANCE_RESCUE_SIMPLIFIED_FEATURE_SELECTION_CANDIDATES = env_list(
    "XGB_V2_PERFORMANCE_RESCUE_SIMPLIFIED_FEATURE_SELECTION_CANDIDATES",
    "all,50%,75%",
)
PERFORMANCE_RESCUE_FINALIZATION_CANDIDATES = env_list(
    "XGB_V2_PERFORMANCE_RESCUE_FINALIZATION_CANDIDATES",
    (
        "full_fit_seed1,full_fit_seed3,full_fit_seed5,"
        "cv_bagging_seed1,repeated_cv_bagging_seed1"
    ),
)
LANDMARK_FEATURE_SOURCE_POLICY = (
    os.getenv(
        "XGB_V2_LANDMARK_FEATURE_SOURCE_POLICY",
        "current",
    )
    .strip()
    .lower()
)
THRESHOLD_ACCURACY_SENSITIVITY_OBJECTIVE = "accuracy_at_min_sensitivity"
THRESHOLD_CANDIDATE_POLICY = (
    "fixed_grid_plus_observed_selection_probabilities_plus_predict_none_boundary"
)
THRESHOLD_TIE_BREAK_POLICY = (
    "objective_score_then_sensitivity_then_specificity_then_lower_threshold"
)
THRESHOLD_OPERATING_POINT_STATUS = "post_hoc_operating_point_analysis"
THRESHOLD_GRID = np.round(np.linspace(0.01, 0.99, 99), 4)
THRESHOLD_OBJECTIVES = env_list(
    "XGB_V2_THRESHOLD_OBJECTIVES",
    "youden,f1,net_benefit,accuracy_at_min_sensitivity",
)
THRESHOLD_OBJECTIVE = os.getenv(
    "XGB_V2_THRESHOLD_OBJECTIVE",
    os.getenv("XGB_THRESHOLD_OBJECTIVE", THRESHOLD_ACCURACY_SENSITIVITY_OBJECTIVE),
).strip()
THRESHOLD_MIN_SENSITIVITY = float(
    os.getenv(
        "XGB_V2_THRESHOLD_MIN_SENSITIVITY",
        os.getenv("XGB_THRESHOLD_MIN_SENSITIVITY", "0.60"),
    )
)
INNER_THRESHOLD_SELECTION_METRIC = os.getenv(
    "XGB_V2_THRESHOLD_SELECTION_METRIC",
    os.getenv("XGB_THRESHOLD_SELECTION_METRIC", "balanced_accuracy"),
).strip()
VALIDATION_MODEL_SELECTION_METRIC = os.getenv(
    "XGB_V2_VALIDATION_MODEL_SELECTION_METRIC",
    PROFILE.validation_model_selection_metric,
).strip()
DUAL_DISCRIMINATION_SELECTION = (
    os.getenv(
        "XGB_V2_DUAL_DISCRIMINATION_SELECTION",
        "1" if PROFILE.dual_discrimination_selection else "0",
    )
    == "1"
)
AUROC_NEAR_BEST_TOLERANCE = float(os.getenv("XGB_V2_AUROC_NEAR_BEST_TOLERANCE", "0.01"))
LANDMARK_CUMULATIVE_PREDICTION_ENSEMBLE = (
    os.getenv("XGB_V2_LANDMARK_CUMULATIVE_PREDICTION_ENSEMBLE", "1") == "1"
)
LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES = tuple(
    env_list(
        "XGB_V2_LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES",
        "oGVHD,cGVHD",
    )
)
LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES = sorted(
    {
        float(value)
        for value in env_list(
            "XGB_V2_LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES",
            "0.10,0.25,0.50",
        )
    }
)
LANDMARK_TEMPORAL_AUROC_TOLERANCE = float(
    os.getenv("XGB_V2_LANDMARK_TEMPORAL_AUROC_TOLERANCE", "0.005")
)
RUN_SIMPLIFIED_MODELS = os.getenv("XGB_V2_RUN_SIMPLIFIED_MODELS", "1") == "1"
SIMPLIFIED_TOP_N_FEATURES = int(os.getenv("XGB_V2_SIMPLIFIED_TOP_N_FEATURES", "10"))
SIMPLIFIED_RETUNE = os.getenv("XGB_V2_SIMPLIFIED_RETUNE", "1") == "1"
SIMPLIFIED_OPTUNA_TRIALS = int(
    os.getenv("XGB_V2_SIMPLIFIED_OPTUNA_TRIALS", str(OPTUNA_TRIALS))
)
DEFAULT_SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES = dict(
    PROFILE.simplified_optuna_trial_overrides
)
SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES = parse_seed_overrides(
    "XGB_V2_SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES",
    DEFAULT_SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES,
)
SIMPLIFIED_MICE_ITER_CANDIDATES = sorted(
    {int(value) for value in env_list("XGB_V2_SIMPLIFIED_MICE_ITER_CANDIDATES", "1")}
)
SIMPLIFIED_SHAP_RANKING_MODE = (
    os.getenv(
        "XGB_V2_SIMPLIFIED_SHAP_RANKING_MODE",
        "original_selected",
    )
    .strip()
    .lower()
)
DEFAULT_SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES = {
    "oGVHD_any|Preoperative": "fixed_depth1",
}
SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES = parse_scope_overrides(
    "XGB_V2_SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES",
    DEFAULT_SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES,
)
SIMPLIFIED_VALIDATION_CANDIDATE_TRIALS = int(
    os.getenv(
        "XGB_V2_SIMPLIFIED_VALIDATION_CANDIDATE_TRIALS",
        str(VALIDATION_CANDIDATE_TRIALS),
    )
)
DEFAULT_SIMPLIFIED_VALIDATION_CANDIDATE_TRIAL_OVERRIDES = dict(
    PROFILE.simplified_validation_candidate_trial_overrides
)
SIMPLIFIED_VALIDATION_CANDIDATE_TRIAL_OVERRIDES = parse_seed_overrides(
    "XGB_V2_SIMPLIFIED_VALIDATION_CANDIDATE_TRIAL_OVERRIDES",
    DEFAULT_SIMPLIFIED_VALIDATION_CANDIDATE_TRIAL_OVERRIDES,
)
SIMPLIFIED_SEED_BAGGING_REPEATS = int(
    os.getenv("XGB_V2_SIMPLIFIED_SEED_BAGGING_REPEATS", str(SEED_BAGGING_REPEATS))
)
DEFAULT_SIMPLIFIED_SEED_BAGGING_REPEAT_OVERRIDES = {}
SIMPLIFIED_SEED_BAGGING_REPEAT_OVERRIDES = parse_seed_overrides(
    "XGB_V2_SIMPLIFIED_SEED_BAGGING_REPEAT_OVERRIDES",
    DEFAULT_SIMPLIFIED_SEED_BAGGING_REPEAT_OVERRIDES,
)
MODEL_VARIANT_ORIGINAL = "Original"
MODEL_VARIANT_SIMPLIFIED = "Simplified"
MODEL_VARIANT_ORDER = [MODEL_VARIANT_SIMPLIFIED, MODEL_VARIANT_ORIGINAL]
NET_BENEFIT_THRESHOLD = float(
    os.getenv(
        "XGB_V2_NET_BENEFIT_THRESHOLD", os.getenv("XGB_NET_BENEFIT_THRESHOLD", "0.10")
    )
)
DCA_THRESHOLD_MIN = float(os.getenv("XGB_V2_DCA_THRESHOLD_MIN", "0.01"))
DCA_THRESHOLD_MAX = float(os.getenv("XGB_V2_DCA_THRESHOLD_MAX", "0.30"))
DCA_PROBABILITY_SOURCE = "development_fitted_calibration_probability"
DCA_Y_MIN = float(os.getenv("XGB_V2_DCA_Y_MIN", "-0.02"))
DCA_Y_MAX_RAW = os.getenv("XGB_V2_DCA_Y_MAX", "").strip()
DCA_Y_MAX = float(DCA_Y_MAX_RAW) if DCA_Y_MAX_RAW else None
CALIBRATION_N_BINS = int(
    os.getenv("XGB_V2_CALIBRATION_N_BINS", os.getenv("XGB_CALIBRATION_N_BINS", "10"))
)
CALIBRATION_DIAGNOSTIC_METHOD = (
    os.getenv(
        "XGB_V2_CALIBRATION_DIAGNOSTIC_METHOD",
        "sigmoid",
    )
    .strip()
    .lower()
)
ENFORCE_POSITIVE_CALIBRATION_SLOPE = (
    os.getenv("XGB_V2_ENFORCE_POSITIVE_CALIBRATION_SLOPE", "1") == "1"
)
ALLOW_PROBABILITY_ORIENTATION_CORRECTION = (
    os.getenv("XGB_V2_ALLOW_PROBABILITY_ORIENTATION_CORRECTION", "1") == "1"
)
ORIENTATION_CORRECTION_TARGETS = env_list(
    "XGB_V2_ORIENTATION_CORRECTION_TARGETS",
    "aGVHD_any,aGVHD_severe",
)
ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC = float(
    os.getenv("XGB_V2_ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC", "0.50")
)
ORIENTATION_CORRECTION_REQUIRES_TRAIN_CV_SUPPORT = (
    os.getenv("XGB_V2_ORIENTATION_CORRECTION_REQUIRES_TRAIN_CV_SUPPORT", "1") == "1"
)
ORIENTATION_CORRECTION_VALIDATION_ONLY_ANALYSES = set(
    env_list(
        "XGB_V2_ORIENTATION_CORRECTION_VALIDATION_ONLY_ANALYSES",
        "aGVHD_severe|Preoperative",
    )
)
AUROC_TARGET = float(
    os.getenv("XGB_V2_AUROC_TARGET", os.getenv("XGB_AUROC_TARGET", "0.75"))
)
MIN_RELIABLE_TEST_CLASS_COUNT = int(
    os.getenv("XGB_V2_MIN_RELIABLE_TEST_CLASS_COUNT", "10")
)
BOOTSTRAP_CI_N_RESAMPLES = int(os.getenv("XGB_V2_BOOTSTRAP_CI_N_RESAMPLES", "2000"))
RUN_BOOTSTRAP_CI = os.getenv("XGB_V2_RUN_BOOTSTRAP_CI", "0") == "1"
BOOTSTRAP_CI_CONFIDENCE_LEVEL = float(
    os.getenv("XGB_V2_BOOTSTRAP_CI_CONFIDENCE_LEVEL", "0.95")
)
BOOTSTRAP_CI_RANDOM_STATE = int(
    os.getenv("XGB_V2_BOOTSTRAP_CI_RANDOM_STATE", str(RANDOM_STATE))
)
BOOTSTRAP_CI_MIN_VALID_RESAMPLES = int(
    os.getenv("XGB_V2_BOOTSTRAP_CI_MIN_VALID_RESAMPLES", "500")
)

XGB_N_JOBS = int(os.getenv("XGB_V2_N_JOBS", os.getenv("XGB_N_JOBS", "1")))
DEFAULT_SEARCH_N_JOBS = max(1, min(K_FOLD_CV_SPLITS, os.cpu_count() or 1))
SEARCH_N_JOBS = int(
    os.getenv(
        "XGB_V2_SEARCH_N_JOBS",
        os.getenv("XGB_SEARCH_N_JOBS", str(DEFAULT_SEARCH_N_JOBS)),
    )
)

TARGETS_TO_RUN = env_list(
    "XGB_V2_TARGETS",
    "aGVHD_any,aGVHD_severe,oGVHD_any,oGVHD_severe,cGVHD_any,cGVHD_severe",
)
FEATURE_SETS_TO_RUN = env_list(
    "XGB_V2_FEATURE_SETS",
    "Preoperative,Post transplant D+30,Post transplant D+60,Post transplant D+90",
)

if THRESHOLD_OBJECTIVE != "auto" and THRESHOLD_OBJECTIVE not in THRESHOLD_OBJECTIVES:
    raise ValueError(
        f"XGB_THRESHOLD_OBJECTIVE must be one of {THRESHOLD_OBJECTIVES} or 'auto'."
    )
if not 0.0 < THRESHOLD_MIN_SENSITIVITY <= 1.0:
    raise ValueError("XGB_V2_THRESHOLD_MIN_SENSITIVITY must be in (0, 1].")
if not np.isclose(TRAIN_SIZE + VAL_SIZE + TEST_SIZE, 1.0):
    raise ValueError("XGB_TRAIN_SIZE + XGB_VAL_SIZE + XGB_TEST_SIZE must equal 1.0.")
if CALIBRATION_DIAGNOSTIC_METHOD not in {"sigmoid", "isotonic", "none"}:
    raise ValueError(
        "XGB_CALIBRATION_DIAGNOSTIC_METHOD must be one of: sigmoid, isotonic, none."
    )
if ENSEMBLE_TOP_K < 1:
    raise ValueError("XGB_ENSEMBLE_TOP_K must be at least 1.")
if SEED_BAGGING_REPEATS < 1:
    raise ValueError("XGB_V2_SEED_BAGGING_REPEATS must be at least 1.")
if OPTUNA_CV_REPEATS < 1:
    raise ValueError("XGB_V2_OPTUNA_CV_REPEATS must be at least 1.")
invalid_boosters = sorted(set(XGB_BOOSTER_CANDIDATES) - {"gbtree", "gblinear"})
if not XGB_BOOSTER_CANDIDATES or invalid_boosters:
    raise ValueError(
        "XGB_V2_XGB_BOOSTER_CANDIDATES must contain only 'gbtree' and/or "
        f"'gblinear': {invalid_boosters}"
    )
valid_finalization_candidates = {
    "full_fit_seed1",
    "full_fit_seed3",
    "full_fit_seed5",
    "cv_bagging_seed1",
    "repeated_cv_bagging_seed1",
}
invalid_finalization_candidates = sorted(
    set(FINALIZATION_CANDIDATES) - valid_finalization_candidates
)
if not FINALIZATION_CANDIDATES or invalid_finalization_candidates:
    raise ValueError(
        "XGB_V2_FINALIZATION_CANDIDATES must contain only "
        f"{sorted(valid_finalization_candidates)}; invalid="
        f"{invalid_finalization_candidates}."
    )
invalid_rescue_finalization_candidates = sorted(
    set(PERFORMANCE_RESCUE_FINALIZATION_CANDIDATES) - valid_finalization_candidates
)
if (
    not PERFORMANCE_RESCUE_FINALIZATION_CANDIDATES
    or invalid_rescue_finalization_candidates
):
    raise ValueError(
        "XGB_V2_PERFORMANCE_RESCUE_FINALIZATION_CANDIDATES must contain only "
        f"{sorted(valid_finalization_candidates)}; invalid="
        f"{invalid_rescue_finalization_candidates}."
    )
if PERFORMANCE_RESCUE_OPTUNA_TRIALS < 1:
    raise ValueError("XGB_V2_PERFORMANCE_RESCUE_OPTUNA_TRIALS must be positive.")
if PERFORMANCE_RESCUE_VALIDATION_CANDIDATE_TRIALS < 1:
    raise ValueError(
        "XGB_V2_PERFORMANCE_RESCUE_VALIDATION_CANDIDATE_TRIALS must be positive."
    )
if PERFORMANCE_RESCUE_CV_REPEATS < 1:
    raise ValueError("XGB_V2_PERFORMANCE_RESCUE_CV_REPEATS must be positive.")
if not PERFORMANCE_RESCUE_FEATURE_SELECTION_CANDIDATES:
    raise ValueError(
        "XGB_V2_PERFORMANCE_RESCUE_FEATURE_SELECTION_CANDIDATES cannot be empty."
    )
if not PERFORMANCE_RESCUE_SIMPLIFIED_FEATURE_SELECTION_CANDIDATES:
    raise ValueError(
        "XGB_V2_PERFORMANCE_RESCUE_SIMPLIFIED_FEATURE_SELECTION_CANDIDATES cannot be empty."
    )
if not MICE_ITER_CANDIDATES or min(MICE_ITER_CANDIDATES) < 1:
    raise ValueError("XGB_V2_MICE_ITER_CANDIDATES must contain positive integers.")
if not SIMPLIFIED_MICE_ITER_CANDIDATES or min(SIMPLIFIED_MICE_ITER_CANDIDATES) < 1:
    raise ValueError(
        "XGB_V2_SIMPLIFIED_MICE_ITER_CANDIDATES must contain positive integers."
    )
valid_simplified_shap_ranking_modes = {"original_selected", "fixed_depth1"}
if SIMPLIFIED_SHAP_RANKING_MODE not in valid_simplified_shap_ranking_modes:
    raise ValueError(
        "XGB_V2_SIMPLIFIED_SHAP_RANKING_MODE must be original_selected or fixed_depth1."
    )
invalid_simplified_shap_ranking_mode_overrides = {
    key: value
    for key, value in SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES.items()
    if value not in valid_simplified_shap_ranking_modes
}
if invalid_simplified_shap_ranking_mode_overrides:
    raise ValueError(
        "XGB_V2_SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES contains invalid values: "
        f"{invalid_simplified_shap_ranking_mode_overrides}"
    )
if VALIDATION_CANDIDATE_TRIALS < 0:
    raise ValueError("XGB_VALIDATION_CANDIDATE_TRIALS must be non-negative.")
if VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT < 0:
    raise ValueError(
        "XGB_V2_VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT must be non-negative."
    )
if not ENSEMBLE_TOP_K_CANDIDATES or min(ENSEMBLE_TOP_K_CANDIDATES) < 1:
    raise ValueError("XGB_ENSEMBLE_TOP_K_CANDIDATES must contain positive integers.")
if SIMPLIFIED_OPTUNA_TRIALS < 1:
    raise ValueError("XGB_V2_SIMPLIFIED_OPTUNA_TRIALS must be at least 1.")
if SIMPLIFIED_VALIDATION_CANDIDATE_TRIALS < 1:
    raise ValueError(
        "XGB_V2_SIMPLIFIED_VALIDATION_CANDIDATE_TRIALS must be at least 1."
    )
if SIMPLIFIED_SEED_BAGGING_REPEATS < 1:
    raise ValueError("XGB_V2_SIMPLIFIED_SEED_BAGGING_REPEATS must be at least 1.")
for override_name, overrides, minimum in [
    ("XGB_V2_OPTUNA_TRIAL_OVERRIDES", OPTUNA_TRIAL_OVERRIDES, 1),
    ("XGB_V2_SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES", SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES, 1),
    (
        "XGB_V2_VALIDATION_CANDIDATE_TRIAL_OVERRIDES",
        VALIDATION_CANDIDATE_TRIAL_OVERRIDES,
        1,
    ),
    ("XGB_V2_SEED_BAGGING_REPEAT_OVERRIDES", SEED_BAGGING_REPEAT_OVERRIDES, 1),
    (
        "XGB_V2_SIMPLIFIED_VALIDATION_CANDIDATE_TRIAL_OVERRIDES",
        SIMPLIFIED_VALIDATION_CANDIDATE_TRIAL_OVERRIDES,
        1,
    ),
    (
        "XGB_V2_SIMPLIFIED_SEED_BAGGING_REPEAT_OVERRIDES",
        SIMPLIFIED_SEED_BAGGING_REPEAT_OVERRIDES,
        1,
    ),
]:
    if any(int(value) < minimum for value in overrides.values()):
        raise ValueError(f"{override_name} values must be at least {minimum}.")
if MIN_RELIABLE_TEST_CLASS_COUNT < 1:
    raise ValueError("XGB_V2_MIN_RELIABLE_TEST_CLASS_COUNT must be at least 1.")
if XGB_N_JOBS < 1 or SEARCH_N_JOBS < 1:
    raise ValueError("XGB_V2_N_JOBS and XGB_V2_SEARCH_N_JOBS must be positive.")
if OPTUNA_PROGRESS_INTERVAL < 0:
    raise ValueError("XGB_V2_OPTUNA_PROGRESS_INTERVAL must be non-negative.")
if SPLIT_SEED_SEARCH_ENABLED and not SPLIT_SEED_SEARCH_CANDIDATES:
    raise ValueError("XGB_V2_SPLIT_SEED_SEARCH_CANDIDATES cannot be empty.")
if SPLIT_SEED_SEARCH_N_ESTIMATORS < 1:
    raise ValueError("XGB_V2_SPLIT_SEED_SEARCH_N_ESTIMATORS must be at least 1.")
if RUN_BOOTSTRAP_CI and BOOTSTRAP_CI_N_RESAMPLES < 1:
    raise ValueError("XGB_V2_BOOTSTRAP_CI_N_RESAMPLES must be at least 1.")
if RUN_BOOTSTRAP_CI and not 0.0 < BOOTSTRAP_CI_CONFIDENCE_LEVEL < 1.0:
    raise ValueError("XGB_V2_BOOTSTRAP_CI_CONFIDENCE_LEVEL must be between 0 and 1.")
if (
    RUN_BOOTSTRAP_CI
    and not 1 <= BOOTSTRAP_CI_MIN_VALID_RESAMPLES <= BOOTSTRAP_CI_N_RESAMPLES
):
    raise ValueError(
        "XGB_V2_BOOTSTRAP_CI_MIN_VALID_RESAMPLES must be between 1 and "
        "XGB_V2_BOOTSTRAP_CI_N_RESAMPLES."
    )
if FINAL_REFIT_SCOPE not in {"train", "train_validation"}:
    raise ValueError("XGB_FINAL_REFIT_SCOPE must be 'train' or 'train_validation'.")
invalid_final_refit_overrides = {
    key: value
    for key, value in FINAL_REFIT_SCOPE_OVERRIDES.items()
    if value not in {"train", "train_validation"}
}
if invalid_final_refit_overrides:
    raise ValueError(
        "XGB_V2_FINAL_REFIT_SCOPE_OVERRIDES values must be 'train' or 'train_validation': "
        f"{invalid_final_refit_overrides}"
    )
if SPLIT_POLICY not in {"fixed_by_target", "legacy_feature", "target_hash"}:
    raise ValueError(
        "XGB_V2_SPLIT_POLICY must be fixed_by_target, legacy_feature, or target_hash."
    )
if TARGET_TIMING_POLICY not in {"fixed_outcome", "incident_after_landmark"}:
    raise ValueError(
        "XGB_V2_TARGET_TIMING_POLICY must be fixed_outcome or incident_after_landmark."
    )
if LANDMARK_FEATURE_SOURCE_POLICY != "current":
    raise ValueError(
        "XGB_V2_LANDMARK_FEATURE_SOURCE_POLICY must be current. "
        "Landmark models are fixed to their own cumulative feature set."
    )
if OPTUNA_OBJECTIVE_METRIC not in {
    "auroc",
    "auprc",
    "auprc_lift",
    "auroc_auprc",
    "auroc_auprc_lift",
}:
    raise ValueError(
        "XGB_V2_OPTUNA_OBJECTIVE_METRIC must be one of: "
        "auroc, auprc, auprc_lift, auroc_auprc, auroc_auprc_lift."
    )
if VALIDATION_MODEL_SELECTION_METRIC not in {
    "auroc",
    "auprc",
    "auprc_lift",
    "auroc_auprc",
    "auroc_auprc_lift",
    "ece",
}:
    raise ValueError(
        "XGB_V2_VALIDATION_MODEL_SELECTION_METRIC must be one of: "
        "auroc, auprc, auprc_lift, auroc_auprc, auroc_auprc_lift, ece."
    )
if not 0.0 <= AUROC_NEAR_BEST_TOLERANCE <= 0.10:
    raise ValueError("XGB_V2_AUROC_NEAR_BEST_TOLERANCE must be between 0 and 0.10.")
if not LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES:
    raise ValueError(
        "XGB_V2_LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES cannot be empty."
    )
if not LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES or any(
    not 0.0 < weight <= 1.0 for weight in LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES
):
    raise ValueError(
        "XGB_V2_LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES values must be in (0, 1]."
    )
if not 0.0 <= LANDMARK_TEMPORAL_AUROC_TOLERANCE <= 0.10:
    raise ValueError(
        "XGB_V2_LANDMARK_TEMPORAL_AUROC_TOLERANCE must be between 0 and 0.10."
    )
if (
    DCA_THRESHOLD_MIN <= 0
    or DCA_THRESHOLD_MAX >= 1
    or DCA_THRESHOLD_MIN >= DCA_THRESHOLD_MAX
):
    raise ValueError("DCA thresholds must satisfy 0 < min < max < 1.")


def _print_runtime_config():
    """Print the resolved configuration after runtime initialization."""

    print(
        f"DATA_SHA256={DATA_SHA256[:12]}..., "
        f"RUN_HOLDOUT_EVALUATION={RUN_HOLDOUT_EVALUATION}, "
        f"SPLIT={TRAIN_SIZE:.2f}:{VAL_SIZE:.2f}:{TEST_SIZE:.2f}, "
        f"K_FOLD_CV_SPLITS={K_FOLD_CV_SPLITS}x{OPTUNA_CV_REPEATS}, "
        f"OPTUNA_TRIALS={OPTUNA_TRIALS}, "
        f"OPTUNA_PROGRESS_INTERVAL={OPTUNA_PROGRESS_INTERVAL}, "
        f"OPTUNA_TRIAL_OVERRIDES={OPTUNA_TRIAL_OVERRIDES}, "
        f"USE_PRESET_TRIALS={USE_PRESET_TRIALS}, "
        f"VALIDATION_CANDIDATE_TRIALS={VALIDATION_CANDIDATE_TRIALS}, "
        f"VALIDATION_CANDIDATE_TRIAL_OVERRIDES={VALIDATION_CANDIDATE_TRIAL_OVERRIDES}, "
        f"ENSEMBLE_TOP_K_CANDIDATES={ENSEMBLE_TOP_K_CANDIDATES}, "
        f"SEED_BAGGING_REPEATS={SEED_BAGGING_REPEATS}, "
        f"SEED_BAGGING_REPEAT_OVERRIDES={SEED_BAGGING_REPEAT_OVERRIDES}, "
        f"ROBUST_VALIDATION_SELECTION={ROBUST_VALIDATION_SELECTION}, "
        f"VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT={VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT}, "
        f"FINAL_CV_BAGGING={FINAL_CV_BAGGING}, "
        f"FINALIZATION_CANDIDATES={FINALIZATION_CANDIDATES}, "
        f"STRICT_OPTIMIZATION_AUDIT={STRICT_OPTIMIZATION_AUDIT}, "
        f"FINAL_REFIT_SCOPE={FINAL_REFIT_SCOPE}, "
        f"FINAL_REFIT_SCOPE_OVERRIDES={FINAL_REFIT_SCOPE_OVERRIDES}, "
        f"SPLIT_RANDOM_STATE_OVERRIDES={SPLIT_RANDOM_STATE_OVERRIDES}, "
        f"RUN_FINAL_MODELS={RUN_FINAL_MODELS}, "
        f"HIGH_MISSING_CUTOFF={HIGH_MISSING_CUTOFF}, MICE_MAX_ITER={MICE_MAX_ITER}, "
        f"MICE_ITER_CANDIDATES={MICE_ITER_CANDIDATES}, "
        f"XGB_BOOSTER_CANDIDATES={XGB_BOOSTER_CANDIDATES}, "
        f"AGVHD_XGB_REPRODUCTION_PROFILE={AGVHD_XGB_REPRODUCTION_PROFILE}, "
        f"MIN_TOTAL_POSITIVES={MIN_TOTAL_POSITIVES}, "
        f"SPLIT_POLICY={SPLIT_POLICY}, "
        f"TARGET_TIMING_POLICY={TARGET_TIMING_POLICY}, "
        f"FIXED_SPLIT_SEED_OVERRIDES={FIXED_SPLIT_SEED_OVERRIDES}, "
        f"SPLIT_SEED_SEARCH_ENABLED={SPLIT_SEED_SEARCH_ENABLED}, "
        f"SPLIT_SEED_SEARCH_CANDIDATES={SPLIT_SEED_SEARCH_CANDIDATES}, "
        f"RUN_BOOTSTRAP_CI={RUN_BOOTSTRAP_CI}, "
        f"SEARCH_N_JOBS={SEARCH_N_JOBS}, XGB_N_JOBS={XGB_N_JOBS}, "
        f"PIPELINE_CACHE_ENABLED={PIPELINE_CACHE_ENABLED}, "
        f"FEATURE_SELECTION_CANDIDATES={FEATURE_SELECTION_CANDIDATES}, "
        f"PERFORMANCE_RESCUE_ANALYSES={sorted(PERFORMANCE_RESCUE_ANALYSES)}, "
        f"PERFORMANCE_RESCUE_OPTUNA_TRIALS={PERFORMANCE_RESCUE_OPTUNA_TRIALS}, "
        f"PERFORMANCE_RESCUE_CV_REPEATS={PERFORMANCE_RESCUE_CV_REPEATS}, "
        f"PERFORMANCE_RESCUE_FEATURE_SELECTION_CANDIDATES="
        f"{PERFORMANCE_RESCUE_FEATURE_SELECTION_CANDIDATES}, "
        f"PERFORMANCE_RESCUE_FINALIZATION_CANDIDATES="
        f"{PERFORMANCE_RESCUE_FINALIZATION_CANDIDATES}, "
        f"LANDMARK_FEATURE_SOURCE_POLICY={LANDMARK_FEATURE_SOURCE_POLICY}, "
        f"LANDMARK_CUMULATIVE_PREDICTION_ENSEMBLE={LANDMARK_CUMULATIVE_PREDICTION_ENSEMBLE}, "
        f"LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES="
        f"{LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES}, "
        f"LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES="
        f"{LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES}, "
        f"LANDMARK_TEMPORAL_AUROC_TOLERANCE={LANDMARK_TEMPORAL_AUROC_TOLERANCE}, "
        f"RUN_SIMPLIFIED_MODELS={RUN_SIMPLIFIED_MODELS}, "
        f"SIMPLIFIED_TOP_N_FEATURES={SIMPLIFIED_TOP_N_FEATURES}, "
        f"SIMPLIFIED_RETUNE={SIMPLIFIED_RETUNE}, "
        f"SIMPLIFIED_OPTUNA_TRIALS={SIMPLIFIED_OPTUNA_TRIALS}, "
        f"SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES={SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES}, "
        f"SIMPLIFIED_MICE_ITER_CANDIDATES={SIMPLIFIED_MICE_ITER_CANDIDATES}, "
        f"SIMPLIFIED_SHAP_RANKING_MODE={SIMPLIFIED_SHAP_RANKING_MODE}, "
        f"SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES="
        f"{SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES}, "
        f"SIMPLIFIED_VALIDATION_CANDIDATE_TRIALS={SIMPLIFIED_VALIDATION_CANDIDATE_TRIALS}, "
        f"SIMPLIFIED_VALIDATION_CANDIDATE_TRIAL_OVERRIDES="
        f"{SIMPLIFIED_VALIDATION_CANDIDATE_TRIAL_OVERRIDES}, "
        f"SIMPLIFIED_SEED_BAGGING_REPEATS={SIMPLIFIED_SEED_BAGGING_REPEATS}, "
        f"SIMPLIFIED_SEED_BAGGING_REPEAT_OVERRIDES="
        f"{SIMPLIFIED_SEED_BAGGING_REPEAT_OVERRIDES}, "
        f"OPTUNA_OBJECTIVE_METRIC={OPTUNA_OBJECTIVE_METRIC}, "
        f"AUPRC_LIFT_WEIGHT={AUPRC_LIFT_WEIGHT}, "
        f"THRESHOLD_OBJECTIVE={THRESHOLD_OBJECTIVE}, "
        f"THRESHOLD_MIN_SENSITIVITY={THRESHOLD_MIN_SENSITIVITY}, "
        f"THRESHOLD_CANDIDATE_POLICY={THRESHOLD_CANDIDATE_POLICY}, "
        f"VALIDATION_MODEL_SELECTION_METRIC={VALIDATION_MODEL_SELECTION_METRIC}, "
        f"DUAL_DISCRIMINATION_SELECTION={DUAL_DISCRIMINATION_SELECTION}, "
        f"AUROC_NEAR_BEST_TOLERANCE={AUROC_NEAR_BEST_TOLERANCE}, "
        f"DCA_THRESHOLD_RANGE=({DCA_THRESHOLD_MIN}, {DCA_THRESHOLD_MAX}), "
        f"CALIBRATION_DIAGNOSTIC_METHOD={CALIBRATION_DIAGNOSTIC_METHOD}, "
        f"ENFORCE_POSITIVE_CALIBRATION_SLOPE={ENFORCE_POSITIVE_CALIBRATION_SLOPE}, "
        f"ALLOW_PROBABILITY_ORIENTATION_CORRECTION={ALLOW_PROBABILITY_ORIENTATION_CORRECTION}, "
        f"ORIENTATION_CORRECTION_TARGETS={ORIENTATION_CORRECTION_TARGETS}, "
        f"ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC={ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC}"
    )

    if AGVHD_XGB_REPRODUCTION_PROFILE and not AGVHD_XGB_REPRODUCTION_DATA_MATCH:
        print(
            "NOTE: the XGBoost aGVHD reproduction profile is active, but the input "
            "data hash differs from the archived run. Exact archived AUROCs require "
            "XGB_V2_DATA_PATH=dataset.xlsx; current-data results are reported as a "
            "separate transport evaluation."
        )


def initialize_runtime():
    """Resolve file-backed settings and create run output directories."""
    global AGVHD_XGB_REPRODUCTION_DATA_MATCH, DATA_SHA256
    global PIPELINE_CACHE_DIR, PIPELINE_MEMORY

    DATA_SHA256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    AGVHD_XGB_REPRODUCTION_DATA_MATCH = (
        DATA_SHA256 == ARCHIVED_AGVHD_REPRODUCTION_DATA_SHA256
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    PIPELINE_CACHE_DIR = Path(
        os.getenv(
            "XGB_V2_PIPELINE_CACHE_DIR",
            str(
                OUTPUT_DIR
                / ".pipeline_cache"
                / f"{PIPELINE_CACHE_VERSION}_{DATA_SHA256[:12]}"
            ),
        )
    )
    PIPELINE_MEMORY = (
        Memory(location=PIPELINE_CACHE_DIR, verbose=0)
        if PIPELINE_CACHE_ENABLED
        else None
    )
    _print_runtime_config()
