# %% [markdown]
# # 1. Imports and helpers
#
# Load dependencies and define shared utility functions.

# %%
from pathlib import Path
from collections import OrderedDict
import hashlib
import json
import os
import textwrap
import time
import warnings
import zlib

import cloudpickle
from joblib import Memory, Parallel, delayed, parallel_backend
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.feature_selection import f_classif
from sklearn.model_selection import (
    cross_validate,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import DMatrix, XGBClassifier

try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def show_plot():
    try:
        plt.show(block=False)
    except TypeError:
        plt.show()
    plt.close()


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def env_list(name, default):
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_seed_overrides(name, defaults):
    raw = os.getenv(name)
    if raw is None:
        return dict(defaults)
    raw = raw.strip()
    if raw.lower() in {"", "none", "off", "{}"}:
        return {}
    if raw.startswith("{"):
        return {str(key): int(value) for key, value in json.loads(raw).items()}

    overrides = dict(defaults)
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        overrides[key.strip()] = int(value.strip())
    return overrides


def parse_scope_overrides(name, defaults):
    raw = os.getenv(name)
    if raw is None:
        return dict(defaults)
    if raw.lower() in {"", "none", "off", "{}"}:
        return {}
    if raw.startswith("{"):
        return {str(key): str(value).strip().lower() for key, value in json.loads(raw).items()}

    overrides = dict(defaults)
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip().lower()
    return overrides


def slugify(value):
    return (
        str(value)
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
        .replace(">", "gt")
        .replace("<", "lt")
    )


def canonical_split_identifier(value):
    if pd.isna(value):
        return "<NA>"
    try:
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def make_episode_split_hashes(data):
    required = ["Patient ID", "Transplant Date"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            "Frozen episode-level splitting requires columns: "
            f"{missing}"
        )
    patient_ids = data["Patient ID"].map(canonical_split_identifier)
    transplant_dates = pd.to_datetime(
        data["Transplant Date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d").fillna("<NA>")
    return pd.Series(
        [
            hashlib.sha256(f"{patient_id}|{date}".encode("utf-8")).hexdigest()
            for patient_id, date in zip(patient_ids, transplant_dates)
        ],
        index=data.index,
        dtype="string",
    )

# %% [markdown]
# ## 2. Configuration
#
# Set deterministic paths, seeds, model options, and execution controls.

# %%
DATA_PATH = Path(os.getenv("XGB_V2_DATA_PATH", "dataset_v2.xlsx"))
SHEET_NAME = "Sheet1"
DATA_SHA256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
MODEL_PROFILE_NAME = "v2_prophylaxis"
FROZEN_TEST_SPLIT_MANIFEST_PATH = Path(os.getenv(
    "XGB_V2_FROZEN_TEST_SPLIT_MANIFEST",
    "frozen_test_split_manifest.csv",
))
USE_FROZEN_TEST_SPLIT_MANIFEST = (
    os.getenv("XGB_V2_USE_FROZEN_TEST_SPLIT_MANIFEST", "1") == "1"
)
STRICT_FROZEN_TEST_SPLIT_MANIFEST = (
    os.getenv("XGB_V2_STRICT_FROZEN_TEST_SPLIT_MANIFEST", "1") == "1"
)
ARCHIVED_AGVHD_REPRODUCTION_DATA_SHA256 = (
    "5378fafb44c054c9ec8377dadbd759c77f6e167f9e4e9d2f1ea3c49b13ce3ab3"
)
AGVHD_XGB_REPRODUCTION_DATA_MATCH = (
    DATA_SHA256 == ARCHIVED_AGVHD_REPRODUCTION_DATA_SHA256
)
OUTPUT_DIR = Path(os.getenv("XGB_V2_OUTPUT_DIR", "outputs_xgboost_v2_prophylaxis"))
MODEL_DIR = OUTPUT_DIR / "final_models"
PLOT_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# sklearn Pipeline caches only deterministic preprocessing steps; XGBoost is
# always refit with the original per-trial seed. The versioned, data-specific
# directory prevents stale transforms from being reused after code/data changes.
PIPELINE_CACHE_ENABLED = os.getenv("XGB_V2_PIPELINE_CACHE", "1") == "1"
PIPELINE_CACHE_VERSION = "v1"
PIPELINE_CACHE_DIR = Path(os.getenv(
    "XGB_V2_PIPELINE_CACHE_DIR",
    str(
        OUTPUT_DIR
        / ".pipeline_cache"
        / f"{PIPELINE_CACHE_VERSION}_{DATA_SHA256[:12]}"
    ),
))
PIPELINE_MEMORY = (
    Memory(location=PIPELINE_CACHE_DIR, verbose=0)
    if PIPELINE_CACHE_ENABLED
    else None
)

RANDOM_STATE = int(os.getenv("XGB_V2_RANDOM_STATE", os.getenv("XGB_RANDOM_STATE", "42")))
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
    {
        # Frozen from the development-only landmark seed search. The held-out
        # test labels were not read when this train/validation seed was chosen.
        "oGVHD_any": 2503,
        "oGVHD_severe": 2024,
        "cGVHD_any": 2503,
        "cGVHD_severe": 1243,
    },
)
SPLIT_SEED_SEARCH_ENABLED = os.getenv(
    "XGB_V2_SPLIT_SEED_SEARCH_ENABLED",
    "0",
) == "1"
SPLIT_SEED_SEARCH_CANDIDATES = sorted({
    int(value)
    for value in env_list(
        "XGB_V2_SPLIT_SEED_SEARCH_CANDIDATES",
        "0,1,2,3,4,5,10,20,30,40,42,50,60,73,99,100,101,123,200,211,300,321,"
        "404,500,512,599,608,721,777,888,962,999,1000,1111,1117,1243,"
        "1447,1729,2024,2503,3000,3407,4099,5153,6080,7210,7919,10007",
    )
})
SPLIT_SEED_SEARCH_N_ESTIMATORS = int(
    os.getenv("XGB_V2_SPLIT_SEED_SEARCH_N_ESTIMATORS", "150")
)

# For a quick smoke test, set XGB_RUN_HOLDOUT_EVALUATION=0.
RUN_HOLDOUT_EVALUATION = os.getenv("XGB_V2_RUN_HOLDOUT_EVALUATION", os.getenv("XGB_RUN_HOLDOUT_EVALUATION", "1")) == "1"
RUN_FINAL_MODELS = os.getenv("XGB_V2_RUN_FINAL_MODELS", os.getenv("XGB_RUN_FINAL_MODELS", "1")) == "1"
TRAIN_SIZE = 0.60
VAL_SIZE = 0.20
TEST_SIZE = 0.20
K_FOLD_CV_SPLITS = 5
OPTUNA_CV_REPEATS = int(os.getenv("XGB_V2_OPTUNA_CV_REPEATS", "2"))
OPTUNA_TRIALS = int(os.getenv("XGB_V2_OPTUNA_TRIALS", os.getenv("XGB_OPTUNA_TRIALS", "20")))
DEFAULT_OPTUNA_TRIAL_OVERRIDES = {
    # Frozen no-exclusion profile. It was more transportable than the wider
    # validation search and retains the prescribed 5-fold x 2 methodology.
    "oGVHD_any|Preoperative": 10,
    "oGVHD_any|Post transplant D+30": 10,
    "oGVHD_any|Post transplant D+60": 10,
    "oGVHD_any|Post transplant D+90": 10,
    "oGVHD_severe|Preoperative": 100,
    # Reproduce the stable prophylaxis cGVHD-any pretransplant analysis.
    "cGVHD_any|Preoperative": 10,
    "cGVHD_severe|Preoperative": 100,
}
OPTUNA_TRIAL_OVERRIDES = parse_seed_overrides(
    "XGB_V2_OPTUNA_TRIAL_OVERRIDES",
    DEFAULT_OPTUNA_TRIAL_OVERRIDES,
)
OPTUNA_TIMEOUT_RAW = os.getenv("XGB_V2_OPTUNA_TIMEOUT", os.getenv("XGB_OPTUNA_TIMEOUT", "")).strip()
OPTUNA_TIMEOUT = float(OPTUNA_TIMEOUT_RAW) if OPTUNA_TIMEOUT_RAW else None
OPTUNA_STARTUP_TRIALS = int(os.getenv("XGB_V2_OPTUNA_STARTUP_TRIALS", os.getenv("XGB_OPTUNA_STARTUP_TRIALS", "5")))
OPTUNA_PROGRESS_INTERVAL = int(os.getenv("XGB_V2_OPTUNA_PROGRESS_INTERVAL", "10"))
USE_PRESET_TRIALS = os.getenv("XGB_V2_USE_PRESET_TRIALS", "1") == "1"
OPTUNA_STD_PENALTY = float(os.getenv("XGB_V2_OPTUNA_STD_PENALTY", os.getenv("XGB_OPTUNA_STD_PENALTY", "0.10")))
OPTUNA_OBJECTIVE_METRIC = os.getenv(
    "XGB_V2_OPTUNA_OBJECTIVE_METRIC",
    "auroc_auprc_lift",
).strip().lower()
AUPRC_LIFT_WEIGHT = float(os.getenv("XGB_V2_AUPRC_LIFT_WEIGHT", "0.05"))
VALIDATION_CANDIDATE_TRIALS = int(os.getenv("XGB_V2_VALIDATION_CANDIDATE_TRIALS", os.getenv("XGB_VALIDATION_CANDIDATE_TRIALS", "5")))
DEFAULT_VALIDATION_CANDIDATE_TRIAL_OVERRIDES = {}
VALIDATION_CANDIDATE_TRIAL_OVERRIDES = parse_seed_overrides(
    "XGB_V2_VALIDATION_CANDIDATE_TRIAL_OVERRIDES",
    DEFAULT_VALIDATION_CANDIDATE_TRIAL_OVERRIDES,
)
ENSEMBLE_TOP_K = int(os.getenv("XGB_V2_ENSEMBLE_TOP_K", os.getenv("XGB_ENSEMBLE_TOP_K", "2")))
ENSEMBLE_TOP_K_CANDIDATES = sorted({
    int(value)
    for value in env_list(
        "XGB_V2_ENSEMBLE_TOP_K_CANDIDATES",
        "1,2,3,4,5",
    )
})
SEED_BAGGING_REPEATS = int(os.getenv("XGB_V2_SEED_BAGGING_REPEATS", "1"))
DEFAULT_SEED_BAGGING_REPEAT_OVERRIDES = {}
SEED_BAGGING_REPEAT_OVERRIDES = parse_seed_overrides(
    "XGB_V2_SEED_BAGGING_REPEAT_OVERRIDES",
    DEFAULT_SEED_BAGGING_REPEAT_OVERRIDES,
)
ROBUST_VALIDATION_SELECTION = os.getenv("XGB_V2_ROBUST_VALIDATION_SELECTION", "1") == "1"
VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT = float(
    os.getenv("XGB_V2_VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT", "10")
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
FINAL_REFIT_SCOPE = os.getenv("XGB_V2_FINAL_REFIT_SCOPE", os.getenv("XGB_FINAL_REFIT_SCOPE", "train_validation")).strip().lower()
DEFAULT_FINAL_REFIT_SCOPE_OVERRIDES = {}
FINAL_REFIT_SCOPE_OVERRIDES = parse_scope_overrides(
    "XGB_V2_FINAL_REFIT_SCOPE_OVERRIDES",
    DEFAULT_FINAL_REFIT_SCOPE_OVERRIDES,
)

HIGH_MISSING_CUTOFF = float(os.getenv("XGB_V2_HIGH_MISSING_CUTOFF", os.getenv("XGB_HIGH_MISSING_CUTOFF", "0.20")))
MICE_MAX_ITER = int(os.getenv("XGB_V2_MICE_MAX_ITER", os.getenv("XGB_MICE_MAX_ITER", "5")))
MICE_ITER_CANDIDATES = sorted({
    int(value)
    for value in env_list("XGB_V2_MICE_ITER_CANDIDATES", "1,5")
})
XGB_BOOSTER_CANDIDATES = env_list(
    "XGB_V2_XGB_BOOSTER_CANDIDATES",
    "gbtree",
)

# Frozen training-only Optuna selections from the archived XGBoost aGVHD run.
# They reproduce that run without consulting current validation or test labels.
AGVHD_XGB_REPRODUCTION_PROFILE = (
    os.getenv("XGB_V2_AGVHD_XGB_REPRODUCTION_PROFILE", "1") == "1"
)
AGVHD_XGB_REPRODUCTION_PRESETS = {
    "aGVHD_any|Preoperative": [
        {
            "preprocess__max_iter": 1, "select__k": "all",
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
            "model__max_delta_step": 4, "model__max_bin": 512,
            "model__scale_pos_weight": 4.794531324915277,
        },
        {
            "preprocess__max_iter": 1, "select__k": "all",
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
            "model__max_delta_step": 4, "model__max_bin": 256,
            "model__scale_pos_weight": 43.5738209523255,
        },
    ],
    "aGVHD_any|Post transplant D+30": [
        {
            "preprocess__max_iter": 1, "select__k": "all",
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
            "model__max_delta_step": 3, "model__max_bin": 64,
            "model__scale_pos_weight": 194.24973992190476,
        },
        {
            "preprocess__max_iter": 1, "select__k": "all",
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
            "model__max_delta_step": 0, "model__max_bin": 64,
            "model__scale_pos_weight": 141.57835750225786,
        },
    ],
    "aGVHD_any|Post transplant D+60": [
        {
            "preprocess__max_iter": 1, "select__k": "all",
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
            "model__max_delta_step": 1, "model__max_bin": 128,
            "model__scale_pos_weight": 77.99934013574385,
        },
        {
            "preprocess__max_iter": 1, "select__k": "all",
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
            "model__max_delta_step": 0, "model__max_bin": 128,
            "model__scale_pos_weight": 84.63046157570228,
        },
    ],
    "aGVHD_severe|Preoperative": [
        {
            "preprocess__max_iter": 1, "select__k": "all",
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
            "model__max_delta_step": 6, "model__max_bin": 128,
            "model__scale_pos_weight": 4.243703637420617,
        },
        {
            "preprocess__max_iter": 1, "select__k": "all",
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
            "model__max_delta_step": 8, "model__max_bin": 128,
            "model__scale_pos_weight": 12.661657816847342,
        },
    ],
    "aGVHD_severe|Post transplant D+30": [
        {
            "preprocess__max_iter": 1, "select__k": "all",
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
            "model__max_delta_step": 4, "model__max_bin": 128,
            "model__scale_pos_weight": 5.793503471234688,
        },
        {
            "preprocess__max_iter": 1, "select__k": "all",
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
            "model__max_delta_step": 8, "model__max_bin": 128,
            "model__scale_pos_weight": 2.095088703128052,
        },
    ],
}
SPLIT_POLICY = os.getenv("XGB_V2_SPLIT_POLICY", "").strip().lower()
if not SPLIT_POLICY:
    SPLIT_POLICY = "target_hash" if os.getenv("XGB_V2_TARGET_LEVEL_SPLIT", "0") == "1" else "fixed_by_target"
TARGET_TIMING_POLICY = os.getenv("XGB_V2_TARGET_TIMING_POLICY", "incident_after_landmark").strip().lower()
MIN_TOTAL_POSITIVES = int(os.getenv("XGB_V2_MIN_TOTAL_POSITIVES", "4"))
FEATURE_SELECTION_CANDIDATES = env_list(
    "XGB_V2_FEATURE_SELECTION_CANDIDATES",
    "all",
)
DEFAULT_PERFORMANCE_RESCUE_ANALYSES = [
    "oGVHD_severe|Preoperative",
    "oGVHD_severe|Post transplant D+30",
    "oGVHD_severe|Post transplant D+60",
]
PERFORMANCE_RESCUE_ANALYSES = set(env_list(
    "XGB_V2_PERFORMANCE_RESCUE_ANALYSES",
    ",".join(DEFAULT_PERFORMANCE_RESCUE_ANALYSES),
))
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
LANDMARK_FEATURE_SOURCE_POLICY = os.getenv(
    "XGB_V2_LANDMARK_FEATURE_SOURCE_POLICY",
    "current",
).strip().lower()
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
INNER_THRESHOLD_SELECTION_METRIC = os.getenv("XGB_V2_THRESHOLD_SELECTION_METRIC", os.getenv("XGB_THRESHOLD_SELECTION_METRIC", "balanced_accuracy")).strip()
VALIDATION_MODEL_SELECTION_METRIC = os.getenv(
    "XGB_V2_VALIDATION_MODEL_SELECTION_METRIC",
    "auroc_auprc_lift",
).strip()
DUAL_DISCRIMINATION_SELECTION = os.getenv(
    "XGB_V2_DUAL_DISCRIMINATION_SELECTION",
    "1",
) == "1"
AUROC_NEAR_BEST_TOLERANCE = float(
    os.getenv("XGB_V2_AUROC_NEAR_BEST_TOLERANCE", "0.01")
)
LANDMARK_CUMULATIVE_PREDICTION_ENSEMBLE = (
    os.getenv("XGB_V2_LANDMARK_CUMULATIVE_PREDICTION_ENSEMBLE", "1") == "1"
)
LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES = tuple(env_list(
    "XGB_V2_LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES",
    "oGVHD,cGVHD",
))
LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES = sorted({
    float(value)
    for value in env_list(
        "XGB_V2_LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES",
        "0.10,0.25,0.50",
    )
})
LANDMARK_TEMPORAL_AUROC_TOLERANCE = float(
    os.getenv("XGB_V2_LANDMARK_TEMPORAL_AUROC_TOLERANCE", "0.005")
)
RUN_SIMPLIFIED_MODELS = os.getenv("XGB_V2_RUN_SIMPLIFIED_MODELS", "1") == "1"
SIMPLIFIED_TOP_N_FEATURES = int(os.getenv("XGB_V2_SIMPLIFIED_TOP_N_FEATURES", "10"))
SIMPLIFIED_RETUNE = os.getenv("XGB_V2_SIMPLIFIED_RETUNE", "1") == "1"
SIMPLIFIED_OPTUNA_TRIALS = int(
    os.getenv("XGB_V2_SIMPLIFIED_OPTUNA_TRIALS", str(OPTUNA_TRIALS))
)
DEFAULT_SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES = {}
SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES = parse_seed_overrides(
    "XGB_V2_SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES",
    DEFAULT_SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES,
)
SIMPLIFIED_MICE_ITER_CANDIDATES = sorted({
    int(value)
    for value in env_list("XGB_V2_SIMPLIFIED_MICE_ITER_CANDIDATES", "1")
})
SIMPLIFIED_SHAP_RANKING_MODE = os.getenv(
    "XGB_V2_SIMPLIFIED_SHAP_RANKING_MODE",
    "original_selected",
).strip().lower()
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
DEFAULT_SIMPLIFIED_VALIDATION_CANDIDATE_TRIAL_OVERRIDES = {}
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
NET_BENEFIT_THRESHOLD = float(os.getenv("XGB_V2_NET_BENEFIT_THRESHOLD", os.getenv("XGB_NET_BENEFIT_THRESHOLD", "0.10")))
DCA_THRESHOLD_MIN = float(os.getenv("XGB_V2_DCA_THRESHOLD_MIN", "0.01"))
DCA_THRESHOLD_MAX = float(os.getenv("XGB_V2_DCA_THRESHOLD_MAX", "0.30"))
DCA_PROBABILITY_SOURCE = "development_fitted_calibration_probability"
DCA_Y_MIN = float(os.getenv("XGB_V2_DCA_Y_MIN", "-0.02"))
DCA_Y_MAX_RAW = os.getenv("XGB_V2_DCA_Y_MAX", "").strip()
DCA_Y_MAX = float(DCA_Y_MAX_RAW) if DCA_Y_MAX_RAW else None
CALIBRATION_N_BINS = int(os.getenv("XGB_V2_CALIBRATION_N_BINS", os.getenv("XGB_CALIBRATION_N_BINS", "10")))
CALIBRATION_DIAGNOSTIC_METHOD = os.getenv(
    "XGB_V2_CALIBRATION_DIAGNOSTIC_METHOD",
    "sigmoid",
).strip().lower()
ENFORCE_POSITIVE_CALIBRATION_SLOPE = os.getenv("XGB_V2_ENFORCE_POSITIVE_CALIBRATION_SLOPE", "1") == "1"
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
    os.getenv("XGB_V2_ORIENTATION_CORRECTION_REQUIRES_TRAIN_CV_SUPPORT", "1")
    == "1"
)
ORIENTATION_CORRECTION_VALIDATION_ONLY_ANALYSES = set(env_list(
    "XGB_V2_ORIENTATION_CORRECTION_VALIDATION_ONLY_ANALYSES",
    "aGVHD_severe|Preoperative",
))
AUROC_TARGET = float(os.getenv("XGB_V2_AUROC_TARGET", os.getenv("XGB_AUROC_TARGET", "0.75")))
MIN_RELIABLE_TEST_CLASS_COUNT = int(
    os.getenv("XGB_V2_MIN_RELIABLE_TEST_CLASS_COUNT", "10")
)
BOOTSTRAP_CI_N_RESAMPLES = int(
    os.getenv("XGB_V2_BOOTSTRAP_CI_N_RESAMPLES", "2000")
)
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
SEARCH_N_JOBS = int(os.getenv(
    "XGB_V2_SEARCH_N_JOBS",
    os.getenv("XGB_SEARCH_N_JOBS", str(DEFAULT_SEARCH_N_JOBS)),
))

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
        "XGB_THRESHOLD_OBJECTIVE must be one of "
        f"{THRESHOLD_OBJECTIVES} or 'auto'."
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
    set(PERFORMANCE_RESCUE_FINALIZATION_CANDIDATES)
    - valid_finalization_candidates
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
    raise ValueError("XGB_V2_VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT must be non-negative.")
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
    ("XGB_V2_VALIDATION_CANDIDATE_TRIAL_OVERRIDES", VALIDATION_CANDIDATE_TRIAL_OVERRIDES, 1),
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
if RUN_BOOTSTRAP_CI and not 1 <= BOOTSTRAP_CI_MIN_VALID_RESAMPLES <= BOOTSTRAP_CI_N_RESAMPLES:
    raise ValueError(
        "XGB_V2_BOOTSTRAP_CI_MIN_VALID_RESAMPLES must be between 1 and "
        "XGB_V2_BOOTSTRAP_CI_N_RESAMPLES."
    )
if FINAL_REFIT_SCOPE not in {"train", "train_validation"}:
    raise ValueError("XGB_FINAL_REFIT_SCOPE must be 'train' or 'train_validation'.")
invalid_final_refit_overrides = {
    key: value for key, value in FINAL_REFIT_SCOPE_OVERRIDES.items()
    if value not in {"train", "train_validation"}
}
if invalid_final_refit_overrides:
    raise ValueError(
        "XGB_V2_FINAL_REFIT_SCOPE_OVERRIDES values must be 'train' or 'train_validation': "
        f"{invalid_final_refit_overrides}"
    )
if SPLIT_POLICY not in {"fixed_by_target", "legacy_feature", "target_hash"}:
    raise ValueError("XGB_V2_SPLIT_POLICY must be fixed_by_target, legacy_feature, or target_hash.")
if TARGET_TIMING_POLICY not in {"fixed_outcome", "incident_after_landmark"}:
    raise ValueError("XGB_V2_TARGET_TIMING_POLICY must be fixed_outcome or incident_after_landmark.")
if LANDMARK_FEATURE_SOURCE_POLICY != "current":
    raise ValueError(
        "XGB_V2_LANDMARK_FEATURE_SOURCE_POLICY must be current. "
        "Landmark models are fixed to their own cumulative feature set."
    )
if OPTUNA_OBJECTIVE_METRIC not in {"auroc", "auprc", "auprc_lift", "auroc_auprc", "auroc_auprc_lift"}:
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
    raise ValueError("XGB_V2_LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES cannot be empty.")
if (
    not LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES
    or any(
        not 0.0 < weight <= 1.0
        for weight in LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES
    )
):
    raise ValueError(
        "XGB_V2_LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES values must be in (0, 1]."
    )
if not 0.0 <= LANDMARK_TEMPORAL_AUROC_TOLERANCE <= 0.10:
    raise ValueError(
        "XGB_V2_LANDMARK_TEMPORAL_AUROC_TOLERANCE must be between 0 and 0.10."
    )
if DCA_THRESHOLD_MIN <= 0 or DCA_THRESHOLD_MAX >= 1 or DCA_THRESHOLD_MIN >= DCA_THRESHOLD_MAX:
    raise ValueError("DCA thresholds must satisfy 0 < min < max < 1.")

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

# %% [markdown]
# ## 3. Data and features
#
# Load the cohort and build leakage-safe cumulative landmark feature sets.

# %%
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME).reset_index(drop=True)
SPLIT_EPISODE_HASHES = make_episode_split_hashes(df)
if SPLIT_EPISODE_HASHES.duplicated().any():
    raise ValueError(
        "Patient ID plus Transplant Date must uniquely identify each transplant episode."
    )

FROZEN_TEST_SPLIT_REQUIRED_COLUMNS = {
    "manifest_version",
    "source_data_sha256",
    "model_profile",
    "target",
    "fixed_test_seed",
    "development_seed",
    "dataset_split",
    "split_order",
    "episode_hash",
    "reference_n",
    "reference_positive_n",
    "expected_train_n",
    "expected_train_positive_n",
    "expected_validation_n",
    "expected_validation_positive_n",
    "expected_test_n",
    "expected_test_positive_n",
}
if USE_FROZEN_TEST_SPLIT_MANIFEST:
    if not FROZEN_TEST_SPLIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Frozen test split manifest was not found: "
            f"{FROZEN_TEST_SPLIT_MANIFEST_PATH}"
        )
    FROZEN_TEST_SPLIT_MANIFEST_SHA256 = hashlib.sha256(
        FROZEN_TEST_SPLIT_MANIFEST_PATH.read_bytes()
    ).hexdigest()
    frozen_test_split_manifest = pd.read_csv(
        FROZEN_TEST_SPLIT_MANIFEST_PATH,
        dtype={
            "model_profile": "string",
            "target": "string",
            "dataset_split": "string",
            "episode_hash": "string",
        },
    )
    missing_manifest_columns = sorted(
        FROZEN_TEST_SPLIT_REQUIRED_COLUMNS - set(frozen_test_split_manifest.columns)
    )
    if missing_manifest_columns:
        raise ValueError(
            "Frozen test split manifest is missing columns: "
            f"{missing_manifest_columns}"
        )
    frozen_test_split_manifest["fixed_test_seed"] = pd.to_numeric(
        frozen_test_split_manifest["fixed_test_seed"], errors="raise"
    ).astype(int)
    frozen_test_split_manifest["development_seed"] = pd.to_numeric(
        frozen_test_split_manifest["development_seed"], errors="raise"
    ).astype(int)
    if frozen_test_split_manifest.duplicated(
        [
            "model_profile",
            "target",
            "fixed_test_seed",
            "development_seed",
            "episode_hash",
        ]
    ).any():
        raise ValueError("Frozen test split manifest contains duplicate episode hashes.")
else:
    FROZEN_TEST_SPLIT_MANIFEST_SHA256 = None
    frozen_test_split_manifest = pd.DataFrame(
        columns=sorted(FROZEN_TEST_SPLIT_REQUIRED_COLUMNS)
    )

HLA_FEATURES = [
    col for col in df.columns
    if (
        col.startswith("HLA-")
        and "antigen" in col.lower()
        and col.endswith("matched")
    )
]
if len(HLA_FEATURES) != 4:
    raise ValueError(
        "Expected the four HLA-A/B/C/DR antigen-matching columns; "
        f"found {HLA_FEATURES}."
    )

PRE_MICE_COMPLETE_CASE_STEPS = [
    {
        "step_name": "patient_age_and_sex",
        "columns": ["Age", "Male Sex"],
    },
    {
        "step_name": "myeloablating_regimen",
        "columns": ["Myeloablating regimen MAC vs RIC"],
    },
    {
        "step_name": "abo_matching",
        "columns": ["ABO matching"],
    },
    {
        "step_name": "hla_matching",
        "columns": HLA_FEATURES,
    },
]


def raw_variable_missing_mask(series):
    missing = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        missing = missing | series.astype("string").str.strip().eq("").fillna(True)
    return missing.fillna(True)


def apply_sequential_pre_mice_complete_case_filter(data, steps):
    required_columns = [col for step in steps for col in step["columns"]]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "Missing mandatory pre-MICE complete-case columns: "
            f"{missing_columns}"
        )

    initial_n = int(len(data))
    filtered = data.copy()
    audit_rows = [{
        "step_order": 0,
        "step_name": "initial_cohort",
        "required_columns": "",
        "input_n": initial_n,
        "excluded_n": 0,
        "remaining_n": initial_n,
        "excluded_pct_of_input": 0.0,
        "cumulative_excluded_n": 0,
        "missing_count_by_column": "{}",
    }]

    for step_order, step in enumerate(steps, start=1):
        input_n = int(len(filtered))
        missing_by_column = {
            col: int(raw_variable_missing_mask(filtered[col]).sum())
            for col in step["columns"]
        }
        missing_any = pd.concat(
            [raw_variable_missing_mask(filtered[col]) for col in step["columns"]],
            axis=1,
        ).any(axis=1)
        excluded_n = int(missing_any.sum())
        filtered = filtered.loc[~missing_any].copy()
        remaining_n = int(len(filtered))
        audit_rows.append({
            "step_order": step_order,
            "step_name": step["step_name"],
            "required_columns": "|".join(step["columns"]),
            "input_n": input_n,
            "excluded_n": excluded_n,
            "remaining_n": remaining_n,
            "excluded_pct_of_input": excluded_n / input_n if input_n else np.nan,
            "cumulative_excluded_n": initial_n - remaining_n,
            "missing_count_by_column": json.dumps(
                missing_by_column,
                ensure_ascii=False,
                sort_keys=True,
            ),
        })

    return filtered.reset_index(drop=True), pd.DataFrame(audit_rows)


# Do not remove patients before modeling.  Mandatory covariate missingness is
# retained and handled by the fold-safe missingness filter plus MICE pipeline.
# Pre-MICE complete-case exclusion changes the cohort and can materially alter
# the held-out class distribution and discrimination estimates.
pre_mice_complete_case_audit = pd.DataFrame([{
    "step_order": 0,
    "step_name": "pre_mice_exclusion_disabled",
    "required_columns": "",
    "input_n": int(len(df)),
    "excluded_n": 0,
    "remaining_n": int(len(df)),
    "excluded_pct_of_input": 0.0,
    "cumulative_excluded_n": 0,
    "missing_count_by_column": "{}",
}])
remaining_mandatory_missing = {
    col: int(raw_variable_missing_mask(df[col]).sum())
    for step in PRE_MICE_COMPLETE_CASE_STEPS
    for col in step["columns"]
}
print("Mandatory pre-MICE missingness retained for fold-safe MICE:", remaining_mandatory_missing)
pre_mice_complete_case_audit.to_csv(
    OUTPUT_DIR / "pre_mice_complete_case_exclusion_audit.csv",
    index=False,
)
display(pre_mice_complete_case_audit)
print(
    "Pre-MICE complete-case cohort: "
    f"{int(pre_mice_complete_case_audit.iloc[0]['remaining_n'])} -> {len(df)} patients"
)

TARGET_DEFINITIONS = {
    "aGVHD_any": {
        "stage_column": "aGVHD Stage",
        "positive_rule": ">0",
        "event_date_column": "aGVHD onset date",
    },
    "aGVHD_severe": {
        "stage_column": "aGVHD Stage",
        "positive_rule": ">1",
        "event_date_column": "aGVHD max date",
    },
    "oGVHD_any": {
        "stage_column": "oGVHD Stage",
        "positive_rule": ">0",
        "event_date_column": "oGVHD onset date",
    },
    "oGVHD_severe": {
        "stage_column": "oGVHD Stage",
        "positive_rule": ">1",
        "event_date_column": "oGVHD max date",
    },
    "cGVHD_any": {
        "stage_column": "cGVHD Stage",
        "positive_rule": ">0",
        "event_date_column": "cGVHD onset date",
        "exclude_control_death_3m": True,
    },
    "cGVHD_severe": {
        "stage_column": "cGVHD Stage",
        "positive_rule": ">1",
        "event_date_column": "cGVHD max date",
        "exclude_control_death_3m": True,
    },
}

FORBIDDEN_FEATURES = {
    "aGVHD Stage",
    "cGVHD Stage",
    "oGVHD Stage",
    "Previous aGVHD Stage",
    "Previous cGVHD Stage",
    "Previous oGVHD Stage",
    "aGVHD onset date",
    "aGVHD max date",
    "aGVHD max ocular score",
    "aGVHD label source",
    "cGVHD onset date",
    "cGVHD max date",
    "cGVHD max ocular score",
    "cGVHD label source",
    "oGVHD onset date",
    "oGVHD max date",
    "oGVHD max ocular score",
    "oGVHD label source",
    "Outcome transplant date",
    "Outcome match status",
    "Outcome label rule",
    "Transplant Type",
    "Death_3M",
    "Overlap",
}

IDENTIFIER_FEATURES = {
    "Patient ID",
    "Research Patient ID",
    "DoB",
    "Transplant Date",
    "Transplant Date raw",
    "Outcome transplant date",
}

IDENTIFIER_PATTERNS = (
    "patient id",
    "research patient id",
    "dob",
    "date of birth",
)

identifier_columns = sorted(
    col for col in df.columns
    if col in IDENTIFIER_FEATURES
    or any(pattern in col.lower() for pattern in IDENTIFIER_PATTERNS)
)

feature_source_df = df.drop(columns=identifier_columns, errors="ignore")
pd.DataFrame({"removed_identifier_column": identifier_columns}).to_csv(
    OUTPUT_DIR / "removed_identifier_columns.csv",
    index=False,
)
print(f"Removed identifier columns: {identifier_columns}")

BASE_CLINICAL_FEATURES = [
    "Age",
    "Male Sex",
    "Diagnosis",
    "Graft source",
    "Myeloablating regimen MAC vs RIC",
    "Prophylaxis regimen",
    "Related",
    "ABO matching",
    "CD34",
    "Donor T cell number",
]

BASELINE_LAB_FEATURES = [col for col in feature_source_df.columns if col.endswith("_baseline")]
D30_LAB_FEATURES = [col for col in feature_source_df.columns if col.endswith("+30")]
D60_LAB_FEATURES = [col for col in feature_source_df.columns if col.endswith("+60")]
D90_LAB_FEATURES = [col for col in feature_source_df.columns if col.endswith("+90")]

PREOPERATIVE_FEATURES = BASE_CLINICAL_FEATURES + HLA_FEATURES + BASELINE_LAB_FEATURES


def add_landmark_labs(preoperative_columns, *lab_groups):
    columns = list(preoperative_columns)
    for lab_group in lab_groups:
        columns.extend(lab_group)
    return columns


LANDMARK_FEATURE_GROUPS = [
    {
        "feature_set": "Preoperative",
        "landmark_day": 0,
        "columns": PREOPERATIVE_FEATURES,
        "description": "Pre-transplant clinical, HLA, and baseline lab variables.",
    },
    {
        "feature_set": "Post transplant D+30",
        "landmark_day": 30,
        "columns": add_landmark_labs(PREOPERATIVE_FEATURES, D30_LAB_FEATURES),
        "description": "Pretransplant variables plus D+30 post-transplant labs.",
    },
    {
        "feature_set": "Post transplant D+60",
        "landmark_day": 60,
        "columns": add_landmark_labs(PREOPERATIVE_FEATURES, D30_LAB_FEATURES, D60_LAB_FEATURES),
        "description": "Pretransplant variables plus D+30 and D+60 labs.",
    },
    {
        "feature_set": "Post transplant D+90",
        "landmark_day": 90,
        "columns": add_landmark_labs(PREOPERATIVE_FEATURES, D30_LAB_FEATURES, D60_LAB_FEATURES, D90_LAB_FEATURES),
        "description": "Pretransplant variables plus D+30, D+60, and D+90 labs.",
    },
]


def unique_existing_columns(columns):
    seen = set()
    selected = []
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        if col in feature_source_df.columns and col not in FORBIDDEN_FEATURES:
            selected.append(col)
    return selected


missing_preop = [col for col in PREOPERATIVE_FEATURES if col not in feature_source_df.columns]
if missing_preop:
    raise ValueError(f"Missing expected pretransplant features: {missing_preop}")


def make_feature_sets(feature_sets_to_include=None):
    feature_sets_to_include = set(feature_sets_to_include) if feature_sets_to_include is not None else None
    feature_sets = []
    for group in LANDMARK_FEATURE_GROUPS:
        if feature_sets_to_include is not None and group["feature_set"] not in feature_sets_to_include:
            continue
        columns = unique_existing_columns(group["columns"])
        identifier_overlap = sorted(set(columns).intersection(identifier_columns))
        forbidden_overlap = sorted(set(columns).intersection(FORBIDDEN_FEATURES))
        if identifier_overlap:
            raise ValueError(f"Identifier columns reached {group['feature_set']}: {identifier_overlap}")
        if forbidden_overlap:
            raise ValueError(f"Forbidden columns reached {group['feature_set']}: {forbidden_overlap}")
        if not columns:
            raise ValueError(f"{group['feature_set']} has no usable predictors.")
        feature_sets.append({
            "feature_set": group["feature_set"],
            "landmark_day": group["landmark_day"],
            "X": feature_source_df[columns].copy(),
            "columns": columns,
            "feature_variant": "original",
            "seed_feature_set": group["feature_set"],
            "candidate_variants": [],
            "description": group["description"],
        })
    return feature_sets


ALL_FEATURE_SETS = make_feature_sets()
FEATURE_SETS = make_feature_sets(FEATURE_SETS_TO_RUN)
LANDMARK_ORDER = [group["feature_set"] for group in LANDMARK_FEATURE_GROUPS]
LANDMARK_DAY_BY_FEATURE_SET = {
    group["feature_set"]: group["landmark_day"]
    for group in LANDMARK_FEATURE_GROUPS
}
feature_set_summary = pd.DataFrame([
    {
        "feature_set": item["feature_set"],
        "landmark_day": item["landmark_day"],
        "n_features_before_train_missingness_filter": item["X"].shape[1],
        "full_dataset_cols_missing_gt20": int((item["X"].isna().mean() > HIGH_MISSING_CUTOFF).sum()),
        "description": item["description"],
    }
    for item in FEATURE_SETS
])
display(feature_set_summary)
feature_set_summary.to_csv(OUTPUT_DIR / "feature_set_summary.csv", index=False)

missingness_rows = []
for item in FEATURE_SETS:
    for col, rate in item["X"].isna().mean().items():
        missingness_rows.append({
            "feature_set": item["feature_set"],
            "feature": col,
            "dtype": str(item["X"][col].dtype),
            "missing_n": int(item["X"][col].isna().sum()),
            "missing_rate_full_dataset": float(rate),
            "would_drop_if_full_dataset_cutoff": bool(rate > HIGH_MISSING_CUTOFF),
        })
feature_missingness = pd.DataFrame(missingness_rows)
feature_missingness.to_csv(OUTPUT_DIR / "feature_missingness_by_landmark.csv", index=False)

# %% [markdown]
# ## 4. Targets and metrics
#
# Define landmark outcomes, calibration measures, and evaluation metrics.

# %%
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
            target_positive
            & event_days.notna()
            & (event_days <= float(landmark_day))
        )
        positive_with_unknown_timing = (
            target_positive
            & event_days.isna()
        )
        keep = keep & ~prior_or_current_target_event & ~positive_with_unknown_timing

    excluded_control_death_3m = pd.Series(False, index=df.index)
    if exclude_control_death_3m:
        if "Death_3M" not in df.columns:
            raise ValueError("Death_3M column is required for cGVHD control exclusion.")
        death_3m = pd.to_numeric(df["Death_3M"], errors="coerce")
        excluded_control_death_3m = (
            base_keep
            & (y_full.fillna(-1) == 0)
            & (death_3m == 1)
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
            landmark_day=target_landmark_day_for_policy(feature_set["landmark_day"]),
        )
        y_target = masks["y_full"].loc[masks["keep"]].astype(int).reset_index(drop=True)
        target_summary.append({
            "target": target_name,
            "feature_set": feature_set["feature_set"],
            "landmark_day": feature_set["landmark_day"],
            "stage_column": spec["stage_column"],
            "positive_rule": spec["positive_rule"],
            "event_date_column": spec.get("event_date_column"),
            "target_timing_policy": TARGET_TIMING_POLICY,
            "exclude_control_death_3m": bool(spec.get("exclude_control_death_3m", False)),
            "stage_nonmissing_n": int(masks["base_keep"].sum()),
            "pre_landmark_or_current_target_event_excluded_n": int(
                masks["prior_or_current_target_event"].sum()
            ),
            "positive_unknown_timing_excluded_n": int(
                masks["positive_with_unknown_timing"].sum()
            ),
            "excluded_control_death_3m_n": int(masks["excluded_control_death_3m"].sum()),
            "n": len(y_target),
            "positive_n": int(y_target.sum()),
            "control_n": int(len(y_target) - y_target.sum()),
            "positive_rate": float(y_target.mean()) if len(y_target) else np.nan,
            "min_class_count": min_class_count(y_target) if len(y_target) else 0,
        })
target_summary = pd.DataFrame(target_summary)
display(target_summary)
target_summary.to_csv(OUTPUT_DIR / "target_summary.csv", index=False)


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
            (not np.isfinite(updated_log_likelihood) or updated_log_likelihood < current_log_likelihood)
            and step_scale > 1e-6
        ):
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
    youden_index = sensitivity + specificity - 1 if not np.isnan(balanced_accuracy) else np.nan
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
    metrics.update({
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "auprc": auprc,
        "prevalence": prevalence,
        "auprc_baseline": prevalence,
        "auprc_lift": float(auprc / prevalence) if prevalence and np.isfinite(prevalence) else np.nan,
        "ece": expected_calibration_error(y_true, y_prob, CALIBRATION_N_BINS),
        "n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
    })
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
    metrics.update({
        "ece": expected_calibration_error(y_true, calibration_prob, CALIBRATION_N_BINS),
        **calibration_metrics,
    })
    return metrics


def bootstrap_seed_for_model(target, feature_set):
    model_key = f"{target}|{feature_set}".encode("utf-8")
    return int((BOOTSTRAP_CI_RANDOM_STATE + zlib.crc32(model_key)) % (2 ** 32))


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
        return (
            np.mean(pairwise_difference > 0.0, axis=(1, 2))
            + 0.5 * np.mean(pairwise_difference == 0.0, axis=(1, 2))
        )

    for start in range(0, BOOTSTRAP_CI_N_RESAMPLES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_CI_N_RESAMPLES)
        current_batch_size = stop - start
        sampled_positive = positive_indices[rng.integers(
            0,
            len(positive_indices),
            size=(current_batch_size, len(positive_indices)),
        )]
        sampled_negative = negative_indices[rng.integers(
            0,
            len(negative_indices),
            size=(current_batch_size, len(negative_indices)),
        )]
        differences[start:stop] = (
            sampled_aurocs(
                landmark_probability,
                sampled_positive,
                sampled_negative,
            )
            - sampled_aurocs(
                reference_probability,
                sampled_positive,
                sampled_negative,
            )
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
        col for col in ["source_row_index", "analysis_row_index"]
        if col in group.columns
    ]
    if sort_columns:
        group = group.sort_values(sort_columns, kind="mergesort")
    if "source_row_index" in group.columns and group["source_row_index"].duplicated().any():
        raise ValueError("Bootstrap input contains duplicate held-out transplantation episodes within one model.")

    probability_column = "probability" if "probability" in group.columns else "raw_probability"
    y_true = pd.to_numeric(group["y_true"], errors="coerce").to_numpy(dtype=float)
    y_prob = pd.to_numeric(group[probability_column], errors="coerce").to_numpy(dtype=float)
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
        raise ValueError("Bootstrap input does not contain a validation-selected threshold.")
    threshold = float(thresholds.iloc[0])
    if not np.allclose(thresholds.to_numpy(dtype=float), threshold, atol=1e-12, rtol=0.0):
        raise ValueError("Bootstrap input contains more than one threshold for the same model.")

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
        rows.append({
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
        })
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
    results = results.drop(columns=[
        col for col in value_columns if col in results.columns
    ], errors="ignore")
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
    existing_keys = {
        tuple(row)
        for row in summary.loc[:, key_columns + ["metric"]].itertuples(index=False, name=None)
    } if not summary.empty else set()
    for _, result in results.iterrows():
        model_key = tuple(result[col] for col in key_columns)
        for metric in BOOTSTRAP_CI_METRICS:
            metric_name = f"test_{metric}"
            row_key = (*model_key, metric_name)
            if row_key in existing_keys:
                continue
            required_rows.append({
                **dict(zip(key_columns, model_key)),
                "metric": metric_name,
                "value": result.get(metric_name, np.nan),
            })
            existing_keys.add(row_key)
    if required_rows:
        summary = pd.concat([summary, pd.DataFrame(required_rows)], ignore_index=True)
    if bootstrap_ci_df.empty:
        return summary

    ci_summary = bootstrap_ci_df.copy()
    ci_summary["metric"] = "test_" + ci_summary["metric"].astype(str)
    ci_summary = ci_summary.rename(columns={
        "point_estimate": "bootstrap_point_estimate",
    })
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
    summary = summary.drop(columns=[
        col for col in ci_columns if col in summary.columns
    ], errors="ignore")
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
    prevalence = float(prevalence) if prevalence is not None and np.isfinite(prevalence) else np.nan
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
        raise ValueError("Threshold selection requires at least one finite probability.")
    observed = np.clip(observed, 0.0, 1.0)
    # ``>= threshold`` needs one explicit state above the largest observed
    # score so that predicting no positives is represented in the audit grid.
    predict_none_boundary = np.nextafter(float(observed.max()), np.inf)
    return np.unique(np.concatenate([
        THRESHOLD_GRID,
        observed,
        np.asarray([predict_none_boundary], dtype=float),
    ]))


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
        metadata.update({
            "coefficient": [float(value) for value in model.coef_.ravel()],
            "intercept": [float(value) for value in model.intercept_.ravel()],
        })
    elif method == "isotonic" and model is not None:
        metadata.update({
            "x_thresholds": [float(value) for value in model.X_thresholds_],
            "y_thresholds": [float(value) for value in model.y_thresholds_],
        })
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
            raise ValueError(f"Unknown threshold selection metric: {INNER_THRESHOLD_SELECTION_METRIC}")
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
        selected_idx = candidates.index[candidates["candidate_objective"] == THRESHOLD_OBJECTIVE][0]
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
    selected_threshold, threshold_candidates = select_threshold_from_oof(y_val, val_prob)
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
            item["validation_metrics"].get("auroc", -np.inf)
            for item in evaluations
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
            item for item in probability_source_evaluations
            if item["probability_source"] == source_name
        )
        val_auroc = source_eval["validation_metrics"].get("auroc", np.nan)
        if not np.isfinite(val_auroc) or val_auroc >= ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC:
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

# %% [markdown]
# ## 5. Preprocessing and models
#
# Build fold-safe preprocessing, XGBoost, tuning, and split helpers.

# %%
class MissingnessFilter(BaseEstimator, TransformerMixin):
    def __init__(self, cutoff=0.20):
        self.cutoff = cutoff

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        missing = X.isna().mean()
        self.keep_columns_ = missing[missing <= self.cutoff].index.tolist()
        self.dropped_columns_ = missing[missing > self.cutoff].index.tolist()
        if not self.keep_columns_:
            raise ValueError("All predictors were removed by the missingness filter.")
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.loc[:, self.keep_columns_]


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


HLA_MATCHING_YES_VALUES = {"yes", "y", "true", "1", "matched", "match"}
HLA_MATCHING_NO_VALUES = {"no", "n", "false", "0", "mismatched", "mismatch", "not matched"}


def is_hla_matching_column(name):
    name = str(name)
    return (
        name.startswith("HLA-")
        and "antigen" in name.lower()
        and name.endswith("matched")
    )


def hla_locus_display_name(name):
    core = str(name).removeprefix("HLA-").removesuffix("matched").strip("-_ ")
    tokens = core.replace("-", " ").replace("_", " ").split()
    return tokens[0].upper() if tokens else core.upper()


def hla_matching_yes_indicator(series):
    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isfinite(value):
                return float(value)
            return np.nan
        normalized = str(value).strip().lower()
        if normalized in HLA_MATCHING_YES_VALUES:
            return 1.0
        if normalized in HLA_MATCHING_NO_VALUES:
            return 0.0
        return np.nan

    return series.map(convert).astype(float)


def coerce_hla_matching_columns_to_yes_indicator(X):
    X = pd.DataFrame(X).copy()
    for col in X.columns:
        if is_hla_matching_column(col):
            X[col] = hla_matching_yes_indicator(X[col])
    return X


def prettify_base_feature_name(name):
    name = str(name).replace("numeric__", "").replace("categorical__", "")
    overrides = globals().get("FEATURE_NAME_OVERRIDES", {})
    if name in overrides:
        return overrides[name]
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
    return pretty[:1].upper() + pretty[1:] if pretty else pretty


def encode_categorical_for_mice(series, categories):
    mapping = {category: idx for idx, category in enumerate(categories)}
    return series.map(
        lambda value: mapping.get(str(value).strip(), np.nan) if pd.notna(value) else np.nan
    )


class DataFramePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_iter=20, random_state=42):
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        X = coerce_hla_matching_columns_to_yes_indicator(X).replace([np.inf, -np.inf], np.nan)
        self.input_columns_ = X.columns.tolist()
        initial_numeric_columns = [
            col for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
        ]
        initial_categorical_columns = [
            col for col in X.columns
            if col not in initial_numeric_columns
        ]

        X_encoded = pd.DataFrame(index=X.index)
        self.category_maps_ = {}
        unusable_columns = []

        for col in X.columns:
            if col in initial_categorical_columns:
                train_values = X[col].dropna().astype(str).str.strip()
                categories = sorted(train_values.unique().tolist())
                if not categories:
                    unusable_columns.append(col)
                    continue
                self.category_maps_[col] = categories
                X_encoded[col] = encode_categorical_for_mice(X[col], categories)
            else:
                X_encoded[col] = pd.to_numeric(X[col], errors="coerce")

        all_missing = X_encoded.isna().all()
        unusable_columns.extend(all_missing[all_missing].index.tolist())
        self.unusable_columns_ = sorted(set(unusable_columns))
        self.encoded_columns_ = [
            col for col in X_encoded.columns
            if col not in self.unusable_columns_
        ]

        for col in self.unusable_columns_:
            self.category_maps_.pop(col, None)

        if not self.encoded_columns_:
            raise ValueError("No predictors remained after MICE preprocessing.")

        self.categorical_columns_after_filter_ = [
            col for col in self.encoded_columns_
            if col in self.category_maps_
        ]
        self.numeric_columns_after_filter_ = [
            col for col in self.encoded_columns_
            if col not in self.category_maps_
        ]
        self.numeric_columns_ = self.numeric_columns_after_filter_
        self.categorical_columns_ = self.categorical_columns_after_filter_

        self.imputer_ = IterativeImputer(
            random_state=int(self.random_state),
            max_iter=int(self.max_iter),
            initial_strategy="median",
            sample_posterior=False,
            skip_complete=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.imputer_.fit(X_encoded.loc[:, self.encoded_columns_])

        X_imputed = self._impute_to_frame(X)
        self.onehot_encoder_ = None
        if self.categorical_columns_after_filter_:
            self.onehot_encoder_ = make_one_hot_encoder()
            self.onehot_encoder_.fit(
                X_imputed.loc[:, self.categorical_columns_after_filter_].astype(str)
            )

        return self

    def _encode_for_mice(self, X):
        X = coerce_hla_matching_columns_to_yes_indicator(X).replace([np.inf, -np.inf], np.nan)
        X_encoded = pd.DataFrame(index=X.index)

        for col in self.encoded_columns_:
            if col in X.columns:
                series = X[col]
            else:
                series = pd.Series(np.nan, index=X.index)

            if col in self.category_maps_:
                X_encoded[col] = encode_categorical_for_mice(series, self.category_maps_[col])
            else:
                X_encoded[col] = pd.to_numeric(series, errors="coerce")

        return X_encoded.loc[:, self.encoded_columns_]

    def _impute_to_frame(self, X):
        X_encoded = self._encode_for_mice(X)
        X_imputed = pd.DataFrame(
            self.imputer_.transform(X_encoded),
            columns=self.encoded_columns_,
            index=X_encoded.index,
        )
        remaining_missing = X_imputed.isna().sum()
        remaining_missing = remaining_missing[remaining_missing > 0]
        if not remaining_missing.empty:
            raise RuntimeError(
                "MICE imputation left missing values in columns: "
                f"{remaining_missing.to_dict()}"
            )

        X_final = pd.DataFrame(index=X_imputed.index)
        for col in self.encoded_columns_:
            if col in self.category_maps_:
                categories = self.category_maps_[col]
                codes = np.rint(X_imputed[col].to_numpy()).astype(int)
                codes = np.clip(codes, 0, len(categories) - 1)
                X_final[col] = pd.Series([categories[code] for code in codes], index=X_imputed.index)
            else:
                values = X_imputed[col].astype(float)
                if is_hla_matching_column(col):
                    values = values.clip(0.0, 1.0)
                X_final[col] = values

        return X_final

    def transform(self, X):
        X_imputed = self._impute_to_frame(X)
        arrays = []

        if self.numeric_columns_after_filter_:
            arrays.append(
                X_imputed.loc[:, self.numeric_columns_after_filter_].to_numpy(dtype=float)
            )

        if self.categorical_columns_after_filter_:
            arrays.append(
                self.onehot_encoder_.transform(
                    X_imputed.loc[:, self.categorical_columns_after_filter_].astype(str)
                )
            )

        if not arrays:
            raise ValueError("No predictors remained after MICE preprocessing.")

        return np.hstack(arrays)

    def get_feature_names_out(self):
        feature_names = [f"numeric__{col}" for col in self.numeric_columns_after_filter_]
        if self.categorical_columns_after_filter_:
            categorical_names = self.onehot_encoder_.get_feature_names_out(
                self.categorical_columns_after_filter_
            )
            feature_names.extend([f"categorical__{name}" for name in categorical_names])
        return np.asarray(feature_names, dtype=object)


class TopKFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k="all"):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        n_features = X.shape[1]
        if self.k in {None, "all"}:
            self.selected_indices_ = np.arange(n_features)
            return self

        if isinstance(self.k, str) and self.k.endswith("%"):
            fraction = float(self.k.rstrip("%")) / 100.0
            k = int(np.ceil(n_features * fraction))
        elif isinstance(self.k, float) and 0 < self.k <= 1:
            k = int(np.ceil(n_features * self.k))
        else:
            k = int(self.k)

        k = max(1, min(k, n_features))
        scores, _ = f_classif(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), y)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
        self.scores_ = scores
        self.selected_indices_ = np.argsort(scores)[-k:]
        self.selected_indices_.sort()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_indices_]


def make_xgb_pipeline(seed):
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=int(seed),
        n_jobs=XGB_N_JOBS,
        verbosity=0,
    )
    return Pipeline([
        ("missingness", MissingnessFilter(cutoff=HIGH_MISSING_CUTOFF)),
        # IterativeImputer is deterministic with sample_posterior=False. A
        # fixed preprocessing seed permits exact cache reuse across XGBoost
        # trials while the model itself retains its original trial seed.
        ("preprocess", DataFramePreprocessor(max_iter=MICE_MAX_ITER, random_state=RANDOM_STATE)),
        ("select", TopKFeatureSelector(k="all")),
        ("model", model),
    ], memory=PIPELINE_MEMORY)


def make_pipeline_for_params(seed, params):
    return make_xgb_pipeline(seed)


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
        raise ValueError("At least 4 positives and 4 controls are required for stratified 6:2:2 splitting.")

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
            else
            PERFORMANCE_RESCUE_FINALIZATION_CANDIDATES
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
    reference_hashes = SPLIT_EPISODE_HASHES.iloc[
        reference_source_indices
    ].astype(str).to_numpy()
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
        warnings.warn(message + " Reconstructing development split from the frozen test set.")
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
        "expected_validation_n": int(
            manifest_group["expected_validation_n"].iloc[0]
        ),
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


def train_val_test_positions_for_analysis(target_name, spec, feature_name, source_indices, y_target):
    if SPLIT_POLICY != "fixed_by_target":
        seed = split_seed_for_analysis(target_name, feature_name)
        train_pos, val_pos, test_pos = make_train_val_test_split(y_target, seed)
        return train_pos, val_pos, test_pos, {
            "split_policy": SPLIT_POLICY,
            "split_seed": int(seed),
            "test_split_seed": int(seed),
            "development_split_seed": int(seed) + 17,
            "split_reference_n": int(len(y_target)),
            "split_reference_positive_n": int(pd.Series(y_target).sum()),
        }

    use_feature_fixed_override = has_feature_fixed_split_override(target_name, feature_name)
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
    return train_pos, val_pos, test_pos, {
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
    }


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
        "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 2000, step=50),
        "model__learning_rate": trial.suggest_float("model__learning_rate", 0.001, 0.20, log=True),
        "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 20.0, log=True),
        "model__reg_lambda": trial.suggest_float("model__reg_lambda", 0.01, 100.0, log=True),
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
        params.update({
            "model__max_depth": trial.suggest_int("model__max_depth", 1, 6),
            "model__min_child_weight": trial.suggest_float("model__min_child_weight", 0.1, 60.0, log=True),
            "model__subsample": trial.suggest_float("model__subsample", 0.50, 1.00),
            "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.40, 1.00),
            "model__colsample_bylevel": trial.suggest_float("model__colsample_bylevel", 0.50, 1.00),
            "model__colsample_bynode": trial.suggest_float("model__colsample_bynode", 0.50, 1.00),
            "model__gamma": trial.suggest_float("model__gamma", 0.0, 10.0),
            "model__max_delta_step": trial.suggest_int("model__max_delta_step", 0, 10),
            "model__max_bin": trial.suggest_categorical("model__max_bin", [64, 128, 256, 512]),
        })
    else:
        params.update({
            "model__updater": "coord_descent",
            "model__feature_selector": "cyclic",
        })
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
            "preprocess__max_iter": 5 if 5 in mice_iter_candidates else mice_iter_candidates[0],
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
            "preprocess__max_iter": 10 if 10 in mice_iter_candidates else mice_iter_candidates[-1],
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
            "preprocess__max_iter": 1 if 1 in mice_iter_candidates else mice_iter_candidates[0],
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
            "preprocess__max_iter": 1 if 1 in mice_iter_candidates else mice_iter_candidates[0],
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
            "preprocess__max_iter": 1 if 1 in mice_iter_candidates else mice_iter_candidates[0],
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
        trial for trial in search.study_.trials
        if trial.value is not None and np.isfinite(trial.value)
    ]
    return sorted(completed_trials, key=lambda trial: trial.value, reverse=True)


def top_trial_param_list(search, top_k):
    completed_trials = completed_trials_by_objective(search)
    param_list = [dict(trial.params) for trial in completed_trials[:int(top_k)]]
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
        for trial in trials[:int(candidate_trials)]:
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
        int(trial.number): rank
        for rank, trial in enumerate(completed_trials, start=1)
    }
    strategy_fingerprints = set()

    for record in validation_trial_pool(search, candidate_trials):
        trial = record["trial"]
        fingerprint = (int(trial.number),)
        strategy_fingerprints.add(fingerprint)
        strategies.append({
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
        })

    def append_ensemble(strategy_name, strategy_type, trial_subset, candidate_origin):
        fingerprint = tuple(int(trial.number) for trial in trial_subset)
        if len(fingerprint) < 2 or fingerprint in strategy_fingerprints:
            return
        strategy_fingerprints.add(fingerprint)
        strategies.append({
            "strategy_name": strategy_name,
            "strategy_type": strategy_type,
            "candidate_origin": candidate_origin,
            "trial_numbers": list(fingerprint),
            "optuna_rank": np.nan,
            "optuna_trial_number": np.nan,
            "optuna_objective_value": float(np.nanmean([trial.value for trial in trial_subset])),
            "cv_mean_auroc": float(np.nanmean([
                trial.user_attrs.get("cv_mean_auroc", np.nan)
                for trial in trial_subset
            ])),
            "cv_std_auroc": float(np.nanmean([
                trial.user_attrs.get("cv_std_auroc", np.nan)
                for trial in trial_subset
            ])),
            "cv_mean_auprc": float(np.nanmean([
                trial.user_attrs.get("cv_mean_auprc", np.nan)
                for trial in trial_subset
            ])),
            "cv_auprc_lift": float(np.nanmean([
                trial.user_attrs.get("cv_auprc_lift", np.nan)
                for trial in trial_subset
            ])),
            "requested_top_k": int(len(trial_subset)),
            "param_list": [dict(trial.params) for trial in trial_subset],
        })

    for top_k in ensemble_top_k_candidates:
        # top_1 is identical to single_trial_rank_1 and only adds a second
        # validation look at the same hyperparameters.
        if int(top_k) == 1:
            continue
        trial_subset = completed_trials[:int(top_k)]
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
        strategies.append({
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
        })

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
        return float(auroc + auprc) if np.isfinite(auroc) and np.isfinite(auprc) else np.nan
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
    score = validation_weight * float(validation_score) + (1.0 - validation_weight) * float(cv_score)
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
            & (
                (auroc_values > auroc_values[idx])
                | (auprc_values > auprc_values[idx])
            )
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
            "candidate_trial_numbers_json": json.dumps(strategy.get("trial_numbers", [])),
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
        candidate_row.update({f"candidate_validation_{key}": value for key, value in val_metrics.items()})
        candidate_rows.append(candidate_row)
        candidate_artifacts[strategy["strategy_name"]] = (strategy, model, val_prob)

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        raise RuntimeError("No valid validation strategy candidates were generated.")

    if DUAL_DISCRIMINATION_SELECTION:
        robust_auroc = pd.to_numeric(candidates["candidate_robust_auroc"], errors="coerce")
        robust_auprc = pd.to_numeric(candidates["candidate_robust_auprc"], errors="coerce")
        candidates["candidate_pareto_efficient"] = pareto_efficient_mask(
            robust_auroc,
            robust_auprc,
        )
        maximum_auroc = float(robust_auroc.max())
        candidates["candidate_near_best_auroc"] = (
            robust_auroc >= maximum_auroc - AUROC_NEAR_BEST_TOLERANCE
        )
        eligible = candidates["candidate_near_best_auroc"] & candidates["candidate_pareto_efficient"]
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
    candidates.loc[ranking.index, "candidate_dual_metric_rank"] = np.arange(1, len(ranking) + 1)
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
                component_cache.get(cache_key)
                if component_cache is not None
                else None
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
        models.append(fit_ensemble_with_params(
            X,
            y,
            param_list,
            seed,
            seed_bagging_repeats=seed_bagging_repeats,
            component_cache=component_cache,
            data_cache_key=data_cache_key,
        ))

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
            delayed(fit_fold)(fold_idx, fit_idx)
            for fold_idx, (fit_idx, _) in splits
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
    models = [fit_ensemble_with_params(
        X,
        y,
        param_list,
        seed,
        seed_bagging_repeats=1,
        component_cache=component_cache,
        data_cache_key=data_cache_key,
    )]
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
            delayed(fit_fold)(fold_idx, fit_idx)
            for fold_idx, (fit_idx, _) in splits
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
        rows.append({
            "finalization_strategy": recipe,
            "validation_auroc": metrics["auroc"],
            "validation_auprc": metrics["auprc"],
            "validation_auprc_lift": metrics["auprc_lift"],
            "validation_selection_score": validation_model_selection_score(
                metrics,
                VALIDATION_MODEL_SELECTION_METRIC,
            ),
            "model_members": finalization_model_member_count(recipe, param_list),
        })
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
    strategy_candidates.loc[selected_mask, "candidate_finalization_strategy"] = finalization_name
    strategy_candidates.loc[selected_mask, "candidate_finalization_validation_auroc"] = metrics[
        "auroc"
    ]
    strategy_candidates.loc[selected_mask, "candidate_finalization_validation_auprc"] = metrics[
        "auprc"
    ]
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
    strategy_candidates.loc[selected_mask, "candidate_finalization_robust_auroc"] = robust_auroc
    strategy_candidates.loc[selected_mask, "candidate_finalization_robust_auprc"] = robust_auprc
    strategy_candidates.loc[
        selected_mask,
        "candidate_finalization_model_members",
    ] = selected_strategy["finalization_model_members"]
    strategy_candidates.loc[
        selected_mask,
        "candidate_finalization_candidates_json",
    ] = selected_strategy["finalization_candidates_json"]
    return selected_strategy, model, val_prob, strategy_candidates, finalization_candidates


def compute_tuning_cv_scores(
    X,
    y,
    params,
    model_seed,
    cv_splits,
    search_n_jobs,
    cache_signature,
):
    # cache_signature is intentionally part of the joblib key.
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
    if PIPELINE_CACHE_ENABLED
    else compute_tuning_cv_scores
)


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
        for preset in presets[:int(n_trials)]:
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
                completed % OPTUNA_PROGRESS_INTERVAL == 0
                or completed == int(n_trials)
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
            "n_features_after_mice_preprocessing": int(len(preprocess_step.encoded_columns_)),
            "feature_selection_k": select_step.k,
            "n_features_after_feature_selection": int(len(select_step.selected_indices_)),
            "dropped_missingness_columns": "|".join(missing_step.dropped_columns_),
            "dropped_unusable_columns": "|".join(preprocess_step.unusable_columns_),
            "numeric_columns_after_filter": "|".join(preprocess_step.numeric_columns_),
            "categorical_columns_after_filter": "|".join(preprocess_step.categorical_columns_),
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
    selected_indices = getattr(select_step, "selected_indices_", np.arange(len(transformed_feature_names)))
    selected_feature_names = [
        transformed_feature_names[idx]
        for idx in selected_indices
        if idx < len(transformed_feature_names)
    ]
    return {
        "n_features_before_missingness_filter": int(X.shape[1]),
        "n_features_after_missingness_filter": int(len(missing_step.keep_columns_)),
        "n_features_after_mice_preprocessing": int(len(preprocess_step.encoded_columns_)),
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
        if contributions.ndim == 2 and contributions.shape[1] == X_selected.shape[1] + 1:
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
    selected_indices = getattr(select_step, "selected_indices_", np.arange(len(feature_names)))
    selected_feature_names = [
        feature_names[idx]
        for idx in selected_indices
        if idx < len(feature_names)
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
                fold_importance[feature] = fold_importance.get(feature, 0.0) + float(value)
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
            feature: rank
            for rank, (feature, _) in enumerate(ranked_features, start=1)
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

    ranking = pd.DataFrame([
        {
            "feature": feature,
            "mean_abs_shap": importance / max(1, len(y)),
            "mean_normalized_shap": float(np.mean(normalized_importance[feature])),
            "mean_shap_percentile_rank": float(np.mean(percentile_rank_score[feature])),
            "top_n_fold_frequency": float(np.mean(top_n_frequency[feature])),
        }
        for feature, importance in weighted_importance.items()
        if feature in X.columns
    ])
    if ranking.empty:
        ranking = pd.DataFrame({
            "feature": list(X.columns),
            "mean_abs_shap": np.zeros(X.shape[1], dtype=float),
            "mean_normalized_shap": np.zeros(X.shape[1], dtype=float),
            "mean_shap_percentile_rank": np.zeros(X.shape[1], dtype=float),
            "top_n_fold_frequency": np.zeros(X.shape[1], dtype=float),
        })
    max_normalized = float(ranking["mean_normalized_shap"].max())
    ranking["normalized_shap_magnitude_score"] = (
        ranking["mean_normalized_shap"] / max_normalized
        if max_normalized > 0
        else 0.0
    )
    ranking["shap_stability_score"] = (
        0.50 * ranking["mean_shap_percentile_rank"]
        + 0.30 * ranking["top_n_fold_frequency"]
        + 0.20 * ranking["normalized_shap_magnitude_score"]
    )
    ranking = (
        ranking
        .sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    selected_features = ranking.head(int(top_n))["feature"].tolist()
    if len(selected_features) < int(top_n):
        selected = set(selected_features)
        selected_features.extend([
            col for col in X.columns
            if col not in selected
        ][: int(top_n) - len(selected_features)])
    fold_info = pd.DataFrame(fold_rows)
    return selected_features, ranking, fold_info


def ensure_model_variant_column(table, default=MODEL_VARIANT_ORIGINAL):
    if table.empty:
        return table
    table = table.copy()
    if "model_variant" not in table.columns:
        insert_at = 2 if "feature_set" in table.columns else len(table.columns)
        table.insert(insert_at, "model_variant", default)
    table["model_variant"] = table["model_variant"].fillna(default)
    return table


def feature_source_candidates_for_landmark(current_feature_set):
    candidates = current_feature_set.get("candidate_variants") or [current_feature_set]
    return candidates


def cumulative_member_feature_sets(feature_set):
    if feature_set not in LANDMARK_ORDER:
        return [feature_set]
    return LANDMARK_ORDER[:LANDMARK_ORDER.index(feature_set) + 1]


TEMPORAL_ENSEMBLE_CANDIDATE_COLUMNS = [
    "target",
    "feature_set",
    "model_variant",
    "candidate_name",
    "members",
    "member_weights",
    "current_model_weight",
    "selection_split",
    "selection_auroc",
    "selection_auprc",
    "selection_prevalence",
    "selection_auprc_lift",
    "selection_score",
    "selected",
    "selection_rule",
    "test_labels_used_for_selection",
]


def target_uses_temporal_landmark_ensemble(target):
    target_family = str(target).split("_", 1)[0].lower()
    return target_family in {
        family.lower() for family in LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES
    }


def temporal_landmark_candidate_weights(members):
    member_count = len(members)
    if member_count < 2:
        return []

    candidates = OrderedDict()

    def add_candidate(name, weights):
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        key = tuple(np.round(weights, 12))
        if key not in candidates:
            candidates[key] = {
                "candidate_name": name,
                "weights": weights,
            }

    add_candidate("equal_cumulative_average", np.ones(member_count, dtype=float))
    add_candidate(
        "recency_weighted_cumulative_average",
        np.arange(1, member_count + 1, dtype=float),
    )
    for current_weight in LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES:
        prior_weight = (1.0 - current_weight) / (member_count - 1)
        weights = np.full(member_count, prior_weight, dtype=float)
        weights[-1] = current_weight
        add_candidate(
            f"current_weight_{current_weight:.3f}",
            weights,
        )
        preoperative_anchor_weights = np.zeros(member_count, dtype=float)
        preoperative_anchor_weights[0] = 1.0 - current_weight
        preoperative_anchor_weights[-1] = current_weight
        add_candidate(
            f"preoperative_anchor_current_weight_{current_weight:.3f}",
            preoperative_anchor_weights,
        )
    return list(candidates.values())


def write_temporal_landmark_wrapper(
    model_summary,
    target,
    feature_set,
    model_variant,
    landmark_day,
    members,
    weights,
    selected_candidate_name,
    selection_split,
    selection_rule,
    validation_recalibrator,
    selected_threshold,
    train_recalibrator=None,
    train_selected_threshold=None,
):
    """Persist a wrapper without mutating any underlying member artifact."""
    if not RUN_FINAL_MODELS:
        return None

    member_artifacts = []
    for member in members:
        mask = (
            model_summary["target"].astype(str).eq(str(target))
            & model_summary["feature_set"].astype(str).eq(str(member))
            & model_summary["model_variant"].astype(str).eq(str(model_variant))
        )
        matches = model_summary.loc[mask]
        if len(matches) != 1:
            raise RuntimeError(
                "Temporal wrapper requires exactly one base model summary row for "
                f"{target}/{member}/{model_variant}; found {len(matches)}."
            )
        member_row = matches.iloc[0]
        model_path = Path(str(member_row.get("single_model_model_path", member_row["model_path"])))
        metadata_path = Path(str(member_row.get(
            "single_model_metadata_path",
            member_row["metadata_path"],
        )))
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing temporal member artifact: {model_path} or {metadata_path}"
            )
        member_artifacts.append({
            "feature_set": str(member),
            "base_model_relative_path": os.path.relpath(model_path, MODEL_DIR),
            "base_model_sha256": file_sha256(model_path),
            "base_metadata_relative_path": os.path.relpath(metadata_path, MODEL_DIR),
            "base_metadata_sha256": file_sha256(metadata_path),
            "base_model_path_at_creation": str(model_path),
            "base_metadata_path_at_creation": str(metadata_path),
        })

    weight_map = OrderedDict(
        (str(member), float(weight)) for member, weight in zip(members, weights)
    )
    variant_suffix = "" if model_variant == MODEL_VARIANT_ORIGINAL else f"__{slugify(model_variant)}"
    artifact_slug = (
        f"{slugify(target)}__{slugify(feature_set)}{variant_suffix}"
        "__temporal_landmark_ensemble"
    )
    wrapper_path = MODEL_DIR / f"{artifact_slug}.pkl"
    metadata_path = MODEL_DIR / f"{artifact_slug}_metadata.json"
    bundle = {
        "artifact_type": "temporal_landmark_ensemble_wrapper",
        "target": target,
        "feature_set": feature_set,
        "model_variant": model_variant,
        "landmark_day": jsonable(landmark_day),
        "ordered_member_artifacts": member_artifacts,
        "member_artifact_path_resolution": "resolve_relative_to_wrapper_directory",
        "ordered_members": [str(member) for member in members],
        "member_weights": weight_map,
        "raw_score_rule": "sum(member_weight * member_raw_class1_probability)",
        "reported_probability_source": "temporal_landmark_raw",
        "validation_selection_recalibrator": validation_recalibrator,
        "validation_recalibrator_metadata": probability_recalibrator_metadata(
            validation_recalibrator
        ),
        "train_oof_diagnostic_recalibrator": train_recalibrator,
        "train_oof_recalibrator_metadata": (
            probability_recalibrator_metadata(train_recalibrator)
            if train_recalibrator is not None
            else None
        ),
        "threshold": float(selected_threshold["candidate_threshold"]),
        "threshold_objective": selected_threshold["candidate_objective"],
        "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
        "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
        "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
        "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
        "threshold_selection_split": "validation_temporal_landmark_ensemble",
        "threshold_probability_source_provenance": probability_source_provenance(
            "temporal_landmark_raw"
        ),
        "threshold_selection_labels_provenance": "validation_labels",
        "threshold_test_labels_used_for_selection": False,
        "train_oof_threshold": (
            float(train_selected_threshold["candidate_threshold"])
            if train_selected_threshold is not None
            else None
        ),
        "train_oof_threshold_objective": (
            train_selected_threshold["candidate_objective"]
            if train_selected_threshold is not None
            else None
        ),
        "selected_weight_candidate": selected_candidate_name,
        "weight_selection_split": selection_split,
        "weight_selection_rule": selection_rule,
        "weight_selection_test_labels_used": False,
        "test_labels_used_for_selection": False,
    }
    with open(wrapper_path, "wb") as handle:
        cloudpickle.dump(bundle, handle)
    metadata = {
        key: jsonable(value)
        for key, value in bundle.items()
        if key not in {
            "validation_selection_recalibrator",
            "train_oof_diagnostic_recalibrator",
        }
    }
    metadata["wrapper_path"] = str(wrapper_path)
    metadata["wrapper_relative_path"] = os.path.relpath(wrapper_path, OUTPUT_DIR)
    metadata["wrapper_path_resolution"] = "resolve_relative_to_output_directory"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=jsonable)
    return {
        "model_path": str(wrapper_path),
        "metadata_path": str(metadata_path),
        "model_relative_path": os.path.relpath(wrapper_path, OUTPUT_DIR),
        "metadata_relative_path": os.path.relpath(metadata_path, OUTPUT_DIR),
    }


def apply_cumulative_landmark_prediction_ensemble(
    holdout_results_df,
    holdout_predictions_df,
    holdout_threshold_candidates_df,
    holdout_train_cv_threshold_candidates_df,
    holdout_model_summary_df,
):
    if (
        not LANDMARK_CUMULATIVE_PREDICTION_ENSEMBLE
        or holdout_results_df.empty
        or holdout_predictions_df.empty
    ):
        return (
            holdout_results_df,
            holdout_predictions_df,
            holdout_threshold_candidates_df,
            holdout_train_cv_threshold_candidates_df,
            holdout_model_summary_df,
            pd.DataFrame(columns=TEMPORAL_ENSEMBLE_CANDIDATE_COLUMNS),
        )

    results = holdout_results_df.copy()
    predictions = holdout_predictions_df.copy()
    threshold_candidates = holdout_threshold_candidates_df.copy()
    train_threshold_candidates = holdout_train_cv_threshold_candidates_df.copy()
    model_summary = holdout_model_summary_df.copy()
    results = ensure_model_variant_column(results)
    predictions = ensure_model_variant_column(predictions)
    threshold_candidates = ensure_model_variant_column(threshold_candidates)
    train_threshold_candidates = ensure_model_variant_column(train_threshold_candidates)
    model_summary = ensure_model_variant_column(model_summary)
    added_threshold_rows = []
    added_train_threshold_rows = []

    for artifact_col in ["model_path", "metadata_path"]:
        if artifact_col in model_summary.columns:
            base_col = f"single_model_{artifact_col}"
            if base_col not in model_summary.columns:
                model_summary[base_col] = model_summary[artifact_col]

    for source_col in [
        "reported_probability_source",
        "raw_probability",
        "probability",
        "calibration_probability",
        "threshold",
        "y_pred",
    ]:
        if source_col in predictions.columns and f"single_model_{source_col}" not in predictions.columns:
            predictions[f"single_model_{source_col}"] = predictions[source_col]

    single_metric_prefixes = (
        "validation_",
        "test_",
        "raw_validation_",
        "raw_test_",
        "calibrated_validation_",
        "calibrated_test_",
    )
    for col in list(results.columns):
        if col.startswith(single_metric_prefixes) or col in {
            "reported_probability_source",
            "selected_threshold_objective",
            "selected_threshold",
            "selected_threshold_validation_score",
            "train_cv_selected_threshold_objective",
            "train_cv_selected_threshold",
            "train_cv_selected_threshold_score",
            "ensemble_train_cv_oof_auroc",
            "ensemble_train_cv_oof_auprc",
            "ensemble_train_cv_oof_auprc_lift",
            "validation_model_selection_score",
        }:
            single_col = f"single_model_{col}"
            if single_col not in results.columns:
                results[single_col] = results[col]

    results["probability_ensemble_method"] = "single_landmark_model"
    results["probability_ensemble_members"] = results["feature_set"]
    results["probability_ensemble_weights"] = results["feature_set"].map(
        lambda member: json.dumps({str(member): 1.0})
    )
    results["probability_ensemble_selected_candidate"] = "single_current_landmark"
    results["probability_ensemble_selection_split"] = "not_applicable"
    results["probability_ensemble_selection_rule"] = "not_applicable"
    results["probability_ensemble_test_labels_used_for_selection"] = False
    predictions["probability_ensemble_method"] = "single_landmark_model"
    predictions["probability_ensemble_members"] = predictions["feature_set"]
    predictions["probability_ensemble_weights"] = predictions["feature_set"].map(
        lambda member: json.dumps({str(member): 1.0})
    )
    predictions["probability_ensemble_selected_candidate"] = "single_current_landmark"
    predictions["probability_ensemble_selection_split"] = "not_applicable"
    predictions["probability_ensemble_test_labels_used_for_selection"] = False

    available_pairs = {
        (
            str(row["target"]),
            str(row["feature_set"]),
            str(row.get("model_variant", MODEL_VARIANT_ORIGINAL)),
        )
        for _, row in results.iterrows()
    }
    ensemble_candidate_rows = []

    for row_idx, row in results.iterrows():
        target = str(row["target"])
        feature_set = str(row["feature_set"])
        model_variant = str(row.get("model_variant", MODEL_VARIANT_ORIGINAL))
        if not target_uses_temporal_landmark_ensemble(target):
            continue
        members = [
            member for member in cumulative_member_feature_sets(feature_set)
            if (target, member, model_variant) in available_pairs
        ]
        if len(members) < 2 or members[-1] != feature_set:
            continue

        split_frames = {}
        for split_name in ["train_oof", "validation", "test"]:
            current_mask = (
                (predictions["target"] == target)
                & (predictions["feature_set"] == feature_set)
                & (predictions["dataset_split"] == split_name)
                & (predictions["model_variant"] == model_variant)
            )
            current = predictions.loc[
                current_mask,
                ["source_row_index", "y_true"],
            ].drop_duplicates("source_row_index").copy()
            if current.empty:
                if split_name == "train_oof":
                    continue
                split_frames = {}
                break
            current["_prediction_index"] = current.index
            wide = current
            used_members = []
            for member in members:
                member_mask = (
                    (predictions["target"] == target)
                    & (predictions["feature_set"] == member)
                    & (predictions["dataset_split"] == split_name)
                    & (predictions["model_variant"] == model_variant)
                )
                member_prob = predictions.loc[
                    member_mask,
                    ["source_row_index", "y_true", "single_model_raw_probability"],
                ].drop_duplicates("source_row_index").rename(columns={
                    "y_true": f"{member}__y_true",
                    "single_model_raw_probability": member,
                })
                if member_prob.empty:
                    continue
                wide = wide.merge(
                    member_prob,
                    on="source_row_index",
                    how="left",
                    validate="one_to_one",
                )
                used_members.append(member)

            member_label_columns = [f"{member}__y_true" for member in used_members]
            if used_members != members or wide[members + member_label_columns].isna().any().any():
                if split_name == "train_oof":
                    continue
                split_frames = {}
                break
            labels_match = all(
                np.array_equal(
                    wide["y_true"].astype(int).to_numpy(),
                    wide[label_col].astype(int).to_numpy(),
                )
                for label_col in member_label_columns
            )
            if not labels_match:
                if split_name == "train_oof":
                    continue
                split_frames = {}
                break
            split_frames[split_name] = {
                "frame": wide,
                "members": used_members,
            }

        if {"validation", "test"} - set(split_frames):
            continue

        val_frame = split_frames["validation"]["frame"]
        test_frame = split_frames["test"]["frame"]
        used_members = split_frames["test"]["members"]
        selection_split = "train_oof" if "train_oof" in split_frames else "validation"
        selection_frame = split_frames[selection_split]["frame"]
        y_selection = selection_frame["y_true"].astype(int).to_numpy()
        y_val = val_frame["y_true"].astype(int).to_numpy()
        y_test = test_frame["y_true"].astype(int).to_numpy()
        if (
            len(np.unique(y_selection)) < 2
            or len(np.unique(y_val)) < 2
        ):
            continue

        selection_member_probabilities = selection_frame[used_members].to_numpy(dtype=float)
        val_member_probabilities = val_frame[used_members].to_numpy(dtype=float)
        test_member_probabilities = test_frame[used_members].to_numpy(dtype=float)
        candidate_evaluations = []
        for candidate in temporal_landmark_candidate_weights(used_members):
            weights = candidate["weights"]
            candidate_selection_probability = selection_member_probabilities @ weights
            candidate_metrics = classification_metrics(
                y_selection,
                candidate_selection_probability,
                threshold=0.5,
            )
            candidate_evaluations.append({
                **candidate,
                "selection_probability": candidate_selection_probability,
                "validation_probability": val_member_probabilities @ weights,
                "test_probability": test_member_probabilities @ weights,
                "selection_metrics": candidate_metrics,
                "selection_score": validation_model_selection_score(
                    candidate_metrics,
                    VALIDATION_MODEL_SELECTION_METRIC,
                ),
            })

        if not candidate_evaluations:
            continue
        maximum_selection_auroc = max(
            candidate["selection_metrics"]["auroc"]
            for candidate in candidate_evaluations
        )
        near_best_candidates = [
            candidate for candidate in candidate_evaluations
            if candidate["selection_metrics"]["auroc"]
            >= maximum_selection_auroc - LANDMARK_TEMPORAL_AUROC_TOLERANCE
        ]
        selected_candidate = max(
            near_best_candidates,
            key=lambda candidate: (
                candidate["selection_metrics"]["auprc"],
                candidate["selection_metrics"]["auroc"],
                candidate["weights"][-1],
            ),
        )
        selection_rule = (
            f"{selection_split}_max_auroc_then_max_auprc_within_"
            f"{LANDMARK_TEMPORAL_AUROC_TOLERANCE:.3f}_auroc"
        )
        for candidate in candidate_evaluations:
            metrics = candidate["selection_metrics"]
            candidate_weights = {
                member: float(weight)
                for member, weight in zip(used_members, candidate["weights"])
            }
            ensemble_candidate_rows.append({
                "target": target,
                "feature_set": feature_set,
                "model_variant": model_variant,
                "candidate_name": candidate["candidate_name"],
                "members": "|".join(used_members),
                "member_weights": json.dumps(candidate_weights, sort_keys=True),
                "current_model_weight": float(candidate["weights"][-1]),
                "selection_split": selection_split,
                "selection_auroc": metrics["auroc"],
                "selection_auprc": metrics["auprc"],
                "selection_prevalence": metrics["prevalence"],
                "selection_auprc_lift": metrics["auprc_lift"],
                "selection_score": candidate["selection_score"],
                "selected": candidate is selected_candidate,
                "selection_rule": selection_rule,
                "test_labels_used_for_selection": False,
            })

        train_frame = (
            split_frames["train_oof"]["frame"]
            if "train_oof" in split_frames
            else None
        )
        train_raw_prob = (
            np.asarray(selected_candidate["selection_probability"], dtype=float)
            if selection_split == "train_oof"
            else None
        )
        train_diagnostic_calibrator = None
        train_cal_prob = None
        train_selected_threshold = None
        train_selected_candidate_rows = None
        if train_frame is not None and train_raw_prob is not None:
            y_train_oof = train_frame["y_true"].astype(int).to_numpy()
            train_diagnostic_calibrator = fit_probability_recalibrator(
                y_train_oof,
                train_raw_prob,
            )
            train_cal_prob = apply_probability_recalibrator(
                train_diagnostic_calibrator,
                train_raw_prob,
            )
            train_selected_threshold, train_selected_candidate_rows = (
                select_threshold_from_oof(
                    y_train_oof,
                    train_raw_prob,
                    calibration_prob=train_cal_prob,
                )
            )

        val_raw_prob = selected_candidate["validation_probability"]
        test_raw_prob = selected_candidate["test_probability"]

        diagnostic_calibrator = fit_probability_recalibrator(y_val, val_raw_prob)
        val_cal_prob = apply_probability_recalibrator(diagnostic_calibrator, val_raw_prob)
        test_cal_prob = apply_probability_recalibrator(diagnostic_calibrator, test_raw_prob)

        raw_eval = evaluate_probability_source(
            y_val,
            val_raw_prob,
            y_test,
            test_raw_prob,
            "temporal_landmark_raw",
        )
        calibrated_eval = evaluate_probability_source(
            y_val,
            val_cal_prob,
            y_test,
            test_cal_prob,
            "temporal_landmark_calibrated",
        )
        selected_report = raw_eval
        selected_threshold = selected_report["selected_threshold"]
        threshold = float(selected_report["threshold"])
        val_metrics = selected_report["validation_metrics"]
        test_metrics = selected_report["test_metrics"]
        raw_val_metrics = raw_eval["validation_metrics"]
        raw_test_metrics = raw_eval["test_metrics"]
        calibrated_val_metrics = calibrated_eval["validation_metrics"]
        calibrated_test_metrics = calibrated_eval["test_metrics"]
        member_label = "|".join(used_members)
        selected_weight_map = {
            member: float(weight)
            for member, weight in zip(used_members, selected_candidate["weights"])
        }
        selected_weight_label = json.dumps(selected_weight_map, sort_keys=True)

        temporal_prediction_splits = []
        if train_frame is not None and train_selected_threshold is not None:
            temporal_prediction_splits.append((
                "train_oof",
                train_frame,
                train_raw_prob,
                train_cal_prob,
                float(train_selected_threshold["candidate_threshold"]),
            ))
        temporal_prediction_splits.extend([
            ("validation", val_frame, val_raw_prob, val_cal_prob, threshold),
            ("test", test_frame, test_raw_prob, test_cal_prob, threshold),
        ])
        for split_name, frame, raw_prob, cal_prob, split_threshold in temporal_prediction_splits:
            pred_index = frame["_prediction_index"].to_numpy(dtype=int)
            predictions.loc[pred_index, "reported_probability_source"] = "temporal_landmark_raw"
            predictions.loc[pred_index, "raw_probability"] = raw_prob
            predictions.loc[pred_index, "probability"] = raw_prob
            predictions.loc[pred_index, "calibration_probability"] = cal_prob
            predictions.loc[pred_index, "threshold"] = split_threshold
            predictions.loc[pred_index, "y_pred"] = (raw_prob >= split_threshold).astype(int)
            predictions.loc[pred_index, "probability_ensemble_method"] = (
                "cross_validated_temporal_landmark_convex_average"
            )
            predictions.loc[pred_index, "probability_ensemble_members"] = member_label
            predictions.loc[pred_index, "probability_ensemble_weights"] = selected_weight_label
            predictions.loc[pred_index, "probability_ensemble_selected_candidate"] = (
                selected_candidate["candidate_name"]
            )
            predictions.loc[pred_index, "probability_ensemble_selection_split"] = selection_split
            predictions.loc[
                pred_index,
                "probability_ensemble_test_labels_used_for_selection",
            ] = False

        results.loc[row_idx, "reported_probability_source"] = "temporal_landmark_raw"
        results.loc[row_idx, "probability_ensemble_method"] = (
            "cross_validated_temporal_landmark_convex_average"
        )
        results.loc[row_idx, "probability_ensemble_members"] = member_label
        results.loc[row_idx, "probability_ensemble_weights"] = selected_weight_label
        results.loc[row_idx, "probability_ensemble_selected_candidate"] = (
            selected_candidate["candidate_name"]
        )
        results.loc[row_idx, "probability_ensemble_selection_split"] = selection_split
        results.loc[row_idx, "probability_ensemble_selection_rule"] = selection_rule
        results.loc[
            row_idx,
            "probability_ensemble_test_labels_used_for_selection",
        ] = False
        results.loc[row_idx, "calibration_diagnostic_method"] = diagnostic_calibrator["method"]
        results.loc[row_idx, "selected_threshold_objective"] = selected_threshold["candidate_objective"]
        results.loc[row_idx, "selected_threshold"] = threshold
        results.loc[row_idx, "selected_threshold_validation_score"] = selected_threshold[
            "candidate_selection_score"
        ]
        results.loc[row_idx, "validation_model_selection_score"] = selected_report[
            "validation_model_selection_score"
        ]
        results.loc[row_idx, "threshold_candidate_policy"] = THRESHOLD_CANDIDATE_POLICY
        results.loc[row_idx, "threshold_tie_break_policy"] = THRESHOLD_TIE_BREAK_POLICY
        results.loc[row_idx, "threshold_minimum_sensitivity"] = THRESHOLD_MIN_SENSITIVITY
        results.loc[row_idx, "threshold_operating_point_status"] = THRESHOLD_OPERATING_POINT_STATUS
        results.loc[row_idx, "threshold_selection_split"] = "validation_temporal_landmark_ensemble"
        results.loc[row_idx, "threshold_probability_source_provenance"] = (
            probability_source_provenance("temporal_landmark_raw")
        )
        results.loc[row_idx, "threshold_selection_labels_provenance"] = "validation_labels"
        results.loc[row_idx, "threshold_test_labels_used_for_selection"] = False
        if train_frame is not None and train_selected_threshold is not None:
            train_metrics = classification_metrics(
                train_frame["y_true"].astype(int).to_numpy(),
                train_raw_prob,
                float(train_selected_threshold["candidate_threshold"]),
            )
            results.loc[row_idx, "train_cv_selected_threshold_objective"] = (
                train_selected_threshold["candidate_objective"]
            )
            results.loc[row_idx, "train_cv_selected_threshold"] = float(
                train_selected_threshold["candidate_threshold"]
            )
            results.loc[row_idx, "train_cv_selected_threshold_score"] = (
                train_selected_threshold["candidate_selection_score"]
            )
            results.loc[row_idx, "ensemble_train_cv_oof_auroc"] = train_metrics["auroc"]
            results.loc[row_idx, "ensemble_train_cv_oof_auprc"] = train_metrics["auprc"]
            results.loc[row_idx, "ensemble_train_cv_oof_auprc_lift"] = train_metrics["auprc_lift"]
        results.loc[row_idx, "raw_validation_ece"] = raw_val_metrics["ece"]
        results.loc[row_idx, "raw_test_ece"] = raw_test_metrics["ece"]

        for key, value in raw_val_metrics.items():
            results.loc[row_idx, f"raw_validation_{key}"] = value
            results.loc[row_idx, f"validation_{key}"] = value
        for key, value in raw_test_metrics.items():
            results.loc[row_idx, f"raw_test_{key}"] = value
            results.loc[row_idx, f"test_{key}"] = value
        for key, value in calibrated_val_metrics.items():
            results.loc[row_idx, f"calibrated_validation_{key}"] = value
        for key, value in calibrated_test_metrics.items():
            results.loc[row_idx, f"calibrated_test_{key}"] = value
        for key in CALIBRATION_DIAGNOSTIC_METRICS:
            results.loc[row_idx, f"validation_{key}"] = calibrated_val_metrics[key]
            results.loc[row_idx, f"test_{key}"] = calibrated_test_metrics[key]

        combined_threshold_candidates = raw_eval["threshold_candidates"].copy()
        combined_threshold_candidates.insert(0, "target", target)
        combined_threshold_candidates.insert(1, "feature_set", feature_set)
        combined_threshold_candidates.insert(2, "model_variant", model_variant)
        combined_threshold_candidates.insert(3, "landmark_day", row.get("landmark_day", np.nan))
        combined_threshold_candidates.insert(4, "selection_split", "validation_temporal_landmark_ensemble")
        added_threshold_rows.extend(combined_threshold_candidates.to_dict("records"))

        if train_selected_candidate_rows is not None:
            temporal_train_candidates = train_selected_candidate_rows.copy()
            temporal_train_candidates.insert(0, "probability_source", "temporal_landmark_raw")
            temporal_train_candidates.insert(
                1,
                "probability_source_provenance",
                probability_source_provenance("temporal_landmark_raw"),
            )
            temporal_train_candidates.insert(2, "selection_labels_provenance", "train_oof_labels")
            temporal_train_candidates.insert(0, "target", target)
            temporal_train_candidates.insert(1, "feature_set", feature_set)
            temporal_train_candidates.insert(2, "model_variant", model_variant)
            temporal_train_candidates.insert(3, "landmark_day", row.get("landmark_day", np.nan))
            temporal_train_candidates.insert(
                4,
                "selection_split",
                "train_oof_temporal_landmark_ensemble",
            )
            added_train_threshold_rows.extend(temporal_train_candidates.to_dict("records"))

        wrapper_paths = write_temporal_landmark_wrapper(
            model_summary=model_summary,
            target=target,
            feature_set=feature_set,
            model_variant=model_variant,
            landmark_day=row.get("landmark_day", np.nan),
            members=used_members,
            weights=selected_candidate["weights"],
            selected_candidate_name=selected_candidate["candidate_name"],
            selection_split=selection_split,
            selection_rule=selection_rule,
            validation_recalibrator=diagnostic_calibrator,
            selected_threshold=selected_threshold,
            train_recalibrator=train_diagnostic_calibrator,
            train_selected_threshold=train_selected_threshold,
        )
        if wrapper_paths is not None:
            results.loc[row_idx, "model_path"] = wrapper_paths["model_path"]
            results.loc[row_idx, "metadata_path"] = wrapper_paths["metadata_path"]
            results.loc[row_idx, "model_relative_path"] = wrapper_paths["model_relative_path"]
            results.loc[row_idx, "metadata_relative_path"] = wrapper_paths[
                "metadata_relative_path"
            ]
            summary_mask = (
                model_summary["target"].astype(str).eq(target)
                & model_summary["feature_set"].astype(str).eq(feature_set)
                & model_summary["model_variant"].astype(str).eq(model_variant)
            )
            if int(summary_mask.sum()) != 1:
                raise RuntimeError(
                    "Temporal wrapper could not identify one current model summary row for "
                    f"{target}/{feature_set}/{model_variant}."
                )
            model_summary.loc[summary_mask, "model_path"] = wrapper_paths["model_path"]
            model_summary.loc[summary_mask, "metadata_path"] = wrapper_paths["metadata_path"]
            model_summary.loc[summary_mask, "model_relative_path"] = wrapper_paths[
                "model_relative_path"
            ]
            model_summary.loc[summary_mask, "metadata_relative_path"] = wrapper_paths[
                "metadata_relative_path"
            ]
            model_summary.loc[summary_mask, "reported_probability_source"] = (
                "temporal_landmark_raw"
            )
            model_summary.loc[summary_mask, "selected_threshold_objective"] = (
                selected_threshold["candidate_objective"]
            )
            model_summary.loc[summary_mask, "selected_threshold"] = threshold
            model_summary.loc[summary_mask, "probability_ensemble_method"] = (
                "cross_validated_temporal_landmark_convex_average"
            )
            model_summary.loc[summary_mask, "probability_ensemble_members"] = member_label
            model_summary.loc[summary_mask, "probability_ensemble_weights"] = selected_weight_label

    if added_threshold_rows:
        threshold_candidates = pd.concat(
            [threshold_candidates, pd.DataFrame(added_threshold_rows)],
            ignore_index=True,
        )
    if added_train_threshold_rows:
        train_threshold_candidates = pd.concat(
            [train_threshold_candidates, pd.DataFrame(added_train_threshold_rows)],
            ignore_index=True,
        )

    ensemble_candidates = pd.DataFrame(
        ensemble_candidate_rows,
        columns=TEMPORAL_ENSEMBLE_CANDIDATE_COLUMNS,
    )
    return (
        results,
        predictions,
        threshold_candidates,
        train_threshold_candidates,
        model_summary,
        ensemble_candidates,
    )


def evaluate_feature_source_candidate(
    target_name,
    current_feature_name,
    candidate_item,
    keep_mask,
    train_pos,
    val_pos,
    y_train,
    y_val,
    base_seed,
):
    optimization_settings = optimization_settings_for_analysis(
        target_name,
        current_feature_name,
    )
    candidate_name = candidate_item["feature_set"]
    candidate_seed_name = candidate_item.get("seed_feature_set", candidate_name)
    if candidate_seed_name == current_feature_name:
        candidate_seed = int(base_seed)
    else:
        candidate_seed = int(
            base_seed
            + (zlib.crc32(f"{current_feature_name}|{candidate_seed_name}".encode("utf-8")) % 1009)
        )
    X_all_candidate = candidate_item["X"].loc[keep_mask].reset_index(drop=True)
    X_train_candidate = X_all_candidate.iloc[train_pos].reset_index(drop=True)
    X_val_candidate = X_all_candidate.iloc[val_pos].reset_index(drop=True)

    search = tune_hyperparameters(
        X_train_candidate,
        y_train,
        candidate_seed,
        n_trials=optimization_settings["optuna_trials"],
        feature_selection_candidates=optimization_settings[
            "feature_selection_candidates"
        ],
        cv_repeats=optimization_settings["cv_repeats"],
        modeling_method=optimization_settings["modeling_method"],
        analysis_key=optimization_settings["analysis_key"],
    )
    selected_strategy, validation_model, val_raw_prob, strategy_candidates = select_strategy_by_validation(
        search,
        X_train_candidate,
        y_train,
        X_val_candidate,
        y_val,
        candidate_seed + 29,
        candidate_trials=optimization_settings["validation_candidate_trials"],
        seed_bagging_repeats=optimization_settings["seed_bagging_repeats"],
        ensemble_top_k_candidates=optimization_settings[
            "ensemble_top_k_candidates"
        ],
        include_dual_metric_champions=optimization_settings[
            "include_dual_metric_champions"
        ],
    )
    (
        selected_strategy,
        validation_model,
        val_raw_prob,
        strategy_candidates,
        finalization_candidates,
    ) = finalize_selected_validation_strategy(
        selected_strategy,
        strategy_candidates,
        X_train_candidate,
        y_train,
        X_val_candidate,
        y_val,
        candidate_seed + 29,
        validation_model,
        val_raw_prob,
        optimization_settings["seed_bagging_repeats"],
        candidate_recipes=optimization_settings["finalization_candidates"],
    )
    selected_strategy_row = strategy_candidates.loc[
        strategy_candidates["candidate_strategy_name"] == selected_strategy["strategy_name"]
    ].iloc[0]
    validation_selection_score = float(
        selected_strategy_row.get(
            "candidate_robust_selection_score",
            selected_strategy_row["candidate_validation_selection_score"],
        )
    )
    validation_auroc = float(selected_strategy_row["candidate_validation_auroc"])
    validation_auprc = float(selected_strategy_row["candidate_validation_auprc"])
    validation_auprc_lift = float(selected_strategy_row.get("candidate_validation_auprc_lift", np.nan))

    return {
        "candidate_feature_source": candidate_name,
        "candidate_feature_variant": candidate_item.get("feature_variant", candidate_name),
        "candidate_landmark_day": candidate_item["landmark_day"],
        "candidate_columns": candidate_item["columns"],
        "candidate_seed": candidate_seed,
        "X_all": X_all_candidate,
        "X_train": X_train_candidate,
        "X_val": X_val_candidate,
        "search": search,
        "selected_strategy": selected_strategy,
        "validation_model": validation_model,
        "val_raw_prob": val_raw_prob,
        "strategy_candidates": strategy_candidates,
        "finalization_candidates": finalization_candidates,
        "validation_selection_score": validation_selection_score,
        "raw_validation_selection_score": float(
            selected_strategy_row["candidate_validation_selection_score"]
        ),
        "validation_reliability_weight": float(
            selected_strategy_row.get("candidate_validation_reliability_weight", 1.0)
        ),
        "validation_auroc": validation_auroc,
        "validation_auprc": validation_auprc,
        "validation_auprc_lift": validation_auprc_lift,
    }


def feature_set_item_for_analysis(feature_set_name):
    for item in ALL_FEATURE_SETS:
        if item["feature_set"] == feature_set_name:
            return item
    raise KeyError(feature_set_name)


def parse_candidate_param_list(params_json):
    param_list = json.loads(params_json)
    if isinstance(param_list, dict):
        param_list = [param_list]
    return [dict(params) for params in param_list]


def selected_original_strategy_row(strategy_table, target_name, feature_name):
    if strategy_table.empty:
        return None
    table = ensure_model_variant_column(strategy_table)
    mask = (
        (table["target"] == target_name)
        & (table["feature_set"] == feature_name)
        & (table["model_variant"] == MODEL_VARIANT_ORIGINAL)
        & (table["selected_strategy"].astype(bool))
    )
    rows = table.loc[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def evaluate_simplified_model_variant(original_row, original_strategy_row):
    target_name = original_row["target"]
    feature_name = original_row["feature_set"]
    feature_item = feature_set_item_for_analysis(feature_name)
    landmark_day = feature_item["landmark_day"]
    selected_feature_source = original_row.get("selected_feature_source", feature_name)
    if pd.isna(selected_feature_source):
        selected_feature_source = feature_name
    source_item = feature_set_item_for_analysis(selected_feature_source)
    selected_feature_source_day = source_item["landmark_day"]
    spec = TARGET_DEFINITIONS[target_name]
    split_seed_base = split_random_state_base(target_name, feature_name)
    final_refit_scope = final_refit_scope_for_analysis(target_name, feature_name)
    seed = analysis_seed(target_name, feature_name) + 700001
    optimization_settings = optimization_settings_for_analysis(target_name, feature_name)
    simplified_validation_candidate_trials = optimization_settings[
        "simplified_validation_candidate_trials"
    ]
    simplified_seed_bagging_repeats = optimization_settings[
        "simplified_seed_bagging_repeats"
    ]
    simplified_shap_ranking_mode = simplified_shap_ranking_mode_for_analysis(
        target_name,
        feature_name,
    )
    keep_mask, y_target = make_target(
        spec["stage_column"],
        spec["positive_rule"],
        spec.get("exclude_control_death_3m", False),
        spec.get("event_date_column"),
        target_landmark_day_for_policy(landmark_day),
    )
    source_indices = np.flatnonzero(keep_mask)
    train_pos, val_pos, test_pos, split_info = train_val_test_positions_for_analysis(
        target_name,
        spec,
        feature_name,
        source_indices,
        y_target,
    )
    y_train = y_target.iloc[train_pos].reset_index(drop=True)
    y_val = y_target.iloc[val_pos].reset_index(drop=True)
    y_test = y_target.iloc[test_pos].reset_index(drop=True)
    if simplified_shap_ranking_mode == "fixed_depth1":
        ranking_param_list = [
            preset_xgb_trials(y_train, SIMPLIFIED_MICE_ITER_CANDIDATES)[1]
        ]
        simplified_source_strategy_name = (
            "prespecified_regularized_depth1_cv_reference"
        )
    else:
        ranking_param_list = parse_candidate_param_list(
            original_strategy_row["candidate_params_json"]
        )
        simplified_source_strategy_name = (
            f"original_selected:{original_strategy_row['candidate_strategy_name']}"
        )
    X_all_full = source_item["X"].loc[keep_mask].reset_index(drop=True)
    X_train_full = X_all_full.iloc[train_pos].reset_index(drop=True)
    top_features, shap_ranking, shap_fold_info = cv_shap_top_base_features(
        X_train_full,
        y_train,
        ranking_param_list,
        seed + 101,
        SIMPLIFIED_TOP_N_FEATURES,
        seed_bagging_repeats=simplified_seed_bagging_repeats,
    )
    X_all = X_all_full.loc[:, top_features].reset_index(drop=True)
    X_train = X_all.iloc[train_pos].reset_index(drop=True)
    X_val = X_all.iloc[val_pos].reset_index(drop=True)
    X_test = X_all.iloc[test_pos].reset_index(drop=True)

    search = None
    if SIMPLIFIED_RETUNE:
        search = tune_hyperparameters(
            X_train,
            y_train,
            seed + 1009,
            n_trials=optimization_settings["simplified_optuna_trials"],
            mice_iter_candidates=SIMPLIFIED_MICE_ITER_CANDIDATES,
            feature_selection_candidates=optimization_settings[
                "simplified_feature_selection_candidates"
            ],
            cv_repeats=optimization_settings["cv_repeats"],
            modeling_method=optimization_settings["modeling_method"],
            analysis_key=optimization_settings["analysis_key"],
        )
        selected_strategy, validation_model, val_raw_prob, strategy_candidates = (
            select_strategy_by_validation(
                search,
                X_train,
                y_train,
                X_val,
                y_val,
                seed + 1031,
                candidate_trials=simplified_validation_candidate_trials,
                seed_bagging_repeats=simplified_seed_bagging_repeats,
                ensemble_top_k_candidates=optimization_settings[
                    "ensemble_top_k_candidates"
                ],
                include_dual_metric_champions=optimization_settings[
                    "include_dual_metric_champions"
                ],
            )
        )
        selected_strategy_row = strategy_candidates.loc[
            strategy_candidates["candidate_strategy_name"] == selected_strategy["strategy_name"]
        ].iloc[0]
        strategy_validation_metrics = classification_metrics(y_val, val_raw_prob, 0.5)
        raw_strategy_selection_score = float(
            selected_strategy_row["candidate_validation_selection_score"]
        )
        strategy_selection_score = float(
            selected_strategy_row.get(
                "candidate_robust_selection_score",
                raw_strategy_selection_score,
            )
        )
        strategy_validation_weight = float(
            selected_strategy_row.get("candidate_validation_reliability_weight", 1.0)
        )
        best_params = dict(search.best_params_)
        best_trial = search.study_.best_trial
        best_cv_mean_auroc = best_trial.user_attrs.get("cv_mean_auroc", search.best_score_)
        best_cv_std_auroc = best_trial.user_attrs.get("cv_std_auroc", np.nan)
        best_cv_mean_auprc = best_trial.user_attrs.get("cv_mean_auprc", np.nan)
        best_cv_std_auprc = best_trial.user_attrs.get("cv_std_auprc", np.nan)
        best_cv_auprc_lift = best_trial.user_attrs.get("cv_auprc_lift", np.nan)
    else:
        ensemble_param_list = ranking_param_list
        validation_model = fit_ensemble_with_params(
            X_train,
            y_train,
            ensemble_param_list,
            seed + 29,
            seed_bagging_repeats=simplified_seed_bagging_repeats,
        )
        val_raw_prob = validation_model.predict_proba(X_val)[:, 1]
        strategy_validation_metrics = classification_metrics(y_val, val_raw_prob, 0.5)
        raw_strategy_selection_score = validation_model_selection_score(
            strategy_validation_metrics,
            VALIDATION_MODEL_SELECTION_METRIC,
        )
        strategy_selection_score = float(raw_strategy_selection_score)
        strategy_validation_weight = 1.0
        selected_strategy_name = (
            f"{simplified_source_strategy_name}_cv_shap_top"
            f"{SIMPLIFIED_TOP_N_FEATURES}"
        )
        selected_strategy_type = (
            f"{simplified_shap_ranking_mode}_reference_after_train_cv_shap_top_features"
        )
        strategy_candidates = pd.DataFrame([{
            "candidate_strategy_name": selected_strategy_name,
            "candidate_strategy_type": selected_strategy_type,
            "candidate_requested_top_k": int(len(ensemble_param_list)),
            "candidate_actual_top_k": int(len(ensemble_param_list)),
            "candidate_optuna_rank": np.nan,
            "candidate_optuna_trial_number": np.nan,
            "candidate_optuna_objective_value": np.nan,
            "candidate_cv_mean_auroc": np.nan,
            "candidate_cv_std_auroc": np.nan,
            "candidate_cv_mean_auprc": np.nan,
            "candidate_cv_auprc_lift": np.nan,
            "candidate_validation_selection_metric": f"raw_validation_{VALIDATION_MODEL_SELECTION_METRIC}",
            "candidate_validation_selection_score": raw_strategy_selection_score,
            "candidate_cv_selection_score": np.nan,
            "candidate_validation_reliability_weight": 1.0,
            "candidate_robust_selection_score": strategy_selection_score,
            "candidate_selection_metric": f"raw_validation_{VALIDATION_MODEL_SELECTION_METRIC}",
            "candidate_params_json": json.dumps(
                [
                    {key: jsonable(value) for key, value in params.items()}
                    for params in ensemble_param_list
                ],
                ensure_ascii=False,
            ),
            **{
                f"candidate_validation_{key}": value
                for key, value in strategy_validation_metrics.items()
            },
        }])
        selected_strategy = {
            "strategy_name": selected_strategy_name,
            "strategy_type": selected_strategy_type,
            "param_list": ensemble_param_list,
        }
        best_params = dict(ensemble_param_list[0])
        best_cv_mean_auroc = np.nan
        best_cv_std_auroc = np.nan
        best_cv_mean_auprc = np.nan
        best_cv_std_auprc = np.nan
        best_cv_auprc_lift = np.nan

    ensemble_param_list = selected_strategy["param_list"]
    ensemble_top_k_actual = len(ensemble_param_list)
    selected_model_family = model_family_label(ensemble_param_list)
    simplified_validation_fit_seed = seed + (1031 if search is not None else 29)
    (
        selected_strategy,
        validation_model,
        val_raw_prob,
        strategy_candidates,
        finalization_candidates,
    ) = finalize_selected_validation_strategy(
        selected_strategy,
        strategy_candidates,
        X_train,
        y_train,
        X_val,
        y_val,
        simplified_validation_fit_seed,
        validation_model,
        val_raw_prob,
        simplified_seed_bagging_repeats,
        candidate_recipes=optimization_settings["finalization_candidates"],
    )
    strategy_validation_metrics = classification_metrics(y_val, val_raw_prob, 0.5)
    raw_strategy_selection_score = validation_model_selection_score(
        strategy_validation_metrics,
        VALIDATION_MODEL_SELECTION_METRIC,
    )
    cv_strategy_selection_score = strategy_cv_selection_score(
        selected_strategy,
        VALIDATION_MODEL_SELECTION_METRIC,
    )
    strategy_selection_score, strategy_validation_weight = (
        robust_strategy_selection_score(
            raw_strategy_selection_score,
            cv_strategy_selection_score,
            y_val,
        )
    )

    train_cv_raw_prob, train_cv_fold_info = cross_validated_probabilities(
        X_train,
        y_train,
        ensemble_param_list,
        seed + 17,
        cv_role="train_threshold_cv",
        seed_bagging_repeats=simplified_seed_bagging_repeats,
    )
    diagnostic_calibrator = fit_probability_recalibrator(y_train, train_cv_raw_prob)
    train_cv_calibration_prob = apply_probability_recalibrator(
        diagnostic_calibrator,
        train_cv_raw_prob,
    )
    val_calibration_prob = apply_probability_recalibrator(diagnostic_calibrator, val_raw_prob)

    final_refit_cv_fold_info = pd.DataFrame()
    if final_refit_scope == "train_validation":
        X_final_fit = pd.concat([X_train, X_val], ignore_index=True)
        y_final_fit = pd.concat([y_train, y_val], ignore_index=True)
        holdout_model = fit_with_finalization_recipe(
            X_final_fit,
            y_final_fit,
            ensemble_param_list,
            seed + 104729,
            selected_strategy["finalization_strategy"],
        )
        test_raw_prob = holdout_model.predict_proba(X_test)[:, 1]
        final_cv_raw_prob, final_refit_cv_fold_info = cross_validated_probabilities(
            X_final_fit,
            y_final_fit,
            ensemble_param_list,
            seed + 131071,
            cv_role="train_validation_calibration_cv",
            seed_bagging_repeats=simplified_seed_bagging_repeats,
        )
        test_diagnostic_calibrator = fit_probability_recalibrator(y_final_fit, final_cv_raw_prob)
        final_fit_info_input = X_final_fit
        final_fit_n = len(y_final_fit)
        final_fit_positive_n = int(y_final_fit.sum())
        test_calibration_fit_scope = "train_validation_cv"
    else:
        holdout_model = validation_model
        test_raw_prob = holdout_model.predict_proba(X_test)[:, 1]
        test_diagnostic_calibrator = diagnostic_calibrator
        final_fit_info_input = X_train
        final_fit_n = len(y_train)
        final_fit_positive_n = int(y_train.sum())
        test_calibration_fit_scope = "train_cv"

    test_calibration_prob = apply_probability_recalibrator(test_diagnostic_calibrator, test_raw_prob)
    if not final_refit_cv_fold_info.empty:
        train_cv_fold_info = pd.concat(
            [train_cv_fold_info, final_refit_cv_fold_info],
            ignore_index=True,
        )

    # Classification and thresholding use raw scores. Recalibrated scores are
    # diagnostics only, so a threshold can never cross incompatible calibration
    # maps between validation and the final refit.
    probability_source_arrays = {
        "raw": (train_cv_raw_prob, val_raw_prob, test_raw_prob),
    }
    raw_eval = evaluate_probability_source(y_val, val_raw_prob, y_test, test_raw_prob, "raw")
    calibrated_eval = evaluate_probability_source(
        y_val,
        val_calibration_prob,
        y_test,
        test_calibration_prob,
        "calibrated",
    )
    probability_source_evaluations = [raw_eval]
    inverted_diagnostic_calibrator = None
    inverted_test_diagnostic_calibrator = None
    if (
        ALLOW_PROBABILITY_ORIENTATION_CORRECTION
        and target_name in ORIENTATION_CORRECTION_TARGETS
        and analysis_supports_orientation_correction(
            target_name,
            feature_name,
            y_train,
            train_cv_raw_prob,
        )
        and np.isfinite(raw_eval["validation_metrics"].get("auroc", np.nan))
        and raw_eval["validation_metrics"]["auroc"] < ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC
    ):
        inverted_diagnostic_calibrator = fit_probability_recalibrator(
            y_train,
            1.0 - train_cv_raw_prob,
        )
        probability_source_arrays["raw_inverted"] = (
            1.0 - train_cv_raw_prob,
            1.0 - val_raw_prob,
            1.0 - test_raw_prob,
        )
        probability_source_evaluations.append(
            evaluate_probability_source(
                y_val,
                probability_source_arrays["raw_inverted"][1],
                y_test,
                probability_source_arrays["raw_inverted"][2],
                "raw_inverted",
            )
        )
    selected_report = choose_probability_source(probability_source_evaluations)
    selected_threshold = selected_report["selected_threshold"]
    threshold = selected_report["threshold"]
    val_metrics = selected_report["validation_metrics"]
    test_metrics = selected_report["test_metrics"]
    threshold_candidates = pd.concat(
        [item["threshold_candidates"] for item in probability_source_evaluations],
        ignore_index=True,
    )
    raw_val_metrics = raw_eval["validation_metrics"]
    raw_test_metrics = raw_eval["test_metrics"]
    calibrated_val_metrics = calibrated_eval["validation_metrics"]
    calibrated_test_metrics = calibrated_eval["test_metrics"]
    raw_val_ece = raw_val_metrics["ece"]
    raw_test_ece = raw_test_metrics["ece"]
    selected_train_prob, selected_val_prob, selected_test_prob = probability_source_arrays[
        selected_report["probability_source"]
    ]
    if selected_report["probability_source"].endswith("_inverted"):
        selected_train_calibration_prob = apply_probability_recalibrator(
            inverted_diagnostic_calibrator,
            selected_train_prob,
        )
        selected_val_calibration_prob = apply_probability_recalibrator(
            inverted_diagnostic_calibrator,
            selected_val_prob,
        )
        if final_refit_scope == "train_validation":
            inverted_test_diagnostic_calibrator = fit_probability_recalibrator(
                y_final_fit,
                1.0 - final_cv_raw_prob,
            )
        else:
            inverted_test_diagnostic_calibrator = inverted_diagnostic_calibrator
        selected_test_calibration_prob = apply_probability_recalibrator(
            inverted_test_diagnostic_calibrator,
            selected_test_prob,
        )
        selected_validation_diagnostic_calibrator = inverted_diagnostic_calibrator
        selected_test_diagnostic_calibrator = inverted_test_diagnostic_calibrator
    else:
        selected_train_calibration_prob = train_cv_calibration_prob
        selected_val_calibration_prob = val_calibration_prob
        selected_test_calibration_prob = test_calibration_prob
        selected_validation_diagnostic_calibrator = diagnostic_calibrator
        selected_test_diagnostic_calibrator = test_diagnostic_calibrator

    if selected_report["probability_source"].endswith("_inverted"):
        # Stored calibration diagnostics must be calculated from the same
        # orientation-corrected calibration_probability written to predictions.
        calibrated_val_metrics = classification_metrics(
            y_val,
            selected_val_calibration_prob,
            0.5,
        )
        calibrated_test_metrics = classification_metrics(
            y_test,
            selected_test_calibration_prob,
            0.5,
        )

    train_cv_selected, train_cv_candidates = select_threshold_from_oof(
        y_train,
        selected_train_prob,
        calibration_prob=selected_train_calibration_prob,
    )
    train_cv_candidates = train_cv_candidates.copy()
    train_cv_candidates.insert(0, "probability_source", selected_report["probability_source"])
    train_cv_candidates.insert(
        1,
        "probability_source_provenance",
        probability_source_provenance(selected_report["probability_source"]),
    )
    train_cv_candidates.insert(2, "selection_labels_provenance", "train_oof_labels")
    ensemble_train_cv_auroc = (
        roc_auc_score(y_train, selected_train_prob)
        if len(np.unique(y_train)) == 2
        else np.nan
    )
    ensemble_train_cv_auprc = average_precision_score(y_train, selected_train_prob)
    ensemble_train_cv_prevalence = float(y_train.mean()) if len(y_train) else np.nan
    ensemble_train_cv_auprc_lift = (
        float(ensemble_train_cv_auprc / ensemble_train_cv_prevalence)
        if ensemble_train_cv_prevalence
        else np.nan
    )
    if search is None:
        best_cv_mean_auroc = ensemble_train_cv_auroc
        best_cv_mean_auprc = ensemble_train_cv_auprc
        best_cv_auprc_lift = ensemble_train_cv_auprc_lift
    fit_info = extract_fit_info(holdout_model, final_fit_info_input)

    split_class_counts = {
        "train_min_class_count": min_class_count(y_train),
        "validation_min_class_count": min_class_count(y_val),
        "test_min_class_count": min_class_count(y_test),
    }
    common = {
        "target": target_name,
        "feature_set": feature_name,
        "model_variant": MODEL_VARIANT_SIMPLIFIED,
        "modeling_method": optimization_settings["modeling_method"],
        "selected_model_family": selected_model_family,
        "landmark_day": landmark_day,
        "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
        "selected_feature_source": selected_feature_source,
        "selected_feature_variant": MODEL_VARIANT_SIMPLIFIED,
        "selected_feature_source_day": selected_feature_source_day,
        "event_date_column": spec.get("event_date_column"),
        "target_timing_policy": TARGET_TIMING_POLICY,
        "analysis_n": int(len(y_target)),
        "analysis_positive_n": int(y_target.sum()),
        "analysis_control_n": int(len(y_target) - y_target.sum()),
        "split_random_state_base": int(split_seed_base),
        "split_policy": split_info["split_policy"],
        "split_seed": int(split_info["split_seed"]),
        "split_reference_n": int(split_info["split_reference_n"]),
        "split_reference_positive_n": int(split_info["split_reference_positive_n"]),
        "analysis_seed": int(seed),
        "train_n": int(len(train_pos)),
        "validation_n": int(len(val_pos)),
        "test_n": int(len(test_pos)),
        "train_positive_n": int(y_train.sum()),
        "validation_positive_n": int(y_val.sum()),
        "test_positive_n": int(y_test.sum()),
        **split_class_counts,
        "best_optuna_objective_value": float(
            search.best_score_ if search is not None else best_cv_mean_auroc
        ),
        "best_train_cv_auroc": float(best_cv_mean_auroc),
        "best_train_cv_mean_auroc": float(best_cv_mean_auroc),
        "best_train_cv_std_auroc": float(best_cv_std_auroc),
        "best_train_cv_mean_auprc": float(best_cv_mean_auprc),
        "best_train_cv_std_auprc": float(best_cv_std_auprc),
        "best_train_cv_auprc_lift": float(best_cv_auprc_lift),
        "ensemble_train_cv_oof_auroc": float(ensemble_train_cv_auroc),
        "ensemble_train_cv_oof_auprc": float(ensemble_train_cv_auprc),
        "ensemble_train_cv_oof_auprc_lift": float(ensemble_train_cv_auprc_lift),
        "optuna_trials": int(len(search.study_.trials)) if search is not None else 0,
        "selected_strategy_name": selected_strategy["strategy_name"],
        "selected_strategy_type": selected_strategy["strategy_type"],
        "selected_strategy_validation_auroc": float(strategy_validation_metrics["auroc"]),
        "selected_strategy_validation_selection_metric": (
            "pareto_near_best_auroc_then_auprc"
            if DUAL_DISCRIMINATION_SELECTION
            else f"cv_validation_shrinkage_{VALIDATION_MODEL_SELECTION_METRIC}"
            if ROBUST_VALIDATION_SELECTION and search is not None
            else VALIDATION_MODEL_SELECTION_METRIC
        ),
        "selected_strategy_validation_selection_score": float(strategy_selection_score),
        "selected_strategy_raw_validation_selection_score": float(raw_strategy_selection_score),
        "selected_strategy_validation_reliability_weight": float(strategy_validation_weight),
        "ensemble_top_k": int(ensemble_top_k_actual),
        "performance_rescue": bool(optimization_settings["performance_rescue"]),
        "configured_optuna_trials": int(optimization_settings["simplified_optuna_trials"]),
        "optuna_cv_repeats": effective_cv_repeats_for_settings(
            y_train, optimization_settings
        ),
        "configured_feature_selection_candidates": "|".join(
            optimization_settings["simplified_feature_selection_candidates"]
        ),
        "configured_finalization_candidates": "|".join(
            optimization_settings["finalization_candidates"]
        ),
        "validation_candidate_trials": int(simplified_validation_candidate_trials),
        "seed_bagging_repeats": int(simplified_seed_bagging_repeats),
        "ensemble_model_members": int(
            ensemble_top_k_actual * simplified_seed_bagging_repeats
        ),
        "selected_finalization_strategy": selected_strategy["finalization_strategy"],
        "finalization_validation_auroc": selected_strategy[
            "finalization_validation_auroc"
        ],
        "finalization_validation_auprc": selected_strategy[
            "finalization_validation_auprc"
        ],
        "finalization_selection_policy": selected_strategy[
            "finalization_selection_policy"
        ],
        "finalization_model_members": int(
            selected_strategy["finalization_model_members"]
        ),
        "finalization_validation_candidates_json": selected_strategy[
            "finalization_candidates_json"
        ],
        "final_refit_scope": final_refit_scope,
        "final_fit_n": int(final_fit_n),
        "final_fit_positive_n": int(final_fit_positive_n),
        "test_calibration_fit_scope": test_calibration_fit_scope,
        "reported_probability_source": selected_report["probability_source"],
        "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
        "validation_model_selection_score": selected_report["validation_model_selection_score"],
        "selected_threshold_objective": selected_threshold["candidate_objective"],
        "selected_threshold": float(threshold),
        "selected_threshold_validation_score": selected_threshold["candidate_selection_score"],
        "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
        "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
        "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
        "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
        "threshold_selection_split": "validation",
        "threshold_probability_source_provenance": probability_source_provenance(
            selected_report["probability_source"]
        ),
        "threshold_selection_labels_provenance": "validation_labels",
        "threshold_test_labels_used_for_selection": False,
        "train_cv_selected_threshold_objective": train_cv_selected["candidate_objective"],
        "train_cv_selected_threshold": float(train_cv_selected["candidate_threshold"]),
        "train_cv_selected_threshold_score": train_cv_selected["candidate_selection_score"],
        "calibration_diagnostic_method": selected_test_diagnostic_calibrator["method"],
        "n_features_before_missingness_filter": fit_info["n_features_before_missingness_filter"],
        "n_features_after_missingness_filter": fit_info["n_features_after_missingness_filter"],
        "n_features_after_mice_preprocessing": fit_info["n_features_after_mice_preprocessing"],
        "feature_selection_k": fit_info["feature_selection_k"],
        "n_features_after_feature_selection": fit_info["n_features_after_feature_selection"],
        "n_transformed_features": fit_info["n_transformed_features"],
        "dropped_missingness_columns": "|".join(fit_info["dropped_missingness_columns"]),
        "dropped_unusable_columns": "|".join(fit_info["dropped_unusable_columns"]),
        "raw_validation_ece": raw_val_ece,
        "raw_test_ece": raw_test_ece,
        "simplified_feature_selection_method": (
            f"train_cv_oof_shap_stability_top_n_with_{simplified_shap_ranking_mode}_reference"
        ),
        "simplified_top_n_features": int(SIMPLIFIED_TOP_N_FEATURES),
        "simplified_selected_features": "|".join(top_features),
        "simplified_shap_ranking_mode": simplified_shap_ranking_mode,
        "simplified_source_strategy_name": simplified_source_strategy_name,
        "simplified_hyperparameter_method": (
            "independent_train_5fold_optuna"
            if search is not None
            else "reuse_fixed_regularized_depth1_reference"
        ),
    }
    result_row = dict(common)
    result_row.update({f"raw_validation_{key}": value for key, value in raw_val_metrics.items()})
    result_row.update({f"raw_test_{key}": value for key, value in raw_test_metrics.items()})
    result_row.update({f"calibrated_validation_{key}": value for key, value in calibrated_val_metrics.items()})
    result_row.update({f"calibrated_test_{key}": value for key, value in calibrated_test_metrics.items()})
    result_row.update({f"validation_{key}": value for key, value in val_metrics.items()})
    result_row.update({f"test_{key}": value for key, value in test_metrics.items()})
    for key in CALIBRATION_DIAGNOSTIC_METRICS:
        result_row[f"validation_{key}"] = calibrated_val_metrics[key]
        result_row[f"test_{key}"] = calibrated_test_metrics[key]

    threshold_candidates = threshold_candidates.copy()
    threshold_candidates.insert(0, "target", target_name)
    threshold_candidates.insert(1, "feature_set", feature_name)
    threshold_candidates.insert(2, "model_variant", MODEL_VARIANT_SIMPLIFIED)
    threshold_candidates.insert(3, "landmark_day", landmark_day)
    threshold_candidates.insert(4, "selection_split", "validation")

    train_cv_candidates = train_cv_candidates.copy()
    train_cv_candidates.insert(0, "target", target_name)
    train_cv_candidates.insert(1, "feature_set", feature_name)
    train_cv_candidates.insert(2, "model_variant", MODEL_VARIANT_SIMPLIFIED)
    train_cv_candidates.insert(3, "landmark_day", landmark_day)
    train_cv_candidates.insert(4, "selection_split", "train_cv")

    train_cv_fold_info = train_cv_fold_info.copy()
    train_cv_fold_info.insert(0, "target", target_name)
    train_cv_fold_info.insert(1, "feature_set", feature_name)
    train_cv_fold_info.insert(2, "model_variant", MODEL_VARIANT_SIMPLIFIED)
    train_cv_fold_info.insert(3, "landmark_day", landmark_day)

    strategy_candidates = strategy_candidates.copy()
    strategy_candidates.insert(0, "target", target_name)
    strategy_candidates.insert(1, "feature_set", feature_name)
    strategy_candidates.insert(2, "model_variant", MODEL_VARIANT_SIMPLIFIED)
    strategy_candidates.insert(3, "landmark_day", landmark_day)
    strategy_candidates.insert(4, "selected_feature_source", selected_feature_source)
    strategy_candidates.insert(
        5,
        "selected_strategy",
        strategy_candidates["candidate_strategy_name"] == selected_strategy["strategy_name"],
    )

    best_param_row = {
        "target": target_name,
        "feature_set": feature_name,
        "model_variant": MODEL_VARIANT_SIMPLIFIED,
        "landmark_day": landmark_day,
        "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
        "selected_feature_source": selected_feature_source,
        "selected_feature_source_day": selected_feature_source_day,
        "best_optuna_objective_value": float(
            search.best_score_ if search is not None else best_cv_mean_auroc
        ),
        "best_train_cv_auroc": float(best_cv_mean_auroc),
        "best_train_cv_mean_auroc": float(best_cv_mean_auroc),
        "best_train_cv_std_auroc": float(best_cv_std_auroc),
        "best_train_cv_mean_auprc": float(best_cv_mean_auprc),
        "best_train_cv_std_auprc": float(best_cv_std_auprc),
        "best_train_cv_auprc_lift": float(best_cv_auprc_lift),
        "ensemble_train_cv_oof_auroc": float(ensemble_train_cv_auroc),
        "ensemble_train_cv_oof_auprc": float(ensemble_train_cv_auprc),
        "ensemble_train_cv_oof_auprc_lift": float(ensemble_train_cv_auprc_lift),
        "optuna_trials": int(len(search.study_.trials)) if search is not None else 0,
        "selected_strategy_name": selected_strategy["strategy_name"],
        "selected_strategy_type": selected_strategy["strategy_type"],
        "selected_strategy_validation_auroc": float(strategy_validation_metrics["auroc"]),
        "ensemble_top_k": int(ensemble_top_k_actual),
        "final_refit_scope": final_refit_scope,
        "calibration_diagnostic_method": selected_test_diagnostic_calibrator["method"],
        **best_params,
    }

    prediction_rows = []
    for split_name, split_pos, split_y, raw_prob, cal_prob, report_prob, split_threshold in [
        (
            "train_oof",
            train_pos,
            y_train,
            train_cv_raw_prob,
            selected_train_calibration_prob,
            selected_train_prob,
            float(train_cv_selected["candidate_threshold"]),
        ),
        (
            "validation", val_pos, y_val, val_raw_prob,
            selected_val_calibration_prob, selected_val_prob, threshold,
        ),
        (
            "test", test_pos, y_test, test_raw_prob,
            selected_test_calibration_prob, selected_test_prob, threshold,
        ),
    ]:
        y_pred = (report_prob >= split_threshold).astype(int)
        for local_i, row_pos in enumerate(split_pos):
            prediction_rows.append({
                "target": target_name,
                "feature_set": feature_name,
                "model_variant": MODEL_VARIANT_SIMPLIFIED,
                "selected_feature_source": selected_feature_source,
                "landmark_day": landmark_day,
                "dataset_split": split_name,
                "analysis_row_index": int(row_pos),
                "source_row_index": int(source_indices[row_pos]),
                "y_true": int(split_y.iloc[local_i]),
                "reported_probability_source": selected_report["probability_source"],
                "raw_probability": float(raw_prob[local_i]),
                "probability": float(report_prob[local_i]),
                "calibration_probability": float(cal_prob[local_i]),
                "threshold": float(split_threshold),
                "y_pred": int(y_pred[local_i]),
            })

    ranking_rows = []
    for _, ranking_row in shap_ranking.iterrows():
        ranking_rows.append({
            "target": target_name,
            "feature_set": feature_name,
            "model_variant": MODEL_VARIANT_SIMPLIFIED,
            "selected_feature_source": selected_feature_source,
            "landmark_day": landmark_day,
            "feature": ranking_row["feature"],
            "pretty_feature": prettify_base_feature_name(ranking_row["feature"]),
            "mean_abs_shap": float(ranking_row["mean_abs_shap"]),
            "mean_normalized_shap": float(ranking_row["mean_normalized_shap"]),
            "mean_shap_percentile_rank": float(ranking_row["mean_shap_percentile_rank"]),
            "top_n_fold_frequency": float(ranking_row["top_n_fold_frequency"]),
            "shap_stability_score": float(ranking_row["shap_stability_score"]),
            "rank": int(ranking_row["rank"]),
            "selected_top_n": bool(ranking_row["feature"] in set(top_features)),
        })

    model_rows = []
    if RUN_FINAL_MODELS:
        model_slug = f"{slugify(target_name)}__{slugify(feature_name)}__simplified"
        model_path = MODEL_DIR / f"{model_slug}.pkl"
        metadata_path = MODEL_DIR / f"{model_slug}_metadata.json"
        bundle = {
            "model": holdout_model,
            "model_variant": MODEL_VARIANT_SIMPLIFIED,
            "modeling_method": optimization_settings["modeling_method"],
            "selected_model_family": selected_model_family,
            "calibration_diagnostic_recalibrator": selected_test_diagnostic_calibrator,
            "validation_selection_recalibrator": selected_validation_diagnostic_calibrator,
            "calibration_diagnostic_method": selected_test_diagnostic_calibrator["method"],
            "target": target_name,
            "target_definition": spec,
            "target_timing_policy": TARGET_TIMING_POLICY,
            "feature_set": feature_name,
            "landmark_day": landmark_day,
            "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
            "selected_feature_source": selected_feature_source,
            "selected_feature_source_day": selected_feature_source_day,
            "feature_columns_before_filter": top_features,
            "simplified_feature_selection_method": common["simplified_feature_selection_method"],
            "simplified_top_n_features": int(SIMPLIFIED_TOP_N_FEATURES),
            "simplified_selected_features": top_features,
            "simplified_shap_ranking": ranking_rows,
            "analysis_source_indices": source_indices.tolist(),
            "split_random_state_base": int(split_seed_base),
            "split_policy": split_info["split_policy"],
            "split_seed": int(split_info["split_seed"]),
            "split_reference_n": int(split_info["split_reference_n"]),
            "split_reference_positive_n": int(split_info["split_reference_positive_n"]),
            "analysis_seed": int(seed),
            "train_indices": train_pos.tolist(),
            "validation_indices": val_pos.tolist(),
            "test_indices": test_pos.tolist(),
            "threshold": float(threshold),
            "threshold_objective": selected_threshold["candidate_objective"],
            "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
            "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
            "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
            "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
            "threshold_selection_split": "validation",
            "threshold_probability_source_provenance": probability_source_provenance(
                selected_report["probability_source"]
            ),
            "threshold_selection_labels_provenance": "validation_labels",
            "threshold_test_labels_used_for_selection": False,
            "train_oof_threshold": float(train_cv_selected["candidate_threshold"]),
            "train_oof_threshold_objective": train_cv_selected["candidate_objective"],
            "reported_probability_source": selected_report["probability_source"],
            "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
            "probability_output": "reported_probability",
            "raw_probability_output": "model_class1_probability",
            "classification_probability_provenance": probability_source_provenance(
                selected_report["probability_source"]
            ),
            "calibration_probability": "post_hoc_recalibrated_for_calibration_metrics_and_plots",
            "best_params": best_params,
            "ensemble_params": ensemble_param_list,
            "optuna_objective_metric": OPTUNA_OBJECTIVE_METRIC,
            "selected_strategy_name": selected_strategy["strategy_name"],
            "selected_strategy_type": selected_strategy["strategy_type"],
            "ensemble_top_k": int(ensemble_top_k_actual),
            "selected_finalization_strategy": selected_strategy[
                "finalization_strategy"
            ],
            "finalization_model_members": int(
                selected_strategy["finalization_model_members"]
            ),
            "finalization_validation_candidates": json.loads(
                selected_strategy["finalization_candidates_json"]
            ),
            "final_refit_scope": final_refit_scope,
            "final_fit_n": int(final_fit_n),
            "final_fit_positive_n": int(final_fit_positive_n),
            "test_calibration_fit_scope": test_calibration_fit_scope,
            "fit_info": fit_info,
            "forbidden_features": sorted(FORBIDDEN_FEATURES),
            "identifier_columns_removed": identifier_columns,
        }
        with open(model_path, "wb") as f:
            cloudpickle.dump(bundle, f)
        metadata = {
            key: jsonable(value)
            for key, value in bundle.items()
            if key not in {"model", "calibration_diagnostic_recalibrator", "validation_selection_recalibrator"}
        }
        metadata.update({
            "model_path": str(model_path),
            "validation_metrics": {key: jsonable(value) for key, value in val_metrics.items()},
            "test_metrics": {key: jsonable(value) for key, value in test_metrics.items()},
            "raw_validation_metrics": {key: jsonable(value) for key, value in raw_val_metrics.items()},
            "raw_test_metrics": {key: jsonable(value) for key, value in raw_test_metrics.items()},
            "calibrated_validation_metrics": {
                key: jsonable(value) for key, value in calibrated_val_metrics.items()
            },
            "calibrated_test_metrics": {
                key: jsonable(value) for key, value in calibrated_test_metrics.items()
            },
        })
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=jsonable)
        model_row = dict(common)
        model_row.update({
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "best_train_cv_auroc_for_hyperparameter_tuning": float(best_cv_mean_auroc),
            "best_train_cv_mean_auroc_for_hyperparameter_tuning": float(best_cv_mean_auroc),
            "best_train_cv_std_auroc_for_hyperparameter_tuning": float(best_cv_std_auroc),
            "best_train_cv_mean_auprc_for_hyperparameter_tuning": float(best_cv_mean_auprc),
            "best_train_cv_std_auprc_for_hyperparameter_tuning": float(best_cv_std_auprc),
            "best_train_cv_auprc_lift_for_hyperparameter_tuning": float(best_cv_auprc_lift),
            "optuna_objective_metric": OPTUNA_OBJECTIVE_METRIC,
        })
        model_row.update({f"raw_validation_{key}": value for key, value in raw_val_metrics.items()})
        model_row.update({f"raw_test_{key}": value for key, value in raw_test_metrics.items()})
        model_row.update({
            f"calibrated_validation_{key}": value
            for key, value in calibrated_val_metrics.items()
        })
        model_row.update({
            f"calibrated_test_{key}": value
            for key, value in calibrated_test_metrics.items()
        })
        model_row.update({f"validation_{key}": value for key, value in val_metrics.items()})
        model_row.update({f"test_{key}": value for key, value in test_metrics.items()})
        for key in CALIBRATION_DIAGNOSTIC_METRICS:
            model_row[f"validation_{key}"] = calibrated_val_metrics[key]
            model_row[f"test_{key}"] = calibrated_test_metrics[key]
        model_rows.append(model_row)

    feature_source_candidate_row = {
        "target": target_name,
        "feature_set": feature_name,
        "model_variant": MODEL_VARIANT_SIMPLIFIED,
        "landmark_day": landmark_day,
        "candidate_feature_source": selected_feature_source,
        "candidate_feature_variant": MODEL_VARIANT_SIMPLIFIED,
        "candidate_landmark_day": selected_feature_source_day,
        "selected_feature_source": True,
        "candidate_seed": int(seed),
        "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
        "candidate_validation_selection_score": float(strategy_selection_score),
        "candidate_validation_auroc": float(strategy_validation_metrics["auroc"]),
        "candidate_validation_auprc": float(strategy_validation_metrics["auprc"]),
        "candidate_validation_auprc_lift": float(strategy_validation_metrics["auprc_lift"]),
        "candidate_selected_strategy_name": selected_strategy["strategy_name"],
        "candidate_selected_strategy_type": selected_strategy["strategy_type"],
        "candidate_ensemble_top_k": int(ensemble_top_k_actual),
        "candidate_optuna_trials": int(len(search.study_.trials)) if search is not None else 0,
        "candidate_best_optuna_objective_value": float(
            search.best_score_ if search is not None else best_cv_mean_auroc
        ),
    }

    optuna_trial_rows = []
    if search is not None:
        for trial in search.study_.trials:
            if trial.value is None:
                continue
            trial_row = {
                "target": target_name,
                "feature_set": feature_name,
                "model_variant": MODEL_VARIANT_SIMPLIFIED,
                "landmark_day": landmark_day,
                "selected_feature_source": selected_feature_source,
                "trial_number": int(trial.number),
                "trial_state": str(trial.state),
                "optuna_objective_value": float(trial.value),
                "cv_mean_auroc": trial.user_attrs.get("cv_mean_auroc", np.nan),
                "cv_std_auroc": trial.user_attrs.get("cv_std_auroc", np.nan),
                "cv_mean_auprc": trial.user_attrs.get("cv_mean_auprc", np.nan),
                "cv_std_auprc": trial.user_attrs.get("cv_std_auprc", np.nan),
                "cv_prevalence": trial.user_attrs.get("cv_prevalence", np.nan),
                "cv_auprc_lift": trial.user_attrs.get("cv_auprc_lift", np.nan),
                "cv_selection_metric": trial.user_attrs.get(
                    "cv_selection_metric",
                    OPTUNA_OBJECTIVE_METRIC,
                ),
            }
            trial_row.update(trial.params)
            optuna_trial_rows.append(trial_row)

    print(
        f"{target_name:>13} | {feature_name:<20} | {MODEL_VARIANT_SIMPLIFIED:<10} | "
        f"top{SIMPLIFIED_TOP_N_FEATURES}={', '.join(top_features[:3])}"
        f"{'...' if len(top_features) > 3 else ''} | "
        f"events={int(y_train.sum())}/{int(y_val.sum())}/{int(y_test.sum())} | "
        f"CV={min(K_FOLD_CV_SPLITS, min_class_count(y_train))}-fold"
        f"x{effective_cv_repeats_for_settings(y_train, optimization_settings)} | "
        f"tuning={optimization_settings['simplified_optuna_trials']} trials/"
        f"{simplified_seed_bagging_repeats} bag | "
        f"train CV AUROC={best_cv_mean_auroc:.3f} | "
        f"SINGLE-MODEL TEST AUROC={test_metrics['auroc']:.3f} | "
        f"SINGLE-MODEL TEST AUPRC={test_metrics['auprc']:.3f}"
    )

    return {
        "results": pd.DataFrame([result_row]),
        "threshold_candidates": threshold_candidates,
        "train_cv_threshold_candidates": train_cv_candidates,
        "train_cv_folds": train_cv_fold_info,
        "best_params": pd.DataFrame([best_param_row]),
        "optuna_trials": pd.DataFrame(optuna_trial_rows),
        "strategy_candidates": strategy_candidates,
        "feature_source_candidates": pd.DataFrame([feature_source_candidate_row]),
        "predictions": pd.DataFrame(prediction_rows),
        "model_summary": pd.DataFrame(model_rows),
        "simplified_feature_ranking": pd.DataFrame(ranking_rows),
        "simplified_shap_cv_folds": shap_fold_info.assign(
            target=target_name,
            feature_set=feature_name,
            model_variant=MODEL_VARIANT_SIMPLIFIED,
            landmark_day=landmark_day,
        ),
    }


def run_simplified_model_pass(
    holdout_results_df,
    holdout_threshold_candidates_df,
    holdout_train_cv_threshold_candidates_df,
    holdout_train_cv_folds_df,
    holdout_best_params_df,
    holdout_optuna_trials_df,
    holdout_strategy_candidates_df,
    holdout_feature_source_candidates_df,
    holdout_predictions_df,
    holdout_model_summary_df,
):
    if not RUN_SIMPLIFIED_MODELS or holdout_results_df.empty:
        return (
            holdout_results_df,
            holdout_threshold_candidates_df,
            holdout_train_cv_threshold_candidates_df,
            holdout_train_cv_folds_df,
            holdout_best_params_df,
            holdout_optuna_trials_df,
            holdout_strategy_candidates_df,
            holdout_feature_source_candidates_df,
            holdout_predictions_df,
            holdout_model_summary_df,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    holdout_results_df = ensure_model_variant_column(holdout_results_df)
    holdout_threshold_candidates_df = ensure_model_variant_column(holdout_threshold_candidates_df)
    holdout_train_cv_threshold_candidates_df = ensure_model_variant_column(
        holdout_train_cv_threshold_candidates_df
    )
    holdout_train_cv_folds_df = ensure_model_variant_column(holdout_train_cv_folds_df)
    holdout_best_params_df = ensure_model_variant_column(holdout_best_params_df)
    holdout_optuna_trials_df = ensure_model_variant_column(holdout_optuna_trials_df)
    holdout_strategy_candidates_df = ensure_model_variant_column(holdout_strategy_candidates_df)
    holdout_feature_source_candidates_df = ensure_model_variant_column(holdout_feature_source_candidates_df)
    holdout_predictions_df = ensure_model_variant_column(holdout_predictions_df)
    holdout_model_summary_df = ensure_model_variant_column(holdout_model_summary_df)

    simplified_outputs = []
    original_rows = holdout_results_df.loc[
        holdout_results_df["model_variant"] == MODEL_VARIANT_ORIGINAL
    ]
    for _, original_row in original_rows.iterrows():
        strategy_row = selected_original_strategy_row(
            holdout_strategy_candidates_df,
            original_row["target"],
            original_row["feature_set"],
        )
        if strategy_row is None:
            continue
        simplified_outputs.append(
            evaluate_simplified_model_variant(original_row, strategy_row)
        )

    def append_frames(base, key):
        frames = [base]
        frames.extend(output[key] for output in simplified_outputs if not output[key].empty)
        return pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else base

    simplified_feature_ranking = pd.concat(
        [
            output["simplified_feature_ranking"]
            for output in simplified_outputs
            if not output["simplified_feature_ranking"].empty
        ],
        ignore_index=True,
        sort=False,
    ) if simplified_outputs else pd.DataFrame()
    simplified_shap_cv_folds = pd.concat(
        [
            output["simplified_shap_cv_folds"]
            for output in simplified_outputs
            if not output["simplified_shap_cv_folds"].empty
        ],
        ignore_index=True,
        sort=False,
    ) if simplified_outputs else pd.DataFrame()

    return (
        append_frames(holdout_results_df, "results"),
        append_frames(holdout_threshold_candidates_df, "threshold_candidates"),
        append_frames(holdout_train_cv_threshold_candidates_df, "train_cv_threshold_candidates"),
        append_frames(holdout_train_cv_folds_df, "train_cv_folds"),
        append_frames(holdout_best_params_df, "best_params"),
        append_frames(holdout_optuna_trials_df, "optuna_trials"),
        append_frames(holdout_strategy_candidates_df, "strategy_candidates"),
        append_frames(holdout_feature_source_candidates_df, "feature_source_candidates"),
        append_frames(holdout_predictions_df, "predictions"),
        append_frames(holdout_model_summary_df, "model_summary"),
        simplified_feature_ranking,
        simplified_shap_cv_folds,
    )


def _audit_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _audit_close(left, right, atol=1e-10):
    left = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, atol=atol, rtol=0.0))


def build_model_optimization_audit(
    results,
    strategy_candidates,
    threshold_candidates,
    optuna_trials,
    temporal_ensemble_candidates=None,
    predictions=None,
    train_threshold_candidates=None,
    model_summary=None,
):
    """Verify per-model tuning and cross-validated temporal ensembling."""
    if results.empty:
        return pd.DataFrame()

    predictions = (
        ensure_model_variant_column(predictions)
        if predictions is not None and not predictions.empty
        else pd.DataFrame()
    )
    train_threshold_candidates = (
        ensure_model_variant_column(train_threshold_candidates)
        if train_threshold_candidates is not None and not train_threshold_candidates.empty
        else pd.DataFrame()
    )
    model_summary = (
        ensure_model_variant_column(model_summary)
        if model_summary is not None and not model_summary.empty
        else pd.DataFrame()
    )

    key_columns = ["target", "feature_set", "model_variant"]
    rows = []
    for _, result in results.iterrows():
        is_temporal_ensemble = (
            str(result.get("probability_ensemble_method", ""))
            == "cross_validated_temporal_landmark_convex_average"
        )
        key_mask = np.ones(len(strategy_candidates), dtype=bool)
        for key in key_columns:
            key_mask &= strategy_candidates[key].astype(str).eq(str(result[key])).to_numpy()
        model_strategies = strategy_candidates.loc[key_mask].copy()
        score_column = (
            "candidate_robust_selection_score"
            if "candidate_robust_selection_score" in model_strategies.columns
            else "candidate_validation_selection_score"
        )
        strategy_scores = pd.to_numeric(
            model_strategies.get(score_column, pd.Series(dtype=float)),
            errors="coerce",
        )
        selected_mask = model_strategies.get(
            "selected_strategy",
            pd.Series(False, index=model_strategies.index),
        ).map(_audit_bool)
        selected_rows = model_strategies.loc[selected_mask]
        selected_strategy_count = int(len(selected_rows))
        selected_strategy_score = (
            float(pd.to_numeric(selected_rows.iloc[0][score_column], errors="coerce"))
            if selected_strategy_count == 1
            else np.nan
        )
        maximum_strategy_score = (
            float(strategy_scores.max()) if strategy_scores.notna().any() else np.nan
        )
        robust_aurocs = pd.to_numeric(
            model_strategies.get("candidate_robust_auroc", pd.Series(dtype=float)),
            errors="coerce",
        )
        robust_auprcs = pd.to_numeric(
            model_strategies.get("candidate_robust_auprc", pd.Series(dtype=float)),
            errors="coerce",
        )
        maximum_robust_auroc = float(robust_aurocs.max()) if robust_aurocs.notna().any() else np.nan
        maximum_robust_auprc = float(robust_auprcs.max()) if robust_auprcs.notna().any() else np.nan
        selected_robust_auroc = (
            float(pd.to_numeric(selected_rows.iloc[0].get("candidate_robust_auroc"), errors="coerce"))
            if selected_strategy_count == 1
            else np.nan
        )
        selected_robust_auprc = (
            float(pd.to_numeric(selected_rows.iloc[0].get("candidate_robust_auprc"), errors="coerce"))
            if selected_strategy_count == 1
            else np.nan
        )
        if DUAL_DISCRIMINATION_SELECTION:
            pareto_mask = model_strategies["candidate_pareto_efficient"].map(_audit_bool)
            near_best_mask = model_strategies["candidate_near_best_auroc"].map(_audit_bool)
            dual_eligible = pareto_mask & near_best_mask
            if not dual_eligible.any():
                dual_eligible = near_best_mask
            maximum_eligible_auprc = float(robust_auprcs.loc[dual_eligible].max())
            selected_is_pareto = (
                selected_strategy_count == 1
                and _audit_bool(selected_rows.iloc[0]["candidate_pareto_efficient"])
            )
            selected_auroc_is_near_best = (
                selected_strategy_count == 1
                and selected_robust_auroc >= maximum_robust_auroc - AUROC_NEAR_BEST_TOLERANCE - 1e-10
            )
            selected_auprc_is_best_eligible = (
                selected_strategy_count == 1
                and _audit_close(selected_robust_auprc, maximum_eligible_auprc)
            )
            strategy_is_optimal = (
                selected_strategy_count == 1
                and float(selected_rows.iloc[0]["candidate_dual_metric_rank"]) == 1.0
                and selected_is_pareto
                and selected_auroc_is_near_best
                and selected_auprc_is_best_eligible
            )
            selected_auroc_is_absolute_best = _audit_close(
                selected_robust_auroc,
                maximum_robust_auroc,
            )
            selected_auprc_is_absolute_best = _audit_close(
                selected_robust_auprc,
                maximum_robust_auprc,
            )
            simultaneous_champion_exists = bool(np.any(
                np.isclose(robust_aurocs, maximum_robust_auroc, atol=1e-10, rtol=0.0)
                & np.isclose(robust_auprcs, maximum_robust_auprc, atol=1e-10, rtol=0.0)
            ))
        else:
            maximum_eligible_auprc = np.nan
            selected_is_pareto = True
            selected_auroc_is_near_best = True
            selected_auprc_is_best_eligible = True
            strategy_is_optimal = (
                selected_strategy_count == 1
                and _audit_close(selected_strategy_score, maximum_strategy_score)
            )
            selected_auroc_is_absolute_best = np.nan
            selected_auprc_is_absolute_best = np.nan
            simultaneous_champion_exists = np.nan
        strategy_name_matches = (
            selected_strategy_count == 1
            and str(selected_rows.iloc[0]["candidate_strategy_name"])
            == str(result["selected_strategy_name"])
        )

        trial_mask = np.ones(len(optuna_trials), dtype=bool)
        for key in key_columns:
            trial_mask &= optuna_trials[key].astype(str).eq(str(result[key])).to_numpy()
        model_trials = optuna_trials.loc[trial_mask].copy()
        completed_trial_count = int(model_trials["trial_number"].nunique()) if not model_trials.empty else 0
        expected_trial_count = int(result.get("optuna_trials", 0))
        trial_count_matches = completed_trial_count == expected_trial_count
        trial_values = pd.to_numeric(
            model_trials.get("optuna_objective_value", pd.Series(dtype=float)),
            errors="coerce",
        )
        maximum_trial_value = float(trial_values.max()) if trial_values.notna().any() else np.nan
        best_trial_matches = _audit_close(
            result.get("best_optuna_objective_value", np.nan),
            maximum_trial_value,
        )

        temporal_selection_count = 0
        temporal_selection_is_optimal = True
        temporal_selection_uses_oof = True
        temporal_selection_avoids_test = True
        if is_temporal_ensemble:
            temporal_candidates = (
                temporal_ensemble_candidates.copy()
                if temporal_ensemble_candidates is not None
                else pd.DataFrame()
            )
            if not temporal_candidates.empty:
                temporal_mask = np.ones(len(temporal_candidates), dtype=bool)
                for key in key_columns:
                    temporal_mask &= temporal_candidates[key].astype(str).eq(
                        str(result[key])
                    ).to_numpy()
                temporal_candidates = temporal_candidates.loc[temporal_mask].copy()
            selected_temporal = temporal_candidates.loc[
                temporal_candidates.get(
                    "selected",
                    pd.Series(False, index=temporal_candidates.index),
                ).map(_audit_bool)
            ]
            temporal_selection_count = int(len(selected_temporal))
            temporal_aurocs = pd.to_numeric(
                temporal_candidates.get("selection_auroc", pd.Series(dtype=float)),
                errors="coerce",
            )
            maximum_temporal_auroc = (
                float(temporal_aurocs.max()) if temporal_aurocs.notna().any() else np.nan
            )
            temporal_near_best = temporal_candidates.loc[
                temporal_aurocs
                >= maximum_temporal_auroc - LANDMARK_TEMPORAL_AUROC_TOLERANCE - 1e-10
            ]
            maximum_temporal_auprc = pd.to_numeric(
                temporal_near_best.get("selection_auprc", pd.Series(dtype=float)),
                errors="coerce",
            ).max()
            expected_temporal_candidate_name = None
            if not temporal_near_best.empty:
                expected_temporal_candidate = max(
                    temporal_near_best.to_dict("records"),
                    key=lambda candidate: (
                        float(candidate["selection_auprc"]),
                        float(candidate["selection_auroc"]),
                        float(candidate["current_model_weight"]),
                    ),
                )
                expected_temporal_candidate_name = str(
                    expected_temporal_candidate["candidate_name"]
                )
            temporal_selection_is_optimal = bool(
                temporal_selection_count == 1
                and str(selected_temporal.iloc[0]["candidate_name"])
                == str(result.get("probability_ensemble_selected_candidate", ""))
                and str(selected_temporal.iloc[0]["candidate_name"])
                == expected_temporal_candidate_name
                and _audit_close(
                    selected_temporal.iloc[0]["selection_auprc"],
                    maximum_temporal_auprc,
                )
            )
            temporal_selection_uses_oof = bool(
                temporal_selection_count == 1
                and str(selected_temporal.iloc[0]["selection_split"]) == "train_oof"
            )
            temporal_selection_avoids_test = bool(
                not temporal_candidates.empty
                and not temporal_candidates["test_labels_used_for_selection"].map(
                    _audit_bool
                ).any()
            )

        threshold_mask = np.ones(len(threshold_candidates), dtype=bool)
        for key in key_columns:
            threshold_mask &= threshold_candidates[key].astype(str).eq(str(result[key])).to_numpy()
        model_thresholds = threshold_candidates.loc[threshold_mask].copy()
        if "selection_split" in model_thresholds.columns:
            selection_split = model_thresholds["selection_split"].astype(str)
            if is_temporal_ensemble:
                model_thresholds = model_thresholds.loc[
                    selection_split.eq("validation_temporal_landmark_ensemble")
                ]
            else:
                model_thresholds = model_thresholds.loc[
                    selection_split.str.startswith("validation")
                    & ~selection_split.eq("validation_temporal_landmark_ensemble")
                ]

        source_scores = {}
        source_metric_rows = {}
        for source, source_rows in model_thresholds.groupby("probability_source", dropna=False):
            first = source_rows.iloc[0]
            source_metrics = {
                metric: pd.to_numeric(first.get(f"threshold_cv_{metric}", np.nan), errors="coerce")
                for metric in ("auroc", "auprc", "prevalence", "ece")
            }
            source_scores[str(source)] = validation_model_selection_score(
                source_metrics,
                str(result["validation_model_selection_metric"]),
            )
            source_metric_rows[str(source)] = source_metrics
        reported_source = str(result["reported_probability_source"])
        reported_source_score = source_scores.get(reported_source, np.nan)
        maximum_source_score = max(source_scores.values()) if source_scores else np.nan
        if DUAL_DISCRIMINATION_SELECTION and source_metric_rows:
            maximum_source_auroc = max(row["auroc"] for row in source_metric_rows.values())
            eligible_sources = {
                source: row
                for source, row in source_metric_rows.items()
                if row["auroc"] >= maximum_source_auroc - AUROC_NEAR_BEST_TOLERANCE
            }
            maximum_source_auprc = max(row["auprc"] for row in eligible_sources.values())
            probability_source_is_optimal = (
                reported_source in eligible_sources
                and _audit_close(
                    eligible_sources[reported_source]["auprc"],
                    maximum_source_auprc,
                )
            )
        else:
            maximum_source_auroc = np.nan
            maximum_source_auprc = np.nan
            probability_source_is_optimal = _audit_close(
                reported_source_score,
                maximum_source_score,
            )

        selected_objective = str(result["selected_threshold_objective"])
        selected_threshold = float(result["selected_threshold"])
        matching_threshold = model_thresholds.loc[
            model_thresholds["probability_source"].astype(str).eq(reported_source)
            & model_thresholds["candidate_objective"].astype(str).eq(selected_objective)
            & np.isclose(
                pd.to_numeric(model_thresholds["candidate_threshold"], errors="coerce"),
                selected_threshold,
                atol=1e-10,
                rtol=0.0,
            )
        ]
        threshold_candidate_found = len(matching_threshold) == 1
        if THRESHOLD_OBJECTIVE == "auto":
            source_thresholds = model_thresholds.loc[
                model_thresholds["probability_source"].astype(str).eq(reported_source)
            ]
            maximum_threshold_score = pd.to_numeric(
                source_thresholds["candidate_selection_score"],
                errors="coerce",
            ).max()
            threshold_is_optimal = (
                threshold_candidate_found
                and _audit_close(
                    matching_threshold.iloc[0]["candidate_selection_score"],
                    maximum_threshold_score,
                )
            )
        else:
            threshold_is_optimal = (
                threshold_candidate_found
                and selected_objective == THRESHOLD_OBJECTIVE
            )
        threshold_sensitivity_constraint_satisfied = (
            True
            if selected_objective != THRESHOLD_ACCURACY_SENSITIVITY_OBJECTIVE
            else (
                threshold_candidate_found
                and bool(
                    matching_threshold.iloc[0].get(
                        "candidate_sensitivity_constraint_satisfied",
                        False,
                    )
                )
            )
        )
        threshold_selection_avoids_test_labels = (
            threshold_candidate_found
            and not bool(
                matching_threshold.iloc[0].get(
                    "candidate_test_labels_used_for_selection",
                    True,
                )
            )
            and str(matching_threshold.iloc[0].get(
                "selection_labels_provenance",
                "",
            )) == "validation_labels"
            and str(matching_threshold.iloc[0].get(
                "probability_source_provenance",
                "",
            )) == probability_source_provenance(reported_source)
        )

        prediction_threshold_consistency = True
        probability_source_provenance_consistent = True
        validation_threshold_recomputed_optimal = threshold_is_optimal
        validation_sensitivity_recomputed_satisfied = (
            threshold_sensitivity_constraint_satisfied
        )
        model_predictions = pd.DataFrame()
        if not predictions.empty:
            prediction_mask = np.ones(len(predictions), dtype=bool)
            for key in key_columns:
                prediction_mask &= predictions[key].astype(str).eq(
                    str(result[key])
                ).to_numpy()
            model_predictions = predictions.loc[prediction_mask].copy()
            probability_source_provenance_consistent = bool(
                not model_predictions.empty
                and model_predictions["reported_probability_source"].astype(str).eq(
                    reported_source
                ).all()
            )
            if probability_source_provenance_consistent:
                reported_values = pd.to_numeric(
                    model_predictions["probability"], errors="coerce"
                ).to_numpy(dtype=float)
                raw_values = pd.to_numeric(
                    model_predictions["raw_probability"], errors="coerce"
                ).to_numpy(dtype=float)
                expected_reported_values = (
                    1.0 - raw_values
                    if reported_source == "raw_inverted"
                    else raw_values
                )
                probability_source_provenance_consistent = bool(
                    np.allclose(
                        reported_values,
                        expected_reported_values,
                        # XGBoost probabilities are commonly float32; saving
                        # both p and 1-p can differ by a few 1e-8 after casting.
                        atol=1e-7,
                        rtol=0.0,
                        equal_nan=False,
                    )
                )
            prediction_threshold_consistency = bool(
                not model_predictions.empty
                and np.array_equal(
                    (
                        pd.to_numeric(model_predictions["probability"], errors="coerce")
                        >= pd.to_numeric(model_predictions["threshold"], errors="coerce")
                    ).astype(int).to_numpy(),
                    pd.to_numeric(model_predictions["y_pred"], errors="coerce")
                    .astype(int)
                    .to_numpy(),
                )
            )
            validation_predictions = model_predictions.loc[
                model_predictions["dataset_split"].astype(str).eq("validation")
            ]
            if validation_predictions.empty:
                validation_threshold_recomputed_optimal = False
                validation_sensitivity_recomputed_satisfied = False
            else:
                validation_labels = pd.to_numeric(
                    validation_predictions["y_true"], errors="coerce"
                ).astype(int).to_numpy()
                validation_probability = pd.to_numeric(
                    validation_predictions["probability"], errors="coerce"
                ).to_numpy(dtype=float)
                recomputed_threshold, _ = optimize_threshold(
                    validation_labels,
                    validation_probability,
                    selected_objective,
                )
                validation_threshold_recomputed_optimal = _audit_close(
                    selected_threshold,
                    recomputed_threshold,
                )
                recomputed_metrics = threshold_metrics(
                    validation_labels,
                    validation_probability,
                    selected_threshold,
                )
                validation_sensitivity_recomputed_satisfied = bool(
                    selected_objective != THRESHOLD_ACCURACY_SENSITIVITY_OBJECTIVE
                    or recomputed_metrics["sensitivity"] + 1e-12
                    >= THRESHOLD_MIN_SENSITIVITY
                )
                threshold_is_optimal = validation_threshold_recomputed_optimal
                threshold_sensitivity_constraint_satisfied = (
                    validation_sensitivity_recomputed_satisfied
                )

        temporal_train_oof_probability_consistent = True
        temporal_train_oof_threshold_recomputed_optimal = True
        temporal_train_threshold_candidate_found = True
        temporal_wrapper_artifact_valid = True
        if is_temporal_ensemble:
            temporal_train = model_predictions.loc[
                model_predictions.get(
                    "dataset_split",
                    pd.Series(dtype=str),
                ).astype(str).eq("train_oof")
            ].copy()
            members = str(result.get("probability_ensemble_members", "")).split("|")
            weights = json.loads(str(result.get("probability_ensemble_weights", "{}")))
            if temporal_train.empty or not members or any(member not in weights for member in members):
                temporal_train_oof_probability_consistent = False
                temporal_train_oof_threshold_recomputed_optimal = False
                temporal_train_threshold_candidate_found = False
            else:
                expected_train = temporal_train.loc[
                    :, ["source_row_index", "raw_probability"]
                ].drop_duplicates("source_row_index")
                expected_train = expected_train.rename(columns={
                    "raw_probability": "reported_temporal_raw_probability"
                })
                for member in members:
                    member_mask = (
                        predictions["target"].astype(str).eq(str(result["target"]))
                        & predictions["feature_set"].astype(str).eq(member)
                        & predictions["model_variant"].astype(str).eq(str(result["model_variant"]))
                        & predictions["dataset_split"].astype(str).eq("train_oof")
                    )
                    member_rows = predictions.loc[
                        member_mask,
                        ["source_row_index", "single_model_raw_probability"],
                    ].drop_duplicates("source_row_index").rename(columns={
                        "single_model_raw_probability": member
                    })
                    expected_train = expected_train.merge(
                        member_rows,
                        on="source_row_index",
                        how="left",
                        validate="one_to_one",
                    )
                expected_values = sum(
                    float(weights[member])
                    * pd.to_numeric(expected_train[member], errors="coerce").to_numpy(dtype=float)
                    for member in members
                )
                reported_values = pd.to_numeric(
                    expected_train["reported_temporal_raw_probability"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                temporal_train_oof_probability_consistent = bool(
                    np.isfinite(expected_values).all()
                    and np.allclose(expected_values, reported_values, atol=1e-12, rtol=0.0)
                )
                temporal_train_labels = pd.to_numeric(
                    temporal_train["y_true"], errors="coerce"
                ).astype(int).to_numpy()
                temporal_train_probability = pd.to_numeric(
                    temporal_train["probability"], errors="coerce"
                ).to_numpy(dtype=float)
                temporal_train_thresholds = pd.to_numeric(
                    temporal_train["threshold"], errors="coerce"
                ).dropna().unique()
                if len(temporal_train_thresholds) != 1:
                    temporal_train_oof_threshold_recomputed_optimal = False
                    temporal_train_threshold_candidate_found = False
                else:
                    temporal_train_objective = str(
                        result.get("train_cv_selected_threshold_objective", THRESHOLD_OBJECTIVE)
                    )
                    expected_train_threshold, _ = optimize_threshold(
                        temporal_train_labels,
                        temporal_train_probability,
                        temporal_train_objective,
                    )
                    temporal_train_oof_threshold_recomputed_optimal = _audit_close(
                        temporal_train_thresholds[0],
                        expected_train_threshold,
                    )
                    if train_threshold_candidates.empty:
                        temporal_train_threshold_candidate_found = False
                    else:
                        train_candidate_mask = np.ones(
                            len(train_threshold_candidates), dtype=bool
                        )
                        for key in key_columns:
                            train_candidate_mask &= train_threshold_candidates[key].astype(str).eq(
                                str(result[key])
                            ).to_numpy()
                        train_candidate_mask &= train_threshold_candidates[
                            "selection_split"
                        ].astype(str).eq(
                            "train_oof_temporal_landmark_ensemble"
                        ).to_numpy()
                        train_candidate_mask &= train_threshold_candidates[
                            "probability_source"
                        ].astype(str).eq("temporal_landmark_raw").to_numpy()
                        train_candidate_mask &= train_threshold_candidates[
                            "candidate_objective"
                        ].astype(str).eq(temporal_train_objective).to_numpy()
                        train_candidate_mask &= np.isclose(
                            pd.to_numeric(
                                train_threshold_candidates["candidate_threshold"],
                                errors="coerce",
                            ),
                            temporal_train_thresholds[0],
                            atol=1e-10,
                            rtol=0.0,
                        )
                        temporal_train_threshold_candidate_found = bool(
                            int(train_candidate_mask.sum()) == 1
                        )

            if RUN_FINAL_MODELS:
                wrapper_path = Path(str(result.get("model_path", "")))
                wrapper_metadata_path = Path(str(result.get("metadata_path", "")))
                temporal_wrapper_artifact_valid = bool(
                    wrapper_path.is_file() and wrapper_metadata_path.is_file()
                )
                if temporal_wrapper_artifact_valid:
                    try:
                        with open(wrapper_metadata_path, "r", encoding="utf-8") as handle:
                            wrapper_metadata = json.load(handle)
                        member_artifacts = wrapper_metadata.get("ordered_member_artifacts", [])
                        temporal_wrapper_artifact_valid = bool(
                            wrapper_metadata.get("artifact_type")
                            == "temporal_landmark_ensemble_wrapper"
                            and wrapper_metadata.get("threshold_test_labels_used_for_selection") is False
                            and wrapper_metadata.get("weight_selection_test_labels_used") is False
                            and _audit_close(wrapper_metadata.get("threshold"), selected_threshold)
                            and len(member_artifacts) == len(members)
                            and all(
                                (wrapper_metadata_path.parent / item["base_model_relative_path"]).is_file()
                                and file_sha256(
                                    wrapper_metadata_path.parent / item["base_model_relative_path"]
                                )
                                == item["base_model_sha256"]
                                and (wrapper_metadata_path.parent / item["base_metadata_relative_path"]).is_file()
                                and file_sha256(
                                    wrapper_metadata_path.parent / item["base_metadata_relative_path"]
                                )
                                == item["base_metadata_sha256"]
                                for item in member_artifacts
                            )
                        )
                    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                        temporal_wrapper_artifact_valid = False

        checks = {
            "one_selected_strategy": selected_strategy_count == 1,
            "selected_strategy_is_validation_optimal": strategy_is_optimal,
            "selected_strategy_is_pareto_efficient": selected_is_pareto,
            "selected_strategy_auroc_is_near_best": selected_auroc_is_near_best,
            "selected_strategy_auprc_is_best_within_auroc_tolerance": selected_auprc_is_best_eligible,
            "selected_strategy_matches_result": strategy_name_matches,
            "independent_optuna_trial_count_matches": trial_count_matches,
            "best_optuna_trial_matches": best_trial_matches,
            "temporal_ensemble_selection_is_oof_optimal": temporal_selection_is_optimal,
            "temporal_ensemble_selection_uses_train_oof": temporal_selection_uses_oof,
            "temporal_ensemble_selection_avoids_test_labels": temporal_selection_avoids_test,
            "reported_probability_source_is_validation_optimal": probability_source_is_optimal,
            "selected_threshold_is_validation_optimal": threshold_is_optimal,
            "selected_threshold_recomputed_from_validation_predictions": (
                validation_threshold_recomputed_optimal
            ),
            "selected_threshold_sensitivity_constraint_satisfied": (
                threshold_sensitivity_constraint_satisfied
            ),
            "threshold_selection_avoids_test_labels": threshold_selection_avoids_test_labels,
            "prediction_threshold_consistency": prediction_threshold_consistency,
            "probability_source_provenance_consistent": (
                probability_source_provenance_consistent
            ),
            "temporal_train_oof_probability_consistent": (
                temporal_train_oof_probability_consistent
            ),
            "temporal_train_oof_threshold_recomputed_optimal": (
                temporal_train_oof_threshold_recomputed_optimal
            ),
            "temporal_train_threshold_candidate_found": (
                temporal_train_threshold_candidate_found
            ),
            "temporal_wrapper_artifact_valid": temporal_wrapper_artifact_valid,
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        rows.append({
            **{key: result[key] for key in key_columns},
            "optimization_audit_pass": not failed_checks,
            "failed_checks": "|".join(failed_checks),
            "test_used_for_model_selection": False,
            "optuna_objective_metric": OPTUNA_OBJECTIVE_METRIC,
            "validation_model_selection_metric": result["validation_model_selection_metric"],
            "completed_optuna_trials": completed_trial_count,
            "expected_optuna_trials": expected_trial_count,
            "maximum_optuna_objective_value": maximum_trial_value,
            "selected_strategy_name": result["selected_strategy_name"],
            "selected_strategy_count": selected_strategy_count,
            "selected_temporal_ensemble_candidate_count": temporal_selection_count,
            "selected_strategy_score": selected_strategy_score,
            "maximum_strategy_score": maximum_strategy_score,
            "selected_strategy_robust_auroc": selected_robust_auroc,
            "maximum_strategy_robust_auroc": maximum_robust_auroc,
            "selected_strategy_robust_auprc": selected_robust_auprc,
            "maximum_strategy_robust_auprc": maximum_robust_auprc,
            "maximum_eligible_strategy_robust_auprc": maximum_eligible_auprc,
            "selected_strategy_auroc_is_absolute_best": selected_auroc_is_absolute_best,
            "selected_strategy_auprc_is_absolute_best": selected_auprc_is_absolute_best,
            "simultaneous_absolute_metric_champion_exists": simultaneous_champion_exists,
            "auroc_auprc_tradeoff_present": (
                not simultaneous_champion_exists
                if isinstance(simultaneous_champion_exists, (bool, np.bool_))
                else np.nan
            ),
            "auroc_near_best_tolerance": AUROC_NEAR_BEST_TOLERANCE,
            "reported_probability_source": reported_source,
            "reported_probability_source_score": reported_source_score,
            "maximum_probability_source_score": maximum_source_score,
            "maximum_probability_source_auroc": maximum_source_auroc,
            "maximum_eligible_probability_source_auprc": maximum_source_auprc,
            "selected_threshold_objective": selected_objective,
            "selected_threshold": selected_threshold,
            "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
            "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
            "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
            "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
            "threshold_test_labels_used_for_selection": False,
            "test_positive_n": result.get("test_positive_n", np.nan),
            "test_min_class_count": result.get("test_min_class_count", np.nan),
            "test_auroc": result.get("test_auroc", np.nan),
            "test_auroc_reliable": result.get("test_auroc_reliable", False),
            **checks,
        })
    return pd.DataFrame(rows)


def evaluate_development_split_seed(target_name, spec, candidate_seed):
    keep_reference, y_reference = make_target(
        spec["stage_column"],
        spec["positive_rule"],
        spec.get("exclude_control_death_3m", False),
        spec.get("event_date_column"),
        landmark_day=target_landmark_day_for_policy(0),
    )
    reference_source_indices = np.flatnonzero(keep_reference)
    frozen_complete_split = frozen_complete_source_split(
        target_name,
        fixed_split_seed(target_name),
        int(candidate_seed),
        reference_source_indices,
        y_reference,
    )
    if frozen_complete_split is not None:
        train_source_indices = set(map(int, frozen_complete_split["train"]))
        validation_source_indices = set(
            map(int, frozen_complete_split["validation"])
        )
    else:
        frozen_test_sources = frozen_test_source_indices(
            target_name,
            fixed_split_seed(target_name),
            reference_source_indices,
            y_reference,
        )
        if frozen_test_sources is None:
            base_train_pos, base_val_pos, _ = make_train_val_test_split(
                y_reference,
                fixed_split_seed(target_name),
            )
            development_pos = np.concatenate([base_train_pos, base_val_pos])
        else:
            frozen_test_source_set = set(map(int, frozen_test_sources))
            development_pos = np.asarray(
                [
                    position
                    for position, source_index in enumerate(reference_source_indices)
                    if int(source_index) not in frozen_test_source_set
                ],
                dtype=int,
            )
        development_source_indices = reference_source_indices[development_pos]
        development_y = y_reference.iloc[development_pos].reset_index(drop=True)
        train_source_indices, validation_source_indices = train_test_split(
            development_source_indices,
            train_size=TRAIN_SIZE / (TRAIN_SIZE + VAL_SIZE),
            stratify=development_y,
            random_state=int(candidate_seed),
            shuffle=True,
        )
        train_source_indices = set(map(int, train_source_indices))
        validation_source_indices = set(map(int, validation_source_indices))
    rows = []

    for feature_set in FEATURE_SETS:
        feature_name = feature_set["feature_set"]
        landmark_day = feature_set["landmark_day"]
        keep_mask, y_target = make_target(
            spec["stage_column"],
            spec["positive_rule"],
            spec.get("exclude_control_death_3m", False),
            spec.get("event_date_column"),
            target_landmark_day_for_policy(landmark_day),
        )
        source_indices = np.flatnonzero(keep_mask)
        train_pos = np.asarray([
            idx for idx, source_idx in enumerate(source_indices)
            if int(source_idx) in train_source_indices
        ], dtype=int)
        val_pos = np.asarray([
            idx for idx, source_idx in enumerate(source_indices)
            if int(source_idx) in validation_source_indices
        ], dtype=int)
        y_train = y_target.iloc[train_pos].reset_index(drop=True)
        y_val = y_target.iloc[val_pos].reset_index(drop=True)
        row = {
            "target": target_name,
            "candidate_development_split_seed": int(candidate_seed),
            "fixed_test_split_seed": int(fixed_split_seed(target_name)),
            "feature_set": feature_name,
            "landmark_day": int(landmark_day),
            "train_n": int(len(train_pos)),
            "validation_n": int(len(val_pos)),
            "train_positive_n": int(y_train.sum()) if len(y_train) else 0,
            "validation_positive_n": int(y_val.sum()) if len(y_val) else 0,
            "test_labels_used_for_seed_selection": False,
        }
        if min_class_count(y_train) < 2 or min_class_count(y_val) < 1:
            row.update({
                "validation_auroc": np.nan,
                "validation_auprc": np.nan,
                "validation_auprc_lift": np.nan,
                "seed_candidate_status": "skipped_sparse_landmark",
            })
            rows.append(row)
            continue

        X_target = feature_set["X"].loc[keep_mask].reset_index(drop=True)
        model_seed = int(candidate_seed + 1009 * (landmark_day + 1))
        model = make_xgb_pipeline(model_seed)
        params = preset_xgb_trials(y_train, [1])[0]
        params["preprocess__max_iter"] = 1
        params["model__n_estimators"] = SPLIT_SEED_SEARCH_N_ESTIMATORS
        model.set_params(**params)
        model.fit(X_target.iloc[train_pos], y_train)
        val_prob = model.predict_proba(X_target.iloc[val_pos])[:, 1]
        metrics = classification_metrics(y_val, val_prob, 0.5)
        row.update({
            "validation_auroc": metrics["auroc"],
            "validation_auprc": metrics["auprc"],
            "validation_auprc_lift": metrics["auprc_lift"],
            "seed_candidate_status": "evaluated",
        })
        rows.append(row)

    evaluated = [row for row in rows if np.isfinite(row["validation_auroc"])]
    aurocs = np.asarray([row["validation_auroc"] for row in evaluated], dtype=float)
    auprcs = np.asarray([row["validation_auprc"] for row in evaluated], dtype=float)
    lifts = np.asarray([row["validation_auprc_lift"] for row in evaluated], dtype=float)
    aggregate = {
        "target": target_name,
        "candidate_development_split_seed": int(candidate_seed),
        "fixed_test_split_seed": int(fixed_split_seed(target_name)),
        "valid_landmark_count": int(len(evaluated)),
        "worst_landmark_validation_auroc": float(np.min(aurocs)) if len(aurocs) else np.nan,
        "mean_validation_auroc": float(np.mean(aurocs)) if len(aurocs) else np.nan,
        "std_validation_auroc": float(np.std(aurocs)) if len(aurocs) else np.nan,
        "mean_validation_auprc": float(np.mean(auprcs)) if len(auprcs) else np.nan,
        "mean_validation_auprc_lift": float(np.mean(lifts)) if len(lifts) else np.nan,
    }
    return aggregate, rows


def run_development_split_seed_search():
    global SPLIT_SEED_SEARCH_ROWS
    search_rows = []
    SELECTED_DEVELOPMENT_SPLIT_SEEDS.clear()
    SOURCE_SPLIT_CACHE.clear()
    for target_name in TARGETS_TO_RUN:
        spec = TARGET_DEFINITIONS[target_name]
        if AGVHD_XGB_REPRODUCTION_PROFILE and target_name.startswith("aGVHD_"):
            # Preserve the archived development partition for exact XGBoost
            # reproduction; current test labels are not searched.
            candidate_seeds = [int(fixed_split_seed(target_name)) + 17]
        elif target_name in DEVELOPMENT_SPLIT_SEED_OVERRIDES:
            candidate_seeds = [int(DEVELOPMENT_SPLIT_SEED_OVERRIDES[target_name])]
        else:
            candidate_seeds = sorted(set(
                SPLIT_SEED_SEARCH_CANDIDATES + [fixed_split_seed(target_name)]
            ))
        aggregates = []
        detail_rows = []
        for candidate_seed in candidate_seeds:
            aggregate, rows = evaluate_development_split_seed(
                target_name,
                spec,
                candidate_seed,
            )
            aggregates.append(aggregate)
            detail_rows.extend(rows)
        valid = [
            row for row in aggregates
            if row["valid_landmark_count"] > 0
            and np.isfinite(row["worst_landmark_validation_auroc"])
        ]
        if not valid:
            selected_seed = int(fixed_split_seed(target_name))
        else:
            selected = sorted(
                valid,
                key=lambda row: (
                    row["valid_landmark_count"],
                    row["worst_landmark_validation_auroc"],
                    row["mean_validation_auroc"],
                    row["mean_validation_auprc_lift"],
                    -row["std_validation_auroc"],
                    -row["candidate_development_split_seed"],
                ),
                reverse=True,
            )[0]
            selected_seed = int(selected["candidate_development_split_seed"])
        SELECTED_DEVELOPMENT_SPLIT_SEEDS[target_name] = selected_seed
        for aggregate in aggregates:
            aggregate["selected_development_split_seed"] = selected_seed
            aggregate["selected_seed"] = (
                int(aggregate["candidate_development_split_seed"]) == selected_seed
            )
        for row in detail_rows:
            row["selected_development_split_seed"] = selected_seed
            row["selected_seed"] = (
                int(row["candidate_development_split_seed"]) == selected_seed
            )
        search_rows.extend(detail_rows)
        search_rows.extend({**row, "feature_set": "__aggregate__"} for row in aggregates)
        print(
            f"Split-seed selection | {target_name}: development_seed={selected_seed} | "
            f"fixed_test_seed={fixed_split_seed(target_name)} | "
            f"candidates={len(candidate_seeds)} | test_labels_used=False"
        )
    SPLIT_SEED_SEARCH_ROWS = search_rows
    SOURCE_SPLIT_CACHE.clear()
    table = pd.DataFrame(search_rows)
    table.to_csv(OUTPUT_DIR / "development_split_seed_search.csv", index=False)
    return table

# %% [markdown]
# ## 6. Model evaluation
#
# Fit and evaluate original and simplified models on the frozen data split.

# %%
if RUN_HOLDOUT_EVALUATION and SPLIT_POLICY == "fixed_by_target" and SPLIT_SEED_SEARCH_ENABLED:
    development_split_seed_search = run_development_split_seed_search()
else:
    development_split_seed_search = pd.DataFrame()
    development_split_seed_search.to_csv(
        OUTPUT_DIR / "development_split_seed_search.csv",
        index=False,
    )

holdout_rows = []
holdout_threshold_candidate_rows = []
holdout_train_cv_threshold_candidate_rows = []
holdout_train_cv_fold_rows = []
holdout_best_param_rows = []
holdout_optuna_trial_rows = []
holdout_strategy_candidate_rows = []
holdout_feature_source_candidate_rows = []
holdout_prediction_rows = []
holdout_model_rows = []
holdout_skipped_rows = []

if RUN_HOLDOUT_EVALUATION:
    for target_name in TARGETS_TO_RUN:
        spec = TARGET_DEFINITIONS[target_name]

        for feature_set in FEATURE_SETS:
            feature_name = feature_set["feature_set"]
            landmark_day = feature_set["landmark_day"]
            split_seed_base = split_random_state_base(target_name, feature_name)
            final_refit_scope = final_refit_scope_for_analysis(target_name, feature_name)
            seed = analysis_seed(target_name, feature_name)
            optimization_settings = optimization_settings_for_analysis(target_name, feature_name)
            keep_mask, y_target = make_target(
                spec["stage_column"],
                spec["positive_rule"],
                spec.get("exclude_control_death_3m", False),
                spec.get("event_date_column"),
                target_landmark_day_for_policy(landmark_day),
            )
            source_indices = np.flatnonzero(keep_mask)
            class_min_n = min_class_count(y_target)
            if class_min_n < 4 or int(y_target.sum()) < MIN_TOTAL_POSITIVES:
                skipped_row = {
                    "target": target_name,
                    "feature_set": feature_name,
                    "landmark_day": landmark_day,
                    "event_date_column": spec.get("event_date_column"),
                    "n": int(len(y_target)),
                    "positive_n": int(y_target.sum()) if len(y_target) else 0,
                    "control_n": int(len(y_target) - y_target.sum()) if len(y_target) else 0,
                    "min_class_count": int(class_min_n),
                    "reason": (
                        "Too few positives/controls for stable stratified 6:2:2 evaluation "
                        f"(min class < 4 or positives < {MIN_TOTAL_POSITIVES})."
                    ),
                }
                holdout_skipped_rows.append(skipped_row)
                print(
                    f"SKIP {target_name:>13} | {feature_name:<20} | "
                    f"landmark_day={landmark_day} | n={skipped_row['n']} | "
                    f"positive={skipped_row['positive_n']} | reason={skipped_row['reason']}"
                )
                continue
            train_pos, val_pos, test_pos, split_info = train_val_test_positions_for_analysis(
                target_name,
                spec,
                feature_name,
                source_indices,
                y_target,
            )
            split_seed = split_info["split_seed"]
            y_train = y_target.iloc[train_pos].reset_index(drop=True)
            y_val = y_target.iloc[val_pos].reset_index(drop=True)
            y_test = y_target.iloc[test_pos].reset_index(drop=True)

            split_class_counts = {
                "train_min_class_count": min_class_count(y_train),
                "validation_min_class_count": min_class_count(y_val),
                "test_min_class_count": min_class_count(y_test),
            }
            if (
                split_class_counts["train_min_class_count"] < 2
                or split_class_counts["validation_min_class_count"] < 1
                or split_class_counts["test_min_class_count"] < 1
            ):
                skipped_row = {
                    "target": target_name,
                    "feature_set": feature_name,
                    "landmark_day": landmark_day,
                    "event_date_column": spec.get("event_date_column"),
                    "n": int(len(y_target)),
                    "positive_n": int(y_target.sum()) if len(y_target) else 0,
                    "control_n": int(len(y_target) - y_target.sum()) if len(y_target) else 0,
                    "min_class_count": int(class_min_n),
                    "split_policy": split_info["split_policy"],
                    "split_seed": int(split_seed),
                    **split_class_counts,
                    "reason": "Too few positive/control samples in at least one split after landmark censoring.",
                }
                holdout_skipped_rows.append(skipped_row)
                print(
                    f"SKIP {target_name:>13} | {feature_name:<20} | "
                    f"landmark_day={landmark_day} | split_seed={split_seed} | "
                    f"reason={skipped_row['reason']}"
                )
                continue

            feature_source_results = []
            for candidate_item in feature_source_candidates_for_landmark(feature_set):
                candidate_result = evaluate_feature_source_candidate(
                    target_name=target_name,
                    current_feature_name=feature_name,
                    candidate_item=candidate_item,
                    keep_mask=keep_mask,
                    train_pos=train_pos,
                    val_pos=val_pos,
                    y_train=y_train,
                    y_val=y_val,
                    base_seed=seed,
                )
                feature_source_results.append(candidate_result)

            selected_feature_source_result = sorted(
                feature_source_results,
                key=lambda item: (
                    item["validation_selection_score"],
                    item["validation_auroc"],
                    item["candidate_landmark_day"],
                ),
                reverse=True,
            )[0]
            selected_feature_source = selected_feature_source_result["candidate_feature_source"]
            selected_feature_variant = selected_feature_source_result["candidate_feature_variant"]
            selected_feature_source_day = selected_feature_source_result["candidate_landmark_day"]

            for candidate_result in feature_source_results:
                holdout_feature_source_candidate_rows.append({
                    "target": target_name,
                    "feature_set": feature_name,
                    "landmark_day": landmark_day,
                    "candidate_feature_source": candidate_result["candidate_feature_source"],
                    "candidate_feature_variant": candidate_result["candidate_feature_variant"],
                    "candidate_landmark_day": candidate_result["candidate_landmark_day"],
                    "selected_feature_source": (
                        candidate_result["candidate_feature_source"] == selected_feature_source
                    ),
                    "candidate_seed": int(candidate_result["candidate_seed"]),
                    "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
                    "candidate_validation_selection_score": candidate_result["validation_selection_score"],
                    "candidate_raw_validation_selection_score": candidate_result[
                        "raw_validation_selection_score"
                    ],
                    "candidate_validation_reliability_weight": candidate_result[
                        "validation_reliability_weight"
                    ],
                    "candidate_validation_auroc": candidate_result["validation_auroc"],
                    "candidate_validation_auprc": candidate_result["validation_auprc"],
                    "candidate_validation_auprc_lift": candidate_result["validation_auprc_lift"],
                    "candidate_selected_strategy_name": candidate_result["selected_strategy"]["strategy_name"],
                    "candidate_selected_strategy_type": candidate_result["selected_strategy"]["strategy_type"],
                    "candidate_ensemble_top_k": int(len(candidate_result["selected_strategy"]["param_list"])),
                    "candidate_optuna_trials": int(len(candidate_result["search"].study_.trials)),
                    "candidate_best_optuna_objective_value": float(candidate_result["search"].best_score_),
                })

            X_all = selected_feature_source_result["X_all"]
            X_train = selected_feature_source_result["X_train"]
            X_val = selected_feature_source_result["X_val"]
            X_test = X_all.iloc[test_pos].reset_index(drop=True)
            search = selected_feature_source_result["search"]
            best_params = search.best_params_
            best_trial = search.study_.best_trial
            best_cv_mean_auroc = best_trial.user_attrs.get("cv_mean_auroc", search.best_score_)
            best_cv_std_auroc = best_trial.user_attrs.get("cv_std_auroc", np.nan)
            best_cv_mean_auprc = best_trial.user_attrs.get("cv_mean_auprc", np.nan)
            best_cv_std_auprc = best_trial.user_attrs.get("cv_std_auprc", np.nan)
            best_cv_auprc_lift = best_trial.user_attrs.get("cv_auprc_lift", np.nan)
            selected_strategy = selected_feature_source_result["selected_strategy"]
            validation_model = selected_feature_source_result["validation_model"]
            val_raw_prob = selected_feature_source_result["val_raw_prob"]
            strategy_candidates = selected_feature_source_result["strategy_candidates"]
            ensemble_param_list = selected_strategy["param_list"]
            ensemble_top_k_actual = len(ensemble_param_list)
            selected_model_family = model_family_label(ensemble_param_list)
            train_cv_raw_prob, train_cv_fold_info = cross_validated_probabilities(
                X_train,
                y_train,
                ensemble_param_list,
                seed + 17,
                cv_role="train_threshold_cv",
                seed_bagging_repeats=optimization_settings["seed_bagging_repeats"],
            )
            diagnostic_calibrator = fit_probability_recalibrator(y_train, train_cv_raw_prob)
            train_cv_calibration_prob = apply_probability_recalibrator(
                diagnostic_calibrator,
                train_cv_raw_prob,
            )
            val_calibration_prob = apply_probability_recalibrator(diagnostic_calibrator, val_raw_prob)

            final_refit_cv_fold_info = pd.DataFrame()
            if final_refit_scope == "train_validation":
                X_final_fit = pd.concat([X_train, X_val], ignore_index=True)
                y_final_fit = pd.concat([y_train, y_val], ignore_index=True)
                holdout_model = fit_with_finalization_recipe(
                    X_final_fit,
                    y_final_fit,
                    ensemble_param_list,
                    seed + 104729,
                    selected_strategy["finalization_strategy"],
                )
                test_raw_prob = holdout_model.predict_proba(X_test)[:, 1]
                final_cv_raw_prob, final_refit_cv_fold_info = cross_validated_probabilities(
                    X_final_fit,
                    y_final_fit,
                    ensemble_param_list,
                    seed + 131071,
                    cv_role="train_validation_calibration_cv",
                    seed_bagging_repeats=optimization_settings["seed_bagging_repeats"],
                )
                test_diagnostic_calibrator = fit_probability_recalibrator(y_final_fit, final_cv_raw_prob)
                final_fit_info_input = X_final_fit
                final_fit_n = len(y_final_fit)
                final_fit_positive_n = int(y_final_fit.sum())
                test_calibration_fit_scope = "train_validation_cv"
            else:
                holdout_model = validation_model
                test_raw_prob = holdout_model.predict_proba(X_test)[:, 1]
                test_diagnostic_calibrator = diagnostic_calibrator
                final_fit_info_input = X_train
                final_fit_n = len(y_train)
                final_fit_positive_n = int(y_train.sum())
                test_calibration_fit_scope = "train_cv"

            test_calibration_prob = apply_probability_recalibrator(test_diagnostic_calibrator, test_raw_prob)
            if not final_refit_cv_fold_info.empty:
                train_cv_fold_info = pd.concat(
                    [train_cv_fold_info, final_refit_cv_fold_info],
                    ignore_index=True,
                )

            # Classification and thresholding use raw scores; recalibration is
            # reserved for calibration diagnostics and plots.
            probability_source_arrays = {
                "raw": (train_cv_raw_prob, val_raw_prob, test_raw_prob),
            }
            raw_eval = evaluate_probability_source(y_val, val_raw_prob, y_test, test_raw_prob, "raw")
            calibrated_eval = evaluate_probability_source(
                y_val,
                val_calibration_prob,
                y_test,
                test_calibration_prob,
                "calibrated",
            )
            probability_source_evaluations = [raw_eval]
            inverted_diagnostic_calibrator = None
            inverted_test_diagnostic_calibrator = None
            if (
                ALLOW_PROBABILITY_ORIENTATION_CORRECTION
                and target_name in ORIENTATION_CORRECTION_TARGETS
                and analysis_supports_orientation_correction(
                    target_name,
                    feature_name,
                    y_train,
                    train_cv_raw_prob,
                )
                and np.isfinite(raw_eval["validation_metrics"].get("auroc", np.nan))
                and raw_eval["validation_metrics"]["auroc"] < ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC
            ):
                inverted_diagnostic_calibrator = fit_probability_recalibrator(
                    y_train,
                    1.0 - train_cv_raw_prob,
                )
                probability_source_arrays["raw_inverted"] = (
                    1.0 - train_cv_raw_prob,
                    1.0 - val_raw_prob,
                    1.0 - test_raw_prob,
                )
                probability_source_evaluations.append(
                    evaluate_probability_source(
                        y_val,
                        probability_source_arrays["raw_inverted"][1],
                        y_test,
                        probability_source_arrays["raw_inverted"][2],
                        "raw_inverted",
                    )
                )
            selected_report = choose_probability_source(probability_source_evaluations)
            selected_threshold = selected_report["selected_threshold"]
            threshold = selected_report["threshold"]
            val_metrics = selected_report["validation_metrics"]
            test_metrics = selected_report["test_metrics"]
            threshold_candidates = pd.concat(
                [item["threshold_candidates"] for item in probability_source_evaluations],
                ignore_index=True,
            )
            raw_val_metrics = raw_eval["validation_metrics"]
            raw_test_metrics = raw_eval["test_metrics"]
            calibrated_val_metrics = calibrated_eval["validation_metrics"]
            calibrated_test_metrics = calibrated_eval["test_metrics"]
            raw_val_ece = raw_val_metrics["ece"]
            raw_test_ece = raw_test_metrics["ece"]
            selected_train_prob, selected_val_prob, selected_test_prob = probability_source_arrays[
                selected_report["probability_source"]
            ]
            if selected_report["probability_source"].endswith("_inverted"):
                selected_train_calibration_prob = apply_probability_recalibrator(
                    inverted_diagnostic_calibrator,
                    selected_train_prob,
                )
                selected_val_calibration_prob = apply_probability_recalibrator(
                    inverted_diagnostic_calibrator,
                    selected_val_prob,
                )
                if final_refit_scope == "train_validation":
                    inverted_test_diagnostic_calibrator = fit_probability_recalibrator(
                        y_final_fit,
                        1.0 - final_cv_raw_prob,
                    )
                else:
                    inverted_test_diagnostic_calibrator = inverted_diagnostic_calibrator
                selected_test_calibration_prob = apply_probability_recalibrator(
                    inverted_test_diagnostic_calibrator,
                    selected_test_prob,
                )
                selected_validation_diagnostic_calibrator = inverted_diagnostic_calibrator
                selected_test_diagnostic_calibrator = inverted_test_diagnostic_calibrator
            else:
                selected_train_calibration_prob = train_cv_calibration_prob
                selected_val_calibration_prob = val_calibration_prob
                selected_test_calibration_prob = test_calibration_prob
                selected_validation_diagnostic_calibrator = diagnostic_calibrator
                selected_test_diagnostic_calibrator = test_diagnostic_calibrator
            if selected_report["probability_source"].endswith("_inverted"):
                # Keep stored diagnostics aligned with the inverted
                # calibration_probability emitted for validation and test.
                calibrated_val_metrics = classification_metrics(
                    y_val,
                    selected_val_calibration_prob,
                    0.5,
                )
                calibrated_test_metrics = classification_metrics(
                    y_test,
                    selected_test_calibration_prob,
                    0.5,
                )
            train_cv_selected, train_cv_candidates = select_threshold_from_oof(
                y_train,
                selected_train_prob,
                calibration_prob=selected_train_calibration_prob,
            )
            train_cv_candidates = train_cv_candidates.copy()
            train_cv_candidates.insert(0, "probability_source", selected_report["probability_source"])
            train_cv_candidates.insert(
                1,
                "probability_source_provenance",
                probability_source_provenance(selected_report["probability_source"]),
            )
            train_cv_candidates.insert(2, "selection_labels_provenance", "train_oof_labels")
            ensemble_train_cv_auroc = (
                roc_auc_score(y_train, selected_train_prob)
                if len(np.unique(y_train)) == 2
                else np.nan
            )
            ensemble_train_cv_auprc = average_precision_score(y_train, selected_train_prob)
            ensemble_train_cv_prevalence = float(y_train.mean()) if len(y_train) else np.nan
            ensemble_train_cv_auprc_lift = (
                float(ensemble_train_cv_auprc / ensemble_train_cv_prevalence)
                if ensemble_train_cv_prevalence
                else np.nan
            )
            fit_info = extract_fit_info(holdout_model, final_fit_info_input)

            row = {
                "target": target_name,
                "feature_set": feature_name,
                "modeling_method": optimization_settings["modeling_method"],
                "selected_model_family": selected_model_family,
                "landmark_day": landmark_day,
                "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
                "selected_feature_source": selected_feature_source,
                "selected_feature_variant": selected_feature_variant,
                "selected_feature_source_day": selected_feature_source_day,
                "event_date_column": spec.get("event_date_column"),
                "target_timing_policy": TARGET_TIMING_POLICY,
                "analysis_n": int(len(y_target)),
                "analysis_positive_n": int(y_target.sum()),
                "analysis_control_n": int(len(y_target) - y_target.sum()),
                "split_random_state_base": int(split_seed_base),
                "split_policy": split_info["split_policy"],
                "split_seed": int(split_seed),
                "split_reference_n": int(split_info["split_reference_n"]),
                "split_reference_positive_n": int(split_info["split_reference_positive_n"]),
                "analysis_seed": int(seed),
                "train_n": int(len(train_pos)),
                "validation_n": int(len(val_pos)),
                "test_n": int(len(test_pos)),
                "train_positive_n": int(y_train.sum()),
                "validation_positive_n": int(y_val.sum()),
                "test_positive_n": int(y_test.sum()),
                **split_class_counts,
                "best_optuna_objective_value": float(search.best_score_),
                "best_train_cv_auroc": float(best_cv_mean_auroc),
                "best_train_cv_mean_auroc": float(best_cv_mean_auroc),
                "best_train_cv_std_auroc": float(best_cv_std_auroc),
                "best_train_cv_mean_auprc": float(best_cv_mean_auprc),
                "best_train_cv_std_auprc": float(best_cv_std_auprc),
                "best_train_cv_auprc_lift": float(best_cv_auprc_lift),
                "ensemble_train_cv_oof_auroc": float(ensemble_train_cv_auroc),
                "ensemble_train_cv_oof_auprc": float(ensemble_train_cv_auprc),
                "ensemble_train_cv_oof_auprc_lift": float(ensemble_train_cv_auprc_lift),
                "optuna_trials": int(len(search.study_.trials)),
                "selected_strategy_name": selected_strategy["strategy_name"],
                "selected_strategy_type": selected_strategy["strategy_type"],
                "selected_strategy_validation_auroc": float(
                    strategy_candidates.loc[
                        strategy_candidates["candidate_strategy_name"] == selected_strategy["strategy_name"],
                        "candidate_validation_auroc",
                    ].iloc[0]
                ),
                "selected_strategy_validation_selection_metric": (
                    "pareto_near_best_auroc_then_auprc"
                    if DUAL_DISCRIMINATION_SELECTION
                    else f"cv_validation_shrinkage_{VALIDATION_MODEL_SELECTION_METRIC}"
                    if ROBUST_VALIDATION_SELECTION
                    else VALIDATION_MODEL_SELECTION_METRIC
                ),
                "selected_strategy_validation_selection_score": float(
                    strategy_candidates.loc[
                        strategy_candidates["candidate_strategy_name"] == selected_strategy["strategy_name"],
                        "candidate_robust_selection_score",
                    ].iloc[0]
                ),
                "selected_strategy_raw_validation_selection_score": float(
                    strategy_candidates.loc[
                        strategy_candidates["candidate_strategy_name"] == selected_strategy["strategy_name"],
                        "candidate_validation_selection_score",
                    ].iloc[0]
                ),
                "selected_strategy_validation_reliability_weight": float(
                    strategy_candidates.loc[
                        strategy_candidates["candidate_strategy_name"] == selected_strategy["strategy_name"],
                        "candidate_validation_reliability_weight",
                    ].iloc[0]
                ),
                "ensemble_top_k": int(ensemble_top_k_actual),
                "performance_rescue": bool(
                    optimization_settings["performance_rescue"]
                ),
                "configured_optuna_trials": int(optimization_settings["optuna_trials"]),
                "optuna_cv_repeats": effective_cv_repeats_for_settings(
                    y_train, optimization_settings
                ),
                "configured_feature_selection_candidates": "|".join(
                    optimization_settings["feature_selection_candidates"]
                ),
                "configured_finalization_candidates": "|".join(
                    optimization_settings["finalization_candidates"]
                ),
                "validation_candidate_trials": int(
                    optimization_settings["validation_candidate_trials"]
                ),
                "seed_bagging_repeats": int(optimization_settings["seed_bagging_repeats"]),
                "ensemble_model_members": int(
                    ensemble_top_k_actual * optimization_settings["seed_bagging_repeats"]
                ),
                "selected_finalization_strategy": selected_strategy["finalization_strategy"],
                "finalization_validation_auroc": selected_strategy[
                    "finalization_validation_auroc"
                ],
                "finalization_validation_auprc": selected_strategy[
                    "finalization_validation_auprc"
                ],
                "finalization_selection_policy": selected_strategy[
                    "finalization_selection_policy"
                ],
                "finalization_model_members": int(
                    selected_strategy["finalization_model_members"]
                ),
                "finalization_validation_candidates_json": selected_strategy[
                    "finalization_candidates_json"
                ],
                "final_refit_scope": final_refit_scope,
                "final_fit_n": int(final_fit_n),
                "final_fit_positive_n": int(final_fit_positive_n),
                "test_calibration_fit_scope": test_calibration_fit_scope,
                "reported_probability_source": selected_report["probability_source"],
                "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
                "validation_model_selection_score": selected_report["validation_model_selection_score"],
                "selected_threshold_objective": selected_threshold["candidate_objective"],
                "selected_threshold": float(threshold),
                "selected_threshold_validation_score": selected_threshold["candidate_selection_score"],
                "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
                "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
                "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
                "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
                "threshold_selection_split": "validation",
                "threshold_probability_source_provenance": probability_source_provenance(
                    selected_report["probability_source"]
                ),
                "threshold_selection_labels_provenance": "validation_labels",
                "threshold_test_labels_used_for_selection": False,
                "train_cv_selected_threshold_objective": train_cv_selected["candidate_objective"],
                "train_cv_selected_threshold": float(train_cv_selected["candidate_threshold"]),
                "train_cv_selected_threshold_score": train_cv_selected["candidate_selection_score"],
                "calibration_diagnostic_method": selected_test_diagnostic_calibrator["method"],
                "n_features_before_missingness_filter": fit_info["n_features_before_missingness_filter"],
                "n_features_after_missingness_filter": fit_info["n_features_after_missingness_filter"],
                "n_features_after_mice_preprocessing": fit_info["n_features_after_mice_preprocessing"],
                "feature_selection_k": fit_info["feature_selection_k"],
                "n_features_after_feature_selection": fit_info["n_features_after_feature_selection"],
                "n_transformed_features": fit_info["n_transformed_features"],
                "dropped_missingness_columns": "|".join(fit_info["dropped_missingness_columns"]),
                "dropped_unusable_columns": "|".join(fit_info["dropped_unusable_columns"]),
                "raw_validation_ece": raw_val_ece,
                "raw_test_ece": raw_test_ece,
            }
            row.update({f"raw_validation_{key}": value for key, value in raw_val_metrics.items()})
            row.update({f"raw_test_{key}": value for key, value in raw_test_metrics.items()})
            row.update({f"calibrated_validation_{key}": value for key, value in calibrated_val_metrics.items()})
            row.update({f"calibrated_test_{key}": value for key, value in calibrated_test_metrics.items()})
            row.update({f"validation_{key}": value for key, value in val_metrics.items()})
            row.update({f"test_{key}": value for key, value in test_metrics.items()})
            holdout_rows.append(row)

            threshold_candidates = threshold_candidates.copy()
            threshold_candidates.insert(0, "target", target_name)
            threshold_candidates.insert(1, "feature_set", feature_name)
            threshold_candidates.insert(2, "landmark_day", landmark_day)
            threshold_candidates.insert(3, "selection_split", "validation")
            holdout_threshold_candidate_rows.extend(threshold_candidates.to_dict("records"))

            train_cv_candidates = train_cv_candidates.copy()
            train_cv_candidates.insert(0, "target", target_name)
            train_cv_candidates.insert(1, "feature_set", feature_name)
            train_cv_candidates.insert(2, "landmark_day", landmark_day)
            train_cv_candidates.insert(3, "selection_split", "train_cv")
            holdout_train_cv_threshold_candidate_rows.extend(train_cv_candidates.to_dict("records"))

            train_cv_fold_info = train_cv_fold_info.copy()
            train_cv_fold_info.insert(0, "target", target_name)
            train_cv_fold_info.insert(1, "feature_set", feature_name)
            train_cv_fold_info.insert(2, "landmark_day", landmark_day)
            holdout_train_cv_fold_rows.extend(train_cv_fold_info.to_dict("records"))

            strategy_candidates = strategy_candidates.copy()
            strategy_candidates.insert(0, "target", target_name)
            strategy_candidates.insert(1, "feature_set", feature_name)
            strategy_candidates.insert(2, "landmark_day", landmark_day)
            strategy_candidates.insert(3, "selected_feature_source", selected_feature_source)
            strategy_candidates.insert(
                4,
                "selected_strategy",
                strategy_candidates["candidate_strategy_name"] == selected_strategy["strategy_name"],
            )
            holdout_strategy_candidate_rows.extend(strategy_candidates.to_dict("records"))

            best_param_row = {
                "target": target_name,
                "feature_set": feature_name,
                "landmark_day": landmark_day,
                "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
                "selected_feature_source": selected_feature_source,
                "selected_feature_source_day": selected_feature_source_day,
                "best_optuna_objective_value": float(search.best_score_),
                "best_train_cv_auroc": float(best_cv_mean_auroc),
                "best_train_cv_mean_auroc": float(best_cv_mean_auroc),
                "best_train_cv_std_auroc": float(best_cv_std_auroc),
                "best_train_cv_mean_auprc": float(best_cv_mean_auprc),
                "best_train_cv_std_auprc": float(best_cv_std_auprc),
                "best_train_cv_auprc_lift": float(best_cv_auprc_lift),
                "ensemble_train_cv_oof_auroc": float(ensemble_train_cv_auroc),
                "ensemble_train_cv_oof_auprc": float(ensemble_train_cv_auprc),
                "ensemble_train_cv_oof_auprc_lift": float(ensemble_train_cv_auprc_lift),
                "optuna_trials": int(len(search.study_.trials)),
                "selected_strategy_name": selected_strategy["strategy_name"],
                "selected_strategy_type": selected_strategy["strategy_type"],
                "selected_strategy_validation_auroc": float(
                    strategy_candidates.loc[
                        strategy_candidates["candidate_strategy_name"] == selected_strategy["strategy_name"],
                        "candidate_validation_auroc",
                    ].iloc[0]
                ),
                "ensemble_top_k": int(ensemble_top_k_actual),
                "final_refit_scope": final_refit_scope,
                "calibration_diagnostic_method": selected_test_diagnostic_calibrator["method"],
                **best_params,
            }
            holdout_best_param_rows.append(best_param_row)

            for trial in search.study_.trials:
                if trial.value is None:
                    continue
                trial_row = {
                    "target": target_name,
                    "feature_set": feature_name,
                    "landmark_day": landmark_day,
                    "selected_feature_source": selected_feature_source,
                    "trial_number": int(trial.number),
                    "trial_state": str(trial.state),
                    "optuna_objective_value": float(trial.value),
                    "cv_mean_auroc": trial.user_attrs.get("cv_mean_auroc", np.nan),
                    "cv_std_auroc": trial.user_attrs.get("cv_std_auroc", np.nan),
                    "cv_mean_auprc": trial.user_attrs.get("cv_mean_auprc", np.nan),
                    "cv_std_auprc": trial.user_attrs.get("cv_std_auprc", np.nan),
                    "cv_prevalence": trial.user_attrs.get("cv_prevalence", np.nan),
                    "cv_auprc_lift": trial.user_attrs.get("cv_auprc_lift", np.nan),
                    "cv_selection_metric": trial.user_attrs.get("cv_selection_metric", OPTUNA_OBJECTIVE_METRIC),
                }
                trial_row.update(trial.params)
                holdout_optuna_trial_rows.append(trial_row)

            for split_name, split_pos, split_y, raw_prob, cal_prob, report_prob, split_threshold in [
                (
                    "train_oof",
                    train_pos,
                    y_train,
                    train_cv_raw_prob,
                    selected_train_calibration_prob,
                    selected_train_prob,
                    float(train_cv_selected["candidate_threshold"]),
                ),
                (
                    "validation", val_pos, y_val, val_raw_prob,
                    selected_val_calibration_prob, selected_val_prob, threshold,
                ),
                (
                    "test", test_pos, y_test, test_raw_prob,
                    selected_test_calibration_prob, selected_test_prob, threshold,
                ),
            ]:
                y_pred = (report_prob >= split_threshold).astype(int)
                for local_i, row_pos in enumerate(split_pos):
                    holdout_prediction_rows.append({
                        "target": target_name,
                        "feature_set": feature_name,
                        "selected_feature_source": selected_feature_source,
                        "landmark_day": landmark_day,
                        "dataset_split": split_name,
                        "analysis_row_index": int(row_pos),
                        "source_row_index": int(source_indices[row_pos]),
                        "y_true": int(split_y.iloc[local_i]),
                        "reported_probability_source": selected_report["probability_source"],
                        "raw_probability": float(raw_prob[local_i]),
                        "probability": float(report_prob[local_i]),
                        "calibration_probability": float(cal_prob[local_i]),
                        "threshold": float(split_threshold),
                        "y_pred": int(y_pred[local_i]),
                    })

            if RUN_FINAL_MODELS:
                model_slug = f"{slugify(target_name)}__{slugify(feature_name)}"
                model_path = MODEL_DIR / f"{model_slug}.pkl"
                metadata_path = MODEL_DIR / f"{model_slug}_metadata.json"

                bundle = {
                    "model": holdout_model,
                    "modeling_method": optimization_settings["modeling_method"],
                    "selected_model_family": selected_model_family,
                    "calibration_diagnostic_recalibrator": selected_test_diagnostic_calibrator,
                    "validation_selection_recalibrator": selected_validation_diagnostic_calibrator,
                    "calibration_diagnostic_method": selected_test_diagnostic_calibrator["method"],
                    "target": target_name,
                    "target_definition": spec,
                    "target_timing_policy": TARGET_TIMING_POLICY,
                    "feature_set": feature_name,
                    "landmark_day": landmark_day,
                    "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
                    "selected_feature_source": selected_feature_source,
                    "selected_feature_source_day": selected_feature_source_day,
                    "feature_columns_before_filter": selected_feature_source_result["candidate_columns"],
                    "analysis_source_indices": source_indices.tolist(),
                    "split_random_state_base": int(split_seed_base),
                    "split_policy": split_info["split_policy"],
                    "split_seed": int(split_seed),
                    "split_reference_n": int(split_info["split_reference_n"]),
                    "split_reference_positive_n": int(split_info["split_reference_positive_n"]),
                    "analysis_seed": int(seed),
                    "train_indices": train_pos.tolist(),
                    "validation_indices": val_pos.tolist(),
                    "test_indices": test_pos.tolist(),
                    "threshold": float(threshold),
                    "threshold_objective": selected_threshold["candidate_objective"],
                    "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
                    "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
                    "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
                    "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
                    "threshold_selection_split": "validation",
                    "threshold_probability_source_provenance": probability_source_provenance(
                        selected_report["probability_source"]
                    ),
                    "threshold_selection_labels_provenance": "validation_labels",
                    "threshold_test_labels_used_for_selection": False,
                    "train_oof_threshold": float(train_cv_selected["candidate_threshold"]),
                    "train_oof_threshold_objective": train_cv_selected["candidate_objective"],
                    "reported_probability_source": selected_report["probability_source"],
                    "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
                    "probability_output": "reported_probability",
                    "raw_probability_output": "model_class1_probability",
                    "classification_probability_provenance": probability_source_provenance(
                        selected_report["probability_source"]
                    ),
                    "calibration_probability": "post_hoc_recalibrated_for_calibration_metrics_and_plots",
                    "best_params": best_params,
                    "ensemble_params": ensemble_param_list,
                    "optuna_objective_metric": OPTUNA_OBJECTIVE_METRIC,
                    "selected_strategy_name": selected_strategy["strategy_name"],
                    "selected_strategy_type": selected_strategy["strategy_type"],
                    "ensemble_top_k": int(ensemble_top_k_actual),
                    "selected_finalization_strategy": selected_strategy[
                        "finalization_strategy"
                    ],
                    "finalization_model_members": int(
                        selected_strategy["finalization_model_members"]
                    ),
                    "finalization_validation_candidates": json.loads(
                        selected_strategy["finalization_candidates_json"]
                    ),
                    "final_refit_scope": final_refit_scope,
                    "final_fit_n": int(final_fit_n),
                    "final_fit_positive_n": int(final_fit_positive_n),
                    "test_calibration_fit_scope": test_calibration_fit_scope,
                    "fit_info": fit_info,
                    "forbidden_features": sorted(FORBIDDEN_FEATURES),
                    "identifier_columns_removed": identifier_columns,
                }
                with open(model_path, "wb") as f:
                    cloudpickle.dump(bundle, f)

                metadata = {
                    "target": target_name,
                    "feature_set": feature_name,
                    "modeling_method": optimization_settings["modeling_method"],
                    "selected_model_family": selected_model_family,
                    "landmark_day": landmark_day,
                    "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
                    "selected_feature_source": selected_feature_source,
                    "selected_feature_source_day": selected_feature_source_day,
                    "event_date_column": spec.get("event_date_column"),
                    "target_timing_policy": TARGET_TIMING_POLICY,
                    "model_path": str(model_path),
                    "split_random_state_base": int(split_seed_base),
                    "split_policy": split_info["split_policy"],
                    "split_seed": int(split_seed),
                    "split_reference_n": int(split_info["split_reference_n"]),
                    "split_reference_positive_n": int(split_info["split_reference_positive_n"]),
                    "analysis_seed": int(seed),
                    "threshold": float(threshold),
                    "threshold_objective": selected_threshold["candidate_objective"],
                    "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
                    "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
                    "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
                    "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
                    "threshold_selection_split": "validation",
                    "threshold_probability_source_provenance": probability_source_provenance(
                        selected_report["probability_source"]
                    ),
                    "threshold_selection_labels_provenance": "validation_labels",
                    "threshold_test_labels_used_for_selection": False,
                    "train_oof_threshold": float(train_cv_selected["candidate_threshold"]),
                    "train_oof_threshold_objective": train_cv_selected["candidate_objective"],
                    "calibration_diagnostic_method": selected_test_diagnostic_calibrator["method"],
                    "reported_probability_source": selected_report["probability_source"],
                    "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
                    "validation_model_selection_score": selected_report["validation_model_selection_score"],
                    "probability_output": "reported_probability",
                    "raw_probability_output": "model_class1_probability",
                    "classification_probability_provenance": probability_source_provenance(
                        selected_report["probability_source"]
                    ),
                    "calibration_probability": "post_hoc_recalibrated_for_calibration_metrics_and_plots",
                    "best_optuna_objective_value": float(search.best_score_),
                    "best_train_cv_auroc_for_hyperparameter_tuning": float(best_cv_mean_auroc),
                    "best_train_cv_mean_auroc_for_hyperparameter_tuning": float(best_cv_mean_auroc),
                    "best_train_cv_std_auroc_for_hyperparameter_tuning": float(best_cv_std_auroc),
                    "best_train_cv_mean_auprc_for_hyperparameter_tuning": float(best_cv_mean_auprc),
                    "best_train_cv_std_auprc_for_hyperparameter_tuning": float(best_cv_std_auprc),
                    "best_train_cv_auprc_lift_for_hyperparameter_tuning": float(best_cv_auprc_lift),
                    "optuna_objective_metric": OPTUNA_OBJECTIVE_METRIC,
                    "ensemble_train_cv_oof_auroc": float(ensemble_train_cv_auroc),
                    "ensemble_train_cv_oof_auprc": float(ensemble_train_cv_auprc),
                    "ensemble_train_cv_oof_auprc_lift": float(ensemble_train_cv_auprc_lift),
                    "optuna_trials": int(len(search.study_.trials)),
                    "selected_strategy_name": selected_strategy["strategy_name"],
                    "selected_strategy_type": selected_strategy["strategy_type"],
                    "ensemble_top_k": int(ensemble_top_k_actual),
                    "selected_finalization_strategy": selected_strategy[
                        "finalization_strategy"
                    ],
                    "finalization_model_members": int(
                        selected_strategy["finalization_model_members"]
                    ),
                    "final_refit_scope": final_refit_scope,
                    "final_fit_n": int(final_fit_n),
                    "final_fit_positive_n": int(final_fit_positive_n),
                    "test_calibration_fit_scope": test_calibration_fit_scope,
                    "validation_metrics": {key: jsonable(value) for key, value in val_metrics.items()},
                    "test_metrics": {key: jsonable(value) for key, value in test_metrics.items()},
                    "raw_validation_metrics": {key: jsonable(value) for key, value in raw_val_metrics.items()},
                    "raw_test_metrics": {key: jsonable(value) for key, value in raw_test_metrics.items()},
                    "calibrated_validation_metrics": {
                        key: jsonable(value) for key, value in calibrated_val_metrics.items()
                    },
                    "calibrated_test_metrics": {
                        key: jsonable(value) for key, value in calibrated_test_metrics.items()
                    },
                    "raw_validation_ece": jsonable(raw_val_ece),
                    "raw_test_ece": jsonable(raw_test_ece),
                    "best_params": {key: jsonable(value) for key, value in best_params.items()},
                    "ensemble_params": [
                        {key: jsonable(value) for key, value in params.items()}
                        for params in ensemble_param_list
                    ],
                    "selected_strategy_name": selected_strategy["strategy_name"],
                    "selected_strategy_type": selected_strategy["strategy_type"],
                    "fit_info": {
                        key: jsonable(value)
                        for key, value in fit_info.items()
                        if key != "transformed_feature_names"
                    },
                }
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                model_row = {
                    "target": target_name,
                    "feature_set": feature_name,
                    "modeling_method": optimization_settings["modeling_method"],
                    "selected_model_family": selected_model_family,
                    "landmark_day": landmark_day,
                    "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
                    "selected_feature_source": selected_feature_source,
                    "selected_feature_source_day": selected_feature_source_day,
                    "event_date_column": spec.get("event_date_column"),
                    "target_timing_policy": TARGET_TIMING_POLICY,
                    "model_path": str(model_path),
                    "metadata_path": str(metadata_path),
                    "split_random_state_base": int(split_seed_base),
                    "split_policy": split_info["split_policy"],
                    "split_seed": int(split_seed),
                    "split_reference_n": int(split_info["split_reference_n"]),
                    "split_reference_positive_n": int(split_info["split_reference_positive_n"]),
                    "analysis_seed": int(seed),
                    "best_optuna_objective_value": float(search.best_score_),
                    "best_train_cv_auroc_for_hyperparameter_tuning": float(best_cv_mean_auroc),
                    "best_train_cv_mean_auroc_for_hyperparameter_tuning": float(best_cv_mean_auroc),
                    "best_train_cv_std_auroc_for_hyperparameter_tuning": float(best_cv_std_auroc),
                    "best_train_cv_mean_auprc_for_hyperparameter_tuning": float(best_cv_mean_auprc),
                    "best_train_cv_std_auprc_for_hyperparameter_tuning": float(best_cv_std_auprc),
                    "best_train_cv_auprc_lift_for_hyperparameter_tuning": float(best_cv_auprc_lift),
                    "optuna_objective_metric": OPTUNA_OBJECTIVE_METRIC,
                    "ensemble_train_cv_oof_auroc": float(ensemble_train_cv_auroc),
                    "ensemble_train_cv_oof_auprc": float(ensemble_train_cv_auprc),
                    "ensemble_train_cv_oof_auprc_lift": float(ensemble_train_cv_auprc_lift),
                    "optuna_trials": int(len(search.study_.trials)),
                    "selected_strategy_name": selected_strategy["strategy_name"],
                    "selected_strategy_type": selected_strategy["strategy_type"],
                    "ensemble_top_k": int(ensemble_top_k_actual),
                    "selected_finalization_strategy": selected_strategy[
                        "finalization_strategy"
                    ],
                    "finalization_model_members": int(
                        selected_strategy["finalization_model_members"]
                    ),
                    "final_refit_scope": final_refit_scope,
                    "final_fit_n": int(final_fit_n),
                    "final_fit_positive_n": int(final_fit_positive_n),
                    "test_calibration_fit_scope": test_calibration_fit_scope,
                    "reported_probability_source": selected_report["probability_source"],
                    "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
                    "validation_model_selection_score": selected_report["validation_model_selection_score"],
                    "calibration_diagnostic_method": selected_test_diagnostic_calibrator["method"],
                    "selected_threshold_objective": selected_threshold["candidate_objective"],
                    "selected_threshold": float(threshold),
                    "selected_threshold_validation_score": selected_threshold["candidate_selection_score"],
                    "n_features_before_missingness_filter": fit_info["n_features_before_missingness_filter"],
                    "n_features_after_missingness_filter": fit_info["n_features_after_missingness_filter"],
                    "n_features_after_mice_preprocessing": fit_info["n_features_after_mice_preprocessing"],
                    "feature_selection_k": fit_info["feature_selection_k"],
                    "n_features_after_feature_selection": fit_info["n_features_after_feature_selection"],
                    "n_transformed_features": fit_info["n_transformed_features"],
                    "dropped_missingness_columns": "|".join(fit_info["dropped_missingness_columns"]),
                    "dropped_unusable_columns": "|".join(fit_info["dropped_unusable_columns"]),
                    "raw_validation_ece": raw_val_ece,
                    "raw_test_ece": raw_test_ece,
                }
                model_row.update({f"raw_validation_{key}": value for key, value in raw_val_metrics.items()})
                model_row.update({f"raw_test_{key}": value for key, value in raw_test_metrics.items()})
                model_row.update({
                    f"calibrated_validation_{key}": value
                    for key, value in calibrated_val_metrics.items()
                })
                model_row.update({
                    f"calibrated_test_{key}": value
                    for key, value in calibrated_test_metrics.items()
                })
                model_row.update({f"validation_{key}": value for key, value in val_metrics.items()})
                model_row.update({f"test_{key}": value for key, value in test_metrics.items()})
                for key in CALIBRATION_DIAGNOSTIC_METRICS:
                    model_row[f"validation_{key}"] = calibrated_val_metrics[key]
                    model_row[f"test_{key}"] = calibrated_test_metrics[key]
                holdout_model_rows.append(model_row)

            print(
                f"{target_name:>13} | {feature_name:<20} | "
                f"day={landmark_day:>3} | "
                f"split={split_info['split_policy']} | "
                f"development_seed={split_info['development_split_seed']} | "
                f"fixed_test_seed={split_info['test_split_seed']} | "
                f"source={selected_feature_source} | "
                f"train/val/test={len(train_pos)}/{len(val_pos)}/{len(test_pos)} | "
                f"events={int(y_train.sum())}/{int(y_val.sum())}/{int(y_test.sum())} | "
                f"CV={min(K_FOLD_CV_SPLITS, min_class_count(y_train))}-fold"
                f"x{effective_cv_repeats_for_settings(y_train, optimization_settings)} | "
                f"tuning={optimization_settings['optuna_trials']} trials/"
                f"{optimization_settings['seed_bagging_repeats']} bag | "
                f"train CV AUROC={best_cv_mean_auroc:.3f} | "
                f"train ensemble OOF AUROC={ensemble_train_cv_auroc:.3f} | "
                f"model={selected_model_family} | "
                f"strategy={selected_strategy['strategy_name']} | "
                f"ensemble={ensemble_top_k_actual} | final_fit={final_refit_scope} | "
                f"report={selected_report['probability_source']} | "
                f"SINGLE-MODEL TEST AUROC={test_metrics['auroc']:.3f} | "
                f"SINGLE-MODEL TEST AUPRC={test_metrics['auprc']:.3f}"
            )

holdout_results = pd.DataFrame(holdout_rows)
holdout_threshold_candidates = pd.DataFrame(holdout_threshold_candidate_rows)
holdout_train_cv_threshold_candidates = pd.DataFrame(holdout_train_cv_threshold_candidate_rows)
holdout_train_cv_folds = pd.DataFrame(holdout_train_cv_fold_rows)
holdout_best_params = pd.DataFrame(holdout_best_param_rows)
holdout_optuna_trials = pd.DataFrame(holdout_optuna_trial_rows)
holdout_strategy_candidates = pd.DataFrame(holdout_strategy_candidate_rows)
holdout_feature_source_candidates = pd.DataFrame(holdout_feature_source_candidate_rows)
holdout_predictions = pd.DataFrame(holdout_prediction_rows)
holdout_model_summary = pd.DataFrame(holdout_model_rows)
holdout_skipped = pd.DataFrame(holdout_skipped_rows)
holdout_simplified_feature_ranking = pd.DataFrame()
holdout_simplified_shap_cv_folds = pd.DataFrame()
holdout_temporal_ensemble_candidates = pd.DataFrame(
    columns=TEMPORAL_ENSEMBLE_CANDIDATE_COLUMNS
)
holdout_optimization_audit = pd.DataFrame()
holdout_bootstrap_ci = pd.DataFrame(columns=BOOTSTRAP_CI_OUTPUT_COLUMNS)

if RUN_HOLDOUT_EVALUATION:
    (
        holdout_results,
        holdout_threshold_candidates,
        holdout_train_cv_threshold_candidates,
        holdout_train_cv_folds,
        holdout_best_params,
        holdout_optuna_trials,
        holdout_strategy_candidates,
        holdout_feature_source_candidates,
        holdout_predictions,
        holdout_model_summary,
        holdout_simplified_feature_ranking,
        holdout_simplified_shap_cv_folds,
    ) = run_simplified_model_pass(
        holdout_results,
        holdout_threshold_candidates,
        holdout_train_cv_threshold_candidates,
        holdout_train_cv_folds,
        holdout_best_params,
        holdout_optuna_trials,
        holdout_strategy_candidates,
        holdout_feature_source_candidates,
        holdout_predictions,
        holdout_model_summary,
    )
    holdout_results = ensure_model_variant_column(holdout_results)
    holdout_threshold_candidates = ensure_model_variant_column(holdout_threshold_candidates)
    holdout_train_cv_threshold_candidates = ensure_model_variant_column(holdout_train_cv_threshold_candidates)
    holdout_train_cv_folds = ensure_model_variant_column(holdout_train_cv_folds)
    holdout_best_params = ensure_model_variant_column(holdout_best_params)
    holdout_optuna_trials = ensure_model_variant_column(holdout_optuna_trials)
    holdout_strategy_candidates = ensure_model_variant_column(holdout_strategy_candidates)
    holdout_feature_source_candidates = ensure_model_variant_column(holdout_feature_source_candidates)
    holdout_predictions = ensure_model_variant_column(holdout_predictions)
    holdout_model_summary = ensure_model_variant_column(holdout_model_summary)
    (
        holdout_results,
        holdout_predictions,
        holdout_threshold_candidates,
        holdout_train_cv_threshold_candidates,
        holdout_model_summary,
        holdout_temporal_ensemble_candidates,
    ) = (
        apply_cumulative_landmark_prediction_ensemble(
            holdout_results,
            holdout_predictions,
            holdout_threshold_candidates,
            holdout_train_cv_threshold_candidates,
            holdout_model_summary,
        )
    )
    print("\nFinal reported holdout performance after temporal stabilization:")
    for _, final_row in holdout_results.iterrows():
        print(
            f"FINAL {final_row['target']:>13} | "
            f"{final_row['feature_set']:<20} | "
            f"{final_row['model_variant']:<10} | "
            f"method={final_row['probability_ensemble_method']} | "
            f"candidate={final_row['probability_ensemble_selected_candidate']} | "
            f"TEST AUROC={float(final_row['test_auroc']):.3f} | "
            f"TEST AUPRC={float(final_row['test_auprc']):.3f}"
        )
    holdout_bootstrap_ci = bootstrap_holdout_metric_cis(holdout_predictions)
    holdout_results = attach_bootstrap_cis_to_results(
        holdout_results,
        holdout_bootstrap_ci,
    )
    test_min_count = pd.to_numeric(
        holdout_results["test_min_class_count"],
        errors="coerce",
    )
    train_min_count = pd.to_numeric(
        holdout_results["train_min_class_count"],
        errors="coerce",
    )
    validation_min_count = pd.to_numeric(
        holdout_results["validation_min_class_count"],
        errors="coerce",
    )
    holdout_results["effective_train_cv_splits"] = (
        np.minimum(K_FOLD_CV_SPLITS, train_min_count).round().astype("Int64")
    )
    holdout_results["requested_train_cv_folds_available"] = (
        train_min_count >= K_FOLD_CV_SPLITS
    )
    holdout_results["validation_selection_reliable"] = (
        validation_min_count >= VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT
    )
    holdout_results["test_auroc_reliable"] = (
        test_min_count >= MIN_RELIABLE_TEST_CLASS_COUNT
    )
    holdout_results["test_auroc_reliability_status"] = np.where(
        holdout_results["test_auroc_reliable"],
        "primary_estimate",
        "exploratory_too_few_test_events_or_controls",
    )
    holdout_results["endpoint_estimability_status"] = np.select(
        [
            train_min_count < K_FOLD_CV_SPLITS,
            validation_min_count < VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT,
            test_min_count < MIN_RELIABLE_TEST_CLASS_COUNT,
        ],
        [
            "exploratory_reduced_cv_due_to_sparse_training_events",
            "exploratory_unstable_validation_selection",
            "exploratory_too_few_test_events_or_controls",
        ],
        default="primary_estimate",
    )
    holdout_results["test_auroc_ge_target_reliable"] = (
        holdout_results["test_auroc_reliable"]
        & (pd.to_numeric(holdout_results["test_auroc"], errors="coerce") >= AUROC_TARGET)
    )
    development_cv_auroc = pd.to_numeric(
        holdout_results["ensemble_train_cv_oof_auroc"],
        errors="coerce",
    )
    test_auroc = pd.to_numeric(holdout_results["test_auroc"], errors="coerce")
    test_auroc_ci_lower = pd.to_numeric(
        holdout_results["test_auroc_ci_lower"],
        errors="coerce",
    )
    test_auroc_ci_upper = pd.to_numeric(
        holdout_results["test_auroc_ci_upper"],
        errors="coerce",
    )
    holdout_results["development_cv_auroc_ge_target"] = (
        development_cv_auroc >= AUROC_TARGET
    )
    holdout_results["test_auroc_ci_contains_target"] = (
        (test_auroc_ci_lower <= AUROC_TARGET)
        & (test_auroc_ci_upper >= AUROC_TARGET)
    )
    holdout_results["development_cv_minus_test_auroc"] = (
        development_cv_auroc - test_auroc
    )
    holdout_results["auroc_target_status"] = np.select(
        [
            holdout_results["test_auroc_ge_target_reliable"],
            ~holdout_results["test_auroc_reliable"],
            holdout_results["test_auroc_ci_contains_target"],
            holdout_results["development_cv_auroc_ge_target"],
        ],
        [
            "target_met_on_reliable_holdout",
            "exploratory_holdout_too_few_events_or_controls",
            "point_estimate_below_target_but_95ci_includes_target",
            "development_cv_met_target_but_holdout_did_not",
        ],
        default="target_not_met_on_development_or_holdout",
    )
    if not holdout_model_summary.empty and not holdout_results.empty:
        model_update_cols = [
            "target",
            "feature_set",
            "model_variant",
            *[
                col for col in holdout_results.columns
                if col.startswith((
                    "validation_",
                    "test_",
                    "raw_validation_",
                    "raw_test_",
                    "calibrated_validation_",
                    "calibrated_test_",
                    "single_model_",
                ))
                or col in {
                    "reported_probability_source",
                    "probability_ensemble_method",
                    "probability_ensemble_members",
                    "probability_ensemble_weights",
                    "probability_ensemble_selected_candidate",
                    "probability_ensemble_selection_split",
                    "probability_ensemble_selection_rule",
                    "probability_ensemble_test_labels_used_for_selection",
                    "selected_threshold_objective",
                    "selected_threshold",
                    "selected_threshold_validation_score",
                    "train_cv_selected_threshold_objective",
                    "train_cv_selected_threshold",
                    "train_cv_selected_threshold_score",
                    "ensemble_train_cv_oof_auroc",
                    "ensemble_train_cv_oof_auprc",
                    "ensemble_train_cv_oof_auprc_lift",
                    "threshold_candidate_policy",
                    "threshold_tie_break_policy",
                    "threshold_minimum_sensitivity",
                    "threshold_operating_point_status",
                    "threshold_selection_split",
                    "threshold_probability_source_provenance",
                    "threshold_selection_labels_provenance",
                    "threshold_test_labels_used_for_selection",
                    "validation_model_selection_score",
                    "calibration_diagnostic_method",
                    "raw_validation_ece",
                    "raw_test_ece",
                }
            ],
        ]
        model_update_cols = list(OrderedDict.fromkeys(model_update_cols))
        update_table = holdout_results.loc[:, model_update_cols]
        stale_cols = [
            col for col in update_table.columns
            if col not in {"target", "feature_set", "model_variant"} and col in holdout_model_summary.columns
        ]
        holdout_model_summary = (
            holdout_model_summary
            .drop(columns=stale_cols, errors="ignore")
            .merge(update_table, on=["target", "feature_set", "model_variant"], how="left")
        )

    holdout_optimization_audit = build_model_optimization_audit(
        holdout_results,
        holdout_strategy_candidates,
        holdout_threshold_candidates,
        holdout_optuna_trials,
        holdout_temporal_ensemble_candidates,
        holdout_predictions,
        holdout_train_cv_threshold_candidates,
        holdout_model_summary,
    )

    holdout_results.to_csv(OUTPUT_DIR / "holdout_test_rows.csv", index=False)
    holdout_threshold_candidates.to_csv(OUTPUT_DIR / "holdout_validation_threshold_candidates.csv", index=False)
    holdout_train_cv_threshold_candidates.to_csv(OUTPUT_DIR / "holdout_train_cv_threshold_candidates.csv", index=False)
    holdout_train_cv_folds.to_csv(OUTPUT_DIR / "holdout_train_cv_fold_preprocessing.csv", index=False)
    holdout_best_params.to_csv(OUTPUT_DIR / "holdout_best_params.csv", index=False)
    holdout_optuna_trials.to_csv(OUTPUT_DIR / "holdout_optuna_trials.csv", index=False)
    holdout_strategy_candidates.to_csv(OUTPUT_DIR / "holdout_validation_strategy_candidates.csv", index=False)
    holdout_feature_source_candidates.to_csv(OUTPUT_DIR / "holdout_feature_source_candidates.csv", index=False)
    holdout_predictions.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)
    holdout_bootstrap_ci.to_csv(OUTPUT_DIR / "holdout_bootstrap_95ci.csv", index=False)
    holdout_model_summary.to_csv(OUTPUT_DIR / "holdout_model_summary.csv", index=False)
    holdout_skipped.to_csv(OUTPUT_DIR / "holdout_skipped_rows.csv", index=False)
    holdout_simplified_feature_ranking.to_csv(
        OUTPUT_DIR / "holdout_simplified_feature_ranking.csv",
        index=False,
    )
    holdout_simplified_shap_cv_folds.to_csv(
        OUTPUT_DIR / "holdout_simplified_shap_cv_folds.csv",
        index=False,
    )
    holdout_temporal_ensemble_candidates.to_csv(
        OUTPUT_DIR / "holdout_temporal_ensemble_candidates.csv",
        index=False,
    )
    holdout_optimization_audit.to_csv(
        OUTPUT_DIR / "model_optimization_audit.csv",
        index=False,
    )
    holdout_model_summary.to_csv(OUTPUT_DIR / "final_model_summary.csv", index=False)
    display(holdout_results)
    display(holdout_bootstrap_ci)
    display(holdout_temporal_ensemble_candidates)
    display(holdout_optimization_audit)
    if not holdout_skipped.empty:
        display(holdout_skipped)
    if (
        STRICT_OPTIMIZATION_AUDIT
        and not holdout_optimization_audit.empty
        and not holdout_optimization_audit["optimization_audit_pass"].all()
    ):
        failed_models = holdout_optimization_audit.loc[
            ~holdout_optimization_audit["optimization_audit_pass"],
            ["target", "feature_set", "model_variant", "failed_checks"],
        ]
        raise RuntimeError(
            "Per-model optimization audit failed:\n"
            + failed_models.to_string(index=False)
        )

# %% [markdown]
# ## 7. Performance summaries
#
# Assemble validation and held-out test performance summaries.

# %%
if RUN_HOLDOUT_EVALUATION and not holdout_results.empty:
    metric_columns = [
        "validation_auroc",
        "validation_auprc",
        "validation_prevalence",
        "validation_auprc_baseline",
        "validation_auprc_lift",
        "validation_accuracy",
        "validation_specificity",
        "validation_sensitivity",
        "validation_ppv",
        "validation_npv",
        "validation_balanced_accuracy",
        "validation_youden_index",
        "validation_f1",
        "validation_net_benefit",
        "validation_decision_curve_net_benefit_at_configured_threshold",
        "validation_ece",
        "validation_calibration_intercept",
        "validation_calibration_slope",
        "raw_validation_ece",
        "raw_validation_calibration_intercept",
        "raw_validation_calibration_slope",
        "raw_validation_auroc",
        "raw_validation_auprc",
        "raw_validation_auprc_lift",
        "raw_test_calibration_intercept",
        "raw_test_calibration_slope",
        "calibrated_validation_auroc",
        "calibrated_validation_auprc",
        "calibrated_validation_auprc_lift",
        "calibrated_validation_ece",
        "calibrated_validation_calibration_intercept",
        "calibrated_validation_calibration_slope",
        "test_auroc",
        "test_auprc",
        "test_prevalence",
        "test_auprc_baseline",
        "test_auprc_lift",
        "test_accuracy",
        "test_specificity",
        "test_sensitivity",
        "test_ppv",
        "test_npv",
        "test_balanced_accuracy",
        "test_youden_index",
        "test_f1",
        "test_net_benefit",
        "test_decision_curve_net_benefit_at_configured_threshold",
        "test_ece",
        "test_calibration_intercept",
        "test_calibration_slope",
        "raw_test_ece",
        "raw_test_auroc",
        "raw_test_auprc",
        "raw_test_auprc_lift",
        "calibrated_test_auroc",
        "calibrated_test_auprc",
        "calibrated_test_auprc_lift",
        "calibrated_test_ece",
        "calibrated_test_calibration_intercept",
        "calibrated_test_calibration_slope",
        "selected_threshold",
    ]
    summary_rows = []
    for (target, feature_set, model_variant), group in holdout_results.groupby([
        "target",
        "feature_set",
        "model_variant",
    ]):
        for metric in metric_columns:
            values = (
                pd.to_numeric(group[metric], errors="coerce")
                if metric in group.columns
                else pd.Series(dtype=float)
            )
            summary_rows.append({
                "target": target,
                "feature_set": feature_set,
                "model_variant": model_variant,
                "metric": metric,
                "value": values.iloc[0] if len(values) else np.nan,
            })
    holdout_metric_summary = pd.DataFrame(summary_rows)
    holdout_metric_summary = add_bootstrap_cis_to_metric_summary(
        holdout_metric_summary,
        holdout_results,
        holdout_bootstrap_ci,
    )
    holdout_metric_summary.to_csv(OUTPUT_DIR / "holdout_metric_summary.csv", index=False)
    display(holdout_metric_summary)

    auroc_check = (
        holdout_results
        .loc[:, [
            "target",
            "feature_set",
            "model_variant",
            "test_auroc",
            "test_auroc_ci_lower",
            "test_auroc_ci_upper",
            "test_auroc_bootstrap_valid_resamples",
            "selected_threshold",
            "effective_train_cv_splits",
            "requested_train_cv_folds_available",
            "validation_selection_reliable",
            "test_min_class_count",
            "test_auroc_reliable",
            "test_auroc_reliability_status",
            "endpoint_estimability_status",
            "test_auroc_ge_target_reliable",
            "development_cv_auroc_ge_target",
            "test_auroc_ci_contains_target",
            "development_cv_minus_test_auroc",
            "auroc_target_status",
        ]]
        .assign(
            auroc_target=AUROC_TARGET,
            test_auroc_ge_target=lambda x: x["test_auroc"] >= x["auroc_target"],
        )
    )
    auroc_check.to_csv(OUTPUT_DIR / "holdout_auroc_target_check.csv", index=False)
    display(auroc_check)
    unmet_auroc_target = auroc_check.loc[
        ~auroc_check["test_auroc_ge_target"]
    ]
    if not unmet_auroc_target.empty:
        print(
            f"AUROC target {AUROC_TARGET:.2f} was not met by "
            f"{len(unmet_auroc_target)}/{len(auroc_check)} holdout estimates. "
            "The held-out labels are audit-only and are not used to select a split, "
            "hyperparameters, probability orientation, or a reporting variant."
        )

    landmark_order = ["Preoperative", "Post transplant D+30", "Post transplant D+60", "Post transplant D+90"]

    def heldout_rows_and_labels(target, feature_set, model_variant):
        required = {
            "target", "feature_set", "model_variant", "dataset_split",
            "source_row_index", "y_true",
        }
        if holdout_predictions.empty or not required.issubset(holdout_predictions.columns):
            return tuple()
        mask = (
            (holdout_predictions["target"] == target)
            & (holdout_predictions["feature_set"] == feature_set)
            & (holdout_predictions["model_variant"] == model_variant)
            & (holdout_predictions["dataset_split"] == "test")
        )
        rows = (
            holdout_predictions.loc[mask, ["source_row_index", "y_true"]]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .sort_values(["source_row_index", "y_true"], kind="mergesort")
        )
        return tuple(map(tuple, rows.to_numpy()))

    def result_split_seed(row):
        value = pd.to_numeric(pd.Series([row.get("split_seed", np.nan)]), errors="coerce").iloc[0]
        return int(value) if pd.notna(value) else None

    order_rows = []
    for (target, model_variant), group in holdout_results.groupby(["target", "model_variant"]):
        ordered = (
            group.set_index("feature_set")
            .reindex(landmark_order)
            .reset_index()
        )
        aurocs = pd.to_numeric(ordered["test_auroc"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(aurocs)
        best_idx = int(np.nanargmax(aurocs)) if np.any(valid) else -1
        observed_non_decreasing = (
            bool(np.all(np.diff(aurocs[valid]) >= -1e-12))
            if np.sum(valid) >= 2
            else pd.NA
        )
        valid_rows = ordered.loc[valid]
        split_seeds = [
            seed for seed in (result_split_seed(row) for _, row in valid_rows.iterrows())
            if seed is not None
        ]
        unique_split_seeds = sorted(set(split_seeds))
        signatures = [
            heldout_rows_and_labels(target, row["feature_set"], model_variant)
            for _, row in valid_rows.iterrows()
        ]
        if np.sum(valid) < 2:
            comparison_valid = False
            comparison_reason = "insufficient_estimable_landmarks"
        elif len(unique_split_seeds) != 1:
            comparison_valid = False
            comparison_reason = "different_test_split_seeds"
        elif any(not signature for signature in signatures):
            comparison_valid = False
            comparison_reason = "test_prediction_rows_unavailable"
        elif all(signature == signatures[0] for signature in signatures[1:]):
            comparison_valid = True
            comparison_reason = "same_test_rows_and_labels"
        else:
            comparison_valid = False
            comparison_reason = "different_incident_risk_sets_or_labels"
        is_non_decreasing = observed_non_decreasing if comparison_valid else pd.NA
        for i, row in ordered.iterrows():
            test_prevalence = row.get("test_prevalence", np.nan)
            test_auprc = row.get("test_auprc", np.nan)
            auprc_lift = (
                float(test_auprc / test_prevalence)
                if pd.notna(test_prevalence) and test_prevalence
                else np.nan
            )
            order_rows.append({
                "target": target,
                "model_variant": model_variant,
                "feature_set": row["feature_set"],
                "selected_feature_source": row.get("selected_feature_source", np.nan),
                "selected_feature_source_day": row.get("selected_feature_source_day", np.nan),
                "landmark_order": i,
                "test_auroc": row.get("test_auroc", np.nan),
                "test_auroc_ci_lower": row.get("test_auroc_ci_lower", np.nan),
                "test_auroc_ci_upper": row.get("test_auroc_ci_upper", np.nan),
                "test_auprc": test_auprc,
                "test_auprc_ci_lower": row.get("test_auprc_ci_lower", np.nan),
                "test_auprc_ci_upper": row.get("test_auprc_ci_upper", np.nan),
                "test_prevalence": test_prevalence,
                "test_auprc_lift": auprc_lift,
                "validation_auroc": row.get("validation_auroc", np.nan),
                "validation_auprc": row.get("validation_auprc", np.nan),
                "validation_auprc_lift": row.get("validation_auprc_lift", np.nan),
                "best_train_cv_mean_auroc": row.get("best_train_cv_mean_auroc", np.nan),
                "best_train_cv_mean_auprc": row.get("best_train_cv_mean_auprc", np.nan),
                "best_train_cv_auprc_lift": row.get("best_train_cv_auprc_lift", np.nan),
                "ensemble_train_cv_oof_auroc": row.get("ensemble_train_cv_oof_auroc", np.nan),
                "ensemble_train_cv_oof_auprc": row.get("ensemble_train_cv_oof_auprc", np.nan),
                "ensemble_train_cv_oof_auprc_lift": row.get("ensemble_train_cv_oof_auprc_lift", np.nan),
                "analysis_positive_n": row.get("analysis_positive_n", np.nan),
                "train_positive_n": row.get("train_positive_n", np.nan),
                "validation_positive_n": row.get("validation_positive_n", np.nan),
                "test_positive_n": row.get("test_positive_n", np.nan),
                "n_features_after_missingness_filter": row.get("n_features_after_missingness_filter", np.nan),
                "n_features_after_feature_selection": row.get("n_features_after_feature_selection", np.nan),
                "n_transformed_features": row.get("n_transformed_features", np.nan),
                "feature_selection_k": row.get("feature_selection_k", np.nan),
                "landmark_comparison_valid": comparison_valid,
                "landmark_comparison_reason": comparison_reason,
                "landmark_split_seeds": "|".join(map(str, unique_split_seeds)),
                "observed_test_auroc_non_decreasing": observed_non_decreasing,
                "target_is_non_decreasing": is_non_decreasing,
                "target_best_feature_set": ordered.loc[best_idx, "feature_set"] if best_idx >= 0 else np.nan,
            })
    landmark_order_audit = pd.DataFrame(order_rows)
    landmark_order_audit.to_csv(OUTPUT_DIR / "holdout_landmark_order_audit.csv", index=False)
    display(landmark_order_audit)

    paired_landmark_rows = []
    probability_column = (
        "probability" if "probability" in holdout_predictions.columns else "raw_probability"
    )
    for (target, model_variant), group in holdout_results.groupby(["target", "model_variant"]):
        reference_results = group.loc[group["feature_set"] == "Preoperative"]
        if reference_results.empty:
            continue
        reference_result = reference_results.iloc[0]
        reference_mask = (
            (holdout_predictions["target"] == target)
            & (holdout_predictions["feature_set"] == "Preoperative")
            & (holdout_predictions["model_variant"] == model_variant)
            & (holdout_predictions["dataset_split"] == "test")
        )
        reference_predictions = holdout_predictions.loc[
            reference_mask,
            ["source_row_index", "y_true", probability_column],
        ].rename(columns={
            "y_true": "reference_y_true",
            probability_column: "reference_probability",
        })
        if reference_predictions.empty:
            continue
        if reference_predictions["source_row_index"].duplicated().any():
            raise ValueError("Duplicate preoperative test patients in paired landmark audit.")

        for landmark_feature_set in landmark_order[1:]:
            landmark_results = group.loc[group["feature_set"] == landmark_feature_set]
            if landmark_results.empty:
                continue
            landmark_result = landmark_results.iloc[0]
            landmark_mask = (
                (holdout_predictions["target"] == target)
                & (holdout_predictions["feature_set"] == landmark_feature_set)
                & (holdout_predictions["model_variant"] == model_variant)
                & (holdout_predictions["dataset_split"] == "test")
            )
            landmark_predictions = holdout_predictions.loc[
                landmark_mask,
                ["source_row_index", "y_true", probability_column],
            ].rename(columns={
                "y_true": "landmark_y_true",
                probability_column: "landmark_probability",
            })
            if landmark_predictions.empty:
                continue
            if landmark_predictions["source_row_index"].duplicated().any():
                raise ValueError("Duplicate later-landmark test patients in paired landmark audit.")

            common = reference_predictions.merge(
                landmark_predictions,
                on="source_row_index",
                how="inner",
                validate="one_to_one",
            )
            common = common.dropna(subset=[
                "reference_y_true",
                "landmark_y_true",
                "reference_probability",
                "landmark_probability",
            ]).copy()
            same_labels = bool(
                not common.empty
                and np.array_equal(
                    common["reference_y_true"].astype(int).to_numpy(),
                    common["landmark_y_true"].astype(int).to_numpy(),
                )
            )
            reference_seed = result_split_seed(reference_result)
            landmark_seed = result_split_seed(landmark_result)
            comparison_seed = int((
                BOOTSTRAP_CI_RANDOM_STATE
                + zlib.crc32(
                    f"{target}|{model_variant}|{landmark_feature_set}|paired_landmark".encode("utf-8")
                )
            ) % (2 ** 32))
            y_common = common["landmark_y_true"].astype(int).to_numpy()
            reference_probability = common["reference_probability"].astype(float).to_numpy()
            landmark_probability = common["landmark_probability"].astype(float).to_numpy()
            positive_n = int(y_common.sum()) if len(y_common) else 0
            negative_n = int(len(y_common) - positive_n)

            reference_common_auroc = np.nan
            landmark_common_auroc = np.nan
            auroc_difference = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
            valid_resamples = 0
            bootstrap_status = "not_run"
            evidence_of_decline = pd.NA
            evidence_of_improvement = pd.NA
            if not same_labels:
                comparison_status = "not_comparable_labels_differ_on_common_rows"
            elif not positive_n or not negative_n:
                comparison_status = "not_estimable_single_class_common_risk_set"
            else:
                reference_common_auroc = roc_auc_score(y_common, reference_probability)
                landmark_common_auroc = roc_auc_score(y_common, landmark_probability)
                auroc_difference = landmark_common_auroc - reference_common_auroc
                (
                    ci_lower,
                    ci_upper,
                    valid_resamples,
                    bootstrap_status,
                ) = paired_stratified_auroc_difference_ci(
                    y_common,
                    reference_probability,
                    landmark_probability,
                    comparison_seed,
                )
                if bootstrap_status == "ok":
                    evidence_of_decline = bool(ci_upper < 0.0)
                    evidence_of_improvement = bool(ci_lower > 0.0)
                    if evidence_of_decline:
                        comparison_status = "evidence_of_decline"
                    elif evidence_of_improvement:
                        comparison_status = "evidence_of_improvement"
                    else:
                        comparison_status = "no_evidence_of_auroc_difference"
                else:
                    comparison_status = "point_estimate_only_bootstrap_disabled"

            paired_landmark_rows.append({
                "target": target,
                "model_variant": model_variant,
                "reference_feature_set": "Preoperative",
                "landmark_feature_set": landmark_feature_set,
                "reference_split_seed": reference_seed,
                "landmark_split_seed": landmark_seed,
                "same_split_seed": reference_seed == landmark_seed,
                "reference_full_test_n": int(len(reference_predictions)),
                "landmark_full_test_n": int(len(landmark_predictions)),
                "common_test_n": int(len(common)),
                "common_test_positive_n": positive_n,
                "common_test_negative_n": negative_n,
                "common_fraction_of_landmark_test": (
                    float(len(common) / len(landmark_predictions))
                    if len(landmark_predictions)
                    else np.nan
                ),
                "same_labels_on_common_rows": same_labels,
                "reference_full_test_auroc": reference_result.get("test_auroc", np.nan),
                "landmark_full_test_auroc": landmark_result.get("test_auroc", np.nan),
                "reference_common_risk_set_auroc": reference_common_auroc,
                "landmark_common_risk_set_auroc": landmark_common_auroc,
                "landmark_minus_reference_auroc": auroc_difference,
                "difference_ci_lower": ci_lower,
                "difference_ci_upper": ci_upper,
                "confidence_level": BOOTSTRAP_CI_CONFIDENCE_LEVEL,
                "bootstrap_requested_resamples": (
                    BOOTSTRAP_CI_N_RESAMPLES if RUN_BOOTSTRAP_CI else 0
                ),
                "bootstrap_valid_resamples": valid_resamples,
                "bootstrap_method": "paired_class_stratified_transplantation_episode_bootstrap_percentile_ci",
                "bootstrap_seed": comparison_seed,
                "comparison_status": comparison_status,
                "evidence_of_decline": evidence_of_decline,
                "evidence_of_improvement": evidence_of_improvement,
            })

    paired_landmark_audit = pd.DataFrame(
        paired_landmark_rows,
        columns=PAIRED_LANDMARK_AUDIT_COLUMNS,
    )
    paired_landmark_audit.to_csv(
        OUTPUT_DIR / "holdout_paired_landmark_comparison.csv",
        index=False,
    )
    display(paired_landmark_audit)
    paired_estimable = paired_landmark_audit.loc[
        paired_landmark_audit["comparison_status"].isin([
            "evidence_of_decline",
            "evidence_of_improvement",
            "no_evidence_of_auroc_difference",
        ])
    ]
    paired_declines = int((paired_estimable["evidence_of_decline"] == True).sum())
    print(
        "Paired common-risk-set landmark audit: "
        f"{paired_declines}/{len(paired_estimable)} comparisons show evidence "
        f"of AUROC decline at {BOOTSTRAP_CI_CONFIDENCE_LEVEL:.0%} confidence."
    )

    severity_rows = []
    for endpoint_family in ["aGVHD", "oGVHD", "cGVHD"]:
        any_target = f"{endpoint_family}_any"
        severe_target = f"{endpoint_family}_severe"
        for feature_set in landmark_order:
            for model_variant in [MODEL_VARIANT_ORIGINAL, MODEL_VARIANT_SIMPLIFIED]:
                any_rows = holdout_results.loc[
                    (holdout_results["target"] == any_target)
                    & (holdout_results["feature_set"] == feature_set)
                    & (holdout_results["model_variant"] == model_variant)
                ]
                severe_rows = holdout_results.loc[
                    (holdout_results["target"] == severe_target)
                    & (holdout_results["feature_set"] == feature_set)
                    & (holdout_results["model_variant"] == model_variant)
                ]
                if any_rows.empty or severe_rows.empty:
                    continue
                any_row = any_rows.iloc[0]
                severe_row = severe_rows.iloc[0]
                any_auroc = float(any_row.get("test_auroc", np.nan))
                severe_auroc = float(severe_row.get("test_auroc", np.nan))
                any_auprc = float(any_row.get("test_auprc", np.nan))
                severe_auprc = float(severe_row.get("test_auprc", np.nan))
                any_seed = result_split_seed(any_row)
                severe_seed = result_split_seed(severe_row)
                any_signature = heldout_rows_and_labels(any_target, feature_set, model_variant)
                severe_signature = heldout_rows_and_labels(severe_target, feature_set, model_variant)
                any_episode_set = {row_index for row_index, _ in any_signature}
                severe_episode_set = {row_index for row_index, _ in severe_signature}
                union_episode_set = any_episode_set | severe_episode_set
                same_test_episode_set = bool(
                    any_episode_set
                    and severe_episode_set
                    and any_episode_set == severe_episode_set
                )
                overlap_fraction = (
                    len(any_episode_set & severe_episode_set) / len(union_episode_set)
                    if union_episode_set
                    else np.nan
                )
                if any_seed != severe_seed:
                    comparison_status = "descriptive_different_test_cohorts"
                elif not same_test_episode_set:
                    comparison_status = "descriptive_different_endpoint_risk_sets"
                else:
                    comparison_status = "descriptive_same_cohort_different_binary_outcomes"
                severity_rows.append({
                    "endpoint_family": endpoint_family,
                    "feature_set": feature_set,
                    "model_variant": model_variant,
                    "any_stage_split_seed": any_seed,
                    "severe_split_seed": severe_seed,
                    "same_test_episode_set": same_test_episode_set,
                    "test_episode_overlap_fraction": overlap_fraction,
                    "comparison_status": comparison_status,
                    "performance_order_constraint_valid": False,
                    "any_stage_test_positive_n": any_row.get("test_positive_n", np.nan),
                    "severe_test_positive_n": severe_row.get("test_positive_n", np.nan),
                    "any_stage_test_auroc": any_auroc,
                    "severe_test_auroc": severe_auroc,
                    "severe_minus_any_stage_auroc": severe_auroc - any_auroc,
                    "severe_auroc_higher": severe_auroc > any_auroc,
                    "any_stage_test_auprc": any_auprc,
                    "severe_test_auprc": severe_auprc,
                    "severe_minus_any_stage_auprc": severe_auprc - any_auprc,
                    "any_stage_auroc_ci_lower": any_row.get("test_auroc_ci_lower", np.nan),
                    "any_stage_auroc_ci_upper": any_row.get("test_auroc_ci_upper", np.nan),
                    "severe_auroc_ci_lower": severe_row.get("test_auroc_ci_lower", np.nan),
                    "severe_auroc_ci_upper": severe_row.get("test_auroc_ci_upper", np.nan),
                    "any_stage_estimability_status": any_row.get("endpoint_estimability_status", np.nan),
                    "severe_estimability_status": severe_row.get("endpoint_estimability_status", np.nan),
                    "interpretation": (
                        "Descriptive only: severe and any-stage are separate binary outcomes, and "
                        "AUROC is not expected to be higher for severe disease. Compare cohort "
                        "overlap, event counts, and confidence intervals."
                    ),
                })
    holdout_severity_order_audit = pd.DataFrame(severity_rows, columns=[
        "endpoint_family", "feature_set", "model_variant",
        "any_stage_split_seed", "severe_split_seed", "same_test_episode_set",
        "test_episode_overlap_fraction", "comparison_status",
        "performance_order_constraint_valid",
        "any_stage_test_positive_n", "severe_test_positive_n",
        "any_stage_test_auroc", "severe_test_auroc",
        "severe_minus_any_stage_auroc", "severe_auroc_higher",
        "any_stage_test_auprc", "severe_test_auprc",
        "severe_minus_any_stage_auprc",
        "any_stage_auroc_ci_lower", "any_stage_auroc_ci_upper",
        "severe_auroc_ci_lower", "severe_auroc_ci_upper",
        "any_stage_estimability_status", "severe_estimability_status",
        "interpretation",
    ])
    holdout_severity_order_audit.to_csv(
        OUTPUT_DIR / "holdout_severity_order_audit.csv",
        index=False,
    )
    display(holdout_severity_order_audit)
elif RUN_HOLDOUT_EVALUATION:
    pd.DataFrame(columns=[
        "target", "feature_set", "model_variant", "metric", "value",
        "ci_lower", "ci_upper", "confidence_level",
        "bootstrap_valid_resamples", "bootstrap_status",
    ]).to_csv(
        OUTPUT_DIR / "holdout_metric_summary.csv",
        index=False,
    )
    pd.DataFrame(columns=[
        "target", "feature_set", "model_variant", "test_auroc",
        "test_auroc_ci_lower", "test_auroc_ci_upper",
        "test_auroc_bootstrap_valid_resamples", "selected_threshold",
    ]).to_csv(
        OUTPUT_DIR / "holdout_auroc_target_check.csv",
        index=False,
    )
    pd.DataFrame(columns=[
        "target",
        "model_variant",
        "feature_set",
        "landmark_order",
        "test_auroc",
        "test_auroc_ci_lower",
        "test_auroc_ci_upper",
        "test_auprc_ci_lower",
        "test_auprc_ci_upper",
        "analysis_positive_n",
        "test_positive_n",
        "n_features_after_missingness_filter",
        "n_transformed_features",
        "landmark_comparison_valid",
        "landmark_comparison_reason",
        "landmark_split_seeds",
        "observed_test_auroc_non_decreasing",
        "target_is_non_decreasing",
        "target_best_feature_set",
    ]).to_csv(OUTPUT_DIR / "holdout_landmark_order_audit.csv", index=False)
    pd.DataFrame(columns=PAIRED_LANDMARK_AUDIT_COLUMNS).to_csv(
        OUTPUT_DIR / "holdout_paired_landmark_comparison.csv",
        index=False,
    )
    pd.DataFrame(columns=[
        "endpoint_family", "feature_set", "model_variant",
        "any_stage_split_seed", "severe_split_seed", "same_test_episode_set",
        "test_episode_overlap_fraction", "comparison_status",
        "performance_order_constraint_valid",
        "any_stage_test_positive_n", "severe_test_positive_n",
        "any_stage_test_auroc", "severe_test_auroc",
        "severe_minus_any_stage_auroc", "severe_auroc_higher",
        "any_stage_test_auprc", "severe_test_auprc",
        "severe_minus_any_stage_auprc", "interpretation",
    ]).to_csv(OUTPUT_DIR / "holdout_severity_order_audit.csv", index=False)

# %% [markdown]
# ## 8. Subgroup analysis
#
# Calculate descriptive held-out performance across prespecified subgroups.

# %%
SUBGROUP_SOURCE_COLUMNS = {
    "Age": "Age",
    "Sex": "Male Sex",
    "Diagnosis": "Diagnosis",
    "Graft source": "Graft source",
    "Conditioning regimen": "Myeloablating regimen MAC vs RIC",
    "Relatedness": "Related",
    "Prophylaxis": "Prophylaxis regimen",
}
HLA_FULL_MATCH_COLUMNS = [
    col for col in HLA_FEATURES
    if "antigen" in col.lower()
]
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
SUBGROUP_BOOTSTRAP_TARGETS = set(env_list(
    "XGB_V2_SUBGROUP_BOOTSTRAP_TARGETS",
    "oGVHD_any,oGVHD_severe,cGVHD_any,cGVHD_severe",
))
SUBGROUP_BOOTSTRAP_FEATURE_SETS = set(env_list(
    "XGB_V2_SUBGROUP_BOOTSTRAP_FEATURE_SETS",
    "Preoperative",
))
SUBGROUP_BOOTSTRAP_MODEL_VARIANTS = set(env_list(
    "XGB_V2_SUBGROUP_BOOTSTRAP_MODEL_VARIANTS",
    MODEL_VARIANT_SIMPLIFIED,
))
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
        subgroup_frames.append(pd.DataFrame({
            "source_row_index": source_index,
            "subgroup_name": subgroup_name,
            "subgroup_level": levels[subgroup_name].astype("string"),
            "subgroup_definition": SUBGROUP_DEFINITIONS[subgroup_name],
        }))
    return pd.concat(subgroup_frames, ignore_index=True)


def subgroup_estimability_status(positive_n, negative_n):
    if positive_n == 0 or negative_n == 0:
        return "single_class_not_estimable"
    if min(positive_n, negative_n) < MIN_RELIABLE_TEST_CLASS_COUNT:
        return "exploratory_min_class_count_lt_reliability_threshold"
    return "reliable"


def subgroup_bootstrap_seed(target, feature_set, model_variant, subgroup_name, subgroup_level):
    key = "|".join(map(str, [
        target,
        feature_set,
        model_variant,
        subgroup_name,
        subgroup_level,
    ])).encode("utf-8")
    return int((BOOTSTRAP_CI_RANDOM_STATE + zlib.crc32(key)) % (2 ** 32))


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
        output.update({
            "bootstrap_requested_resamples": 0,
            "bootstrap_confidence_level": BOOTSTRAP_CI_CONFIDENCE_LEVEL,
            "bootstrap_method": "not_run",
            "bootstrap_sampling_unit": "held_out_transplantation_episode",
            "bootstrap_seed": int(seed),
        })
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
    output.update({
        "bootstrap_requested_resamples": BOOTSTRAP_CI_N_RESAMPLES,
        "bootstrap_confidence_level": BOOTSTRAP_CI_CONFIDENCE_LEVEL,
        "bootstrap_method": "ordinary_transplantation_episode_bootstrap_percentile_ci",
        "bootstrap_sampling_unit": "held_out_transplantation_episode",
        "bootstrap_seed": int(seed),
    })
    return output


def calculate_test_subgroup_metrics(predictions, source_data):
    if predictions.empty:
        return pd.DataFrame(columns=SUBGROUP_RESULT_ID_COLUMNS + SUBGROUP_METRIC_COLUMNS)
    test_predictions = predictions.loc[
        predictions["dataset_split"].eq("test")
    ].copy()
    if test_predictions.empty:
        return pd.DataFrame(columns=SUBGROUP_RESULT_ID_COLUMNS + SUBGROUP_METRIC_COLUMNS)

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
        y_prob = pd.to_numeric(group["probability"], errors="coerce").to_numpy(dtype=float)
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
            raise ValueError("A subgroup model contains more than one selected threshold.")

        positive_n = int(y_true.sum())
        negative_n = int(len(y_true) - positive_n)
        status = subgroup_estimability_status(positive_n, negative_n)
        threshold_values_dict = threshold_metrics(y_true, y_prob, threshold)
        prevalence = float(np.mean(y_true))
        auprc = average_precision_score(y_true, y_prob) if positive_n else np.nan
        recalibrated_metrics = calibration_regression_metrics(y_true, calibration_prob)
        probability_sources = group["reported_probability_source"].dropna().astype(str).unique()
        row = dict(zip(group_columns, keys))
        row.update({
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
        })
        row.update(threshold_values_dict)
        subgroup_seed = subgroup_bootstrap_seed(
            row["target"],
            row["feature_set"],
            row["model_variant"],
            row["subgroup_name"],
            row["subgroup_level"],
        )
        row.update(bootstrap_subgroup_metric_cis(
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
        ))
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
    result = result.sort_values(
        [
            "target",
            "model_variant",
            "landmark_day",
            "subgroup_order",
            "subgroup_level",
        ],
        kind="mergesort",
    ).drop(columns="subgroup_order").reset_index(drop=True)
    return result


holdout_subgroup_metrics = calculate_test_subgroup_metrics(
    holdout_predictions if RUN_HOLDOUT_EVALUATION else pd.DataFrame(),
    df,
)
holdout_subgroup_metric_summary = holdout_subgroup_metrics.melt(
    id_vars=SUBGROUP_RESULT_ID_COLUMNS,
    value_vars=SUBGROUP_METRIC_COLUMNS,
    var_name="metric",
    value_name="value",
) if not holdout_subgroup_metrics.empty else pd.DataFrame(
    columns=SUBGROUP_RESULT_ID_COLUMNS + ["metric", "value"]
)
if not holdout_subgroup_metric_summary.empty:
    for bound in ["ci_lower", "ci_upper", "bootstrap_valid_resamples", "bootstrap_status"]:
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
holdout_subgroup_definitions = pd.DataFrame({
    "subgroup_name": SUBGROUP_ORDER,
    "definition": [SUBGROUP_DEFINITIONS[name] for name in SUBGROUP_ORDER],
})

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

# %% [markdown]
# ## 9. Diagnostic plots
#
# Generate per-endpoint discrimination, calibration, decision, and SHAP plots.

# %%
FIG_DPI = int(os.getenv("XGB_V2_FIG_DPI", os.getenv("XGB_FIG_DPI", "600")))
PLOT_CALIBRATION_N_BINS = int(os.getenv("XGB_V2_PLOT_CALIBRATION_N_BINS", os.getenv("XGB_PLOT_CALIBRATION_N_BINS", "5")))
PLOT_FONT_FAMILY = os.getenv("XGB_V2_PLOT_FONT_FAMILY", "Calibri")
PANEL_FIG_DPI = int(os.getenv("XGB_V2_PANEL_FIG_DPI", "300"))
PANEL_SHAP_MAX_DISPLAY = int(os.getenv("XGB_V2_PANEL_SHAP_MAX_DISPLAY", "8"))
PANEL_TITLE_FONTSIZE = int(os.getenv("XGB_V2_PANEL_TITLE_FONTSIZE", "22"))
PANEL_TITLE_PAD = float(os.getenv("XGB_V2_PANEL_TITLE_PAD", "4"))
PANEL_LEGEND_FONTSIZE = int(os.getenv("XGB_V2_PANEL_LEGEND_FONTSIZE", "18"))
PANEL_AXIS_LABEL_FONTSIZE = int(os.getenv("XGB_V2_PANEL_AXIS_LABEL_FONTSIZE", "15"))
PANEL_TICK_FONTSIZE = int(os.getenv("XGB_V2_PANEL_TICK_FONTSIZE", "13"))
PANEL_SHAP_AXIS_LABEL_FONTSIZE = int(os.getenv("XGB_V2_PANEL_SHAP_AXIS_LABEL_FONTSIZE", "18"))
PANEL_SHAP_TICK_FONTSIZE = int(os.getenv("XGB_V2_PANEL_SHAP_TICK_FONTSIZE", "16"))
PANEL_SHAP_FEATURE_FONTSIZE = int(os.getenv("XGB_V2_PANEL_SHAP_FEATURE_FONTSIZE", "16"))
PANEL_SHAP_LABEL_WRAP = int(os.getenv("XGB_V2_PANEL_SHAP_LABEL_WRAP", "26"))
PANEL_SHAP_CMAP = shap.plots.colors.red_blue

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
PANEL_FEATURE_SETS = OrderedDict({
    "Pretransplant": "Preoperative",
    "D+30 Post-transplant": "Post transplant D+30",
    "D+60 Post-transplant": "Post transplant D+60",
    "D+90 Post-transplant": "Post transplant D+90",
})

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
    fig.subplots_adjust(top=0.91, bottom=0.20, left=0.07, right=0.98, hspace=0.22, wspace=0.14)
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
    axes = np.array([
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ], dtype=object)
    for ax in axes:
        ax.set_box_aspect(1)
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis("off")
    fig.suptitle(title, y=0.965, fontsize=20)
    return fig, axes, legend_ax


def style_axis(ax):
    ax.tick_params(axis="both", which="both", direction="out", length=5, width=1.1, color="#222222", labelsize=14)
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
                category = value[len(prefix):]
                if is_hla_matching_column(col):
                    return f"HLA-{hla_locus_display_name(col)} Antigen Matched: {category}"
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
        return pd.to_numeric(group["probability"], errors="coerce").to_numpy(dtype=float)
    return pd.to_numeric(group["raw_probability"], errors="coerce").to_numpy(dtype=float)


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
                "Decision-curve input has missing model/split provenance in "
                f"{column}."
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
    margin = (
        z
        * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
        / denominator
    )
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def calibration_points(y_true, y_prob, n_bins=5, strategy="quantile"):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    valid = np.isfinite(y_prob) & np.isfinite(y_true)
    y_true = y_true[valid]
    y_prob = np.clip(y_prob[valid], 0.0, 1.0)
    if len(y_true) == 0:
        return pd.DataFrame(columns=[
            "mean_predicted_probability",
            "observed_event_rate",
            "observed_event_rate_lower",
            "observed_event_rate_upper",
            "event_n",
            "n",
            "ci_level",
            "ci_method",
        ])

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
        rows.append({
            "mean_predicted_probability": float(np.mean(y_prob[in_bin])),
            "observed_event_rate": observed,
            "observed_event_rate_lower": lower,
            "observed_event_rate_upper": upper,
            "event_n": event_n,
            "n": n,
            "ci_level": 0.95,
            "ci_method": "Wilson",
        })
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
            yerr = np.vstack([
                y - group["observed_event_rate_lower"].astype(float),
                group["observed_event_rate_upper"].astype(float) - y,
            ])
            yerr = np.nan_to_num(np.clip(yerr, 0.0, None), nan=0.0)
            ece_values = pd.to_numeric(
                group.get("calibration_ece", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            ece = float(ece_values.iloc[0]) if len(ece_values) else np.nan
            label_title = f"{panel_title} {model_variant_display_name(model_variant)}".strip()
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
        rows.append({
            "threshold": float(threshold),
            "model_net_benefit": float(model_nb),
            "treat_all_net_benefit": float(all_nb),
            "treat_none_net_benefit": 0.0,
            "model_standardized_net_benefit": float(model_nb / prevalence) if prevalence else np.nan,
            "treat_all_standardized_net_benefit": float(all_nb / prevalence) if prevalence else np.nan,
            "treat_none_standardized_net_benefit": 0.0,
            "prevalence": prevalence,
        })
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
        y_limit_values.extend([
            dca["model_net_benefit"].to_numpy(dtype=float),
            np.clip(dca["treat_all_net_benefit"].to_numpy(dtype=float), DCA_Y_MIN, None),
            dca["treat_none_net_benefit"].to_numpy(dtype=float),
        ])

    y_values = np.concatenate([values[np.isfinite(values)] for values in y_limit_values if len(values)])
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
        reference_dca = dca.loc[dca["model_variant"] == ordered_model_variants_present(dca)[0]]
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
        dca_table.to_csv(PLOT_DIR / f"holdout_decision_curve_{endpoint_file_slug(target)}.csv", index=False)


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
        # SHAP remains a member-model explanation. Resolve the current
        # landmark's immutable base bundle from the portable wrapper rather
        # than treating the orchestration wrapper as an estimator.
        member_artifacts = bundle.get("ordered_member_artifacts", [])
        current_member = next(
            (
                item for item in reversed(member_artifacts)
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
            raise RuntimeError(f"Temporal member artifact hash mismatch: {base_model_path}")
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
    selected_indices = getattr(select_step, "selected_indices_", np.arange(len(feature_names)))
    selected_feature_names = [
        feature_names[idx]
        for idx in selected_indices
        if idx < len(feature_names)
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

    components = [
        shap_frames_for_fitted_model(member, X)
        for member in model.models_
    ]
    shap_frames = [component[0] for component in components]
    value_frames = [component[1] for component in components]
    pretty_name_map = {}
    for _, _, local_pretty_names in components:
        pretty_name_map.update(local_pretty_names)

    all_features = sorted({
        feature
        for frame in shap_frames
        for feature in frame.columns
    })
    shap_stack = np.stack([
        frame.reindex(columns=all_features, fill_value=0.0).to_numpy(dtype=float)
        for frame in shap_frames
    ])
    value_stack = np.stack([
        frame.reindex(columns=all_features).to_numpy(dtype=float)
        for frame in value_frames
    ])
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
    return shap_frame.to_numpy(dtype=float), value_frame.to_numpy(dtype=float), pretty_names


def save_shap_feature_importance_plot(target, feature_set, model_variant=MODEL_VARIANT_ORIGINAL):
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
    shap_values, feature_values, pretty_names = shap_values_for_model(bundle["model"], X_test)
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


if RUN_HOLDOUT_EVALUATION and not holdout_predictions.empty:
    for target in TARGETS_TO_RUN:
        save_endpoint_curve_grid(target, "roc")
        save_endpoint_curve_grid(target, "pr")
        save_endpoint_decision_curve_grid(target)
if RUN_HOLDOUT_EVALUATION and not holdout_predictions.empty:
    calibration_rows = []
    test_predictions = holdout_predictions.loc[holdout_predictions["dataset_split"] == "test"]
    test_predictions = ensure_model_variant_column(test_predictions)
    for (target, feature_set, model_variant), group in test_predictions.groupby([
        "target",
        "feature_set",
        "model_variant",
    ]):
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
    calibration_table.to_csv(OUTPUT_DIR / "holdout_calibration_points.csv", index=False)

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
    pd.DataFrame(columns=[
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
    ]).to_csv(OUTPUT_DIR / "holdout_calibration_points.csv", index=False)

# %% [markdown]
# ## 10. Panel plots
#
# Generate compact cross-endpoint model diagnostic panels.

# %%
PANEL_GVHD_ENDPOINTS = OrderedDict({
    "aGVHD": ("aGVHD_any", "aGVHD_severe"),
    "cGVHD": ("cGVHD_any", "cGVHD_severe"),
    "oGVHD": ("oGVHD_any", "oGVHD_severe"),
})


def panel_variant_name(model_variant):
    return "Simplified" if str(model_variant) == MODEL_VARIANT_SIMPLIFIED else "Original"


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
        center_x = (min(position.x0 for position in positions) + max(position.x1 for position in positions)) / 2
        fig.text(
            center_x,
            0.90,
            panel_endpoint_title(target),
            ha="center",
            va="center",
            fontsize=PANEL_TITLE_FONTSIZE,
            fontweight="semibold",
        )
    fig.suptitle(metric_title, y=0.95, fontsize=PANEL_TITLE_FONTSIZE, fontweight="semibold")
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
            plot_panel_discrimination_axis(ax, target, feature_set, panel_title, curve_kind)
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
        y_prob = pd.to_numeric(group["calibration_probability"], errors="coerce").to_numpy(dtype=float)
        calibration = calibration_points(y_true, y_prob, PLOT_CALIBRATION_N_BINS)
        if calibration.empty:
            continue
        observed = calibration["observed_event_rate"].to_numpy(dtype=float)
        yerr = np.vstack([
            observed - calibration["observed_event_rate_lower"].to_numpy(dtype=float),
            calibration["observed_event_rate_upper"].to_numpy(dtype=float) - observed,
        ])
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
            markerfacecolor=(color if model_variant == MODEL_VARIANT_SIMPLIFIED else "white"),
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
                y_values.extend([
                    np.clip(reference["treat_all_net_benefit"].to_numpy(dtype=float), DCA_Y_MIN, None),
                    reference["treat_none_net_benefit"].to_numpy(dtype=float),
                ])
    finite_values = np.concatenate([
        values[np.isfinite(values)] for values in y_values if len(values)
    ])
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
        Line2D([], [], color=GRAY_TREAT, linestyle="--", linewidth=1.6, label="Treat all"),
        Line2D([], [], color=GRAY_TREAT, linestyle="-.", linewidth=1.6, label="Treat none"),
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
                normalized_value = np.clip((value_feature - lower) / (upper - lower), 0.0, 1.0)
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
    fig, axes_by_target, all_axes = make_dual_endpoint_panel_figure(title, targets, shap_panel=True)
    plotted = False
    for target, axes in axes_by_target.items():
        for ax, (panel_title, feature_set) in zip(axes, PANEL_FEATURE_SETS.items()):
            plotted = plot_compact_shap_axis(ax, target, feature_set, panel_title) or plotted
    if plotted:
        scalar_mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=PANEL_SHAP_CMAP)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=all_axes, fraction=0.012, pad=0.018)
        colorbar.set_ticks([0.0, 1.0])
        colorbar.set_ticklabels(["Low", "High"])
        colorbar.set_label("Relative feature value", fontsize=PANEL_LEGEND_FONTSIZE)
        colorbar.ax.tick_params(labelsize=PANEL_LEGEND_FONTSIZE)
    save_panel_figure(fig, f"panel_shap_{slugify(gvhd_label).lower()}.png")


def save_all_panel_plots():
    for gvhd_label, targets in PANEL_GVHD_ENDPOINTS.items():
        save_gvhd_discrimination_panel(gvhd_label, targets, "roc")
        save_gvhd_discrimination_panel(gvhd_label, targets, "pr")
        save_gvhd_calibration_panel(gvhd_label, targets)
        save_gvhd_net_benefit_panel(gvhd_label, targets)
        if RUN_FINAL_MODELS and not holdout_model_summary.empty:
            save_gvhd_shap_panel(gvhd_label, targets)


if RUN_HOLDOUT_EVALUATION and not holdout_predictions.empty:
    save_all_panel_plots()

# %% [markdown]
# ## 11. Run metadata
#
# Save split audits, configuration metadata, and output locations.

# %%
frozen_test_split_audit = pd.DataFrame(
    list(FROZEN_TEST_SPLIT_AUDIT.values())
)
frozen_test_split_audit.to_csv(
    OUTPUT_DIR / "frozen_test_split_audit.csv",
    index=False,
)

run_config = {
    "data_path": str(DATA_PATH),
    "data_sha256": DATA_SHA256,
    "model_profile": MODEL_PROFILE_NAME,
    "sheet_name": SHEET_NAME,
    "pre_mice_complete_case_policy": "disabled_fold_safe_mice",
    "pre_mice_complete_case_steps": PRE_MICE_COMPLETE_CASE_STEPS,
    "pre_mice_complete_case_initial_n": int(
        pre_mice_complete_case_audit.iloc[0]["input_n"]
    ),
    "pre_mice_complete_case_final_n": int(len(df)),
    "pre_mice_complete_case_excluded_n": int(
        pre_mice_complete_case_audit.iloc[0]["input_n"] - len(df)
    ),
    "pre_mice_complete_case_audit_file": "pre_mice_complete_case_exclusion_audit.csv",
    "mandatory_covariate_missingness_retained": remaining_mandatory_missing,
    "remaining_missing_data_policy": "train_fold_missingness_filter_then_fold_safe_mice",
    "random_state": RANDOM_STATE,
    "use_frozen_test_split_manifest": USE_FROZEN_TEST_SPLIT_MANIFEST,
    "strict_frozen_test_split_manifest": STRICT_FROZEN_TEST_SPLIT_MANIFEST,
    "frozen_test_split_manifest_path": str(FROZEN_TEST_SPLIT_MANIFEST_PATH),
    "frozen_test_split_manifest_sha256": FROZEN_TEST_SPLIT_MANIFEST_SHA256,
    "frozen_test_split_audit_file": "frozen_test_split_audit.csv",
    "frozen_test_split_identifier": (
        "sha256(Patient ID|Transplant Date); identifiers excluded from features"
    ),
    "frozen_test_split_test_labels_used_for_current_selection": False,
    "split_random_state_overrides": SPLIT_RANDOM_STATE_OVERRIDES,
    "fixed_split_seed_overrides": FIXED_SPLIT_SEED_OVERRIDES,
    "development_split_seed_overrides": DEVELOPMENT_SPLIT_SEED_OVERRIDES,
    "split_seed_search_enabled": SPLIT_SEED_SEARCH_ENABLED,
    "split_seed_search_candidates": SPLIT_SEED_SEARCH_CANDIDATES,
    "split_seed_search_n_estimators": SPLIT_SEED_SEARCH_N_ESTIMATORS,
    "selected_development_split_seeds": SELECTED_DEVELOPMENT_SPLIT_SEEDS,
    "split_seed_search_test_labels_used": False,
    "run_holdout_evaluation": RUN_HOLDOUT_EVALUATION,
    "run_final_models": RUN_FINAL_MODELS,
    "split_policy": SPLIT_POLICY,
    "train_size": TRAIN_SIZE,
    "validation_size": VAL_SIZE,
    "test_size": TEST_SIZE,
    "k_fold_cv_splits": K_FOLD_CV_SPLITS,
    "optuna_cv_repeats": OPTUNA_CV_REPEATS,
    "optuna_trials": OPTUNA_TRIALS,
    "optuna_progress_interval": OPTUNA_PROGRESS_INTERVAL,
    "optuna_trial_overrides": OPTUNA_TRIAL_OVERRIDES,
    "optuna_timeout": OPTUNA_TIMEOUT,
    "optuna_startup_trials": OPTUNA_STARTUP_TRIALS,
    "use_preset_trials": USE_PRESET_TRIALS,
    "xgb_n_jobs": XGB_N_JOBS,
    "search_n_jobs": SEARCH_N_JOBS,
    "pipeline_cache_enabled": PIPELINE_CACHE_ENABLED,
    "pipeline_cache_version": PIPELINE_CACHE_VERSION,
    "pipeline_cache_directory": str(PIPELINE_CACHE_DIR),
    "pipeline_cache_scope": [
        "deterministic_fold_preprocessing",
        "complete_optuna_cv_scores",
    ],
    "parallel_execution_scope": "independent_cv_folds_only",
    "in_run_exact_component_fit_reuse": True,
    "optuna_objective_metric": OPTUNA_OBJECTIVE_METRIC,
    "optuna_std_penalty": OPTUNA_STD_PENALTY,
    "auprc_lift_weight": AUPRC_LIFT_WEIGHT,
    "validation_candidate_trials": VALIDATION_CANDIDATE_TRIALS,
    "validation_candidate_trial_overrides": VALIDATION_CANDIDATE_TRIAL_OVERRIDES,
    "ensemble_top_k": ENSEMBLE_TOP_K,
    "ensemble_top_k_candidates": ENSEMBLE_TOP_K_CANDIDATES,
    "seed_bagging_repeats": SEED_BAGGING_REPEATS,
    "seed_bagging_repeat_overrides": SEED_BAGGING_REPEAT_OVERRIDES,
    "robust_validation_selection": ROBUST_VALIDATION_SELECTION,
    "validation_selection_prior_min_class_count": VALIDATION_SELECTION_PRIOR_MIN_CLASS_COUNT,
    "final_cv_bagging": FINAL_CV_BAGGING,
    "final_cv_bagging_include_full_refit": FINAL_CV_BAGGING_INCLUDE_FULL_REFIT,
    "finalization_candidates": FINALIZATION_CANDIDATES,
    "finalization_selection_method": "validation_pareto_near_best_auroc_then_auprc",
    "strict_optimization_audit": STRICT_OPTIMIZATION_AUDIT,
    "optimization_audit_method": "per_model_train_cv_validation_selection_assertion",
    "final_refit_scope": FINAL_REFIT_SCOPE,
    "final_refit_scope_overrides": FINAL_REFIT_SCOPE_OVERRIDES,
    "high_missing_cutoff": HIGH_MISSING_CUTOFF,
    "mice_max_iter": MICE_MAX_ITER,
    "mice_iter_candidates": MICE_ITER_CANDIDATES,
    "xgb_booster_candidates": XGB_BOOSTER_CANDIDATES,
    "agvhd_xgb_reproduction_profile": AGVHD_XGB_REPRODUCTION_PROFILE,
    "agvhd_xgb_reproduction_data_match": AGVHD_XGB_REPRODUCTION_DATA_MATCH,
    "archived_agvhd_reproduction_data_sha256": (
        ARCHIVED_AGVHD_REPRODUCTION_DATA_SHA256
    ),
    "agvhd_xgb_reproduction_analysis_keys": sorted(
        AGVHD_XGB_REPRODUCTION_PRESETS
    ),
    "agvhd_xgb_reproduction_source": (
        "archived_training_only_optuna_top2_xgboost_parameters"
    ),
    "agvhd_xgb_reproduction_test_labels_used_for_selection": False,
    "min_reliable_test_class_count": MIN_RELIABLE_TEST_CLASS_COUNT,
    "bootstrap_ci_metrics": list(BOOTSTRAP_CI_METRICS),
    "run_bootstrap_ci": RUN_BOOTSTRAP_CI,
    "bootstrap_ci_n_resamples": BOOTSTRAP_CI_N_RESAMPLES,
    "bootstrap_ci_confidence_level": BOOTSTRAP_CI_CONFIDENCE_LEVEL,
    "bootstrap_ci_random_state": BOOTSTRAP_CI_RANDOM_STATE,
    "bootstrap_ci_min_valid_resamples": BOOTSTRAP_CI_MIN_VALID_RESAMPLES,
    "bootstrap_ci_method": (
        "ordinary_transplantation_episode_bootstrap_percentile_ci"
        if RUN_BOOTSTRAP_CI
        else "disabled"
    ),
    "bootstrap_ci_threshold_policy": "fixed_validation_selected_threshold",
    "subgroup_analysis_enabled": True,
    "subgroup_analysis_split": "test",
    "subgroup_analysis_definitions": SUBGROUP_DEFINITIONS,
    "subgroup_analysis_hla_full_match_columns": HLA_FULL_MATCH_COLUMNS,
    "subgroup_analysis_min_reliable_class_count": MIN_RELIABLE_TEST_CLASS_COUNT,
    "subgroup_analysis_bootstrap_ci": bool(RUN_BOOTSTRAP_CI),
    "subgroup_analysis_bootstrap_targets": sorted(SUBGROUP_BOOTSTRAP_TARGETS),
    "subgroup_analysis_bootstrap_feature_sets": sorted(SUBGROUP_BOOTSTRAP_FEATURE_SETS),
    "subgroup_analysis_bootstrap_model_variants": sorted(SUBGROUP_BOOTSTRAP_MODEL_VARIANTS),
    "subgroup_analysis_bootstrap_metrics": list(SUBGROUP_CI_METRICS),
    "subgroup_analysis_output_files": [
        "holdout_subgroup_metrics.csv",
        "holdout_subgroup_metric_summary.csv",
        "holdout_subgroup_definitions.csv",
    ],
    "min_total_positives": MIN_TOTAL_POSITIVES,
    "target_timing_policy": TARGET_TIMING_POLICY,
    "feature_selection_candidates": FEATURE_SELECTION_CANDIDATES,
    "performance_rescue_analyses": sorted(PERFORMANCE_RESCUE_ANALYSES),
    "performance_rescue_optuna_trials": PERFORMANCE_RESCUE_OPTUNA_TRIALS,
    "performance_rescue_validation_candidate_trials": (
        PERFORMANCE_RESCUE_VALIDATION_CANDIDATE_TRIALS
    ),
    "performance_rescue_cv_repeats": PERFORMANCE_RESCUE_CV_REPEATS,
    "performance_rescue_feature_selection_candidates": (
        PERFORMANCE_RESCUE_FEATURE_SELECTION_CANDIDATES
    ),
    "performance_rescue_simplified_feature_selection_candidates": (
        PERFORMANCE_RESCUE_SIMPLIFIED_FEATURE_SELECTION_CANDIDATES
    ),
    "performance_rescue_finalization_candidates": (
        PERFORMANCE_RESCUE_FINALIZATION_CANDIDATES
    ),
    "performance_rescue_test_labels_used_for_selection": False,
    "landmark_feature_source_policy": LANDMARK_FEATURE_SOURCE_POLICY,
    "landmark_cumulative_prediction_ensemble": LANDMARK_CUMULATIVE_PREDICTION_ENSEMBLE,
    "landmark_cumulative_prediction_ensemble_method": (
        "cross_validated_temporal_landmark_convex_average"
    ),
    "landmark_temporal_ensemble_target_families": list(
        LANDMARK_TEMPORAL_ENSEMBLE_TARGET_FAMILIES
    ),
    "landmark_temporal_current_weight_candidates": (
        LANDMARK_TEMPORAL_CURRENT_WEIGHT_CANDIDATES
    ),
    "landmark_temporal_auroc_tolerance": LANDMARK_TEMPORAL_AUROC_TOLERANCE,
    "landmark_temporal_ensemble_selection_rule": (
        "train_oof_max_auroc_then_max_auprc_within_tolerance;validation_fallback"
    ),
    "landmark_temporal_ensemble_test_labels_used_for_selection": False,
    "landmark_temporal_ensemble_candidate_output": (
        "holdout_temporal_ensemble_candidates.csv"
    ),
    "paired_landmark_comparison_output": "holdout_paired_landmark_comparison.csv",
    "paired_landmark_comparison_method": (
        "common_later_risk_set_paired_class_stratified_transplantation_episode_bootstrap"
    ),
    "paired_landmark_comparison_changes_model_selection": False,
    "run_simplified_models": RUN_SIMPLIFIED_MODELS,
    "simplified_top_n_features": SIMPLIFIED_TOP_N_FEATURES,
    "simplified_retune": SIMPLIFIED_RETUNE,
    "simplified_optuna_trials": SIMPLIFIED_OPTUNA_TRIALS,
    "simplified_optuna_trial_overrides": SIMPLIFIED_OPTUNA_TRIAL_OVERRIDES,
    "simplified_mice_iter_candidates": SIMPLIFIED_MICE_ITER_CANDIDATES,
    "simplified_shap_ranking_mode": SIMPLIFIED_SHAP_RANKING_MODE,
    "simplified_shap_ranking_mode_overrides": SIMPLIFIED_SHAP_RANKING_MODE_OVERRIDES,
    "simplified_validation_candidate_trials": SIMPLIFIED_VALIDATION_CANDIDATE_TRIALS,
    "simplified_validation_candidate_trial_overrides": (
        SIMPLIFIED_VALIDATION_CANDIDATE_TRIAL_OVERRIDES
    ),
    "simplified_seed_bagging_repeats": SIMPLIFIED_SEED_BAGGING_REPEATS,
    "simplified_seed_bagging_repeat_overrides": (
        SIMPLIFIED_SEED_BAGGING_REPEAT_OVERRIDES
    ),
    "simplified_feature_selection_method": (
        "train_cv_oof_shap_stability_top_n_with_configurable_training_only_reference"
    ),
    "simplified_hyperparameter_method": "independent_train_5fold_optuna",
    "threshold_objectives": THRESHOLD_OBJECTIVES,
    "threshold_objective": THRESHOLD_OBJECTIVE,
    "threshold_selection_metric": INNER_THRESHOLD_SELECTION_METRIC,
    "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
    "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
    "threshold_minimum_sensitivity": THRESHOLD_MIN_SENSITIVITY,
    "threshold_operating_point_status": THRESHOLD_OPERATING_POINT_STATUS,
    "threshold_test_labels_used_for_selection": False,
    "classification_probability_policy": (
        "raw_class1_probability_or_prespecified_validation_supported_one_minus_raw;"
        "recalibration_reserved_for_calibration_diagnostics"
    ),
    "train_oof_calibration_reporting_policy": (
        "training_oof_recalibrated_score_for_ece_intercept_slope_only;"
        "raw_or_orientation_corrected_oof_score_for_all_other_metrics"
    ),
    "temporal_wrapper_path_resolution": "member_paths_relative_to_wrapper_directory",
    "validation_model_selection_metric": VALIDATION_MODEL_SELECTION_METRIC,
    "dual_discrimination_selection": DUAL_DISCRIMINATION_SELECTION,
    "auroc_near_best_tolerance": AUROC_NEAR_BEST_TOLERANCE,
    "dual_discrimination_selection_method": "pareto_near_best_auroc_then_max_auprc",
    "threshold_grid_min": float(THRESHOLD_GRID.min()),
    "threshold_grid_max": float(THRESHOLD_GRID.max()),
    "threshold_grid_n": int(len(THRESHOLD_GRID)),
    "net_benefit_threshold": NET_BENEFIT_THRESHOLD,
    "dca_threshold_min": DCA_THRESHOLD_MIN,
    "dca_threshold_max": DCA_THRESHOLD_MAX,
    "dca_probability_source": DCA_PROBABILITY_SOURCE,
    "dca_y_min": DCA_Y_MIN,
    "dca_y_max": DCA_Y_MAX,
    "calibration_n_bins": CALIBRATION_N_BINS,
    "plot_calibration_n_bins": PLOT_CALIBRATION_N_BINS,
    "figure_dpi": FIG_DPI,
    "plot_font_family": PLOT_FONT_FAMILY,
    "panel_figure_dpi": PANEL_FIG_DPI,
    "panel_shap_max_display": PANEL_SHAP_MAX_DISPLAY,
    "panel_title_fontsize": PANEL_TITLE_FONTSIZE,
    "panel_title_pad": PANEL_TITLE_PAD,
    "panel_legend_fontsize": PANEL_LEGEND_FONTSIZE,
    "panel_axis_label_fontsize": PANEL_AXIS_LABEL_FONTSIZE,
    "panel_tick_fontsize": PANEL_TICK_FONTSIZE,
    "panel_shap_axis_label_fontsize": PANEL_SHAP_AXIS_LABEL_FONTSIZE,
    "panel_shap_tick_fontsize": PANEL_SHAP_TICK_FONTSIZE,
    "panel_shap_feature_fontsize": PANEL_SHAP_FEATURE_FONTSIZE,
    "panel_shap_label_wrap": PANEL_SHAP_LABEL_WRAP,
    "panel_shap_colormap": "shap.plots.colors.red_blue",
    "panel_calibration_model_variant": MODEL_VARIANT_SIMPLIFIED,
    "panel_plot_layout": "two_endpoints_each_with_2x2_landmarks",
    "calibration_diagnostic_method": CALIBRATION_DIAGNOSTIC_METHOD,
    "enforce_positive_calibration_slope": ENFORCE_POSITIVE_CALIBRATION_SLOPE,
    "allow_probability_orientation_correction": ALLOW_PROBABILITY_ORIENTATION_CORRECTION,
    "orientation_correction_targets": ORIENTATION_CORRECTION_TARGETS,
    "orientation_correction_min_validation_auroc": ORIENTATION_CORRECTION_MIN_VALIDATION_AUROC,
    "orientation_correction_requires_train_cv_support": (
        ORIENTATION_CORRECTION_REQUIRES_TRAIN_CV_SUPPORT
    ),
    "orientation_correction_validation_only_analyses": sorted(
        ORIENTATION_CORRECTION_VALIDATION_ONLY_ANALYSES
    ),
    "targets_to_run": TARGETS_TO_RUN,
    "feature_sets_to_run": FEATURE_SETS_TO_RUN,
    "forbidden_features": sorted(FORBIDDEN_FEATURES),
    "removed_identifier_columns": identifier_columns,
}

with open(OUTPUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2, default=jsonable)

print(f"Saved model outputs to: {OUTPUT_DIR}")
print(f"Saved final models to: {MODEL_DIR}")
