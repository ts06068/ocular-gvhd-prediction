"""Shared imports and small utility helpers."""

# Re-exported dependencies form the shared import facade.
# ruff: noqa: F401

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
        return {
            str(key): str(value).strip().lower()
            for key, value in json.loads(raw).items()
        }

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


# Hash identifiers only for split matching; identifiers never enter the predictor matrix.
def make_episode_split_hashes(data):
    required = ["Patient ID", "Transplant Date"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Frozen episode-level splitting requires columns: {missing}")
    patient_ids = data["Patient ID"].map(canonical_split_identifier)
    transplant_dates = (
        pd.to_datetime(data["Transplant Date"], errors="coerce")
        .dt.strftime("%Y-%m-%d")
        .fillna("<NA>")
    )
    return pd.Series(
        [
            hashlib.sha256(f"{patient_id}|{date}".encode("utf-8")).hexdigest()
            for patient_id, date in zip(patient_ids, transplant_dates)
        ],
        index=data.index,
        dtype="string",
    )
