"""Cohort loading and landmark feature construction."""

from .utils import (
    display,
    hashlib,
    json,
    make_episode_split_hashes,
    np,
    pd,
)
from .config import (
    DATA_PATH,
    FEATURE_SETS_TO_RUN,
    FROZEN_TEST_SPLIT_MANIFEST_PATH,
    HIGH_MISSING_CUTOFF,
    OUTPUT_DIR,
    PROFILE,
    SHEET_NAME,
    USE_FROZEN_TEST_SPLIT_MANIFEST,
)

# Import-safe placeholders are populated by run_stage().
df = pd.DataFrame()
SPLIT_EPISODE_HASHES = pd.Series(dtype="string")
FROZEN_TEST_SPLIT_MANIFEST_SHA256 = None
frozen_test_split_manifest = pd.DataFrame()
HLA_FEATURES = []
PRE_MICE_COMPLETE_CASE_STEPS = []
pre_mice_complete_case_audit = pd.DataFrame()
remaining_mandatory_missing = {}
TARGET_DEFINITIONS = {}
FORBIDDEN_FEATURES = set()
identifier_columns = []
ALL_FEATURE_SETS = []
FEATURE_SETS = []
LANDMARK_ORDER = []


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
            f"Missing mandatory pre-MICE complete-case columns: {missing_columns}"
        )

    initial_n = int(len(data))
    filtered = data.copy()
    audit_rows = [
        {
            "step_order": 0,
            "step_name": "initial_cohort",
            "required_columns": "",
            "input_n": initial_n,
            "excluded_n": 0,
            "remaining_n": initial_n,
            "excluded_pct_of_input": 0.0,
            "cumulative_excluded_n": 0,
            "missing_count_by_column": "{}",
        }
    ]

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
        audit_rows.append(
            {
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
            }
        )

    return filtered.reset_index(drop=True), pd.DataFrame(audit_rows)


# Do not remove patients before modeling.  Mandatory covariate missingness is
# retained and handled by the fold-safe missingness filter plus MICE pipeline.
# Pre-MICE complete-case exclusion changes the cohort and can materially alter
# the held-out class distribution and discrimination estimates.


def add_landmark_labs(preoperative_columns, *lab_groups):
    columns = list(preoperative_columns)
    for lab_group in lab_groups:
        columns.extend(lab_group)
    return columns


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


def make_feature_sets(feature_sets_to_include=None):
    feature_sets_to_include = (
        set(feature_sets_to_include) if feature_sets_to_include is not None else None
    )
    feature_sets = []
    for group in LANDMARK_FEATURE_GROUPS:
        if (
            feature_sets_to_include is not None
            and group["feature_set"] not in feature_sets_to_include
        ):
            continue
        columns = unique_existing_columns(group["columns"])
        identifier_overlap = sorted(set(columns).intersection(identifier_columns))
        forbidden_overlap = sorted(set(columns).intersection(FORBIDDEN_FEATURES))
        if identifier_overlap:
            raise ValueError(
                f"Identifier columns reached {group['feature_set']}: {identifier_overlap}"
            )
        if forbidden_overlap:
            raise ValueError(
                f"Forbidden columns reached {group['feature_set']}: {forbidden_overlap}"
            )
        if not columns:
            raise ValueError(f"{group['feature_set']} has no usable predictors.")
        feature_sets.append(
            {
                "feature_set": group["feature_set"],
                "landmark_day": group["landmark_day"],
                "X": feature_source_df[columns].copy(),
                "columns": columns,
                "feature_variant": "original",
                "seed_feature_set": group["feature_set"],
                "candidate_variants": [],
                "description": group["description"],
            }
        )
    return feature_sets


# Cohort I/O is staged so importing feature helpers cannot touch clinical files.
def run_stage():
    """Load the cohort and build landmark feature sets."""
    global ALL_FEATURE_SETS, BASELINE_LAB_FEATURES, BASE_CLINICAL_FEATURES
    global D30_LAB_FEATURES, D60_LAB_FEATURES, D90_LAB_FEATURES
    global FEATURE_SETS, FORBIDDEN_FEATURES, FROZEN_TEST_SPLIT_MANIFEST_SHA256
    global FROZEN_TEST_SPLIT_REQUIRED_COLUMNS, HLA_FEATURES, IDENTIFIER_FEATURES
    global IDENTIFIER_PATTERNS, LANDMARK_DAY_BY_FEATURE_SET, LANDMARK_FEATURE_GROUPS
    global LANDMARK_ORDER, PREOPERATIVE_FEATURES, PRE_MICE_COMPLETE_CASE_STEPS
    global SPLIT_EPISODE_HASHES, TARGET_DEFINITIONS, df
    global feature_missingness, feature_set_summary, feature_source_df
    global frozen_test_split_manifest, identifier_columns, pre_mice_complete_case_audit
    global remaining_mandatory_missing

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
                f"Frozen test split manifest is missing columns: {missing_manifest_columns}"
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
            raise ValueError(
                "Frozen test split manifest contains duplicate episode hashes."
            )
    else:
        FROZEN_TEST_SPLIT_MANIFEST_SHA256 = None
        frozen_test_split_manifest = pd.DataFrame(
            columns=sorted(FROZEN_TEST_SPLIT_REQUIRED_COLUMNS)
        )

    HLA_FEATURES = [
        col
        for col in df.columns
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

    pre_mice_complete_case_audit = pd.DataFrame(
        [
            {
                "step_order": 0,
                "step_name": "pre_mice_exclusion_disabled",
                "required_columns": "",
                "input_n": int(len(df)),
                "excluded_n": 0,
                "remaining_n": int(len(df)),
                "excluded_pct_of_input": 0.0,
                "cumulative_excluded_n": 0,
                "missing_count_by_column": "{}",
            }
        ]
    )

    remaining_mandatory_missing = {
        col: int(raw_variable_missing_mask(df[col]).sum())
        for step in PRE_MICE_COMPLETE_CASE_STEPS
        for col in step["columns"]
    }

    print(
        "Mandatory pre-MICE missingness retained for fold-safe MICE:",
        remaining_mandatory_missing,
    )

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
        "Prophylaxis regimen",
        "Death_3M",
        "Overlap",
    }

    if PROFILE.include_prophylaxis:
        FORBIDDEN_FEATURES.discard("Prophylaxis regimen")

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
        col
        for col in df.columns
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
        "Related",
        "ABO matching",
        "CD34",
        "Donor T cell number",
    ]

    if PROFILE.include_prophylaxis:
        BASE_CLINICAL_FEATURES.insert(
            BASE_CLINICAL_FEATURES.index("Myeloablating regimen MAC vs RIC") + 1,
            "Prophylaxis regimen",
        )

    BASELINE_LAB_FEATURES = [
        col for col in feature_source_df.columns if col.endswith("_baseline")
    ]

    D30_LAB_FEATURES = [col for col in feature_source_df.columns if col.endswith("+30")]

    D60_LAB_FEATURES = [col for col in feature_source_df.columns if col.endswith("+60")]

    D90_LAB_FEATURES = [col for col in feature_source_df.columns if col.endswith("+90")]

    PREOPERATIVE_FEATURES = (
        BASE_CLINICAL_FEATURES + HLA_FEATURES + BASELINE_LAB_FEATURES
    )

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
            "columns": add_landmark_labs(
                PREOPERATIVE_FEATURES, D30_LAB_FEATURES, D60_LAB_FEATURES
            ),
            "description": "Pretransplant variables plus D+30 and D+60 labs.",
        },
        {
            "feature_set": "Post transplant D+90",
            "landmark_day": 90,
            "columns": add_landmark_labs(
                PREOPERATIVE_FEATURES,
                D30_LAB_FEATURES,
                D60_LAB_FEATURES,
                D90_LAB_FEATURES,
            ),
            "description": "Pretransplant variables plus D+30, D+60, and D+90 labs.",
        },
    ]

    missing_preop = [
        col for col in PREOPERATIVE_FEATURES if col not in feature_source_df.columns
    ]

    if missing_preop:
        raise ValueError(f"Missing expected pretransplant features: {missing_preop}")

    ALL_FEATURE_SETS = make_feature_sets()

    FEATURE_SETS = make_feature_sets(FEATURE_SETS_TO_RUN)

    LANDMARK_ORDER = [group["feature_set"] for group in LANDMARK_FEATURE_GROUPS]

    LANDMARK_DAY_BY_FEATURE_SET = {
        group["feature_set"]: group["landmark_day"] for group in LANDMARK_FEATURE_GROUPS
    }

    feature_set_summary = pd.DataFrame(
        [
            {
                "feature_set": item["feature_set"],
                "landmark_day": item["landmark_day"],
                "n_features_before_train_missingness_filter": item["X"].shape[1],
                "full_dataset_cols_missing_gt20": int(
                    (item["X"].isna().mean() > HIGH_MISSING_CUTOFF).sum()
                ),
                "description": item["description"],
            }
            for item in FEATURE_SETS
        ]
    )

    display(feature_set_summary)

    feature_set_summary.to_csv(OUTPUT_DIR / "feature_set_summary.csv", index=False)

    missingness_rows = []

    for item in FEATURE_SETS:
        for col, rate in item["X"].isna().mean().items():
            missingness_rows.append(
                {
                    "feature_set": item["feature_set"],
                    "feature": col,
                    "dtype": str(item["X"][col].dtype),
                    "missing_n": int(item["X"][col].isna().sum()),
                    "missing_rate_full_dataset": float(rate),
                    "would_drop_if_full_dataset_cutoff": bool(
                        rate > HIGH_MISSING_CUTOFF
                    ),
                }
            )

    feature_missingness = pd.DataFrame(missingness_rows)

    feature_missingness.to_csv(
        OUTPUT_DIR / "feature_missingness_by_landmark.csv", index=False
    )
