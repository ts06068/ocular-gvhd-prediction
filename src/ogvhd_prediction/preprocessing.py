"""Fold-safe preprocessing and estimator pipelines."""

from .utils import (
    BaseEstimator,
    IterativeImputer,
    OneHotEncoder,
    Pipeline,
    TransformerMixin,
    XGBClassifier,
    f_classif,
    np,
    pd,
    warnings,
)
from .config import (
    HIGH_MISSING_CUTOFF,
    MICE_MAX_ITER,
    PIPELINE_MEMORY,
    RANDOM_STATE,
    XGB_N_JOBS,
)


# Fit this transformer inside each fold; a global fit would leak missingness patterns.
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
HLA_MATCHING_NO_VALUES = {
    "no",
    "n",
    "false",
    "0",
    "mismatched",
    "mismatch",
    "not matched",
}


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
        lambda value: (
            mapping.get(str(value).strip(), np.nan) if pd.notna(value) else np.nan
        )
    )


class DataFramePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_iter=20, random_state=42):
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        X = coerce_hla_matching_columns_to_yes_indicator(X).replace(
            [np.inf, -np.inf], np.nan
        )
        self.input_columns_ = X.columns.tolist()
        initial_numeric_columns = [
            col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
        ]
        initial_categorical_columns = [
            col for col in X.columns if col not in initial_numeric_columns
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
            col for col in X_encoded.columns if col not in self.unusable_columns_
        ]

        for col in self.unusable_columns_:
            self.category_maps_.pop(col, None)

        if not self.encoded_columns_:
            raise ValueError("No predictors remained after MICE preprocessing.")

        self.categorical_columns_after_filter_ = [
            col for col in self.encoded_columns_ if col in self.category_maps_
        ]
        self.numeric_columns_after_filter_ = [
            col for col in self.encoded_columns_ if col not in self.category_maps_
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
        X = coerce_hla_matching_columns_to_yes_indicator(X).replace(
            [np.inf, -np.inf], np.nan
        )
        X_encoded = pd.DataFrame(index=X.index)

        for col in self.encoded_columns_:
            if col in X.columns:
                series = X[col]
            else:
                series = pd.Series(np.nan, index=X.index)

            if col in self.category_maps_:
                X_encoded[col] = encode_categorical_for_mice(
                    series, self.category_maps_[col]
                )
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
                X_final[col] = pd.Series(
                    [categories[code] for code in codes], index=X_imputed.index
                )
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
                X_imputed.loc[:, self.numeric_columns_after_filter_].to_numpy(
                    dtype=float
                )
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
        feature_names = [
            f"numeric__{col}" for col in self.numeric_columns_after_filter_
        ]
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
        scores = np.nan_to_num(
            scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf
        )
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
    return Pipeline(
        [
            ("missingness", MissingnessFilter(cutoff=HIGH_MISSING_CUTOFF)),
            # Fix the imputation seed so deterministic transforms share a cache;
            # XGBoost trials still keep their own model seeds.
            (
                "preprocess",
                DataFramePreprocessor(
                    max_iter=MICE_MAX_ITER, random_state=RANDOM_STATE
                ),
            ),
            ("select", TopKFeatureSelector(k="all")),
            ("model", model),
        ],
        memory=PIPELINE_MEMORY,
    )


def make_pipeline_for_params(seed, params):
    return make_xgb_pipeline(seed)
