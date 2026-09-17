from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_logistic_pipeline(frame: pd.DataFrame, feature_names: list[str]) -> Pipeline:
    numeric = [c for c in feature_names if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in feature_names if c not in numeric]

    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            # Preserve probability calibration for causal probability differences.
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
