from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class ControlRiskModel:
    """Predicts untreated success probability from control observations."""

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.feature_names: list[str] | None = None
        self.is_fitted = False

    def fit(self, frame: pd.DataFrame, feature_names: list[str]):
        self.feature_names = feature_names
        control = frame[frame["treatment"] == 0]
        self.model.fit(control[feature_names], control["outcome"])
        self.is_fitted = True
        return self

    def predict_success(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.feature_names is None:
            raise RuntimeError("Model must be fit before predict")
        return self.model.predict_proba(frame[self.feature_names])[:, 1]

    def predict_risk(self, frame: pd.DataFrame) -> np.ndarray:
        return 1.0 - self.predict_success(frame)
