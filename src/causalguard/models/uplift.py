from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression


@dataclass
class UpliftPredictions:
    p0: np.ndarray
    p1: np.ndarray
    uplift: np.ndarray


class TLearner:
    """Simple, interpretable uplift baseline using separate outcome models."""

    def __init__(self, base_estimator=None):
        if base_estimator is None:
            base_estimator = LogisticRegression(max_iter=1000)
        self.model_control = clone(base_estimator)
        self.model_treated = clone(base_estimator)
        self.feature_names: list[str] | None = None
        self.is_fitted = False

    def fit(self, frame: pd.DataFrame, feature_names: list[str]):
        self.feature_names = feature_names
        control = frame[frame["treatment"] == 0]
        treated = frame[frame["treatment"] == 1]
        if control["outcome"].nunique() < 2 or treated["outcome"].nunique() < 2:
            raise ValueError("Both treatment groups need both outcome classes to fit TLearner")
        self.model_control.fit(control[feature_names], control["outcome"])
        self.model_treated.fit(treated[feature_names], treated["outcome"])
        self.is_fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> UpliftPredictions:
        if not self.is_fitted or self.feature_names is None:
            raise RuntimeError("Model must be fit before predict")
        p0 = self.model_control.predict_proba(frame[self.feature_names])[:, 1]
        p1 = self.model_treated.predict_proba(frame[self.feature_names])[:, 1]
        return UpliftPredictions(p0=p0, p1=p1, uplift=p1 - p0)
