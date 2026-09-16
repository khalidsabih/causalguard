from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    return float(np.mean((y_true - probability) ** 2))
