from __future__ import annotations

import numpy as np


def top_fraction(score: np.ndarray, fraction: float, eligible: np.ndarray | None = None) -> np.ndarray:
    """Treat the highest-scoring fraction of eligible observations."""
    score = np.asarray(score, dtype=float)
    n = len(score)
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    if eligible is None:
        eligible = np.ones(n, dtype=bool)
    else:
        eligible = np.asarray(eligible, dtype=bool)
    idx = np.flatnonzero(eligible)
    k = int(np.floor(len(idx) * fraction))
    treatment = np.zeros(n, dtype=int)
    if k <= 0:
        return treatment
    ranked = idx[np.argsort(score[idx])[::-1]]
    treatment[ranked[:k]] = 1
    return treatment


def profit_score(uplift: np.ndarray, outcome_value, treatment_cost: float) -> np.ndarray:
    return np.asarray(uplift, dtype=float) * np.asarray(outcome_value, dtype=float) - treatment_cost
