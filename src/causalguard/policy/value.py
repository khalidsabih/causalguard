from __future__ import annotations

import numpy as np
import pandas as pd


def true_incremental_value(
    frame: pd.DataFrame,
    treatment: np.ndarray,
    outcome_value,
    treatment_cost: float,
) -> float:
    """Oracle expected incremental value, available only in simulation."""
    treatment = np.asarray(treatment)
    uplift = frame["cate_true"].to_numpy()
    value = np.asarray(outcome_value, dtype=float)
    return float(np.sum(treatment * (uplift * value - treatment_cost)))


def randomized_holdout_value(
    frame: pd.DataFrame,
    outcome_value: float,
    treatment_cost: float,
) -> float:
    """Estimate incremental value from a randomized treatment/control holdout.

    The estimator is intentionally simple: difference in outcome rates times
    population size and outcome value, less treatment cost. It is unbiased when
    treatment assignment in the supplied frame is randomized with equal odds.
    """
    treated = frame[frame["treatment"] == 1]
    control = frame[frame["treatment"] == 0]
    if len(treated) == 0 or len(control) == 0:
        return float("nan")
    effect = treated["outcome"].mean() - control["outcome"].mean()
    expected_incremental_outcomes = effect * len(frame)
    expected_treatments = len(frame) / 2.0
    return float(expected_incremental_outcomes * outcome_value - expected_treatments * treatment_cost)
