from __future__ import annotations

import numpy as np
import pandas as pd


def ips_incremental_policy_value_per_customer(
    randomized_frame: pd.DataFrame,
    outcome_value: float | None,
    treatment_cost: float,
    propensity: float = 0.5,
    value_column: str | None = None,
) -> float:
    """IPS estimate of incremental value of target policy vs never-treat.

    Required columns:
      outcome: observed binary outcome
      treatment: randomized treatment actually received
      policy_action: action recommended by the deployed policy

    This estimator is intentionally transparent rather than variance-optimal.
    A future research version can add doubly robust estimators and confidence
    intervals.
    """
    if len(randomized_frame) == 0:
        return float("nan")
    if not 0 < propensity < 1:
        raise ValueError("propensity must be in (0, 1)")

    y = randomized_frame["outcome"].to_numpy(dtype=float)
    a = randomized_frame["treatment"].to_numpy(dtype=int)
    pi = randomized_frame["policy_action"].to_numpy(dtype=int)
    if value_column is not None:
        value = randomized_frame[value_column].to_numpy(dtype=float)
    elif outcome_value is not None:
        value = np.full(len(randomized_frame), float(outcome_value))
    else:
        raise ValueError("Provide outcome_value or value_column")

    monetary_outcome = y * value
    match_policy = (a == pi).astype(float)
    prob_policy_action = np.where(pi == 1, propensity, 1.0 - propensity)
    policy_outcome_value = np.mean(match_policy * monetary_outcome / prob_policy_action)

    control_outcome_value = np.mean(
        (a == 0).astype(float) * monetary_outcome / (1.0 - propensity)
    )
    treatment_rate = float(np.mean(pi))

    return float(policy_outcome_value - control_outcome_value - treatment_rate * treatment_cost)
