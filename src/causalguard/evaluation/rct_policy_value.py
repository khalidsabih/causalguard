from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RCTPolicyValueEstimate:
    estimate: float
    standard_error: float
    lower: float
    upper: float
    n: int
    selected_n: int
    treated_selected_n: int
    control_selected_n: int


def rct_incremental_policy_value_with_ci(
    randomized_frame: pd.DataFrame,
    outcome_value: float | None,
    treatment_cost: float,
    value_column: str | None = None,
    confidence_level: float = 0.95,
) -> RCTPolicyValueEstimate:
    """Estimate policy value vs never-treat from a randomized evaluation sample.

    The deployed policy is represented by ``policy_action``. For customers the
    policy would treat, randomized treatment assignment identifies the average
    treatment effect in that selected subgroup. Multiplying that subgroup effect
    by the policy treatment rate yields incremental outcome value per customer in
    the full evaluation population.

    This estimator is intended for held-out randomized data where policy actions
    depend only on pre-treatment information and models fit outside the evaluation
    observations. It is a transparent difference-in-means baseline and is usually
    much less noisy than unnormalized Horvitz-Thompson IPS when outcomes have a
    large nonzero baseline level.
    """
    if len(randomized_frame) == 0:
        return RCTPolicyValueEstimate(
            estimate=float("nan"),
            standard_error=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            n=0,
            selected_n=0,
            treated_selected_n=0,
            control_selected_n=0,
        )

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    y = randomized_frame["outcome"].to_numpy(dtype=float)
    treatment = randomized_frame["treatment"].to_numpy(dtype=int)
    policy = randomized_frame["policy_action"].to_numpy(dtype=int)

    if value_column is not None:
        value = randomized_frame[value_column].to_numpy(dtype=float)
    elif outcome_value is not None:
        value = np.full(len(randomized_frame), float(outcome_value))
    else:
        raise ValueError("Provide outcome_value or value_column")

    selected = policy == 1
    selected_n = int(np.sum(selected))
    n = len(randomized_frame)

    if selected_n == 0:
        return RCTPolicyValueEstimate(
            estimate=0.0,
            standard_error=0.0,
            lower=0.0,
            upper=0.0,
            n=n,
            selected_n=0,
            treated_selected_n=0,
            control_selected_n=0,
        )

    treated_selected = selected & (treatment == 1)
    control_selected = selected & (treatment == 0)
    treated_selected_n = int(np.sum(treated_selected))
    control_selected_n = int(np.sum(control_selected))

    if treated_selected_n == 0 or control_selected_n == 0:
        return RCTPolicyValueEstimate(
            estimate=float("nan"),
            standard_error=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            n=n,
            selected_n=selected_n,
            treated_selected_n=treated_selected_n,
            control_selected_n=control_selected_n,
        )

    monetary_outcome = y * value
    treated_values = monetary_outcome[treated_selected]
    control_values = monetary_outcome[control_selected]

    subgroup_effect = float(treated_values.mean() - control_values.mean())
    treatment_rate = selected_n / n

    estimate = treatment_rate * (subgroup_effect - float(treatment_cost))

    if treated_selected_n < 2 or control_selected_n < 2:
        standard_error = float("nan")
    else:
        subgroup_variance = (
            np.var(treated_values, ddof=1) / treated_selected_n
            + np.var(control_values, ddof=1) / control_selected_n
        )
        standard_error = float(treatment_rate * np.sqrt(subgroup_variance))

    if np.isfinite(standard_error):
        critical = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        lower = estimate - critical * standard_error
        upper = estimate + critical * standard_error
    else:
        lower = float("nan")
        upper = float("nan")

    return RCTPolicyValueEstimate(
        estimate=float(estimate),
        standard_error=standard_error,
        lower=float(lower),
        upper=float(upper),
        n=n,
        selected_n=selected_n,
        treated_selected_n=treated_selected_n,
        control_selected_n=control_selected_n,
    )
