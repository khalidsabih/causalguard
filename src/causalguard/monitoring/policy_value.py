from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicyValueEstimate:
    estimate: float
    standard_error: float
    lower: float
    upper: float
    upper_one_sided: float
    n: int


def ips_incremental_policy_value_with_ci(
    randomized_frame: pd.DataFrame,
    outcome_value: float | None,
    treatment_cost: float,
    propensity: float = 0.5,
    value_column: str | None = None,
    confidence_level: float = 0.95,
) -> PolicyValueEstimate:
    """Estimate incremental policy value with approximate uncertainty bounds.

    The two-sided confidence interval uses the empirical standard error
    of per-customer IPS contributions and a normal approximation.

    The one-sided upper confidence bound is intended for decisions such
    as testing whether policy value is confidently below a threshold.

    These bounds capture sampling uncertainty in the randomized monitoring
    sample. They should not be interpreted as accounting for every source
    of uncertainty under temporal distribution shift.
    """
    if len(randomized_frame) == 0:
        return PolicyValueEstimate(
            estimate=float("nan"),
            standard_error=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            upper_one_sided=float("nan"),
            n=0,
        )

    if not 0 < propensity < 1:
        raise ValueError(
            "propensity must be in (0, 1)"
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be in (0, 1)"
        )

    y = randomized_frame[
        "outcome"
    ].to_numpy(dtype=float)

    a = randomized_frame[
        "treatment"
    ].to_numpy(dtype=int)

    pi = randomized_frame[
        "policy_action"
    ].to_numpy(dtype=int)

    if value_column is not None:
        value = randomized_frame[
            value_column
        ].to_numpy(dtype=float)

    elif outcome_value is not None:
        value = np.full(
            len(randomized_frame),
            float(outcome_value),
        )

    else:
        raise ValueError(
            "Provide outcome_value or value_column"
        )

    monetary_outcome = y * value

    match_policy = (
        a == pi
    ).astype(float)

    prob_policy_action = np.where(
        pi == 1,
        propensity,
        1.0 - propensity,
    )

    policy_contribution = (
        match_policy
        * monetary_outcome
        / prob_policy_action
    )

    control_contribution = (
        (a == 0).astype(float)
        * monetary_outcome
        / (1.0 - propensity)
    )

    treatment_cost_contribution = (
        pi.astype(float)
        * treatment_cost
    )

    contribution = (
        policy_contribution
        - control_contribution
        - treatment_cost_contribution
    )

    estimate = float(
        np.mean(contribution)
    )

    n = len(contribution)

    if n < 2:
        return PolicyValueEstimate(
            estimate=estimate,
            standard_error=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            upper_one_sided=float("nan"),
            n=n,
        )

    standard_error = float(
        np.std(
            contribution,
            ddof=1,
        )
        / np.sqrt(n)
    )

    # Two-sided confidence interval.
    #
    # For confidence_level = 0.95 this gives approximately 1.96.
    two_sided_critical = NormalDist().inv_cdf(
        0.5
        + confidence_level / 2.0
    )

    lower = (
        estimate
        - two_sided_critical
        * standard_error
    )

    upper = (
        estimate
        + two_sided_critical
        * standard_error
    )

    # One-sided upper confidence bound.
    #
    # For confidence_level = 0.95 this gives approximately 1.645.
    # This is the quantity used by the confidence-aware retraining
    # trigger when asking:
    #
    # "Is policy value confidently below the threshold?"
    one_sided_critical = NormalDist().inv_cdf(
        confidence_level
    )

    upper_one_sided = (
        estimate
        + one_sided_critical
        * standard_error
    )

    return PolicyValueEstimate(
        estimate=estimate,
        standard_error=standard_error,
        lower=float(lower),
        upper=float(upper),
        upper_one_sided=float(
            upper_one_sided
        ),
        n=n,
    )


def ips_incremental_policy_value_per_customer(
    randomized_frame: pd.DataFrame,
    outcome_value: float | None,
    treatment_cost: float,
    propensity: float = 0.5,
    value_column: str | None = None,
) -> float:
    """IPS point estimate of incremental value vs never-treat.

    Kept for backwards compatibility with existing experiments.
    """
    result = (
        ips_incremental_policy_value_with_ci(
            randomized_frame=(
                randomized_frame
            ),
            outcome_value=outcome_value,
            treatment_cost=treatment_cost,
            propensity=propensity,
            value_column=value_column,
        )
    )

    return result.estimate