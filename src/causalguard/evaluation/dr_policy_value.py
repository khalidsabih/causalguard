from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class DRPolicyValueEstimate:
    estimate: float
    standard_error: float
    lower: float
    upper: float
    n: int


def _validate_inputs(
    outcome,
    treatment,
    policy_action,
    p0,
    p1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = [
        np.asarray(outcome, dtype=float),
        np.asarray(treatment, dtype=int),
        np.asarray(policy_action, dtype=float),
        np.asarray(p0, dtype=float),
        np.asarray(p1, dtype=float),
    ]
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("All inputs must have the same length")
    if not set(np.unique(arrays[1])).issubset({0, 1}):
        raise ValueError("treatment must be binary 0/1")
    if not np.all(np.isfinite(arrays[2])):
        raise ValueError("policy_action must be finite")
    if np.any((arrays[2] < 0) | (arrays[2] > 1)):
        raise ValueError("policy_action must lie in [0, 1]")
    return tuple(arrays)  # type: ignore[return-value]


def dr_incremental_policy_contributions(
    *,
    outcome,
    treatment,
    policy_action,
    p0,
    p1,
    propensity: float,
    outcome_value: float = 1.0,
    treatment_cost: float = 0.0,
) -> np.ndarray:
    """Cross-fitted AIPW contributions for policy value vs never-treat.

    ``policy_action`` may be binary for a deterministic policy or a probability
    in ``[0, 1]`` for a stochastic policy. ``p0`` and ``p1`` must be predictions
    produced without using the corresponding observation's realized outcome.
    With randomized treatment, the propensity is known by design or can be
    represented by the randomized assignment rate when only the marginal rate
    is available.
    """
    if not 0 < propensity < 1:
        raise ValueError("propensity must be in (0, 1)")

    y, t, action, mu0, mu1 = _validate_inputs(
        outcome,
        treatment,
        policy_action,
        p0,
        p1,
    )
    if len(y) == 0:
        return np.asarray([], dtype=float)

    aipw_tau = (
        (mu1 - mu0)
        + t / propensity * (y - mu1)
        - (1 - t) / (1 - propensity) * (y - mu0)
    )
    return action * (
        float(outcome_value) * aipw_tau - float(treatment_cost)
    )


def _mean_with_normal_ci(
    contributions: np.ndarray,
    confidence_level: float,
) -> DRPolicyValueEstimate:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    n = len(contributions)
    if n == 0:
        return DRPolicyValueEstimate(
            estimate=float("nan"),
            standard_error=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            n=0,
        )

    estimate = float(np.mean(contributions))
    if n < 2:
        standard_error = float("nan")
    else:
        standard_error = float(np.std(contributions, ddof=1) / np.sqrt(n))

    if np.isfinite(standard_error):
        critical = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        lower = estimate - critical * standard_error
        upper = estimate + critical * standard_error
    else:
        lower = float("nan")
        upper = float("nan")

    return DRPolicyValueEstimate(
        estimate=estimate,
        standard_error=standard_error,
        lower=float(lower),
        upper=float(upper),
        n=n,
    )


def dr_incremental_policy_value_with_ci(
    *,
    outcome,
    treatment,
    policy_action,
    p0,
    p1,
    propensity: float,
    outcome_value: float = 1.0,
    treatment_cost: float = 0.0,
    confidence_level: float = 0.95,
) -> DRPolicyValueEstimate:
    contributions = dr_incremental_policy_contributions(
        outcome=outcome,
        treatment=treatment,
        policy_action=policy_action,
        p0=p0,
        p1=p1,
        propensity=propensity,
        outcome_value=outcome_value,
        treatment_cost=treatment_cost,
    )
    return _mean_with_normal_ci(contributions, confidence_level)


def paired_dr_policy_difference_with_ci(
    *,
    outcome,
    treatment,
    first_action,
    second_action,
    p0,
    p1,
    propensity: float,
    outcome_value: float = 1.0,
    treatment_cost: float = 0.0,
    confidence_level: float = 0.95,
) -> DRPolicyValueEstimate:
    first = dr_incremental_policy_contributions(
        outcome=outcome,
        treatment=treatment,
        policy_action=first_action,
        p0=p0,
        p1=p1,
        propensity=propensity,
        outcome_value=outcome_value,
        treatment_cost=treatment_cost,
    )
    second = dr_incremental_policy_contributions(
        outcome=outcome,
        treatment=treatment,
        policy_action=second_action,
        p0=p0,
        p1=p1,
        propensity=propensity,
        outcome_value=outcome_value,
        treatment_cost=treatment_cost,
    )
    return _mean_with_normal_ci(first - second, confidence_level)
