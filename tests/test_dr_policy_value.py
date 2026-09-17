import numpy as np

from causalguard.evaluation import (
    dr_incremental_policy_value_with_ci,
    paired_dr_policy_difference_with_ci,
)


def test_dr_policy_value_recovers_exact_effect_with_exact_nuisance_models():
    outcome = np.array([0.8, 0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3])
    treatment = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    p0 = np.full(8, 0.3)
    p1 = np.full(8, 0.8)
    action = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    result = dr_incremental_policy_value_with_ci(
        outcome=outcome,
        treatment=treatment,
        policy_action=action,
        p0=p0,
        p1=p1,
        propensity=0.5,
    )

    assert np.isclose(result.estimate, 0.25)
    assert result.n == 8


def test_dr_policy_value_subtracts_policy_treatment_cost():
    outcome = np.array([0.8, 0.3, 0.8, 0.3])
    treatment = np.array([1, 0, 1, 0])
    p0 = np.full(4, 0.3)
    p1 = np.full(4, 0.8)
    action = np.ones(4, dtype=int)

    result = dr_incremental_policy_value_with_ci(
        outcome=outcome,
        treatment=treatment,
        policy_action=action,
        p0=p0,
        p1=p1,
        propensity=0.5,
        treatment_cost=0.1,
    )

    assert np.isclose(result.estimate, 0.4)


def test_paired_dr_difference_is_zero_for_identical_policies():
    outcome = np.array([1.0, 0.0, 1.0, 0.0])
    treatment = np.array([1, 0, 1, 0])
    p0 = np.full(4, 0.25)
    p1 = np.full(4, 0.75)
    action = np.array([1, 0, 1, 0])

    result = paired_dr_policy_difference_with_ci(
        outcome=outcome,
        treatment=treatment,
        first_action=action,
        second_action=action,
        p0=p0,
        p1=p1,
        propensity=0.5,
    )

    assert result.estimate == 0.0
    assert result.standard_error == 0.0
    assert result.lower == 0.0
    assert result.upper == 0.0


def test_dr_policy_value_supports_stochastic_policy_probability():
    outcome = np.array([0.8, 0.3, 0.8, 0.3])
    treatment = np.array([1, 0, 1, 0])
    p0 = np.full(4, 0.3)
    p1 = np.full(4, 0.8)
    policy_probability = np.full(4, 0.25)

    result = dr_incremental_policy_value_with_ci(
        outcome=outcome,
        treatment=treatment,
        policy_action=policy_probability,
        p0=p0,
        p1=p1,
        propensity=0.5,
    )

    assert np.isclose(result.estimate, 0.125)


def test_dr_policy_value_rejects_invalid_policy_probability():
    outcome = np.array([1.0, 0.0])
    treatment = np.array([1, 0])
    p0 = np.full(2, 0.25)
    p1 = np.full(2, 0.75)

    with np.testing.assert_raises(ValueError):
        dr_incremental_policy_value_with_ci(
            outcome=outcome,
            treatment=treatment,
            policy_action=np.array([1.2, 0.0]),
            p0=p0,
            p1=p1,
            propensity=0.5,
        )
