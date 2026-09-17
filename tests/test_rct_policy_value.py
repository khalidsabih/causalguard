import numpy as np
import pandas as pd

from causalguard.evaluation import rct_incremental_policy_value_with_ci


def test_never_treat_has_zero_incremental_value():
    frame = pd.DataFrame(
        {
            "outcome": [1, 0, 1, 0],
            "treatment": [1, 1, 0, 0],
            "policy_action": [0, 0, 0, 0],
        }
    )

    result = rct_incremental_policy_value_with_ci(
        randomized_frame=frame,
        outcome_value=1.0,
        treatment_cost=0.0,
    )

    assert result.estimate == 0.0
    assert result.standard_error == 0.0
    assert result.selected_n == 0


def test_treat_all_matches_difference_in_means():
    frame = pd.DataFrame(
        {
            "outcome": [1, 1, 0, 0, 1, 0, 0, 0],
            "treatment": [1, 1, 1, 1, 0, 0, 0, 0],
            "policy_action": np.ones(8, dtype=int),
        }
    )

    result = rct_incremental_policy_value_with_ci(
        randomized_frame=frame,
        outcome_value=1.0,
        treatment_cost=0.0,
    )

    expected = frame.loc[frame["treatment"] == 1, "outcome"].mean() - frame.loc[
        frame["treatment"] == 0, "outcome"
    ].mean()

    assert np.isclose(result.estimate, expected)
    assert result.selected_n == 8
    assert result.treated_selected_n == 4
    assert result.control_selected_n == 4


def test_selected_subgroup_effect_is_scaled_by_policy_rate_and_cost():
    frame = pd.DataFrame(
        {
            "outcome": [1, 0, 0, 0, 1, 1, 0, 0],
            "treatment": [1, 0, 1, 0, 1, 0, 1, 0],
            "policy_action": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )

    result = rct_incremental_policy_value_with_ci(
        randomized_frame=frame,
        outcome_value=2.0,
        treatment_cost=0.25,
    )

    # Selected subgroup: treated outcomes [1, 0], control outcomes [0, 0].
    # Monetary subgroup effect = 2 * (0.5 - 0.0) = 1.0.
    # Policy rate = 0.5, so value = 0.5 * (1.0 - 0.25) = 0.375.
    assert np.isclose(result.estimate, 0.375)
