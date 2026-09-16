import numpy as np

from causalguard.policy import profit_score, top_fraction


def test_top_fraction_selects_highest_scores():
    score = np.array([0.1, 0.9, 0.8, 0.2])
    action = top_fraction(score, 0.5)
    assert action.tolist() == [0, 1, 1, 0]


def test_profit_score():
    uplift = np.array([0.0, 0.1])
    score = profit_score(uplift, outcome_value=100.0, treatment_cost=5.0)
    assert np.allclose(score, [-5.0, 5.0])
