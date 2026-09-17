import numpy as np
import pytest

from causalguard.pipelines.run_orange_crossfit import (
    foldwise_top_fraction,
    parse_budgets,
)


def test_foldwise_top_fraction_respects_budget_inside_each_fold():
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    fold_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    action = foldwise_top_fraction(score, fold_ids, fraction=0.50)

    assert action.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert action[fold_ids == 0].sum() == 2
    assert action[fold_ids == 1].sum() == 2


def test_parse_budgets_validates_range():
    assert parse_budgets("0.10,0.25,0.50") == [0.10, 0.25, 0.50]

    with pytest.raises(ValueError):
        parse_budgets("0.10,1.20")
