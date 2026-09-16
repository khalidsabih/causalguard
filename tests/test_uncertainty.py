from causalguard.evaluation import bootstrap_mean_ci


def test_bootstrap_mean_ci_contains_mean_for_simple_sample():
    mean, low, high = bootstrap_mean_ci([1, 2, 3, 4, 5], n_bootstrap=500, seed=1)
    assert mean == 3.0
    assert low <= mean <= high
