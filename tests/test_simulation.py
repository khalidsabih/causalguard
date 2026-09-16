import numpy as np

from causalguard.simulation import CausalEnvironment, SimulationConfig


def test_no_drift_is_reproducible():
    cfg = SimulationConfig(seed=7, n_features=4, drift_type="none")
    env1 = CausalEnvironment(cfg)
    env2 = CausalEnvironment(cfg)
    a = env1.generate_batch(50, step=2)
    b = env2.generate_batch(50, step=2)
    assert np.allclose(a[["x0", "x1", "x2", "x3"]], b[["x0", "x1", "x2", "x3"]])


def test_treatment_drift_changes_true_cate():
    cfg = SimulationConfig(seed=11, n_features=4, drift_type="treatment", drift_start=3, drift_strength=1.0)
    env = CausalEnvironment(cfg)
    x = np.zeros((500, 4))
    p0_before, p1_before = env.probabilities(x, step=2)
    p0_after, p1_after = env.probabilities(x, step=4)
    assert np.mean(p1_after - p0_after) < np.mean(p1_before - p0_before)
