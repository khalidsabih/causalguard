import numpy as np
import pandas as pd

from causalguard.monitoring import feature_drift_score, ips_incremental_policy_value_per_customer


def test_feature_drift_detects_large_shift():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x0": rng.normal(0, 1, 1000), "x1": rng.normal(0, 1, 1000)})
    same = pd.DataFrame({"x0": rng.normal(0, 1, 1000), "x1": rng.normal(0, 1, 1000)})
    shifted = pd.DataFrame({"x0": rng.normal(2, 1, 1000), "x1": rng.normal(2, 1, 1000)})
    assert feature_drift_score(ref, shifted, ["x0", "x1"]) > feature_drift_score(ref, same, ["x0", "x1"])


def test_ips_policy_value_runs():
    frame = pd.DataFrame(
        {
            "outcome": [1, 0, 1, 0, 1, 0, 1, 0],
            "treatment": [1, 0, 1, 0, 0, 1, 0, 1],
            "policy_action": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    value = ips_incremental_policy_value_per_customer(frame, outcome_value=100.0, treatment_cost=5.0)
    assert np.isfinite(value)
