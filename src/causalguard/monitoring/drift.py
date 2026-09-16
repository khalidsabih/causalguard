from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def feature_drift_score(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_names: list[str],
) -> float:
    """Mean two-sample KS statistic across features.

    This is deliberately simple and interpretable. The project can later add
    classifier-based drift, PSI, Wasserstein distance, and multiple-testing
    corrections as stronger baselines.
    """
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    stats = [
        ks_2samp(reference[name].to_numpy(), current[name].to_numpy()).statistic
        for name in feature_names
    ]
    return float(np.mean(stats))
