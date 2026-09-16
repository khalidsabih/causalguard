from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(values.mean()), float(low), float(high)
