from __future__ import annotations

import numpy as np
import pandas as pd


def qini_like_gain(frame: pd.DataFrame, score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Simple uplift gain table for randomized data.

    This is an educational diagnostic, not a publication-grade Qini estimator.
    It reports cumulative treated-control outcome differences by score quantile.
    """
    work = frame[["treatment", "outcome"]].copy()
    work["score"] = np.asarray(score)
    work = work.sort_values("score", ascending=False).reset_index(drop=True)
    work["bin"] = pd.qcut(work.index + 1, q=bins, labels=False, duplicates="drop")

    rows = []
    for b in sorted(work["bin"].unique()):
        subset = work[work["bin"] <= b]
        treated = subset[subset["treatment"] == 1]["outcome"]
        control = subset[subset["treatment"] == 0]["outcome"]
        if len(treated) == 0 or len(control) == 0:
            gain = float("nan")
        else:
            gain = float((treated.mean() - control.mean()) * len(subset))
        rows.append({"fraction": len(subset) / len(work), "incremental_outcomes": gain})
    return pd.DataFrame(rows)
