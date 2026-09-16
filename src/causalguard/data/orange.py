from __future__ import annotations

import pandas as pd

from causalguard.data.openml import load_orange_belgium


def _binary01(series: pd.Series, name: str) -> pd.Series:
    values = list(pd.Series(series.dropna().unique()).sort_values())
    if set(values).issubset({0, 1, False, True}):
        return series.astype(int)
    string_values = {str(v).strip().lower(): v for v in values}
    if {"0", "1"}.issubset(string_values):
        return series.astype(str).astype(int)
    raise ValueError(f"Expected binary 0/1 column for {name}; found {values}")


def load_orange_retention_frame(data_id: int = 45580) -> tuple[pd.DataFrame, list[str]]:
    """Load Orange Belgium and standardize to CausalGuard conventions.

    Public benchmark columns:
      t: randomized retention treatment
      y: churn outcome (1 means churn)

    CausalGuard uses outcome=1 as desirable, so outcome is defined as 1-y
    (retention). Raw t/y columns are kept for auditability.
    """
    bunch = load_orange_belgium(data_id=data_id)
    frame = bunch.frame.copy()
    if "t" not in frame.columns or "y" not in frame.columns:
        raise ValueError("Orange benchmark must contain treatment 't' and churn outcome 'y'")
    frame["treatment"] = _binary01(frame["t"], "t")
    churn = _binary01(frame["y"], "y")
    frame["outcome"] = 1 - churn
    feature_names = [c for c in frame.columns if c not in {"t", "y", "treatment", "outcome"}]
    return frame, feature_names
