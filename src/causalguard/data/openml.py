from __future__ import annotations

from sklearn.datasets import fetch_openml


def load_orange_belgium(data_id: int = 45580):
    """Fetch the public Orange Belgium churn-uplift benchmark from OpenML.

    Network access is required at runtime. The function intentionally returns
    the raw OpenML Bunch so the real-data adapter can be written after the
    dataset schema is inspected rather than hard-coding assumptions.
    """
    return fetch_openml(data_id=data_id, as_frame=True, parser="auto")
