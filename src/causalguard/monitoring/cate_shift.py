from __future__ import annotations

import numpy as np
import pandas as pd

from causalguard.models import TLearner


def cate_shift_score(
    randomized_recent: pd.DataFrame,
    anchor_features: pd.DataFrame,
    deployed_model: TLearner,
    feature_names: list[str],
) -> float:
    """Proxy score for change in conditional treatment effects.

    A temporary T-learner is fit on recent randomized observations and compared
    with the deployed model on a fixed anchor population. The score is mean
    absolute change in predicted uplift.

    This is an interpretable baseline, not a formal sequential CATE
    change-point detector. A publication version should benchmark a method from
    the change-detection literature alongside it.
    """
    if len(randomized_recent) < 100:
        return float("nan")
    try:
        recent_model = TLearner().fit(randomized_recent, feature_names)
    except ValueError:
        return float("nan")
    deployed = deployed_model.predict(anchor_features).uplift
    recent = recent_model.predict(anchor_features).uplift
    return float(np.mean(np.abs(recent - deployed)))
