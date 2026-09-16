import numpy as np

from causalguard.models import TLearner
from causalguard.simulation import CausalEnvironment, SimulationConfig


def test_tlearner_predicts_valid_probabilities():
    env = CausalEnvironment(SimulationConfig(seed=3, n_features=5))
    frame = env.generate_batch(1000, step=0)
    rng = np.random.default_rng(9)
    frame = env.realize_outcomes(frame, rng.binomial(1, 0.5, size=len(frame)))
    features = [f"x{i}" for i in range(5)]
    model = TLearner().fit(frame, features)
    pred = model.predict(frame.head(20))
    assert np.all((pred.p0 >= 0) & (pred.p0 <= 1))
    assert np.all((pred.p1 >= 0) & (pred.p1 <= 1))
    assert len(pred.uplift) == 20
