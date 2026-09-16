from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import expit

DriftType = Literal[
    "none",
    "covariate",
    "outcome",
    "treatment",
    "gradual_treatment",
    "recurring_treatment",
]


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42
    n_features: int = 8
    drift_type: DriftType = "none"
    drift_start: int = 14
    drift_strength: float = 1.25
    drift_duration: int = 8


class CausalEnvironment:
    """Synthetic environment with known counterfactual probabilities.

    Outcome 1 is desirable (retained / converted). Treatment can change the
    probability of the outcome. Drift mechanisms alter P(X), baseline outcome,
    or treatment effect separately so monitoring strategies can be evaluated.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.beta = np.linspace(0.55, -0.35, config.n_features)
        self.tau_beta = np.zeros(config.n_features)
        self.tau_beta[: min(4, config.n_features)] = np.array([0.65, -0.45, 0.35, 0.25])[
            : min(4, config.n_features)
        ]

    def _drift_progress(self, step: int) -> float:
        if step < self.config.drift_start:
            return 0.0
        if self.config.drift_type == "gradual_treatment":
            return min(1.0, (step - self.config.drift_start + 1) / max(1, self.config.drift_duration))
        if self.config.drift_type == "recurring_treatment":
            period = max(2, self.config.drift_duration)
            phase = (step - self.config.drift_start) % period
            return 1.0 if phase < period / 2 else 0.0
        return 1.0

    def sample_features(self, n: int, step: int) -> np.ndarray:
        mean = np.zeros(self.config.n_features)
        if self.config.drift_type == "covariate" and step >= self.config.drift_start:
            mean[: min(3, self.config.n_features)] = self.config.drift_strength
        return self.rng.normal(loc=mean, scale=1.0, size=(n, self.config.n_features))

    def probabilities(self, x: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
        base_logit = -0.85 + x @ self.beta / np.sqrt(self.config.n_features)

        if self.config.drift_type == "outcome" and step >= self.config.drift_start:
            base_logit = base_logit - self.config.drift_strength

        heterogeneous_effect = 0.35 + x @ self.tau_beta / np.sqrt(max(1, self.config.n_features))
        progress = self._drift_progress(step)

        if self.config.drift_type in {"treatment", "gradual_treatment", "recurring_treatment"}:
            heterogeneous_effect = heterogeneous_effect - progress * self.config.drift_strength

        p0 = expit(base_logit)
        p1 = expit(base_logit + heterogeneous_effect)
        return p0, p1

    def generate_batch(self, n: int, step: int) -> pd.DataFrame:
        x = self.sample_features(n=n, step=step)
        p0, p1 = self.probabilities(x, step=step)
        frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(self.config.n_features)])
        frame["step"] = step
        value_signal = x[:, 0] + (0.35 * x[:, 2] if self.config.n_features > 2 else 0.0)
        frame["value_multiplier"] = 0.5 + expit(value_signal)
        frame["p0_true"] = p0
        frame["p1_true"] = p1
        frame["cate_true"] = p1 - p0
        return frame

    def realize_outcomes(self, frame: pd.DataFrame, treatment: np.ndarray) -> pd.DataFrame:
        treatment = np.asarray(treatment, dtype=int)
        if treatment.shape[0] != len(frame):
            raise ValueError("treatment length must match frame length")
        p = np.where(treatment == 1, frame["p1_true"].to_numpy(), frame["p0_true"].to_numpy())
        y = self.rng.binomial(1, p)
        out = frame.copy()
        out["treatment"] = treatment
        out["outcome"] = y
        out["observed_probability"] = p
        return out
