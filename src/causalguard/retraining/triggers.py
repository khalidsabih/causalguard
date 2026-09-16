from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TriggerState:
    step: int
    steps_since_retrain: int
    feature_drift: float
    brier: float
    cate_shift: float
    policy_value: float
    policy_value_upper: float = float("nan")


class RetrainingTrigger:
    name = "base"

    def should_retrain(self, state: TriggerState) -> bool:
        raise NotImplementedError


class NeverTrigger(RetrainingTrigger):
    name = "never"

    def should_retrain(self, state: TriggerState) -> bool:
        return False


class PeriodicTrigger(RetrainingTrigger):
    name = "periodic"

    def __init__(self, every: int):
        self.every = every

    def should_retrain(self, state: TriggerState) -> bool:
        return state.steps_since_retrain >= self.every


class FeatureDriftTrigger(RetrainingTrigger):
    name = "feature_drift"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_retrain(self, state: TriggerState) -> bool:
        return (
            state.feature_drift == state.feature_drift
            and state.feature_drift > self.threshold
        )


class PerformanceTrigger(RetrainingTrigger):
    name = "performance"

    def __init__(self, max_brier: float):
        self.max_brier = max_brier

    def should_retrain(self, state: TriggerState) -> bool:
        return (
            state.brier == state.brier
            and state.brier > self.max_brier
        )


class CateShiftTrigger(RetrainingTrigger):
    name = "cate_shift"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_retrain(self, state: TriggerState) -> bool:
        return (
            state.cate_shift == state.cate_shift
            and state.cate_shift > self.threshold
        )


class PolicyValueTrigger(RetrainingTrigger):
    name = "policy_value"

    def __init__(self, min_value: float):
        self.min_value = min_value

    def should_retrain(self, state: TriggerState) -> bool:
        return (
            state.policy_value == state.policy_value
            and state.policy_value < self.min_value
        )


class ConfidencePolicyValueTrigger(RetrainingTrigger):
    name = "policy_value_confident"

    def __init__(self, min_value: float):
        self.min_value = min_value

    def should_retrain(self, state: TriggerState) -> bool:
        return (
            state.policy_value_upper == state.policy_value_upper
            and state.policy_value_upper < self.min_value
        )


def build_trigger(
    name: str,
    config: dict,
) -> RetrainingTrigger:
    if name == "never":
        return NeverTrigger()

    if name == "periodic":
        return PeriodicTrigger(
            int(config.get("periodic_every", 5))
        )

    if name == "feature_drift":
        return FeatureDriftTrigger(
            float(
                config.get(
                    "feature_drift_threshold",
                    0.18,
                )
            )
        )

    if name == "performance":
        return PerformanceTrigger(
            float(
                config.get(
                    "performance_threshold",
                    0.26,
                )
            )
        )

    if name == "cate_shift":
        return CateShiftTrigger(
            float(
                config.get(
                    "cate_shift_threshold",
                    0.08,
                )
            )
        )

    if name == "policy_value":
        return PolicyValueTrigger(
            float(
                config.get(
                    "policy_value_threshold",
                    0.0,
                )
            )
        )

    if name == "policy_value_confident":
        return ConfidencePolicyValueTrigger(
            float(
                config.get(
                    "policy_value_threshold",
                    0.0,
                )
            )
        )

    raise ValueError(f"Unknown trigger: {name}")