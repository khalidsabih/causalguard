from causalguard.retraining import TriggerState, build_trigger


def test_confidence_trigger_does_not_fire_when_upper_bound_positive():
    trigger = build_trigger(
        "policy_value_confident",
        {"policy_value_threshold": 0.0},
    )

    state = TriggerState(
        step=13,
        steps_since_retrain=8,
        feature_drift=0.0,
        brier=0.2,
        cate_shift=0.0,
        policy_value=-0.269,
        policy_value_upper=20.817,
    )

    assert trigger.should_retrain(state) is False


def test_confidence_trigger_fires_when_upper_bound_negative():
    trigger = build_trigger(
        "policy_value_confident",
        {"policy_value_threshold": 0.0},
    )

    state = TriggerState(
        step=15,
        steps_since_retrain=10,
        feature_drift=0.0,
        brier=0.2,
        cate_shift=0.0,
        policy_value=-21.780,
        policy_value_upper=-2.053,
    )

    assert trigger.should_retrain(state) is True