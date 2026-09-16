from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from causalguard.mlops import log_experiment_if_enabled
from causalguard.models import TLearner
from causalguard.monitoring import (
    brier_score,
    cate_shift_score,
    feature_drift_score,
    ips_incremental_policy_value_with_ci,
)
from causalguard.policy import profit_score, top_fraction, true_incremental_value
from causalguard.retraining import TriggerState, build_trigger
from causalguard.simulation import CausalEnvironment, SimulationConfig


def _coerce(value: str) -> Any:
    low = value.lower()

    if low in {"true", "false"}:
        return low == "true"

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(
                f"Override must look like key=value: {item}"
            )

        key, value = item.split("=", 1)
        config[key] = _coerce(value)

    return config


def _fit_model(
    training: pd.DataFrame,
    feature_names: list[str],
) -> TLearner:
    return TLearner().fit(
        training,
        feature_names,
    )


def run_experiment(
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    seed = int(config.get("seed", 42))
    n_steps = int(config.get("n_steps", 30))
    batch_size = int(config.get("batch_size", 800))
    n_features = int(config.get("n_features", 8))

    initial_train_steps = int(
        config.get("initial_train_steps", 5)
    )

    budget_fraction = float(
        config.get("budget_fraction", 0.25)
    )

    policy_name = str(
        config.get("policy", "profit")
    )

    exploration_rate = float(
        config.get("exploration_rate", 0.10)
    )

    outcome_value = float(
        config.get("outcome_value", 250.0)
    )

    treatment_cost = float(
        config.get("treatment_cost", 8.0)
    )

    retraining_cost = float(
        config.get("retraining_cost", 250.0)
    )

    monitor_window = int(
        config.get("monitor_window_steps", 3)
    )

    max_train_rows = int(
        config.get("max_train_rows", 12000)
    )

    retraining_data_strategy = str(
        config.get(
            "retraining_data_strategy",
            "full_history",
        )
    )

    retraining_window_steps = int(
        config.get(
            "retraining_window_steps",
            5,
        )
    )

    retraining_cooldown_steps = int(
        config.get(
            "retraining_cooldown_steps",
            0,
        )
    )

    policy_value_confidence_level = float(
        config.get(
            "policy_value_confidence_level",
            0.95,
        )
    )

    sim_config = SimulationConfig(
        seed=seed,
        n_features=n_features,
        drift_type=str(
            config.get("drift_type", "none")
        ),
        drift_start=int(
            config.get("drift_start", 14)
        ),
        drift_strength=float(
            config.get("drift_strength", 1.25)
        ),
        drift_duration=int(
            config.get("drift_duration", 8)
        ),
    )

    env = CausalEnvironment(sim_config)

    assignment_rng = np.random.default_rng(
        seed + 101
    )

    feature_names = [
        f"x{i}" for i in range(n_features)
    ]

    # ---------------------------------------------------------
    # Initial randomized causal training experiment
    # ---------------------------------------------------------

    initial_frames: list[pd.DataFrame] = []

    for step in range(initial_train_steps):
        raw = env.generate_batch(
            batch_size,
            step,
        )

        treatment = assignment_rng.binomial(
            1,
            0.5,
            size=batch_size,
        )

        observed = env.realize_outcomes(
            raw,
            treatment,
        )

        observed["exploration"] = True
        observed["policy_action"] = treatment

        initial_frames.append(observed)

    # ---------------------------------------------------------
    # Two deliberately separate histories
    #
    # randomized_history:
    #   data that can be used to train/retrain the causal model.
    #   Includes the initial randomized experiment and subsequent
    #   production exploration observations.
    #
    # monitoring_history:
    #   production exploration observations only.
    #   Used for online monitoring so the initial RCT does not
    #   contaminate the rolling production monitoring window.
    # ---------------------------------------------------------

    randomized_history = [
        frame.copy()
        for frame in initial_frames
    ]

    monitoring_history: list[pd.DataFrame] = []

    training = pd.concat(
        randomized_history,
        ignore_index=True,
    )

    model = _fit_model(
        training,
        feature_names,
    )

    reference_features = training[
        feature_names
    ].copy()

    anchor_features = reference_features.sample(
        n=min(
            1000,
            len(reference_features),
        ),
        random_state=seed,
    ).copy()

    trigger = build_trigger(
        str(
            config.get(
                "trigger",
                "never",
            )
        ),
        config,
    )

    # ---------------------------------------------------------
    # Production state
    # ---------------------------------------------------------

    steps_since_retrain = 0
    last_retrain_step: int | None = None
    cumulative_net_value = 0.0

    rows: list[dict] = []

    # ---------------------------------------------------------
    # Production simulation
    # ---------------------------------------------------------

    for step in range(
        initial_train_steps,
        n_steps,
    ):
        raw = env.generate_batch(
            batch_size,
            step,
        )

        predictions = model.predict(raw)

        customer_value = (
            outcome_value
            * raw["value_multiplier"].to_numpy()
        )

        raw["customer_value"] = customer_value

        # -----------------------------------------------------
        # Policy decision
        # -----------------------------------------------------

        if policy_name == "profit":
            score = profit_score(
                predictions.uplift,
                customer_value,
                treatment_cost,
            )

            policy_action = top_fraction(
                score,
                budget_fraction,
            )

            policy_action = np.where(
                score > 0,
                policy_action,
                0,
            ).astype(int)

        elif policy_name == "uplift":
            score = predictions.uplift

            policy_action = top_fraction(
                score,
                budget_fraction,
            )

            policy_action = np.where(
                score > 0,
                policy_action,
                0,
            ).astype(int)

        elif policy_name == "risk":
            score = (
                1.0
                - predictions.p0
            )

            policy_action = top_fraction(
                score,
                budget_fraction,
            )

        elif policy_name == "random":
            score = assignment_rng.random(
                batch_size
            )

            policy_action = top_fraction(
                score,
                budget_fraction,
            )

        else:
            raise ValueError(
                f"Unknown policy: {policy_name}"
            )

        # -----------------------------------------------------
        # Persistent randomized exploration
        # -----------------------------------------------------

        exploration = (
            assignment_rng.random(
                batch_size
            )
            < exploration_rate
        )

        treatment = policy_action.copy()

        treatment[exploration] = (
            assignment_rng.binomial(
                1,
                0.5,
                size=int(
                    exploration.sum()
                ),
            )
        )

        # -----------------------------------------------------
        # Realize outcomes
        # -----------------------------------------------------

        observed = env.realize_outcomes(
            raw,
            treatment,
        )

        observed["exploration"] = exploration
        observed["policy_action"] = policy_action
        observed["p0_hat"] = predictions.p0
        observed["p1_hat"] = predictions.p1
        observed["uplift_hat"] = predictions.uplift

        # -----------------------------------------------------
        # Predictive monitoring
        # -----------------------------------------------------

        observed_probability = np.where(
            treatment == 1,
            predictions.p1,
            predictions.p0,
        )

        current_brier = brier_score(
            observed[
                "outcome"
            ].to_numpy(),
            observed_probability,
        )

        # -----------------------------------------------------
        # Feature drift
        # -----------------------------------------------------

        current_feature_drift = (
            feature_drift_score(
                reference_features,
                raw,
                feature_names,
            )
        )

        # -----------------------------------------------------
        # Production randomized exploration sample
        # -----------------------------------------------------

        randomized_batch = observed[
            observed["exploration"]
        ].copy()

        # Used for future causal retraining.
        randomized_history.append(
            randomized_batch
        )

        # Used only for production monitoring.
        monitoring_history.append(
            randomized_batch
        )

        recent_randomized = pd.concat(
            monitoring_history[
                -monitor_window:
            ],
            ignore_index=True,
        )

        # -----------------------------------------------------
        # CATE-shift monitoring
        # -----------------------------------------------------

        current_cate_shift = (
            cate_shift_score(
                recent_randomized,
                anchor_features=(
                    anchor_features
                ),
                deployed_model=model,
                feature_names=(
                    feature_names
                ),
            )
        )

        # -----------------------------------------------------
        # Policy-value monitoring + uncertainty
        # -----------------------------------------------------

        policy_value_result = (
            ips_incremental_policy_value_with_ci(
                recent_randomized,
                outcome_value=None,
                treatment_cost=(
                    treatment_cost
                ),
                propensity=0.5,
                value_column=(
                    "customer_value"
                ),
                confidence_level=(
                    policy_value_confidence_level
                ),
            )
        )

        current_policy_value = (
            policy_value_result.estimate
        )

        # -----------------------------------------------------
        # Simulator-only oracle value
        #
        # This is unavailable in real production and must not
        # be used by deployable monitoring logic.
        # -----------------------------------------------------

        oracle_value = true_incremental_value(
            observed,
            treatment=treatment,
            outcome_value=observed[
                "customer_value"
            ].to_numpy(),
            treatment_cost=treatment_cost,
        )

        # -----------------------------------------------------
        # Trigger state
        # -----------------------------------------------------

        steps_since_retrain += 1

        state = TriggerState(
            step=step,
            steps_since_retrain=(
                steps_since_retrain
            ),
            feature_drift=(
                current_feature_drift
            ),
            brier=current_brier,
            cate_shift=current_cate_shift,
            policy_value=(
                current_policy_value
            ),
            policy_value_upper=(
                policy_value_result.upper_one_sided
            ),
        )

        should_retrain = (
            trigger.should_retrain(state)
        )

        # -----------------------------------------------------
        # Optional fixed cooldown
        # -----------------------------------------------------

        cooldown_complete = (
            last_retrain_step is None
            or (
                step
                - last_retrain_step
                > retraining_cooldown_steps
            )
        )

        retraining_blocked_by_cooldown = bool(
            should_retrain
            and not cooldown_complete
        )

        retrained = bool(
            should_retrain
            and cooldown_complete
        )

        # -----------------------------------------------------
        # Economic accounting
        # -----------------------------------------------------

        step_net_value = oracle_value

        # -----------------------------------------------------
        # Retraining
        # -----------------------------------------------------

        if retrained:
            step_net_value -= (
                retraining_cost
            )

            if (
                retraining_data_strategy
                == "full_history"
            ):
                causal_training = pd.concat(
                    randomized_history,
                    ignore_index=True,
                ).tail(
                    max_train_rows
                )

            elif (
                retraining_data_strategy
                == "rolling_window"
            ):
                causal_training = pd.concat(
                    randomized_history[
                        -retraining_window_steps:
                    ],
                    ignore_index=True,
                )

            else:
                raise ValueError(
                    "Unknown "
                    "retraining_data_strategy: "
                    f"{retraining_data_strategy}"
                )

            try:
                model = _fit_model(
                    causal_training,
                    feature_names,
                )

                reference_features = (
                    causal_training[
                        feature_names
                    ].copy()
                )

                steps_since_retrain = 0
                last_retrain_step = step

            except ValueError:
                # Keep the old model when the randomized
                # retraining sample is temporarily degenerate.
                retrained = False

                # Undo the retraining charge because fitting
                # did not successfully occur.
                step_net_value += (
                    retraining_cost
                )

        cumulative_net_value += (
            step_net_value
        )

        # -----------------------------------------------------
        # Diagnostics
        # -----------------------------------------------------

        rows.append(
            {
                "step": step,
                "drift_type": (
                    sim_config.drift_type
                ),
                "trigger": trigger.name,
                "policy": policy_name,
                "feature_drift": (
                    current_feature_drift
                ),
                "brier": current_brier,
                "cate_shift": (
                    current_cate_shift
                ),
                "policy_value_estimate_per_customer": (
                    current_policy_value
                ),
                "policy_value_standard_error": (
                    policy_value_result.standard_error
                ),
                "policy_value_lower": (
                    policy_value_result.lower
                ),
                "policy_value_upper": (
                    policy_value_result.upper
                ),
                "policy_value_upper_one_sided": (
                    policy_value_result.upper_one_sided
                ),
                "policy_value_monitor_n": (
                    policy_value_result.n
                ),
                "oracle_incremental_value": (
                    oracle_value
                ),
                "oracle_value_per_customer": (
                    oracle_value
                    / batch_size
                ),
                "true_mean_cate": (
                    observed[
                        "cate_true"
                    ].mean()
                ),
                "predicted_mean_uplift": (
                    predictions.uplift.mean()
                ),
                "policy_treatment_rate": (
                    policy_action.mean()
                ),
                "actual_treatment_rate": (
                    treatment.mean()
                ),
                "exploration_rate_realized": (
                    exploration.mean()
                ),
                "retraining_requested": bool(
                    should_retrain
                ),
                "retrained": bool(
                    retrained
                ),
                "retraining_blocked_by_cooldown": (
                    retraining_blocked_by_cooldown
                ),
                "step_net_value": (
                    step_net_value
                ),
                "cumulative_net_value": (
                    cumulative_net_value
                ),
            }
        )

    # ---------------------------------------------------------
    # Experiment outputs
    # ---------------------------------------------------------

    results = pd.DataFrame(rows)

    summary = {
        "drift_type": (
            sim_config.drift_type
        ),
        "trigger": trigger.name,
        "policy": policy_name,
        "seed": seed,
        "steps_evaluated": (
            len(results)
        ),
        "retraining_data_strategy": (
            retraining_data_strategy
        ),
        "retraining_window_steps": (
            retraining_window_steps
        ),
        "retraining_cooldown_steps": (
            retraining_cooldown_steps
        ),
        "monitor_window_steps": (
            monitor_window
        ),
        "policy_value_confidence_level": (
            policy_value_confidence_level
        ),
        "retrainings": (
            int(
                results[
                    "retrained"
                ].sum()
            )
            if len(results)
            else 0
        ),
        "retraining_requests": (
            int(
                results[
                    "retraining_requested"
                ].sum()
            )
            if len(results)
            else 0
        ),
        "final_cumulative_net_value": (
            float(
                results[
                    "cumulative_net_value"
                ].iloc[-1]
            )
            if len(results)
            else 0.0
        ),
        "mean_oracle_value_per_customer": (
            float(
                results[
                    "oracle_value_per_customer"
                ].mean()
            )
            if len(results)
            else 0.0
        ),
        "mean_feature_drift": (
            float(
                results[
                    "feature_drift"
                ].mean()
            )
            if len(results)
            else 0.0
        ),
        "mean_brier": (
            float(
                results[
                    "brier"
                ].mean()
            )
            if len(results)
            else 0.0
        ),
        "mean_cate_shift": (
            float(
                results[
                    "cate_shift"
                ].mean()
            )
            if len(results)
            else 0.0
        ),
    }

    return results, summary


def save_results(
    results: pd.DataFrame,
    summary: dict,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output / "results.csv",
        index=False,
    )

    with open(
        output / "summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CausalGuard simulation experiment"
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/experiments/base.yaml"
        ),
    )

    parser.add_argument(
        "--output",
        default="experiments/latest",
    )

    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override key=value",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = load_config(
        args.config,
        args.set,
    )

    results, summary = run_experiment(
        config
    )

    save_results(
        results,
        summary,
        args.output,
    )

    log_experiment_if_enabled(
        config,
        summary,
        args.output,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()