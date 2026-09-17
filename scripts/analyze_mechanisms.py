from pathlib import Path

import pandas as pd
import yaml


ANALYSIS = Path("analysis/main_trigger_benchmark")
CONFIG = Path("configs/experiments/trigger_benchmark_frozen.yaml")

config = yaml.safe_load(CONFIG.read_text())
RETRAINING_COST = float(config["retraining_cost"])

runs = pd.read_csv(
    ANALYSIS / "run_level_metrics.csv"
)

traj = pd.read_csv(
    ANALYSIS / "trajectory_summary.csv"
)


CASES = [
    ("covariate", "feature_drift"),
    ("treatment", "policy_value"),
    ("recurring_treatment", "policy_value"),
]


print()
print("=" * 95)
print("ECONOMIC MECHANISM DECOMPOSITION")
print("=" * 95)

decomposition_rows = []

for drift, trigger in CASES:
    baseline = runs[
        (runs["drift_type"] == drift)
        & (runs["trigger"] == "never")
    ].set_index("seed")

    candidate = runs[
        (runs["drift_type"] == drift)
        & (runs["trigger"] == trigger)
    ].set_index("seed")

    net_delta = (
        candidate["net_value"]
        - baseline["net_value"]
    )

    mean_net_delta = net_delta.mean()

    mean_retrainings = (
        candidate["retrainings"].mean()
    )

    direct_retraining_cost = (
        mean_retrainings
        * RETRAINING_COST
    )

    # never has zero retraining cost, so adding the
    # candidate's direct retraining charge back gives
    # the difference in policy/decision value before
    # retraining charges.
    decision_value_delta = (
        mean_net_delta
        + direct_retraining_cost
    )

    decomposition_rows.append(
        {
            "drift": drift,
            "trigger": trigger,
            "mean_net_delta_vs_never": mean_net_delta,
            "mean_retrainings": mean_retrainings,
            "direct_retraining_cost": direct_retraining_cost,
            "decision_value_delta_vs_never": decision_value_delta,
        }
    )

decomp = pd.DataFrame(decomposition_rows)

print(
    decomp
    .round(2)
    .to_string(index=False)
)

decomp.to_csv(
    ANALYSIS / "mechanism_decomposition.csv",
    index=False,
)


print()
print("=" * 95)
print("TEMPORAL MECHANISM AROUND DRIFT")
print("=" * 95)

all_cases = []

for drift, trigger in CASES:
    base = traj[
        (traj["benchmark_drift"] == drift)
        & (traj["benchmark_trigger"] == "never")
    ].copy()

    cand = traj[
        (traj["benchmark_drift"] == drift)
        & (traj["benchmark_trigger"] == trigger)
    ].copy()

    merged = cand.merge(
        base,
        on="step",
        suffixes=("_candidate", "_never"),
    )

    # Remove direct retraining cost from the step value
    # so we can distinguish retraining expense from
    # downstream policy-quality changes.
    merged["candidate_decision_value"] = (
        merged["mean_step_net_value_candidate"]
        + RETRAINING_COST
        * merged["retrain_rate_candidate"]
    )

    merged["never_decision_value"] = (
        merged["mean_step_net_value_never"]
    )

    merged["decision_value_delta"] = (
        merged["candidate_decision_value"]
        - merged["never_decision_value"]
    )

    merged["net_value_delta"] = (
        merged["mean_step_net_value_candidate"]
        - merged["mean_step_net_value_never"]
    )

    merged["treatment_rate_delta"] = (
        merged["mean_policy_treatment_rate_candidate"]
        - merged["mean_policy_treatment_rate_never"]
    )

    merged["predicted_uplift_delta"] = (
        merged["mean_predicted_uplift_candidate"]
        - merged["mean_predicted_uplift_never"]
    )

    out = pd.DataFrame(
        {
            "drift": drift,
            "trigger": trigger,
            "step": merged["step"],
            "request_rate": merged[
                "request_rate_candidate"
            ],
            "retrain_rate": merged[
                "retrain_rate_candidate"
            ],
            "true_cate_oracle": merged[
                "mean_true_cate_candidate"
            ],
            "candidate_treatment_rate": merged[
                "mean_policy_treatment_rate_candidate"
            ],
            "never_treatment_rate": merged[
                "mean_policy_treatment_rate_never"
            ],
            "treatment_rate_delta": merged[
                "treatment_rate_delta"
            ],
            "candidate_predicted_uplift": merged[
                "mean_predicted_uplift_candidate"
            ],
            "never_predicted_uplift": merged[
                "mean_predicted_uplift_never"
            ],
            "net_value_delta": merged[
                "net_value_delta"
            ],
            "decision_value_delta": merged[
                "decision_value_delta"
            ],
        }
    )

    all_cases.append(out)

    print()
    print(f"{drift}: {trigger} vs never")
    print("-" * 95)

    # Concentrate on the region around drift_start=14.
    print(
        out[
            (out["step"] >= 12)
            & (out["step"] <= 24)
        ]
        .round(3)
        .to_string(index=False)
    )

mechanisms = pd.concat(
    all_cases,
    ignore_index=True,
)

mechanisms.to_csv(
    ANALYSIS / "mechanism_trajectories.csv",
    index=False,
)

print()
print("=" * 95)
print("FILES WRITTEN")
print("=" * 95)
print(ANALYSIS / "mechanism_decomposition.csv")
print(ANALYSIS / "mechanism_trajectories.csv")
