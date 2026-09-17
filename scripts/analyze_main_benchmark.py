from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ============================================================
# Frozen benchmark definition
# ============================================================

ROOT = Path("experiments/main_trigger_benchmark")
CONFIG = Path("configs/experiments/trigger_benchmark_frozen.yaml")
OUTPUT = Path("analysis/main_trigger_benchmark")

DRIFTS = [
    "none",
    "covariate",
    "outcome",
    "treatment",
    "gradual_treatment",
    "recurring_treatment",
]

TRIGGERS = [
    "never",
    "periodic",
    "feature_drift",
    "performance",
    "cate_shift",
    "policy_value",
    "policy_value_confident",
]

PRIMARY_TRIGGERS = [
    "never",
    "periodic",
    "feature_drift",
    "performance",
    "cate_shift",
    "policy_value",
]

SEEDS = list(range(1000, 1030))

N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 42


# ============================================================
# Load frozen configuration
# ============================================================

config = yaml.safe_load(
    CONFIG.read_text()
)

DRIFT_START = int(
    config.get("drift_start", 14)
)

RETRAINING_COST = float(
    config.get("retraining_cost", 250.0)
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helpers
# ============================================================


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n == 0:
        return np.nan, np.nan

    if n == 1:
        return values[0], values[0]

    indices = rng.integers(
        0,
        n,
        size=(N_BOOTSTRAP, n),
    )

    samples = values[
        indices
    ].mean(axis=1)

    alpha = (
        1.0 - confidence
    )

    low, high = np.quantile(
        samples,
        [
            alpha / 2.0,
            1.0 - alpha / 2.0,
        ],
    )

    return (
        float(low),
        float(high),
    )


def paired_bootstrap(
    left: pd.Series,
    right: pd.Series,
    rng: np.random.Generator,
) -> dict:
    paired = pd.concat(
        [
            left.rename("left"),
            right.rename("right"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    diff = (
        paired["left"]
        - paired["right"]
    ).to_numpy(dtype=float)

    n = len(diff)

    if n == 0:
        return {
            "n": 0,
            "mean_diff": np.nan,
            "median_diff": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }

    indices = rng.integers(
        0,
        n,
        size=(N_BOOTSTRAP, n),
    )

    boot = diff[
        indices
    ].mean(axis=1)

    low, high = np.quantile(
        boot,
        [0.025, 0.975],
    )

    return {
        "n": n,
        "mean_diff": float(
            diff.mean()
        ),
        "median_diff": float(
            np.median(diff)
        ),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "wins": int(
            (diff > 0).sum()
        ),
        "ties": int(
            (diff == 0).sum()
        ),
        "losses": int(
            (diff < 0).sum()
        ),
    }


# ============================================================
# Completeness validation
# ============================================================

expected_runs = (
    len(DRIFTS)
    * len(TRIGGERS)
    * len(SEEDS)
)

missing: list[str] = []
run_rows: list[dict] = []
trajectory_frames: list[pd.DataFrame] = []

for drift in DRIFTS:
    for trigger in TRIGGERS:
        for seed in SEEDS:
            folder = (
                ROOT
                / drift
                / trigger
                / f"seed_{seed}"
            )

            summary_path = (
                folder
                / "summary.json"
            )

            results_path = (
                folder
                / "results.csv"
            )

            if (
                not summary_path.exists()
                or not results_path.exists()
            ):
                missing.append(
                    str(folder)
                )
                continue

            summary = json.loads(
                summary_path.read_text()
            )

            results = pd.read_csv(
                results_path
            )

            # --------------------------------------------
            # Basic integrity checks
            # --------------------------------------------

            if int(summary["seed"]) != seed:
                raise ValueError(
                    f"Seed mismatch in {folder}"
                )

            if (
                summary["drift_type"]
                != drift
            ):
                raise ValueError(
                    f"Drift mismatch in {folder}"
                )

            if (
                summary["trigger"]
                != trigger
            ):
                raise ValueError(
                    f"Trigger mismatch in {folder}"
                )

            if len(results) != 25:
                raise ValueError(
                    f"Expected 25 production steps in "
                    f"{folder}, got {len(results)}"
                )

            if (
                "retraining_requested"
                not in results.columns
            ):
                raise ValueError(
                    f"Missing retraining_requested "
                    f"in {results_path}"
                )

            # --------------------------------------------
            # Run-level operational diagnostics
            # --------------------------------------------

            request_mask = (
                results[
                    "retraining_requested"
                ]
                .astype(bool)
            )

            retrain_mask = (
                results[
                    "retrained"
                ]
                .astype(bool)
            )

            if drift == "none":
                pre_drift_requests = 0
                post_drift_requests = 0
                first_post_request_step = np.nan
                first_response_delay = np.nan
                post_drift_response = False

            else:
                pre_mask = (
                    results["step"]
                    < DRIFT_START
                )

                post_mask = (
                    results["step"]
                    >= DRIFT_START
                )

                pre_drift_requests = int(
                    (
                        request_mask
                        & pre_mask
                    ).sum()
                )

                post_drift_requests = int(
                    (
                        request_mask
                        & post_mask
                    ).sum()
                )

                post_request_steps = (
                    results.loc[
                        request_mask
                        & post_mask,
                        "step",
                    ]
                    .astype(int)
                    .tolist()
                )

                post_drift_response = (
                    len(
                        post_request_steps
                    )
                    > 0
                )

                if post_drift_response:
                    first_post_request_step = int(
                        post_request_steps[0]
                    )

                    first_response_delay = (
                        first_post_request_step
                        - DRIFT_START
                    )

                else:
                    first_post_request_step = np.nan
                    first_response_delay = np.nan

            retrainings = int(
                retrain_mask.sum()
            )

            requests = int(
                request_mask.sum()
            )

            net_value = float(
                summary[
                    "final_cumulative_net_value"
                ]
            )

            # Remove ONLY direct retraining charges.
            # Exploration and treatment economics remain.
            value_before_retraining_cost = (
                net_value
                + RETRAINING_COST
                * retrainings
            )

            run_rows.append(
                {
                    "drift_type": drift,
                    "trigger": trigger,
                    "primary_strategy": (
                        trigger
                        in PRIMARY_TRIGGERS
                    ),
                    "seed": seed,
                    "net_value": net_value,
                    "value_before_retraining_cost": (
                        value_before_retraining_cost
                    ),
                    "retrainings": retrainings,
                    "requests": requests,
                    "any_request": (
                        requests > 0
                    ),
                    "pre_drift_requests": (
                        pre_drift_requests
                    ),
                    "pre_drift_false_alarm": (
                        pre_drift_requests > 0
                    ),
                    "post_drift_requests": (
                        post_drift_requests
                    ),
                    "post_drift_response": (
                        post_drift_response
                    ),
                    "first_post_request_step": (
                        first_post_request_step
                    ),
                    "first_response_delay": (
                        first_response_delay
                    ),
                    "mean_oracle_value_per_customer": float(
                        summary[
                            "mean_oracle_value_per_customer"
                        ]
                    ),
                }
            )

            trajectory = (
                results.copy()
            )

            trajectory[
                "seed"
            ] = seed

            trajectory[
                "benchmark_drift"
            ] = drift

            trajectory[
                "benchmark_trigger"
            ] = trigger

            trajectory_frames.append(
                trajectory
            )


if missing:
    print(
        "\nMISSING RUNS"
    )
    print("=" * 80)

    for item in missing:
        print(item)

    raise SystemExit(
        f"\nFound {len(missing)} missing "
        f"runs out of {expected_runs}."
    )


runs = pd.DataFrame(
    run_rows
)

trajectories = pd.concat(
    trajectory_frames,
    ignore_index=True,
)

print()
print("=" * 80)
print("BENCHMARK COMPLETENESS")
print("=" * 80)
print(
    f"Expected runs: {expected_runs}"
)
print(
    f"Loaded runs:   {len(runs)}"
)

if len(runs) != expected_runs:
    raise ValueError(
        "Run count mismatch."
    )


# ============================================================
# Save run-level data
# ============================================================

runs.to_csv(
    OUTPUT
    / "run_level_metrics.csv",
    index=False,
)


# ============================================================
# Aggregate economic / operational summary
# ============================================================

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

summary_rows: list[dict] = []

for (
    drift,
    trigger,
), group in runs.groupby(
    [
        "drift_type",
        "trigger",
    ],
    sort=False,
):
    values = group[
        "net_value"
    ].to_numpy()

    low, high = (
        bootstrap_mean_ci(
            values,
            rng,
        )
    )

    summary_rows.append(
        {
            "drift_type": drift,
            "trigger": trigger,
            "n": len(group),
            "mean_net_value": float(
                group[
                    "net_value"
                ].mean()
            ),
            "median_net_value": float(
                group[
                    "net_value"
                ].median()
            ),
            "std_net_value": float(
                group[
                    "net_value"
                ].std(ddof=1)
            ),
            "mean_net_ci95_low": low,
            "mean_net_ci95_high": high,
            "mean_value_before_retraining_cost": float(
                group[
                    "value_before_retraining_cost"
                ].mean()
            ),
            "mean_retrainings": float(
                group[
                    "retrainings"
                ].mean()
            ),
            "mean_requests": float(
                group[
                    "requests"
                ].mean()
            ),
            "request_run_rate": float(
                group[
                    "any_request"
                ].mean()
            ),
            "mean_oracle_value_per_customer": float(
                group[
                    "mean_oracle_value_per_customer"
                ].mean()
            ),
        }
    )

group_summary = pd.DataFrame(
    summary_rows
)

group_summary.to_csv(
    OUTPUT
    / "group_summary.csv",
    index=False,
)


# ============================================================
# No-drift false-alarm analysis
# ============================================================

stationary = runs[
    runs[
        "drift_type"
    ]
    == "none"
].copy()

stationary_rows: list[dict] = []

never_stationary = (
    stationary[
        stationary["trigger"]
        == "never"
    ]
    .set_index("seed")
    ["net_value"]
)

for trigger in TRIGGERS:
    group = stationary[
        stationary["trigger"]
        == trigger
    ].copy()

    row = {
        "trigger": trigger,
        "n": len(group),
        "run_false_alarm_rate": float(
            (
                group["requests"]
                > 0
            ).mean()
        ),
        "mean_false_requests": float(
            group[
                "requests"
            ].mean()
        ),
        "median_false_requests": float(
            group[
                "requests"
            ].median()
        ),
        "max_false_requests": int(
            group[
                "requests"
            ].max()
        ),
        "mean_retrainings": float(
            group[
                "retrainings"
            ].mean()
        ),
        "mean_net_value": float(
            group[
                "net_value"
            ].mean()
        ),
    }

    if trigger != "never":
        trigger_values = (
            group
            .set_index("seed")
            ["net_value"]
        )

        comparison = (
            paired_bootstrap(
                trigger_values,
                never_stationary,
                rng,
            )
        )

        row.update(
            {
                "mean_net_delta_vs_never": (
                    comparison[
                        "mean_diff"
                    ]
                ),
                "net_delta_ci95_low": (
                    comparison[
                        "ci95_low"
                    ]
                ),
                "net_delta_ci95_high": (
                    comparison[
                        "ci95_high"
                    ]
                ),
                "wins_vs_never": (
                    comparison[
                        "wins"
                    ]
                ),
                "ties_vs_never": (
                    comparison[
                        "ties"
                    ]
                ),
                "losses_vs_never": (
                    comparison[
                        "losses"
                    ]
                ),
            }
        )

    stationary_rows.append(
        row
    )

stationary_summary = (
    pd.DataFrame(
        stationary_rows
    )
)

stationary_summary.to_csv(
    OUTPUT
    / "stationary_false_alarms.csv",
    index=False,
)


# ============================================================
# Post-drift response diagnostics
#
# Important:
# For periodic retraining this is a response delay,
# NOT causal "detection".
# For data-driven triggers it can be interpreted as
# first trigger response after the known simulator shift.
# ============================================================

response_rows: list[dict] = []

for drift in DRIFTS:
    if drift == "none":
        continue

    for trigger in TRIGGERS:
        group = runs[
            (
                runs[
                    "drift_type"
                ]
                == drift
            )
            & (
                runs[
                    "trigger"
                ]
                == trigger
            )
        ].copy()

        detected = group[
            group[
                "post_drift_response"
            ]
        ]

        response_rows.append(
            {
                "drift_type": drift,
                "trigger": trigger,
                "n": len(group),
                "pre_drift_false_alarm_rate": float(
                    group[
                        "pre_drift_false_alarm"
                    ].mean()
                ),
                "mean_pre_drift_requests": float(
                    group[
                        "pre_drift_requests"
                    ].mean()
                ),
                "post_drift_response_rate": float(
                    group[
                        "post_drift_response"
                    ].mean()
                ),
                "missed_response_rate": float(
                    1.0
                    - group[
                        "post_drift_response"
                    ].mean()
                ),
                "mean_first_response_delay": (
                    float(
                        detected[
                            "first_response_delay"
                        ].mean()
                    )
                    if len(detected)
                    else np.nan
                ),
                "median_first_response_delay": (
                    float(
                        detected[
                            "first_response_delay"
                        ].median()
                    )
                    if len(detected)
                    else np.nan
                ),
                "mean_post_drift_requests": float(
                    group[
                        "post_drift_requests"
                    ].mean()
                ),
                "mean_retrainings": float(
                    group[
                        "retrainings"
                    ].mean()
                ),
            }
        )

response_summary = (
    pd.DataFrame(
        response_rows
    )
)

response_summary.to_csv(
    OUTPUT
    / "drift_response_summary.csv",
    index=False,
)


# ============================================================
# Paired net-value comparison versus NEVER
# ============================================================

vs_never_rows: list[dict] = []

for drift in DRIFTS:
    drift_data = runs[
        runs[
            "drift_type"
        ]
        == drift
    ]

    baseline = (
        drift_data[
            drift_data["trigger"]
            == "never"
        ]
        .set_index("seed")
        ["net_value"]
    )

    for trigger in TRIGGERS:
        if trigger == "never":
            continue

        candidate = (
            drift_data[
                drift_data["trigger"]
                == trigger
            ]
            .set_index("seed")
            ["net_value"]
        )

        result = paired_bootstrap(
            candidate,
            baseline,
            rng,
        )

        vs_never_rows.append(
            {
                "drift_type": drift,
                "trigger": trigger,
                "comparison": (
                    f"{trigger} - never"
                ),
                **result,
            }
        )

paired_vs_never = pd.DataFrame(
    vs_never_rows
)

paired_vs_never.to_csv(
    OUTPUT
    / "paired_vs_never.csv",
    index=False,
)


# ============================================================
# Paired net-value comparison versus PERIODIC
# ============================================================

vs_periodic_rows: list[dict] = []

for drift in DRIFTS:
    drift_data = runs[
        runs[
            "drift_type"
        ]
        == drift
    ]

    baseline = (
        drift_data[
            drift_data["trigger"]
            == "periodic"
        ]
        .set_index("seed")
        ["net_value"]
    )

    for trigger in TRIGGERS:
        if trigger == "periodic":
            continue

        candidate = (
            drift_data[
                drift_data["trigger"]
                == trigger
            ]
            .set_index("seed")
            ["net_value"]
        )

        result = paired_bootstrap(
            candidate,
            baseline,
            rng,
        )

        vs_periodic_rows.append(
            {
                "drift_type": drift,
                "trigger": trigger,
                "comparison": (
                    f"{trigger} - periodic"
                ),
                **result,
            }
        )

paired_vs_periodic = pd.DataFrame(
    vs_periodic_rows
)

paired_vs_periodic.to_csv(
    OUTPUT
    / "paired_vs_periodic.csv",
    index=False,
)


# ============================================================
# All pairwise comparisons
# ============================================================

pairwise_rows: list[dict] = []

for drift in DRIFTS:
    drift_data = runs[
        runs[
            "drift_type"
        ]
        == drift
    ]

    for left_trigger, right_trigger in itertools.combinations(
        TRIGGERS,
        2,
    ):
        left = (
            drift_data[
                drift_data["trigger"]
                == left_trigger
            ]
            .set_index("seed")
            ["net_value"]
        )

        right = (
            drift_data[
                drift_data["trigger"]
                == right_trigger
            ]
            .set_index("seed")
            ["net_value"]
        )

        result = paired_bootstrap(
            left,
            right,
            rng,
        )

        pairwise_rows.append(
            {
                "drift_type": drift,
                "trigger_a": left_trigger,
                "trigger_b": right_trigger,
                "comparison": (
                    f"{left_trigger} - "
                    f"{right_trigger}"
                ),
                "primary_comparison": (
                    left_trigger
                    in PRIMARY_TRIGGERS
                    and right_trigger
                    in PRIMARY_TRIGGERS
                ),
                **result,
            }
        )

pairwise = pd.DataFrame(
    pairwise_rows
)

pairwise.to_csv(
    OUTPUT
    / "paired_all_strategies.csv",
    index=False,
)


# ============================================================
# Aggregate time-series trajectories
#
# These are important for the next stage:
# recovery dynamics and figures.
# ============================================================

trajectory_summary = (
    trajectories.groupby(
        [
            "benchmark_drift",
            "benchmark_trigger",
            "step",
        ],
        as_index=False,
    )
    .agg(
        mean_step_net_value=(
            "step_net_value",
            "mean",
        ),
        mean_cumulative_net_value=(
            "cumulative_net_value",
            "mean",
        ),
        mean_oracle_value_per_customer=(
            "oracle_value_per_customer",
            "mean",
        ),
        mean_true_cate=(
            "true_mean_cate",
            "mean",
        ),
        mean_predicted_uplift=(
            "predicted_mean_uplift",
            "mean",
        ),
        mean_policy_treatment_rate=(
            "policy_treatment_rate",
            "mean",
        ),
        mean_actual_treatment_rate=(
            "actual_treatment_rate",
            "mean",
        ),
        mean_feature_drift=(
            "feature_drift",
            "mean",
        ),
        mean_brier=(
            "brier",
            "mean",
        ),
        mean_cate_shift=(
            "cate_shift",
            "mean",
        ),
        request_rate=(
            "retraining_requested",
            "mean",
        ),
        retrain_rate=(
            "retrained",
            "mean",
        ),
    )
)

trajectory_summary.to_csv(
    OUTPUT
    / "trajectory_summary.csv",
    index=False,
)


# ============================================================
# Display order
# ============================================================

trigger_dtype = (
    pd.CategoricalDtype(
        categories=TRIGGERS,
        ordered=True,
    )
)

drift_dtype = (
    pd.CategoricalDtype(
        categories=DRIFTS,
        ordered=True,
    )
)

group_summary[
    "trigger"
] = group_summary[
    "trigger"
].astype(
    trigger_dtype
)

group_summary[
    "drift_type"
] = group_summary[
    "drift_type"
].astype(
    drift_dtype
)


# ============================================================
# Console results
# ============================================================

print()
print("=" * 110)
print("MEAN NET VALUE BY DRIFT")
print("=" * 110)

net_pivot = (
    group_summary.pivot(
        index="drift_type",
        columns="trigger",
        values="mean_net_value",
    )
    .reindex(
        index=DRIFTS,
        columns=TRIGGERS,
    )
)

print(
    net_pivot
    .round(0)
    .to_string()
)


print()
print("=" * 110)
print("MEAN RETRAININGS BY DRIFT")
print("=" * 110)

retrain_pivot = (
    group_summary.pivot(
        index="drift_type",
        columns="trigger",
        values="mean_retrainings",
    )
    .reindex(
        index=DRIFTS,
        columns=TRIGGERS,
    )
)

print(
    retrain_pivot
    .round(2)
    .to_string()
)


print()
print("=" * 110)
print("STATIONARY FALSE-ALARM RESULTS")
print("=" * 110)

stationary_display = (
    stationary_summary[
        [
            "trigger",
            "run_false_alarm_rate",
            "mean_false_requests",
            "max_false_requests",
            "mean_net_value",
            "mean_net_delta_vs_never",
            "net_delta_ci95_low",
            "net_delta_ci95_high",
        ]
    ]
    .copy()
)

print(
    stationary_display
    .round(3)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("POST-DRIFT RESPONSE SUMMARY")
print("=" * 110)

print(
    response_summary[
        [
            "drift_type",
            "trigger",
            "pre_drift_false_alarm_rate",
            "post_drift_response_rate",
            "missed_response_rate",
            "mean_first_response_delay",
            "mean_post_drift_requests",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("PAIRED NET VALUE VS NEVER")
print("=" * 110)

print(
    paired_vs_never[
        [
            "drift_type",
            "trigger",
            "mean_diff",
            "ci95_low",
            "ci95_high",
            "wins",
            "ties",
            "losses",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("PAIRED NET VALUE VS PERIODIC")
print("=" * 110)

print(
    paired_vs_periodic[
        [
            "drift_type",
            "trigger",
            "mean_diff",
            "ci95_low",
            "ci95_high",
            "wins",
            "ties",
            "losses",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("OUTPUT FILES")
print("=" * 110)

for path in sorted(
    OUTPUT.glob("*.csv")
):
    print(path)

print()
print("Analysis complete.")
