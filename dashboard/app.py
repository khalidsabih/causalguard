import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="CausalGuard",
    page_icon="🛡️",
    layout="wide",
)


# =========================================================
# Paths
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_DIR = (
    ROOT
    / "analysis"
    / "main_trigger_benchmark"
)

ORANGE_ANALYSIS_DIR = (
    ROOT
    / "analysis"
    / "orange"
)


# =========================================================
# Data loading
# =========================================================

@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    files = {
        "group_summary": "group_summary.csv",
        "stationary": "stationary_false_alarms.csv",
        "response": "drift_response_summary.csv",
        "vs_never": "paired_vs_never.csv",
        "vs_periodic": "paired_vs_periodic.csv",
        "runs": "run_level_metrics.csv",
        "trajectory": "trajectory_summary.csv",
        "mechanism": "mechanism_trajectories.csv",
        "decomposition": "mechanism_decomposition.csv",
    }

    loaded = {}

    for key, filename in files.items():
        path = ANALYSIS_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required analysis file: {path}"
            )

        loaded[key] = pd.read_csv(path)

    return loaded


@st.cache_data
def load_orange_data() -> dict[str, object]:
    files = {
        "audit": "data_audit.json",
        "primary_policy": "primary_policy_results.csv",
        "primary_paired": "primary_paired_results.csv",
        "stability_paired": "partition_stability_paired.csv",
        "stability_summary": "partition_stability_summary.csv",
    }

    loaded: dict[str, object] = {}

    for key, filename in files.items():
        path = ORANGE_ANALYSIS_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required Orange analysis file: {path}"
            )

        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                loaded[key] = json.load(handle)
        else:
            loaded[key] = pd.read_csv(path)

    return loaded


try:
    data = load_data()

except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


# =========================================================
# Labels and ordering
# =========================================================

DRIFT_LABELS = {
    "none": "No drift",
    "covariate": "Covariate drift",
    "outcome": "Outcome drift",
    "treatment": "Abrupt treatment-effect drift",
    "gradual_treatment": "Gradual treatment-effect drift",
    "recurring_treatment": "Recurring treatment-effect drift",
}

TRIGGER_LABELS = {
    "never": "Never retrain",
    "periodic": "Periodic",
    "feature_drift": "Feature drift",
    "performance": "Predictive performance",
    "cate_shift": "CATE shift",
    "policy_value": "Policy value",
    "policy_value_confident": "Policy value + confidence",
}

TRIGGER_ORDER = [
    "never",
    "periodic",
    "feature_drift",
    "performance",
    "cate_shift",
    "policy_value",
    "policy_value_confident",
]

SECONDARY_STRATEGIES = {
    "policy_value_confident",
}


def strategy_type(trigger: str) -> str:
    if trigger in SECONDARY_STRATEGIES:
        return "Secondary sensitivity"

    return "Primary benchmark"


def prepare_strategy_labels(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    frame["trigger"] = pd.Categorical(
        frame["trigger"],
        categories=TRIGGER_ORDER,
        ordered=True,
    )

    frame = frame.sort_values(
        "trigger"
    ).reset_index(drop=True)

    frame["strategy_label"] = (
        frame["trigger"]
        .astype(str)
        .map(TRIGGER_LABELS)
    )

    frame["strategy_type"] = (
        frame["trigger"]
        .astype(str)
        .apply(strategy_type)
    )

    return frame




def render_simulation_tab() -> None:
    # =========================================================
    # Header
    # =========================================================

    st.header("Simulation Benchmark")

    st.subheader(
        "Monitoring and Retraining Causal Targeting Policies "
        "Under Distribution Shift"
    )

    st.markdown(
        """
CausalGuard studies when retraining a causal targeting
policy creates value — and when reacting to distribution
shift can make the deployed policy worse.
"""
    )


    # =========================================================
    # Benchmark summary metrics
    # =========================================================

    runs = data["runs"]

    n_runs = len(runs)
    n_seeds = runs["seed"].nunique()
    n_drifts = runs["drift_type"].nunique()
    n_triggers = runs["trigger"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Benchmark runs",
        f"{n_runs:,}",
    )

    col2.metric(
        "Evaluation seeds",
        n_seeds,
    )

    col3.metric(
        "Drift regimes",
        n_drifts,
    )

    col4.metric(
        "Strategies",
        n_triggers,
    )


    # =========================================================
    # Research question
    # =========================================================

    st.divider()

    st.header("Research question")

    st.info(
        """
How do distribution-based, prediction-based,
treatment-effect-based, and policy-value-based retraining
signals compare in preserving the net decision value of a
causal targeting policy under temporal distribution shift?
"""
    )


    # =========================================================
    # Main finding
    # =========================================================

    st.header("Main finding")

    st.markdown(
        """
**Drift detection alone is not sufficient evidence that a
causal policy should be retrained.**

The economic value of retraining depends on:

- what part of the data-generating process changed,
- whether the change affects treatment effects or decisions,
- how persistent the new regime is,
- and whether the model can adapt before the environment
  changes again.
"""
    )


    # =========================================================
    # Benchmark Overview
    # =========================================================

    st.divider()

    st.header("Benchmark Overview")

    available_drifts = [
        drift
        for drift in DRIFT_LABELS
        if drift in data["group_summary"]["drift_type"].unique()
    ]

    selected_drift = st.selectbox(
        "Select drift regime",
        options=available_drifts,
        format_func=lambda value: DRIFT_LABELS.get(
            value,
            value,
        ),
        key="benchmark_drift_selector",
    )


    # ---------------------------------------------------------
    # Filter benchmark summary
    # ---------------------------------------------------------

    summary = data["group_summary"].copy()

    summary = summary[
        summary["drift_type"] == selected_drift
    ].copy()

    summary = prepare_strategy_labels(
        summary
    )


    # ---------------------------------------------------------
    # Selected regime heading
    # ---------------------------------------------------------

    st.subheader(
        DRIFT_LABELS.get(
            selected_drift,
            selected_drift,
        )
    )

    st.caption(
        "Policy value + confidence is shown as a secondary, "
        "pilot-informed sensitivity analysis rather than a "
        "primary pre-specified benchmark strategy."
    )


    # ---------------------------------------------------------
    # Mean net value
    # ---------------------------------------------------------

    fig_value = go.Figure()

    for group_name in [
        "Primary benchmark",
        "Secondary sensitivity",
    ]:
        subset = summary[
            summary["strategy_type"] == group_name
        ]

        fig_value.add_trace(
            go.Bar(
                x=subset["strategy_label"],
                y=subset["mean_net_value"],
                name=group_name,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (
                        subset["mean_net_ci95_high"]
                        - subset["mean_net_value"]
                    ),
                    "arrayminus": (
                        subset["mean_net_value"]
                        - subset["mean_net_ci95_low"]
                    ),
                },
                customdata=subset[
                    [
                        "median_net_value",
                        "std_net_value",
                        "mean_retrainings",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Mean net value: €%{y:,.0f}<br>"
                    "Median: €%{customdata[0]:,.0f}<br>"
                    "SD: €%{customdata[1]:,.0f}<br>"
                    "Mean retrainings: "
                    "%{customdata[2]:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig_value.add_hline(
        y=0,
        line_width=1,
    )

    fig_value.update_layout(
        title="Mean net value",
        xaxis_title=None,
        yaxis_title="Net value (€)",
        legend_title=None,
    )

    st.plotly_chart(
        fig_value,
        use_container_width=True,
    )


    # ---------------------------------------------------------
    # Retraining burden
    # ---------------------------------------------------------

    fig_retraining = go.Figure()

    fig_retraining.add_trace(
        go.Bar(
            x=summary["strategy_label"],
            y=summary["mean_retrainings"],
            customdata=summary[
                [
                    "mean_requests",
                    "request_run_rate",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean retrainings: %{y:.2f}<br>"
                "Mean requests: %{customdata[0]:.2f}<br>"
                "Runs with any request: "
                "%{customdata[1]:.1%}"
                "<extra></extra>"
            ),
        )
    )

    fig_retraining.update_layout(
        title="Retraining burden",
        xaxis_title=None,
        yaxis_title="Mean retrainings per run",
        showlegend=False,
    )

    st.plotly_chart(
        fig_retraining,
        use_container_width=True,
    )


    # ---------------------------------------------------------
    # Paired comparison versus never
    # ---------------------------------------------------------

    st.subheader(
        "Paired value difference vs never retraining"
    )

    st.caption(
        "Positive values indicate higher net value than the "
        "never-retrain baseline. Error bars show paired 95% "
        "bootstrap confidence intervals."
    )

    paired = data["vs_never"].copy()

    paired = paired[
        paired["drift_type"] == selected_drift
    ].copy()

    paired = prepare_strategy_labels(
        paired
    )

    fig_paired = go.Figure()

    for group_name in [
        "Primary benchmark",
        "Secondary sensitivity",
    ]:
        subset = paired[
            paired["strategy_type"] == group_name
        ]

        fig_paired.add_trace(
            go.Bar(
                x=subset["strategy_label"],
                y=subset["mean_diff"],
                name=group_name,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (
                        subset["ci95_high"]
                        - subset["mean_diff"]
                    ),
                    "arrayminus": (
                        subset["mean_diff"]
                        - subset["ci95_low"]
                    ),
                },
                customdata=subset[
                    [
                        "wins",
                        "ties",
                        "losses",
                        "median_diff",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Mean difference: €%{y:,.0f}<br>"
                    "Median difference: "
                    "€%{customdata[3]:,.0f}<br>"
                    "Wins: %{customdata[0]}<br>"
                    "Ties: %{customdata[1]}<br>"
                    "Losses: %{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    fig_paired.add_hline(
        y=0,
        line_width=1,
    )

    fig_paired.update_layout(
        xaxis_title=None,
        yaxis_title="Δ net value vs never (€)",
        legend_title=None,
    )

    st.plotly_chart(
        fig_paired,
        use_container_width=True,
    )


    # ---------------------------------------------------------
    # Detailed benchmark table
    # ---------------------------------------------------------

    with st.expander(
        "Show detailed benchmark statistics"
    ):
        display = summary[
            [
                "strategy_label",
                "n",
                "mean_net_value",
                "mean_net_ci95_low",
                "mean_net_ci95_high",
                "mean_retrainings",
                "mean_requests",
                "request_run_rate",
            ]
        ].copy()

        display["request_run_rate_pct"] = (
            100 * display["request_run_rate"]
        )

        display = display[
            [
                "strategy_label",
                "n",
                "mean_net_value",
                "mean_net_ci95_low",
                "mean_net_ci95_high",
                "mean_retrainings",
                "mean_requests",
                "request_run_rate_pct",
            ]
        ]

        display = display.rename(
            columns={
                "strategy_label": "Strategy",
                "n": "Runs",
                "mean_net_value": "Mean net value",
                "mean_net_ci95_low": "95% CI low",
                "mean_net_ci95_high": "95% CI high",
                "mean_retrainings": "Mean retrainings",
                "mean_requests": "Mean requests",
                "request_run_rate_pct": "Runs with request (%)",
            }
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Mean net value":
                    st.column_config.NumberColumn(
                        format="€ %.0f"
                    ),
                "95% CI low":
                    st.column_config.NumberColumn(
                        format="€ %.0f"
                    ),
                "95% CI high":
                    st.column_config.NumberColumn(
                        format="€ %.0f"
                    ),
                "Mean retrainings":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                "Mean requests":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                "Runs with request (%)":
                    st.column_config.NumberColumn(
                        format="%.1f%%"
                    ),
            },
        )


    # ---------------------------------------------------------
    # Frozen benchmark notice
    # ---------------------------------------------------------

    st.info(
        "All values shown above come from the frozen held-out "
        "benchmark. Dashboard controls only change the view; "
        "they do not rerun, retrain, or tune any model."
    )


    # =========================================================
    # Monitoring Behavior
    # =========================================================

    st.divider()

    st.header("Monitoring Behavior")

    st.markdown(
        """
This section separates two questions:

1. **How often does a strategy act when nothing has changed?**
2. **How does it respond after a real distributional change?**

For periodic retraining, actions are scheduled rather than
statistical false alarms.
"""
    )


    # =========================================================
    # Stationary behavior
    # =========================================================

    st.subheader("Behavior under stationarity")

    stationary_runs = data["runs"][
        data["runs"]["drift_type"] == "none"
    ].copy()

    stationary_summary = (
        stationary_runs
        .groupby(
            "trigger",
            as_index=False,
            observed=True,
        )
        .agg(
            action_rate=("any_request", "mean"),
            mean_requests=("requests", "mean"),
            mean_retrainings=("retrainings", "mean"),
        )
    )

    stationary_summary = prepare_strategy_labels(
        stationary_summary
    )

    stationary_summary["action_rate_pct"] = (
        100 * stationary_summary["action_rate"]
    )


    # ---------------------------------------------------------
    # Stationary action chart
    # ---------------------------------------------------------

    fig_stationary = go.Figure()

    for group_name in [
        "Primary benchmark",
        "Secondary sensitivity",
    ]:
        subset = stationary_summary[
            stationary_summary["strategy_type"] == group_name
        ]

        fig_stationary.add_trace(
            go.Bar(
                x=subset["strategy_label"],
                y=subset["action_rate_pct"],
                name=group_name,
                customdata=subset[
                    [
                        "mean_requests",
                        "mean_retrainings",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Runs with action: %{y:.1f}%<br>"
                    "Mean requests: "
                    "%{customdata[0]:.2f}<br>"
                    "Mean retrainings: "
                    "%{customdata[1]:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig_stationary.update_layout(
        title="Stationary action rate",
        xaxis_title=None,
        yaxis_title="Runs with ≥1 action (%)",
        yaxis={
            "range": [0, 105],
        },
        legend_title=None,
    )

    st.plotly_chart(
        fig_stationary,
        use_container_width=True,
    )

    st.caption(
        "For data-driven monitors, an action under stationarity "
        "is interpreted as a false alarm. Periodic retraining is "
        "different: its stationary actions are scheduled by design."
    )


    # ---------------------------------------------------------
    # Stationary request burden
    # ---------------------------------------------------------

    fig_stationary_requests = go.Figure()

    fig_stationary_requests.add_trace(
        go.Bar(
            x=stationary_summary["strategy_label"],
            y=stationary_summary["mean_requests"],
            customdata=stationary_summary[
                [
                    "mean_retrainings",
                    "action_rate_pct",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean requests: %{y:.2f}<br>"
                "Mean retrainings: "
                "%{customdata[0]:.2f}<br>"
                "Runs with action: "
                "%{customdata[1]:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig_stationary_requests.update_layout(
        title="Stationary monitoring burden",
        xaxis_title=None,
        yaxis_title="Mean requests per run",
        showlegend=False,
    )

    st.plotly_chart(
        fig_stationary_requests,
        use_container_width=True,
    )


    # =========================================================
    # Post-drift response
    # =========================================================

    st.subheader("Response after drift")

    response_drifts = [
        drift
        for drift in DRIFT_LABELS
        if (
            drift != "none"
            and drift
            in data["response"]["drift_type"].unique()
        )
    ]

    default_response_index = 0

    if "treatment" in response_drifts:
        default_response_index = (
            response_drifts.index("treatment")
        )

    response_drift = st.selectbox(
        "Select drift regime for response analysis",
        options=response_drifts,
        index=default_response_index,
        format_func=lambda value: DRIFT_LABELS.get(
            value,
            value,
        ),
        key="response_drift_selector",
    )

    response = data["response"].copy()

    response = response[
        response["drift_type"] == response_drift
    ].copy()

    response = prepare_strategy_labels(
        response
    )

    response["response_rate_pct"] = (
        100 * response["post_drift_response_rate"]
    )

    response["missed_rate_pct"] = (
        100 * response["missed_response_rate"]
    )

    response["pre_drift_action_pct"] = (
        100 * response["pre_drift_false_alarm_rate"]
    )


    # ---------------------------------------------------------
    # Response rate
    # ---------------------------------------------------------

    fig_response = go.Figure()

    for group_name in [
        "Primary benchmark",
        "Secondary sensitivity",
    ]:
        subset = response[
            response["strategy_type"] == group_name
        ]

        fig_response.add_trace(
            go.Bar(
                x=subset["strategy_label"],
                y=subset["response_rate_pct"],
                name=group_name,
                customdata=subset[
                    [
                        "missed_rate_pct",
                        "mean_first_response_delay",
                        "mean_post_drift_requests",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Post-drift response: %{y:.1f}%<br>"
                    "Missed response: "
                    "%{customdata[0]:.1f}%<br>"
                    "Mean response delay: "
                    "%{customdata[1]:.2f} steps<br>"
                    "Mean post-drift requests: "
                    "%{customdata[2]:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig_response.update_layout(
        title=(
            "Post-drift response rate — "
            + DRIFT_LABELS.get(
                response_drift,
                response_drift,
            )
        ),
        xaxis_title=None,
        yaxis_title=(
            "Runs with ≥1 post-drift response (%)"
        ),
        yaxis={
            "range": [0, 105],
        },
        legend_title=None,
    )

    st.plotly_chart(
        fig_response,
        use_container_width=True,
    )


    # ---------------------------------------------------------
    # Response delay
    # ---------------------------------------------------------

    delay_data = response[
        response["mean_first_response_delay"].notna()
    ].copy()

    fig_delay = go.Figure()

    fig_delay.add_trace(
        go.Bar(
            x=delay_data["strategy_label"],
            y=delay_data["mean_first_response_delay"],
            customdata=delay_data[
                [
                    "median_first_response_delay",
                    "response_rate_pct",
                    "mean_post_drift_requests",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean first response delay: "
                "%{y:.2f} steps<br>"
                "Median delay: "
                "%{customdata[0]:.2f} steps<br>"
                "Response rate: "
                "%{customdata[1]:.1f}%<br>"
                "Mean post-drift requests: "
                "%{customdata[2]:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig_delay.update_layout(
        title="First-response delay",
        xaxis_title=None,
        yaxis_title="Production steps after drift",
        showlegend=False,
    )

    st.plotly_chart(
        fig_delay,
        use_container_width=True,
    )

    st.caption(
        "Delay is computed only among runs that produced a "
        "post-drift response. A low delay should therefore be "
        "interpreted together with the response rate above."
    )


    # ---------------------------------------------------------
    # Post-drift request burden
    # ---------------------------------------------------------

    fig_post_requests = go.Figure()

    fig_post_requests.add_trace(
        go.Bar(
            x=response["strategy_label"],
            y=response["mean_post_drift_requests"],
            customdata=response[
                [
                    "mean_retrainings",
                    "response_rate_pct",
                    "pre_drift_action_pct",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean post-drift requests: %{y:.2f}<br>"
                "Mean total retrainings: "
                "%{customdata[0]:.2f}<br>"
                "Post-drift response rate: "
                "%{customdata[1]:.1f}%<br>"
                "Pre-drift action rate: "
                "%{customdata[2]:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig_post_requests.update_layout(
        title="Post-drift monitoring burden",
        xaxis_title=None,
        yaxis_title="Mean post-drift requests",
        showlegend=False,
    )

    st.plotly_chart(
        fig_post_requests,
        use_container_width=True,
    )


    # ---------------------------------------------------------
    # Detailed monitoring table
    # ---------------------------------------------------------

    with st.expander(
        "Show detailed monitoring statistics"
    ):
        monitoring_table = response[
            [
                "strategy_label",
                "pre_drift_action_pct",
                "response_rate_pct",
                "missed_rate_pct",
                "mean_first_response_delay",
                "median_first_response_delay",
                "mean_post_drift_requests",
                "mean_retrainings",
            ]
        ].copy()

        monitoring_table = monitoring_table.rename(
            columns={
                "strategy_label": "Strategy",
                "pre_drift_action_pct":
                    "Pre-drift action (%)",
                "response_rate_pct":
                    "Response (%)",
                "missed_rate_pct":
                    "Missed (%)",
                "mean_first_response_delay":
                    "Mean delay",
                "median_first_response_delay":
                    "Median delay",
                "mean_post_drift_requests":
                    "Post-drift requests",
                "mean_retrainings":
                    "Mean retrainings",
            }
        )

        st.dataframe(
            monitoring_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pre-drift action (%)":
                    st.column_config.NumberColumn(
                        format="%.1f%%"
                    ),
                "Response (%)":
                    st.column_config.NumberColumn(
                        format="%.1f%%"
                    ),
                "Missed (%)":
                    st.column_config.NumberColumn(
                        format="%.1f%%"
                    ),
                "Mean delay":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                "Median delay":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                "Post-drift requests":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                "Mean retrainings":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),
            },
        )

    # =========================================================
    # Mechanism Explorer
    # =========================================================

    st.divider()

    st.header("Mechanism Explorer")

    st.markdown(
        """
The aggregate benchmark tells us **whether** retraining
created value. This section investigates **how** that value
was created or destroyed over time.

Oracle treatment effects are shown only because this is a
controlled simulation. They are not available to a deployed
monitoring system.
"""
    )


    # ---------------------------------------------------------
    # Mechanism cases
    # ---------------------------------------------------------

    MECHANISM_CASES = {
        "Covariate drift → harmful retraining": {
            "drift": "covariate",
            "trigger": "feature_drift",
            "change_steps": [14],
            "summary": (
                "Feature drift is detected correctly, but "
                "retraining changes the targeting policy and "
                "reduces decision value."
            ),
        },
        "Persistent treatment drift → useful adaptation": {
            "drift": "treatment",
            "trigger": "policy_value",
            "change_steps": [14],
            "summary": (
                "Treatment effects become harmful. Retraining "
                "learns to suppress treatment and recovers value."
            ),
        },
        "Recurring treatment drift → phase mismatch": {
            "drift": "recurring_treatment",
            "trigger": "policy_value",
            "change_steps": [14, 18, 22],
            "summary": (
                "The environment changes again before adaptation "
                "fully settles, causing the learned policy to lag "
                "behind the current regime."
            ),
        },
    }


    mechanism_choice = st.selectbox(
        "Select mechanism",
        options=list(MECHANISM_CASES),
        key="mechanism_selector",
    )

    case = MECHANISM_CASES[
        mechanism_choice
    ]

    st.subheader(
        mechanism_choice
    )

    st.caption(
        case["summary"]
    )


    # ---------------------------------------------------------
    # Filter mechanism data
    # ---------------------------------------------------------

    mechanism = data["mechanism"].copy()

    mechanism = mechanism[
        (
            mechanism["drift"]
            == case["drift"]
        )
        & (
            mechanism["trigger"]
            == case["trigger"]
        )
    ].copy()

    mechanism = mechanism.sort_values(
        "step"
    ).reset_index(drop=True)


    decomposition = data["decomposition"].copy()

    decomposition = decomposition[
        (
            decomposition["drift"]
            == case["drift"]
        )
        & (
            decomposition["trigger"]
            == case["trigger"]
        )
    ].copy()


    if mechanism.empty:
        st.warning(
            "No mechanism trajectory found for this case."
        )

    else:
        # -----------------------------------------------------
        # Economic decomposition
        # -----------------------------------------------------

        if not decomposition.empty:
            row = decomposition.iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Net value Δ vs never",
                (
                    f"€{row['mean_net_delta_vs_never']:,.0f}"
                ),
            )

            col2.metric(
                "Direct retraining cost",
                (
                    f"€{row['direct_retraining_cost']:,.0f}"
                ),
            )

            col3.metric(
                "Decision value Δ vs never",
                (
                    f"€{row['decision_value_delta_vs_never']:,.0f}"
                ),
            )

            st.caption(
                "Decision value removes the direct retraining "
                "charge, helping separate model/policy effects "
                "from the cost of running retraining itself."
            )


        # -----------------------------------------------------
        # Helper for regime-change markers
        # -----------------------------------------------------

        def add_change_markers(
            figure: go.Figure,
        ) -> None:
            for step in case["change_steps"]:
                figure.add_vline(
                    x=step,
                    line_dash="dash",
                    line_width=1,
                    opacity=0.6,
                )


        # =====================================================
        # Row 1
        # =====================================================

        left, right = st.columns(2)


        # -----------------------------------------------------
        # Causal signal
        # -----------------------------------------------------

        with left:
            fig_effect = go.Figure()

            fig_effect.add_trace(
                go.Scatter(
                    x=mechanism["step"],
                    y=mechanism["true_cate_oracle"],
                    mode="lines",
                    name="True mean CATE (oracle)",
                    line={
                        "width": 3,
                    },
                )
            )

            fig_effect.add_trace(
                go.Scatter(
                    x=mechanism["step"],
                    y=mechanism[
                        "candidate_predicted_uplift"
                    ],
                    mode="lines",
                    name="Retraining strategy",
                    line={
                        "width": 2,
                    },
                )
            )

            fig_effect.add_trace(
                go.Scatter(
                    x=mechanism["step"],
                    y=mechanism[
                        "never_predicted_uplift"
                    ],
                    mode="lines",
                    name="Never retrain",
                    line={
                        "width": 2,
                        "dash": "dot",
                    },
                )
            )

            fig_effect.add_hline(
                y=0,
                line_width=1,
            )

            add_change_markers(
                fig_effect
            )

            fig_effect.update_layout(
                title="Treatment effect and model belief",
                xaxis_title="Production step",
                yaxis_title="Treatment effect",
                legend_title=None,
                hovermode="x unified",
            )

            st.plotly_chart(
                fig_effect,
                use_container_width=True,
            )


        # -----------------------------------------------------
        # Treatment behavior
        # -----------------------------------------------------

        with right:
            fig_treatment = go.Figure()

            fig_treatment.add_trace(
                go.Scatter(
                    x=mechanism["step"],
                    y=mechanism[
                        "candidate_treatment_rate"
                    ],
                    mode="lines",
                    name="Retraining strategy",
                    line={
                        "width": 3,
                    },
                )
            )

            fig_treatment.add_trace(
                go.Scatter(
                    x=mechanism["step"],
                    y=mechanism[
                        "never_treatment_rate"
                    ],
                    mode="lines",
                    name="Never retrain",
                    line={
                        "width": 2,
                        "dash": "dot",
                    },
                )
            )

            add_change_markers(
                fig_treatment
            )

            fig_treatment.update_layout(
                title="Policy treatment rate",
                xaxis_title="Production step",
                yaxis_title="Fraction treated",
                legend_title=None,
                hovermode="x unified",
            )

            st.plotly_chart(
                fig_treatment,
                use_container_width=True,
            )


        # =====================================================
        # Row 2
        # =====================================================

        left, right = st.columns(2)


        # -----------------------------------------------------
        # Retraining activity
        # -----------------------------------------------------

        with left:
            fig_activity = go.Figure()

            fig_activity.add_trace(
                go.Scatter(
                    x=mechanism["step"],
                    y=mechanism["request_rate"],
                    mode="lines",
                    name="Request rate",
                    line={
                        "width": 2,
                    },
                )
            )

            fig_activity.add_trace(
                go.Scatter(
                    x=mechanism["step"],
                    y=mechanism["retrain_rate"],
                    mode="lines",
                    name="Retrain rate",
                    line={
                        "width": 2,
                        "dash": "dash",
                    },
                )
            )

            add_change_markers(
                fig_activity
            )

            fig_activity.update_layout(
                title="Retraining activity",
                xaxis_title="Production step",
                yaxis_title="Fraction of runs",
                yaxis={
                    "range": [0, 1.05],
                },
                legend_title=None,
                hovermode="x unified",
            )

            st.plotly_chart(
                fig_activity,
                use_container_width=True,
            )


        # -----------------------------------------------------
        # Economic consequence
        # -----------------------------------------------------

        with right:
            fig_economic = go.Figure()

            fig_economic.add_trace(
                go.Scatter(
                    x=mechanism["step"],
                    y=mechanism[
                        "decision_value_delta"
                    ],
                    mode="lines",
                    name="Decision value Δ",
                    line={
                        "width": 3,
                    },
                )
            )

            fig_economic.add_trace(
                go.Scatter(
                    x=mechanism["step"],
                    y=mechanism[
                        "net_value_delta"
                    ],
                    mode="lines",
                    name="Net value Δ",
                    line={
                        "width": 2,
                        "dash": "dash",
                    },
                )
            )

            fig_economic.add_hline(
                y=0,
                line_width=1,
            )

            add_change_markers(
                fig_economic
            )

            fig_economic.update_layout(
                title="Economic consequence vs never retraining",
                xaxis_title="Production step",
                yaxis_title="Value difference (€ / step)",
                legend_title=None,
                hovermode="x unified",
            )

            st.plotly_chart(
                fig_economic,
                use_container_width=True,
            )


        # =====================================================
        # Interpretation
        # =====================================================

        if case["drift"] == "covariate":
            st.info(
                """
**Interpretation:** the feature monitor detects the shift,
but both strategies continue treating approximately the same
fraction of customers. The loss therefore appears after the
causal refit changes the policy rather than because treatment
volume increases.

This is consistent with a change in customer ranking or
composition, although the aggregate trajectory does not by
itself prove that ranking mechanism.
"""
            )

        elif case["drift"] == "treatment":
            st.success(
                """
**Interpretation:** after the treatment effect becomes
negative, the stale policy continues predicting positive
uplift. Retraining progressively lowers predicted uplift and
reduces treatment exposure. The resulting decision-value
gain is much larger than the direct retraining cost.
"""
            )

        elif case["drift"] == "recurring_treatment":
            st.warning(
                """
**Interpretation:** the policy adapts to one treatment-effect
regime, but the environment switches again before the model
fully catches up. The adapted policy can therefore become
misaligned with the current regime — an adaptation
**phase mismatch**.
"""
            )


        # -----------------------------------------------------
        # Raw mechanism data
        # -----------------------------------------------------

        with st.expander(
            "Show mechanism trajectory data"
        ):
            st.dataframe(
                mechanism,
                use_container_width=True,
                hide_index=True,
            )
    # =========================================================
    # Footer
    # =========================================================

    st.divider()

    st.caption(
        "Simulation benchmark — frozen held-out evaluation. "
        "Oracle quantities are used only for simulation "
        "evaluation and are not deployable monitoring signals."
    )


def render_orange_tab() -> None:
    st.header("Orange Real Data")

    st.markdown(
        """
This study evaluates causal and predictive targeting on the
**Orange Belgium randomized retention benchmark**. It is a
separate real-data study from the temporal-drift simulation.
"""
    )

    st.info(
        "Orange provides randomized treatment assignments for "
        "policy evaluation, but no controlled temporal drift "
        "process. It therefore evaluates targeting behavior — "
        "not real-world drift detection or retraining."
    )

    try:
        orange = load_orange_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.caption(
            "Commit the frozen Orange analysis artifacts under "
            "analysis/orange/ before deploying this tab."
        )
        return

    audit = orange["audit"]
    primary_policy = orange["primary_policy"].copy()
    primary_paired = orange["primary_paired"].copy()
    stability_paired = orange["stability_paired"].copy()

    # =====================================================
    # Dataset summary
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customers",
        f"{int(audit['rows']):,}",
    )
    col2.metric(
        "Features",
        f"{int(audit['feature_count']):,}",
    )
    col3.metric(
        "Treatment rate",
        f"{100 * float(audit['treatment_rate']):.2f}%",
    )
    col4.metric(
        "Primary budget",
        "25%",
    )

    st.caption(
        "Primary evaluation: 5-fold cross-fitting with a "
        "cross-fitted doubly robust (DR/AIPW) policy-value "
        "estimator. RCT difference-in-means is reported as a "
        "secondary estimator."
    )

    # =====================================================
    # Primary result
    # =====================================================

    st.divider()
    st.subheader("Primary result — 25% targeting budget")

    primary_contrast = primary_paired[
        (primary_paired["budget_fraction"] == 0.25)
        & (primary_paired["first_policy"] == "risk")
        & (primary_paired["second_policy"] == "uplift")
    ].copy()

    if "crossfit_seed" in primary_contrast.columns:
        seed_1000 = primary_contrast[
            primary_contrast["crossfit_seed"] == 1000
        ]
        if not seed_1000.empty:
            primary_contrast = seed_1000

    if primary_contrast.empty:
        st.warning(
            "Primary 25% risk-vs-uplift contrast is missing."
        )
    else:
        row = primary_contrast.iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Risk − Uplift",
            f"{row['dr_difference_per_customer']:+.6f}",
        )
        col2.metric(
            "95% CI low",
            f"{row['dr_ci95_low']:+.6f}",
        )
        col3.metric(
            "95% CI high",
            f"{row['dr_ci95_high']:+.6f}",
        )

        st.markdown(
            """
The primary point estimate is positive, but the confidence
interval includes zero. The final analysis therefore does
**not provide clear evidence of a difference** between
predictive-risk and uplift targeting at the pre-specified
25% budget.
"""
        )

    # =====================================================
    # Policy values by budget
    # =====================================================

    st.divider()
    st.subheader("Policy values")

    budget = st.selectbox(
        "Targeting budget",
        options=[0.25, 0.10, 0.50],
        format_func=lambda value: f"{value:.0%}",
        key="orange_budget_selector",
    )

    if budget != 0.25:
        st.caption(
            "10% and 50% budgets are secondary sensitivity "
            "analyses; 25% is the pre-specified primary budget."
        )

    policy_labels = {
        "never": "Never treat",
        "random": "Random",
        "risk": "Predictive risk",
        "uplift": "Uplift",
        "profit_uplift": "Profit-aware uplift",
        "treat_all": "Treat all",
    }
    policy_order = list(policy_labels)

    policy_view = primary_policy[
        primary_policy["budget_fraction"] == budget
    ].copy()

    policy_view["policy"] = pd.Categorical(
        policy_view["policy"],
        categories=policy_order,
        ordered=True,
    )
    policy_view = policy_view.sort_values("policy")
    policy_view["policy_label"] = (
        policy_view["policy"].astype(str).map(policy_labels)
    )

    fig_policy = go.Figure()
    fig_policy.add_trace(
        go.Bar(
            x=policy_view["policy_label"],
            y=policy_view[
                "dr_incremental_value_per_customer"
            ],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": (
                    policy_view["dr_ci95_high"]
                    - policy_view[
                        "dr_incremental_value_per_customer"
                    ]
                ),
                "arrayminus": (
                    policy_view[
                        "dr_incremental_value_per_customer"
                    ]
                    - policy_view["dr_ci95_low"]
                ),
            },
            customdata=policy_view[
                [
                    "treatment_rate",
                    "rct_incremental_value_per_customer",
                    "rct_standard_error",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "DR value/customer: %{y:.6f}<br>"
                "Treatment rate: %{customdata[0]:.1%}<br>"
                "RCT value/customer: %{customdata[1]:.6f}<br>"
                "RCT SE: %{customdata[2]:.6f}"
                "<extra></extra>"
            ),
        )
    )
    fig_policy.add_hline(y=0, line_width=1)
    fig_policy.update_layout(
        title=f"Cross-fitted DR policy value — {budget:.0%} budget",
        xaxis_title=None,
        yaxis_title="Incremental retained customers / customer",
        showlegend=False,
    )
    st.plotly_chart(
        fig_policy,
        use_container_width=True,
    )

    with st.expander("Show detailed Orange policy results"):
        display = policy_view[
            [
                "policy_label",
                "treatment_rate",
                "dr_incremental_value_per_customer",
                "dr_standard_error",
                "dr_ci95_low",
                "dr_ci95_high",
                "rct_incremental_value_per_customer",
                "rct_standard_error",
            ]
        ].copy()
        display = display.rename(
            columns={
                "policy_label": "Policy",
                "treatment_rate": "Treatment rate",
                "dr_incremental_value_per_customer": "DR value/customer",
                "dr_standard_error": "DR SE",
                "dr_ci95_low": "DR 95% CI low",
                "dr_ci95_high": "DR 95% CI high",
                "rct_incremental_value_per_customer": "RCT value/customer",
                "rct_standard_error": "RCT SE",
            }
        )
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

    # =====================================================
    # Partition stability
    # =====================================================

    st.divider()
    st.subheader("Cross-fit partition stability")

    stability = stability_paired[
        (stability_paired["budget_fraction"] == 0.25)
        & (stability_paired["first_policy"] == "risk")
        & (stability_paired["second_policy"] == "uplift")
    ].copy()

    seed_column = (
        "seed"
        if "seed" in stability.columns
        else "crossfit_seed"
    )
    stability = stability.sort_values(seed_column)

    if stability.empty:
        st.warning(
            "No 25% risk-vs-uplift stability results found."
        )
    else:
        differences = stability["dr_difference_per_customer"]
        positive_count = int((differences > 0).sum())
        ci_excludes_zero = int(
            (
                (stability["dr_ci95_low"] > 0)
                | (stability["dr_ci95_high"] < 0)
            ).sum()
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Mean difference",
            f"{differences.mean():+.6f}",
        )
        col2.metric(
            "Median difference",
            f"{differences.median():+.6f}",
        )
        col3.metric(
            "Positive estimates",
            f"{positive_count}/{len(stability)}",
        )
        col4.metric(
            "CIs excluding zero",
            f"{ci_excludes_zero}/{len(stability)}",
        )

        fig_stability = go.Figure()
        fig_stability.add_trace(
            go.Scatter(
                x=stability[seed_column].astype(str),
                y=stability["dr_difference_per_customer"],
                mode="markers",
                name="Risk − Uplift",
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (
                        stability["dr_ci95_high"]
                        - stability[
                            "dr_difference_per_customer"
                        ]
                    ),
                    "arrayminus": (
                        stability[
                            "dr_difference_per_customer"
                        ]
                        - stability["dr_ci95_low"]
                    ),
                },
                customdata=stability[
                    ["dr_ci95_low", "dr_ci95_high"]
                ].to_numpy(),
                hovertemplate=(
                    "<b>Partition %{x}</b><br>"
                    "Risk − Uplift: %{y:.6f}<br>"
                    "95% CI: [%{customdata[0]:.6f}, "
                    "%{customdata[1]:.6f}]"
                    "<extra></extra>"
                ),
            )
        )
        fig_stability.add_hline(y=0, line_width=1)
        fig_stability.update_layout(
            title="Risk − Uplift across cross-fit partitions",
            xaxis_title="Cross-fit partition seed",
            yaxis_title="DR difference / customer",
            showlegend=False,
        )
        st.plotly_chart(
            fig_stability,
            use_container_width=True,
        )

        st.caption(
            "These partitions reuse the same 11,896 customers. "
            "They measure sensitivity to fold assignment and "
            "must not be interpreted as independent replications."
        )

        with st.expander("Show partition-level contrasts"):
            st.dataframe(
                stability,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.caption(
        "Orange real-data study — targeting evaluation only. "
        "Temporal drift and retraining claims remain based on "
        "the controlled simulation benchmark."
    )


# =========================================================
# Application shell
# =========================================================

st.title("CausalGuard")
st.markdown(
    "Decision-aware causal targeting under distribution shift, "
    "with controlled simulation and randomized real-data evidence."
)

simulation_tab, orange_tab = st.tabs(
    [
        "Simulation Benchmark",
        "Orange Real Data",
    ]
)

with simulation_tab:
    render_simulation_tab()

with orange_tab:
    render_orange_tab()
