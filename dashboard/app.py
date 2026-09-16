from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="CausalGuard", layout="wide")
st.title("CausalGuard")
st.caption("Monitoring and retraining causal targeting policies under distribution shift")

path = st.sidebar.text_input("Results CSV", "experiments/latest/results.csv")
file_path = Path(path)
if not file_path.exists():
    st.info("Run an experiment first: causalguard-run --output experiments/latest")
    st.stop()

df = pd.read_csv(file_path)
last = df.iloc[-1]

overview, monitoring, decision_map, research_trace = st.tabs(
    ["Overview", "Monitoring", "Decision map", "Research trace"]
)

with overview:
    cols = st.columns(5)
    cols[0].metric("Cumulative net value", f"{last['cumulative_net_value']:,.0f}")
    cols[1].metric(
        "Estimated policy value / customer",
        f"{last['policy_value_estimate_per_customer']:.2f}",
    )
    cols[2].metric("Feature drift", f"{last['feature_drift']:.3f}")
    cols[3].metric("CATE shift proxy", f"{last.get('cate_shift', float('nan')):.3f}")
    cols[4].metric("Retrainings", int(df["retrained"].sum()))

    st.subheader("Decision value over time")
    st.line_chart(
        df.set_index("step")[["oracle_value_per_customer", "policy_value_estimate_per_customer"]]
    )
    st.caption(
        "Oracle value is available only in the simulator. A real production monitor sees the estimated policy value, not the counterfactual ground truth."
    )

with monitoring:
    st.subheader("Signals that can disagree")
    available = [c for c in ["feature_drift", "brier", "cate_shift"] if c in df.columns]
    st.line_chart(df.set_index("step")[available])

    st.subheader("Retraining events")
    event_cols = [
        "step",
        "feature_drift",
        "brier",
        "cate_shift",
        "policy_value_estimate_per_customer",
    ]
    event_cols = [c for c in event_cols if c in df.columns]
    events = df.loc[df["retrained"], event_cols]
    if events.empty:
        st.write("No retraining events in this run.")
    else:
        st.dataframe(events, use_container_width=True)

with decision_map:
    st.subheader("How evidence becomes a retraining decision")
    st.graphviz_chart(
        """
        digraph causalguard {
          rankdir=TB;
          Customer [shape=box];
          Prediction [label="Outcome / churn model", shape=box];
          CATE [label="Treatment-effect model", shape=box];
          Value [label="Customer value + treatment cost", shape=box];
          Policy [label="Treatment policy", shape=box];
          Outcome [label="Observed outcomes", shape=box];
          Drift [label="Feature drift", shape=ellipse];
          Perf [label="Predictive performance", shape=ellipse];
          CATEShift [label="CATE shift", shape=ellipse];
          PolicyValue [label="Causal policy value", shape=ellipse];
          Retrain [label="Retrain?", shape=diamond];

          Customer -> Prediction;
          Customer -> CATE;
          CATE -> Policy;
          Value -> Policy;
          Prediction -> Policy;
          Policy -> Outcome;
          Customer -> Drift;
          Outcome -> Perf;
          Outcome -> CATEShift;
          Outcome -> PolicyValue;
          Drift -> Retrain;
          Perf -> Retrain;
          CATEShift -> Retrain;
          PolicyValue -> Retrain;
        }
        """,
        use_container_width=True,
    )
    st.markdown(
        "The research question is whether these monitoring signals lead to the same retraining decisions. They need not: input drift can be harmless, and treatment value can deteriorate while predictive metrics remain stable."
    )

with research_trace:
    st.subheader("Full experiment trace")
    st.dataframe(df, use_container_width=True)
