from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path(
    "analysis/main_trigger_benchmark/"
    "mechanism_trajectories.csv"
)

OUTPUT = Path(
    "analysis/main_trigger_benchmark/figures"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

df = pd.read_csv(INPUT)


CASES = [
    {
        "drift": "covariate",
        "trigger": "feature_drift",
        "title": (
            "Covariate drift: detection triggers harmful retraining"
        ),
        "filename": "mechanism_covariate",
        "change_steps": [14],
    },
    {
        "drift": "treatment",
        "trigger": "policy_value",
        "title": (
            "Persistent treatment-effect drift: "
            "causal adaptation creates value"
        ),
        "filename": "mechanism_treatment",
        "change_steps": [14],
    },
    {
        "drift": "recurring_treatment",
        "trigger": "policy_value",
        "title": (
            "Recurring treatment-effect drift: "
            "adaptation becomes out of phase"
        ),
        "filename": "mechanism_recurring_treatment",
        "change_steps": [14, 18, 22],
    },
]


def add_change_lines(
    ax,
    steps,
):
    for step in steps:
        ax.axvline(
            step,
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )


for case in CASES:
    data = df[
        (df["drift"] == case["drift"])
        & (df["trigger"] == case["trigger"])
    ].copy()

    if data.empty:
        raise ValueError(
            f"No data for {case['drift']} / "
            f"{case['trigger']}"
        )

    data = data.sort_values("step")

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 12),
        sharex=True,
    )

    # ========================================================
    # Panel 1 — true CATE and predicted uplift
    # ========================================================

    ax = axes[0]

    ax.plot(
        data["step"],
        data["true_cate_oracle"],
        label="True mean CATE (oracle)",
        linewidth=2,
    )

    ax.plot(
        data["step"],
        data["candidate_predicted_uplift"],
        label="Retraining strategy: predicted uplift",
        linewidth=2,
    )

    ax.plot(
        data["step"],
        data["never_predicted_uplift"],
        label="Never retrain: predicted uplift",
        linewidth=2,
        linestyle=":",
    )

    ax.axhline(
        0,
        linewidth=1,
        alpha=0.5,
    )

    add_change_lines(
        ax,
        case["change_steps"],
    )

    ax.set_ylabel(
        "Treatment effect"
    )

    ax.set_title(
        case["title"],
        fontsize=14,
        pad=12,
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    ax.grid(
        alpha=0.2
    )

    # ========================================================
    # Panel 2 — targeting behavior
    # ========================================================

    ax = axes[1]

    ax.plot(
        data["step"],
        data["candidate_treatment_rate"],
        label="Retraining strategy",
        linewidth=2,
    )

    ax.plot(
        data["step"],
        data["never_treatment_rate"],
        label="Never retrain",
        linewidth=2,
        linestyle=":",
    )

    add_change_lines(
        ax,
        case["change_steps"],
    )

    ax.set_ylabel(
        "Policy treatment rate"
    )

    ax.set_ylim(
        bottom=0
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    ax.grid(
        alpha=0.2
    )

    # ========================================================
    # Panel 3 — retraining activity
    # ========================================================

    ax = axes[2]

    ax.plot(
        data["step"],
        data["request_rate"],
        label="Request rate",
        linewidth=2,
    )

    ax.plot(
        data["step"],
        data["retrain_rate"],
        label="Retrain rate",
        linewidth=2,
        linestyle="--",
    )

    add_change_lines(
        ax,
        case["change_steps"],
    )

    ax.set_ylabel(
        "Fraction of runs"
    )

    ax.set_ylim(
        -0.03,
        1.03,
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    ax.grid(
        alpha=0.2
    )

    # ========================================================
    # Panel 4 — economic consequence
    # ========================================================

    ax = axes[3]

    ax.plot(
        data["step"],
        data["decision_value_delta"],
        label=(
            "Decision-value difference "
            "(excluding retraining fee)"
        ),
        linewidth=2,
    )

    ax.plot(
        data["step"],
        data["net_value_delta"],
        label="Net-value difference",
        linewidth=2,
        linestyle="--",
    )

    ax.axhline(
        0,
        linewidth=1,
        alpha=0.6,
    )

    add_change_lines(
        ax,
        case["change_steps"],
    )

    ax.set_ylabel(
        "Value vs never (€ / step)"
    )

    ax.set_xlabel(
        "Production step"
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    ax.grid(
        alpha=0.2
    )

    # ========================================================
    # Figure formatting
    # ========================================================

    fig.tight_layout()

    png_path = (
        OUTPUT
        / f"{case['filename']}.png"
    )

    pdf_path = (
        OUTPUT
        / f"{case['filename']}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "saved:",
        png_path,
    )

    print(
        "saved:",
        pdf_path,
    )


print()
print(
    "All mechanism figures complete."
)
