from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_line(df: pd.DataFrame, x: str, ys: list[str], title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for y in ys:
        if y in df.columns:
            ax.plot(df[x], df[y], label=y)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create simple paper-ready diagnostic figures")
    parser.add_argument("--input", default="experiments/latest/results.csv")
    parser.add_argument("--output-dir", default="reports/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_line(
        df,
        "step",
        ["oracle_value_per_customer", "policy_value_estimate_per_customer"],
        "True versus estimated incremental policy value",
        out / "policy_value_timeline.png",
    )
    save_line(
        df,
        "step",
        ["feature_drift", "cate_shift"],
        "Distribution and treatment-effect monitoring signals",
        out / "monitoring_signals.png",
    )
    save_line(
        df,
        "step",
        ["cumulative_net_value"],
        "Cumulative net value",
        out / "cumulative_net_value.png",
    )


if __name__ == "__main__":
    main()
