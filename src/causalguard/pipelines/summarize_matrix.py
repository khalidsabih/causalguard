from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from causalguard.evaluation import bootstrap_mean_ci


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize repeated experiment matrix with bootstrap CIs")
    parser.add_argument("--input", default="experiments/matrix_summary.csv")
    parser.add_argument("--output", default="reports/tables/matrix_with_ci.csv")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    rows = []
    group_cols = ["drift_type", "trigger"]
    if "policy" in df.columns:
        group_cols.append("policy")
    for keys, group in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mean, low, high = bootstrap_mean_ci(
            group["final_cumulative_net_value"], n_bootstrap=args.bootstrap
        )
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_runs": len(group),
                "mean_net_value": mean,
                "ci_low": low,
                "ci_high": high,
                "mean_retrainings": group["retrainings"].mean(),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
