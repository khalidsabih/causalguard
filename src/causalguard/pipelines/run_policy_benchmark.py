from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from causalguard.pipelines.run_experiment import load_config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare simple treatment policies")
    parser.add_argument("--config", default="configs/experiments/base.yaml")
    parser.add_argument("--output", default="experiments/policy_benchmark.csv")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--drift-type", default="none")
    args = parser.parse_args()

    base = load_config(args.config)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for seed in seeds:
        for policy in ["random", "risk", "uplift", "profit"]:
            config = dict(base)
            config.update({"seed": seed, "policy": policy, "drift_type": args.drift_type, "trigger": "never"})
            _, summary = run_experiment(config)
            rows.append(summary)

    results = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results.groupby("policy")["final_cumulative_net_value"].agg(["mean", "std"]).round(1))


if __name__ == "__main__":
    main()
