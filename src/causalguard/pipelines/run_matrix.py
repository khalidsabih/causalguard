from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from causalguard.pipelines.run_experiment import load_config, run_experiment


def run_matrix(base_config: dict, seeds: list[int]) -> pd.DataFrame:
    drift_types = ["none", "covariate", "outcome", "treatment", "gradual_treatment"]
    triggers = ["never", "periodic", "feature_drift", "performance", "cate_shift", "policy_value"]
    rows = []
    for seed in seeds:
        for drift_type in drift_types:
            for trigger in triggers:
                config = dict(base_config)
                config.update({"seed": seed, "drift_type": drift_type, "trigger": trigger})
                _, summary = run_experiment(config)
                rows.append(summary)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small CausalGuard experiment matrix")
    parser.add_argument("--config", default="configs/experiments/base.yaml")
    parser.add_argument("--output", default="experiments/matrix_summary.csv")
    parser.add_argument("--seeds", default="1,2,3")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    results = run_matrix(config, seeds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results.groupby(["drift_type", "trigger"])["final_cumulative_net_value"].mean().unstack())


if __name__ == "__main__":
    main()
