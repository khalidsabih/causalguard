from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from causalguard.data import build_logistic_pipeline, load_orange_retention_frame
from causalguard.models import TLearner
from causalguard.monitoring import ips_incremental_policy_value_per_customer
from causalguard.policy import top_fraction


def evaluate_policy(test: pd.DataFrame, action: np.ndarray, name: str) -> dict:
    work = test[["outcome", "treatment"]].copy()
    work["policy_action"] = np.asarray(action, dtype=int)
    # Randomization in the benchmark is approximately balanced; this first
    # baseline uses 0.5. A publication version should estimate/verify the known
    # assignment propensity from the benchmark documentation/data.
    value = ips_incremental_policy_value_per_customer(
        work,
        outcome_value=1.0,
        treatment_cost=0.0,
        propensity=0.5,
    )
    return {
        "policy": name,
        "treatment_rate": float(np.mean(action)),
        "incremental_retention_per_customer_ips": value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Orange Belgium risk-vs-uplift baseline")
    parser.add_argument("--output", default="experiments/orange/policy_baseline.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-fraction", type=float, default=0.25)
    args = parser.parse_args()

    frame, features = load_orange_retention_frame()
    stratify = frame["treatment"].astype(str) + "_" + frame["outcome"].astype(str)
    train, test = train_test_split(
        frame,
        test_size=0.30,
        random_state=args.seed,
        stratify=stratify,
    )

    estimator = build_logistic_pipeline(train, features)
    model = TLearner(base_estimator=estimator).fit(train, features)
    pred = model.predict(test)

    rng = np.random.default_rng(args.seed)
    policies = {
        "random": top_fraction(rng.random(len(test)), args.budget_fraction),
        "risk": top_fraction(1.0 - pred.p0, args.budget_fraction),
        "uplift": top_fraction(pred.uplift, args.budget_fraction),
    }
    rows = [evaluate_policy(test, action, name) for name, action in policies.items()]
    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
