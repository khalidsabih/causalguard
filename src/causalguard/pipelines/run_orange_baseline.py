from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from causalguard.data import build_logistic_pipeline, load_orange_retention_frame
from causalguard.evaluation import rct_incremental_policy_value_with_ci
from causalguard.models import TLearner
from causalguard.monitoring import ips_incremental_policy_value_with_ci
from causalguard.policy import profit_score, top_fraction


def evaluate_policy(
    test: pd.DataFrame,
    action: np.ndarray,
    name: str,
    *,
    propensity: float,
    outcome_value: float,
    treatment_cost: float,
) -> dict:
    work = test[["outcome", "treatment"]].copy()
    work["policy_action"] = np.asarray(action, dtype=int)

    rct = rct_incremental_policy_value_with_ci(
        randomized_frame=work,
        outcome_value=outcome_value,
        treatment_cost=treatment_cost,
        confidence_level=0.95,
    )

    ips = ips_incremental_policy_value_with_ci(
        randomized_frame=work,
        outcome_value=outcome_value,
        treatment_cost=treatment_cost,
        propensity=propensity,
        confidence_level=0.95,
    )

    return {
        "policy": name,
        "treatment_rate": float(np.mean(action)),
        "rct_incremental_value_per_customer": rct.estimate,
        "rct_standard_error": rct.standard_error,
        "rct_ci95_low": rct.lower,
        "rct_ci95_high": rct.upper,
        "selected_n": rct.selected_n,
        "selected_treated_n": rct.treated_selected_n,
        "selected_control_n": rct.control_selected_n,
        "ips_incremental_value_per_customer": ips.estimate,
        "ips_standard_error": ips.standard_error,
        "ips_ci95_low": ips.lower,
        "ips_ci95_high": ips.upper,
        "propensity": propensity,
        "n_test": rct.n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orange Belgium real-data targeting smoke test"
    )
    parser.add_argument("--output", default="experiments/orange/policy_baseline.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-fraction", type=float, default=0.25)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--propensity", type=float, default=None)
    parser.add_argument("--outcome-value", type=float, default=1.0)
    parser.add_argument("--treatment-cost", type=float, default=0.0)
    args = parser.parse_args()

    frame, features = load_orange_retention_frame()

    propensity = (
        float(args.propensity)
        if args.propensity is not None
        else float(frame["treatment"].mean())
    )

    if not 0 < propensity < 1:
        raise ValueError("propensity must be in (0, 1)")

    stratify = frame["treatment"].astype(str) + "_" + frame["outcome"].astype(str)
    train, test = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    estimator = build_logistic_pipeline(train, features)
    model = TLearner(base_estimator=estimator).fit(train, features)
    pred = model.predict(test)

    rng = np.random.default_rng(args.seed)
    uplift_action = top_fraction(pred.uplift, args.budget_fraction)

    profit = profit_score(
        pred.uplift,
        outcome_value=args.outcome_value,
        treatment_cost=args.treatment_cost,
    )
    profit_action = top_fraction(profit, args.budget_fraction)
    profit_action = np.where(profit > 0, profit_action, 0).astype(int)

    policies = {
        "never": np.zeros(len(test), dtype=int),
        "random": top_fraction(rng.random(len(test)), args.budget_fraction),
        "risk": top_fraction(1.0 - pred.p0, args.budget_fraction),
        "uplift": uplift_action,
        "profit_uplift": profit_action,
        "treat_all": np.ones(len(test), dtype=int),
    }

    rows = [
        evaluate_policy(
            test,
            action,
            name,
            propensity=propensity,
            outcome_value=args.outcome_value,
            treatment_cost=args.treatment_cost,
        )
        for name, action in policies.items()
    ]

    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    display_columns = [
        "policy",
        "treatment_rate",
        "rct_incremental_value_per_customer",
        "rct_standard_error",
        "rct_ci95_low",
        "rct_ci95_high",
        "selected_treated_n",
        "selected_control_n",
        "ips_incremental_value_per_customer",
        "ips_standard_error",
    ]

    print(
        f"rows={len(frame):,} features={len(features)} "
        f"empirical_treatment_rate={frame['treatment'].mean():.4f} "
        f"evaluation_propensity={propensity:.4f}"
    )
    print(out[display_columns].to_string(index=False))
    print(f"\nSaved baseline to {output}")


if __name__ == "__main__":
    main()
