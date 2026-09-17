from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from causalguard.data import build_logistic_pipeline, load_orange_retention_frame
from causalguard.evaluation import (
    dr_incremental_policy_value_with_ci,
    paired_dr_policy_difference_with_ci,
    rct_incremental_policy_value_with_ci,
)
from causalguard.models import TLearner
from causalguard.monitoring import ips_incremental_policy_value_with_ci
from causalguard.policy import profit_score, top_fraction


def parse_budgets(value: str) -> list[float]:
    budgets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets:
        raise ValueError("At least one budget fraction is required")
    if any(not 0 < budget <= 1 for budget in budgets):
        raise ValueError("Budget fractions must be in (0, 1]")
    return budgets


def foldwise_top_fraction(
    score: np.ndarray,
    fold_ids: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Select the top fraction separately inside each held-out fold.

    The foldwise threshold prevents an evaluation observation from influencing
    its own action indirectly through scores for observations in other folds.
    """
    score = np.asarray(score, dtype=float)
    fold_ids = np.asarray(fold_ids)

    if len(score) != len(fold_ids):
        raise ValueError("score and fold_ids must have the same length")

    action = np.zeros(len(score), dtype=int)
    for fold_id in np.unique(fold_ids):
        mask = fold_ids == fold_id
        action[mask] = top_fraction(score[mask], fraction)
    return action


def cross_fit_predictions(
    frame: pd.DataFrame,
    features: list[str],
    *,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate out-of-fold T-learner predictions for every observation."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    stratify = frame["treatment"].astype(str) + "_" + frame["outcome"].astype(str)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    p0 = np.full(len(frame), np.nan, dtype=float)
    p1 = np.full(len(frame), np.nan, dtype=float)
    uplift = np.full(len(frame), np.nan, dtype=float)
    fold_ids = np.full(len(frame), -1, dtype=int)

    for fold_id, (train_idx, eval_idx) in enumerate(splitter.split(frame, stratify)):
        train = frame.iloc[train_idx]
        held_out = frame.iloc[eval_idx]

        estimator = build_logistic_pipeline(train, features)
        model = TLearner(base_estimator=estimator).fit(train, features)
        prediction = model.predict(held_out)

        p0[eval_idx] = prediction.p0
        p1[eval_idx] = prediction.p1
        uplift[eval_idx] = prediction.uplift
        fold_ids[eval_idx] = fold_id

    if (
        np.isnan(p0).any()
        or np.isnan(p1).any()
        or np.isnan(uplift).any()
        or np.any(fold_ids < 0)
    ):
        raise RuntimeError("Cross-fitting did not produce predictions for every row")

    return p0, p1, uplift, fold_ids


def evaluate_policy(
    frame: pd.DataFrame,
    action: np.ndarray,
    name: str,
    *,
    p0: np.ndarray,
    p1: np.ndarray,
    budget_fraction: float,
    propensity: float,
    outcome_value: float,
    treatment_cost: float,
    seed: int,
    n_splits: int,
) -> dict:
    policy_weight = np.asarray(action, dtype=float)
    is_binary_policy = np.all(np.isin(policy_weight, [0.0, 1.0]))

    dr = dr_incremental_policy_value_with_ci(
        outcome=frame["outcome"].to_numpy(dtype=float),
        treatment=frame["treatment"].to_numpy(dtype=int),
        policy_action=policy_weight,
        p0=p0,
        p1=p1,
        propensity=propensity,
        outcome_value=outcome_value,
        treatment_cost=treatment_cost,
        confidence_level=0.95,
    )

    if is_binary_policy:
        work = frame[["outcome", "treatment"]].copy()
        work["policy_action"] = policy_weight.astype(int)

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
        rct_estimate = rct.estimate
        rct_standard_error = rct.standard_error
        rct_lower = rct.lower
        rct_upper = rct.upper
        selected_n = rct.selected_n
        selected_treated_n = rct.treated_selected_n
        selected_control_n = rct.control_selected_n
        ips_estimate = ips.estimate
        ips_standard_error = ips.standard_error
        n_evaluation = rct.n
    else:
        if not np.allclose(policy_weight, policy_weight[0]):
            raise ValueError(
                "Non-binary policies are supported only for a constant "
                "stochastic treatment probability"
            )
        random_probability = float(policy_weight[0])
        all_work = frame[["outcome", "treatment"]].copy()
        all_work["policy_action"] = 1

        rct_all = rct_incremental_policy_value_with_ci(
            randomized_frame=all_work,
            outcome_value=outcome_value,
            treatment_cost=treatment_cost,
            confidence_level=0.95,
        )
        ips_all = ips_incremental_policy_value_with_ci(
            randomized_frame=all_work,
            outcome_value=outcome_value,
            treatment_cost=treatment_cost,
            propensity=propensity,
            confidence_level=0.95,
        )
        rct_estimate = random_probability * rct_all.estimate
        rct_standard_error = random_probability * rct_all.standard_error
        rct_lower = random_probability * rct_all.lower
        rct_upper = random_probability * rct_all.upper
        selected_n = float("nan")
        selected_treated_n = float("nan")
        selected_control_n = float("nan")
        ips_estimate = random_probability * ips_all.estimate
        ips_standard_error = random_probability * ips_all.standard_error
        n_evaluation = rct_all.n

    return {
        "crossfit_seed": seed,
        "n_splits": n_splits,
        "budget_fraction": budget_fraction,
        "policy": name,
        "treatment_rate": float(np.mean(policy_weight)),
        "dr_incremental_value_per_customer": dr.estimate,
        "dr_standard_error": dr.standard_error,
        "dr_ci95_low": dr.lower,
        "dr_ci95_high": dr.upper,
        "rct_incremental_value_per_customer": rct_estimate,
        "rct_standard_error": rct_standard_error,
        "rct_ci95_low": rct_lower,
        "rct_ci95_high": rct_upper,
        "selected_n": selected_n,
        "selected_treated_n": selected_treated_n,
        "selected_control_n": selected_control_n,
        "ips_incremental_value_per_customer": ips_estimate,
        "ips_standard_error": ips_standard_error,
        "propensity": propensity,
        "n_evaluation": n_evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orange Belgium fold-safe cross-fitted targeting evaluation"
    )
    parser.add_argument(
        "--output",
        default="experiments/orange/crossfit_smoke.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--budgets", default="0.10,0.25,0.50")
    parser.add_argument("--propensity", type=float, default=None)
    parser.add_argument("--outcome-value", type=float, default=1.0)
    parser.add_argument("--treatment-cost", type=float, default=0.0)
    args = parser.parse_args()

    budgets = parse_budgets(args.budgets)
    frame, features = load_orange_retention_frame()

    propensity = (
        float(args.propensity)
        if args.propensity is not None
        else float(frame["treatment"].mean())
    )
    if not 0 < propensity < 1:
        raise ValueError("propensity must be in (0, 1)")

    p0, p1, uplift, fold_ids = cross_fit_predictions(
        frame,
        features,
        n_splits=args.folds,
        seed=args.seed,
    )

    risk_score = 1.0 - p0
    profit = profit_score(
        uplift,
        outcome_value=args.outcome_value,
        treatment_cost=args.treatment_cost,
    )

    rows: list[dict] = []
    paired_rows: list[dict] = []
    for budget in budgets:
        # Expected random targeting: every customer is treated with probability
        # equal to the budget. This removes unnecessary Monte Carlo noise from
        # drawing one arbitrary random subset.
        random_action = np.full(len(frame), budget, dtype=float)
        risk_action = foldwise_top_fraction(risk_score, fold_ids, budget)
        uplift_action = foldwise_top_fraction(uplift, fold_ids, budget)
        profit_action = foldwise_top_fraction(profit, fold_ids, budget)
        profit_action = np.where(profit > 0, profit_action, 0).astype(int)

        policies = {
            "never": np.zeros(len(frame), dtype=int),
            "random": random_action,
            "risk": risk_action,
            "uplift": uplift_action,
            "profit_uplift": profit_action,
            "treat_all": np.ones(len(frame), dtype=int),
        }

        for name, action in policies.items():
            rows.append(
                evaluate_policy(
                    frame,
                    action,
                    name,
                    p0=p0,
                    p1=p1,
                    budget_fraction=budget,
                    propensity=propensity,
                    outcome_value=args.outcome_value,
                    treatment_cost=args.treatment_cost,
                    seed=args.seed,
                    n_splits=args.folds,
                )
            )

        for first_name, second_name in [
            ("risk", "uplift"),
            ("uplift", "random"),
            ("risk", "random"),
        ]:
            difference = paired_dr_policy_difference_with_ci(
                outcome=frame["outcome"].to_numpy(dtype=float),
                treatment=frame["treatment"].to_numpy(dtype=int),
                first_action=policies[first_name],
                second_action=policies[second_name],
                p0=p0,
                p1=p1,
                propensity=propensity,
                outcome_value=args.outcome_value,
                treatment_cost=args.treatment_cost,
                confidence_level=0.95,
            )
            paired_rows.append(
                {
                    "crossfit_seed": args.seed,
                    "n_splits": args.folds,
                    "budget_fraction": budget,
                    "first_policy": first_name,
                    "second_policy": second_name,
                    "dr_difference_per_customer": difference.estimate,
                    "dr_standard_error": difference.standard_error,
                    "dr_ci95_low": difference.lower,
                    "dr_ci95_high": difference.upper,
                    "n_evaluation": difference.n,
                }
            )

    out = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    paired_output = output.with_name(f"{output.stem}_paired.csv")
    paired.to_csv(paired_output, index=False)

    uplift_quantiles = np.quantile(uplift, [0.05, 0.25, 0.50, 0.75, 0.95])
    control = frame["treatment"].to_numpy(dtype=int) == 0
    treated = ~control
    outcome = frame["outcome"].to_numpy(dtype=float)

    print(
        f"rows={len(frame):,} features={len(features)} folds={args.folds} "
        f"seed={args.seed} treatment_rate={frame['treatment'].mean():.4f}"
    )
    print(
        "OOF predictions: "
        f"mean_p0={p0.mean():.6f} mean_p1={p1.mean():.6f} "
        f"mean_uplift={uplift.mean():.6f} positive_uplift_share={np.mean(uplift > 0):.4f}"
    )
    print(
        "Observed-arm calibration: "
        f"control observed={outcome[control].mean():.6f} "
        f"mean_p0={p0[control].mean():.6f} "
        f"gap={p0[control].mean() - outcome[control].mean():+.6f} "
        f"brier={brier_score_loss(outcome[control], p0[control]):.6f}; "
        f"treated observed={outcome[treated].mean():.6f} "
        f"mean_p1={p1[treated].mean():.6f} "
        f"gap={p1[treated].mean() - outcome[treated].mean():+.6f} "
        f"brier={brier_score_loss(outcome[treated], p1[treated]):.6f}"
    )
    print(
        "OOF uplift quantiles (5/25/50/75/95%): "
        + ", ".join(f"{value:.6f}" for value in uplift_quantiles)
    )

    display_columns = [
        "budget_fraction",
        "policy",
        "treatment_rate",
        "dr_incremental_value_per_customer",
        "dr_standard_error",
        "dr_ci95_low",
        "dr_ci95_high",
        "rct_incremental_value_per_customer",
        "rct_standard_error",
    ]
    print("\nPolicy value (DR primary; RCT difference-in-means secondary):")
    print(out[display_columns].to_string(index=False))

    paired_columns = [
        "budget_fraction",
        "first_policy",
        "second_policy",
        "dr_difference_per_customer",
        "dr_standard_error",
        "dr_ci95_low",
        "dr_ci95_high",
    ]
    print("\nPaired DR policy differences (first - second):")
    print(paired[paired_columns].to_string(index=False))
    print(f"\nSaved cross-fitted smoke results to {output}")
    print(f"Saved paired differences to {paired_output}")


if __name__ == "__main__":
    main()
