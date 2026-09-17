from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from causalguard.data import load_orange_retention_frame


def _safe_float(value: float) -> float:
    return float(value) if pd.notna(value) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the Orange Belgium randomized retention dataset"
    )
    parser.add_argument("--data-id", type=int, default=45580)
    parser.add_argument(
        "--output",
        default="analysis/orange/data_audit.json",
    )
    args = parser.parse_args()

    frame, feature_names = load_orange_retention_frame(data_id=args.data_id)

    numeric = [
        column
        for column in feature_names
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical = [column for column in feature_names if column not in numeric]

    missing_counts = frame[feature_names].isna().sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0]

    treatment_rate = float(frame["treatment"].mean())
    retention_by_treatment = frame.groupby("treatment")["outcome"].mean()
    churn_by_treatment = 1.0 - retention_by_treatment

    control_retention = _safe_float(retention_by_treatment.get(0, float("nan")))
    treated_retention = _safe_float(retention_by_treatment.get(1, float("nan")))
    control_churn = _safe_float(churn_by_treatment.get(0, float("nan")))
    treated_churn = _safe_float(churn_by_treatment.get(1, float("nan")))

    possible_time_columns = [
        column
        for column in feature_names
        if any(token in column.lower() for token in ("date", "time", "month", "campaign"))
    ]

    audit = {
        "data_id": args.data_id,
        "rows": int(len(frame)),
        "feature_count": int(len(feature_names)),
        "numeric_feature_count": int(len(numeric)),
        "categorical_feature_count": int(len(categorical)),
        "treatment_counts": {
            str(key): int(value)
            for key, value in frame["treatment"].value_counts().sort_index().items()
        },
        "treatment_rate": treatment_rate,
        "control_rate": 1.0 - treatment_rate,
        "retention_rate_overall": float(frame["outcome"].mean()),
        "retention_rate_control": control_retention,
        "retention_rate_treated": treated_retention,
        "retention_ate_treated_minus_control": treated_retention - control_retention,
        "churn_rate_control": control_churn,
        "churn_rate_treated": treated_churn,
        "churn_reduction_control_minus_treated": control_churn - treated_churn,
        "features_with_missing": int(len(missing_counts)),
        "missing_cells": int(frame[feature_names].isna().sum().sum()),
        "top_missing_features": [
            {
                "feature": column,
                "missing_count": int(count),
                "missing_rate": float(count / len(frame)),
            }
            for column, count in missing_counts.head(15).items()
        ],
        "possible_time_or_campaign_columns": possible_time_columns,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps(audit, indent=2))
    print(f"\nSaved audit to {output}")


if __name__ == "__main__":
    main()
