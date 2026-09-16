from __future__ import annotations

from pathlib import Path


def log_experiment_if_enabled(config: dict, summary: dict, output_dir: str | Path) -> None:
    """Optionally log a completed experiment to MLflow.

    MLflow is deliberately optional. The experiment remains fully runnable
    without it; tracking is infrastructure around the research, not a runtime
    requirement of the simulator.
    """
    if not bool(config.get("mlflow_enabled", False)):
        return
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow logging was enabled but mlflow is not installed. "
            "Install with: pip install -e '.[mlops]'"
        ) from exc

    experiment_name = str(config.get("mlflow_experiment", "CausalGuard"))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        safe_params = {
            key: value
            for key, value in config.items()
            if isinstance(value, (str, int, float, bool))
        }
        mlflow.log_params(safe_params)
        numeric_summary = {
            key: float(value)
            for key, value in summary.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        mlflow.log_metrics(numeric_summary)
        output = Path(output_dir)
        for name in ["results.csv", "summary.json"]:
            path = output / name
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="experiment")
