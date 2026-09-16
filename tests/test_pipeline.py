from causalguard.pipelines.run_experiment import run_experiment


def test_small_experiment_runs_end_to_end():
    config = {
        "seed": 4,
        "n_steps": 9,
        "batch_size": 250,
        "n_features": 5,
        "initial_train_steps": 3,
        "budget_fraction": 0.25,
        "exploration_rate": 0.15,
        "outcome_value": 200.0,
        "treatment_cost": 5.0,
        "retraining_cost": 50.0,
        "drift_type": "treatment",
        "drift_start": 5,
        "drift_strength": 0.8,
        "drift_duration": 3,
        "trigger": "periodic",
        "periodic_every": 2,
        "monitor_window_steps": 2,
        "max_train_rows": 3000,
    }
    results, summary = run_experiment(config)
    assert len(results) == 6
    assert "cumulative_net_value" in results.columns
    assert summary["steps_evaluated"] == 6
