# CausalGuard

**Monitoring and retraining causal targeting policies under distribution shift.**

CausalGuard is a research-first Causal ML + MLOps project for studying a production question that ordinary model monitoring does not answer:

> **When should a causal targeting policy be retrained if the real objective is preserving decision value rather than predictive accuracy?**

The project compares feature-distribution, predictive-performance, treatment-effect, and policy-value signals under controlled temporal shift. It evaluates not only whether a monitor reacts, but whether the resulting retraining decision improves the economics of the deployed treatment policy.

## Main result

The frozen benchmark supports a simple conclusion:

> **Detecting distribution shift is not equivalent to detecting a need to retrain.** Retraining creates value when a persistent shift damages the causal decision rule, but can destroy value when the shift is decision-irrelevant or when adaptation lags a recurring environment.

Three mechanisms are especially important:

- **Covariate drift:** the feature monitor detects the shift, but retraining reduces decision value even though the treatment rate remains unchanged. The loss is driven primarily by downstream policy quality, not the direct retraining fee.
- **Persistent treatment-effect drift:** causal retraining learns to suppress treatment after treatment effects become harmful and recovers substantial economic value.
- **Recurring treatment-effect drift:** the same adaptive behavior can become out of phase with the environment, so fast retraining can perform worse than leaving the policy stale.

## Frozen benchmark

The final held-out benchmark contains:

- **6 drift regimes**;
- **7 retraining strategies**;
- **30 untouched evaluation seeds** (`1000-1029`);
- **1,260 total runs**.

### Drift regimes

1. no drift;
2. covariate drift;
3. outcome drift;
4. abrupt treatment-effect drift;
5. gradual treatment-effect drift;
6. recurring treatment-effect drift.

### Retraining strategies

Primary benchmark strategies:

1. never retrain;
2. periodic retraining;
3. feature-drift trigger;
4. predictive-performance trigger;
5. CATE-shift trigger;
6. policy-value trigger.

Secondary sensitivity analysis:

7. confidence-aware policy-value trigger.

The confidence-aware strategy uses a pilot-informed 80% confidence setting and is therefore reported as a **secondary sensitivity**, not as a primary pre-specified benchmark strategy.

The frozen configuration is stored at:

```text
configs/experiments/trigger_benchmark_frozen.yaml
```

Important frozen settings include:

```yaml
feature_drift_threshold: 0.04406
performance_threshold: 0.2296
cate_shift_threshold: 0.2326
policy_value_threshold: 0.0
monitor_window_steps: 3
policy_value_confidence_level: 0.80
retraining_data_strategy: rolling_window
retraining_window_steps: 5
retraining_cooldown_steps: 0
```

## Selected benchmark findings

Mean net value varied strongly by drift regime, so the project should not be interpreted as a trigger leaderboard.

- Under **no drift**, never retraining achieved about **€220.7k** mean net value. Periodic retraining lost about **€37.4k** versus never retraining.
- Under **covariate drift**, feature-triggered retraining lost about **€30.3k** versus never retraining despite detecting the shift immediately.
- Under **abrupt treatment-effect drift**, policy-value retraining improved net value by about **€51.8k** versus never retraining.
- Under **gradual treatment-effect drift**, policy-value retraining improved net value by about **€23.8k** versus never retraining.
- Under **recurring treatment-effect drift**, policy-value retraining lost about **€37.4k** versus never retraining because the adapted policy lagged the changing regime.

Paired bootstrap intervals, stationary-action behavior, response delays, and full run-level summaries are available in `analysis/main_trigger_benchmark/`.

## Mechanism analysis

The final analysis separates direct retraining expense from downstream decision effects.

| Regime / strategy | Mean net delta vs never | Direct retraining cost | Decision-value delta vs never |
|---|---:|---:|---:|
| Covariate / feature drift | -€30,255 | €1,267 | -€28,988 |
| Treatment / policy value | +€51,825 | €3,200 | +€55,025 |
| Recurring treatment / policy value | -€37,432 | €2,450 | -€34,982 |

This decomposition shows that the major gains and losses are mostly caused by **policy adaptation**, not by the nominal retraining fee itself.

Publication figures are stored in:

```text
analysis/main_trigger_benchmark/figures/
```

## Dashboard

The Streamlit dashboard is a read-only exploration layer over the frozen benchmark. It does **not** rerun, tune, or retrain models.

It contains:

- **Benchmark Overview** — mean net value, uncertainty, retraining burden, and paired comparisons;
- **Monitoring Behavior** — stationary actions, post-drift response rates, response delay, and request burden;
- **Mechanism Explorer** — covariate, persistent-treatment, and recurring-treatment mechanisms over time.

Install and run it with:

```bash
pip install -e '.[dashboard]'
python -m streamlit run dashboard/app.py
```

## Scientific design

CausalGuard separates three parts of the data-generating process:

- `P(X)`: customer characteristics;
- baseline outcome behavior;
- heterogeneous treatment effect `tau(X)`.

That separation allows controlled shifts in which input distributions change without treatment value changing, as well as shifts in which the treatment effect changes while conventional model-monitoring signals remain comparatively stable.

The production simulation also maintains **persistent randomized exploration traffic**. Causal monitoring and causal retraining use observations with known treatment propensities rather than silently treating policy-generated treatment assignments as randomized data.

### Economic objective

The primary target is cumulative incremental net value. Treatment cost is included in the per-customer incremental value calculation, and retraining cost is charged when a retraining event succeeds. Persistent randomized exploration can also reduce value endogenously because it overrides the policy action for part of the traffic.

The project therefore treats retraining as an economic decision rather than an automatic reaction to detected shift.

## Oracle vs deployable signals

The simulator exposes privileged counterfactual quantities such as true CATE and oracle incremental value. These are used **only for evaluation**.

They are never inputs to deployable monitoring or retraining triggers.

This distinction is central to the project: a monitoring strategy must act using signals that could plausibly exist in production, while the simulator can use oracle information afterward to determine whether those decisions were good.

## Repository layout

```text
causalguard/
├── analysis/
│   └── main_trigger_benchmark/   frozen aggregate results and figures
├── configs/
│   └── experiments/              experiment definitions and frozen config
├── dashboard/                    Streamlit results dashboard
├── data/                         local/external data locations
├── experiments/                  raw experiment outputs
├── notebooks/                    exploratory work only
├── reports/
│   └── paper/                    protocol, claims log, and related-work notes
├── scripts/                      benchmark analysis and figure generation
├── src/causalguard/
│   ├── data/                     external data loaders
│   ├── models/                   predictive and causal estimators
│   ├── policy/                   treatment allocation and value
│   ├── simulation/               causal DGP and drift regimes
│   ├── monitoring/               monitoring signals
│   ├── retraining/               retraining rules
│   ├── evaluation/               metrics and summaries
│   └── pipelines/                reproducible experiment runners
└── tests/
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,dashboard]'
```

Conda works equally well; the project was developed and validated with Python 3.11.

## Run a single experiment

```bash
causalguard-run \
  --config configs/experiments/base.yaml \
  --output experiments/latest
```

Override individual settings:

```bash
causalguard-run \
  --config configs/experiments/base.yaml \
  --output experiments/covariate_periodic \
  --set drift_type=covariate \
  --set trigger=periodic
```

## Reproduce the frozen analysis

With the final raw benchmark outputs present under `experiments/main_trigger_benchmark/`:

```bash
python scripts/analyze_main_benchmark.py
python scripts/analyze_mechanisms.py
python scripts/plot_main_mechanisms.py
```

These commands regenerate the aggregate CSVs and mechanism figures under `analysis/main_trigger_benchmark/`.

## Tests and code quality

Run:

```bash
ruff check src tests
pytest -q
```

Current audit state: **11 tests passing**.

GitHub Actions runs linting and the automated test suite on pushes and pull requests.

## Real-data role

The repository also includes an OpenML adapter for the Orange Belgium randomized retention benchmark. Real randomized data are useful for studying estimator behavior and the difficulty of causal targeting, but they do not provide oracle ground truth for temporal drift mechanisms.

For that reason, the main monitoring/retraining claims are explicitly framed as **controlled simulation evidence**, not as claims about real Orange customers.

## Research safeguards

The project follows several rules throughout the study:

- do not assume uplift targeting must outperform predictive targeting;
- do not assume sophisticated retraining triggers must beat periodic or no retraining;
- keep real-data evidence separate from simulator-oracle evidence;
- never use simulator oracle quantities as deployable monitoring signals;
- retain randomized exploration for causal monitoring and retraining;
- report paired uncertainty rather than relying on mean rankings alone;
- preserve simple baselines such as never and periodic retraining;
- freeze thresholds before final evaluation seeds;
- treat negative and null findings as scientifically useful;
- avoid first-of-its-kind novelty claims without a systematic literature review.

## Limitations and extensions

The current study intentionally uses transparent causal baselines and a controlled simulator. Important extensions include:

- sequentially valid causal monitoring / confidence sequences;
- doubly robust off-policy evaluation;
- formal CATE change-point methods;
- delayed-outcome modeling;
- exploration-rate sensitivity;
- broader cost and drift-strength sensitivity analysis;
- stronger causal learners;
- contextual-bandit adaptation;
- larger real-data uncertainty studies.

These are **extensions to the frozen benchmark**, not missing pieces required to interpret the reported main experiment.

## Related work and claims discipline

Working literature notes are in:

```text
reports/paper/related_work.md
```

The repository also maintains a claims log distinguishing simulation evidence, randomized real-data evidence, oracle quantities, and unsupported novelty claims:

```text
reports/paper/claims_log.md
```

## Status

The main CausalGuard experiment, final aggregate analysis, mechanism analysis, figures, tests, and dashboard are complete and frozen.

Remaining work is primarily publication/repository packaging: manuscript refinement, artifact selection, Git cleanup, and public release metadata.
