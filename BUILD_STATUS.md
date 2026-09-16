# Build status

## Working now

- End-to-end causal production simulator
- Covariate, outcome, abrupt treatment-effect, gradual treatment-effect, and recurring treatment-effect drift
- T-learner baseline
- Random / risk / uplift / profit-aware policies
- Heterogeneous customer value and treatment cost
- Randomized exploration traffic
- Feature drift monitor
- Predictive Brier monitor
- CATE-shift proxy monitor
- IPS causal policy-value monitor
- Never / periodic / feature / predictive / CATE / policy-value retraining triggers
- Net-value accounting with retraining cost
- Repeated experiment matrix runner
- Bootstrap summary utility
- Orange Belgium OpenML adapter and real-data baseline runner
- Streamlit explanatory dashboard scaffold
- DVC stage, optional MLflow logging, Dockerfile, GitHub Actions CI
- 9 automated tests

## Intentionally not finished yet

These should be added only after the baseline experiment design is frozen:

- formal sequential CATE change-point detector from the literature;
- doubly robust policy-value monitor;
- confidence-aware retraining trigger;
- explicit delayed-label queue;
- exploration-fraction optimization;
- contextual-bandit adaptive baseline;
- larger causal learners / causal forests;
- full Orange/Hillstrom uncertainty analysis;
- publication-scale experiment grid;
- final dashboard and manuscript figures.

The repository is therefore a **working research scaffold**, not a finished paper or production service.
