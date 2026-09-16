# CausalGuard research protocol - draft 0.1

## Primary research question

How do distribution-based, prediction-based, treatment-effect-based, and policy-value-based retraining triggers compare in preserving the net decision value of a causal targeting policy under temporal distribution shift?

## Planned hypotheses

H1. Large covariate drift does not necessarily imply material loss of causal policy value.

H2. Causal policy value can deteriorate without a proportionally large deterioration in conventional predictive metrics.

H3. Feature-drift-triggered retraining will cause unnecessary retraining in at least some drift regimes.

H4. Policy-value monitoring will detect at least some economically harmful treatment-effect shifts that feature-drift monitoring misses.

H5. No single retraining strategy will dominate across all drift regimes and cost assumptions.

H6. Persistent randomized exploration improves causal monitoring but produces an exploration-cost versus information trade-off.

These are hypotheses, not expected conclusions. Negative results are acceptable.

## Primary outcome

Cumulative incremental net value over the deployment horizon:

    incremental outcome value
  - treatment cost
  - retraining cost
  - exploration cost / opportunity cost

The first executable simulator currently includes outcome value, treatment cost, and retraining cost. A formal exploration-opportunity-cost term will be added before the main study.

## Secondary outcomes

- number of retrainings;
- false retraining alarms;
- harmful shifts missed;
- detection delay;
- time under a degraded policy;
- oracle policy regret (simulation only);
- predictive Brier score / calibration;
- feature-drift score;
- estimated causal policy value;
- compute/runtime cost.

## Drift regimes

1. no drift;
2. covariate shift only;
3. baseline-outcome concept drift;
4. abrupt treatment-effect drift;
5. gradual treatment-effect drift;
6. recurring treatment-effect drift.

## Retraining strategies

1. never;
2. periodic;
3. feature-drift-triggered;
4. predictive-performance-triggered;
5. CATE-change-triggered (planned);
6. policy-value-triggered;
7. hybrid (planned);
8. adaptive contextual bandit (advanced benchmark, planned).

## Causal data collection

The deployment simulation maintains randomized exploration traffic. Causal retraining and policy-value estimation should use data whose treatment propensities are known. Policy-generated treatment logs are not silently treated as randomized data.

## Uncertainty plan

The final study should use:

- multiple random seeds;
- repeated experiment replications;
- bootstrap or simulation-based confidence intervals;
- sensitivity analysis over treatment cost, outcome value, retraining cost, exploration fraction, drift strength, and label delay;
- effect sizes in addition to significance tests.

## Real-data role

Orange Belgium and/or Hillstrom establish real-world behavior and estimator difficulty. They do not provide ground truth for temporal drift mechanisms. Simulation conclusions must be described as simulation evidence, not claims about Orange customers.

## Reproducibility

Every final experiment should store:

- Git commit;
- configuration;
- seed;
- model/policy definition;
- monitoring rule;
- retraining rule;
- software environment;
- result table.
