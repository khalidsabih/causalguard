# Orange Belgium validation protocol

Status: **frozen before final Orange evaluation**.

The Orange study is a real-data causal-targeting validation track. It is not a
temporal-drift or retraining validation of the CausalGuard simulator benchmark.

## Research question

On randomized real-world retention data, how do predictive-risk targeting and
causal uplift targeting compare in out-of-fold incremental retention under a
fixed targeting budget?

## Data

- OpenML dataset: 45580
- Rows: 11,896
- Features: 178 (160 numeric, 18 categorical)
- Treatment column: `t` -> `treatment`
- Outcome: retention, defined as `1 - y` where `y` is churn
- Randomized treatment fraction used for evaluation: empirical assignment rate
  9,010 / 11,896 = 0.7573974445
- No temporal ordering is used or inferred.

## Model and policy learning

- Logistic T-learner with no class weighting
- 5-fold stratified cross-fitting
- Every reported evaluation observation receives predictions from a model that
  was not trained on that observation
- Risk score: predicted untreated churn risk, `1 - p0`
- Uplift score: predicted retention effect, `p1 - p0`
- Targeting is capacity constrained within each held-out fold

## Policies

Primary policy comparison:

- predictive-risk targeting
- causal uplift targeting

Reference policies:

- expected random targeting at the same budget
- never treat
- treat all

`profit_uplift` is secondary/exploratory in the zero-cost Orange analysis. It
acts as an uplift policy with a non-positive-uplift gate and is not part of the
primary causal-vs-risk comparison. Economic cost sensitivity, if added, will be
labeled exploratory unless separately pre-specified.

## Budgets

- Primary budget: 25%
- Secondary sensitivity budgets: 10% and 50%

The 25% primary budget matches the pre-existing CausalGuard targeting setup and
was not selected from the Orange pilot result.

## Estimation

Primary estimator:

- cross-fitted doubly robust / AIPW incremental policy value per customer

Secondary estimator:

- randomized selected-subgroup difference in means

Diagnostic only:

- raw Horvitz-Thompson IPS

The random reference is evaluated as the expected stochastic policy that treats
each customer with probability equal to the budget. It is not based on one
arbitrary random subset.

## Primary contrast and uncertainty

Primary contrast:

- risk targeting minus uplift targeting at the 25% budget

Report the paired DR estimate, standard error, and two-sided 95% confidence
interval. Secondary contrasts against expected random targeting and the 10% /
50% budgets are descriptive sensitivity analyses.

## Cross-fitting randomness

- Seed 42 is pilot/exploratory only and is excluded from the final primary result.
- Primary frozen partition seed: 1000
- Partition-stability sensitivity seeds: 1001-1009

The sensitivity seeds reuse the same customers and therefore are **not
independent experimental replications**. They are used only to assess sensitivity
to fold assignment. Do not compute a confidence interval or p-value by treating
these seeds as independent observations.

## Interpretation safeguards

- Do not infer individual treatment effects from observed outcomes.
- Do not claim Orange validates temporal drift or retraining behavior.
- Do not declare one targeting family superior from the pilot seed.
- Keep real randomized-data evidence separate from simulator oracle evidence.
- Report null and negative results without retuning the protocol.
