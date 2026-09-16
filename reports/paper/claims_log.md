# Claims log

Use this file during the project to prevent accidental overclaiming.

| Claim type | Allowed evidence | Example wording |
|---|---|---|
| Real-data descriptive result | Orange/Hillstrom analysis | "In the Orange holdout, the estimated policy value was..." |
| Simulation mechanism | Simulator definition | "We introduced treatment-effect drift by changing..." |
| Simulation performance result | Repeated controlled experiments | "Under the simulated abrupt-treatment-shift regime..." |
| Causal claim about real customers | Randomized real data + valid estimator | "The randomized campaign supports an average treatment-effect estimate of..." |
| Individual counterfactual claim | Not identifiable from ordinary RCT data | Do not say "customer X was saved by treatment" |
| General industry claim | External literature | Cite source and scope |
| Novelty claim | Systematic literature search | Avoid "first" until verified |

## Simulator-only quantities

The following are privileged oracle quantities and must never be presented as if observable in production:

- `p0_true`;
- `p1_true`;
- `cate_true`;
- oracle incremental value;
- oracle policy regret (when added).

These are used to evaluate monitoring systems, not as inputs to deployable triggers.
