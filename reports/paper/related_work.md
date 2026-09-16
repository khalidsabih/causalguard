# Related-work map

This note is a working map for the eventual article, not a finished literature review.

## Uplift and causal targeting

### Gutierrez & Gerardy (2017) - Causal Inference and Uplift Modelling: A Review
Core lesson: uplift modeling is a causal-inference problem. Prediction of an outcome and prediction of the effect of an intervention answer different questions.

Link: https://proceedings.mlr.press/v67/gutierrez17a.html

### De Caigny et al. - churn prediction versus uplift for retention
Core lesson: in at least some real retention settings, treatment-aware targeting can improve campaign profitability over ordinary churn targeting. This is application evidence, not a universal guarantee.

Link: https://www.sciencedirect.com/science/article/pii/S0020025519312022

### Verhelst et al. - Uplift vs. predictive modeling: a theoretical analysis
Core lesson: predictive targeting can outperform uplift modeling in some regimes. Estimation variance, treatment-effect heterogeneity, feature information, sample size, and economics all matter.

Link: https://arxiv.org/abs/2309.12036

### Orange Belgium churn-uplift benchmark
Core lesson: real churn-uplift data are difficult; small treatment effects and limited sample size make uncertainty and estimator stability central issues.

Link: https://arxiv.org/abs/2312.07206

## Evaluation and uncertainty

### Bokelmann & Lessmann - Improving uplift model evaluation on randomized controlled trial data
Core lesson: common uplift evaluation metrics can have high variance. Small differences between Qini/AUUC values should not automatically be interpreted as meaningful model superiority.

Link: https://www.sciencedirect.com/science/article/pii/S037722172300721X

## Dynamic treatment effects

### Berrevoets, Verboven & Verbeke - Treatment effect optimisation in dynamic environments
Core lesson: static uplift policies can become stale as environments change; adaptive contextual-bandit approaches are one way to handle nonstationarity.

Link: https://www.degruyter.com/document/doi/10.1515/jci-2020-0009/html

### Madrid Padilla & Yu - sequential change detection for CATE
Core lesson: abrupt changes in conditional treatment effects can be treated as a sequential change-point-detection problem. CausalGuard should eventually benchmark a formal method from this literature rather than rely only on its simple CATE-shift proxy.

Link: https://arxiv.org/abs/2206.09092

## Decision quality under shift

### Fischer-Abaigar, Kern & Kreuter - The Missing Link: Allocation Performance in Causal Machine Learning
Core lesson: CATE-prediction quality and downstream allocation quality are not interchangeable. Distribution shift can affect allocation differently depending on the decision rule and budget.

Link: https://arxiv.org/abs/2407.10779

### Ren, Byun & Wilder - decision-focused distribution shift
Core lesson: shifts that are worst for prediction accuracy can differ from shifts that are worst for downstream decision quality. Monitoring should be aligned with the objective of the deployed decision system.

Link: https://proceedings.mlr.press/v244/ren24a.html

## Shift detection and retraining

### Rabanser, Guennemann & Lipton - Failing Loudly
Core lesson: statistical methods can detect dataset shift, but detecting that a distribution changed is different from establishing that the change harms a deployed model or decision policy.

Link: https://proceedings.neurips.cc/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html

### Regol et al. - When to retrain a machine learning model
Core lesson: retraining is a decision under uncertainty and cost. Shift detection alone does not answer whether updating is economically justified.

Link: https://proceedings.mlr.press/v267/regol25a.html

### Cost-aware retraining research
Core lesson: frequent retraining can waste resources while infrequent retraining can leave a stale model in production. Retraining cost belongs in the objective rather than being treated as zero.

Link: https://www.sciencedirect.com/science/article/pii/S0950705124002454

## Working gap

The literature contains strong work on each component separately:

- uplift/CATE estimation;
- treatment effects under nonstationarity;
- CATE change detection;
- policy evaluation under shift;
- downstream allocation under shift;
- generic shift detection;
- cost-aware model retraining.

The working CausalGuard question is narrower:

> How do distribution-based, prediction-based, treatment-effect-based, and policy-value-based retraining triggers compare in preserving the net decision value of a causal targeting policy under temporal distribution shift?

This should be treated as a *candidate gap*, not a first-of-its-kind claim, until a formal systematic literature search is completed.
