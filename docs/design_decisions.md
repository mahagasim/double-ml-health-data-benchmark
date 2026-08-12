# Design decisions and QA log

This file records substantive choices made **before the 200-replication primary Monte Carlo** and distinguishes scientific design changes from software-engineering changes.

## Final primary design

1. **Primary causal model:** partially linear regression (PLR).
2. **Primary target:** constant structural treatment effect \(\theta_0=1\), which is also the ATE in every DGP draw.
3. **Secondary causal model:** binary-treatment IRM/AIPW, retained to connect DML to doubly robust ATE estimation.
4. **Sample size:** \(N=1000\).
5. **Raw dimensionality:** \(p=10\) in Scenario A; \(p=400\) in B-D.
6. **Rich nuisance dimensionality:** \(p^*=5p-1=1999\) in B-D.
7. **DGP sequence:** low-dimensional linear → sparse many-control linear → nonlinear/interacting → same nonlinear DGP with weak overlap.
8. **Primary estimators:** difference in means, OLS, naive LASSO plug-in, unpenalized parametric IPW, unpenalized parametric AIPW, DML-PLR raw LASSO, DML-PLR rich-dictionary LASSO.
9. **Cross-fitting:** five outer folds.
10. **Monte Carlo size:** 200 replications per scenario, 800 datasets total.
11. **No silent trimming or clipping:** weak-overlap pathologies are part of the experiment.

## Why PLR is primary

The project originally considered IRM/AIPW as the main DML estimator. Because the DGP has a constant treatment effect and satisfies

\[
Y=D\theta_0+g_0(X)+\varepsilon,
\]

the PLR coefficient is exactly the ATE. PLR therefore gives a cleaner demonstration of residualization, orthogonalization, and cross-fitting while IRM remains available as a secondary extension.

## High-dimensional design

Early QA with \(p=200\) was computationally convenient but did not create a particularly demanding regularization problem. The final raw design uses \(p=400\) with only 20 signal variables. The rich nonlinear dictionary then expands this to 1,999 candidate nuisance features, exceeding the sample size.

The rich dictionary is deliberately deterministic and applies every transformation to every covariate. No true active-variable list is passed to the estimator.

## Learner evolution during QA

The initial primary flexible learner was Random Forest and later XGBoost. Smoke tests revealed two issues:

- forests underfit the sparse/high-dimensional nuisance structure under reasonable pre-specified settings;
- repeated XGBoost fits made the large Monte Carlo unnecessarily dependent on native-library runtime behavior in the available environment.

The DGP was **not simplified to make a tree learner win**. Instead, before the scientific 200-replication run, the flexible primary learner was changed to a rich nonlinear basis plus LASSO. This is methodologically closer to the high-dimensional econometric motivation of the project and makes the nuisance design explicitly \(p^*>N\).

XGBoost and Random Forest code are retained as optional sensitivity analyses rather than deleted.

## Rich dictionary

The final flexible nuisance dictionary contains, for all covariates:

- raw terms;
- centered squares;
- sine transforms;
- centered threshold indicators;
- adjacent interactions.

These transformation families are capable of representing the nonlinear terms in Scenarios C-D. This is intentional: Scenario C is a controlled test of what happens when the nuisance representation is adequate versus inadequate, not a leaderboard of arbitrary black-box algorithms.

## L1 penalty policy

Nested CV inside every outer fold was initially implemented. It was removed before the scientific run because it created substantial repeated computation without being central to the causal question.

Fixed penalties were selected on **separate calibration seeds 9101-9103** using only nuisance-prediction diagnostics. Treatment-effect bias/RMSE was not an input to the calibration criterion.

Frozen values:

- LASSO alpha = 0.05;
- L1-logistic C = 0.05.

The calibration grid and averages are committed under `results/`.

## Overlap calibration

Scenario D's treatment-index multiplier is **2.25**. In 20 fixed-seed calibration draws:

- mean treatment prevalence ≈ 0.488;
- true propensity outside [0.05,0.95] ≈ 15.9%;
- true propensity outside [0.10,0.90] ≈ 29.1%.

For Scenario C the corresponding [0.05,0.95] extreme share is only about 0.6%. The outcome equation is identical in C and D.

## Inference policy

- Difference in means and OLS use HC3 robust SEs.
- Naive LASSO plug-in is a point estimator only.
- Parametric IPW uses a plug-in influence-function SE and is interpreted primarily as an overlap/weighting stress benchmark.
- Parametric AIPW and DML use influence-function-based SEs.
- Coverage is always interpreted jointly with estimator failures and interval width.

This last point became empirically important: in B and D the unpenalized weighting estimators can report high coverage only because rare estimates/intervals become enormous.

## Final Monte Carlo execution

Long single-process runs exposed operating-system resource accumulation from repeated high-dimensional native numerical routines. This did not change any scientific estimate, but it made a single 800-dataset invocation unreliable in this container.

The final solution was to execute each high-dimensional scenario in **five independent blocks of 40 replications** and merge the raw CSVs. Batch seeds were fixed as:

- 20260811
- 20360814
- 20460817
- 20560820
- 20660823

Scenario A was rerun under the same batch-seed plan for consistent provenance.

The batch strategy changes only process lifetime. Estimators, DGPs, folds, penalties, and metric definitions are identical across blocks.

## Final primary-result status

The primary 800-dataset experiment is complete. QA/smoke outputs generated before the final design are not included in the frozen primary raw outputs or any primary figure.

Pre-specified sample-size, fold-count, Random Forest, and XGBoost sensitivities remain future extensions and must be reported separately if executed.

## Package validation status

Manual PLR and IRM validation hooks against `DoubleMLPLR`/`DoubleMLIRM` remain in the repository. `doubleml` was not available in the execution environment used for the primary run, so exact package agreement is not claimed.
