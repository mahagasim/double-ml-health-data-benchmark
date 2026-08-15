# Design decisions and QA log

This file records substantive choices made **before the 200-replication primary Monte Carlo** and distinguishes scientific design changes from software-engineering changes.

## Final primary design

1. **Primary causal model:** partially linear regression (PLR).
2. **Primary target:** constant structural treatment effect \(\theta_0=1\), which is also the ATE in every DGP draw.
3. **Secondary causal model:** binary-treatment IRM/AIPW, retained to connect DML to doubly robust ATE estimation.
4. **Sample size:** \(N=1000\).
5. **Raw dimensionality:** \(p=10\) in Scenario A; \(p=400\) in B-D.
6. **Rich nuisance dimensionality:** \(p^*=5p-1=1999\) in B-D.
7. **DGP sequence:** low-dimensional linear → sparse many-control linear → nonlinear/interacting → same nonlinear outcome structure with weak overlap.
8. **Primary estimators:** difference in means, OLS, naive LASSO plug-in, unpenalized parametric IPW, unpenalized parametric AIPW, DML-PLR raw LASSO, DML-PLR rich-dictionary LASSO.
9. **Cross-fitting:** five outer folds.
10. **Monte Carlo size:** 200 reported replications per scenario, 800 datasets total.
11. **No silent trimming or clipping:** weak-overlap pathologies are part of the experiment.

## Why PLR is primary

The project originally considered IRM/AIPW as the main DML estimator. Because the DGP has a constant treatment effect and satisfies

\[
Y=D\theta_0+g_0(X)+\varepsilon,
\]

the PLR coefficient is exactly the ATE **in this simulation design**. PLR therefore gives a cleaner demonstration of residualization, orthogonalization, and cross-fitting while IRM remains available as a secondary extension. Under treatment-effect heterogeneity, the PLR coefficient would require separate interpretation and should not automatically be called the population ATE.

## High-dimensional design

Early QA with \(p=200\) was computationally convenient but did not create a particularly demanding regularization problem. The final raw design uses \(p=400\), while only the first 20 covariates have nonzero direct structural coefficients in at least one equation. Because the covariates are AR(1)-correlated, variables with zero direct coefficients can still act as marginal predictive proxies.

The rich nonlinear dictionary expands the raw design to 1,999 candidate nuisance features, exceeding the sample size. The dictionary is deliberately deterministic and applies every transformation to every covariate. No true active-variable list is passed to the estimator.

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

This design is **not oracle variable selection** because the estimator receives no active-variable list. It is, however, DGP-informed at the level of transformation families. The adjacent-interaction centering also uses the known AR(1) correlation parameter \(\rho\); because the feature pipeline subsequently standardizes within training folds, this centering does not identify active variables or inject treatment-effect information.

## L1 penalty policy

Nested CV inside every outer fold was initially implemented. It was removed before the scientific run because it created substantial repeated computation without being central to the causal question.

Fixed penalties were frozen using **separate calibration seeds 9101-9103** and nuisance-prediction diagnostics only. Treatment-effect bias/RMSE was not an input to the calibration criterion.

Frozen values:

- LASSO alpha = 0.05;
- L1-logistic C = 0.05.

Important qualification: the calibration script evaluates the **raw-feature** nuisance specification and uses simulation truth to assess nuisance error. The same values were carried into the rich-dictionary learner; they were not separately optimized for the 1,999-feature basis. The chosen values are therefore best described as fixed pilot-calibrated penalties rather than oracle-optimal hyperparameters.

The calibration grid and averages are committed under `results/`.

## Overlap calibration

Scenario D's treatment-index multiplier is **2.25**. In 20 fixed-seed calibration draws:

- mean treatment prevalence ≈ 0.488;
- true propensity outside [0.05,0.95] ≈ 15.9%;
- true propensity outside [0.10,0.90] ≈ 29.1%.

For Scenario C the corresponding [0.05,0.95] extreme share is only about 0.6%. The outcome equation is identical in C and D.

Scenario D is a **weak-overlap** design. Under the logistic assignment mechanism, propensity scores remain strictly between zero and one, so this should not be described as a literal mathematical positivity violation.

## Inference policy

- Difference in means and OLS use HC3 robust SEs.
- Naive LASSO plug-in is a point estimator only.
- Parametric IPW uses a plug-in influence-function SE conditional on fitted propensity estimates and is interpreted primarily as an overlap/weighting stress benchmark; it is not presented as a full first-step-adjusted M-estimation sandwich.
- Parametric AIPW and DML use influence-function-based SEs.
- Coverage is always interpreted jointly with estimator failures and interval width.

This last point became empirically important: in B and D the unpenalized weighting estimators can report high coverage only because rare estimates/intervals become enormous.

## Final Monte Carlo execution

Long single-process runs exposed operating-system resource accumulation from repeated high-dimensional native numerical routines. This did not change the scientific specification, but it made a single long-lived invocation unreliable in the available execution environment.

The final results are reported as **five independent blocks of 40 replications** with fixed batch seeds:

- 20260811
- 20360814
- 20460817
- 20560820
- 20660823

Scenario A was reported under the same batch-seed plan for consistent provenance.

The batch strategy changes only process lifetime. Estimators, DGPs, folds, penalties, and metric definitions are identical across blocks. `scripts/merge_batch_outputs.py` now validates the expected number of batches, replications, methods, and replication-seed uniqueness before writing merged outputs.

## Frozen-result verification boundary

The lightweight GitHub snapshot omits the replication-level raw CSV files. The committed manifest records SHA-256 hashes for those omitted artifacts, but the hashes cannot be independently verified from the snapshot alone unless the files are supplied or regenerated.

`python scripts/verify_frozen_results.py` therefore performs the strongest validation available without rerunning the experiment: it checks internal agreement among the manifest, combined summary, scenario summaries, and status counts, and hash-verifies any manifest-listed archive file that is actually present locally.

## Final primary-result status

The reported primary 800-dataset experiment is complete. QA/smoke outputs generated before the final design are not included in the primary figures or committed summary tables.

Pre-specified sample-size, fold-count, Random Forest, and XGBoost sensitivities remain future extensions and must be reported separately if executed.

## Package validation status

Manual PLR and IRM validation hooks against `DoubleMLPLR`/`DoubleMLIRM` remain in the repository. External package agreement is deliberately kept separate from the frozen primary-run claim unless the optional validation is actually executed and its output recorded.
