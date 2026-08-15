# Methodology specification

## 1. Causal target and structural model

The primary simulation uses a partially linear model (PLR)

\[
Y = D\theta_0 + g_0(X) + \varepsilon,
\qquad E[\varepsilon\mid D,X]=0,
\]

with binary treatment \(D\in\{0,1\}\), a continuous outcome, and a constant structural treatment effect

\[
\theta_0=1.
\]

Because treatment effects are constant in the DGP,

\[
E[Y(1)-Y(0)] = \theta_0 = 1,
\]

so the PLR coefficient is also the ATE in this benchmark. This equivalence relies on the constant-effect design; under treatment-effect heterogeneity, a PLR coefficient need not equal the population ATE. The binary-treatment interactive regression model (IRM) and its AIPW-type orthogonal score remain in the codebase as a secondary extension.

## 2. Identification assumptions represented by the DGP

1. **Consistency:** the observed outcome equals the potential outcome under the observed treatment.
2. **Conditional exchangeability:** \((Y(1),Y(0))\perp D\mid X\).
3. **Positivity:** \(0<P(D=1\mid X)<1\). Scenario D preserves strict positivity under the logistic DGP but deliberately creates **weak/limited overlap**, i.e. many propensity scores close to the boundaries.
4. **No interference:** one simulated unit's treatment does not affect another unit's outcome.

DML addresses estimation with complex nuisance functions. It does not repair violations of causal identification assumptions or eliminate the information loss caused by weak overlap.

## 3. Causal graph

```mermaid
graph LR
  X[Baseline covariates X] --> D[Treatment D]
  X --> Y[Outcome Y]
  D --> Y
```

## 4. Covariates and structural roles

Covariates follow an AR(1)-type Gaussian design

\[
X\sim N(0,\Sigma),\qquad \Sigma_{jk}=\rho^{|j-k|},\qquad \rho=0.30.
\]

In Scenarios B-D:

- X1-X10 enter both the treatment and outcome functions directly;
- X11-X15 enter the treatment index directly but not the outcome mean;
- X16-X20 enter the outcome mean directly but not the treatment index;
- X21-X400 have zero direct structural coefficients in both equations.

Because \(X\) is correlated, a variable with a zero direct coefficient can still be marginally predictive through correlation with active variables. The categories above therefore describe **direct structural entry**, not marginal predictive irrelevance.

## 5. Four scenarios

| Scenario | N | p | Nuisance structure | Overlap |
|---|---:|---:|---|---|
| A | 1,000 | 10 | Linear, low-dimensional | Good |
| B | 1,000 | 400 | Sparse linear | Good |
| C | 1,000 | 400 | Nonlinear + interactions | Good |
| D | 1,000 | 400 | Same nonlinear structure as C | Weak |

A→B primarily stresses dimensionality; B→C stresses functional-form complexity; C→D stresses overlap while holding the nonlinear outcome structure fixed.

Scenario D multiplies the Scenario C treatment index by 2.25. Across 20 fixed-seed pre-run calibration draws, the share of true propensity scores outside [0.05,0.95] rises from about 0.6% in C to 15.9% in D, while treatment prevalence remains close to one-half.

## 6. Primary estimator ladder

1. Difference in means.
2. OLS adjustment with all observed covariates.
3. Naive LASSO outcome-regression g-formula plug-in; point estimate only.
4. Unpenalized parametric logistic IPW.
5. Unpenalized-parametric AIPW with linear outcome regressions.
6. **DML-PLR + raw-covariate LASSO.**
7. **DML-PLR + rich nonlinear dictionary LASSO.**

XGBoost and Random Forest are retained as optional learner-sensitivity implementations but are not part of the frozen primary Monte Carlo benchmark.

## 7. PLR orthogonal score

Define

\[
\ell_0(X)=E[Y\mid X],\qquad m_0(X)=E[D\mid X].
\]

The partialling-out score is

\[
\psi(W;\theta,\eta)
=
\{Y-\ell(X)-\theta[D-m(X)]\}\{D-m(X)\}.
\]

For cross-fitted nuisance predictions \(\hat\ell_i\) and \(\hat m_i\), let

\[
\tilde Y_i=Y_i-\hat\ell_i,
\qquad
\tilde D_i=D_i-\hat m_i.
\]

Then

\[
\hat\theta
=
\frac{\sum_i \tilde D_i\tilde Y_i}
{\sum_i \tilde D_i^2}.
\]

The estimated influence function is

\[
\hat\phi_i
=
\frac{\tilde D_i(\tilde Y_i-\hat\theta\tilde D_i)}
{N^{-1}\sum_j\tilde D_j^2},
\]

with

\[
\widehat{SE}(\hat\theta)=\frac{sd(\hat\phi_i)}{\sqrt N}.
\]

## 8. Five-fold cross-fitting

The primary specification uses five outer folds. For every observation, nuisance predictions are generated only by models trained on the other four folds. The treatment-effect score is then evaluated using the pooled out-of-fold predictions.

The implementation uses stratified folds based on treatment status. This is a pragmatic finite-sample choice that stabilizes class balance across folds; it differs slightly from the simplest theoretical presentation in which an external random partition is drawn independently of the data values.

The implementation is manual: the project does not call a DML package to obtain the primary estimates.

## 9. Fixed L1 nuisance penalties

Repeated nested cross-validation created substantial computational overhead without advancing the causal question. Before the scientific Monte Carlo, L1 penalties were therefore frozen using **separate seeds 9101-9103** and nuisance-prediction diagnostics only. Treatment-effect error was not used for hyperparameter selection.

Frozen values:

- standardized outcome LASSO: \(\alpha=0.05\);
- standardized L1-logistic treatment nuisance: \(C=0.05\).

Important scope qualification: the calibration script evaluates the **raw-feature** nuisance specification. It uses simulation truth to score nuisance error, which is appropriate as a pilot simulation-design aid but is not a realistic observational-data tuning procedure. The same frozen values are then used for the rich-dictionary specification; they were not separately optimized over the 1,999-feature basis. They should therefore be understood as **fixed pilot-calibrated penalties**, not oracle-optimal values.

The complete calibration grid is saved in `results/l1_penalty_calibration.csv` and its aggregated version in `results/l1_penalty_calibration_summary.csv`.

## 10. Raw versus rich nuisance dictionaries

### Raw LASSO

The raw learner uses the original \(p\) covariates only. It is well matched to sparse linear structure but cannot directly represent the nonlinear terms in Scenarios C-D.

### Rich-dictionary LASSO

The flexible primary learner applies the same deterministic basis to **every** observed covariate:

1. \(X_j\);
2. \(X_j^2-1\);
3. \(\sin(X_j)\);
4. \(I(X_j>0)-1/2\);
5. adjacent interactions \(X_jX_{j+1}-\rho\).

For \(p>1\), the transformed dimension is

\[
p^* = 4p+(p-1)=5p-1.
\]

Thus the high-dimensional scenarios have

\[
p=400 \quad\Rightarrow\quad p^*=1999>N=1000.
\]

The dictionary is fixed and non-data-adaptive. It does not select the known active variables; all transformations are applied indiscriminately to all 400 covariates. LASSO then performs regularized nuisance selection in a genuinely \(p^*>N\) design.

The transformation **families are nevertheless DGP-informed**: they were deliberately chosen to span the nonlinearities used in the simulation. This makes Scenario C a controlled test of representation adequacy, not an oracle active-variable experiment and not a leaderboard among arbitrary black-box learners.

The adjacent interactions are centered using the known design correlation \(\rho\). This uses DGP knowledge for centering, but the nuisance pipelines subsequently standardize features within training folds; subtracting a constant therefore does not reveal active variables or provide the estimator with treatment-effect information.

## 11. Secondary IRM/AIPW score

For

\[
g_d(X)=E[Y\mid D=d,X],\qquad e(X)=P(D=1\mid X),
\]

the secondary ATE orthogonal score is

\[
\psi_i(\tau)
=
g_1(X_i)-g_0(X_i)
+\frac{D_i(Y_i-g_1(X_i))}{e(X_i)}
-\frac{(1-D_i)(Y_i-g_0(X_i))}{1-e(X_i)}-\tau.
\]

This keeps the relationship among AIPW, double robustness, orthogonal scores, and cross-fitted DML explicit.

## 12. Monte Carlo design

The reported frozen primary experiment uses

\[
R=200
\]

replications per scenario, for **800 simulated datasets** in total. Each estimator is evaluated against the known truth \(\theta_0=1\).

For each estimator the project records:

- mean and median estimate;
- bias and Monte Carlo SE of bias;
- RMSE;
- median and 95th-percentile absolute error;
- empirical SD;
- average estimated SE where inference is defined;
- 95% CI coverage and Monte Carlo SE of coverage;
- average CI width;
- numerical failure rate;
- runtime.

At nominal 95% coverage and \(R=200\), the Monte Carlo standard error of the coverage proportion is about 1.5 percentage points. Small differences between close-performing estimators should therefore not be overinterpreted without paired Monte Carlo uncertainty calculations.

### Execution provenance

The final runs are reported as five independent 40-replication blocks with batch seeds

`20260811`, `20360814`, `20460817`, `20560820`, and `20660823`.

The batch CSVs are merged deterministically by `scripts/merge_batch_outputs.py`. The repository's lightweight GitHub snapshot omits the replication-level CSVs, so the committed summaries/status counts can be internally checked but the SHA-256 hashes of the omitted raw files cannot be independently verified from the snapshot alone.

The batch execution strategy changes only process lifetime; it does not alter any DGP, fold, nuisance learner, estimating equation, or result definition.

## 13. Nuisance diagnostics

For both DML-PLR estimators the simulation records:

- RMSE of \(\hat\ell(X)\) against the known true \(E[Y\mid X]\);
- propensity RMSE against true \(e(X)\);
- Brier score;
- log loss;
- AUC;
- calibration error;
- estimated extreme-propensity share;
- nuisance feature dimension.

These metrics should not be collapsed into a single statement that nuisance prediction is globally “better.” In particular, from C to D the rich learner's treatment AUC rises substantially while propensity RMSE-to-truth and calibration error do not improve. The supported causal lesson is narrower: **better treatment discrimination can coincide with worse causal estimation when treatment becomes more deterministic and overlap deteriorates.**

## 14. Overlap policy

The primary experiment does **not** silently trim observations or truncate propensity scores. It records true and estimated propensity diagnostics and explicitly records estimator failures.

Scenario D is described as a **weak-overlap / limited-overlap** design rather than a literal positivity violation: under the logistic DGP, propensities remain strictly between zero and one, but many are close enough to the boundaries to make estimation unstable.

Trimming, if studied later, is a sensitivity analysis because it changes the procedure and potentially the target population.

## 15. Configuration files

The YAML files under `configs/` document the frozen scenario specification for reviewer readability. The current execution path uses the scenario definitions in the Python source; the YAML files are not independently parsed by the primary CLI and should not be described as a separate executable source of truth.

## 16. Pre-specified but not yet executed sensitivities

1. Scenario C sample size: \(N\in\{500,1000,2000\}\).
2. Scenario C cross-fitting folds: \(K\in\{2,5\}\).
3. Optional learner sensitivities using Random Forest or XGBoost.

These remain extensions; they are not mixed into the primary 800-dataset results.

## 17. Inference caveat for parametric weighting benchmarks

Difference in means and OLS use HC3 robust standard errors. The naive LASSO plug-in is a point estimator only. Parametric AIPW and the DML estimators use influence-function-based standard errors.

The parametric IPW implementation uses a plug-in influence-function standard error conditional on the fitted propensity estimates; it is not presented as a full first-step-adjusted M-estimation sandwich. Its coverage is therefore treated as a descriptive stress-benchmark diagnostic rather than as a claim of generally valid parametric IPW inference in these high-dimensional settings.

## 18. Software validation

The manual PLR-LASSO estimator has a validation hook for comparison with `DoubleMLPLR` using identical externally supplied folds and the `partialling out` score. The manual IRM-LASSO implementation has an analogous `DoubleMLIRM` hook.

`doubleml` is an optional validation dependency. Agreement with the package is not part of the frozen primary-run claim unless the external validation is actually executed and its result recorded. The current CI separately checks the repository's internal tests and frozen-result consistency.
