# When Does Double Machine Learning Improve Causal Estimation?

**A Monte Carlo benchmark of dimensionality, nonlinear nuisance structure, regularization, and overlap in simulated health data**

> **Status:** primary experiment complete. This lightweight GitHub snapshot contains frozen summaries from a reported 800-dataset Monte Carlo run (200 replications × 4 scenarios), reviewer-facing notebooks, figures, diagnostics, deterministic seeds, and SHA-256 reference hashes for omitted replication-level outputs. No real patient data are used.

## Research question

**When does Double Machine Learning improve causal estimation, and when does it fail to do so?**

The project asks how conventional causal estimators and DML behave as observed confounding becomes high-dimensional, nonlinear, and difficult to represent, and as treatment overlap deteriorates.

A second question is deliberately predictive-versus-causal:

> Does better treatment discrimination necessarily imply better causal-effect estimation?

## Main finding

The simulation does **not** show that DML automatically beats classical estimators.

It shows something more specific:

> **In this benchmark, DML performs well when the nuisance representation is adequate; orthogonalization does not compensate for severe nuisance misspecification, and it does not solve weak overlap.**

The strongest contrast occurs in the nonlinear Scenario C. The true effect is 1.0:

| Estimator | Bias | RMSE | 95% coverage |
|---|---:|---:|---:|
| OLS adjustment | 0.498 | 0.511 | 0.040 |
| DML — raw LASSO | 0.492 | 0.501 | 0.000 |
| **DML — rich-dictionary LASSO** | **0.018** | **0.078** | **0.945** |

Under weak overlap (Scenario D), rich-dictionary DML remains much more accurate than the alternatives, but coverage falls to 90.0% and RMSE rises to 0.099. DML therefore does not solve weak overlap.

See [`docs/results.md`](docs/results.md) for the full interpretation.

![Core estimator bias](figures/bias_core_methods.svg)

## Why this project

Machine-learning algorithms optimize prediction. A causal estimand is a different statistical object. Naively inserting regularized or flexible nuisance predictions into a causal procedure can transmit model-selection and overfitting bias into a low-dimensional target.

Double/Debiased Machine Learning addresses this problem through:

1. **Neyman-orthogonal scores**, which reduce first-order sensitivity to nuisance-estimation error; and
2. **cross-fitting**, which evaluates the score using out-of-fold nuisance predictions.

This repository therefore implements the estimating equation directly rather than treating DML as a black-box package call.

## Primary causal model

The DGP satisfies a partially linear model

\[
Y=D\theta_0+g_0(X)+\varepsilon,
\qquad E[\varepsilon\mid D,X]=0.
\]

Treatment is binary, the outcome is continuous, and

\[
\theta_0=1.
\]

Because treatment effects are constant,

\[
ATE=E[Y(1)-Y(0)]=1.
\]

The project also retains a manual binary-treatment IRM/AIPW DML implementation as a secondary extension.

## Four simulation scenarios

| Scenario | N | Raw p | Structure | Overlap | Purpose |
|---|---:|---:|---|---|---|
| **A** | 1,000 | 10 | Linear, low-dimensional | Good | Classical benchmark |
| **B** | 1,000 | 400 | Sparse linear, many controls | Good | Stress regularization/dimensionality |
| **C** | 1,000 | 400 | Nonlinear + interactions | Good | Stress nuisance representation |
| **D** | 1,000 | 400 | Same nonlinear structure as C | Weak | Stress overlap |

Covariates follow a correlated Gaussian AR(1)-type design with \(\rho=0.30\).

In B-D:

- X1-X10 enter both the treatment and outcome functions directly;
- X11-X15 enter the treatment index directly but not the outcome mean;
- X16-X20 enter the outcome mean directly but not the treatment index;
- X21-X400 have zero direct structural coefficients in both equations.

Because covariates are correlated, variables with zero direct coefficients can still be marginally predictive proxies. The labels above therefore refer to structural entry, not marginal predictive irrelevance.

## A genuinely high-dimensional nuisance dictionary

The primary flexible DML learner is not a model-zoo addition. It creates a fixed nonlinear basis for **every** covariate:

- \(X_j\);
- \(X_j^2-1\);
- \(\sin(X_j)\);
- \(I(X_j>0)-1/2\);
- adjacent interactions \(X_jX_{j+1}-\rho\).

For \(p=400\), this produces

\[
p^*=5p-1=1999>N=1000.
\]

LASSO therefore performs regularized nuisance learning in a true \(p^*>N\) design. The dictionary does **not** receive an oracle list of active variables: the same transformations are applied to all covariates. However, the transformation families were deliberately chosen to span the nonlinearities used in the simulation DGP, so Scenario C is a controlled representation-adequacy experiment rather than an unrestricted black-box benchmark.

## Estimator ladder

The primary benchmark contains seven estimators:

1. **Difference in means** — unadjusted confounding benchmark.
2. **OLS adjustment** — classical regression benchmark.
3. **Naive LASSO g-formula plug-in** — regularized outcome prediction without an orthogonal score.
4. **Parametric IPW** — unpenalized logistic propensity weighting.
5. **Parametric AIPW** — doubly robust parametric benchmark.
6. **DML-PLR + raw LASSO** — sparse linear nuisance learning.
7. **DML-PLR + rich-dictionary LASSO** — nonlinear high-dimensional nuisance learning.

Random Forest and XGBoost implementations remain available as optional sensitivities; they are not used in the frozen primary results.

## Manual DML implementation

Define

\[
\ell_0(X)=E[Y\mid X],
\qquad
m_0(X)=E[D\mid X].
\]

The partialling-out score is

\[
\psi(W;\theta,\eta)
=
\{Y-\ell(X)-\theta[D-m(X)]\}\{D-m(X)\}.
\]

The code manually:

1. creates five outer folds;
2. fits nuisance models on the complement of each fold;
3. generates held-out \(\hat\ell(X)\) and \(\hat m(X)\);
4. residualizes outcome and treatment;
5. estimates \(\theta\) from the orthogonal score; and
6. computes an influence-function standard error.

No DML package is required for the primary estimator.

## Frozen L1 penalties

L1 hyperparameters were frozen **before the scientific run** using separate seeds 9101-9103 and nuisance-prediction diagnostics only. Treatment-effect error was not used for selection.

Frozen values:

- outcome LASSO \(\alpha=0.05\);
- L1-logistic treatment nuisance \(C=0.05\).

The pilot calibration used the raw-feature nuisance specification and simulation truth to assess nuisance error; it was not a separate optimization of the 1,999-feature rich dictionary. The values should therefore be read as fixed pilot-calibrated penalties, not as oracle-optimal hyperparameters for every learner specification. The full grid is committed in `results/l1_penalty_calibration.csv`.

## Monte Carlo design

The reported primary experiment contains

\[
4\times200=800
\]

simulated datasets.

For each estimator the repository records:

- bias and Monte Carlo SE of bias;
- RMSE;
- empirical SD;
- mean estimated SE;
- empirical 95% coverage and its Monte Carlo SE;
- average interval width;
- median and 95th-percentile absolute error;
- estimator failure rate;
- runtime.

Coverage is interpreted jointly with interval width and failures. This matters because unstable IPW/AIPW procedures can obtain superficially high coverage only through enormous intervals.

## Primary results by scenario

### A — classical model is correctly specified

OLS is essentially unbiased (bias -0.005), with RMSE 0.067 and 96.5% coverage. DML is stable but not superior; fixed regularization produces small finite-sample bias and some undercoverage.

**Lesson:** complexity is unnecessary when the simple model is right.

### B — sparse many-control setting

OLS remains excellent because the structural outcome model is linear and \(N>p\). Raw DML-LASSO has similar RMSE but more bias and lower coverage.

Unpenalized parametric IPW/AIPW become heavy-tailed and fail numerically in 6.5% of replications.

**Lesson:** “many controls” alone does not imply DML must dominate correctly specified regression, but unregularized propensity modeling can become fragile.

### C — nonlinear confounding

Raw-linear OLS and raw-linear DML both retain approximately +0.50 bias. Rich-dictionary DML reduces bias to +0.018 and restores 94.5% coverage.

Outcome-nuisance RMSE falls from 1.171 for raw DML-LASSO to 0.458 for rich-dictionary DML.

**Lesson:** orthogonalization requires adequate nuisance learning; DML is not a cure for arbitrary nuisance misspecification.

### D — weak overlap

The rich learner's mean propensity AUC rises from 0.646 in C to 0.825 in D, yet propensity RMSE-to-truth and calibration error do not improve. At the same time causal RMSE worsens from 0.078 to 0.099 and coverage falls from 94.5% to 90.0%.

The share of **true** propensity scores outside [0.05,0.95] rises from roughly 0.6% to 15.9%.

**Lesson:** better treatment discrimination can indicate worse causal overlap; predictive discrimination is not the same object as causal identification or stable effect estimation.

![RMSE on log scale](figures/rmse_all_methods_log.svg)

![True overlap stress](figures/true_overlap_extremes.svg)

## Reproducibility

The final high-dimensional simulations were reported as five independent 40-replication blocks so long-lived native numerical state could not contaminate the run. The batch seeds are recorded in the methodology and run manifest, and the merge procedure is deterministic.

The GitHub snapshot includes:

- deterministic DGP and seeds;
- configuration files documenting the frozen specification;
- manual estimator source code;
- unit/smoke tests;
- nuisance-only hyperparameter calibration;
- overlap calibration;
- frozen Monte Carlo summaries plus SHA-256 reference hashes for omitted replication-level outputs;
- automated figures;
- reviewer-facing notebooks;
- CI checks;
- an optional DoubleML validation hook.

Current committed test suite at the frozen snapshot: **9 tests**. CI on this revision also checks internal consistency of the committed frozen-result artifacts.

> **Repository-size note:** the GitHub portfolio snapshot intentionally omits the multi-megabyte replication-level `*_raw.csv` files and the generated example dataset. The manifest records deterministic seeds and SHA-256 reference hashes for those omitted artifacts. Because the raw files are not committed, this snapshot can verify the internal consistency of the committed summaries and status counts, but it cannot independently hash-verify the omitted raw files unless they are supplied or regenerated.

The YAML files under `configs/` document the frozen scenario specification. The executable scenario definitions currently live in the Python source; the YAML files should not be interpreted as an independently parsed source of truth.

## Repository structure

```text
double-ml-health-data-benchmark/
├── README.md
├── pyproject.toml
├── references.bib
├── configs/
├── data/
├── docs/
│   ├── methodology.md
│   ├── results.md
│   ├── literature_matrix.md
│   └── design_decisions.md
├── src/dml_health_benchmark/
│   ├── dgp.py
│   ├── features.py
│   ├── learners.py
│   ├── estimators.py
│   ├── diagnostics.py
│   ├── monte_carlo.py
│   ├── plotting.py
│   └── validation.py
├── scripts/
│   ├── calibrate_lasso_penalties.py
│   ├── calibrate_overlap.py
│   ├── run_experiment.py
│   ├── merge_batch_outputs.py
│   ├── verify_frozen_results.py
│   ├── export_example_data.py
│   └── make_figures.py
├── tests/
├── notebooks/
├── results/
│   ├── run_manifest.json
│   ├── key_results.csv
│   ├── combined_summary.csv
│   └── scenario_*_summary.csv
└── figures/
```

## Run the project

Install the primary environment:

```bash
pip install -e .
```

Run a short reproducibility check:

```bash
dml-health-run --scenario C --replications 10 --folds 5 --jobs 1
```

Run one full scenario in an environment that supports a long process:

```bash
dml-health-run --scenario C --replications 200 --folds 5 --jobs 5
```

For isolated blocks, run separate 40-replication jobs and merge them with:

```bash
python scripts/merge_batch_outputs.py \
  --scenario C \
  --batch-root results/batches_C \
  --output-dir results
```

Check the internal consistency of the committed frozen summaries/status/manifest without rerunning the Monte Carlo:

```bash
python scripts/verify_frozen_results.py
```

Generate figures from saved raw outputs when those outputs are available:

```bash
python scripts/make_figures.py \
  results/scenario_A_raw.csv \
  results/scenario_B_raw.csv \
  results/scenario_C_raw.csv \
  results/scenario_D_raw.csv
```

## Reviewer notebooks

1. `01_theory_and_dgp.ipynb` — causal target, assumptions, DGP and overlap.
2. `02_manual_dml_walkthrough.ipynb` — cross-fitting and orthogonal-score derivation from scratch.
3. `03_monte_carlo_analysis.ipynb` — reads the frozen committed summaries and reproduces the substantive diagnostics.

The notebooks are intentionally lightweight reviewer aids. Saved execution outputs are not used as evidence for the primary numerical claims.

## Boundaries and limitations

- This is a simulation study, not a clinical-effect analysis.
- Treatment effects are constant by construction; under heterogeneous effects, the PLR coefficient need not equal the population ATE.
- The rich basis intentionally contains transformation families capable of representing the nonlinear DGP; it is a controlled learner-adequacy experiment.
- Fixed nuisance penalties are one finite-sample design choice and were pilot-calibrated on raw features rather than separately optimized for the rich dictionary.
- The unpenalized parametric propensity benchmark is deliberately stressed in the 400-control settings; its plug-in SE does not constitute a full first-step-adjusted M-estimation sandwich.
- Sample-size/fold-count and tree-learner sensitivities remain extensions.
- `DoubleML` package agreement is not part of the frozen scientific run claim. An optional independent validation hook is retained separately.
- With 200 replications, Monte Carlo uncertainty is non-negligible; small differences between close-performing methods should not be overinterpreted.

## Methodological foundation

The project is anchored to the high-dimensional treatment-effect, doubly robust, DML/orthogonality, cross-fitting, and overlap literature. See [`docs/literature_matrix.md`](docs/literature_matrix.md) and [`references.bib`](references.bib).
