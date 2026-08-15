# Primary Monte Carlo results

## Executive result

The simulation does **not** support the claim that DML automatically outperforms classical estimators. Instead it supports a more specific conclusion:

> **In this benchmark, DML performs well when the nuisance representation is sufficiently adequate; orthogonalization and cross-fitting do not rescue severe nuisance misspecification, and they do not solve weak overlap.**

The true effect is \(\theta_0=1\). The committed summaries report 200 Monte Carlo replications for each of four scenarios.

## Selected primary results

| Scenario | Estimator | Bias | RMSE | 95% coverage | Failure rate |
|---|---|---:|---:|---:|---:|
| A | OLS adjustment | -0.005 | 0.067 | 0.965 | 0.000 |
| A | DML — raw LASSO | 0.044 | 0.079 | 0.900 | 0.000 |
| A | DML — rich LASSO | 0.048 | 0.080 | 0.920 | 0.000 |
| B | OLS adjustment | 0.003 | 0.091 | 0.985 | 0.000 |
| B | DML — raw LASSO | 0.056 | 0.089 | 0.905 | 0.000 |
| B | DML — rich LASSO | 0.059 | 0.095 | 0.850 | 0.000 |
| C | OLS adjustment | 0.498 | 0.511 | 0.040 | 0.000 |
| C | DML — raw LASSO | 0.492 | 0.501 | 0.000 | 0.000 |
| C | **DML — rich LASSO** | **0.018** | **0.078** | **0.945** | **0.000** |
| D | OLS adjustment | 0.880 | 0.889 | 0.000 | 0.000 |
| D | DML — raw LASSO | 0.853 | 0.859 | 0.000 | 0.000 |
| D | **DML — rich LASSO** | **0.051** | **0.099** | **0.900** | **0.000** |

The full estimator table, including Monte Carlo uncertainty, medians, tail errors, interval width, and runtimes, is in `results/combined_summary.csv`.

## Scenario A — simple linear benchmark

OLS is the natural benchmark and performs as expected: bias is approximately zero (-0.005), RMSE is 0.067, and coverage is 96.5%.

Both DML estimators remain stable but show small positive finite-sample bias of about 0.04-0.05 and modest undercoverage. The result is deliberately not framed as a DML victory: when the structural outcome model is correctly specified and low-dimensional, the classical estimator is excellent and simpler.

The naive LASSO plug-in is more biased (0.161) despite using regularized prediction, illustrating that “fit ML and plug predictions into a causal contrast” is not equivalent to orthogonal causal estimation.

## Scenario B — many sparse controls

With 400 observed controls and a sparse structural signal, OLS still performs very well because \(N=1000>p\) and the structural outcome equation remains linear. Its bias is 0.003 and coverage is 98.5%.

Raw DML-LASSO has slightly lower RMSE than OLS (0.089 versus 0.091) but higher bias and only 90.5% coverage. Given only 200 Monte Carlo replications and the shared simulated datasets, this very small RMSE difference should not be interpreted as evidence that DML dominates OLS. The rich dictionary adds unnecessary degrees of freedom in a setting where the nonlinear basis is not needed; its coverage falls to 85%.

This is an important negative result: **regularization and DML do not dominate a correctly specified classical model merely because many controls are present.**

The unpenalized parametric propensity estimators are a different story. IPW and AIPW fail numerically in 6.5% of replications and have extremely heavy-tailed estimates among the successful runs. Their median estimates remain much less extreme than their means, but rare near-separation/extreme-weight events make means and RMSE explode. This is why the project reports medians, failure rates, and interval widths rather than interpreting coverage in isolation.

## Scenario C — nonlinear high-dimensional confounding

This is the central learner-adequacy experiment.

OLS, the naive LASSO plug-in, and DML with raw linear LASSO nuisances all retain roughly one-half unit of positive bias:

- OLS bias: 0.498;
- naive LASSO plug-in bias: 0.550;
- raw-LASSO DML bias: 0.492.

Orthogonalization alone therefore does not rescue a nuisance learner that cannot represent the nonlinear confounding structure.

The rich-dictionary DML estimator changes the result sharply. Its bias falls to 0.018, RMSE to 0.078, and empirical 95% coverage is 94.5%. Outcome-nuisance RMSE falls from 1.171 under raw LASSO to 0.458 under the rich dictionary.

This is the clearest demonstration of the project's central mechanism:

\[
\text{DML} + \text{adequate nuisance representation}
\neq
\text{DML} + \text{arbitrary ML learner}.
\]

The rich representation is intentionally DGP-informed at the level of transformation families: it contains squares, sine terms, threshold indicators, and adjacent interactions capable of spanning the nonlinear simulation structure. It does **not** receive the active-variable set. Scenario C should therefore be read as a controlled representation-adequacy contrast, not as an unrestricted real-world model-selection contest.

## Scenario D — nonlinear confounding plus weak overlap

Scenario D keeps the Scenario C nonlinear outcome structure but strengthens treatment assignment. True propensity scores outside [0.05,0.95] rise from about 0.6% in C to 15.9% in D. The logistic DGP still has strict propensities between zero and one, so the issue is **weak/limited overlap**, not a literal mathematical failure of positivity.

Raw-linear methods deteriorate substantially. Rich-dictionary DML remains far more accurate than the alternatives, but its performance also worsens relative to Scenario C:

- bias: 0.018 → 0.051;
- RMSE: 0.078 → 0.099;
- coverage: 94.5% → 90.0%;
- average CI width: 0.278 → 0.322.

The result therefore does **not** say that flexible DML solves overlap. It says that good nuisance learning can address functional-form complexity, while poor overlap remains a separate information and inference limitation.

## Treatment discrimination is not causal identification

The rich-dictionary treatment nuisance becomes much better at discriminating treatment in Scenario D:

- mean propensity AUC in C: 0.646;
- mean propensity AUC in D: 0.825.

Yet this is not a general improvement across propensity-quality metrics: propensity RMSE-to-truth and calibration error do not improve from C to D. At the same time, causal RMSE rises and coverage falls while the true extreme-propensity share increases dramatically.

The supported lesson is therefore deliberately narrow:

> **Better treatment discrimination can coincide with worse causal estimation when treatment becomes more deterministic and comparable treated/untreated observations disappear.**

This is not evidence that “better nuisance prediction” in every sense causes worse estimation; AUC, calibration, probability accuracy, and causal utility are distinct objects.

## Why IPW “coverage” can be misleading

In Scenario D, unpenalized parametric IPW/AIPW fail outright in 62.5% of replications. Among successful replications, reported intervals can be enormous: average widths are on the order of hundreds of thousands because of extreme inverse weights.

Consequently, a seemingly high empirical coverage number for IPW does not indicate useful inference. Coverage must be interpreted jointly with failure rate, RMSE, interval width, and tail behavior. In addition, the IPW implementation uses a plug-in influence-function SE conditional on the fitted propensity estimates rather than a full first-step-adjusted sandwich, so its coverage is treated as a descriptive stress-benchmark diagnostic.

## Main conclusion

The primary experiment supports four conclusions:

1. **Correctly specified classical models remain hard to beat.** In A and B, OLS is excellent.
2. **DML is not a substitute for adequate nuisance learning.** Raw-linear DML fails under the nonlinear DGP in C.
3. **High-dimensional regularized nuisance learning can make DML work when the representation is adequate.** In this controlled DGP, the rich dictionary expands 400 raw covariates to 1,999 candidate features and recovers near-nominal inference in C.
4. **Overlap is a separate information problem.** Higher treatment AUC in D coincides with worse causal performance because treatment assignment becomes more separated.

## Limitations

- The data are simulated and deliberately structured; no clinical effect is claimed.
- Treatment effects are constant, making the PLR coefficient equal to the ATE by construction. With heterogeneous treatment effects, the PLR target would require separate interpretation.
- The rich dictionary intentionally includes transformation families capable of spanning the nonlinear DGP. It applies them to every covariate, so it is not oracle variable selection, but it is DGP-informed at the representation-family level.
- L1 penalties are fixed after nuisance-only pilot calibration on separate seeds. The calibration uses raw features and simulation truth; it does not separately optimize the rich dictionary.
- The unpenalized parametric propensity model is intentionally stressed by \(p=400\); its instability should not be generalized to all modern propensity implementations.
- The committed GitHub snapshot omits replication-level raw CSVs. It supports internal consistency checks of summaries/status counts and records raw-file hashes, but those hashes cannot be independently verified without the omitted files or a regeneration.
- Pre-specified sample-size, fold-count, Random Forest, and XGBoost sensitivities have not been added to the primary results.
- The manual estimator has an optional independent `DoubleML` validation hook, but package agreement is not part of the frozen scientific-run claim unless that validation is actually executed and recorded.
