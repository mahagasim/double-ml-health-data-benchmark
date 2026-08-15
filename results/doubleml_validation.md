# External DoubleML implementation validation

This check is **separate from the frozen primary Monte Carlo experiment**. It does not change any reported Monte Carlo result and does not validate the rich-dictionary specification directly. Its purpose is narrower: compare the repository's manual **raw-LASSO PLR partialling-out implementation** with `DoubleMLPLR` on the same fixed simulated data and exactly the same externally supplied five-fold sample split.

## Environment

- Python: 3.11.15
- DoubleML: 0.11.4
- primary direct dependency versions: `requirements-ci-pinned.txt`
- cross-fitting folds: 5
- fold random state: 314159

## Fixed validation draws

| Scenario | Data seed | Manual estimate | DoubleML estimate | Estimate difference | Manual SE | DoubleML SE | SE difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 20260811 | 1.0235945620 | 1.0235949932 | -4.31e-07 | 0.0661879905 | 0.0661549274 | 3.31e-05 |
| C | 20260813 | 1.4574501028 | 1.4574521898 | -2.09e-06 | 0.0925834801 | 0.0925371936 | 4.63e-05 |

The validation criterion is an absolute point-estimate difference below `1e-5` and an absolute standard-error difference below `1e-3`. These tolerances allow for small numerical-optimization and finite-sample normalization differences while being much tighter than any scientifically meaningful difference in this benchmark.

## Interpretation

The fixed-draw comparison provides an independent implementation check for the manual **raw-LASSO PLR** estimator: its point estimate and influence-function standard error numerically agree with `DoubleMLPLR` to substantially tighter tolerances than the benchmark reports results.

This should **not** be interpreted as:

- an independent re-analysis of all 800 Monte Carlo datasets;
- validation of the omitted replication-level archive hashes;
- a validation of the rich 1,999-feature dictionary against DoubleML; or
- evidence that DoubleML/DML is generally superior to other estimators.

The reproducible runner is `scripts/validate_doubleml.py`, and CI executes it in a dedicated validation job.
