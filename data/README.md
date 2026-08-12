# Data provenance

The primary project uses **synthetic data only**. No patient, SHARE, BRFSS, administrative, or other external microdata are required.

Each Monte Carlo dataset is generated deterministically from:

- a scenario configuration;
- the data-generating-process code in `src/dml_health_benchmark/dgp.py`; and
- a recorded random seed.

The causal estimators observe only:

- `Y`: observed continuous outcome;
- `D`: observed binary treatment;
- `X1 ... Xp`: observed baseline covariates.

The simulation additionally retains hidden truth such as potential outcomes, the true propensity score, and the true nuisance functions. Those quantities are used **only for simulation evaluation** and are not passed to the estimators.

Generated Monte Carlo datasets are not committed because thousands of large CSV files would be redundant. The experiment regenerates them from seeds and stores estimator-level results instead.

Use `scripts/export_example_data.py` when a human-readable example CSV is useful. By default it exports only the observed analysis variables; hidden simulation truth is included only with an explicit flag.
