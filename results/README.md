# Results directory

This directory contains the **frozen primary Monte Carlo results**.

## Primary run

- 4 scenarios: A-D
- 200 replications per scenario
- 800 simulated datasets total
- 7 primary estimators per dataset
- true treatment effect: 1.0
- 5-fold cross-fitting for DML
- high-dimensional raw design: p=400
- rich DML nuisance design: p*=1,999

The high-dimensional scenarios were executed as five independent 40-replication blocks with deterministic batch seeds and then merged. Batch execution changes only process lifetime; it does not alter the statistical design.

## Main files

- `key_results.csv`: compact reviewer-facing subset of the main inferential results.
- `scenario_A_raw.csv` ... `scenario_D_raw.csv`: replication-level estimator results and diagnostics in the full frozen run archive; intentionally omitted from the lightweight GitHub snapshot and reproducible from the recorded seeds.
- `scenario_A_summary.csv` ... `scenario_D_summary.csv`: scenario-specific Monte Carlo summaries.
- `combined_raw.csv`: all primary replication-level results in the full frozen run archive; intentionally omitted from the lightweight GitHub snapshot.
- `combined_summary.csv`: full primary estimator summary.
- `combined_nuisance.csv`: average nuisance diagnostics.
- `status_counts.csv`: estimator status/failure counts.
- `nuisance_error_associations.csv`: within-scenario associations between nuisance diagnostics and absolute causal error.
- `overlap_calibration.csv`: pre-run true-overlap calibration summary.
- `overlap_calibration_raw.csv`: calibration draws.
- `l1_penalty_calibration.csv`: separate-seed nuisance-only L1 calibration grid.
- `l1_penalty_calibration_summary.csv`: averaged penalty-calibration diagnostics.
- `run_manifest.json`: frozen run configuration and software versions.

The narrative interpretation is in [`docs/results.md`](../docs/results.md).

Raw synthetic datasets are not committed. Every dataset is reproducible from the DGP, scenario configuration, and recorded seed.
