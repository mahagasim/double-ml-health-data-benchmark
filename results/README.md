# Results directory

This directory contains the **committed frozen summaries and diagnostics** for the reported primary Monte Carlo experiment.

## Primary run

- 4 scenarios: A-D
- 200 reported replications per scenario
- 800 reported simulated datasets total
- 7 primary estimators per dataset
- true treatment effect: 1.0
- 5-fold cross-fitting for DML
- high-dimensional raw design: p=400
- rich DML nuisance design: p*=1,999

The final runs are reported as five independent 40-replication blocks with deterministic batch seeds and then merged. Batch execution changes only process lifetime; it does not alter the statistical design.

## Main files

- `key_results.csv`: compact reviewer-facing subset of the main inferential results.
- `scenario_A_raw.csv` ... `scenario_D_raw.csv`: replication-level estimator results in the full frozen run archive; intentionally omitted from this lightweight GitHub snapshot.
- `scenario_A_summary.csv` ... `scenario_D_summary.csv`: scenario-specific Monte Carlo summaries.
- `combined_raw.csv`: all primary replication-level results in the full frozen run archive; intentionally omitted from this lightweight GitHub snapshot.
- `combined_summary.csv`: full primary estimator summary.
- `combined_nuisance.csv`: average nuisance diagnostics.
- `status_counts.csv`: estimator status/failure counts.
- `nuisance_error_associations.csv`: within-scenario associations between nuisance diagnostics and absolute causal error.
- `overlap_calibration.csv`: pre-run true-overlap calibration summary.
- `overlap_calibration_raw.csv`: calibration draws.
- `l1_penalty_calibration.csv`: separate-seed nuisance-only L1 calibration grid.
- `l1_penalty_calibration_summary.csv`: averaged penalty-calibration diagnostics.
- `run_manifest.json`: frozen scientific configuration, seeds, omitted-artifact hashes, and validation-status metadata. It does **not** contain a complete software-version lock.

The narrative interpretation is in [`docs/results.md`](../docs/results.md).

## Integrity verification

Run:

```bash
python scripts/verify_frozen_results.py
```

This checks that:

- the manifest, combined summary, scenario summaries, and status counts agree;
- all four scenarios and all seven primary methods are present;
- each scenario-method combination reports 200 replications;
- successful-replication counts match `status_counts.csv`;
- manifest SHA-256 strings are well formed; and
- if an omitted archive artifact is present locally, its SHA-256 hash matches the manifest.

Because the replication-level CSV files are not committed, the lightweight GitHub snapshot **cannot independently hash-verify those omitted files** unless they are supplied or regenerated. The committed summaries and status counts can nevertheless be checked for internal consistency without rerunning the Monte Carlo.

Raw synthetic datasets are not committed. The DGP and deterministic seeds are retained so they can be regenerated.
