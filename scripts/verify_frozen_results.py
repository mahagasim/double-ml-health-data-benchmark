#!/usr/bin/env python
"""Verify internal consistency of committed frozen-result artifacts.

This script does not rerun the Monte Carlo experiment. It checks that the
committed manifest, summaries, and status counts agree with one another. If
replication-level files named in the manifest are present locally, their
SHA-256 hashes are also verified; omitted files are reported but do not make
the lightweight GitHub snapshot fail validation.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPECTED_SCENARIOS = ["A", "B", "C", "D"]
EXPECTED_METHODS = [
    "difference_in_means",
    "ols_adjusted",
    "lasso_plugin",
    "parametric_ipw",
    "parametric_aipw",
    "dml_plr_lasso",
    "dml_plr_lasso_rich",
]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads((RESULTS / "run_manifest.json").read_text())
    primary = manifest["primary_experiment"]

    assert primary["scenarios"] == EXPECTED_SCENARIOS
    assert primary["replications_per_scenario"] == 200
    assert primary["total_simulated_datasets"] == 800
    assert primary["primary_estimators_per_dataset"] == len(EXPECTED_METHODS)
    assert primary["crossfit_folds"] == 5
    assert len(primary["batch_seeds"]) == 5
    assert len(set(primary["batch_seeds"])) == 5

    combined = pd.read_csv(RESULTS / "combined_summary.csv")
    assert not combined.duplicated(["scenario", "method"]).any()
    assert set(combined["scenario"]) == set(EXPECTED_SCENARIOS)
    assert set(combined["method"]) == set(EXPECTED_METHODS)
    assert len(combined) == len(EXPECTED_SCENARIOS) * len(EXPECTED_METHODS)
    assert (combined["replications"] == 200).all()

    for scenario in EXPECTED_SCENARIOS:
        scenario_summary = pd.read_csv(RESULTS / f"scenario_{scenario}_summary.csv")
        expected = combined.loc[combined["scenario"] == scenario].reset_index(drop=True)
        actual = scenario_summary.reset_index(drop=True)
        assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )

    status = pd.read_csv(RESULTS / "status_counts.csv")
    status_totals = status.groupby(["scenario", "method"], as_index=False)["count"].sum()
    assert not status_totals.duplicated(["scenario", "method"]).any()
    assert set(status_totals["scenario"]) == set(EXPECTED_SCENARIOS)
    assert set(status_totals["method"]) == set(EXPECTED_METHODS)
    assert len(status_totals) == len(EXPECTED_SCENARIOS) * len(EXPECTED_METHODS)
    assert (status_totals["count"] == 200).all()

    merged = combined.merge(status_totals, on=["scenario", "method"], how="left", validate="one_to_one")
    expected_success = merged["successful_replications"].round().astype(int)
    ok_counts = (
        status.loc[status["status"] == "ok"]
        .set_index(["scenario", "method"])["count"]
        .reindex(pd.MultiIndex.from_frame(merged[["scenario", "method"]]), fill_value=0)
        .to_numpy()
    )
    assert (ok_counts == expected_success.to_numpy()).all()

    omitted = manifest.get("github_snapshot_omitted_reproducible_files", {})
    verified = []
    absent = []
    for rel_path, expected_hash in omitted.items():
        assert HEX64.fullmatch(expected_hash), f"Invalid SHA-256 string for {rel_path}"
        path = ROOT / rel_path
        if path.exists():
            actual_hash = sha256_file(path)
            assert actual_hash == expected_hash, f"SHA-256 mismatch: {rel_path}"
            verified.append(rel_path)
        else:
            absent.append(rel_path)

    print("Frozen-result integrity checks passed.")
    print(f"Scenarios: {len(EXPECTED_SCENARIOS)}; methods/scenario: {len(EXPECTED_METHODS)}; reported datasets: 800")
    if verified:
        print(f"Hash-verified present omitted/archive artifacts: {len(verified)}")
    if absent:
        print(f"Manifest-listed artifacts absent from lightweight snapshot (not hash-verified): {len(absent)}")


if __name__ == "__main__":
    main()
