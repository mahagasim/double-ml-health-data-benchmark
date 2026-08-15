#!/usr/bin/env python
"""Merge independently executed Monte Carlo batch outputs into one scenario result.

The frozen primary design used five batches of 40 replications. Integrity
checks are performed before any merged output is written so incomplete or
malformed batches cannot silently enter the frozen summaries. The expected
batch counts can be overridden for small development runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dml_health_benchmark.monte_carlo import summarize_monte_carlo, summarize_nuisance

EXPECTED_METHODS = {
    "difference_in_means",
    "ols_adjusted",
    "lasso_plugin",
    "parametric_ipw",
    "parametric_aipw",
    "dml_plr_lasso",
    "dml_plr_lasso_rich",
}


def validate_batch(df: pd.DataFrame, scenario: str, expected_reps: int, path: Path) -> None:
    required = {"scenario", "replication", "replication_seed", "method"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    if set(df["scenario"].dropna().astype(str)) != {scenario}:
        raise ValueError(f"{path}: scenario column is not exclusively {scenario}")

    if df.duplicated(["replication", "method"]).any():
        raise ValueError(f"{path}: duplicate (replication, method) rows")

    reps = list(pd.unique(df["replication"]))
    if len(reps) != expected_reps:
        raise ValueError(f"{path}: expected {expected_reps} replications, found {len(reps)}")

    for rep, group in df.groupby("replication", sort=False):
        methods = set(group["method"])
        if methods != EXPECTED_METHODS:
            missing_methods = sorted(EXPECTED_METHODS - methods)
            extra_methods = sorted(methods - EXPECTED_METHODS)
            raise ValueError(
                f"{path}: replication {rep} has unexpected method set; "
                f"missing={missing_methods}, extra={extra_methods}"
            )
        seeds = group["replication_seed"].dropna().unique()
        if len(seeds) != 1:
            raise ValueError(f"{path}: replication {rep} does not have exactly one replication seed")

    rep_seed_table = df[["replication", "replication_seed"]].drop_duplicates()
    if rep_seed_table["replication_seed"].duplicated().any():
        raise ValueError(f"{path}: replication seeds are not unique within the batch")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list("ABCD"), required=True)
    ap.add_argument("--batch-root", required=True)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--expected-batches", type=int, default=5)
    ap.add_argument("--expected-reps-per-batch", type=int, default=40)
    args = ap.parse_args()

    root = Path(args.batch_root)
    files = sorted(root.glob(f"batch_*/scenario_{args.scenario}_raw.csv"))
    if not files:
        raise FileNotFoundError(f"No batch raw files found under {root}")
    if len(files) != args.expected_batches:
        raise ValueError(f"Expected {args.expected_batches} batch files, found {len(files)}")

    parts = []
    global_rep = 0
    all_replication_seeds: set[int] = set()

    for batch_id, path in enumerate(files):
        df = pd.read_csv(path)
        validate_batch(df, args.scenario, args.expected_reps_per_batch, path)

        batch_seed_values = set(df["replication_seed"].dropna().astype(int).unique())
        overlap = all_replication_seeds.intersection(batch_seed_values)
        if overlap:
            raise ValueError(f"{path}: replication seeds duplicated across batches: {sorted(overlap)[:5]}")
        all_replication_seeds.update(batch_seed_values)

        batch_reps = list(pd.unique(df["replication"]))
        rep_map = {rep: global_rep + i for i, rep in enumerate(batch_reps)}
        df["replication"] = df["replication"].map(rep_map)
        df["batch_id"] = batch_id
        parts.append(df)
        global_rep += len(batch_reps)

    expected_total = args.expected_batches * args.expected_reps_per_batch
    if global_rep != expected_total:
        raise ValueError(f"Expected {expected_total} merged replications, found {global_rep}")

    raw = pd.concat(parts, ignore_index=True)
    if raw.duplicated(["scenario", "replication", "method"]).any():
        raise ValueError("Merged output contains duplicate (scenario, replication, method) rows")

    counts = raw.groupby("replication")["method"].nunique()
    if not (counts == len(EXPECTED_METHODS)).all():
        raise ValueError("Merged output does not contain exactly seven methods for every replication")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / f"scenario_{args.scenario}_raw.csv", index=False)
    summarize_monte_carlo(raw).to_csv(out / f"scenario_{args.scenario}_summary.csv", index=False)
    nuisance = summarize_nuisance(raw)
    if not nuisance.empty:
        nuisance.to_csv(out / f"scenario_{args.scenario}_nuisance.csv", index=False)

    print(
        f"validated and merged {global_rep} replications from {len(files)} batches "
        f"({len(EXPECTED_METHODS)} estimators per replication)"
    )


if __name__ == "__main__":
    main()
