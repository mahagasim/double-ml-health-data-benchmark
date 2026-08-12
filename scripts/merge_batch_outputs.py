#!/usr/bin/env python
"""Merge independently executed Monte Carlo batch outputs into one scenario result."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dml_health_benchmark.monte_carlo import summarize_monte_carlo, summarize_nuisance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list("ABCD"), required=True)
    ap.add_argument("--batch-root", required=True)
    ap.add_argument("--output-dir", default="results")
    args = ap.parse_args()

    root = Path(args.batch_root)
    files = sorted(root.glob(f"batch_*/scenario_{args.scenario}_raw.csv"))
    if not files:
        raise FileNotFoundError(f"No batch raw files found under {root}")

    parts = []
    global_rep = 0
    for batch_id, path in enumerate(files):
        df = pd.read_csv(path)
        batch_reps = list(pd.unique(df["replication"]))
        rep_map = {rep: global_rep + i for i, rep in enumerate(batch_reps)}
        df["replication"] = df["replication"].map(rep_map)
        df["batch_id"] = batch_id
        parts.append(df)
        global_rep += len(batch_reps)

    raw = pd.concat(parts, ignore_index=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / f"scenario_{args.scenario}_raw.csv", index=False)
    summarize_monte_carlo(raw).to_csv(out / f"scenario_{args.scenario}_summary.csv", index=False)
    nuisance = summarize_nuisance(raw)
    if not nuisance.empty:
        nuisance.to_csv(out / f"scenario_{args.scenario}_nuisance.csv", index=False)
    print(f"merged {global_rep} replications from {len(files)} batches")


if __name__ == "__main__":
    main()
