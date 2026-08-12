from __future__ import annotations

import argparse
from pathlib import Path

from .dgp import SCENARIOS
from .monte_carlo import run_monte_carlo, summarize_monte_carlo, summarize_nuisance


def main():
    parser = argparse.ArgumentParser(description="Run the DML health-data Monte Carlo benchmark.")
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), required=True)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = run_monte_carlo(SCENARIOS[args.scenario], args.replications, args.folds, args.seed, args.jobs)
    raw.to_csv(outdir / f"scenario_{args.scenario}_raw.csv", index=False)
    summarize_monte_carlo(raw).to_csv(outdir / f"scenario_{args.scenario}_summary.csv", index=False)
    nuisance = summarize_nuisance(raw)
    if not nuisance.empty:
        nuisance.to_csv(outdir / f"scenario_{args.scenario}_nuisance.csv", index=False)
