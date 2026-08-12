#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dml_health_benchmark.monte_carlo import summarize_monte_carlo, summarize_nuisance
from dml_health_benchmark.plotting import (
    plot_all_rmse_log,
    plot_ci_width_log,
    plot_core_bias,
    plot_coverage_stable,
    plot_estimate_distributions,
    plot_failure_rate,
    plot_nuisance_metric,
    plot_nuisance_vs_causal_error,
    plot_true_extreme_overlap,
)


def main():
    parser = argparse.ArgumentParser(description="Create benchmark figures from saved Monte Carlo outputs.")
    parser.add_argument("csv", nargs="+", help="One or more raw scenario CSV files")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--overlap-summary", default="results/overlap_calibration.csv")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    summary = summarize_monte_carlo(raw)
    nuisance = summarize_nuisance(raw)

    plot_core_bias(summary, outdir / "bias_core_methods.svg")
    plot_all_rmse_log(summary, outdir / "rmse_all_methods_log.svg")
    plot_coverage_stable(summary, outdir / "coverage_selected_methods.svg")
    plot_ci_width_log(summary, outdir / "ci_width_all_methods_log.svg")
    plot_failure_rate(summary, outdir / "failure_rate.svg")

    if not nuisance.empty:
        plot_nuisance_metric(nuisance, "l_rmse_truth", outdir / "nuisance_outcome_rmse.svg", "Outcome-nuisance RMSE")
        plot_nuisance_metric(nuisance, "propensity_rmse_truth", outdir / "nuisance_propensity_rmse.svg", "Propensity RMSE")

    for scenario in raw["scenario"].drop_duplicates():
        plot_estimate_distributions(raw, scenario, outdir / f"sampling_distribution_{scenario}.svg")
    for method in ["dml_plr_lasso", "dml_plr_lasso_rich"]:
        if method in set(raw["method"]):
            plot_nuisance_vs_causal_error(raw, method, outdir / f"nuisance_vs_error_{method}.svg")

    overlap_path = Path(args.overlap_summary)
    if overlap_path.exists():
        plot_true_extreme_overlap(pd.read_csv(overlap_path), outdir / "true_overlap_extremes.svg")


if __name__ == "__main__":
    main()
