"""Monte Carlo orchestration and summary metrics."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from threadpoolctl import threadpool_limits

from .dgp import ScenarioConfig, generate_data
from .estimators import run_primary_estimators


def _run_one_replication(config: ScenarioConfig, replication: int, rep_seed: int, n_folds: int) -> List[dict]:
    data = generate_data(config, rep_seed)
    results = run_primary_estimators(data, n_folds=n_folds, seed=rep_seed % (2**31 - 1))
    rows: List[dict] = []
    for method, res in results.items():
        estimate = res.estimate
        err = estimate - config.tau if np.isfinite(estimate) else np.nan
        covered = float(res.ci_lower <= config.tau <= res.ci_upper) if res.inference_available and np.isfinite(res.ci_lower) else np.nan
        row = {
            "scenario": config.name,
            "replication": replication,
            "replication_seed": rep_seed,
            "method": method,
            "true_ate": config.tau,
            "estimate": estimate,
            "error": err,
            "abs_error": abs(err) if np.isfinite(err) else np.nan,
            "sq_error": err**2 if np.isfinite(err) else np.nan,
            "std_error": res.std_error,
            "ci_lower": res.ci_lower,
            "ci_upper": res.ci_upper,
            "covered_95": covered,
            "ci_width": res.ci_upper - res.ci_lower if res.inference_available else np.nan,
            "runtime_seconds": res.runtime_seconds,
            "inference_available": res.inference_available,
            "status": res.status,
        }
        row.update(res.diagnostics)
        rows.append(row)
    return rows


def _run_one_replication_capped(task) -> List[dict]:
    config, replication, rep_seed, n_folds = task
    with threadpool_limits(limits=1):
        return _run_one_replication(config, replication, rep_seed, n_folds)


def run_monte_carlo(config: ScenarioConfig, n_replications: int = 500, n_folds: int = 5, seed: int = 20260811, n_jobs: int = 1) -> pd.DataFrame:
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_replications)
    rep_seeds = [int(child.generate_state(1)[0]) for child in child_seeds]
    tasks = [(config, r, rep_seed, n_folds) for r, rep_seed in enumerate(rep_seeds)]
    if n_jobs == 1:
        chunks = [_run_one_replication_capped(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx, max_tasks_per_child=10) as pool:
            chunks = list(pool.map(_run_one_replication_capped, tasks, chunksize=1))
    rows = [row for chunk in chunks for row in chunk]
    return pd.DataFrame(rows)


def summarize_monte_carlo(results: pd.DataFrame) -> pd.DataFrame:
    def summarize_group(g: pd.DataFrame) -> pd.Series:
        ok = g[np.isfinite(g["estimate"])]
        infer = g[g["inference_available"] & np.isfinite(g["std_error"])]
        return pd.Series({
            "replications": len(g),
            "successful_replications": len(ok),
            "failure_rate": 1 - len(ok) / len(g) if len(g) else np.nan,
            "mean_estimate": ok["estimate"].mean(),
            "median_estimate": ok["estimate"].median(),
            "bias": ok["error"].mean(),
            "mcse_bias": ok["estimate"].std(ddof=1) / np.sqrt(len(ok)) if len(ok) > 1 else np.nan,
            "rmse": np.sqrt(ok["sq_error"].mean()),
            "median_abs_error": ok["abs_error"].median(),
            "p95_abs_error": ok["abs_error"].quantile(0.95),
            "empirical_sd": ok["estimate"].std(ddof=1),
            "average_estimated_se": infer["std_error"].mean() if len(infer) else np.nan,
            "coverage_95": infer["covered_95"].mean() if len(infer) else np.nan,
            "mcse_coverage_95": np.sqrt(infer["covered_95"].mean() * (1 - infer["covered_95"].mean()) / len(infer)) if len(infer) else np.nan,
            "average_ci_width": infer["ci_width"].mean() if len(infer) else np.nan,
            "mean_runtime_seconds": g["runtime_seconds"].mean(),
        })
    return results.groupby(["scenario", "method"], sort=False, dropna=False).apply(summarize_group, include_groups=False).reset_index()


def summarize_nuisance(results: pd.DataFrame) -> pd.DataFrame:
    cols = ["propensity_brier","propensity_log_loss","propensity_auc","propensity_calibration_error","propensity_rmse_truth","l_rmse_truth","mu0_rmse_truth","mu1_rmse_truth","estimated_extreme_propensity_share_005"]
    available = [c for c in cols if c in results.columns]
    if not available:
        return pd.DataFrame()
    return results.groupby(["scenario", "method"], sort=False)[available].mean(numeric_only=True).reset_index()
