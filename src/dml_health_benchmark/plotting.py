"""Publication-style figures driven from saved Monte Carlo outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary_metric(summary: pd.DataFrame, metric: str, path, ylabel: str | None = None):
    pivot = summary.pivot(index="method", columns="scenario", values=metric)
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel("Estimator")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title())
    ax.set_title(f"{ylabel or metric.replace('_', ' ').title()} across DGP scenarios")
    ax.legend(title="Scenario")
    _save(ax.figure, path)


def plot_estimate_distributions(results: pd.DataFrame, scenario: str, path):
    subset = results[(results["scenario"] == scenario) & np.isfinite(results["estimate"])].copy()
    methods = list(subset["method"].drop_duplicates())
    data = [subset.loc[subset["method"] == m, "estimate"].to_numpy() for m in methods]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, tick_labels=methods, showfliers=False)
    ax.axhline(float(subset["true_ate"].iloc[0]), linestyle="--", linewidth=1.5)
    ax.set_ylabel("Estimated ATE")
    ax.set_title(f"Sampling distributions — Scenario {scenario}")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, path)


def plot_nuisance_vs_causal_error(results: pd.DataFrame, method: str, path):
    subset = results[(results["method"] == method) & np.isfinite(results["abs_error"])].copy()
    if "propensity_auc" not in subset.columns:
        raise ValueError("propensity_auc is not available for this method.")
    subset = subset[np.isfinite(subset["propensity_auc"])]
    fig, ax = plt.subplots(figsize=(7, 5))
    for scenario, group in subset.groupby("scenario"):
        # Keep reviewer-facing SVGs lightweight while preserving all raw diagnostics in the data.
        if len(group) > 50:
            group = group.sample(n=50, random_state=20260812)
        ax.scatter(group["propensity_auc"], group["abs_error"], alpha=0.6, label=f"Scenario {scenario}")
    ax.set_xlabel("Out-of-fold propensity AUC")
    ax.set_ylabel("Absolute ATE error")
    ax.set_title(f"Treatment prediction vs causal error — {method}")
    ax.legend()
    _save(fig, path)


def plot_overlap(e_true: np.ndarray, d: np.ndarray, scenario: str, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(e_true[d == 0], bins=30, alpha=0.55, density=True, label="Untreated")
    ax.hist(e_true[d == 1], bins=30, alpha=0.55, density=True, label="Treated")
    ax.set_xlabel("True propensity score")
    ax.set_ylabel("Density")
    ax.set_title(f"Treatment overlap — Scenario {scenario}")
    ax.legend()
    _save(fig, path)

METHOD_LABELS = {
    "difference_in_means": "Difference in means",
    "ols_adjusted": "OLS adjustment",
    "lasso_plugin": "Naive LASSO plug-in",
    "parametric_ipw": "Parametric IPW",
    "parametric_aipw": "Parametric AIPW",
    "dml_plr_lasso": "DML: raw LASSO",
    "dml_plr_lasso_rich": "DML: rich-dictionary LASSO",
}


def _label_methods(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["method_label"] = out["method"].map(METHOD_LABELS).fillna(out["method"])
    return out


def plot_core_bias(summary: pd.DataFrame, path):
    """Bias for the stable/core estimators, leaving exploding weighting estimators to log-scale plots."""
    methods = [
        "difference_in_means", "ols_adjusted", "lasso_plugin",
        "dml_plr_lasso", "dml_plr_lasso_rich",
    ]
    sub = _label_methods(summary[summary["method"].isin(methods)])
    pivot = sub.pivot(index="scenario", columns="method_label", values="bias")
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Bias")
    ax.set_title("Causal-effect bias across simulation scenarios")
    ax.legend(title="Estimator", fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    _save(ax.figure, path)


def plot_all_rmse_log(summary: pd.DataFrame, path):
    sub = _label_methods(summary[np.isfinite(summary["rmse"])])
    pivot = sub.pivot(index="scenario", columns="method_label", values="rmse")
    ax = pivot.plot(kind="bar", figsize=(12, 5), logy=True)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title("RMSE across all estimators — log scale exposes instability")
    ax.legend(title="Estimator", fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    _save(ax.figure, path)


def plot_coverage_stable(summary: pd.DataFrame, path):
    methods = ["ols_adjusted", "parametric_aipw", "dml_plr_lasso", "dml_plr_lasso_rich"]
    sub = _label_methods(summary[summary["method"].isin(methods) & np.isfinite(summary["coverage_95"])])
    pivot = sub.pivot(index="scenario", columns="method_label", values="coverage_95")
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.axhline(0.95, linestyle="--", linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Empirical 95% CI coverage")
    ax.set_title("Empirical confidence-interval coverage")
    ax.legend(title="Estimator", fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    _save(ax.figure, path)


def plot_ci_width_log(summary: pd.DataFrame, path):
    sub = _label_methods(summary[np.isfinite(summary["average_ci_width"])])
    pivot = sub.pivot(index="scenario", columns="method_label", values="average_ci_width")
    ax = pivot.plot(kind="bar", figsize=(12, 5), logy=True)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Average 95% CI width (log scale)")
    ax.set_title("Interval width reveals when nominal coverage is uninformative")
    ax.legend(title="Estimator", fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    _save(ax.figure, path)


def plot_failure_rate(summary: pd.DataFrame, path):
    sub = _label_methods(summary[summary["failure_rate"] > 0])
    if sub.empty:
        return
    pivot = sub.pivot(index="scenario", columns="method_label", values="failure_rate").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Estimator failure rate")
    ax.set_title("Numerical failures under high dimensionality and weak overlap")
    ax.legend(title="Estimator", fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    _save(ax.figure, path)


def plot_nuisance_metric(nuisance: pd.DataFrame, metric: str, path, ylabel: str):
    methods = ["dml_plr_lasso", "dml_plr_lasso_rich"]
    sub = _label_methods(nuisance[nuisance["method"].isin(methods)])
    pivot = sub.pivot(index="scenario", columns="method_label", values=metric)
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by nuisance learner")
    ax.legend(title="DML nuisance learner", fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    _save(ax.figure, path)


def plot_true_extreme_overlap(overlap_summary: pd.DataFrame, path):
    frame = overlap_summary.set_index("scenario")[["true_extreme_share_005", "true_extreme_share_010"]]
    frame = frame.rename(columns={
        "true_extreme_share_005": "Outside [0.05, 0.95]",
        "true_extreme_share_010": "Outside [0.10, 0.90]",
    })
    ax = frame.plot(kind="bar", figsize=(8, 5))
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Share of true propensity scores")
    ax.set_title("True overlap stress by scenario")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Extreme propensity definition")
    _save(ax.figure, path)
