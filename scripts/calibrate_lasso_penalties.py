#!/usr/bin/env python
"""Calibrate fixed L1 nuisance penalties using nuisance prediction only.

The calibration intentionally never evaluates treatment-effect error. It uses
separate seeds that are excluded from the scientific Monte Carlo and chooses a
single penalty pair that is reasonable across the sparse-linear, nonlinear,
and weak-overlap scenarios.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dml_health_benchmark.dgp import SCENARIOS, generate_data


def _reg(alpha: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", Lasso(alpha=alpha, max_iter=3_000, tol=1e-4)),
    ])


def _prop(C: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=C, l1_ratio=1.0, solver="liblinear", max_iter=500, tol=1e-4
        )),
    ])


def evaluate(scenario: str, seed: int, alpha: float, C: float, folds: int) -> dict:
    data = generate_data(SCENARIOS[scenario], seed)
    l_hat = np.full(len(data.y), np.nan)
    m_hat = np.full(len(data.y), np.nan)
    split = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in split.split(data.X, data.d):
        r = _reg(alpha)
        p = _prop(C)
        r.fit(data.X[tr], data.y[tr])
        p.fit(data.X[tr], data.d[tr])
        l_hat[te] = r.predict(data.X[te])
        m_hat[te] = p.predict_proba(data.X[te])[:, 1]

    l_true = data.mu0 + data.config.tau * data.e_true
    return {
        "scenario": scenario,
        "seed": seed,
        "alpha": alpha,
        "C": C,
        "l_rmse_truth": mean_squared_error(l_true, l_hat) ** 0.5,
        "propensity_log_loss": log_loss(data.d, m_hat),
        "propensity_rmse_truth": mean_squared_error(data.e_true, m_hat) ** 0.5,
        "propensity_auc": roc_auc_score(data.d, m_hat),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/l1_penalty_calibration.csv")
    args = ap.parse_args()

    seeds = [9101, 9102, 9103]
    alphas = [0.02, 0.03, 0.05, 0.08]
    Cs = [0.05, 0.10, 0.20]
    rows = []
    for scenario in ["B", "C", "D"]:
        for seed in seeds:
            for alpha in alphas:
                for C in Cs:
                    rows.append(evaluate(scenario, seed, alpha, C, folds=5))
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["alpha", "C"], as_index=False)[
            ["l_rmse_truth", "propensity_log_loss", "propensity_rmse_truth", "propensity_auc"]
        ]
        .mean()
        .sort_values(["propensity_log_loss", "l_rmse_truth"])
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out, index=False)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print("\nFrozen primary values: alpha=0.05, C=0.05")


if __name__ == "__main__":
    main()
