"""Causal estimators used in the benchmark.

The primary DML estimator is a manually implemented cross-fitted partialling-
out estimator for the partially linear model. A binary-treatment IRM/AIPW DML
estimator is retained as a secondary extension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from time import perf_counter
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .diagnostics import nuisance_diagnostics, overlap_diagnostics, plr_nuisance_diagnostics
from .dgp import SimulatedData
from .features import rich_nonlinear_dictionary
from .learners import (
    make_lasso_propensity,
    make_lasso_regressor,
    make_rf_propensity,
    make_rf_regressor,
    make_xgb_propensity,
    make_xgb_regressor,
)


Z_975 = 1.959963984540054


@dataclass
class EstimatorResult:
    method: str
    estimate: float
    std_error: float = np.nan
    ci_lower: float = np.nan
    ci_upper: float = np.nan
    runtime_seconds: float = np.nan
    inference_available: bool = False
    diagnostics: Dict[str, float] = field(default_factory=dict)
    status: str = "ok"


def _finalize(method: str, estimate: float, std_error: float, started: float, diagnostics=None, inference=True) -> EstimatorResult:
    if diagnostics is None:
        diagnostics = {}
    if inference and np.isfinite(std_error):
        lo = estimate - Z_975 * std_error
        hi = estimate + Z_975 * std_error
    else:
        lo = hi = np.nan
    return EstimatorResult(
        method=method,
        estimate=float(estimate),
        std_error=float(std_error) if np.isfinite(std_error) else np.nan,
        ci_lower=float(lo) if np.isfinite(lo) else np.nan,
        ci_upper=float(hi) if np.isfinite(hi) else np.nan,
        runtime_seconds=float(perf_counter() - started),
        inference_available=bool(inference and np.isfinite(std_error)),
        diagnostics=diagnostics,
    )


def difference_in_means(data: SimulatedData) -> EstimatorResult:
    started = perf_counter()
    X = sm.add_constant(data.d.astype(float))
    fit = sm.OLS(data.y, X).fit(cov_type="HC3")
    return _finalize("difference_in_means", fit.params[1], fit.bse[1], started)


def ols_adjusted(data: SimulatedData) -> EstimatorResult:
    started = perf_counter()
    X = sm.add_constant(np.column_stack([data.d, data.X]))
    fit = sm.OLS(data.y, X).fit(cov_type="HC3")
    return _finalize("ols_adjusted", fit.params[1], fit.bse[1], started)


def _fit_parametric_propensity(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=500)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Setting penalty=None will ignore.*")
        model.fit(X, d)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def _valid_propensity(e: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(e)) and np.all(e > 0.0) and np.all(e < 1.0))


def parametric_ipw(data: SimulatedData) -> EstimatorResult:
    started = perf_counter()
    e = _fit_parametric_propensity(data.X, data.d)
    if not _valid_propensity(e):
        return EstimatorResult(method="parametric_ipw", estimate=np.nan, runtime_seconds=perf_counter() - started, status="invalid_propensity")
    pseudo = data.d * data.y / e - (1 - data.d) * data.y / (1 - e)
    tau_hat = float(np.mean(pseudo))
    se = float(np.std(pseudo - tau_hat, ddof=1) / np.sqrt(len(data.y)))
    diag = overlap_diagnostics(data.d, e)
    return _finalize("parametric_ipw", tau_hat, se, started, diag, inference=True)


def parametric_aipw(data: SimulatedData) -> EstimatorResult:
    started = perf_counter()
    e = _fit_parametric_propensity(data.X, data.d)
    if not _valid_propensity(e):
        return EstimatorResult(method="parametric_aipw", estimate=np.nan, runtime_seconds=perf_counter() - started, status="invalid_propensity")
    m0 = LinearRegression().fit(data.X[data.d == 0], data.y[data.d == 0])
    m1 = LinearRegression().fit(data.X[data.d == 1], data.y[data.d == 1])
    mu0_hat = m0.predict(data.X)
    mu1_hat = m1.predict(data.X)
    pseudo = mu1_hat - mu0_hat + data.d * (data.y - mu1_hat) / e - (1 - data.d) * (data.y - mu0_hat) / (1 - e)
    tau_hat = float(np.mean(pseudo))
    se = float(np.std(pseudo - tau_hat, ddof=1) / np.sqrt(len(data.y)))
    diag = overlap_diagnostics(data.d, e)
    return _finalize("parametric_aipw", tau_hat, se, started, diag, inference=True)


def _crossfit_outcome_predictions(data: SimulatedData, reg_factory: Callable[[int], object], n_folds: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(data.y)
    mu0_hat = np.full(n, np.nan)
    mu1_hat = np.full(n, np.nan)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_id, (train, test) in enumerate(splitter.split(data.X, data.d)):
        tr0 = train[data.d[train] == 0]
        tr1 = train[data.d[train] == 1]
        if len(tr0) < 10 or len(tr1) < 10:
            raise RuntimeError("Too few treated/control observations in an outer training fold.")
        r0 = reg_factory(seed + 1000 + fold_id)
        r1 = reg_factory(seed + 2000 + fold_id)
        r0.fit(data.X[tr0], data.y[tr0])
        r1.fit(data.X[tr1], data.y[tr1])
        mu0_hat[test] = r0.predict(data.X[test])
        mu1_hat[test] = r1.predict(data.X[test])
    return mu0_hat, mu1_hat


def lasso_plugin(data: SimulatedData, n_folds: int = 5, seed: int = 1) -> EstimatorResult:
    """Naive full-sample LASSO g-formula plug-in estimator."""
    started = perf_counter()
    tr0 = np.flatnonzero(data.d == 0)
    tr1 = np.flatnonzero(data.d == 1)
    r0 = make_lasso_regressor(seed + 1000)
    r1 = make_lasso_regressor(seed + 2000)
    r0.fit(data.X[tr0], data.y[tr0])
    r1.fit(data.X[tr1], data.y[tr1])
    mu0_hat = r0.predict(data.X)
    mu1_hat = r1.predict(data.X)
    tau_hat = float(np.mean(mu1_hat - mu0_hat))
    diag = {
        "mu0_rmse_truth": float(np.sqrt(np.mean((mu0_hat - data.mu0) ** 2))),
        "mu1_rmse_truth": float(np.sqrt(np.mean((mu1_hat - data.mu1) ** 2))),
    }
    return _finalize("lasso_plugin", tau_hat, np.nan, started, diag, inference=False)


def crossfit_irm(data: SimulatedData, learner: str, n_folds: int = 5, seed: int = 1) -> EstimatorResult:
    started = perf_counter()
    if learner == "lasso":
        reg_factory = make_lasso_regressor
        prop_factory = make_lasso_propensity
        method = "dml_irm_lasso"
    elif learner == "rf":
        reg_factory = make_rf_regressor
        prop_factory = make_rf_propensity
        method = "dml_irm_random_forest"
    else:
        raise ValueError("learner must be 'lasso' or 'rf'")
    n = len(data.y)
    mu0_hat = np.full(n, np.nan)
    mu1_hat = np.full(n, np.nan)
    e_hat = np.full(n, np.nan)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_id, (train, test) in enumerate(splitter.split(data.X, data.d)):
        tr0 = train[data.d[train] == 0]
        tr1 = train[data.d[train] == 1]
        if len(tr0) < 10 or len(tr1) < 10:
            return EstimatorResult(method=method, estimate=np.nan, runtime_seconds=perf_counter() - started, status="small_training_arm")
        r0 = reg_factory(seed + 1000 + fold_id)
        r1 = reg_factory(seed + 2000 + fold_id)
        pm = prop_factory(seed + 3000 + fold_id)
        r0.fit(data.X[tr0], data.y[tr0])
        r1.fit(data.X[tr1], data.y[tr1])
        pm.fit(data.X[train], data.d[train])
        mu0_hat[test] = r0.predict(data.X[test])
        mu1_hat[test] = r1.predict(data.X[test])
        e_hat[test] = pm.predict_proba(data.X[test])[:, 1]
    if not _valid_propensity(e_hat):
        diag = overlap_diagnostics(data.d, e_hat)
        return EstimatorResult(method=method, estimate=np.nan, runtime_seconds=perf_counter() - started, diagnostics=diag, status="invalid_propensity")
    pseudo = mu1_hat - mu0_hat + data.d * (data.y - mu1_hat) / e_hat - (1 - data.d) * (data.y - mu0_hat) / (1 - e_hat)
    tau_hat = float(np.mean(pseudo))
    influence = pseudo - tau_hat
    se = float(np.std(influence, ddof=1) / np.sqrt(n))
    diag = nuisance_diagnostics(data.d, mu0_hat, mu1_hat, e_hat, mu0_true=data.mu0, mu1_true=data.mu1, e_true=data.e_true)
    diag.update({f"overlap_{k}": v for k, v in overlap_diagnostics(data.d, e_hat).items()})
    return _finalize(method, tau_hat, se, started, diag, inference=True)


def crossfit_plr(data: SimulatedData, learner: str, n_folds: int = 5, seed: int = 1) -> EstimatorResult:
    started = perf_counter()
    if learner == "lasso":
        outcome_factory = make_lasso_regressor
        treatment_factory = make_lasso_propensity
        method = "dml_plr_lasso"
    elif learner == "lasso_rich":
        outcome_factory = make_lasso_regressor
        treatment_factory = make_lasso_propensity
        method = "dml_plr_lasso_rich"
    elif learner == "xgboost":
        outcome_factory = make_xgb_regressor
        treatment_factory = make_xgb_propensity
        method = "dml_plr_xgboost"
    elif learner == "rf":
        outcome_factory = make_rf_regressor
        treatment_factory = make_rf_propensity
        method = "dml_plr_random_forest"
    else:
        raise ValueError("learner must be 'lasso', 'lasso_rich', 'xgboost', or 'rf'")
    n = len(data.y)
    X_nuisance = rich_nonlinear_dictionary(data.X, data.config.rho) if learner == "lasso_rich" else data.X
    l_hat = np.full(n, np.nan)
    m_hat = np.full(n, np.nan)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_id, (train, test) in enumerate(splitter.split(X_nuisance, data.d)):
        ly = outcome_factory(seed + 1000 + fold_id)
        md = treatment_factory(seed + 2000 + fold_id)
        ly.fit(X_nuisance[train], data.y[train])
        md.fit(X_nuisance[train], data.d[train])
        l_hat[test] = ly.predict(X_nuisance[test])
        m_hat[test] = md.predict_proba(X_nuisance[test])[:, 1]
    if not _valid_propensity(m_hat):
        return EstimatorResult(method=method, estimate=np.nan, runtime_seconds=perf_counter() - started, diagnostics=overlap_diagnostics(data.d, m_hat), status="invalid_propensity")
    y_res = data.y - l_hat
    d_res = data.d - m_hat
    denom = float(np.mean(d_res**2))
    if not np.isfinite(denom) or denom <= 1e-12:
        return EstimatorResult(method=method, estimate=np.nan, runtime_seconds=perf_counter()-started, status="degenerate_residual_treatment")
    theta_hat = float(np.mean(d_res * y_res) / denom)
    score = d_res * (y_res - theta_hat * d_res)
    influence = score / denom
    se = float(np.std(influence, ddof=1) / np.sqrt(n))
    l_true = data.mu0 + data.config.tau * data.e_true
    diag = plr_nuisance_diagnostics(data.d, l_hat, m_hat, l_true=l_true, m_true=data.e_true)
    diag.update({f"overlap_{k}": v for k, v in overlap_diagnostics(data.d, m_hat).items()})
    diag["nuisance_feature_dimension"] = float(X_nuisance.shape[1])
    return _finalize(method, theta_hat, se, started, diag, inference=True)


def oracle_aipw(data: SimulatedData) -> EstimatorResult:
    started = perf_counter()
    e = data.e_true
    pseudo = data.mu1 - data.mu0 + data.d * (data.y - data.mu1) / e - (1 - data.d) * (data.y - data.mu0) / (1 - e)
    tau_hat = float(np.mean(pseudo))
    se = float(np.std(pseudo - tau_hat, ddof=1) / np.sqrt(len(data.y)))
    return _finalize("oracle_aipw", tau_hat, se, started)


def run_primary_estimators(data: SimulatedData, n_folds: int = 5, seed: int = 1) -> Dict[str, EstimatorResult]:
    estimators = [
        difference_in_means(data),
        ols_adjusted(data),
        lasso_plugin(data, n_folds=n_folds, seed=seed),
        parametric_ipw(data),
        parametric_aipw(data),
        crossfit_plr(data, "lasso", n_folds=n_folds, seed=seed),
        crossfit_plr(data, "lasso_rich", n_folds=n_folds, seed=seed),
    ]
    return {res.method: res for res in estimators}
