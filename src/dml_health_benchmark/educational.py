"""Pedagogical partially-linear DML implementation.

This is not the main Monte Carlo estimand. It exists to make residualization,
partialling-out, Neyman orthogonality, and cross-fitting transparent before the
project moves to the binary-treatment IRM ATE score.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from .learners import make_lasso_regressor


def manual_plr_dml(X: np.ndarray, d: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 1):
    n = len(y)
    l_hat = np.full(n, np.nan)  # E[Y|X]
    m_hat = np.full(n, np.nan)  # E[D|X]
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_id, (train, test) in enumerate(splitter.split(X)):
        ly = make_lasso_regressor(seed + 1000 + fold_id)
        md = make_lasso_regressor(seed + 2000 + fold_id)
        ly.fit(X[train], y[train])
        md.fit(X[train], d[train])
        l_hat[test] = ly.predict(X[test])
        m_hat[test] = md.predict(X[test])

    y_res = y - l_hat
    d_res = d - m_hat
    theta = float(np.sum(d_res * y_res) / np.sum(d_res**2))
    score = d_res * (y_res - theta * d_res)
    jac = float(np.mean(d_res**2))
    influence = score / jac
    se = float(np.std(influence, ddof=1) / np.sqrt(n))
    return {"theta": theta, "std_error": se, "y_residual": y_res, "d_residual": d_res}
