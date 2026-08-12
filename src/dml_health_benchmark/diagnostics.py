"""Diagnostics for nuisance prediction and overlap."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, roc_auc_score


def calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Simple equal-frequency expected calibration error for binary probabilities."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    total = len(y)
    err = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        err += (len(idx) / total) * abs(float(np.mean(y[idx])) - float(np.mean(p[idx])))
    return float(err)


def nuisance_diagnostics(
    d: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
    e_hat: np.ndarray,
    mu0_true: np.ndarray | None = None,
    mu1_true: np.ndarray | None = None,
    e_true: np.ndarray | None = None,
) -> Dict[str, float]:
    p = np.clip(e_hat, 1e-12, 1 - 1e-12)
    out: Dict[str, float] = {
        "propensity_brier": float(brier_score_loss(d, p)),
        "propensity_log_loss": float(log_loss(d, p, labels=[0, 1])),
        "propensity_auc": float(roc_auc_score(d, p)),
        "propensity_calibration_error": calibration_error(d, p),
        "estimated_extreme_propensity_share_005": float(np.mean((p < 0.05) | (p > 0.95))),
    }
    if mu0_true is not None:
        out["mu0_rmse_truth"] = float(np.sqrt(mean_squared_error(mu0_true, mu0_hat)))
    if mu1_true is not None:
        out["mu1_rmse_truth"] = float(np.sqrt(mean_squared_error(mu1_true, mu1_hat)))
    if e_true is not None:
        out["propensity_rmse_truth"] = float(np.sqrt(mean_squared_error(e_true, p)))
    return out


def overlap_diagnostics(d: np.ndarray, e: np.ndarray) -> Dict[str, float]:
    """Compute overlap diagnostics without trimming the analysis sample."""
    p = np.asarray(e, dtype=float)
    valid = np.isfinite(p) & (p > 0) & (p < 1)
    if not np.all(valid):
        return {
            "valid_propensity_share": float(np.mean(valid)),
            "extreme_propensity_share_005": np.nan,
            "extreme_propensity_share_010": np.nan,
            "max_ipw_weight": np.inf,
            "ipw_ess": np.nan,
        }
    w = d / p + (1 - d) / (1 - p)
    ess = (w.sum() ** 2) / np.sum(w**2)
    return {
        "valid_propensity_share": 1.0,
        "extreme_propensity_share_005": float(np.mean((p < 0.05) | (p > 0.95))),
        "extreme_propensity_share_010": float(np.mean((p < 0.10) | (p > 0.90))),
        "max_ipw_weight": float(np.max(w)),
        "ipw_ess": float(ess),
    }


def plr_nuisance_diagnostics(
    d: np.ndarray,
    l_hat: np.ndarray,
    m_hat: np.ndarray,
    l_true: np.ndarray | None = None,
    m_true: np.ndarray | None = None,
) -> Dict[str, float]:
    """Diagnostics for PLR nuisances l(X)=E[Y|X] and m(X)=E[D|X]."""
    p = np.clip(m_hat, 1e-12, 1 - 1e-12)
    out: Dict[str, float] = {
        "propensity_brier": float(brier_score_loss(d, p)),
        "propensity_log_loss": float(log_loss(d, p, labels=[0, 1])),
        "propensity_auc": float(roc_auc_score(d, p)),
        "propensity_calibration_error": calibration_error(d, p),
        "estimated_extreme_propensity_share_005": float(np.mean((p < 0.05) | (p > 0.95))),
    }
    if l_true is not None:
        out["l_rmse_truth"] = float(np.sqrt(mean_squared_error(l_true, l_hat)))
    if m_true is not None:
        out["propensity_rmse_truth"] = float(np.sqrt(mean_squared_error(m_true, p)))
    return out
