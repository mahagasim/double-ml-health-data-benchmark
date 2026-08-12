"""Nuisance learner factories.

Primary L1 penalties are fixed *before* the scientific Monte Carlo. They were
selected on separate calibration seeds using nuisance-prediction metrics only
(not treatment-effect error). This avoids outcome-driven tuning and makes the
large simulation substantially more reproducible than nested CV inside every
outer fold.

XGBoost settings are likewise pre-specified. Random Forest is retained as an
optional learner-sensitivity check rather than a primary estimator.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Frozen after nuisance-only calibration on seeds 9101-9103.
LASSO_ALPHA = 0.05
L1_LOGIT_C = 0.05


def make_lasso_regressor(seed: int | None = None) -> Pipeline:
    # Lasso is deterministic for the default coordinate-descent selection;
    # `seed` remains in the factory signature for a uniform learner API.
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", Lasso(alpha=LASSO_ALPHA, max_iter=3_000, tol=1e-4)),
    ])


def make_lasso_propensity(seed: int | None = None) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=L1_LOGIT_C,
            l1_ratio=1.0,
            solver="liblinear",
            max_iter=500,
            tol=1e-4,
            random_state=seed,
        )),
    ])


def _xgb_common(seed: int) -> dict:
    return dict(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.50,
        reg_lambda=1.0,
        min_child_weight=5,
        n_jobs=1,
        random_state=seed,
        verbosity=0,
    )


def make_xgb_regressor(seed: int):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - optional sensitivity
        raise ImportError("XGBoost is optional; install requirements-validation.txt.") from exc
    return XGBRegressor(**_xgb_common(seed))


def make_xgb_propensity(seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - optional sensitivity
        raise ImportError("XGBoost is optional; install requirements-validation.txt.") from exc
    return XGBClassifier(**_xgb_common(seed), eval_metric="logloss")


def make_rf_regressor(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )


def make_rf_propensity(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight=None,
        random_state=seed,
        n_jobs=1,
    )
