"""Synthetic data-generating processes for the DML health benchmark.

The core study is deliberately simulation-based so the true ATE and nuisance
functions are known. Estimators only receive (Y, D, X); truth is retained for
Monte Carlo evaluation and nuisance diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    n: int = 1_000
    p: int = 10
    rho: float = 0.30
    tau: float = 1.0
    outcome_noise_sd: float = 1.0
    nonlinear: bool = False
    overlap_scale: float = 1.0
    description: str = ""


@dataclass
class SimulatedData:
    X: np.ndarray
    d: np.ndarray
    y: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    e_true: np.ndarray
    config: ScenarioConfig

    @property
    def true_ate(self) -> float:
        return float(np.mean(self.y1 - self.y0))


SCENARIOS: Dict[str, ScenarioConfig] = {
    "A": ScenarioConfig(
        name="A",
        p=10,
        description="Low-dimensional linear confounding with good overlap.",
    ),
    "B": ScenarioConfig(
        name="B",
        p=400,
        description="High-dimensional sparse linear confounding with good overlap.",
    ),
    "C": ScenarioConfig(
        name="C",
        p=400,
        nonlinear=True,
        description="High-dimensional nonlinear/interacting confounding with good overlap.",
    ),
    "D": ScenarioConfig(
        name="D",
        p=400,
        nonlinear=True,
        overlap_scale=2.25,
        description="Scenario C outcome structure with deliberately weakened overlap.",
    ),
}


def ar1_gaussian(n: int, p: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Generate N(0, Sigma) covariates with Sigma[j,k] = rho**|j-k| efficiently."""
    z = rng.normal(size=(n, p))
    X = np.empty_like(z)
    X[:, 0] = z[:, 0]
    innovation_sd = np.sqrt(1.0 - rho**2)
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + innovation_sd * z[:, j]
    return X


def covariate_roles(p: int) -> Dict[str, np.ndarray]:
    """Return zero-based indices for causal/predictive roles used in the DGP."""
    if p < 10:
        raise ValueError("The benchmark requires at least 10 covariates.")
    roles = {
        "confounders": np.arange(0, min(10, p)),
        "treatment_only": np.arange(10, min(15, p)),
        "outcome_only": np.arange(15, min(20, p)),
        "noise": np.arange(min(20, p), p),
    }
    return roles


def _linear_treatment_index(X: np.ndarray) -> np.ndarray:
    conf = np.array([0.55, -0.50, 0.45, -0.40, 0.35, -0.30, 0.25, -0.20, 0.15, -0.10])
    eta = X[:, :10] @ conf
    if X.shape[1] >= 15:
        treatment_only = np.array([0.30, -0.25, 0.20, -0.18, 0.15])
        eta = eta + X[:, 10:15] @ treatment_only
    return eta


def _linear_outcome_mean(X: np.ndarray) -> np.ndarray:
    conf = np.array([0.80, -0.70, 0.60, -0.55, 0.50, -0.45, 0.40, -0.35, 0.30, -0.25])
    mu = X[:, :10] @ conf
    if X.shape[1] >= 20:
        outcome_only = np.array([0.45, -0.40, 0.35, -0.30, 0.25])
        mu = mu + X[:, 15:20] @ outcome_only
    return mu


def _nonlinear_treatment_index(X: np.ndarray, rho: float) -> np.ndarray:
    x = lambda j: X[:, j - 1]
    # Center nonlinear terms where their expectations are known under the AR(1) Gaussian DGP.
    eta = (
        0.55 * x(1)
        - 0.45 * x(2)
        + 0.35 * (x(3) * x(4) - rho)
        + 0.30 * (x(5) ** 2 - 1.0)
        - 0.30 * np.sin(x(6))
        + 0.25 * ((x(7) > 0).astype(float) - 0.5)
        + 0.20 * (x(8) * x(9) - rho)
        - 0.15 * x(10)
    )
    if X.shape[1] >= 15:
        treatment_only = np.array([0.30, -0.25, 0.20, -0.18, 0.15])
        eta = eta + X[:, 10:15] @ treatment_only
    return eta


def _nonlinear_outcome_mean(X: np.ndarray, rho: float) -> np.ndarray:
    x = lambda j: X[:, j - 1]
    mu = (
        0.75 * x(1)
        - 0.65 * x(2)
        + 0.55 * (x(3) * x(4) - rho)
        + 0.45 * (x(5) ** 2 - 1.0)
        + 0.40 * np.sin(x(6))
        + 0.35 * ((x(7) > 0).astype(float) - 0.5)
        + 0.30 * (x(8) * x(9) - rho)
        - 0.25 * x(10)
    )
    if X.shape[1] >= 20:
        outcome_only = np.array([0.45, -0.40, 0.35, -0.30, 0.25])
        mu = mu + X[:, 15:20] @ outcome_only
    return mu


def true_nuisance_functions(X: np.ndarray, config: ScenarioConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return true (mu0, mu1, e) for a supplied covariate matrix."""
    if config.nonlinear:
        eta = _nonlinear_treatment_index(X, config.rho)
        mu0 = _nonlinear_outcome_mean(X, config.rho)
    else:
        eta = _linear_treatment_index(X)
        mu0 = _linear_outcome_mean(X)
    e_true = expit(config.overlap_scale * eta)
    mu1 = mu0 + config.tau
    return mu0, mu1, e_true


def generate_data(config: ScenarioConfig, seed: int) -> SimulatedData:
    """Generate one observed dataset and retain hidden simulation truth."""
    rng = np.random.default_rng(seed)
    X = ar1_gaussian(config.n, config.p, config.rho, rng)
    mu0, mu1, e_true = true_nuisance_functions(X, config)
    d = rng.binomial(1, e_true).astype(int)
    eps = rng.normal(0.0, config.outcome_noise_sd, size=config.n)
    y0 = mu0 + eps
    y1 = mu1 + eps
    y = np.where(d == 1, y1, y0)
    return SimulatedData(X=X, d=d, y=y, y0=y0, y1=y1, mu0=mu0, mu1=mu1, e_true=e_true, config=config)


def with_overrides(base: ScenarioConfig, **kwargs) -> ScenarioConfig:
    """Create a modified immutable scenario config for sensitivity analyses."""
    values = base.__dict__.copy()
    values.update(kwargs)
    return ScenarioConfig(**values)
