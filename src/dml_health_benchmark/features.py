"""Deterministic feature dictionaries used by high-dimensional nuisance learners.

The rich dictionary is intentionally *not* data-adaptive: it applies the same
transformations to every covariate, so there is no variable-selection oracle or
outcome leakage. In p=400 scenarios it expands the nuisance design to p*=1999,
which is larger than N=1000 and creates a genuine regularized high-dimensional
prediction problem.
"""
from __future__ import annotations

import numpy as np


def rich_nonlinear_dictionary(X: np.ndarray, rho: float = 0.30) -> np.ndarray:
    """Return a fixed nonlinear dictionary for every input column.

    Blocks:
    - raw linear terms X_j;
    - centered squares X_j^2 - 1;
    - sin(X_j);
    - centered zero-threshold indicators 1(X_j>0)-1/2;
    - centered adjacent interactions X_j X_{j+1} - rho.

    Under the benchmark's standard-normal AR(1) covariates, the square,
    threshold, and adjacent-interaction blocks are centered in expectation.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[1] < 1:
        raise ValueError("X must contain at least one covariate.")

    blocks = [
        X,
        X**2 - 1.0,
        np.sin(X),
        (X > 0.0).astype(float) - 0.5,
    ]
    if X.shape[1] > 1:
        blocks.append(X[:, :-1] * X[:, 1:] - rho)
    return np.column_stack(blocks)


def rich_dictionary_dimension(p: int) -> int:
    if p < 1:
        raise ValueError("p must be positive.")
    return 4 * p + max(p - 1, 0)
