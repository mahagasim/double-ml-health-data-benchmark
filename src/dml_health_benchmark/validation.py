"""Optional external validation against the DoubleML package."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .dgp import SimulatedData
from .estimators import crossfit_irm, crossfit_plr
from .learners import make_lasso_propensity, make_lasso_regressor


def validate_manual_irm_with_doubleml(data: SimulatedData, n_folds: int = 5, seed: int = 1):
    try:
        import doubleml as dml
    except ImportError as exc:
        raise RuntimeError("DoubleML is not installed. Install requirements-validation.txt and rerun this check in a network-enabled environment.") from exc
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    smpls = [(train, test) for train, test in splitter.split(data.X, data.d)]
    manual = crossfit_irm(data, learner="lasso", n_folds=n_folds, seed=seed)
    dml_data = dml.DoubleMLData.from_arrays(data.X, data.y, data.d)
    model = dml.DoubleMLIRM(dml_data, ml_g=make_lasso_regressor(seed), ml_m=make_lasso_propensity(seed), n_folds=n_folds, score="ATE", trimming_rule="truncate", trimming_threshold=1e-12, draw_sample_splitting=False)
    model.set_sample_splitting(smpls)
    model.fit()
    package_estimate = float(np.asarray(model.coef).reshape(-1)[0])
    package_se = float(np.asarray(model.se).reshape(-1)[0])
    return {"manual_estimate": manual.estimate,"doubleml_estimate": package_estimate,"estimate_difference": manual.estimate-package_estimate,"manual_se": manual.std_error,"doubleml_se": package_se,"se_difference": manual.std_error-package_se}


def validate_manual_plr_with_doubleml(data: SimulatedData, n_folds: int = 5, seed: int = 1):
    try:
        import doubleml as dml
    except ImportError as exc:
        raise RuntimeError("DoubleML is not installed. Install requirements-validation.txt and rerun this check in a network-enabled environment.") from exc
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    smpls = [(train, test) for train, test in splitter.split(data.X, data.d)]
    manual = crossfit_plr(data, learner="lasso", n_folds=n_folds, seed=seed)
    dml_data = dml.DoubleMLData.from_arrays(data.X, data.y, data.d)
    model = dml.DoubleMLPLR(dml_data, ml_l=make_lasso_regressor(seed), ml_m=make_lasso_propensity(seed), n_folds=n_folds, score="partialling out", draw_sample_splitting=False)
    model.set_sample_splitting(smpls)
    model.fit()
    package_estimate = float(np.asarray(model.coef).reshape(-1)[0])
    package_se = float(np.asarray(model.se).reshape(-1)[0])
    return {"manual_estimate": manual.estimate,"doubleml_estimate": package_estimate,"estimate_difference": manual.estimate-package_estimate,"manual_se": manual.std_error,"doubleml_se": package_se,"se_difference": manual.std_error-package_se}
