import numpy as np

from dml_health_benchmark.dgp import SCENARIOS, generate_data, with_overrides
from dml_health_benchmark.estimators import crossfit_irm, crossfit_plr, oracle_aipw


def test_oracle_aipw_recovers_true_effect_in_large_sample():
    data = generate_data(with_overrides(SCENARIOS["C"], n=8_000), seed=1234)
    res = oracle_aipw(data)
    assert abs(res.estimate - 1.0) < 0.08
    assert res.std_error > 0


def test_manual_irm_lasso_smoke():
    data = generate_data(with_overrides(SCENARIOS["A"], n=350), seed=11)
    res = crossfit_irm(data, learner="lasso", n_folds=2, seed=99)
    assert res.status == "ok"
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)
    assert "propensity_auc" in res.diagnostics


def test_manual_plr_lasso_smoke():
    data = generate_data(with_overrides(SCENARIOS["A"], n=350), seed=12)
    res = crossfit_plr(data, learner="lasso", n_folds=2, seed=98)
    assert res.status == "ok"
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)
    assert "l_rmse_truth" in res.diagnostics


def test_manual_plr_rich_lasso_smoke():
    data = generate_data(with_overrides(SCENARIOS["C"], n=350, p=60), seed=13)
    res = crossfit_plr(data, learner="lasso_rich", n_folds=2, seed=97)
    assert res.status == "ok"
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)
    assert res.diagnostics["nuisance_feature_dimension"] > data.X.shape[1]
