import numpy as np

from dml_health_benchmark.dgp import SCENARIOS, generate_data, with_overrides


def test_true_ate_is_exact_by_construction():
    data = generate_data(with_overrides(SCENARIOS["A"], n=500), seed=123)
    assert np.allclose(data.y1 - data.y0, 1.0)
    assert np.isclose(data.true_ate, 1.0)


def test_high_dimensional_roles_have_expected_shape():
    data = generate_data(with_overrides(SCENARIOS["B"], n=100), seed=42)
    assert data.X.shape == (100, 400)
    assert data.d.shape == (100,)
    assert np.all((data.e_true > 0) & (data.e_true < 1))


def test_scenario_d_has_weaker_overlap_than_c():
    c = generate_data(with_overrides(SCENARIOS["C"], n=20_000), seed=7)
    d = generate_data(with_overrides(SCENARIOS["D"], n=20_000), seed=7)
    extreme_c = np.mean((c.e_true < 0.05) | (c.e_true > 0.95))
    extreme_d = np.mean((d.e_true < 0.05) | (d.e_true > 0.95))
    assert extreme_d > extreme_c + 0.08
