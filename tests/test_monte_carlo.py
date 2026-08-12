from dml_health_benchmark.dgp import SCENARIOS, with_overrides
from dml_health_benchmark.monte_carlo import run_monte_carlo, summarize_monte_carlo


def test_summary_columns_from_tiny_run():
    # Keep this deliberately tiny; it is a smoke test, not a scientific result.
    cfg = with_overrides(SCENARIOS["A"], n=250)
    raw = run_monte_carlo(cfg, n_replications=1, n_folds=2, seed=5)
    summary = summarize_monte_carlo(raw)
    assert set(["bias", "rmse", "empirical_sd", "coverage_95"]).issubset(summary.columns)
    assert len(raw) == 7
