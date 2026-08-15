#!/usr/bin/env python
"""Compare the manual raw-LASSO PLR implementation with DoubleMLPLR.

This is an implementation check on two fixed simulated draws; it does not
rerun or alter the frozen Monte Carlo experiment.
"""
from dml_health_benchmark.dgp import SCENARIOS, generate_data
from dml_health_benchmark.validation import validate_manual_plr_with_doubleml

ESTIMATE_TOLERANCE = 1e-5
SE_TOLERANCE = 1e-3

for scenario, seed in {"A": 20260811, "C": 20260813}.items():
    data = generate_data(SCENARIOS[scenario], seed)
    result = validate_manual_plr_with_doubleml(data, n_folds=5, seed=314159)
    print(scenario, result)
    assert abs(result["estimate_difference"]) < ESTIMATE_TOLERANCE
    assert abs(result["se_difference"]) < SE_TOLERANCE

print("External DoubleML PLR validation passed.")
