#!/usr/bin/env python
from dml_health_benchmark.dgp import SCENARIOS, generate_data
from dml_health_benchmark.validation import validate_manual_plr_with_doubleml

for scenario, seed in {"A": 20260811, "C": 20260813}.items():
    data = generate_data(SCENARIOS[scenario], seed)
    result = validate_manual_plr_with_doubleml(data, n_folds=5, seed=314159)
    print(scenario, result)
    assert abs(result["estimate_difference"]) < 1e-6
    assert abs(result["se_difference"]) < 1e-3

print("External DoubleML PLR validation passed.")
