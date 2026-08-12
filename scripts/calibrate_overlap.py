#!/usr/bin/env python
"""Reproduce the pre-Monte-Carlo overlap calibration diagnostics."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from dml_health_benchmark.dgp import SCENARIOS, generate_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = []
    for scenario, config in SCENARIOS.items():
        for r in range(args.draws):
            data = generate_data(config, args.seed + r)
            e = data.e_true
            rows.append({
                "scenario": scenario,
                "draw": r,
                "treatment_prevalence": float(data.d.mean()),
                "true_extreme_share_005": float(np.mean((e < .05) | (e > .95))),
                "true_extreme_share_010": float(np.mean((e < .10) | (e > .90))),
                "true_propensity_q01": float(np.quantile(e, .01)),
                "true_propensity_q99": float(np.quantile(e, .99)),
            })
    raw = pd.DataFrame(rows)
    summary = raw.groupby("scenario", as_index=False).mean(numeric_only=True).drop(columns="draw")
    print(summary.round(4).to_string(index=False))
    if args.output:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(path.with_name(path.stem + "_raw.csv"), index=False)
        summary.to_csv(path, index=False)


if __name__ == "__main__":
    main()
