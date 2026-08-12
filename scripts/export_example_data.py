#!/usr/bin/env python
"""Export one reproducible synthetic dataset for inspection or teaching."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dml_health_benchmark.dgp import SCENARIOS, generate_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), default="A")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-truth", action="store_true")
    args = parser.parse_args()

    data = generate_data(SCENARIOS[args.scenario], args.seed)
    frame = pd.DataFrame(data.X, columns=[f"X{j+1}" for j in range(data.X.shape[1])])
    frame.insert(0, "D", data.d)
    frame.insert(0, "Y", data.y)

    if args.include_truth:
        frame["Y0_true"] = data.y0
        frame["Y1_true"] = data.y1
        frame["mu0_true"] = data.mu0
        frame["mu1_true"] = data.mu1
        frame["propensity_true"] = data.e_true

    output = Path(args.output or f"data/examples/scenario_{args.scenario}_seed{args.seed}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
