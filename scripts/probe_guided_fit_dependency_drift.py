"""Probe guided-fit across pinned SciPy versions for a real drift case.

This script is deterministic: NumPy is pinned by the workflow, the synthetic
measurement perturbations are analytic (no RNG), and guided-fit receives a
fixed seed for repeated cross-validation.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

from cds2 import guided_fit as gf


def _candidate_cases() -> tuple[tuple[str, gf.ModelName, gf.FitDataset], ...]:
    cases: list[tuple[str, gf.ModelName, gf.FitDataset]] = []

    x = np.linspace(0.0, 0.9, 96)
    y = 1.8 * np.exp(0.075 * x) + 2.2
    y += 7e-4 * np.sin(7.0 * x) + 3e-4 * np.cos(13.0 * x)
    cases.append(
        (
            "exponential_weak_offset",
            "exponential",
            gf.FitDataset(
                "exponential_weak_offset",
                x,
                y,
                np.full_like(x, 0.002),
            ),
        )
    )

    x = np.linspace(-0.55, 0.85, 110)
    y = 0.4 + (5.8 - 0.4) / (1.0 + np.exp(-0.75 * (x - 0.1)))
    y += 0.003 * np.sin(9.0 * x) - 0.002 * np.cos(5.0 * x)
    cases.append(
        (
            "logistic_partial_window",
            "logistic",
            gf.FitDataset(
                "logistic_partial_window",
                x,
                y,
                np.full_like(x, 0.01),
            ),
        )
    )

    x = np.linspace(0.8, 2.6, 100)
    y = 0.7 + (6.0 - 0.7) / (1.0 + np.exp(-1.05 * (x - 0.15)))
    y += 0.0025 * np.sin(11.0 * x)
    cases.append(
        (
            "logistic_upper_shoulder",
            "logistic",
            gf.FitDataset(
                "logistic_upper_shoulder",
                x,
                y,
                np.full_like(x, 0.012),
            ),
        )
    )

    x = np.linspace(0.94, 1.22, 90)
    y = 2.05 * np.power(x, 1.32) + 3.1
    y += 4e-4 * np.sin(19.0 * x) + 2e-4 * np.cos(31.0 * x)
    cases.append(
        (
            "power_narrow_window",
            "power",
            gf.FitDataset(
                "power_narrow_window",
                x,
                y,
                np.full_like(x, 0.0015),
            ),
        )
    )

    x = np.linspace(0.0, 3.0, 120)
    y = 2.4 * np.exp(0.18 * x) + 0.9
    y += 0.015 * np.sin(4.0 * x) + 0.006 * np.cos(13.0 * x)
    cases.append(
        (
            "exponential_moderate",
            "exponential",
            gf.FitDataset(
                "exponential_moderate",
                x,
                y,
                np.full_like(x, 0.03),
            ),
        )
    )

    x = np.linspace(-3.0, 4.0, 140)
    y = 0.55 + (5.4 - 0.55) / (1.0 + np.exp(-1.15 * (x - 0.35)))
    y += 0.02 * np.sin(2.7 * x) + 0.008 * np.cos(7.1 * x)
    cases.append(
        (
            "logistic_full_window",
            "logistic",
            gf.FitDataset(
                "logistic_full_window",
                x,
                y,
                np.full_like(x, 0.04),
            ),
        )
    )

    return tuple(cases)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for name, model, dataset in _candidate_cases():
        try:
            result = gf.run_guided_fit(
                (dataset,),
                model,
                missing_policy="interpolate",
                outlier_policy="keep",
                seed=20261006,
            )
            item = result.datasets[0]
            records.append(
                {
                    "case": name,
                    "status": "ok",
                    "model": model,
                    "params": item.params.tolist(),
                    "parameter_std": item.parameter_std.tolist(),
                    "confidence_95": item.confidence_95.tolist(),
                    "rmse": item.rmse,
                    "cv_rmse": item.cv_rmse,
                    "r_squared": item.r_squared,
                    "cross_check_error": item.cross_check_error,
                    "trust": result.trust,
                }
            )
        except Exception as exc:  # noqa: BLE001 - probe records candidate failures
            records.append(
                {
                    "case": name,
                    "status": "error",
                    "model": model,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "seed": 20261006,
        "cases": records,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
