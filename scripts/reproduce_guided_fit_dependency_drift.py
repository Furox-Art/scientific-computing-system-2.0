"""Reproduce the archived guided-fit numerical-stack upgrade case."""

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

SEED = 20261006


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_input(path: Path) -> None:
    x = np.linspace(0.0, 0.9, 96)
    y = 1.8 * np.exp(0.075 * x) + 2.2
    y += 7e-4 * np.sin(7.0 * x) + 3e-4 * np.cos(13.0 * x)
    pd.DataFrame(
        {
            "x": x,
            "y": y,
            "sigma": np.full_like(x, 0.002),
        }
    ).to_csv(path, index=False)


def _result_payload(result: gf.GuidedFitResult) -> dict[str, object]:
    item = result.datasets[0]
    return {
        "model": result.model,
        "trust": result.trust,
        "params": item.params.tolist(),
        "parameter_std": item.parameter_std.tolist(),
        "confidence_95": item.confidence_95.tolist(),
        "rmse": item.rmse,
        "cv_rmse": item.cv_rmse,
        "r_squared": item.r_squared,
        "cross_check_error": item.cross_check_error,
    }


def baseline(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "input.csv"
    _make_input(csv_path)
    dataset = gf.load_csv_dataset(csv_path, "x", "y", "sigma")
    result = gf.run_guided_fit(
        (dataset,),
        "exponential",
        missing_policy="interpolate",
        outlier_policy="keep",
        seed=SEED,
    )
    manifest = gf.save_manifest(
        result,
        (dataset,),
        output_dir / "guided_fit_manifest.json",
        x_column="x",
        y_column="y",
        sigma_column="sigma",
    )
    payload: dict[str, object] = {
        "case": "exponential_weak_offset",
        "mode": "baseline",
        "seed": SEED,
        "environment": _environment(),
        "manifest": manifest.name,
        "result": _result_payload(result),
    }
    _write_json(output_dir / "baseline-result.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def rerun(manifest: Path, output: Path) -> int:
    result = gf.rerun_manifest(manifest)
    payload: dict[str, object] = {
        "case": "exponential_weak_offset",
        "mode": "rerun",
        "seed": SEED,
        "environment": _environment(),
        "stability_warning": result.stability_warning,
        "stability_details": list(result.stability_details),
        "result": _result_payload(result),
    }
    _write_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("--output-dir", type=Path, required=True)

    rerun_parser = subparsers.add_parser("rerun")
    rerun_parser.add_argument("--manifest", type=Path, required=True)
    rerun_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.mode == "baseline":
        return baseline(args.output_dir)
    return rerun(args.manifest, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
