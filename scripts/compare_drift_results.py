#!/usr/bin/env python3
"""Derive side-by-side drift metrics from a baseline/rerun result pair.

The guided-fit drift case produces two JSON records: a ``baseline`` run and a
``rerun`` replay of the saved manifest. This tool compares them, prints an
old-vs-new side-by-side table, reports absolute and relative deviation for
every fitted quantity, and emits an explicit ``WARNING`` when any measured
change exceeds the configured materiality threshold.

Usage
-----
Compare a real baseline/rerun pair produced by
``scripts/reproduce_guided_fit_dependency_drift.py``::

    python scripts/compare_drift_results.py \
        --baseline case/baseline-result.json \
        --rerun case/upgraded-rerun.json \
        --output case/drift-comparison.json

Add ``--annotated`` to persist the local-run provenance of this comparison.

Exit status is ``0`` when no material drift was detected and ``1`` when a
change exceeded the materiality threshold, so the tool doubles as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

# Fractional change above which a difference is reported as material rather
# than as ordinary floating-point noise. 0.05 == 5 %.
DEFAULT_MATERIALITY_THRESHOLD = 0.05

# Environment record for this comparison itself, recorded so the comparison
# step is as auditable as the runs it compares.
COMPARISON_TOOL = "scripts/compare_drift_results.py"


def _as_matrix(value: Any) -> np.ndarray:
    return np.asarray(cast("list[list[float]]", value), dtype=np.float64)


def _as_vector(value: Any) -> np.ndarray:
    return np.asarray(cast("list[float]", value), dtype=np.float64)


def _relative_change(new: float, old: float) -> float:
    return abs(new - old) / max(abs(old), 1e-12)


def compare(
    baseline: dict[str, Any],
    rerun: dict[str, Any],
    *,
    materiality_threshold: float = DEFAULT_MATERIALITY_THRESHOLD,
) -> dict[str, Any]:
    """Return the full drift comparison between a baseline and a rerun record.

    Both records are the JSON payloads written by
    ``scripts/reproduce_guided_fit_dependency_drift.py``.
    """
    old = cast("dict[str, Any]", baseline["result"])
    new = cast("dict[str, Any]", rerun["result"])

    old_ci, new_ci = _as_matrix(old["confidence_95"]), _as_matrix(new["confidence_95"])
    if old_ci.shape != new_ci.shape:
        msg = (
            f"confidence interval shape changed: {old_ci.shape} -> {new_ci.shape}; "
            "cannot compare element-wise"
        )
        raise ValueError(msg)

    ci_delta = new_ci - old_ci
    ci_scale = float(np.linalg.norm(old_ci))
    max_abs_ci_bound_shift = float(np.max(np.abs(ci_delta)))
    relative_ci_l2 = float(np.linalg.norm(ci_delta)) / max(ci_scale, 1e-12)

    old_params, new_params = _as_vector(old["params"]), _as_vector(new["params"])
    param_delta = new_params - old_params
    parameter_l2_shift = float(np.linalg.norm(param_delta))
    parameter_relative = parameter_l2_shift / max(float(np.linalg.norm(old_params)), 1e-12)

    old_std, new_std = _as_vector(old["parameter_std"]), _as_vector(new["parameter_std"])

    rmse_relative = _relative_change(float(new["rmse"]), float(old["rmse"]))
    cv_rmse_relative = _relative_change(float(new["cv_rmse"]), float(old["cv_rmse"]))
    r_squared_change = abs(float(new["r_squared"]) - float(old["r_squared"]))
    cross_check_change = abs(float(new["cross_check_error"]) - float(old["cross_check_error"]))

    material_ci_drift = relative_ci_l2 > materiality_threshold
    material_fit_drift = max(rmse_relative, parameter_relative) > materiality_threshold
    trust_changed = old["trust"] != new["trust"]

    def per_parameter() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index in range(old_params.size):
            rows.append(
                {
                    "index": index,
                    "baseline": float(old_params[index]),
                    "rerun": float(new_params[index]),
                    "absolute_deviation": float(param_delta[index]),
                    "relative_deviation": _relative_change(
                        float(new_params[index]), float(old_params[index])
                    ),
                    "baseline_std": float(old_std[index]),
                    "rerun_std": float(new_std[index]),
                }
            )
        return rows

    def per_ci_bound() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(old_ci.shape[0]):
            for col, bound in enumerate(("lower", "upper")):
                old_bound = float(old_ci[row, col])
                new_bound = float(new_ci[row, col])
                rows.append(
                    {
                        "parameter_index": row,
                        "bound": bound,
                        "baseline": old_bound,
                        "rerun": new_bound,
                        "absolute_deviation": abs(new_bound - old_bound),
                        "relative_deviation": _relative_change(new_bound, old_bound),
                    }
                )
        return rows

    return {
        "schema_version": 1,
        "case": baseline.get("case"),
        "model": old["model"],
        "seed": baseline.get("seed"),
        "comparison_tool": COMPARISON_TOOL,
        "environments": {
            "baseline": baseline.get("environment", {}),
            "rerun": rerun.get("environment", {}),
        },
        "rerun_stability_warning": bool(rerun.get("stability_warning", False)),
        "rerun_stability_details": list(rerun.get("stability_details", [])),
        "materiality_threshold": materiality_threshold,
        "metrics": {
            "max_abs_ci_bound_shift": max_abs_ci_bound_shift,
            "relative_ci_l2": relative_ci_l2,
            "relative_ci_percent": relative_ci_l2 * 100.0,
            "parameter_l2_shift": parameter_l2_shift,
            "parameter_l2_relative": parameter_relative,
            "rmse_relative_change": rmse_relative,
            "cv_rmse_relative_change": cv_rmse_relative,
            "r_squared_absolute_change": r_squared_change,
            "cross_check_error_absolute_change": cross_check_change,
        },
        "scalars": {
            "rmse": {
                "baseline": float(old["rmse"]),
                "rerun": float(new["rmse"]),
                "relative_change": rmse_relative,
            },
            "cv_rmse": {
                "baseline": float(old["cv_rmse"]),
                "rerun": float(new["cv_rmse"]),
                "relative_change": cv_rmse_relative,
            },
            "r_squared": {
                "baseline": float(old["r_squared"]),
                "rerun": float(new["r_squared"]),
                "absolute_change": r_squared_change,
            },
            "cross_check_error": {
                "baseline": float(old["cross_check_error"]),
                "rerun": float(new["cross_check_error"]),
                "absolute_change": cross_check_change,
            },
            "trust": {
                "baseline": old["trust"],
                "rerun": new["trust"],
                "changed": trust_changed,
            },
        },
        "per_parameter": per_parameter(),
        "per_ci_bound": per_ci_bound(),
        "verdict": {
            "material_ci_drift": material_ci_drift,
            "material_fit_drift": material_fit_drift,
            "trust_changed": trust_changed,
            "material_drift": bool(material_ci_drift or material_fit_drift or trust_changed),
        },
    }


def _side_by_side(report: dict[str, Any]) -> list[str]:
    metrics = cast("dict[str, float]", report["metrics"])
    scalars = cast("dict[str, Any]", report["scalars"])
    lines = [
        "== Side-by-side fitted result ==",
        f"{'quantity':<22}{'baseline':>24}{'rerun':>24}{'deviation':>16}",
    ]
    for row in cast("list[dict[str, Any]]", report["per_parameter"]):
        lines.append(
            f"param[{row['index']}]{'':<14}"
            f"{row['baseline']:>24.17g}{row['rerun']:>24.17g}"
            f"{row['relative_deviation']:>15.3%}"
        )
    for key, label in (
        ("rmse", "rmse"),
        ("cv_rmse", "cv_rmse"),
        ("r_squared", "r_squared"),
        ("cross_check_error", "cross_check_error"),
    ):
        entry = cast("dict[str, Any]", scalars[key])
        change = entry.get("relative_change", entry.get("absolute_change"))
        lines.append(
            f"{label:<22}{entry['baseline']:>24.17g}{entry['rerun']:>24.17g}{change:>15.3%}"
        )
    lines.append("")
    lines.append("== Deviation metrics ==")
    for key in (
        "max_abs_ci_bound_shift",
        "relative_ci_l2",
        "parameter_l2_shift",
        "parameter_l2_relative",
        "rmse_relative_change",
        "cv_rmse_relative_change",
        "r_squared_absolute_change",
        "cross_check_error_absolute_change",
    ):
        lines.append(f"{key:<38}{metrics[key]:.12g}")
    return lines


def print_report(report: dict[str, Any]) -> None:
    baseline_env = cast("dict[str, str]", report["environments"]["baseline"])
    rerun_env = cast("dict[str, str]", report["environments"]["rerun"])
    print("== Environments ==")
    print(f"{'package':<14}{'baseline':>16}{'rerun':>16}")
    for package in sorted(set(baseline_env) | set(rerun_env)):
        print(f"{package:<14}{baseline_env.get(package, '-'):>16}{rerun_env.get(package, '-'):>16}")
    print()
    print(*_side_by_side(report), sep="\n")
    print()
    print("== Environment drift reported by the rerun ==")
    if report["rerun_stability_warning"]:
        for detail in cast("list[str]", report["rerun_stability_details"]):
            print(f"  ! {detail}")
    else:
        print("  (none)")
    print()
    threshold = cast("float", report["materiality_threshold"])
    verdict = cast("dict[str, bool]", report["verdict"])
    print("== Verdict ==")
    print(f"materiality threshold      : {threshold:.0%} relative change")
    print(f"material CI drift          : {verdict['material_ci_drift']}")
    print(f"material fit drift         : {verdict['material_fit_drift']}")
    print(f"reliability label changed  : {verdict['trust_changed']}")
    if verdict["material_drift"]:
        print()
        print(
            f"WARNING: a measured change exceeds the {threshold:.0%} materiality "
            "threshold. The fitted result is NOT numerically stable across this "
            "environment change; treat the saved analysis as invalidated."
        )
    else:
        print()
        print(
            f"OK: environment drift was detected, but every measured change stayed "
            f"below the {threshold:.0%} materiality threshold and the reliability "
            "label is unchanged. The scientific result is reproducible."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotated", type=Path, default=None)
    parser.add_argument(
        "--materiality-threshold",
        type=float,
        default=DEFAULT_MATERIALITY_THRESHOLD,
        help="fractional change above which drift is reported as material (default 0.05)",
    )
    args = parser.parse_args(argv)

    baseline = cast("dict[str, Any]", json.loads(args.baseline.read_text(encoding="utf-8")))
    rerun = cast("dict[str, Any]", json.loads(args.rerun.read_text(encoding="utf-8")))
    report = compare(baseline, rerun, materiality_threshold=args.materiality_threshold)
    print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\ncomparison written to {args.output}")

    if args.annotated is not None:
        args.annotated.parent.mkdir(parents=True, exist_ok=True)
        args.annotated.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"provenance copy written to {args.annotated}")

    return 1 if cast("dict[str, bool]", report["verdict"])["material_drift"] else 0


if __name__ == "__main__":
    sys.exit(main())
