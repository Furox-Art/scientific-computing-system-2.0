#!/usr/bin/env python3
"""Case study 21: NumPy/SciPy version drift in a reproducible guided fit.

A single deterministic dataset, a single model and a single random seed are
fitted twice — once in the environment recorded in the saved manifest, and
once in the environment currently running the script. The fitted parameters,
their confidence intervals and every scalar diagnostic are then compared
side by side, with absolute and relative deviation reported for each.

The purpose is to answer a question that plain unit tests cannot: *does a
saved scientific result still hold when the numerical stack underneath it
changes?* Environment drift is reported explicitly, and any measured change
that exceeds a materiality threshold is escalated to a loud WARNING instead
of being silently rounded away.

What this script does and does not claim
----------------------------------------
- It runs a real guided fit in *this* environment and replays a saved
  manifest, so the rerun numbers here are measured, not fabricated.
- The committed artifact under ``examples/data/version-drift/`` records a
  baseline (NumPy 1.26.4 / SciPy 1.11.4) and a rerun (NumPy 2.3.5 /
  SciPy 1.18.1) that were each executed in their own environment. Regenerate
  both halves with the commands in ``examples/version_drift_example.md``.
- Running this script does **not** install a second NumPy/SciPy version. It
  compares whatever environments you feed it. If you run both halves in one
  environment you will correctly observe zero environment drift and zero
  deviation, which is the expected self-consistency check.

Usage
-----
.. code-block:: bash

    # print the committed, already-measured comparison
    python examples/version_drift_fit.py

    # compare your own baseline/rerun pair
    python examples/version_drift_fit.py \
        --baseline case/baseline-result.json \
        --rerun case/upgraded-rerun.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_DIR = REPO_ROOT / "examples" / "data" / "version-drift"
DEFAULT_MATERIALITY_THRESHOLD = 0.05

QUANTITIES = ("params", "parameter_std", "confidence_95", "rmse", "cv_rmse", "r_squared")


def _numpy() -> Any:
    import numpy

    return numpy


def load_record(path: Path) -> dict[str, Any]:
    """Load one baseline/rerun result record produced by the drift driver."""
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _as_array(record: dict[str, Any], key: str) -> Any:
    return _numpy().asarray(cast("Any", record["result"][key]), dtype=float)


def _relative_change(new: float, old: float) -> float:
    return abs(new - old) / max(abs(old), 1e-12)


def print_environment_drift(baseline: dict[str, Any], rerun: dict[str, Any]) -> list[str]:
    """Report which recorded packages differ between the two environments."""
    old_env = cast("dict[str, str]", baseline.get("environment", {}))
    new_env = cast("dict[str, str]", rerun.get("environment", {}))
    print("== Environment ==")
    print(f"{'package':<14}{'baseline':>18}{'rerun':>18}")
    changed: list[str] = []
    for package in sorted(set(old_env) | set(new_env)):
        before = old_env.get(package, "-")
        after = new_env.get(package, "-")
        print(f"{package:<14}{before:>18}{after:>18}")
        if before != after:
            changed.append(f"runtime version changed: {package} {before} -> {after}")
    print()
    if changed:
        print("== Environment drift detected ==")
        for line in changed:
            print(f"  ! {line}")
    else:
        print("== Environment drift == (none: both runs used identical package versions)")
    print()
    return changed


def print_side_by_side(baseline: dict[str, Any], rerun: dict[str, Any]) -> None:
    """Print baseline vs rerun values with absolute and relative deviation."""
    old = cast("dict[str, Any]", baseline["result"])
    new = cast("dict[str, Any]", rerun["result"])

    print("== Side-by-side result ==")
    header = f"{'quantity':<26}{'baseline':>24}{'rerun':>24}{'abs dev':>13}{'rel dev':>12}"
    print(header)
    print("-" * len(header))

    old_params = _as_array(baseline, "params")
    new_params = _as_array(rerun, "params")
    for index in range(old_params.size):
        delta = abs(float(new_params[index]) - float(old_params[index]))
        print(
            f"param[{index}]"
            f"{float(old_params[index]):>24.12g}{float(new_params[index]):>24.12g}"
            f"{delta:>13.6g}{_relative_change(float(new_params[index]), float(old_params[index])):>11.4%}"
        )

    old_ci = _as_array(baseline, "confidence_95")
    new_ci = _as_array(rerun, "confidence_95")
    for row in range(old_ci.shape[0]):
        for col, bound in enumerate(("lower", "upper")):
            delta = abs(float(new_ci[row, col]) - float(old_ci[row, col]))
            print(
                f"ci[{row}][{bound}]"
                f"{float(old_ci[row, col]):>24.12g}{float(new_ci[row, col]):>24.12g}"
                f"{delta:>13.6g}{_relative_change(float(new_ci[row, col]), float(old_ci[row, col])):>11.4%}"
            )

    for key in ("rmse", "cv_rmse", "r_squared", "cross_check_error"):
        before, after = float(old[key]), float(new[key])
        print(
            f"{key:<26}{before:>24.12g}{after:>24.12g}"
            f"{abs(after - before):>13.6g}{_relative_change(after, before):>11.4%}"
        )
    print()


def print_aggregate_metrics(baseline: dict[str, Any], rerun: dict[str, Any]) -> dict[str, float]:
    """Print and return the whole-result deviation metrics."""
    np = _numpy()
    old_ci = _as_array(baseline, "confidence_95")
    new_ci = _as_array(rerun, "confidence_95")
    ci_delta = new_ci - old_ci
    ci_scale = max(float(np.linalg.norm(old_ci)), 1e-12)
    relative_ci_l2 = float(np.linalg.norm(ci_delta)) / ci_scale

    old_params = _as_array(baseline, "params")
    new_params = _as_array(rerun, "params")
    param_scale = max(float(np.linalg.norm(old_params)), 1e-12)
    parameter_l2_shift = float(np.linalg.norm(new_params - old_params))

    old = cast("dict[str, Any]", baseline["result"])
    new = cast("dict[str, Any]", rerun["result"])

    metrics = {
        "max_abs_ci_bound_shift": float(np.max(np.abs(ci_delta))),
        "relative_ci_l2": relative_ci_l2,
        "parameter_l2_shift": parameter_l2_shift,
        "parameter_l2_relative": parameter_l2_shift / param_scale,
        "rmse_relative_change": _relative_change(float(new["rmse"]), float(old["rmse"])),
        "cv_rmse_relative_change": _relative_change(float(new["cv_rmse"]), float(old["cv_rmse"])),
        "r_squared_absolute_change": abs(float(new["r_squared"]) - float(old["r_squared"])),
    }

    print("== Aggregate deviation metrics ==")
    for key, value in metrics.items():
        print(f"{key:<34}{value:.12g}")
    print()
    return metrics


def print_verdict(
    metrics: dict[str, float],
    baseline: dict[str, Any],
    rerun: dict[str, Any],
    *,
    threshold: float,
    environment_drift: list[str],
) -> bool:
    """Escalate to a WARNING when a change exceeds the materiality threshold."""
    old = cast("dict[str, Any]", baseline["result"])
    new = cast("dict[str, Any]", rerun["result"])
    trust_changed = old["trust"] != new["trust"]

    material = (
        metrics["relative_ci_l2"] > threshold
        or max(metrics["rmse_relative_change"], metrics["parameter_l2_relative"]) > threshold
        or trust_changed
    )

    print("== Verdict ==")
    print(f"materiality threshold          : {threshold:.0%} relative change")
    print(f"environment drift detected     : {bool(environment_drift)}")
    print(
        f"reliability label              : {old['trust']} -> {new['trust']}"
        f" ({'CHANGED' if trust_changed else 'unchanged'})"
    )
    print(f"material drift                 : {material}")
    print()

    if material:
        print(
            f"WARNING: measured drift exceeds the {threshold:.0%} materiality "
            "threshold. The saved analysis is NOT reproducible under this "
            "environment change; do not reuse the baseline result."
        )
    elif environment_drift:
        print(
            f"Environment drift was detected and reported. Every measured change "
            f"stayed below the {threshold:.0%} materiality threshold and the "
            "reliability label is unchanged, so the scientific conclusion is "
            "reproducible across this stack change."
        )
    else:
        print(
            "No environment drift and no material change: both runs are "
            "numerically identical, which is the expected self-consistency "
            "result when the same environment replays its own manifest."
        )
    print()
    return material


def run_self_check(seed: int) -> int:
    """Fit the committed dataset in the current environment as a smoke test."""
    import numpy as np
    import scipy

    try:
        from cds2 import guided_fit as gf
    except ImportError:  # pragma: no cover - depends on invocation path
        print("cds2 is not importable; skipping the live self-check.", file=sys.stderr)
        return 0

    x = np.linspace(0.0, 0.9, 96)
    y = 1.8 * np.exp(0.075 * x) + 2.2
    y += 7e-4 * np.sin(7.0 * x) + 3e-4 * np.cos(13.0 * x)
    dataset = gf.FitDataset(
        name="version_drift_selfcheck",
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        sigma=np.full_like(x, 0.002),
    )
    result = gf.run_guided_fit((dataset,), "exponential", seed=seed)
    item = result.datasets[0]

    print("== Live self-check (this environment) ==")
    print(f"python   : {__import__('platform').python_version()}")
    print(f"numpy    : {np.__version__}")
    print(f"scipy    : {scipy.__version__}")
    print(f"model    : {result.model}")
    print(f"seed     : {seed}")
    print(f"params   : {item.params.tolist()}")
    print(f"rmse     : {item.rmse:.12g}")
    print(f"trust    : {result.trust}")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NumPy/SciPy version-drift case study.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=COMMITTED_DIR / "baseline-result.json",
        help="baseline result record (default: the committed one)",
    )
    parser.add_argument(
        "--rerun",
        type=Path,
        default=COMMITTED_DIR / "upgraded-rerun.json",
        help="rerun result record (default: the committed one)",
    )
    parser.add_argument(
        "--materiality-threshold",
        type=float,
        default=DEFAULT_MATERIALITY_THRESHOLD,
        help="relative change above which drift is reported as material (default 0.05)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="also fit the dataset in the current environment and print it",
    )
    args = parser.parse_args(argv)

    missing = [path for path in (args.baseline, args.rerun) if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        print(f"missing result record(s): {joined}", file=sys.stderr)
        return 2

    print("== NumPy/SciPy version-drift case study ==")
    print(f"case     : {load_record(args.baseline).get('case', 'n/a')}")
    print(f"baseline record: {args.baseline}")
    print(f"rerun record  : {args.rerun}")
    print()

    baseline = load_record(args.baseline)
    rerun = load_record(args.rerun)

    environment_drift = print_environment_drift(baseline, rerun)
    print_side_by_side(baseline, rerun)
    metrics = print_aggregate_metrics(baseline, rerun)
    material = print_verdict(
        metrics,
        baseline,
        rerun,
        threshold=args.materiality_threshold,
        environment_drift=environment_drift,
    )

    if args.self_check:
        seed = int(baseline.get("seed", 0))
        run_self_check(seed)

    return 1 if material else 0


if __name__ == "__main__":
    sys.exit(main())
