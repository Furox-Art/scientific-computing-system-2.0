"""Run CDS2 benchmarks with reproducibility and machine provenance metadata.

This is the research-facing wrapper around :mod:`benchmarks.run_benchmarks`.
It preserves the existing timing suite and augments ``results.json`` with a
machine-readable protocol plus a separate ``provenance.json`` snapshot.

Run from the repository root::

    python -m benchmarks.research_run --quick
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.run_benchmarks import (
    CORE_BENCHMARKS,
    OPTIONAL_BENCHMARKS,
    BenchResult,
    environment_info,
    format_table,
    run_all,
)

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _git_output(*args: str) -> str | None:
    """Return stripped git output, or ``None`` when git is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _numpy_configuration() -> str:
    """Capture NumPy's BLAS/LAPACK and compiler configuration as text."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_config()
    return buffer.getvalue().strip()


def collect_provenance(command: list[str] | None = None) -> dict[str, Any]:
    """Collect environment details required to interpret performance results."""
    git_status = _git_output("status", "--porcelain")
    base_environment = environment_info()
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command or []),
        "git": {
            "commit": _git_output("rev-parse", "HEAD")
            or base_environment.get("git_commit", "unknown"),
            "dirty": None if git_status is None else bool(git_status),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "byteorder": sys.byteorder,
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "threading_environment": {
            name: os.environ[name] for name in THREAD_ENV_VARS if name in os.environ
        },
        "package_versions": base_environment.get("versions", {}),
        "numpy_configuration": _numpy_configuration(),
    }


def run_research_benchmarks(
    output_dir: str | Path,
    *,
    quick: bool = False,
    benchmarks: list[str] | None = None,
    include_optional: bool = True,
    command: list[str] | None = None,
) -> list[BenchResult]:
    """Run the existing suite and persist protocol/provenance beside results."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    # Capture provenance before the benchmark writes files so the git-dirty
    # flag describes the source checkout rather than generated artifacts.
    provenance = collect_provenance(command=command)
    results = run_all(
        output_dir=target,
        quick=quick,
        benchmarks=benchmarks,
        include_optional=include_optional,
    )

    results_path = target / "results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["protocol"] = {
        "quick_mode": quick,
        "selected_benchmarks": "all" if benchmarks is None else list(benchmarks),
        "optional_benchmarks_enabled": include_optional,
        "timer": "time.perf_counter",
        "timing_aggregation": "minimum of benchmark-specific repeated wall-clock timings",
        "suite_warmup": "none; benchmark-specific setup/warmup is defined in run_benchmarks.py",
        "randomness": "fixed per-benchmark seeds where the benchmark is stochastic",
    }
    payload["provenance"] = provenance
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for provenance-rich benchmark runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use reduced benchmark sizes")
    parser.add_argument(
        "--output-dir",
        default="benchmarks/artifacts/latest",
        help="directory for results.json and provenance.json",
    )
    parser.add_argument("--core-only", action="store_true", help="skip optional third-party races")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(CORE_BENCHMARKS) + sorted(OPTIONAL_BENCHMARKS),
        help="run only selected benchmark cases",
    )
    args = parser.parse_args(argv)
    command = list(sys.argv[1:] if argv is None else argv)

    results = run_research_benchmarks(
        args.output_dir,
        quick=args.quick,
        benchmarks=args.only,
        include_optional=not args.core_only,
        command=command,
    )
    print(format_table(results))
    print(f"\nArtifacts: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
