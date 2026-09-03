"""Regression report generator for benchmark history.

``run_regression_check`` loads the benchmark history, computes tolerance bands,
and emits a pass/fail report. It is used by ``benchmarks/run_benchmarks.py`` and
can be invoked standalone after a benchmark run.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cds2.prof import BenchHistory, RegressionGate  # noqa: E402


@dataclasses.dataclass(frozen=True)
class RegressionReport:
    """Summary of a regression check run."""

    results: list[GateLine]
    passed: bool

    def to_table(self) -> str:
        lines = ["name                      latest    limit   status"]
        lines += ["-" * 52]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"{r.name:26s} {r.latest:7.3f} {r.limit:7.3f}   {status}")
        lines += ["-" * 52]
        verdict = "ALL PASS" if self.passed else "FAILURES DETECTED"
        lines.append(verdict)
        return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class GateLine:
    name: str
    latest: float
    limit: float
    passed: bool


def run_regression_check(
    tolerance: float = 0.10,
    lookback: int = 10,
    history: BenchHistory | None = None,
) -> RegressionReport:
    """Load history, run the regression gate, return a report."""
    gate = RegressionGate(tolerance=tolerance, lookback=lookback, history=history)
    raw = gate.check_latest()
    lines = [
        GateLine(name=r.name, latest=r.latest_ratio, limit=r.limit, passed=r.passed) for r in raw
    ]
    return RegressionReport(results=lines, passed=all(r.passed for r in lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark regression check")
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--lookback", type=int, default=10)
    args = parser.parse_args()

    report = run_regression_check(tolerance=args.tolerance, lookback=args.lookback)
    print(report.to_table())
    sys.exit(0 if report.passed else 1)
