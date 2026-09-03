"""Tolerance-based regression gates for benchmark history.

``RegressionGate`` fits a tolerance band over recent history and asserts that
the latest ratio for each benchmark stays within band. It is used both as a
pytest plugin (``--regression``) and as a standalone CLI check.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from .history import BenchHistory


@dataclasses.dataclass(frozen=True)
class GateResult:
    """Outcome of a single benchmark regression check."""

    name: str
    latest_ratio: float
    limit: float
    passed: bool

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{self.name}: ratio {self.latest_ratio:.3f} vs limit {self.limit:.3f} [{status}]"


class RegressionGate:
    """Check that the latest benchmark ratios stay within tolerance bands.

    Args:
        tolerance: fractional tolerance above 1.0 (0.10 = 10%). Applied to
            the historical median ratio for each benchmark.
        lookback: number of recent runs to include when fitting the band.
        history: a ``BenchHistory`` instance; defaults to the repo's own.
    """

    def __init__(
        self,
        tolerance: float = 0.10,
        lookback: int = 10,
        history: BenchHistory | None = None,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.tolerance = tolerance
        self.lookback = lookback
        self.history = history or BenchHistory()

    def _band(self, ratios: pd.Series) -> float:
        """Compute the upper limit for a benchmark's ratio series.

        The limit is ``median * (1 + tolerance)`` but never below 1.0 (we never
        require cds2 to be faster than the baseline).
        """
        if ratios.empty:
            return 1.0 + self.tolerance
        return max(float(ratios.median()) * (1.0 + self.tolerance), 1.0)

    def check_latest(
        self, df: pd.DataFrame | None = None, name: str | None = None
    ) -> list[GateResult]:
        """Check the latest run against the historical band.

        Args:
            df: pre-loaded history frame. If None, loads the last *lookback*
                runs from the history directory.
            name: if given, only check this benchmark; otherwise check all.
        """
        if df is None:
            df = self.history.load_range()
        if df.empty:
            return []
        if "name" not in df.columns:
            return []

        results: list[GateResult] = []
        for bench, group in df.groupby("name"):
            if name and bench != name:
                continue
            recent = group.sort_values("timestamp").tail(self.lookback)
            if recent.empty:  # pragma: no cover - defensive; groupby groups are never empty
                continue
            latest = float(recent["ratio"].iloc[-1])
            limit = self._band(recent["ratio"])
            results.append(
                GateResult(
                    name=str(bench),
                    latest_ratio=latest,
                    limit=limit,
                    passed=latest <= limit,
                )
            )
        return results

    def assert_latest(self, df: pd.DataFrame | None = None) -> None:
        """Like ``check_latest`` but raises ``AssertionError`` on any failure."""
        results = self.check_latest(df)
        failures = [r for r in results if not r.passed]
        if failures:
            msg = "regression gate failures:\n" + "\n".join(f"  {f}" for f in failures)
            raise AssertionError(msg)


# pytest plugin --------------------------------------------------------------


def pytest_addoption(parser: Any) -> None:
    """Register the ``--regression`` CLI flag."""
    parser.addoption(
        "--regression",
        action="store_true",
        default=False,
        help="Run benchmark regression gates after tests",
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Skip regression tests unless ``--regression`` is passed."""
    if config.getoption("--regression"):
        return
    skip = pytest.mark.skip(reason="need --regression option to run")
    for item in items:
        if "regression" in item.keywords:
            item.add_marker(skip)


try:
    import pytest
except ImportError:  # pragma: no cover - pytest is optional at runtime
    pass
else:

    @pytest.mark.regression
    def test_regression_gate() -> None:  # pragma: no cover - runs only under --regression
        """Run the regression gate as a pytest test."""
        gate = RegressionGate()
        gate.assert_latest()
