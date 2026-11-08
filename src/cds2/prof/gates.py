"""Tolerance-based regression gates for benchmark history.

``RegressionGate`` fits a tolerance band over the runs that *precede* the
latest one and asserts that the latest ratio for each benchmark stays inside
that band. It is used both as a pytest plugin (``--regression``) and as a
standalone CLI check.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from .history import BenchHistory


@dataclasses.dataclass(frozen=True)
class GateResult:
    """Outcome of a single benchmark regression check.

    Attributes:
        name: the benchmark name.
        latest_ratio: cds2/baseline ratio of the most recent run.
        limit: upper bound of the acceptance band.
        passed: whether ``latest_ratio`` lies inside the band.
        lower_limit: lower bound of the acceptance band, or ``None`` when there
            is no prior run to fit a band on (the check then passes vacuously).
    """

    name: str
    latest_ratio: float
    limit: float
    passed: bool
    lower_limit: float | None = None
    #: The latest run's own relative measurement uncertainty, if it recorded one.
    noise: float = 0.0

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        if self.lower_limit is None:
            return f"{self.name}: ratio {self.latest_ratio:.3f}, no prior baseline [{status}]"
        return (
            f"{self.name}: ratio {self.latest_ratio:.3f} vs band "
            f"[{self.lower_limit:.3f}, {self.limit:.3f}] [{status}]"
        )


class RegressionGate:
    """Check that the latest benchmark ratios stay within tolerance bands.

    The band is fitted on the runs that precede the latest one, never on a
    window that contains it. Fitting on a window that includes the latest ratio
    is self-referential: with a single row the "band" is that row's own value,
    so every profile passes, and with several rows a gradual drift keeps
    re-centring the band on itself and is never caught.

    Args:
        tolerance: fractional tolerance around the baseline median (0.10 = 10%).
            Applied in both directions: the latest ratio may rise to
            ``baseline * (1 + tolerance)`` and must not fall below
            ``baseline / (1 + tolerance)``. A suspiciously fast result -- a
            no-op measurement rather than a real speedup -- therefore fails too.
        lookback: number of prior runs used when fitting the band.
        dispersion_multiplier: how many multiples of the latest run's *own*
            measurement uncertainty the band must also cover. The band is never
            narrower than ``tolerance`` and never wider than ``max_tolerance``.
            The widening comes from the within-run uncertainty, not from the
            spread of the history: a gradual regression does not inflate the
            uncertainty of one run, so it cannot hide behind a band that grew
            because it drifted.
        max_tolerance: upper bound for that widening, so a regression larger
            than this still fails.
        history: a ``BenchHistory`` instance; defaults to the repo's own.
    """

    def __init__(
        self,
        tolerance: float = 0.10,
        lookback: int = 10,
        dispersion_multiplier: float = 2.0,
        max_tolerance: float = 0.30,
        history: BenchHistory | None = None,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if dispersion_multiplier < 0:
            raise ValueError("dispersion_multiplier must be >= 0")
        if max_tolerance < tolerance:
            raise ValueError("max_tolerance must be >= tolerance")
        self.tolerance = tolerance
        self.lookback = lookback
        self.dispersion_multiplier = dispersion_multiplier
        self.max_tolerance = max_tolerance
        self.history = history or BenchHistory()

    def _width(self, noise: float) -> float:
        """Fractional band half-width: the tolerance floor, widened by noise.

        ``noise`` is the run's own measurement uncertainty: how far the two
        halves of its rounds disagreed. It is capped by ``max_tolerance`` so a
        broken measurement cannot buy itself an arbitrarily wide band.
        """
        if noise <= 0 or not np.isfinite(noise):
            return self.tolerance
        return min(max(self.tolerance, self.dispersion_multiplier * noise), self.max_tolerance)

    def _band(self, ratios: pd.Series, noise: float = 0.0) -> float:
        """Upper limit of the band for a baseline ratio series."""
        if ratios.empty:
            return 1.0 + self._width(noise)
        return float(ratios.median()) * (1.0 + self._width(noise))

    def _lower_band(self, ratios: pd.Series, noise: float = 0.0) -> float:
        """Lower limit of the band for a baseline ratio series."""
        if ratios.empty:
            return 1.0 / (1.0 + self._width(noise))
        return float(ratios.median()) / (1.0 + self._width(noise))

    def check_latest(
        self, df: pd.DataFrame | None = None, name: str | None = None
    ) -> list[GateResult]:
        """Check the latest run against the band fitted on the runs before it.

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
            ordered = group.sort_values(["timestamp", "ratio"], kind="stable")
            ratios = ordered["ratio"]
            latest = float(ratios.iloc[-1])
            # Ratios are comparable within one hardware class only. Comparing a
            # run against a run from a different CPU is how a healthy release
            # got eight "regressions": the same commit measured on two runner
            # models differs by 12-17% on the sub-second benchmarks.
            if "cpu" in ordered.columns:
                latest_cpu = ordered["cpu"].iloc[-1]
                if isinstance(latest_cpu, str) and latest_cpu:
                    ordered = ordered[ordered["cpu"] == latest_cpu]
                    ratios = ordered["ratio"]
            noise = 0.0
            if "noise" in ordered.columns:
                raw_noise = ordered["noise"].iloc[-1]
                if isinstance(raw_noise, (int, float)) and np.isfinite(raw_noise):
                    noise = float(raw_noise)
            # Drop the latest row *before* trimming to `lookback`, so the band
            # is always fitted on prior runs only.
            baseline = ratios.iloc[:-1].tail(self.lookback)
            if baseline.empty:
                # A first run has nothing to regress against — but a non-finite
                # ratio is a broken measurement (division by zero, a timeout, an
                # instrumentation fault) and must not pass silently: an
                # unconditional PASS here is exactly how the gate stops
                # catching anything at all. A first, finite run still passes:
                # there is genuinely nothing to compare it against.
                results.append(
                    GateResult(
                        name=str(bench),
                        latest_ratio=latest,
                        limit=self._band(baseline),
                        passed=bool(np.isfinite(latest)),
                        lower_limit=None,
                        noise=noise,
                    )
                )
                continue
            lower = self._lower_band(baseline, noise)
            limit = self._band(baseline, noise)
            results.append(
                GateResult(
                    name=str(bench),
                    latest_ratio=latest,
                    limit=limit,
                    passed=lower <= latest <= limit,
                    lower_limit=lower,
                    noise=noise,
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
