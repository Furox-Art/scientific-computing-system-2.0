"""Determinism and non-flakiness of the perf regression gate.

The gate exists to fail on genuine regressions and nothing else, so its verdict
must be a pure function of the measured runs. These tests pin:

- the same runs produce the same band regardless of the row order the caller
  used (equal timestamps are ordered by ratio, not by accident);
- one genuinely slow run fails, quantitatively reproducible noise does not;
- a run with no baseline passes when finite, and a non-finite ratio never
  passes silently;
- `timed` measures exactly one execution and preserves exceptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cds2.prof.gates import RegressionGate
from cds2.prof.profiler import timed


def _frame(ratios, name="bench", step=1.0):
    return pd.DataFrame(
        {
            "name": [name] * len(ratios),
            "timestamp": [step * (i + 1) for i in range(len(ratios))],
            "ratio": ratios,
        }
    )


def test_band_is_built_from_the_median_and_tolerance() -> None:
    gate = RegressionGate(tolerance=0.2)
    ratios = pd.Series([1.0, 2.0])
    assert gate._band(ratios) == pytest.approx(2.4)
    assert gate._lower_band(ratios) == pytest.approx(2.0 / 1.2)
    assert gate._baseline_window(ratios).tolist() == [1.0, 2.0]


def test_lookback_trims_the_baseline_window() -> None:
    gate = RegressionGate(lookback=2)
    ratios = pd.Series([1.0] * 5)
    assert gate._baseline_window(ratios).tolist() == [1.0, 1.0]


def test_band_with_invalid_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError):
        RegressionGate(tolerance=-0.1)
    with pytest.raises(ValueError):
        RegressionGate(lookback=0)


def test_band_is_independent_of_row_order_for_equal_timestamps() -> None:
    ratios = [1.00, 1.01, 0.99, 1.02, 1.00, 0.98, 1.01, 1.00, 0.99, 1.01]
    df = _frame(ratios)
    shuffled = df.iloc[np.random.RandomState(0).permutation(len(df))]

    a = RegressionGate().check_latest(df)[0]
    b = RegressionGate().check_latest(shuffled)[0]
    assert a.limit == b.limit and a.latest_ratio == b.latest_ratio
    assert a.passed == b.passed


def test_same_runs_always_give_the_same_verdict() -> None:
    df = _frame([1.0] * 9 + [1.4])
    results = [RegressionGate().check_latest(df)[0] for _ in range(5)]
    assert len({(r.limit, r.passed) for r in results}) == 1
    assert not results[0].passed


def test_gate_detects_a_genuine_regression() -> None:
    steady = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    result = RegressionGate().check_latest(_frame(steady + [1.8]))[0]
    assert result.passed is False


@pytest.mark.parametrize("factor", [1.01, 0.99, 1.05, 0.95])
def test_gate_tolerates_reproducible_noise(factor: float) -> None:
    # +/- 5% around a steady 1.00 sit inside the default 10% band.
    base = [1.0] * 9
    noisy = [r * factor for r in base]
    result = RegressionGate().check_latest(_frame(noisy))[0]
    assert result.passed is True


def test_gate_first_run_without_baseline_must_be_finite() -> None:
    passing = RegressionGate().check_latest(_frame([1.0]))[0]
    assert passing.passed is True
    failing = RegressionGate().check_latest(_frame([np.nan]))[0]
    assert failing.passed is False


def test_timed_runs_once_and_propagates_exceptions() -> None:
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise RuntimeError("boom")

    wrapped = timed(boom)
    with pytest.raises(RuntimeError):
        wrapped()
    assert calls["n"] == 1


def test_timed_measures_exactly_one_execution() -> None:
    calls = {"n": 0}

    def work():
        calls["n"] += 1
        return 41 + 1

    wrapped = timed(work)
    value = wrapped()
    assert calls["n"] == 1
    assert value == 42
    # The stored profile describes the same invocation that returned the value.
    assert wrapped.last_profile.wall_median is not None


def test_timed_is_reproducible_in_shape() -> None:
    wrapped = timed(lambda: 7)
    first = (wrapped(), wrapped.last_profile.wall_median is not None)
    again = (wrapped(), wrapped.last_profile.wall_median is not None)
    assert first == again
