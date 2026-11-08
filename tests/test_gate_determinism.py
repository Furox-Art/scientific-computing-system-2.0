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


def test_band_follows_the_baseline_median() -> None:
    """The band is centred on the median of the runs before the latest one."""
    # Five runs at 1.0 then one at 2.0: the baseline window is the five runs
    # before the last, so the fitted median is 1.0 and the band is 0.9-1.1.
    result = RegressionGate().check_latest(_frame([1.0] * 5 + [2.0]))[0]
    assert result.passed is False
    assert result.lower_limit == pytest.approx(1.0 / 1.1)
    assert result.limit == pytest.approx(1.0 * 1.1)


def test_lookback_limits_the_baseline_window() -> None:
    # With lookback=1 the band is fitted on the single run before the latest,
    # so a spike starting from a drifted baseline is judged against that run.
    ratios = [1.0, 1.0, 1.0, 1.0, 5.0]
    gate = RegressionGate(lookback=1)
    result = gate.check_latest(_frame(ratios))[0]
    assert result.passed is False
    # baseline median = 1.0 → band [0.9, 1.1]; latest 5.0 must fail.
    assert result.limit == pytest.approx(1.1)


def test_invalid_gate_configuration_is_rejected() -> None:
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


def _frame_with_noise(ratios, noise):
    """A history frame that carries the per-run measurement uncertainty."""
    return pd.DataFrame(
        {
            "name": ["bench"] * len(ratios),
            "timestamp": [float(i + 1) for i in range(len(ratios))],
            "ratio": ratios,
            "noise": noise,
        }
    )


def test_a_noisy_run_widens_the_band_by_its_own_uncertainty() -> None:
    """A run that cannot reproduce itself is not failed for a small difference.

    Measured on the shared CI runners, benchmarks disagree with themselves by
    10-20% between the two halves of one set of rounds. The gate widens by that
    much instead of using one fixed tolerance for every benchmark and host.
    """
    tight = RegressionGate().check_latest(_frame_with_noise([1.0, 1.0, 1.15], [0.0, 0.0, 0.0]))[0]
    noisy = RegressionGate().check_latest(_frame_with_noise([1.0, 1.0, 1.15], [0.0, 0.0, 0.10]))[0]
    assert tight.passed is False
    assert noisy.passed is True
    # width = max(tolerance 0.10, 2 * noise 0.10) = 0.20 around a median of 1.0
    assert noisy.limit == pytest.approx(1.20)
    assert noisy.lower_limit == pytest.approx(1.0 / 1.20)
    assert noisy.noise == pytest.approx(0.10)


def test_the_widening_is_capped_so_a_broken_measurement_cannot_hide_a_regression() -> None:
    result = RegressionGate().check_latest(_frame_with_noise([1.0, 1.0, 5.0], [0.0, 0.0, 9.0]))[0]
    assert result.passed is False
    # width capped at max_tolerance 0.30 around a median of 1.0
    assert result.limit == pytest.approx(1.30)


def test_history_without_noise_keeps_the_plain_tolerance() -> None:
    frame = _frame([1.0, 1.0, 1.15])
    result = RegressionGate().check_latest(frame)[0]
    assert result.passed is False
    assert result.noise == 0.0
    assert result.limit == pytest.approx(1.10)


def test_invalid_widening_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        RegressionGate(dispersion_multiplier=-1.0)
    with pytest.raises(ValueError):
        RegressionGate(tolerance=0.2, max_tolerance=0.1)


def test_a_non_numeric_noise_value_is_ignored() -> None:
    """Old or damaged rows must not widen or crash the gate."""
    frame = _frame_with_noise([1.0, 1.0, 1.15], [0.0, 0.0, float("nan")])
    result = RegressionGate().check_latest(frame)[0]
    assert result.noise == 0.0
    assert result.passed is False
    assert result.limit == pytest.approx(1.10)
