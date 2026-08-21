"""Tests for cds2.timeseries."""

import numpy as np
import pandas as pd
import pytest

from cds2 import timeseries


def _seasonal_series(periods: int = 6, period: int = 12) -> pd.Series:
    t_values = np.arange(periods * period, dtype=float)
    trend = 0.5 * t_values
    seasonal = 10.0 * np.sin(2 * np.pi * t_values / period)
    return pd.Series(trend + seasonal, name="value")


class TestSmoothing:
    def test_moving_average_length_and_nan_edges(self) -> None:
        smoothed = timeseries.moving_average(list(range(20)), window=3)
        assert len(smoothed) == 20
        assert np.isnan(smoothed.iloc[0])
        assert smoothed.iloc[1] == pytest.approx(1.0)

    def test_exponential_smoothing_recurses(self) -> None:
        smoothed = timeseries.exponential_smoothing([0.0, 10.0, 10.0], alpha=0.5)
        assert smoothed.iloc[0] == pytest.approx(0.0)
        assert smoothed.iloc[1] == pytest.approx(5.0)

    def test_alpha_validation(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            timeseries.exponential_smoothing([1.0, 2.0], alpha=1.5)

    def test_difference(self) -> None:
        differenced = timeseries.difference([1.0, 4.0, 9.0, 16.0], lag=1)
        assert list(differenced) == [3.0, 5.0, 7.0]

    def test_window_validation(self) -> None:
        with pytest.raises(ValueError, match="window"):
            timeseries.moving_average([1.0, 2.0], window=99)


class TestSeasonalDecompose:
    def test_additive_recovers_seasonality(self) -> None:
        series = _seasonal_series()
        result = timeseries.seasonal_decompose(series, period=12, model="additive")
        assert len(result.seasonal) == len(series)
        amplitude = result.seasonal.max() - result.seasonal.min()
        assert amplitude == pytest.approx(20.0, abs=1.5)

    def test_residual_small_for_perfect_signal(self) -> None:
        series = _seasonal_series()
        result = timeseries.seasonal_decompose(series, period=12)
        interior = result.residual.dropna().abs().max()
        assert interior < 1e-9 or interior < 0.5

    def test_multiplicative_runs(self) -> None:
        t_values = np.arange(48, dtype=float)
        series = pd.Series((1.0 + 0.05 * t_values) * (1.0 + 0.2 * np.sin(2 * np.pi * t_values / 8)))
        result = timeseries.seasonal_decompose(series, period=8, model="multiplicative")
        assert result.seasonal.notna().sum() > 0

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="periods"):
            timeseries.seasonal_decompose(pd.Series(np.ones(10)), period=12)


class TestACF:
    def test_lag_zero_is_one(self) -> None:
        correlations = timeseries.acf(np.random.default_rng(1).normal(size=100), nlags=10)
        assert correlations[0] == pytest.approx(1.0)

    def test_ar1_decay_geometric(self) -> None:
        rng_values = np.random.default_rng(2)
        n = 5000
        shocks = rng_values.normal(size=n)
        ar = np.zeros(n)
        for i in range(1, n):
            ar[i] = 0.8 * ar[i - 1] + shocks[i]
        correlations = timeseries.acf(ar, nlags=3)
        assert correlations[1] == pytest.approx(0.8, abs=0.05)
        assert correlations[2] < correlations[1]

    def test_constant_series_guard(self) -> None:
        correlations = timeseries.acf(np.full(50, 3.0), nlags=5)
        assert correlations[0] == 1.0
        assert np.allclose(correlations[1:], 0.0)


class TestPACF:
    def test_ar1_cutoff_after_first_lag(self) -> None:
        rng_values = np.random.default_rng(3)
        n = 3000
        shocks = rng_values.normal(size=n)
        ar = np.zeros(n)
        for i in range(1, n):
            ar[i] = 0.7 * ar[i - 1] + shocks[i]
        partial = timeseries.pacf(ar, nlags=4)
        assert partial[1] == pytest.approx(0.7, abs=0.04)
        assert abs(partial[2]) < 0.08
        assert abs(partial[3]) < 0.08


class TestLjungBox:
    def test_white_noise_not_rejected(self) -> None:
        noise = np.random.default_rng(4).uniform(-1.0, 1.0, size=400)
        assert timeseries.ljung_box(noise, lags=10).p_value > 0.01

    def test_trending_series_rejected(self) -> None:
        trending = np.arange(300, dtype=float) + 5.0 * np.sin(np.arange(300) / 3.0)
        assert timeseries.ljung_box(trending, lags=10).p_value < 1e-6
