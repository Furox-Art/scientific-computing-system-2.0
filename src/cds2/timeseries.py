"""Time-series analysis built on pandas and numpy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import TestResult

__all__ = [
    "DecompositionResult",
    "moving_average",
    "exponential_smoothing",
    "difference",
    "seasonal_decompose",
    "acf",
    "pacf",
    "ljung_box",
]


@dataclass(frozen=True)
class DecompositionResult:
    """Classical decomposition of a series into trend/seasonal/residual."""

    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series


def _as_series(series: object) -> pd.Series:
    result = pd.Series(series, dtype=float)
    return result


def moving_average(series: object, window: int) -> pd.Series:
    """Centered rolling mean of a series."""
    values = _as_series(series)
    if window < 1 or window > len(values):
        msg = "window must be between 1 and the series length"
        raise ValueError(msg)
    return values.rolling(window=window, center=True).mean()


def exponential_smoothing(series: object, alpha: float = 0.3) -> pd.Series:
    """Simple exponential smoothing with smoothing factor ``alpha``."""
    if not 0.0 < alpha <= 1.0:
        msg = "alpha must be in (0, 1]"
        raise ValueError(msg)
    return _as_series(series).ewm(alpha=alpha, adjust=False).mean()


def difference(series: object, lag: int = 1) -> pd.Series:
    """Lag-differenced series with NaN entries dropped."""
    values = _as_series(series)
    if lag < 1 or lag >= len(values):
        msg = "lag must be between 1 and len(series)-1"
        raise ValueError(msg)
    return values.diff(lag).dropna()


def seasonal_decompose(series: object, period: int, model: str = "additive") -> DecompositionResult:
    """Classical decomposition using a centered moving-average trend."""
    values = _as_series(series)
    n = len(values)
    if period < 2:
        msg = "period must be at least 2"
        raise ValueError(msg)
    if n < 2 * period:
        msg = "series must have at least two full periods"
        raise ValueError(msg)
    trend = values.rolling(window=period, center=True).mean()
    positions = np.arange(n) % period
    if model == "additive":
        detrended = values - trend
        profile = np.array(
            [np.nanmean(np.asarray(detrended)[positions == p]) for p in range(period)]
        )
        seasonal_values = profile[positions]
        residual_values = np.asarray(values) - np.asarray(trend) - seasonal_values
    elif model == "multiplicative":
        ratio = values / trend
        profile = np.array([np.nanmean(np.asarray(ratio)[positions == p]) for p in range(period)])
        seasonal_values = profile[positions]
        residual_values = np.asarray(values) / (np.asarray(trend) * seasonal_values)
    else:
        msg = f"unsupported decomposition model: {model!r}"
        raise ValueError(msg)
    index = values.index
    return DecompositionResult(
        observed=values,
        trend=trend,
        seasonal=pd.Series(seasonal_values, index=index),
        residual=pd.Series(residual_values, index=index),
    )


def acf(x: object, nlags: int = 20) -> np.ndarray:
    """Sample autocorrelation function (biased estimator), lags 0..nlags."""
    values = np.asarray(x, dtype=float).ravel()
    n = values.size
    if not 0 <= nlags <= n - 1:
        msg = "nlags must be between 0 and n-1"
        raise ValueError(msg)
    centered = values - values.mean()
    denominator = float((centered**2).sum())
    if denominator == 0.0:
        return np.concatenate([[1.0], np.zeros(nlags)])
    corr = np.correlate(centered, centered, mode="full")[n - 1 :] / denominator
    return np.asarray(corr[: nlags + 1])


def pacf(x: object, nlags: int = 20) -> np.ndarray:
    """Partial autocorrelation function via the Durbin-Levinson recursion."""
    values = np.asarray(x, dtype=float).ravel()
    n = values.size
    if nlags < 1 or nlags > n - 2:
        msg = "nlags must satisfy 1 <= nlags <= n-2"
        raise ValueError(msg)
    rho = acf(values, nlags)
    partial = np.ones(nlags + 1)
    phi = np.zeros((nlags + 1, nlags + 1))
    if nlags >= 1:
        phi[1, 1] = rho[1]
        partial[1] = rho[1]
    for m in range(2, nlags + 1):
        previous = phi[m - 1, 1:m]
        numerator = rho[m] - previous @ rho[m - 1 : 0 : -1]
        denominator = 1.0 - previous @ rho[1:m]
        coefficient = float(numerator / denominator) if abs(denominator) > 1e-12 else 0.0
        phi[m, m] = coefficient
        phi[m, 1:m] = previous - coefficient * phi[m - 1, m - 1 : 0 : -1]
        partial[m] = coefficient
    return partial


def ljung_box(x: object, lags: int = 10) -> TestResult:
    """Ljung-Box portmanteau test for remaining autocorrelation."""
    values = np.asarray(x, dtype=float).ravel()
    n = values.size
    if lags < 1 or lags > n - 2:
        msg = "lags must satisfy 1 <= lags <= n-2"
        raise ValueError(msg)
    correlations = acf(values, lags)[1:]
    q_statistic = float(n * (n + 2) * np.sum(correlations**2 / (n - np.arange(1, lags + 1))))
    from scipy import stats as sps

    p_value = float(sps.chi2.sf(q_statistic, df=lags))
    return TestResult(statistic=q_statistic, p_value=p_value)
