"""Quantitative finance basics: returns, drawdowns, risk metrics and Black-Scholes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

__all__ = [
    "BSResult",
    "DrawdownResult",
    "annualized_volatility",
    "black_scholes",
    "historical_var",
    "log_returns",
    "max_drawdown",
    "monte_carlo_var",
    "sharpe_ratio",
    "simple_returns",
    "sortino_ratio",
]

FloatArray = NDArray[np.float64]

ArrayLike = Sequence[float] | FloatArray


def _as_prices(values: ArrayLike) -> FloatArray:
    """Validate and coerce a price series to a positive finite 1-D float array."""
    prices: FloatArray = np.asarray(values, dtype=np.float64)
    if (
        prices.ndim != 1
        or prices.size < 2
        or not bool(np.all(np.isfinite(prices)))
        or not bool(np.all(prices > 0.0))
    ):
        msg = "prices must be positive finite numbers"
        raise ValueError(msg)
    return prices


def log_returns(prices: ArrayLike) -> FloatArray:
    """Logarithmic returns ``ln(P_t / P_{t-1})``; length ``n - 1``."""
    values = _as_prices(prices)
    return np.diff(np.log(values))


def simple_returns(prices: ArrayLike) -> FloatArray:
    """Simple percentage returns ``P_t / P_{t-1} - 1``; length ``n - 1``."""
    values = _as_prices(prices)
    return values[1:] / values[:-1] - 1.0


@dataclass(frozen=True)
class DrawdownResult:
    """Deepest peak-to-trough decline of a price series."""

    max_drawdown: float
    peak_index: int
    trough_index: int


def max_drawdown(prices: ArrayLike) -> DrawdownResult:
    """Most severe drawdown ``P_t / running_max_t - 1`` with its peak/trough indices."""
    values = _as_prices(prices)
    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1.0
    trough = int(np.argmin(drawdowns))
    peak = int(np.argmax(values[: trough + 1]))
    return DrawdownResult(
        max_drawdown=float(drawdowns[trough]),
        peak_index=peak,
        trough_index=trough,
    )


def _validate_periods(periods_per_year: int) -> None:
    if periods_per_year < 1:
        msg = "periods_per_year must be at least 1"
        raise ValueError(msg)


def annualized_volatility(returns: ArrayLike, periods_per_year: int = 252) -> float:
    """Sample volatility scaled by ``sqrt(periods_per_year)``."""
    _validate_periods(periods_per_year)
    values = np.asarray(returns, dtype=np.float64)
    return float(np.std(values, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: ArrayLike,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized mean excess return over per-period sample volatility."""
    _validate_periods(periods_per_year)
    values = np.asarray(returns, dtype=np.float64)
    excess = values - risk_free_annual / periods_per_year
    return float(np.mean(excess) / np.std(values, ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: ArrayLike,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
    target: float = 0.0,
) -> float:
    """Mean excess return divided by downside deviation against a per-period target.

    The downside deviation averages the squared negative deviations
    ``min(r - target, 0)`` over *all* periods; if there are none it is zero
    and the ratio is infinite.
    """
    _validate_periods(periods_per_year)
    values = np.asarray(returns, dtype=np.float64)
    deviations = np.minimum(values - target, 0.0)
    downside = float(np.sqrt(np.mean(deviations**2)))
    if downside == 0.0:
        return float("inf")
    mean_excess = float(np.mean(values)) - risk_free_annual / periods_per_year
    return mean_excess / downside


@dataclass(frozen=True)
class BSResult:
    """Black-Scholes option price and first-order Greeks."""

    price: float
    delta: float
    gamma: float
    vega: float


def black_scholes(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    option: str = "call",
) -> BSResult:
    """Black-Scholes European call/put price with delta, gamma and vega.

    Vega is quoted per unit change in volatility (not per percent).
    A zero maturity collapses to intrinsic value with step-function delta.
    """
    if spot <= 0 or strike <= 0:
        msg = "spot and strike must be positive"
        raise ValueError(msg)
    if volatility <= 0:
        msg = "volatility must be positive"
        raise ValueError(msg)
    if maturity < 0:
        msg = "maturity must be non-negative"
        raise ValueError(msg)
    if option not in {"call", "put"}:
        msg = "option must be call or put"
        raise ValueError(msg)
    if maturity == 0.0:
        if option == "call":
            intrinsic = max(spot - strike, 0.0)
            delta = 1.0 if spot > strike else 0.0
        else:
            intrinsic = max(strike - spot, 0.0)
            delta = -1.0 if spot < strike else 0.0
        return BSResult(price=intrinsic, delta=delta, gamma=0.0, vega=0.0)
    root_t = float(np.sqrt(maturity))
    d1 = (float(np.log(spot / strike)) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * root_t
    )
    d2 = d1 - volatility * root_t
    density = float(sp_stats.norm.pdf(d1))
    gamma = density / (spot * volatility * root_t)
    vega = spot * density * root_t
    discount = float(np.exp(-rate * maturity))
    if option == "call":
        price = spot * float(sp_stats.norm.cdf(d1)) - strike * discount * float(
            sp_stats.norm.cdf(d2)
        )
        delta = float(sp_stats.norm.cdf(d1))
    else:
        price = strike * discount * float(sp_stats.norm.cdf(-d2)) - spot * float(
            sp_stats.norm.cdf(-d1)
        )
        delta = float(sp_stats.norm.cdf(d1)) - 1.0
    return BSResult(price=price, delta=delta, gamma=gamma, vega=vega)


def historical_var(returns: ArrayLike, confidence: float = 0.95) -> float:
    """Historical Value-at-Risk as the positive loss at the given confidence level."""
    if not 0.0 < confidence < 1.0:
        msg = "confidence must lie strictly between 0 and 1"
        raise ValueError(msg)
    values = np.asarray(returns, dtype=np.float64)
    return float(-np.quantile(values, 1.0 - confidence))


def monte_carlo_var(
    current_value: float,
    mean_return: float,
    sigma: float,
    horizon_days: int = 10,
    confidence: float = 0.99,
    simulations: int = 50_000,
    seed: int | None = None,
) -> float:
    """Monte-Carlo Value-at-Risk of a lognormal terminal value over the horizon.

    Terminal prices follow ``V0 * exp((mu - sigma^2/2) h + sigma sqrt(h) Z)``;
    the reported number is the loss percentile of that distribution.
    """
    if current_value <= 0:
        msg = "current_value must be positive"
        raise ValueError(msg)
    if sigma <= 0:
        msg = "sigma must be positive"
        raise ValueError(msg)
    if horizon_days < 1:
        msg = "horizon_days must be at least 1"
        raise ValueError(msg)
    if not 0.0 < confidence < 1.0:
        msg = "confidence must lie strictly between 0 and 1"
        raise ValueError(msg)
    if simulations < 100:
        msg = "simulations must be at least 100"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(simulations)
    drift = (mean_return - 0.5 * sigma**2) * horizon_days
    diffusion = sigma * float(np.sqrt(horizon_days))
    terminal = current_value * np.exp(drift + diffusion * shocks)
    losses = current_value - terminal
    return float(np.quantile(losses, confidence))
