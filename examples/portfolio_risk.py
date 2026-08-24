"""Case study 18: Portfolio risk metrics.

A two-asset synthetic price history is converted into return statistics -
volatility, Sharpe and Sortino ratios, maximum drawdown - and tail risk is
quantified with historical and Monte Carlo VaR plus a Black-Scholes hedge
cost.
"""

from __future__ import annotations

import numpy as np

import cds2


def main() -> None:
    rng = np.random.default_rng(11)
    days = 750
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.018, days)))

    returns = cds2.finance.log_returns(prices.tolist())
    vol = cds2.finance.annualized_volatility(returns)
    sharpe = cds2.finance.sharpe_ratio(returns, risk_free_annual=0.03)
    sortino = cds2.finance.sortino_ratio(returns, risk_free_annual=0.03)
    drawdown = cds2.finance.max_drawdown(prices.tolist())

    print("== Portfolio statistics (3y daily) ==")
    print(f"annualized volatility : {vol:.2%}")
    print(f"Sharpe ratio          : {sharpe:.2f}")
    print(f"Sortino ratio         : {sortino:.2f}")
    print(
        f"max drawdown          : {drawdown.max_drawdown:.1%} "
        f"(day {drawdown.peak_index} -> {drawdown.trough_index})"
    )

    var95 = cds2.finance.historical_var(returns, confidence=0.95)
    mc_var99 = cds2.finance.monte_carlo_var(
        current_value=250_000.0,
        mean_return=float(np.mean(returns)),
        sigma=float(np.std(returns, ddof=1)),
        horizon_days=10,
        confidence=0.99,
        simulations=100_000,
        seed=5,
    )
    print("\n== Value at Risk ==")
    print(f"1-day historical VaR 95%   : {var95:.2%}")
    print(f"10-day MC VaR 99% (250k)   : {mc_var99:,.0f}")

    bs = cds2.finance.black_scholes(
        spot=prices[-1],
        strike=prices[-1] * 0.95,
        rate=0.04,
        volatility=vol,
        maturity=0.5,
        option="put",
    )
    print("\n== Protective put (5% OTM, 6m) ==")
    print(f"price {bs.price:.2f}  delta {bs.delta:.3f}  gamma {bs.gamma:.5f}  vega {bs.vega:.1f}")


if __name__ == "__main__":
    main()
