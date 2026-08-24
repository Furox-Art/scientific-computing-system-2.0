"""Tests for cds2.finance."""

import math

import numpy as np
import pytest
from scipy import stats as sp_stats

from cds2 import finance as fin


class TestReturns:
    def test_log_returns_hand_checked(self) -> None:
        result = fin.log_returns([100.0, 110.0, 121.0])
        expected = np.array([math.log(1.1), math.log(1.1)])
        assert result.shape == (2,)
        assert result == pytest.approx(expected)

    def test_simple_returns_hand_checked(self) -> None:
        result = fin.simple_returns([100.0, 110.0, 121.0])
        assert result.shape == (2,)
        assert result == pytest.approx([0.10, 0.10])

    def test_returns_accept_arrays(self) -> None:
        prices = np.array([100.0, 110.0, 121.0])
        logs = fin.log_returns(prices)
        assert fin.simple_returns(prices) == pytest.approx(np.exp(logs) - 1.0)

    def test_rejects_wrong_ndim(self) -> None:
        with pytest.raises(ValueError, match="positive finite"):
            fin.log_returns([[100.0, 110.0]])

    def test_rejects_short_series(self) -> None:
        with pytest.raises(ValueError, match="positive finite"):
            fin.simple_returns([100.0])

    def test_rejects_nonpositive_prices(self) -> None:
        with pytest.raises(ValueError, match="positive finite"):
            fin.max_drawdown([100.0, -5.0])

    def test_rejects_zero_price(self) -> None:
        with pytest.raises(ValueError, match="positive finite"):
            fin.max_drawdown([100.0, 0.0])

    def test_rejects_nonfinite_prices(self) -> None:
        for bad in (np.nan, np.inf):
            with pytest.raises(ValueError, match="positive finite"):
                fin.log_returns([100.0, float(bad)])


class TestMaxDrawdown:
    def test_crafted_path(self) -> None:
        result = fin.max_drawdown([100.0, 120.0, 90.0, 95.0, 80.0, 130.0])
        assert result.max_drawdown == pytest.approx(80.0 / 120.0 - 1.0)
        assert result.max_drawdown < 0.0
        assert result.peak_index == 1
        assert result.trough_index == 4

    def test_monotonic_rise_has_zero_drawdown(self) -> None:
        result = fin.max_drawdown([100.0, 110.0, 120.0])
        assert result.max_drawdown == pytest.approx(0.0)
        assert (result.peak_index, result.trough_index) == (0, 0)

    def test_result_is_frozen(self) -> None:
        result = fin.max_drawdown([100.0, 50.0])
        with pytest.raises(AttributeError):
            result.max_drawdown = -0.5  # type: ignore[misc]


class TestVolatilityAndRatios:
    returns = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]

    def test_annualized_volatility_matches_manual(self) -> None:
        expected = float(np.std(np.asarray(self.returns), ddof=1)) * math.sqrt(252)
        assert fin.annualized_volatility(self.returns) == pytest.approx(expected)

    def test_annualized_volatility_custom_periods(self) -> None:
        expected = float(np.std(np.asarray(self.returns), ddof=1)) * math.sqrt(12)
        got = fin.annualized_volatility(self.returns, periods_per_year=12)
        assert got == pytest.approx(expected)

    def test_periods_per_year_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            fin.annualized_volatility(self.returns, periods_per_year=0)

    def test_sharpe_ratio_matches_manual(self) -> None:
        rng = np.random.default_rng(7)
        sample = rng.normal(loc=0.0005, scale=0.01, size=500)
        expected = (
            (float(np.mean(sample)) - 0.02 / 252) / float(np.std(sample, ddof=1)) * math.sqrt(252)
        )
        got = fin.sharpe_ratio(sample, risk_free_annual=0.02)
        assert got == pytest.approx(expected)

    def test_sharpe_zero_rate_default(self) -> None:
        sample = [0.01, -0.01, 0.02]
        expected = (float(np.mean(sample)) / float(np.std(sample, ddof=1))) * math.sqrt(252)
        assert fin.sharpe_ratio(sample) == pytest.approx(expected)

    def test_sharpe_periods_validation(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            fin.sharpe_ratio(self.returns, periods_per_year=-3)

    def test_sortino_ratio_matches_manual(self) -> None:
        rng = np.random.default_rng(3)
        sample = rng.normal(loc=0.001, scale=0.02, size=400)
        deviations = np.minimum(sample - 0.0005, 0.0)
        downside = math.sqrt(float(np.mean(deviations**2)))
        expected = (float(np.mean(sample)) - 0.03 / 252) / downside
        got = fin.sortino_ratio(sample, risk_free_annual=0.03, target=0.0005)
        assert got == pytest.approx(expected)

    def test_sortino_infinite_without_downside(self) -> None:
        assert math.isinf(fin.sortino_ratio([0.01, 0.02, 0.03]))

    def test_sortino_periods_validation(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            fin.sortino_ratio(self.returns, periods_per_year=0)


class TestBlackScholes:
    def test_put_call_parity(self) -> None:
        spot, strike, rate = 100.0, 95.0, 0.05
        call = fin.black_scholes(spot, strike, rate, 0.2, 1.0, option="call")
        put = fin.black_scholes(spot, strike, rate, 0.2, 1.0, option="put")
        parity = call.price - put.price
        assert parity == pytest.approx(spot - strike * math.exp(-rate))

    def test_delta_bounds_and_shared_greeks(self) -> None:
        call = fin.black_scholes(100.0, 100.0, 0.05, 0.2, 1.0)
        put = fin.black_scholes(100.0, 100.0, 0.05, 0.2, 1.0, option="put")
        assert 0.0 < call.delta < 1.0
        assert -1.0 < put.delta < 0.0
        assert call.delta == pytest.approx(put.delta + 1.0)
        assert call.gamma == pytest.approx(put.gamma)
        assert call.vega == pytest.approx(put.vega)
        assert call.vega > 0.0 and call.gamma > 0.0

    def test_gamma_and_vega_match_manual_formulas(self) -> None:
        spot, strike, rate, vol, maturity = 100.0, 105.0, 0.03, 0.25, 0.5
        d1 = (math.log(spot / strike) + (rate + 0.5 * vol**2) * maturity) / (
            vol * math.sqrt(maturity)
        )
        density = float(sp_stats.norm.pdf(d1))
        expected_gamma = density / (spot * vol * math.sqrt(maturity))
        expected_vega = spot * density * math.sqrt(maturity)
        result = fin.black_scholes(spot, strike, rate, vol, maturity)
        assert result.gamma == pytest.approx(expected_gamma)
        assert result.vega == pytest.approx(expected_vega)

    def test_zero_maturity_intrinsic_values(self) -> None:
        call = fin.black_scholes(110.0, 100.0, 0.05, 0.2, 0.0)
        put = fin.black_scholes(90.0, 100.0, 0.05, 0.2, 0.0, option="put")
        assert call.price == pytest.approx(10.0)
        assert call.delta == 1.0
        assert put.price == pytest.approx(10.0)
        assert put.delta == -1.0
        assert call.gamma == 0.0 and call.vega == 0.0

    def test_zero_maturity_worthless_side(self) -> None:
        otm_call = fin.black_scholes(90.0, 100.0, 0.0, 0.2, 0.0)
        otm_put = fin.black_scholes(110.0, 100.0, 0.0, 0.2, 0.0, option="put")
        atm_call = fin.black_scholes(100.0, 100.0, 0.0, 0.2, 0.0)
        atm_put = fin.black_scholes(100.0, 100.0, 0.0, 0.2, 0.0, option="put")
        assert otm_call.price == 0.0 and otm_put.price == 0.0
        assert atm_call.delta == 0.0 and atm_put.delta == 0.0

    def test_invalid_option_name_raises(self) -> None:
        with pytest.raises(ValueError, match="option must be call or put"):
            fin.black_scholes(100.0, 100.0, 0.0, 0.2, 1.0, option="straddle")

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="spot and strike"):
            fin.black_scholes(0.0, 100.0, 0.0, 0.2, 1.0)
        with pytest.raises(ValueError, match="spot and strike"):
            fin.black_scholes(100.0, -1.0, 0.0, 0.2, 1.0)
        with pytest.raises(ValueError, match="volatility"):
            fin.black_scholes(100.0, 100.0, 0.0, 0.0, 1.0)
        with pytest.raises(ValueError, match="maturity"):
            fin.black_scholes(100.0, 100.0, 0.0, 0.2, -0.5)


class TestHistoricalVar:
    def test_matches_direct_quantile(self) -> None:
        rng = np.random.default_rng(11)
        sample = rng.normal(loc=0.0, scale=0.01, size=1000)
        for confidence in (0.9, 0.95, 0.99):
            expected = float(-np.quantile(sample, 1.0 - confidence))
            assert fin.historical_var(sample, confidence=confidence) == pytest.approx(expected)

    def test_confidence_must_be_strictly_inside_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="strictly between"):
            fin.historical_var([0.1, -0.1], confidence=0.0)
        with pytest.raises(ValueError, match="strictly between"):
            fin.historical_var([0.1, -0.1], confidence=1.0)


class TestMonteCarloVar:
    def test_deterministic_with_seed(self) -> None:
        first = fin.monte_carlo_var(1_000_000.0, 0.05, 0.2, horizon_days=10, seed=42)
        second = fin.monte_carlo_var(1_000_000.0, 0.05, 0.2, horizon_days=10, seed=42)
        assert first == second
        assert first > 0.0

    def test_close_to_lognormal_closed_form(self) -> None:
        value, mean_return, sigma, horizon = 1_000_000.0, 0.05, 0.2, 10
        lower_z = float(sp_stats.norm.ppf(1.0 - 0.99))
        closed_form = value * (
            1.0
            - math.exp(
                (mean_return - 0.5 * sigma**2) * horizon + sigma * math.sqrt(horizon) * lower_z
            )
        )
        estimate = fin.monte_carlo_var(value, mean_return, sigma, horizon_days=horizon, seed=0)
        assert estimate == pytest.approx(closed_form, rel=0.05)

    def test_validations_raise(self) -> None:
        with pytest.raises(ValueError, match="current_value"):
            fin.monte_carlo_var(-1.0, 0.05, 0.2)
        with pytest.raises(ValueError, match="sigma"):
            fin.monte_carlo_var(1e6, 0.05, 0.0)
        with pytest.raises(ValueError, match="horizon_days"):
            fin.monte_carlo_var(1e6, 0.05, 0.2, horizon_days=0)
        with pytest.raises(ValueError, match="confidence"):
            fin.monte_carlo_var(1e6, 0.05, 0.2, confidence=1.5)
        with pytest.raises(ValueError, match="simulations"):
            fin.monte_carlo_var(1e6, 0.05, 0.2, simulations=99)
