"""Tests for cds2.chaos."""

import numpy as np
import pytest

from cds2 import chaos


@pytest.fixture()
def chaotic_series() -> np.ndarray:
    return chaos.logistic_map(3.99, length=600, x0=0.2, seed=42)


class TestDelayEmbed:
    def test_shape(self) -> None:
        result = chaos.delay_embed(np.arange(10.0), dimension=3, delay=2)
        assert result.embedded.shape == (6, 3)
        assert result.dimension == 3
        assert result.delay == 2

    def test_content(self) -> None:
        series = np.arange(10.0)
        embedded = chaos.delay_embed(series, dimension=2, delay=1).embedded
        assert embedded[0, 0] == 0.0
        assert embedded[0, 1] == 1.0

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="samples"):
            chaos.delay_embed([1.0, 2.0], dimension=5)

    def test_invalid_dimension(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            chaos.delay_embed([1.0, 2.0], dimension=0)

    def test_invalid_delay(self) -> None:
        with pytest.raises(ValueError, match="delay"):
            chaos.delay_embed([1.0, 2.0], delay=0)


class TestFalseNearest:
    def test_low_dimension_for_regular_series(self) -> None:
        sine = np.sin(np.linspace(0.0, 40.0 * np.pi, 800))
        assert chaos.false_nearest_fraction(sine, dimension=1) > chaos.false_nearest_fraction(
            sine, dimension=3
        )


class TestLyapunov:
    def test_positive_for_chaotic_logistic(self, chaotic_series: np.ndarray) -> None:
        result = chaos.largest_lyapunov_exponent(chaotic_series)
        assert result.exponent > 0.1

    def test_nonpositive_for_fixed_point(self) -> None:
        stable = chaos.logistic_map(2.5, length=400, x0=0.3, seed=7)
        chaotic = chaos.logistic_map(3.99, length=400, x0=0.2, seed=8)
        stable_exponent = chaos.largest_lyapunov_exponent(stable).exponent
        chaotic_exponent = chaos.largest_lyapunov_exponent(chaotic).exponent
        assert chaotic_exponent > stable_exponent + 0.2

    def test_horizons_match_divergence_length(self, chaotic_series: np.ndarray) -> None:
        result = chaos.largest_lyapunov_exponent(chaotic_series, max_horizon=6)
        assert result.horizons.size == result.mean_log_divergence.size == 6

    def test_short_series_raises(self) -> None:
        with pytest.raises(ValueError):
            chaos.largest_lyapunov_exponent([1.0, 2.0])


class TestCorrelationDimension:
    def test_below_embedding_dimension(self, chaotic_series: np.ndarray) -> None:
        result = chaos.correlation_dimension(chaotic_series, embedding_dimension=4)
        assert 0.0 < result.dimension < 4.0

    def test_explicit_radii(self, chaotic_series: np.ndarray) -> None:
        radii = [0.01, 0.03, 0.1, 0.3]
        result = chaos.correlation_dimension(chaotic_series, radii=radii)
        assert result.radii.size == len(radii)
        assert np.all(np.diff(result.correlations[result.correlations > 0]) >= 0)

    def test_single_radius_raises(self, chaotic_series: np.ndarray) -> None:
        with pytest.raises(ValueError, match="two distinct"):
            chaos.correlation_dimension(chaotic_series, radii=[0.1])

    def test_constant_series_raises(self) -> None:
        with pytest.raises(ValueError, match="degenerate|too small"):
            chaos.correlation_dimension(np.ones(200))


class TestSampleEntropy:
    def test_noise_more_complex_than_sine(self) -> None:
        rng = np.random.default_rng(1)
        noise = rng.normal(size=600)
        sine = np.sin(np.linspace(0.0, 60.0 * np.pi, 600))
        assert chaos.sample_entropy(noise) > chaos.sample_entropy(sine)

    def test_short_series_raises(self) -> None:
        with pytest.raises(ValueError, match="short"):
            chaos.sample_entropy([1.0])

    def test_bad_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            chaos.sample_entropy(np.ones(50), tolerance=0.0)


class TestHurst:
    def test_random_walk_persists(self) -> None:
        rng = np.random.default_rng(2)
        walk = np.cumsum(rng.normal(size=3000))
        assert chaos.hurst_exponent(walk) > 0.6

    def test_white_noise_around_half(self) -> None:
        rng = np.random.default_rng(3)
        noise = rng.normal(size=4000)
        assert 0.35 < chaos.hurst_exponent(noise) < 0.65

    def test_short_series_raises(self) -> None:
        with pytest.raises(ValueError, match="R/S"):
            chaos.hurst_exponent([1.0] * 10)


class TestLogisticMap:
    def test_range_and_length(self) -> None:
        out = chaos.logistic_map(3.9, length=100, seed=5)
        assert out.shape == (100,)
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_r_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 4\]"):
            chaos.logistic_map(4.5)

    def test_invalid_length(self) -> None:
        with pytest.raises(ValueError, match="length"):
            chaos.logistic_map(3.0, length=0)


class TestBifurcationScan:
    def test_flat_arrays(self) -> None:
        params, states = chaos.bifurcation_scan(
            lambda r, x: r * x * (1.0 - x),
            np.linspace(2.5, 4.0, num=20),
            iterations=150,
            last_values=20,
            seed=1,
        )
        assert params.shape == (20 * 20,) == states.shape
        assert np.all((states >= -1e-9) & (states <= 1.0 + 1e-9))

    def test_iterations_must_exceed_window(self) -> None:
        with pytest.raises(ValueError, match="exceed"):
            chaos.bifurcation_scan(lambda r, x: x, [1.0], iterations=5, last_values=10)


class TestCoverageEdges:
    def test_false_nearest_too_short(self) -> None:
        with pytest.raises(ValueError, match="short"):
            chaos.false_nearest_fraction([1.0, 2.0, 3.0], dimension=1)

    def test_lyapunov_single_point_raises(self) -> None:
        with pytest.raises(ValueError, match="short"):
            chaos.largest_lyapunov_exponent([1.0], embedding_dimension=1)

    def test_lyapunov_usable_window_too_small(self) -> None:
        with pytest.raises(ValueError, match="short"):
            chaos.largest_lyapunov_exponent([1.0, 2.0, 3.0], embedding_dimension=1, max_horizon=5)

    def test_lyapunov_no_admissible_neighbours(self) -> None:
        with pytest.raises(ValueError, match="admissible"):
            chaos.largest_lyapunov_exponent(np.arange(6.0), embedding_dimension=1, max_horizon=4)

    def test_lyapunov_constant_series_zero_exponent(self) -> None:
        result = chaos.largest_lyapunov_exponent(np.full(60, 2.5), embedding_dimension=1)
        assert result.exponent == 0.0

    def test_correlation_dimension_radii_below_scale(self) -> None:
        series = chaos.logistic_map(3.9, length=200, seed=3)
        with pytest.raises(ValueError, match="too small"):
            chaos.correlation_dimension(series, radii=[1e-15, 2e-15])

    def test_sample_entropy_no_matches_returns_zero(self) -> None:
        assert chaos.sample_entropy(np.arange(100.0), dimension=2, tolerance=1e-6) == 0.0

    def test_hurst_constant_series_raises(self) -> None:
        with pytest.raises(ValueError, match="window sizes"):
            chaos.hurst_exponent([7.0] * 64)

    def test_hurst_alternating_low(self) -> None:
        alternating = np.array([(-1.0) ** k for k in range(2048)])
        assert chaos.hurst_exponent(alternating) < 0.4

    def test_hurst_trend_high(self) -> None:
        trend = np.arange(2048, dtype=float)
        assert chaos.hurst_exponent(trend) > 0.9
