"""Tests for cds2.montecarlo."""

import math

import pytest

from cds2 import montecarlo


class TestPiEstimate:
    def test_converges_to_pi(self) -> None:
        estimate = montecarlo.pi_estimate(n=200_000, seed=42)
        assert estimate == pytest.approx(math.pi, abs=0.02)

    def test_seeded_reproducible(self) -> None:
        first = montecarlo.pi_estimate(n=5_000, seed=7)
        second = montecarlo.pi_estimate(n=5_000, seed=7)
        assert first == second


class TestMCIntegrate:
    def test_known_integral(self) -> None:
        estimate = montecarlo.mc_integrate(lambda x: x**2, 0.0, 1.0, n=100_000, seed=1)
        assert estimate == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_constant_function_exact_in_expectation(self) -> None:
        estimate = montecarlo.mc_integrate(lambda _x: 2.5, -1.0, 3.0, n=10_000, seed=5)
        assert estimate == pytest.approx(10.0, abs=1e-12)


class TestMCExpectation:
    def test_uniform_expectation(self) -> None:
        sampler = lambda rng, size: rng.uniform(-1.0, 1.0, size=size)  # noqa: E731
        result = montecarlo.mc_expectation(lambda x: x**2, sampler, n=100_000, seed=3)
        assert result == pytest.approx(1.0 / 3.0, abs=0.01)


class TestHitOrMiss:
    def test_quarter_circle_area(self) -> None:
        area = montecarlo.hit_or_miss(
            lambda x: math.sqrt(max(1.0 - x * x, 0.0)),
            0.0,
            1.0,
            1.0,
            n=100_000,
            seed=11,
        )
        assert area == pytest.approx(math.pi / 4, abs=0.02)

    def test_rectangle_area(self) -> None:
        area = montecarlo.hit_or_miss(lambda _x: 2.0, 0.0, 5.0, 3.0, n=20_000, seed=2)
        assert area == pytest.approx(10.0, abs=0.5)
