"""Tests for cds2.epidemiology."""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.optimize import brentq

from cds2 import epidemiology as epi


class TestSimulateSIR:
    def test_grid_and_shapes(self) -> None:
        result = epi.simulate_sir(1000.0, 0.4, 0.1, 120)
        assert result.times.size == result.susceptible.size == 121
        assert result.infected.size == result.recovered.size == 121
        assert result.times[0] == 0.0
        assert result.times[-1] == 120.0

    def test_mass_conservation(self) -> None:
        result = epi.simulate_sir(1000.0, 0.4, 0.1, 120)
        total = result.susceptible + result.infected + result.recovered
        assert np.allclose(total, 1000.0, atol=1e-6)

    def test_susceptible_non_increasing(self) -> None:
        result = epi.simulate_sir(1000.0, 0.4, 0.1, 120)
        assert np.all(np.diff(result.susceptible) <= 1e-9)

    def test_recovered_non_decreasing(self) -> None:
        result = epi.simulate_sir(1000.0, 0.4, 0.1, 120)
        assert np.all(np.diff(result.recovered) >= -1e-9)

    def test_peak_day_matches_argmax(self) -> None:
        result = epi.simulate_sir(1000.0, 0.4, 0.1, 120)
        assert result.peak_day == result.times[int(np.argmax(result.infected))]

    def test_r0_and_attack_rate_match_final_size(self) -> None:
        result = epi.simulate_sir(1000.0, 0.4, 0.1, 120)
        assert result.r0 == 4.0
        assert abs(result.attack_rate - epi.final_size_iteration(4.0)) < 0.01

    def test_growth_when_r0_above_one(self) -> None:
        result = epi.simulate_sir(1000.0, 0.5, 0.1, 10, i0=5.0)
        assert result.infected[1] > result.infected[0]

    def test_decay_when_r0_below_one(self) -> None:
        result = epi.simulate_sir(1000.0, 0.1, 0.5, 10, i0=5.0)
        assert result.infected[1] < result.infected[0]

    def test_custom_i0_and_single_step_per_day(self) -> None:
        result = epi.simulate_sir(500.0, 0.3, 0.15, 40, i0=50.0, steps_per_day=1)
        total = result.susceptible + result.infected + result.recovered
        assert result.infected[0] == 50.0
        assert np.allclose(total, 500.0, atol=1e-6)

    def test_result_is_frozen(self) -> None:
        result = epi.simulate_sir(1000.0, 0.4, 0.1, 5)
        with pytest.raises(FrozenInstanceError):
            result.population = 2000.0


class TestSimulateSEIR:
    def test_initial_conditions_with_seed_exposures(self) -> None:
        result = epi.simulate_seir(1000.0, 0.5, 0.2, 0.1, 60, i0=20.0, e0=10.0)
        assert result.susceptible[0] == 970.0
        assert result.exposed[0] == 10.0
        assert result.infected[0] == 20.0
        assert result.recovered[0] == 0.0

    def test_mass_conservation(self) -> None:
        result = epi.simulate_seir(1000.0, 0.5, 0.2, 0.1, 120)
        total = result.susceptible + result.exposed + result.infected + result.recovered
        assert np.allclose(total, 1000.0, atol=1e-6)

    def test_exposed_nonzero_mid_run_from_zero_seed(self) -> None:
        result = epi.simulate_seir(1000.0, 0.5, 0.2, 0.1, 120)
        assert result.exposed[0] == 0.0
        assert result.exposed[60] > 0.0

    def test_recovered_non_decreasing(self) -> None:
        result = epi.simulate_seir(1000.0, 0.5, 0.2, 0.1, 120)
        assert np.all(np.diff(result.recovered) >= -1e-9)

    def test_peak_day_and_r0(self) -> None:
        result = epi.simulate_seir(1000.0, 0.5, 0.2, 0.1, 120)
        assert result.r0 == 5.0
        assert result.peak_day == result.times[int(np.argmax(result.infected))]

    def test_attack_rate_matches_final_recovered(self) -> None:
        result = epi.simulate_seir(1000.0, 0.5, 0.2, 0.1, 120)
        assert abs(result.attack_rate - result.recovered[-1] / 1000.0) < 1e-12
        assert 0.0 < result.attack_rate < 1.0


class TestHerdImmunityThreshold:
    def test_known_value(self) -> None:
        assert epi.herd_immunity_threshold(4.0) == 0.75

    def test_critical_value(self) -> None:
        assert epi.herd_immunity_threshold(1.0) == 0.0

    @pytest.mark.parametrize("r0", [0.0, -2.5])
    def test_invalid_raises(self, r0: float) -> None:
        with pytest.raises(ValueError, match="r0 must be positive"):
            epi.herd_immunity_threshold(r0)


class TestEffectiveReproduction:
    def test_basic_math(self) -> None:
        assert epi.effective_reproduction(3.0, 0.5) == 1.5

    def test_bounds(self) -> None:
        assert epi.effective_reproduction(2.0, 0.0) == 0.0
        assert epi.effective_reproduction(4.0, 1.0) == 4.0


class TestFinalSizeIteration:
    @staticmethod
    def _objective(z: float, r0: float) -> float:
        return z - (1.0 - math.exp(-r0 * z))

    def test_matches_brentq_root_for_r0_two(self) -> None:
        root = brentq(self._objective, 1e-9, 1.0 - 1e-9, args=(2.0,))
        assert abs(epi.final_size_iteration(2.0) - root) < 1e-6

    def test_matches_brentq_root_for_r0_five(self) -> None:
        root = brentq(self._objective, 1e-9, 1.0 - 1e-9, args=(5.0,))
        assert abs(epi.final_size_iteration(5.0) - root) < 1e-6

    def test_subcritical_converges_to_zero(self) -> None:
        assert 0.0 <= epi.final_size_iteration(0.5) < 1e-8

    def test_max_iter_exceeded_raises(self) -> None:
        with pytest.raises(RuntimeError, match="final-size iteration did not converge"):
            epi.final_size_iteration(2.0, max_iter=1)


class TestSIRValidation:
    def test_population_not_positive(self) -> None:
        with pytest.raises(ValueError, match="population must be positive"):
            epi.simulate_sir(0.0, 0.4, 0.1, 10)

    def test_days_below_one(self) -> None:
        with pytest.raises(ValueError, match="days must be at least 1"):
            epi.simulate_sir(1000.0, 0.4, 0.1, 0)

    def test_steps_per_day_below_one(self) -> None:
        with pytest.raises(ValueError, match="steps_per_day must be at least 1"):
            epi.simulate_sir(1000.0, 0.4, 0.1, 10, steps_per_day=0)

    def test_beta_not_positive(self) -> None:
        with pytest.raises(ValueError, match="beta must be positive"):
            epi.simulate_sir(1000.0, 0.0, 0.1, 10)

    def test_gamma_negative(self) -> None:
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            epi.simulate_sir(1000.0, 0.4, -0.1, 10)

    @pytest.mark.parametrize("i0", [-1.0, 1001.0])
    def test_i0_outside_population(self, i0: float) -> None:
        with pytest.raises(ValueError, match="initial infections must lie within the population"):
            epi.simulate_sir(1000.0, 0.4, 0.1, 10, i0=i0)


class TestSEIRValidation:
    def test_population_not_positive(self) -> None:
        with pytest.raises(ValueError, match="population must be positive"):
            epi.simulate_seir(-5.0, 0.5, 0.2, 0.1, 10)

    def test_days_below_one(self) -> None:
        with pytest.raises(ValueError, match="days must be at least 1"):
            epi.simulate_seir(1000.0, 0.5, 0.2, 0.1, 0)

    def test_steps_per_day_below_one(self) -> None:
        with pytest.raises(ValueError, match="steps_per_day must be at least 1"):
            epi.simulate_seir(1000.0, 0.5, 0.2, 0.1, 10, steps_per_day=-3)

    def test_beta_not_positive(self) -> None:
        with pytest.raises(ValueError, match="beta must be positive"):
            epi.simulate_seir(1000.0, -1.0, 0.2, 0.1, 10)

    def test_sigma_not_positive(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            epi.simulate_seir(1000.0, 0.5, 0.0, 0.1, 10)

    def test_gamma_negative(self) -> None:
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            epi.simulate_seir(1000.0, 0.5, 0.2, -1.0, 10)

    def test_i0_outside_population(self) -> None:
        with pytest.raises(ValueError, match="initial infections must lie within the population"):
            epi.simulate_seir(1000.0, 0.5, 0.2, 0.1, 10, i0=-1.0)

    def test_e0_negative(self) -> None:
        with pytest.raises(ValueError, match="initial exposures must be non-negative"):
            epi.simulate_seir(1000.0, 0.5, 0.2, 0.1, 10, e0=-0.5)

    def test_e0_plus_i0_exceeds_population(self) -> None:
        with pytest.raises(
            ValueError,
            match="initial exposed and infected must fit within the population",
        ):
            epi.simulate_seir(1000.0, 0.5, 0.2, 0.1, 10, i0=600.0, e0=500.0)
