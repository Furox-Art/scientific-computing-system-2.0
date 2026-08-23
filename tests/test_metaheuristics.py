"""Tests for cds2.metaheuristics."""

import numpy as np
import pytest

from cds2 import metaheuristics

SPHERE = [(-10.0, 10.0), (-10.0, 10.0)]


def quadratic(vector: np.ndarray) -> float:
    return float((vector[0] - 3.0) ** 2 + (vector[1] + 2.0) ** 2)


class TestGeneticOptions:
    def test_defaults_valid(self) -> None:
        options = metaheuristics.GeneticOptions()
        assert options.population_size == 60

    def test_invalid_population(self) -> None:
        with pytest.raises(ValueError, match="population_size"):
            metaheuristics.GeneticOptions(population_size=2)

    def test_invalid_rates(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            metaheuristics.GeneticOptions(crossover_rate=1.5)

    def test_invalid_elitism(self) -> None:
        with pytest.raises(ValueError, match="elitism"):
            metaheuristics.GeneticOptions(elitism=99)


class TestGenetic:
    def test_finds_minimum_region(self) -> None:
        result = metaheuristics.genetic_minimize(
            quadratic,
            SPHERE,
            options=metaheuristics.GeneticOptions(population_size=30, generations=80, elitism=2),
            seed=4,
        )
        assert abs(result.x[0] - 3.0) < 1.0
        assert abs(result.x[1] + 2.0) < 1.0

    def test_history_monotone(self) -> None:
        result = metaheuristics.genetic_minimize(
            quadratic,
            SPHERE,
            options=metaheuristics.GeneticOptions(population_size=12, generations=25),
            seed=9,
        )
        values = np.asarray(result.history)
        assert np.all(np.diff(values) <= 1e-12)
        assert len(result.history) == 26

    def test_bounds_respected_and_bad_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="pairs"):
            metaheuristics.genetic_minimize(quadratic, [-1.0, 1.0], seed=1)

    def test_result_inside_bounds(self) -> None:
        result = metaheuristics.genetic_minimize(quadratic, SPHERE, seed=2)
        assert np.all(result.x >= -10.0)
        assert np.all(result.x <= 10.0)


class TestPSO:
    def test_converges_to_quadratic_minimum(self) -> None:
        result = metaheuristics.pso_minimize(quadratic, SPHERE, iterations=120, seed=11)
        assert result.fun < 1e-6
        assert abs(result.x[0] - 3.0) < 0.05

    def test_history_length(self) -> None:
        result = metaheuristics.pso_minimize(quadratic, SPHERE, iterations=30, seed=1)
        assert len(result.history) == 31
        assert result.iterations == 30

    def test_invalid_sizes_raise(self) -> None:
        with pytest.raises(ValueError, match="non-positive"):
            metaheuristics.pso_minimize(quadratic, SPHERE, swarm_size=1)


class TestSimulatedAnnealing:
    def test_unbounded_one_dimensional(self) -> None:
        result = metaheuristics.simulated_annealing(
            lambda v: (v[0] + 4.0) ** 2, initial=[20.0], steps=4000, seed=21
        )
        assert abs(result.x[0] + 4.0) < 0.75

    def test_bounded_two_dimensional(self) -> None:
        result = metaheuristics.simulated_annealing(
            quadratic, initial=[0.0, 0.0], bounds=SPHERE, steps=5000, seed=31
        )
        assert result.fun < 1.0

    def test_invalid_cooling_raises(self) -> None:
        with pytest.raises(ValueError, match="cooling_rate"):
            metaheuristics.simulated_annealing(quadratic, [0.0], cooling_rate=1.2)

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            metaheuristics.simulated_annealing(quadratic, [0.0], bounds=[(-1.0, 1.0), (-1.0, 1.0)])


class TestCoverageEdges:
    def test_odd_child_slots_handled(self) -> None:
        options = metaheuristics.GeneticOptions(population_size=13, generations=6, elitism=2)
        result = metaheuristics.genetic_minimize(quadratic, SPHERE, options=options, seed=5)
        assert len(result.history) == 7
