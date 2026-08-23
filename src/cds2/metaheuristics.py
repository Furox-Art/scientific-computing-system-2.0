"""Population and annealing metaheuristics: genetic algorithm, PSO, simulated annealing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "HeuristicResult",
    "GeneticOptions",
    "genetic_minimize",
    "pso_minimize",
    "simulated_annealing",
]

FloatArray = NDArray[np.float64]

Objective = Callable[[FloatArray], float]


@dataclass(frozen=True)
class HeuristicResult:
    """Best point found by a metaheuristic plus its search statistics."""

    x: FloatArray
    fun: float
    iterations: int
    history: tuple[float, ...]


@dataclass(frozen=True)
class GeneticOptions:
    """Knobs of the real-coded genetic algorithm."""

    population_size: int = 60
    generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.15
    tournament_size: int = 3
    elitism: int = 2

    def __post_init__(self) -> None:
        if self.population_size < 4 or self.generations < 1:
            msg = "population_size >= 4 and generations >= 1 are required"
            raise ValueError(msg)
        if not 0.0 <= self.crossover_rate <= 1.0 or not 0.0 <= self.mutation_rate <= 1.0:
            msg = "crossover_rate and mutation_rate must lie in [0, 1]"
            raise ValueError(msg)
        if self.tournament_size < 2 or not 0 <= self.elitism < self.population_size:
            msg = "tournament_size >= 2 and 0 <= elitism < population_size required"
            raise ValueError(msg)


def _clip_to_bounds(point: FloatArray, bounds: FloatArray) -> FloatArray:
    return np.clip(point, bounds[:, 0], bounds[:, 1])


def genetic_minimize(
    objective: Objective,
    bounds: Sequence[tuple[float, float]],
    options: GeneticOptions | None = None,
    seed: int | None = None,
) -> HeuristicResult:
    """Real-coded genetic algorithm with tournament selection and blend crossover."""
    config = options or GeneticOptions()
    rng = np.random.default_rng(seed)
    limits = np.asarray(bounds, dtype=float)
    if limits.ndim != 2 or limits.shape[1] != 2:
        msg = "bounds must be a sequence of (low, high) pairs"
        raise ValueError(msg)
    dimension = limits.shape[0]
    low = limits[:, 0]
    high = limits[:, 1]

    population = rng.uniform(low, high, size=(config.population_size, dimension))
    fitness = np.array([objective(individual) for individual in population])
    best_index = int(np.argmin(fitness))
    best_point = population[best_index].copy()
    best_value = float(fitness[best_index])
    history: list[float] = [best_value]

    for generation in range(config.generations):
        order = np.argsort(fitness)
        elite = population[order[: config.elitism]].copy()

        children = np.empty_like(population)
        slot = 0
        while slot < config.population_size - config.elitism:
            parent_a = _tournament(population, fitness, config.tournament_size, rng)
            parent_b = _tournament(population, fitness, config.tournament_size, rng)
            child_a = parent_a.copy()
            child_b = parent_b.copy()
            if rng.random() < config.crossover_rate:
                weight = rng.random(dimension)
                child_a = parent_a + weight * (parent_b - parent_a)
                child_b = parent_b + weight * (parent_a - parent_b)
            for child in (child_a, child_b):
                mask = rng.random(dimension) < config.mutation_rate
                perturbation = rng.normal(scale=0.1 * (high - low), size=dimension)
                child[mask] += perturbation[mask]
            children[slot] = _clip_to_bounds(child_a, limits)
            if slot + 1 < config.population_size - config.elitism:
                children[slot + 1] = _clip_to_bounds(child_b, limits)
            slot += 2

        keep = max(config.population_size - config.elitism, 0)
        population[:keep] = children[:keep]
        population[keep:] = elite
        fitness = np.array([objective(individual) for individual in population])
        current_best = int(np.argmin(fitness))
        if float(fitness[current_best]) < best_value:
            best_value = float(fitness[current_best])
            best_point = population[current_best].copy()
        history.append(best_value)

    return HeuristicResult(
        x=_clip_to_bounds(best_point, limits),
        fun=best_value,
        iterations=config.generations,
        history=tuple(history),
    )


def _tournament(
    population: FloatArray, fitness: FloatArray, size: int, rng: np.random.Generator
) -> FloatArray:
    contenders = rng.integers(0, population.shape[0], size=size)
    winner = int(contenders[np.argmin(fitness[contenders])])
    return np.asarray(population[winner], dtype=float)


def pso_minimize(
    objective: Objective,
    bounds: Sequence[tuple[float, float]],
    swarm_size: int = 40,
    iterations: int = 150,
    inertia: float = 0.72,
    cognitive: float = 1.49,
    social: float = 1.49,
    seed: int | None = None,
) -> HeuristicResult:
    """Particle swarm optimization over a box-constrained continuous space."""
    rng = np.random.default_rng(seed)
    limits = np.asarray(bounds, dtype=float)
    if limits.ndim != 2 or limits.shape[1] != 2 or swarm_size < 2 or iterations < 1:
        msg = "invalid bounds or non-positive sizes"
        raise ValueError(msg)

    positions = rng.uniform(limits[:, 0], limits[:, 1], size=(swarm_size, limits.shape[0]))
    values = np.array([objective(p) for p in positions])
    personal_best_positions = positions.copy()
    personal_best_values = values.copy()
    global_index = int(np.argmin(values))
    global_best_position = positions[global_index].copy()
    global_best_value = float(values[global_index])
    velocities = rng.uniform(-1.0, 1.0, size=positions.shape) * (limits[:, 1] - limits[:, 0]) * 0.1
    history: list[float] = [global_best_value]

    for _ in range(iterations):
        r_cognitive = rng.random(positions.shape)
        r_social = rng.random(positions.shape)
        velocities = (
            inertia * velocities
            + cognitive * r_cognitive * (personal_best_positions - positions)
            + social * r_social * (global_best_position - positions)
        )
        positions = _clip_to_bounds(positions + velocities, limits)
        values = np.array([objective(p) for p in positions])

        improved = values < personal_best_values
        personal_best_values = np.where(improved, values, personal_best_values)
        personal_best_positions[improved] = positions[improved]
        current_global = int(np.argmin(personal_best_values))
        if personal_best_values[current_global] < global_best_value:
            global_best_value = float(personal_best_values[current_global])
            global_best_position = personal_best_positions[current_global].copy()
        history.append(global_best_value)

    return HeuristicResult(
        x=global_best_position,
        fun=global_best_value,
        iterations=iterations,
        history=tuple(history),
    )


def simulated_annealing(
    objective: Objective,
    initial: Sequence[float],
    bounds: Sequence[tuple[float, float]] | None = None,
    steps: int = 2_000,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.995,
    step_scale: float = 0.3,
    seed: int | None = None,
) -> HeuristicResult:
    """Gaussian-proposal simulated annealing with exponential cooling."""
    if steps < 1 or initial_temperature <= 0 or not 0.0 < cooling_rate <= 1.0:
        msg = "steps >= 1, temperature > 0 and 0 < cooling_rate <= 1 are required"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    current = np.atleast_1d(np.asarray(initial, dtype=float)).copy()
    limits: FloatArray | None = None
    if bounds is not None:
        candidate = np.asarray(bounds, dtype=float)
        if candidate.ndim != 2 or candidate.shape[1] != 2 or candidate.shape[0] != current.size:
            msg = "bounds must match the state dimension"
            raise ValueError(msg)
        limits = candidate
        current = _clip_to_bounds(current, limits)
    current_value = objective(current)
    best_point = current.copy()
    best_value = current_value
    temperature = initial_temperature
    history: list[float] = [best_value]

    for _ in range(steps):
        proposal = current + rng.normal(scale=step_scale, size=current.size)
        if limits is not None:
            proposal = _clip_to_bounds(proposal, limits)
        proposal_value = objective(proposal)
        delta = proposal_value - current_value
        if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
            current = proposal
            current_value = proposal_value
            if current_value < best_value:
                best_value = current_value
                best_point = current.copy()
        temperature *= cooling_rate
        history.append(best_value)

    result_x = _clip_to_bounds(best_point, limits) if limits is not None else best_point
    return HeuristicResult(
        x=result_x,
        fun=float(best_value),
        iterations=steps,
        history=tuple(history),
    )
