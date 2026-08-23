"""Case study 10: Global optimization race on a multimodal landscape.

Three cds2 metaheuristics - genetic algorithm, particle swarm and
simulated annealing - attack the same rugged potential and are compared
against scipy's differential evolution wrapper.
"""

from __future__ import annotations

import numpy as np

import cds2

BOUNDS = [(-5.0, 5.0), (-5.0, 5.0)]
GLOBAL_MIN = (3.0, 0.5)


def rugged(x: np.ndarray) -> float:
    bowl = (x[0] - GLOBAL_MIN[0]) ** 2 + (x[1] - GLOBAL_MIN[1]) ** 2
    ripples = 0.6 * np.sin(3.0 * x[0]) * np.sin(3.0 * x[1])
    return float(bowl + ripples + 0.05 * bowl**2)


def report(
    label: str, result: cds2.metaheuristics.HeuristicResult | cds2.optimize.GlobalResult
) -> None:
    distance = float(np.linalg.norm(np.asarray(result.x) - np.array(GLOBAL_MIN)))
    iters = int(getattr(result, "n_iterations", getattr(result, "iterations", 0)))
    print(f"{label:20s} f={result.fun:8.4f}  |x-x*|={distance:7.4f}  iters={iters}")


def main() -> None:
    print("== Metaheuristic race on a rippled bowl ==")
    report(
        "genetic algorithm",
        cds2.metaheuristics.genetic_minimize(
            rugged,
            BOUNDS,
            options=cds2.metaheuristics.GeneticOptions(population_size=40, generations=120),
            seed=42,
        ),
    )
    report(
        "particle swarm",
        cds2.metaheuristics.pso_minimize(rugged, BOUNDS, iterations=150, seed=43),
    )
    report(
        "simulated annealing",
        cds2.metaheuristics.simulated_annealing(
            rugged, [0.0, 0.0], bounds=BOUNDS, steps=6000, seed=44
        ),
    )
    report(
        "differential evolution",
        cds2.optimize.differential_evolution(rugged, BOUNDS, seed=45),
    )


if __name__ == "__main__":
    main()
