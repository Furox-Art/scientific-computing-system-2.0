"""Seeded Monte Carlo estimation routines built on numpy.random."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["pi_estimate", "mc_integrate", "mc_expectation", "hit_or_miss"]

FloatArray = NDArray[np.float64]


def pi_estimate(n: int = 100_000, seed: int | None = None) -> float:
    """Estimate pi by uniform sampling inside the unit square.

    Counts the fraction of points falling within the quarter unit circle.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    y = rng.uniform(0.0, 1.0, size=n)
    inside = np.count_nonzero(x * x + y * y <= 1.0)
    return 4.0 * inside / n


def mc_integrate(
    func: Callable[[FloatArray], FloatArray],
    a: float,
    b: float,
    n: int = 100_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo estimate of the integral of ``func`` over [a, b].

    ``func`` is applied elementwise to arrays of sample locations.
    """
    rng = np.random.default_rng(seed)
    samples = rng.uniform(a, b, size=n)
    values = np.asarray(func(samples), dtype=float)
    return float((b - a) * np.mean(values))


def mc_expectation(
    func: Callable[[FloatArray], FloatArray],
    sampler: Callable[[np.random.Generator, int], FloatArray],
    n: int = 100_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo expectation E[func(X)] where ``sampler(rng, n)`` draws X.

    Passing a seeded ``seed`` makes both sampling and the result reproducible.
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(sampler(rng, n), dtype=float)
    values = np.asarray(func(samples), dtype=float)
    return float(np.mean(values))


def hit_or_miss(
    func: Callable[[float], float],
    a: float,
    b: float,
    y_max: float,
    n: int = 100_000,
    seed: int | None = None,
) -> float:
    """Hit-or-miss area estimate for non-negative ``func`` on [a, b] x [0, y_max]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n)
    y = rng.uniform(0.0, y_max, size=n)
    try:
        f_values = np.asarray(func(x), dtype=float)
        if f_values.shape != x.shape:
            raise ValueError("vectorized evaluation returned wrong shape")
    except Exception:
        f_values = np.array([func(xi) for xi in x], dtype=float)
    hits = int(np.count_nonzero(y <= f_values))
    box_area = (b - a) * y_max
    return box_area * hits / n
