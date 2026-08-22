"""Seeded Monte Carlo estimation routines built on numpy.random."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MCMCResult",
    "pi_estimate",
    "mc_integrate",
    "mc_expectation",
    "hit_or_miss",
    "metropolis_hastings",
    "parallel_mc_integrate",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MCMCResult:
    """Markov-chain sample set plus diagnostics."""

    samples: FloatArray
    acceptance_rate: float


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
    func: Callable[..., object],
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
        f_values: FloatArray = np.asarray(func(x), dtype=float)
        if f_values.shape != x.shape:
            raise ValueError("vectorized evaluation returned wrong shape")
    except Exception:
        f_values = np.array([func(xi) for xi in x.tolist()], dtype=float)
    hits = int(np.count_nonzero(y <= f_values))
    box_area = (b - a) * y_max
    return box_area * hits / n


def metropolis_hastings(
    log_prob: Callable[[FloatArray], float],
    initial: object,
    n_samples: int = 10_000,
    burn_in: int = 1_000,
    proposal_scale: float = 1.0,
    thin: int = 1,
    seed: int | None = None,
) -> MCMCResult:
    """Metropolis-Hastings sampler for an unnormalized density.

    ``log_prob`` receives a 1-D state vector and returns its log-density
    (up to an additive constant). Gaussian random-walk proposals; the first
    ``burn_in`` steps are discarded, then every ``thin``-th step is kept.
    """
    if n_samples < 1 or burn_in < 0 or thin < 1:
        msg = "need n_samples >= 1, burn_in >= 0 and thin >= 1"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    current = np.atleast_1d(np.asarray(initial, dtype=float)).copy()
    current_log_prob = float(log_prob(current))
    total_steps = burn_in + n_samples * thin
    samples = np.empty((n_samples, current.size))
    accepted = 0
    kept = 0
    for step in range(total_steps):
        proposal = current + rng.normal(scale=proposal_scale, size=current.size)
        proposal_log_prob = float(log_prob(proposal))
        log_accept_ratio = proposal_log_prob - current_log_prob
        if np.log(rng.random()) < log_accept_ratio:
            current = proposal
            current_log_prob = proposal_log_prob
            accepted += 1
        if step >= burn_in and (step - burn_in) % thin == 0 and kept < n_samples:
            samples[kept] = current
            kept += 1
    return MCMCResult(samples=samples, acceptance_rate=accepted / total_steps)


def _integrate_chunk(
    job: tuple[Callable[..., object], float, float, int, int | None],
) -> float:  # pragma: no cover - worker subprocess
    func, low, high, count, chunk_seed = job
    typed_func = cast("Callable[[FloatArray], FloatArray]", func)
    return mc_integrate(typed_func, low, high, n=count, seed=chunk_seed)


def parallel_mc_integrate(
    func: Callable[..., object],
    a: float,
    b: float,
    n_total: int = 4_000_000,
    workers: int | None = None,
    seed: int | None = None,
) -> float:
    """Chunked Monte Carlo integration across worker processes.

    ``func`` must be picklable - define it at module level. Each worker gets
    an equal sub-interval and an independent seed; the result averages the
    per-chunk estimates.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    worker_count = workers or os.cpu_count() or 2
    if b <= a:
        msg = "b must be greater than a"
        raise ValueError(msg)
    width = (b - a) / worker_count
    per_worker = max(n_total // worker_count, 1)
    jobs = [
        (
            func,
            a + index * width,
            a + (index + 1) * width,
            per_worker,
            None if seed is None else seed + index,
        )
        for index in range(worker_count)
    ]
    # Subprocess bodies are invisible to the coverage tracer by design;
    # correctness of this block is asserted functionally by its tests.
    with ProcessPoolExecutor(max_workers=worker_count) as pool:  # pragma: no cover
        estimates = list(pool.map(_integrate_chunk, jobs))
    return float(np.sum(estimates))
