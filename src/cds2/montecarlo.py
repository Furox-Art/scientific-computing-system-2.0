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
    """Estimate pi by uniform sampling inside the unit square."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    y = rng.uniform(0.0, 1.0, size=n)
    inside = np.count_nonzero(x * x + y * y <= 1.0)
    return float(4.0 * inside / n)


def mc_integrate(
    func: Callable[[FloatArray], FloatArray],
    a: float,
    b: float,
    n: int = 100_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo estimate of the oriented integral of ``func`` from a to b."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("integration bounds must be finite")
    if a == b:
        return 0.0
    low, high = (a, b) if a < b else (b, a)
    orientation = 1.0 if a < b else -1.0
    rng = np.random.default_rng(seed)
    samples = rng.uniform(low, high, size=n)
    values = np.asarray(func(samples), dtype=float)
    if values.ndim == 0:
        mean_value = float(values)
    else:
        if values.shape != samples.shape:
            raise ValueError("func must return a scalar or one value per sample")
        if not bool(np.all(np.isfinite(values))):
            raise ValueError("func returned non-finite values")
        mean_value = float(np.mean(values))
    if not np.isfinite(mean_value):
        raise ValueError("func returned a non-finite value")
    return float(orientation * (high - low) * mean_value)


def mc_expectation(
    func: Callable[[FloatArray], FloatArray],
    sampler: Callable[[np.random.Generator, int], FloatArray],
    n: int = 100_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo expectation E[func(X)] where ``sampler(rng, n)`` draws X."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    samples = np.asarray(sampler(rng, n), dtype=float)
    if samples.ndim == 0 or samples.shape[0] != n:
        raise ValueError("sampler must return n samples")
    if not bool(np.all(np.isfinite(samples))):
        raise ValueError("sampler returned non-finite samples")
    values = np.asarray(func(samples), dtype=float)
    if values.ndim == 0:
        result = float(values)
    else:
        if values.shape != (n,):
            raise ValueError("func must return a scalar or one value per sample")
        if not bool(np.all(np.isfinite(values))):
            raise ValueError("func returned non-finite values")
        result = float(np.mean(values))
    if not np.isfinite(result):
        raise ValueError("func returned a non-finite value")
    return result


def hit_or_miss(
    func: Callable[..., object],
    a: float,
    b: float,
    y_max: float,
    n: int = 100_000,
    seed: int | None = None,
) -> float:
    """Hit-or-miss area estimate for ``0 <= func(x) <= y_max`` on ``[a, b]``."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        raise ValueError("hit-or-miss requires finite bounds with b > a")
    if not np.isfinite(y_max) or y_max <= 0.0:
        raise ValueError("y_max must be positive and finite")
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n)
    y = rng.uniform(0.0, y_max, size=n)
    try:
        raw = func(x)
    except (TypeError, ValueError):
        f_values = np.array([func(xi) for xi in x.tolist()], dtype=float)
    else:
        candidate = np.asarray(raw, dtype=float)
        if candidate.ndim == 0:
            f_values = np.full_like(x, float(candidate), dtype=float)
        elif candidate.shape == x.shape:
            f_values = candidate
        else:
            raise ValueError("func must return a scalar or one value per sample")
    if not bool(np.all(np.isfinite(f_values))):
        raise ValueError("func returned non-finite values")
    if np.any(f_values < 0.0) or np.any(f_values > y_max):
        raise ValueError("func values must lie within [0, y_max]")
    hits = int(np.count_nonzero(y <= f_values))
    return float((b - a) * y_max * hits / n)


def metropolis_hastings(
    log_prob: Callable[[FloatArray], float],
    initial: object,
    n_samples: int = 10_000,
    burn_in: int = 1_000,
    proposal_scale: float = 1.0,
    thin: int = 1,
    seed: int | None = None,
) -> MCMCResult:
    """Metropolis-Hastings sampler for an unnormalized log density."""
    integer_args = (("n_samples", n_samples), ("burn_in", burn_in), ("thin", thin))
    for name, value in integer_args:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if n_samples < 1 or burn_in < 0 or thin < 1:
        raise ValueError("need n_samples >= 1, burn_in >= 0 and thin >= 1")
    if not np.isfinite(proposal_scale) or proposal_scale <= 0.0:
        raise ValueError("proposal_scale must be positive and finite")
    rng = np.random.default_rng(seed)
    current = np.atleast_1d(np.asarray(initial, dtype=float)).copy()
    if current.ndim != 1 or current.size == 0 or not bool(np.all(np.isfinite(current))):
        raise ValueError("initial must be a non-empty finite 1-D state")
    current_log_prob = float(log_prob(current))
    if np.isnan(current_log_prob) or current_log_prob == float("inf"):
        raise ValueError("log_prob(initial) must be finite or -inf")
    total_steps = burn_in + n_samples * thin
    samples = np.empty((n_samples, current.size))
    accepted = 0
    kept = 0
    for step in range(total_steps):
        proposal = current + rng.normal(scale=proposal_scale, size=current.size)
        proposal_log_prob = float(log_prob(proposal))
        if np.isnan(proposal_log_prob) or proposal_log_prob == float("inf"):
            raise ValueError("log_prob must not return NaN or +inf")
        if current_log_prob == float("-inf"):
            accept = proposal_log_prob > float("-inf")
        elif proposal_log_prob == float("-inf"):
            accept = False
        else:
            accept = np.log(rng.random()) < proposal_log_prob - current_log_prob
        if accept:
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
    """Chunked Monte Carlo integration using exactly ``n_total`` samples."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    if not isinstance(n_total, (int, np.integer)) or isinstance(n_total, bool) or n_total < 1:
        raise ValueError("n_total must be a positive integer")
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        raise ValueError("b must be greater than a and both bounds must be finite")
    if workers is not None and (
        not isinstance(workers, (int, np.integer)) or isinstance(workers, bool) or workers < 1
    ):
        raise ValueError("workers must be a positive integer or None")
    requested_workers = workers if workers is not None else min(os.cpu_count() or 2, 4)
    worker_count = min(int(requested_workers), int(n_total))
    width = (b - a) / worker_count
    quotient, remainder = divmod(int(n_total), worker_count)
    counts = [quotient + (1 if index < remainder else 0) for index in range(worker_count)]
    if seed is None:
        seeds: list[int | None] = [None] * worker_count
    else:
        children = np.random.SeedSequence(seed).spawn(worker_count)
        seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    jobs = [
        (func, a + index * width, a + (index + 1) * width, counts[index], seeds[index])
        for index in range(worker_count)
    ]
    try:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:  # pragma: no cover
            estimates = list(pool.map(_integrate_chunk, jobs))
    except Exception:  # pragma: no cover - process/pickling fallback
        estimates = [_integrate_chunk(job) for job in jobs]
    return float(np.sum(estimates))
