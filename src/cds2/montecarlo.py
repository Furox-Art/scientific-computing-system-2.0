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
    "split_rhat",
    "effective_sample_size",
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


#: Sample budget is split into this many fixed chunks (or fewer for tiny
#: budgets). The chunk count must never derive from ``workers``: doing so made
#: the estimate depend on the CPU topology, so the same seed produced different
#: numbers on different machines.
_PARALLEL_CHUNKS = 32


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
    """Chunked Monte Carlo integration using exactly ``n_total`` samples.

    The sample budget is split into a fixed number of chunks independent of
    ``workers``, and each chunk is seeded by its global index. With a fixed
    ``seed`` the estimate is therefore identical for every worker count and on
    every CPU topology; ``workers`` only controls how many run in parallel.
    """
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
    chunk_count = min(_PARALLEL_CHUNKS, int(n_total))
    width = (b - a) / chunk_count
    quotient, remainder = divmod(int(n_total), chunk_count)
    counts = [quotient + (1 if index < remainder else 0) for index in range(chunk_count)]
    if seed is None:
        seeds: list[int | None] = [None] * chunk_count
    else:
        children = np.random.SeedSequence(seed).spawn(chunk_count)
        seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    jobs = [
        (func, a + index * width, a + (index + 1) * width, counts[index], seeds[index])
        for index in range(chunk_count)
    ]
    pool_workers = min(worker_count, chunk_count)
    try:
        with ProcessPoolExecutor(max_workers=pool_workers) as pool:  # pragma: no cover
            estimates = list(pool.map(_integrate_chunk, jobs))
    except Exception:  # pragma: no cover - process/pickling fallback
        estimates = [_integrate_chunk(job) for job in jobs]
    return float(np.sum(estimates))


def _chain_matrix(chains: object) -> FloatArray:
    """Validate MCMC draws as an ``(m, n)`` matrix of ``m`` chains of length ``n``.

    A single 1-D sequence is treated as one chain. Ragged sequences, non-finite
    draws and chains shorter than four draws are rejected: the split-R-hat and
    ESS estimators below need at least two draws per half chain.
    """
    try:
        arr = np.asarray(chains, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "chains must be a 2-D array or a sequence of equal-length 1-D chains"
        ) from exc
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("chains must be a 2-D array or a sequence of 1-D chains")
    m, n = arr.shape
    if m < 1 or n < 4:
        raise ValueError("each chain needs at least four draws")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("chains must contain only finite draws")
    return arr


def split_rhat(chains: object) -> float:
    """Split-R-hat convergence diagnostic (Gelman et al., BDA3, ch. 11).

    Each chain is split in half (an odd final draw is dropped), then the
    potential scale reduction factor ``sqrt(var_plus / W)`` is computed from
    the between-half-chain variance ``B`` and the mean within-half-chain
    variance ``W``. Values near 1 indicate the chains mix to the same
    distribution; values above ~1.01 flag non-convergence. Splitting also
    detects non-stationarity within a single chain (e.g. a drifting mean).

    Args:
        chains: ``(m, n)`` array-like of draws, or a sequence of ``m``
            equal-length 1-D chains. At least two chains of four draws each.

    Raises:
        ValueError: fewer than two chains, invalid shape, non-finite draws,
            or zero within-chain variance (constant chains).
    """
    arr = _chain_matrix(chains)
    m, n = arr.shape
    if m < 2:
        raise ValueError("split_rhat needs at least two chains")
    if n % 2:
        arr = arr[:, :-1]
        n -= 1
    half = n // 2
    split = np.concatenate((arr[:, :half], arr[:, half:]), axis=0)
    chain_means = split.mean(axis=1)
    within = float(split.var(axis=1, ddof=1).mean())
    if within == 0.0:
        raise ValueError("split_rhat is undefined for chains with zero within-chain variance")
    between = half * float(chain_means.var(ddof=1))
    var_plus = (half - 1) / half * within + between / half
    return float(np.sqrt(var_plus / within))


def effective_sample_size(chains: object) -> float:
    """Bulk effective sample size (ESS) of MCMC draws (Gelman et al., BDA3).

    Autocorrelation inflates the variance of MCMC averages: ``n`` correlated
    draws carry the information of fewer independent ones. This estimates
    ``m * n / tau`` with the integrated autocorrelation time ``tau`` built
    from the initial monotone positive sequence of paired autocorrelations
    (Geyer's truncation), using the combined within/between-chain variance
    so disagreeing chains are penalised. The result is capped at the total
    draw count ``m * n`` (negatively autocorrelated chains can otherwise
    exceed it).

    Args:
        chains: ``(m, n)`` array-like of draws, a sequence of equal-length
            1-D chains, or a single 1-D chain. Each chain needs four draws.

    Raises:
        ValueError: invalid shape, non-finite draws, or constant chains
            (zero within-chain variance makes autocorrelation undefined).
    """
    arr = _chain_matrix(chains)
    m, n = arr.shape
    chain_means = arr.mean(axis=1)
    within = float(arr.var(axis=1, ddof=1).mean())
    if within == 0.0:
        raise ValueError("effective_sample_size is undefined for constant chains")
    between = n * float(chain_means.var(ddof=1)) if m > 1 else 0.0
    var_plus = (n - 1) / n * within + between / n
    centered = arr - chain_means[:, None]
    # Biased autocovariance per chain via zero-padded FFT (denominator n).
    size = 1 << (2 * n - 1).bit_length()
    freq = np.fft.rfft(centered, n=size, axis=1)
    acov = np.fft.irfft(freq * np.conj(freq), n=size, axis=1)[:, :n] / n
    mean_acov = acov.mean(axis=0)
    rho = 1.0 - (within - mean_acov) / var_plus
    rho[0] = 1.0
    total = 0.0
    monotone = float("inf")
    for lag in range(n // 2):
        pair = float(rho[2 * lag] + rho[2 * lag + 1])
        if pair <= 0.0:
            break
        monotone = min(monotone, pair)
        total += monotone
    tau = max(1.0, 2.0 * total - 1.0)
    return float(m * n / tau)
