"""Nonlinear dynamics: attractor reconstruction, Lyapunov exponents, fractal dimensions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EmbeddingResult",
    "LyapunovResult",
    "CorrDimResult",
    "delay_embed",
    "false_nearest_fraction",
    "largest_lyapunov_exponent",
    "correlation_dimension",
    "sample_entropy",
    "hurst_exponent",
    "logistic_map",
    "bifurcation_scan",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class EmbeddingResult:
    """Time-delay embedding of a scalar series."""

    embedded: FloatArray
    dimension: int
    delay: int


@dataclass(frozen=True)
class LyapunovResult:
    """Largest Lyapunov exponent with its fitting diagnostics."""

    exponent: float
    mean_log_divergence: FloatArray
    horizons: IntArray


@dataclass(frozen=True)
class CorrDimResult:
    """Grassberger-Procaccia correlation-dimension fit."""

    dimension: float
    radii: FloatArray
    correlations: FloatArray


def delay_embed(
    signal: Sequence[float] | FloatArray, dimension: int = 3, delay: int = 1
) -> EmbeddingResult:
    """Takens time-delay embedding of a scalar series.

    Returns the matrix whose rows are ``[x(t), x(t+delay), ..., x(t+(d-1)*delay)]``.
    """
    series = np.asarray(signal, dtype=float)
    if dimension < 1:
        msg = "dimension must be at least 1"
        raise ValueError(msg)
    if delay < 1:
        msg = "delay must be at least 1"
        raise ValueError(msg)
    needed = (dimension - 1) * delay + 1
    if series.ndim != 1 or series.size < needed:
        msg = f"series needs at least {needed} samples"
        raise ValueError(msg)
    n_rows = series.size - (dimension - 1) * delay
    offsets = np.arange(dimension) * delay
    embedded = np.stack([series[index + offsets] for index in range(n_rows)], axis=0)
    return EmbeddingResult(
        embedded=np.ascontiguousarray(embedded), dimension=dimension, delay=delay
    )


def false_nearest_fraction(
    signal: Sequence[float] | FloatArray,
    dimension: int,
    delay: int = 1,
    rtol: float = 10.0,
    atol_factor: float = 2.0,
) -> float:
    """Fraction of false nearest neighbours when embedding at ``dimension``.

    Kennel's test: a neighbour is false if adding the extra coordinate moves
    it far away relative to its current distance or relative to the attractor
    spread.
    """
    current = delay_embed(signal, dimension=dimension, delay=delay).embedded
    nxt = delay_embed(signal, dimension=dimension + 1, delay=delay).embedded
    limit = min(current.shape[0], nxt.shape[0])
    if limit < 3:
        msg = "series too short for nearest-neighbour analysis"
        raise ValueError(msg)
    current = current[:limit]
    nxt = nxt[:limit]

    diff = current[:, None, :] - current[None, :, :]
    distances = np.sqrt((diff**2).sum(axis=-1))
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argmin(distances, axis=1)
    base_distance = distances[np.arange(limit), neighbours]

    extra = np.abs(nxt[:, -1] - nxt[neighbours, -1])
    sigma = float(np.std(np.asarray(signal, dtype=float)))
    false_mask = (extra / np.maximum(base_distance, 1e-12) > rtol) | (
        extra > atol_factor * sigma * np.sqrt(dimension + 1)
    )
    return float(false_mask.mean())


def largest_lyapunov_exponent(
    signal: Sequence[float] | FloatArray,
    embedding_dimension: int = 3,
    delay: int = 1,
    max_horizon: int | None = None,
    fit_points: int = 8,
) -> LyapunovResult:
    """Rosenstein largest-Lyapunov-exponent estimate.

    Tracks the mean logarithmic divergence of each point's nearest neighbour
    over prediction horizons; ``exponent`` is the slope of the linear growth
    region (first ``fit_points`` finite horizons).
    """
    embedded = delay_embed(signal, dimension=embedding_dimension, delay=delay).embedded
    n = embedded.shape[0]
    horizon_limit = min(max_horizon or fit_points + 4, n - 1)
    if horizon_limit < 1:
        msg = "series too short for divergence tracking"
        raise ValueError(msg)

    diff = embedded[:, None, :] - embedded[None, :, :]
    distances = np.sqrt((diff**2).sum(axis=-1))
    usable = n - horizon_limit
    if usable < 2:
        msg = "series too short for divergence tracking"
        raise ValueError(msg)
    # Base points and their neighbours must both leave room to evolve
    # ``horizon_limit`` steps forward without leaving the embedding.
    distances[:, usable:] = np.inf
    for offset in range(1, max(embedding_dimension * delay, 1) + 1):
        pairs = np.arange(n - offset)
        distances[pairs, pairs + offset] = np.inf
        distances[pairs + offset, pairs] = np.inf
    np.fill_diagonal(distances, np.inf)
    if not np.isfinite(distances[:usable]).any():
        msg = "no admissible nearest neighbours within the horizon window"
        raise ValueError(msg)
    neighbours = np.argmin(distances, axis=1)

    log_divergence = np.full(horizon_limit, np.nan)
    for horizon in range(horizon_limit):
        diverged = np.abs(
            embedded[neighbours[:usable] + horizon + 1] - embedded[np.arange(usable) + horizon + 1]
        )
        positive = diverged > 0
        if np.any(positive):
            log_divergence[horizon] = float(np.log(diverged[positive]).mean())

    horizons = np.arange(horizon_limit)
    finite = np.isfinite(log_divergence)
    count = min(fit_points, int(finite.sum()))
    if count < 2:
        exponent_value = 0.0
    else:
        chosen = horizons[finite][:count]
        values = log_divergence[finite][:count]
        exponent_value = float(np.polyfit(chosen, values, 1)[0])
    return LyapunovResult(
        exponent=exponent_value,
        mean_log_divergence=log_divergence,
        horizons=horizons.astype(np.int64),
    )


def correlation_dimension(
    signal: Sequence[float] | FloatArray,
    embedding_dimension: int = 3,
    delay: int = 1,
    radii: Sequence[float] | None = None,
    sample_limit: int = 400,
) -> CorrDimResult:
    """Grassberger-Procaccia correlation-dimension estimate.

    Computes the correlation sum C(r) on a (sub-sampled) delay embedding and
    returns the slope of log C(r) versus log r over the supplied or generated
    radius grid.
    """
    embedded = delay_embed(signal, dimension=embedding_dimension, delay=delay).embedded
    n = embedded.shape[0]
    if n > sample_limit:
        step = int(np.ceil(n / sample_limit))
        embedded = np.ascontiguousarray(embedded[::step])
        n = embedded.shape[0]
    if radii is None:
        pairwise = np.sqrt(((embedded[:, None, :] - embedded[None, :, :]) ** 2).sum(-1))
        upper = pairwise[np.triu_indices(n, k=1)]
        positive = upper[upper > 0]
        if positive.size == 0:
            msg = "degenerate trajectory: no positive pairwise distances"
            raise ValueError(msg)
        low = max(float(np.percentile(positive, 5)), 1e-9)
        high = max(float(np.percentile(positive, 95)), low * 10)
        grid = np.geomspace(low, high, num=12)
    else:
        grid_values = sorted({float(r) for r in radii if r > 0})
        grid = np.asarray(grid_values, dtype=float)
        if grid.size < 2:
            msg = "need at least two distinct positive radii"
            raise ValueError(msg)

    correlations = np.empty(grid.size)
    for i, radius in enumerate(grid):
        hits = 0
        total = 0
        chunk = max(64, int(np.ceil(np.sqrt(n))))
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            block = embedded[start:stop]
            local = np.sqrt(((block[:, None, :] - embedded[None, :, :]) ** 2).sum(-1))
            # Exclude each point's distance to itself (block rows map to
            # absolute indices start:stop, so the self-pairs sit there).
            local[np.arange(stop - start), np.arange(start, stop)] = np.inf
            hits += int((local <= radius).sum())
            total += local.size
        correlations[i] = hits / total

    usable = correlations > 0
    if int(usable.sum()) < 2:
        msg = "radii grid too small for the trajectory scale"
        raise ValueError(msg)
    slope = np.polyfit(np.log(grid[usable]), np.log(correlations[usable]), 1)[0]
    return CorrDimResult(dimension=float(slope), radii=grid, correlations=correlations)


def sample_entropy(
    signal: Sequence[float] | FloatArray, dimension: int = 2, tolerance: float | None = None
) -> float:
    """Sample entropy of a scalar series (Pincus).

    Defaults to the standard tolerance of ``0.2 * std``; higher values mean
    more complexity, 0 means perfectly predictable.
    """
    series = np.asarray(signal, dtype=float)
    if series.ndim != 1 or series.size < dimension + 2:
        msg = "series too short for sample entropy"
        raise ValueError(msg)
    if tolerance is None:
        tolerance = 0.2 * float(series.std())
    if tolerance <= 0:
        msg = "tolerance must be positive"
        raise ValueError(msg)

    def matches(pattern_length: int) -> int:
        windows = np.lib.stride_tricks.sliding_window_view(series, pattern_length)
        total = 0
        for i in range(windows.shape[0]):
            distances = np.max(np.abs(windows[i + 1 :] - windows[i]), axis=1)
            total += int((distances <= tolerance).sum())
        return total

    numerator = matches(dimension + 1)
    denominator = matches(dimension)
    if denominator == 0 or numerator == 0:
        return 0.0
    return float(-np.log(numerator / denominator))


def hurst_exponent(signal: Sequence[float] | FloatArray, min_window: int = 8) -> float:
    """Rescaled-range (R/S) Hurst exponent estimate for a scalar series."""
    series = np.asarray(signal, dtype=float)
    if series.ndim != 1 or series.size < min_window * 2:
        msg = "series too short for an R/S analysis"
        raise ValueError(msg)
    sizes: list[int] = []
    estimates: list[float] = []
    size = series.size // 2
    while size >= min_window:
        chunks = series.size // size
        rs_values: list[float] = []
        for c in range(chunks):
            segment = series[c * size : (c + 1) * size]
            walk = np.cumsum(segment - segment.mean())
            spread = float(walk.max() - walk.min())
            deviation = float(segment.std())
            if deviation > 0 and spread > 0:
                rs_values.append(spread / deviation)
        if rs_values:
            sizes.append(size)
            estimates.append(float(np.mean(rs_values)))
        size //= 2
    if len(sizes) < 2:
        msg = "not enough window sizes for a regression"
        raise ValueError(msg)
    slope = np.polyfit(np.log(np.asarray(sizes, dtype=float)), np.log(np.asarray(estimates)), 1)[0]
    return float(slope)


def logistic_map(
    r: float, length: int = 200, x0: float = 0.4, discard: int = 100, seed: int | None = None
) -> FloatArray:
    """Iterate the logistic map ``x_{n+1} = r * x_n * (1 - x_n)``.

    The first ``discard`` transient iterates are dropped; a tiny amount of
    noise keeps trajectories off unstable fixed points.
    """
    if not 0.0 <= r <= 4.0:
        msg = "r must lie in [0, 4]"
        raise ValueError(msg)
    if length < 1 or discard < 0:
        msg = "need length >= 1 and discard >= 0"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    state = float(x0) % 1.0
    out = np.empty(length)
    for i in range(discard + length):
        state = r * state * (1.0 - state)
        state += rng.normal(scale=1e-9)
        state = min(max(state, 0.0), 1.0)
        if i >= discard:
            out[i - discard] = state
    return out


def bifurcation_scan(
    map_function: Callable[[float, float], float],
    parameter_values: Sequence[float],
    iterations: int = 300,
    last_values: int = 40,
    x0: float = 0.4,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Generic bifurcation scan of a one-parameter one-dimensional map.

    Returns flat arrays ``(parameters, states)`` holding the asymptotic states
    of each parameter value - ready to scatter-plot.
    """
    if iterations < last_values + 1:
        msg = "iterations must exceed last_values"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    params_out: list[float] = []
    states_out: list[float] = []
    for parameter in parameter_values:
        state = float(x0)
        for i in range(iterations):
            state = map_function(parameter, state)
            state += rng.normal(scale=1e-9)
            if i >= iterations - last_values:
                params_out.append(float(parameter))
                states_out.append(state)
    return np.asarray(params_out), np.asarray(states_out)
