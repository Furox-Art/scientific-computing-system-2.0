"""Design of experiments: factorial designs and space-filling sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DesignResult",
    "full_factorial",
    "fractional_factorial_2k",
    "latin_hypercube",
    "central_composite",
    "pluck_factors",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DesignResult:
    """Coded design matrix with factor names and run order."""

    matrix: FloatArray
    factor_names: tuple[str, ...]
    randomized: bool


def _validate_levels(levels: Sequence[int]) -> list[int]:
    cleaned = [int(level) for level in levels]
    if not cleaned or any(level < 2 for level in cleaned):
        msg = "every factor needs at least two levels"
        raise ValueError(msg)
    return cleaned


def _names(factor_names: Sequence[str] | None, count: int) -> tuple[str, ...]:
    if factor_names is None:
        return tuple(f"x{i + 1}" for i in range(count))
    if len(factor_names) != count:
        msg = "factor_names length must match the number of factors"
        raise ValueError(msg)
    return tuple(factor_names)


def _random_order(total: int, randomize: bool, seed: int | None) -> IntArray:
    if not randomize:
        return np.arange(total)
    return np.random.default_rng(seed).permutation(total)


def full_factorial(
    levels: Sequence[int],
    factor_names: Sequence[str] | None = None,
    randomize: bool = False,
    seed: int | None = None,
) -> DesignResult:
    """Full factorial design over ``len(levels)`` factors.

    Rows are coded runs; column 0 cycles fastest and later factors slower
    (column j has period ``prod(levels[:j+1])``).
    """
    counts = _validate_levels(levels)
    total = int(np.prod(counts))
    if total >= 1_000_000:
        msg = f"refusing to build a design with {total} runs"
        raise ValueError(msg)
    matrix = np.empty((total, len(counts)), dtype=float)
    cycle = 1
    for j, count in enumerate(counts):
        pattern = np.tile(np.repeat(np.arange(count), cycle), total // (count * cycle))
        matrix[:, j] = pattern
        cycle *= count
    names = _names(factor_names, len(counts))
    order = _random_order(total, randomize, seed)
    return DesignResult(matrix[order], names, randomize)


def fractional_factorial_2k(
    n_factors: int,
    generator: str | None = None,
    randomize: bool = False,
    seed: int | None = None,
) -> DesignResult:
    """Two-level fractional factorial via a defining-relation generator.

    ``generator`` uses capital letters - e.g. ``"D=ABC"`` builds the classic
    2^(4-1) half fraction where column D equals the A*B*C interaction.
    Without a generator the full 2^k design is returned.
    """
    if n_factors < 2 or n_factors > 26:
        msg = "n_factors must lie between 2 and 26"
        raise ValueError(msg)
    if generator is None:
        runs = _base_two_level(n_factors).T
        names = tuple(chr(ord("A") + i) for i in range(n_factors))
        order = _random_order(runs.shape[0], randomize, seed)
        return DesignResult(runs[order], names, randomize)

    parts = generator.split("=")
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        msg = 'generator must look like "D=ABC"'
        raise ValueError(msg)
    new_letter = parts[0].strip()
    interaction_letters = parts[1].strip().replace("+", "")
    if len(new_letter) != 1 or not new_letter.isalpha():
        msg = 'generator must look like "D=ABC"'
        raise ValueError(msg)
    new_index = ord(new_letter) - ord("A")
    interaction_indices = [ord(ch) - ord("A") for ch in interaction_letters]
    if (
        new_index != n_factors - 1
        or any(not ch.isalpha() for ch in interaction_letters)
        or any(index < 0 or index >= n_factors - 1 for index in interaction_indices)
    ):
        msg = "generator references unknown factors"
        raise ValueError(msg)
    base = _base_two_level(n_factors - 1)
    extra = np.prod(base[interaction_indices], axis=0)
    runs = np.vstack([base, extra[None, :]]).T
    names = tuple(chr(ord("A") + i) for i in range(n_factors))
    order = _random_order(runs.shape[0], randomize, seed)
    return DesignResult(runs[order], names, randomize)


def _base_two_level(n_factors: int) -> FloatArray:
    runs = 2**n_factors
    matrix = np.empty((n_factors, runs))
    for j in range(n_factors):
        block = 2**j
        column = np.repeat([-1.0, 1.0], runs // (2 * block))
        matrix[j] = np.tile(column, block)
    return matrix


def latin_hypercube(
    n_samples: int,
    n_dimensions: int,
    seed: int | None = None,
    centered: bool = False,
) -> FloatArray:
    """Latin hypercube sample in the unit hypercube [0, 1)^d.

    Each dimension is stratified into ``n_samples`` equally probable bins
    holding exactly one draw - far better coverage than plain Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    if n_samples < 2 or n_dimensions < 1:
        msg = "need at least two samples in at least one dimension"
        raise ValueError(msg)
    bins = np.arange(n_samples, dtype=float)
    samples = np.empty((n_samples, n_dimensions))
    for d in range(n_dimensions):
        permuted = rng.permutation(bins)
        inside = np.full(n_samples, 0.5) if centered else rng.random(n_samples)
        samples[:, d] = (permuted + inside) / n_samples
    return samples


def central_composite(
    n_factors: int, alpha: float | None = None, face_centered: bool = False
) -> DesignResult:
    """Face-centred or rotatable central composite design on k factors.

    Combines the two-level factorial cube, axial (star) points at +/-alpha
    and centre points; alpha = (2^k)^(1/4) gives a rotatable design.
    """
    if not 2 <= n_factors <= 8:
        msg = "n_factors must lie between 2 and 8"
        raise ValueError(msg)
    alpha_value = (
        1.0 if face_centered else (alpha if alpha is not None else float((2**n_factors) ** 0.25))
    )

    cube = _base_two_level(n_factors).T
    star_rows: list[list[float]] = []
    for axis in range(n_factors):
        for sign in (-alpha_value, alpha_value):
            row = [0.0] * n_factors
            row[axis] = sign
            star_rows.append(row)
    centre = np.zeros((2 * n_factors, n_factors))
    runs = np.vstack([cube, np.asarray(star_rows), centre])
    names = tuple(chr(ord("A") + i) for i in range(n_factors))
    return DesignResult(runs, names, False)


def pluck_factors(
    design: Sequence[Sequence[float]] | DesignResult,
    low: Sequence[float],
    high: Sequence[float],
) -> FloatArray:
    """Map coded design levels onto physical factor ranges.

    Each design column is rescaled from its own [min, max] span onto
    [low_j, high_j], so any coding convention (-1/1, 0/1, axial points)
    maps sensibly onto the real factor ranges.
    """
    matrix = design.matrix if isinstance(design, DesignResult) else np.asarray(design, dtype=float)
    lows = np.asarray(low, dtype=float)
    highs = np.asarray(high, dtype=float)
    if lows.shape != highs.shape or lows.shape != (matrix.shape[1],):
        msg = "low/high ranges must match the design width"
        raise ValueError(msg)
    col_min = np.asarray(matrix.min(axis=0), dtype=float)
    col_max = np.asarray(matrix.max(axis=0), dtype=float)
    span = np.where(col_max > col_min, col_max - col_min, 1.0)
    normalized = (matrix - col_min) / span
    return np.asarray(lows + normalized * (highs - lows), dtype=float)
