"""Spatial statistics: contiguity weights, autocorrelation and point patterns."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import spatial

__all__ = [
    "MoranResult",
    "GearyResult",
    "NearestNeighborIndex",
    "build_weight_matrix",
    "morans_i",
    "gearys_c",
    "nearest_neighbor_index",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class MoranResult:
    """Outcome of the global Moran's I autocorrelation test."""

    index: float
    expected: float
    z_score: float


@dataclass(frozen=True)
class GearyResult:
    """Outcome of the global Geary's C autocorrelation test."""

    c: float
    z_score: float
    expected: float = 1.0


@dataclass(frozen=True)
class NearestNeighborIndex:
    """Mean nearest-neighbour distance compared with complete spatial randomness."""

    observed_mean: float
    expected_mean: float
    ratio: float
    pattern: str


def _as_points(points: Sequence[Sequence[float]] | FloatArray) -> FloatArray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 3:
        msg = "points must be a non-empty (n, 2) array with at least three rows"
        raise ValueError(msg)
    return array


def build_weight_matrix(
    points: Sequence[Sequence[float]] | FloatArray,
    cutoff: float | None = None,
) -> FloatArray:
    """Row-standardised binary contiguity weights linking points within ``cutoff``.

    ``cutoff`` defaults to the mean nearest-neighbour distance; isolated nodes
    receive an all-zero row and the diagonal stays zero.
    """
    pts = _as_points(points)
    tree = spatial.cKDTree(pts)
    query = tree.query(pts, k=2)
    nearest = np.asarray(query[0], dtype=float)[:, 1]
    limit = float(nearest.mean()) if cutoff is None else float(cutoff)
    if limit <= 0.0:
        msg = "cutoff must be positive"
        raise ValueError(msg)
    count = pts.shape[0]
    differences = pts[:, None, :] - pts[None, :, :]
    distances = np.sqrt((differences**2).sum(axis=-1))
    adjacency = (distances <= limit) & ~np.eye(count, dtype=bool)
    neighbour_counts = adjacency.sum(axis=1)
    weights = np.zeros((count, count), dtype=float)
    connected = neighbour_counts > 0
    weights[connected] = adjacency[connected] / neighbour_counts[connected, None]
    return weights


def _validated_inputs(
    values: Sequence[float] | FloatArray,
    weight_matrix: FloatArray,
) -> tuple[FloatArray, FloatArray, int]:
    observed = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weight_matrix, dtype=float)
    n = int(weights.shape[0])
    if weights.ndim != 2 or weights.shape[1] != n or observed.size != n:
        msg = "values and weight matrix sizes differ"
        raise ValueError(msg)
    if n < 3:
        msg = "weight matrix needs at least three rows"
        raise ValueError(msg)
    if float(weights.sum()) <= 0.0:
        msg = "weight matrix has no links"
        raise ValueError(msg)
    deviations = observed - observed.mean()
    if float(deviations @ deviations) == 0.0:
        msg = "values must not be constant"
        raise ValueError(msg)
    return deviations, weights, n


def morans_i(values: Sequence[float] | FloatArray, weight_matrix: FloatArray) -> MoranResult:
    """Global Moran's I; the z score uses the simplified randomisation variance 1/n."""
    deviations, weights, n = _validated_inputs(values, weight_matrix)
    s0 = float(weights.sum())
    numerator = float(deviations @ (weights @ deviations))
    denominator = float(deviations @ deviations)
    index = (n / s0) * (numerator / denominator)
    expected = -1.0 / (n - 1)
    z_score = (index - expected) / math.sqrt(1.0 / n)
    return MoranResult(index=index, expected=expected, z_score=z_score)


def gearys_c(values: Sequence[float] | FloatArray, weight_matrix: FloatArray) -> GearyResult:
    """Global Geary's C; the z score uses the simplified randomisation variance 1/(2n)."""
    deviations, weights, n = _validated_inputs(values, weight_matrix)
    s0 = float(weights.sum())
    pairwise_squared = (deviations[:, None] - deviations[None, :]) ** 2
    neighbour_sum = float((weights * pairwise_squared).sum())
    denominator = float(deviations @ deviations)
    c_value = ((n - 1) / (2.0 * s0)) * (neighbour_sum / denominator)
    z_score = (c_value - 1.0) / math.sqrt(1.0 / (2 * n))
    return GearyResult(c=c_value, z_score=z_score, expected=1.0)


def nearest_neighbor_index(
    points: Sequence[Sequence[float]] | FloatArray,
    area: float | None = None,
) -> NearestNeighborIndex:
    """Clark-Evans nearest-neighbour index against complete spatial randomness.

    ``area`` defaults to the bounding-box area; a ratio below 0.95 flags a
    clustered pattern and above 1.05 a dispersed one.
    """
    pts = _as_points(points)
    count = pts.shape[0]
    if area is None:
        spans = pts.max(axis=0) - pts.min(axis=0)
        study_area = float(spans[0] * spans[1])
    else:
        study_area = float(area)
    if study_area <= 0.0:
        msg = "area must be positive"
        raise ValueError(msg)
    tree = spatial.cKDTree(pts)
    query = tree.query(pts, k=2)
    nearest = np.asarray(query[0], dtype=float)[:, 1]
    observed_mean = float(nearest.mean())
    expected_mean = float(0.5 / np.sqrt(count / study_area))
    ratio = observed_mean / expected_mean
    if ratio < 0.95:
        pattern = "clustered"
    elif ratio > 1.05:
        pattern = "dispersed"
    else:
        pattern = "random"
    return NearestNeighborIndex(
        observed_mean=observed_mean,
        expected_mean=expected_mean,
        ratio=ratio,
        pattern=pattern,
    )
