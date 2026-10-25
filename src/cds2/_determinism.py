"""Helpers that keep seeded numerical routines reproducible."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def start_vector(n: int, seed: int | None) -> FloatArray | None:
    """ARPACK start vector derived from ``seed``; ``None`` keeps scipy's entropy."""
    if seed is None:
        return None
    return np.asarray(np.random.default_rng(seed).standard_normal(n), dtype=np.float64)


def canonicalize_signs(vectors: FloatArray) -> FloatArray:
    """Flip each eigenvector so its largest-magnitude component is positive."""
    result = np.array(vectors, dtype=np.float64, copy=True)
    for column in range(result.shape[1]):
        vector = result[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            result[:, column] = -vector
    return result
