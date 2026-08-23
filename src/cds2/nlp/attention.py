"""Attention mechanisms: scaled dot-product and multi-head attention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AttentionWeights",
    "softmax",
    "scaled_dot_product_attention",
    "multi_head_attention",
]

FloatArray = NDArray[np.float64]


def softmax(values: NDArray[np.float64], axis: int = -1) -> FloatArray:
    """Numerically stable softmax along ``axis``."""
    shifted = values - values.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return np.asarray(exponentials / exponentials.sum(axis=axis, keepdims=True))


@dataclass(frozen=True)
class AttentionWeights:
    """Attention output plus the weight matrix for inspection."""

    output: FloatArray
    weights: FloatArray


def scaled_dot_product_attention(
    queries: FloatArray,
    keys: FloatArray,
    values: FloatArray,
    mask: FloatArray | None = None,
) -> AttentionWeights:
    """Classic scaled dot-product attention over a batch or 2-D input."""
    dimension = queries.shape[-1]
    scores = queries @ np.swapaxes(keys, -1, -2) / np.sqrt(dimension)
    if mask is not None:
        scores = np.where(mask.astype(bool), scores, -np.inf)
    weights = softmax(scores, axis=-1)
    output = weights @ values
    return AttentionWeights(
        output=np.asarray(output, dtype=float), weights=np.asarray(weights, dtype=float)
    )


def multi_head_attention(
    queries: FloatArray,
    keys: FloatArray,
    values: FloatArray,
    heads: int,
    mask: FloatArray | None = None,
) -> AttentionWeights:
    """Split the last dimension into ``heads`` and attend independently."""
    if queries.shape[-1] % heads != 0:
        msg = "embedding dimension must be divisible by the head count"
        raise ValueError(msg)
    batch_shape = queries.shape[:-1]
    head_dim = queries.shape[-1] // heads

    def split(array: FloatArray) -> FloatArray:
        reshaped = array.reshape(*batch_shape, heads, head_dim)
        return np.swapaxes(reshaped, -2, -3)

    per_head = scaled_dot_product_attention(split(queries), split(keys), split(values), mask)
    merged = np.swapaxes(per_head.output, -2, -3).reshape(*batch_shape, heads * head_dim)
    return AttentionWeights(output=merged, weights=per_head.weights)
