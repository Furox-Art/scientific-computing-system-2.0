"""Information theory: entropy, divergence and dependence measures."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "entropy",
    "joint_entropy",
    "conditional_entropy",
    "kl_divergence",
    "js_divergence",
    "cross_entropy",
    "mutual_information",
    "normalized_mutual_information",
    "permutation_entropy",
]

FloatArray = NDArray[np.float64]


def _as_probabilities(values: Sequence[float] | FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        msg = "probabilities must be a 1-D sequence"
        raise ValueError(msg)
    if array.size == 0:
        msg = "probabilities cannot be empty"
        raise ValueError(msg)
    if np.any(array < 0) or not math.isclose(float(array.sum()), 1.0, abs_tol=1e-9):
        msg = "probabilities must be non-negative and sum to 1"
        raise ValueError(msg)
    return array


def entropy(probabilities: Sequence[float] | FloatArray, base: float = 2.0) -> float:
    """Shannon entropy of a discrete distribution in units of ``base``."""
    if base <= 0 or math.isclose(base, 1.0):
        msg = "base must be positive and different from 1"
        raise ValueError(msg)
    probs = _as_probabilities(probabilities)
    positive = probs[probs > 0]
    return float(-(positive * np.log(positive)).sum() / math.log(base))


def joint_entropy(joint: Sequence[Sequence[float]] | FloatArray, base: float = 2.0) -> float:
    """Entropy of a joint distribution given as a 2-D probability table."""
    matrix = np.asarray(joint, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        msg = "joint distribution must be a non-empty 2-D table"
        raise ValueError(msg)
    flat = _as_probabilities(matrix.ravel())
    positive = flat[flat > 0]
    return float(-(positive * np.log(positive)).sum() / math.log(base))


def conditional_entropy(
    joint: Sequence[Sequence[float]] | FloatArray,
    marginal: Sequence[float] | FloatArray | None = None,
    base: float = 2.0,
) -> float:
    """H(Y|X) from a joint P(X, Y) table (rows indexed by X)."""
    matrix = np.asarray(joint, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        msg = "joint distribution must be a non-empty 2-D table"
        raise ValueError(msg)
    row_sums = matrix.sum(axis=1)
    if marginal is None:
        marginal_values = row_sums
    else:
        marginal_values = np.asarray(marginal, dtype=float)
        if marginal_values.shape != (matrix.shape[0],):
            msg = "marginal shape does not match the joint table rows"
            raise ValueError(msg)
    total = float(row_sums.sum())
    entropy_value = 0.0
    for row, p_x in zip(matrix, marginal_values, strict=True):
        if p_x <= 0:
            continue
        conditional = row / p_x
        positive = conditional[conditional > 0]
        entropy_value += p_x / total * -(positive * np.log(positive)).sum() / math.log(base)
    return float(entropy_value)


def kl_divergence(
    p: Sequence[float] | FloatArray, q: Sequence[float] | FloatArray, base: float = 2.0
) -> float:
    """Kullback-Leibler divergence D_KL(P || Q); zero terms where P is 0."""
    p_array = _as_probabilities(p)
    q_array = _as_probabilities(q)
    if p_array.shape != q_array.shape:
        msg = "p and q must have the same length"
        raise ValueError(msg)
    support = (q_array > 0) & (p_array > 0)
    ratio_terms = p_array[support] * (np.log(p_array[support]) - np.log(q_array[support]))
    return float(ratio_terms.sum() / math.log(base))


def js_divergence(
    p: Sequence[float] | FloatArray, q: Sequence[float] | FloatArray, base: float = 2.0
) -> float:
    """Jensen-Shannon divergence between P and Q (symmetric, bounded)."""
    p_array = _as_probabilities(p)
    q_array = _as_probabilities(q)
    if p_array.shape != q_array.shape:
        msg = "p and q must have the same length"
        raise ValueError(msg)
    mixture = 0.5 * (p_array + q_array)
    return 0.5 * kl_divergence(p_array, mixture, base) + 0.5 * kl_divergence(q_array, mixture, base)


def cross_entropy(
    p: Sequence[float] | FloatArray, q: Sequence[float] | FloatArray, base: float = 2.0
) -> float:
    """Cross-entropy H(P, Q) = H(P) + D_KL(P || Q)."""
    p_array = _as_probabilities(p)
    q_array = _as_probabilities(q)
    if p_array.shape != q_array.shape:
        msg = "p and q must have the same length"
        raise ValueError(msg)
    support = (q_array > 0) & (p_array > 0)
    terms = -p_array[support] * np.log(q_array[support])
    return float(terms.sum() / math.log(base))


def mutual_information(joint: Sequence[Sequence[float]] | FloatArray, base: float = 2.0) -> float:
    """Mutual information I(X; Y) from a joint P(X, Y) table."""
    matrix = np.asarray(joint, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        msg = "joint distribution must be a non-empty 2-D table"
        raise ValueError(msg)
    total = float(matrix.sum())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        msg = "joint probabilities must sum to 1"
        raise ValueError(msg)
    row_sums = matrix.sum(axis=1, keepdims=True)
    col_sums = matrix.sum(axis=0, keepdims=True)
    independent = row_sums @ col_sums
    support = matrix > 0
    terms = matrix[support] * np.log(matrix[support] / independent[support])
    return float(terms.sum() / math.log(base))


def normalized_mutual_information(
    joint: Sequence[Sequence[float]] | FloatArray, base: float = 2.0
) -> float:
    """I(X; Y) scaled into [0, 1] by the square-root entropy product."""
    matrix = np.asarray(joint, dtype=float)
    h_x = entropy(matrix.sum(axis=1), base)
    h_y = entropy(matrix.sum(axis=0), base)
    denominator = math.sqrt(h_x * h_y)
    if denominator == 0:
        return 0.0
    return mutual_information(matrix, base) / denominator


def permutation_entropy(
    signal: Sequence[float] | FloatArray, order: int = 3, delay: int = 1, base: float = 2.0
) -> float:
    """Bandt-Pompe permutation entropy of an ordinal-pattern histogram."""
    series = np.asarray(signal, dtype=float)
    if order < 2:
        msg = "order must be at least 2"
        raise ValueError(msg)
    if delay < 1:
        msg = "delay must be at least 1"
        raise ValueError(msg)
    needed = (order - 1) * delay + 1
    if series.size < needed:
        msg = f"signal needs at least {needed} samples for order={order}, delay={delay}"
        raise ValueError(msg)
    windows = np.lib.stride_tricks.sliding_window_view(series, needed)[::delay]
    embedded = windows[:, ::delay][:, :order]
    codes = np.argsort(embedded, axis=1, kind="stable")
    weights = np.array([order**i for i in range(order)], dtype=np.int64)
    patterns = codes @ weights
    _, counts = np.unique(patterns, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities, base)
