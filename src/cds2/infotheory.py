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


def _log_base(base: float) -> float:
    if not math.isfinite(base) or base <= 0 or math.isclose(base, 1.0):
        msg = "base must be finite, positive and different from 1"
        raise ValueError(msg)
    return math.log(base)


def _as_joint_distribution(joint: Sequence[Sequence[float]] | FloatArray) -> FloatArray:
    matrix = np.asarray(joint, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        msg = "joint distribution must be a non-empty 2-D table"
        raise ValueError(msg)
    _as_probabilities(matrix.ravel())
    return matrix


def _entropy(probabilities: FloatArray, logarithm: float) -> float:
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log(positive)).sum() / logarithm)


def entropy(probabilities: Sequence[float] | FloatArray, base: float = 2.0) -> float:
    """Shannon entropy of a discrete distribution in units of ``base``."""
    return _entropy(_as_probabilities(probabilities), _log_base(base))


def joint_entropy(joint: Sequence[Sequence[float]] | FloatArray, base: float = 2.0) -> float:
    """Entropy of a joint distribution given as a 2-D probability table."""
    matrix = _as_joint_distribution(joint)
    return _entropy(matrix.ravel(), _log_base(base))


def conditional_entropy(
    joint: Sequence[Sequence[float]] | FloatArray,
    marginal: Sequence[float] | FloatArray | None = None,
    base: float = 2.0,
) -> float:
    """H(Y|X) from a joint P(X, Y) table (rows indexed by X)."""
    matrix = _as_joint_distribution(joint)
    logarithm = _log_base(base)
    row_sums = matrix.sum(axis=1)
    if marginal is not None:
        marginal_values = np.asarray(marginal, dtype=float)
        if marginal_values.shape != (matrix.shape[0],):
            msg = "marginal shape does not match the joint table rows"
            raise ValueError(msg)
        if not np.allclose(_as_probabilities(marginal_values), row_sums, rtol=1e-9, atol=1e-9):
            msg = "marginal probabilities must match the joint table row sums"
            raise ValueError(msg)
    total = float(row_sums.sum())
    return math.fsum(
        float(p_x) / total * _entropy(row / p_x, logarithm)
        for row, p_x in zip(matrix, row_sums, strict=True)
        if p_x > 0
    )


def _as_probability_pair(
    p: Sequence[float] | FloatArray, q: Sequence[float] | FloatArray
) -> tuple[FloatArray, FloatArray]:
    p_array = _as_probabilities(p)
    q_array = _as_probabilities(q)
    if p_array.shape != q_array.shape:
        msg = "p and q must have the same length"
        raise ValueError(msg)
    return p_array, q_array


def kl_divergence(
    p: Sequence[float] | FloatArray, q: Sequence[float] | FloatArray, base: float = 2.0
) -> float:
    """Kullback-Leibler divergence D_KL(P || Q), with zero terms where P is 0."""
    p_array, q_array = _as_probability_pair(p, q)
    logarithm = _log_base(base)
    support = p_array > 0
    if np.any(q_array[support] == 0):
        return math.inf / logarithm
    ratio_terms = p_array[support] * (np.log(p_array[support]) - np.log(q_array[support]))
    return float(ratio_terms.sum() / logarithm)


def js_divergence(
    p: Sequence[float] | FloatArray, q: Sequence[float] | FloatArray, base: float = 2.0
) -> float:
    """Jensen-Shannon divergence between P and Q (symmetric, bounded)."""
    p_array, q_array = _as_probability_pair(p, q)
    logarithm = _log_base(base)
    combined = p_array + q_array
    divergence = 0.0
    for probabilities in (p_array, q_array):
        support = probabilities > 0
        positive = probabilities[support]
        # Use 2P/(P+Q) to keep the smallest positive probabilities in support.
        divergence += float((positive * np.log(2.0 * positive / combined[support])).sum())
    return 0.5 * divergence / logarithm


def cross_entropy(
    p: Sequence[float] | FloatArray, q: Sequence[float] | FloatArray, base: float = 2.0
) -> float:
    """Cross-entropy H(P, Q) = H(P) + D_KL(P || Q)."""
    p_array, q_array = _as_probability_pair(p, q)
    logarithm = _log_base(base)
    support = p_array > 0
    if np.any(q_array[support] == 0):
        return math.inf / logarithm
    terms = -p_array[support] * np.log(q_array[support])
    return float(terms.sum() / logarithm)


def _mutual_information(matrix: FloatArray, logarithm: float) -> float:
    rows, cols = np.nonzero(matrix)
    positive = matrix[rows, cols]
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    log_ratio = np.log(positive) - np.log(row_sums[rows]) - np.log(col_sums[cols])
    return float((positive * log_ratio).sum() / logarithm)


def mutual_information(joint: Sequence[Sequence[float]] | FloatArray, base: float = 2.0) -> float:
    """Mutual information I(X; Y) from a joint P(X, Y) table."""
    return _mutual_information(_as_joint_distribution(joint), _log_base(base))


def normalized_mutual_information(
    joint: Sequence[Sequence[float]] | FloatArray, base: float = 2.0
) -> float:
    """I(X; Y) scaled into [0, 1] by the square-root entropy product."""
    matrix = _as_joint_distribution(joint)
    _log_base(base)
    # The logarithm base cancels in the normalized result.
    h_x = _entropy(matrix.sum(axis=1), 1.0)
    h_y = _entropy(matrix.sum(axis=0), 1.0)
    denominator = math.sqrt(abs(h_x)) * math.sqrt(abs(h_y))
    if denominator == 0:
        return 0.0
    return _mutual_information(matrix, 1.0) / denominator


def permutation_entropy(
    signal: Sequence[float] | FloatArray, order: int = 3, delay: int = 1, base: float = 2.0
) -> float:
    """Bandt-Pompe permutation entropy of an ordinal-pattern histogram."""
    series = np.asarray(signal, dtype=float)
    if series.ndim != 1 or not np.all(np.isfinite(series)):
        msg = "signal must be a finite 1-D sequence"
        raise ValueError(msg)
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
    windows = np.lib.stride_tricks.sliding_window_view(series, needed)
    embedded = windows[:, ::delay]
    codes = np.argsort(embedded, axis=1, kind="stable")
    _, counts = np.unique(codes, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities, base)
