"""Numerical differentiation: derivatives, Jacobians, Hessians, uncertainty."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ErrorPropagationResult",
    "derivative",
    "complex_step_gradient",
    "jacobian",
    "hessian",
    "propagate_error",
]


@dataclass(frozen=True)
class ErrorPropagationResult:
    """First-order propagated output mean and covariance."""

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]


def propagate_error(
    f: Callable[[np.ndarray], object],
    x: object,
    cov: object,
    step: float | None = None,
) -> ErrorPropagationResult:
    """Linearized (first-order) uncertainty propagation via the Jacobian.

    For ``f: R^n -> R^m`` with input covariance ``cov`` (n x n), returns the
    output mean ``f(x)`` and covariance ``J cov J^T``.
    """
    point = np.atleast_1d(np.asarray(x, dtype=float))
    jacobian_matrix = jacobian(f, point, step=step)
    input_cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if input_cov.shape[0] != point.size:
        msg = "cov must be an n x n matrix matching x"
        raise ValueError(msg)
    output_mean = np.atleast_1d(np.asarray(f(point.copy()), dtype=float))
    output_cov = jacobian_matrix @ input_cov @ jacobian_matrix.T
    return ErrorPropagationResult(mean=output_mean, covariance=output_cov)


_CUBRT_EPS = float(np.cbrt(np.finfo(float).eps))


def _adaptive_step(x_value: float, order: int = 1) -> float:
    """Step size scaled to machine precision for an ``order``-th operator."""
    exponent = 1.0 / (2.0 + order)
    return (
        _CUBRT_EPS**order * max(1.0, abs(x_value))
        if order == 1
        else float(np.finfo(float).eps) ** exponent * max(1.0, abs(x_value))
    )


def derivative(
    f: Callable[[float], float],
    x: float,
    method: str = "central",
    step: float | None = None,
) -> float:
    """First derivative of a scalar function at ``x``.

    Methods: ``central`` (default, second-order accurate), ``forward``,
    ``backward``.
    """
    x_value = float(x)
    h = step if step is not None else _adaptive_step(x_value)
    if method == "central":
        return float((f(x_value + h) - f(x_value - h)) / (2.0 * h))
    if method == "forward":
        return float((f(x_value + h) - f(x_value)) / h)
    if method == "backward":
        return float((f(x_value) - f(x_value - h)) / h)
    msg = f"unknown differentiation method: {method!r}"
    raise ValueError(msg)


def complex_step_gradient(f: Callable[..., object], x: object) -> np.ndarray:
    """Gradient via the complex-step trick - accurate to machine precision.

    Requires ``f`` to propagate complex arithmetic (plain NumPy formulas do).
    """
    point = np.atleast_1d(np.asarray(x, dtype=float))
    h = 1e-20
    gradient = np.empty(point.size)
    for index in range(point.size):
        perturbed = point.astype(np.complex128)
        perturbed[index] += 1j * h
        value = f(perturbed)
        gradient[index] = float(np.imag(value)) / h
    return gradient


def jacobian(
    f: Callable[[np.ndarray], object],
    x: object,
    step: float | None = None,
) -> np.ndarray:
    """Finite-difference Jacobian of ``f: R^n -> R^m`` evaluated at ``x``."""
    point = np.atleast_1d(np.asarray(x, dtype=float)).copy()
    n = point.size
    base = np.atleast_1d(np.asarray(f(point.copy()), dtype=float))
    result = np.empty((base.size, n))
    for index in range(n):
        h = step if step is not None else _adaptive_step(float(abs(point[index])))
        plus = point.copy()
        minus = point.copy()
        plus[index] += h
        minus[index] -= h
        f_plus = np.atleast_1d(np.asarray(f(plus), dtype=float))
        f_minus = np.atleast_1d(np.asarray(f(minus), dtype=float))
        result[:, index] = (f_plus - f_minus) / (2.0 * h)
    return result


def hessian(
    f: Callable[[np.ndarray], float],
    x: object,
    step: float | None = None,
) -> np.ndarray:
    """Central-difference Hessian of a scalar field ``f: R^n -> R``."""
    point = np.atleast_1d(np.asarray(x, dtype=float))
    n = point.size
    steps = np.full(n, step if step is not None else float(np.finfo(float).eps) ** (1.0 / 4.0))
    f0 = float(f(point.copy()))
    diagonal = np.empty(n)
    forward = np.empty((n, n))
    backward = np.empty((n, n))
    for i in range(n):
        hi = steps[i]
        p_plus = point.copy()
        p_minus = point.copy()
        p_plus[i] += hi
        p_minus[i] -= hi
        forward[i] = p_plus
        backward[i] = p_minus
        diagonal[i] = (float(f(p_plus)) - 2.0 * f0 + float(f(p_minus))) / (hi * hi)
    result = np.zeros((n, n))
    for i in range(n):
        result[i, i] = diagonal[i]
    for i in range(n):
        for j in range(i + 1, n):
            hi, hj = steps[i], steps[j]
            pp = point.copy()
            pp[i] += hi
            pp[j] += hj
            pm = point.copy()
            pm[i] += hi
            pm[j] -= hj
            mp = point.copy()
            mp[i] -= hi
            mp[j] += hj
            mm = point.copy()
            mm[i] -= hi
            mm[j] -= hj
            mixed = (float(f(pp)) - float(f(pm)) - float(f(mp)) + float(f(mm))) / (4.0 * hi * hj)
            result[i, j] = mixed
            result[j, i] = mixed
    return result
