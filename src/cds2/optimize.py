"""Optimization and root finding built on scipy.optimize."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize as spo

__all__ = [
    "OptimizationResult",
    "LinprogResult",
    "FitResult",
    "GlobalResult",
    "minimize",
    "minimize_scalar",
    "root",
    "find_root_scalar",
    "newton_root",
    "linprog",
    "least_squares",
    "curve_fit",
    "differential_evolution",
]


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of an optimization or root-finding run."""

    x: object
    fun: float | None
    success: bool
    message: str
    n_iterations: int | None


@dataclass(frozen=True)
class LinprogResult:
    """Outcome of a linear-programming solve."""

    x: np.ndarray | None
    fun: float | None
    success: bool
    message: str


@dataclass(frozen=True)
class FitResult:
    """Fitted parameters and covariance from ``curve_fit``."""

    params: np.ndarray
    covariance: np.ndarray | None


def minimize(
    fun: Callable[..., float],
    x0: Sequence[float],
    method: str = "BFGS",
    **kwargs: object,
) -> OptimizationResult:
    """Minimize an unconstrained (or method-constrained) multivariate function."""
    res = spo.minimize(fun, np.asarray(x0, dtype=float), method=method, **kwargs)
    return OptimizationResult(
        x=res.x,
        fun=float(res.fun),
        success=bool(res.success),
        message=str(getattr(res, "message", "")),
        n_iterations=getattr(res, "nit", None),
    )


def minimize_scalar(
    fun: Callable[[float], float],
    bracket: tuple[float, float] | None = None,
    bounds: tuple[float, float] | None = None,
    method: str | None = None,
) -> OptimizationResult:
    """Minimize a univariate function, optionally over a bounded interval."""
    if method is None:
        method = "bounded" if bounds is not None else "brent"
    kwargs: dict[str, object] = {}
    if bracket is not None:
        kwargs["bracket"] = bracket
    if bounds is not None:
        kwargs["bounds"] = bounds
    res = spo.minimize_scalar(fun, method=method, **kwargs)
    return OptimizationResult(
        x=float(res.x),
        fun=float(res.fun),
        success=bool(res.success),
        message=str(getattr(res, "message", "")),
        n_iterations=int(res.nit) if hasattr(res, "nit") else None,
    )


def root(
    fun: Callable[..., object],
    x0: Sequence[float],
    method: str = "hybr",
    **kwargs: object,
) -> OptimizationResult:
    """Find a root of a system of equations starting from ``x0``."""
    res = spo.root(fun, np.asarray(x0, dtype=float), method=method, **kwargs)
    return OptimizationResult(
        x=res.x,
        fun=None,
        success=bool(res.success),
        message=str(res.message),
        n_iterations=None,
    )


def find_root_scalar(f: Callable[[float], float], a: float, b: float, **kwargs: float) -> float:
    """Root of a univariate function on a sign-changing bracket via Brent's method."""
    return float(spo.brentq(f, a, b, **kwargs))


def newton_root(
    f: Callable[[float], float],
    x0: float,
    fprime: Callable[[float], float] | None = None,
    tol: float = 1.48e-08,
    maxiter: int = 50,
) -> float:
    """Newton-Raphson (or secant when ``fprime`` omitted) root of a scalar function."""
    return float(spo.newton(f, x0, fprime=fprime, tol=tol, maxiter=maxiter))


def linprog(
    c: Sequence[float],
    A_ub: Sequence[Sequence[float]] | None = None,
    b_ub: Sequence[float] | None = None,
    A_eq: Sequence[Sequence[float]] | None = None,
    b_eq: Sequence[float] | None = None,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
) -> LinprogResult:
    """Minimize ``c @ x`` subject to linear inequality/equality constraints."""
    res = spo.linprog(
        np.asarray(c, dtype=float), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds
    )
    solution = None if res.x is None else np.asarray(res.x, dtype=float)
    value = None if res.fun is None else float(res.fun)
    return LinprogResult(x=solution, fun=value, success=bool(res.success), message=str(res.message))


def least_squares(
    fun: Callable[..., object],
    x0: Sequence[float],
    **kwargs: object,
) -> OptimizationResult:
    """Solve a nonlinear least-squares problem given a residual function."""
    res = spo.least_squares(fun, np.asarray(x0, dtype=float), **kwargs)
    return OptimizationResult(
        x=res.x,
        fun=float(res.cost),
        success=bool(res.success),
        message=str(res.message),
        n_iterations=int(res.nfev),
    )


@dataclass(frozen=True)
class GlobalResult:
    """Outcome of a stochastic global optimization run."""

    x: np.ndarray
    fun: float
    success: bool
    message: str
    n_iterations: int
    n_evaluations: int


def curve_fit(
    f: Callable[..., object],
    xdata: Sequence[float],
    ydata: Sequence[float],
    p0: Sequence[float] | None = None,
) -> FitResult:
    """Fit model parameters by non-linear least squares."""
    params, covariance = spo.curve_fit(f, np.asarray(xdata), np.asarray(ydata), p0=p0)
    return FitResult(params=np.asarray(params, dtype=float), covariance=covariance)


def differential_evolution(
    fun: Callable[..., float],
    bounds: Sequence[Sequence[float]],
    maxiter: int = 1000,
    popsize: int = 15,
    seed: int | None = None,
    **kwargs: object,
) -> GlobalResult:
    """Stochastic global minimization over a box via differential evolution."""
    box = [tuple(bound) for bound in bounds]
    res = spo.differential_evolution(
        fun,
        box,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        **kwargs,
    )
    return GlobalResult(
        x=np.asarray(res.x, dtype=float),
        fun=float(res.fun),
        success=bool(res.success),
        message=str(res.message),
        n_iterations=int(res.nit),
        n_evaluations=int(res.nfev),
    )
