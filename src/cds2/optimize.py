"""Optimization — convenience re-export of scipy.optimize with normalized result types; see cds2.metaheuristics and cds2.bayesopt for CDS-native global search."""

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
    "minimize_constrained",
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
    """Non-linear least-squares fit and residual diagnostics.

    The first two fields preserve the pre-5.x construction API. Additional
    diagnostics default to ``None`` so existing manual ``FitResult``
    construction remains valid.
    """

    params: np.ndarray
    covariance: np.ndarray | None
    parameter_std: np.ndarray | None = None
    predictions: np.ndarray | None = None
    residuals: np.ndarray | None = None
    rss: float | None = None
    rmse: float | None = None
    r_squared: float | None = None
    dof: int | None = None


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
    *,
    sigma: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None = None,
    absolute_sigma: bool = False,
    bounds: tuple[Sequence[float] | float, Sequence[float] | float] = (-np.inf, np.inf),
    method: str | None = None,
    jac: Callable[..., object] | str | None = None,
) -> FitResult:
    """Fit model parameters by non-linear least squares.

    Parameters mirror the highest-value controls from :func:`scipy.optimize.curve_fit`:
    measurement uncertainty, absolute/relative uncertainty semantics, parameter
    bounds, solver selection and an optional Jacobian. The returned diagnostics
    are computed on the original (unweighted) residuals for easy model checking.
    """
    x_values = np.asarray(xdata, dtype=float)
    y_values = np.asarray(ydata, dtype=float)
    sigma_values = None if sigma is None else np.asarray(sigma, dtype=float)
    params, covariance = spo.curve_fit(
        f,
        x_values,
        y_values,
        p0=p0,
        sigma=sigma_values,
        absolute_sigma=absolute_sigma,
        bounds=bounds,
        method=method,
        jac=jac,
    )
    params_array = np.asarray(params, dtype=float)
    covariance_array = np.asarray(covariance, dtype=float)
    predictions = np.asarray(f(x_values, *params_array), dtype=float)
    residuals = y_values - predictions
    squared_residuals = residuals * residuals
    rss = float(np.sum(squared_residuals))
    rmse = float(np.sqrt(np.mean(squared_residuals)))
    centered = y_values - float(np.mean(y_values))
    total_sum_squares = float(np.sum(centered * centered))
    r_squared = None if total_sum_squares == 0.0 else float(1.0 - rss / total_sum_squares)
    with np.errstate(invalid="ignore"):
        parameter_std = np.sqrt(np.diag(covariance_array))

    return FitResult(
        params=params_array,
        covariance=covariance_array,
        parameter_std=np.asarray(parameter_std, dtype=float),
        predictions=predictions,
        residuals=np.asarray(residuals, dtype=float),
        rss=rss,
        rmse=rmse,
        r_squared=r_squared,
        dof=int(y_values.size - params_array.size),
    )


def minimize_constrained(
    fun: Callable[..., float],
    x0: Sequence[float],
    constraints: Sequence[dict[str, object]] | None = None,
    bounds: Sequence[Sequence[float | None]] | None = None,
    method: str = "SLSQP",
    jac: Callable[..., object] | None = None,
    **kwargs: object,
) -> OptimizationResult:
    """Minimize under equality/inequality constraints (SLSQP or trust-constr).

    ``constraints`` uses SciPy's dict form, e.g.
    ``{"type": "eq", "fun": lambda v: v[0] + v[1] - 1}``.
    """
    res = spo.minimize(
        fun,
        np.asarray(x0, dtype=float),
        method=method,
        bounds=bounds,
        constraints=constraints,
        jac=jac,
        **kwargs,
    )
    return OptimizationResult(
        x=res.x,
        fun=float(res.fun),
        success=bool(res.success),
        message=str(res.message),
        n_iterations=getattr(res, "nit", None),
    )


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
