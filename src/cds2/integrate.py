"""Numerical integration and ODE solving built on scipy.integrate."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import integrate as spi

__all__ = [
    "QuadResult",
    "OdeResult",
    "quad",
    "integrate_2d",
    "integrate_3d",
    "solve_ivp",
    "trapezoid",
    "cumulative_trapezoid",
    "simpson",
]


@dataclass(frozen=True)
class QuadResult:
    """Integral estimate with its absolute error bound."""

    value: float
    error: float


@dataclass(frozen=True)
class OdeResult:
    """Trajectory produced by an initial-value-problem solver."""

    t: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    t_events: tuple[np.ndarray, ...] | None = None
    y_events: tuple[np.ndarray, ...] | None = None


def quad(func: Callable[[float], float], a: float, b: float) -> QuadResult:
    """Adaptive definite integral of a scalar function over [a, b]."""
    value, error = spi.quad(func, a, b)
    return QuadResult(value=value, error=error)


def integrate_2d(
    func: Callable[[float, float], float],
    xa: float,
    xb: float,
    ya: float,
    yb: float,
) -> QuadResult:
    """Double integral of ``func(x, y)`` over the rectangle [xa, xb] x [ya, yb]."""
    value, error = spi.dblquad(lambda y, x: func(x, y), xa, xb, ya, yb)
    return QuadResult(value=value, error=error)


def integrate_3d(
    func: Callable[[float, float, float], float],
    xa: float,
    xb: float,
    ya: float,
    yb: float,
    za: float,
    zb: float,
) -> QuadResult:
    """Triple integral of ``func(x, y, z)`` over the axis-aligned box."""
    value, error = spi.tplquad(lambda z, y, x: func(x, y, z), xa, xb, ya, yb, za, zb)
    return QuadResult(value=value, error=error)


def solve_ivp(
    func: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: Sequence[float],
    t_eval: Sequence[float] | None = None,
    method: str = "RK45",
    events: Sequence[Callable[[float, np.ndarray], float]] | None = None,
    **kwargs: object,
) -> OdeResult:
    """Integrate an initial-value problem dy/dt = func(t, y).

    Backward integration works naturally when ``t_span[1] < t_span[0]``.
    Pass ``events`` (callables ``g(t, y) -> float``; zero crossings trigger
    them, set ``terminal = True`` on a callable to stop integration) to
    receive ``t_events`` / ``y_events`` on the result. For stiff problems
    use ``method="Radau"``, ``"BDF"`` or ``"LSODA"``.
    """
    res = spi.solve_ivp(
        func,
        tuple(t_span),
        np.asarray(y0, dtype=float),
        t_eval=None if t_eval is None else np.asarray(t_eval, dtype=float),
        method=method,
        events=events,
        **kwargs,
    )
    t_events = None
    y_events = None
    if events is not None:
        t_events = tuple(np.asarray(block) for block in res.t_events)
        y_events = tuple(np.asarray(block) for block in res.y_events)
    return OdeResult(
        t=np.asarray(res.t),
        y=np.asarray(res.y),
        success=bool(res.success),
        message=str(res.message),
        t_events=t_events,
        y_events=y_events,
    )


def trapezoid(y: Sequence[float], x: Sequence[float] | None = None, dx: float = 1.0) -> float:
    """Composite trapezoidal integral of sampled values."""
    return float(spi.trapezoid(np.asarray(y, dtype=float), x=x, dx=dx))


def cumulative_trapezoid(
    y: Sequence[float],
    x: Sequence[float] | None = None,
    dx: float = 1.0,
) -> np.ndarray:
    """Running trapezoidal integral, returned with the same length as ``y``."""
    values = spi.cumulative_trapezoid(np.asarray(y, dtype=float), x=x, dx=dx, initial=0.0)
    return np.asarray(values)


def simpson(y: Sequence[float], x: Sequence[float] | None = None, dx: float = 1.0) -> float:
    """Composite Simpson-rule integral of sampled values."""
    return float(spi.simpson(np.asarray(y, dtype=float), x=x, dx=dx))
