"""Interpolation built on scipy.interpolate and numpy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import interpolate as spi

__all__ = [
    "linear_interp",
    "cubic_spline",
    "pchip_interpolator",
    "lagrange_poly",
    "grid_interp",
    "regular_grid_interp",
]


def linear_interp(x_new: object, x_known: Sequence[float], y_known: Sequence[float]) -> np.ndarray:
    """Piecewise-linear interpolation at ``x_new`` through known points."""
    return np.asarray(
        np.interp(np.asarray(x_new, dtype=float), np.asarray(x_known), np.asarray(y_known))
    )


def cubic_spline(x: Sequence[float], y: Sequence[float]) -> spi.CubicSpline:
    """Natural cubic spline interpolant callable on any point in range."""
    return spi.CubicSpline(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def pchip_interpolator(x: Sequence[float], y: Sequence[float]) -> spi.PchipInterpolator:
    """Monotone shape-preserving PCHIP interpolant."""
    return spi.PchipInterpolator(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def lagrange_poly(x: Sequence[float], y: Sequence[float]) -> np.poly1d:
    """Lagrange polynomial passing exactly through all given points."""
    poly = spi.lagrange(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if isinstance(poly, np.ndarray):
        msg = "lagrange returned an unexpected array result"
        raise TypeError(msg)
    return np.poly1d(poly)


def grid_interp(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    query_points: Sequence[Sequence[float]],
    method: str = "linear",
) -> np.ndarray:
    """Interpolate values defined at scattered N-D points onto new locations."""
    result = spi.griddata(
        np.asarray(points, dtype=float),
        np.asarray(values, dtype=float),
        np.asarray(query_points, dtype=float),
        method=method,
    )
    return np.asarray(result)


def regular_grid_interp(
    grid_axes: Sequence[Sequence[float]],
    values: object,
    query_points: object,
) -> np.ndarray:
    """Interpolate on a regular (rectilinear) grid built from axis vectors."""
    interpolator = spi.RegularGridInterpolator(
        [np.asarray(axis, dtype=float) for axis in grid_axes],
        np.asarray(values, dtype=float),
    )
    return np.asarray(interpolator(np.atleast_2d(query_points)))
