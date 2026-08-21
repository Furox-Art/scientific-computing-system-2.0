"""Special mathematical functions built on scipy.special."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import special as sps

__all__ = [
    "gamma_fn",
    "gammaln",
    "erf",
    "erfc",
    "erfinv",
    "beta_fn",
    "betaln",
    "bessel_j0",
    "bessel_j1",
    "bessel_y0",
    "zeta",
]


def _as_array(x: object) -> NDArray[np.float64]:
    return np.atleast_1d(np.asarray(x, dtype=float))


def gamma_fn(x: object) -> NDArray[np.float64]:
    """Euler gamma function for real arguments."""
    return _as_array(sps.gamma(_as_array(x)))


def gammaln(x: object) -> NDArray[np.float64]:
    """Log-gamma function - numerically stable for large inputs."""
    return _as_array(sps.gammaln(_as_array(x)))


def erf(x: object) -> NDArray[np.float64]:
    """Gaussian error function."""
    return _as_array(sps.erf(_as_array(x)))


def erfc(x: object) -> NDArray[np.float64]:
    """Complementary error function (accurate in the far tail)."""
    return _as_array(sps.erfc(_as_array(x)))


def erfinv(x: object) -> NDArray[np.float64]:
    """Inverse error function on [-1, 1]."""
    return _as_array(sps.erfinv(_as_array(x)))


def beta_fn(a: float, b: float) -> float:
    """Euler beta function."""
    return float(sps.beta(a, b))


def betaln(a: float, b: float) -> float:
    """Log-beta function."""
    return float(sps.betaln(a, b))


def bessel_j0(x: object) -> NDArray[np.float64]:
    """Bessel function of the first kind, order 0."""
    return _as_array(sps.j0(_as_array(x)))


def bessel_j1(x: object) -> NDArray[np.float64]:
    """Bessel function of the first kind, order 1."""
    return _as_array(sps.j1(_as_array(x)))


def bessel_y0(x: object) -> NDArray[np.float64]:
    """Bessel function of the second kind (Neumann), order 0."""
    return _as_array(sps.y0(_as_array(x)))


def zeta(x: float, q: float | None = None) -> float:
    """Riemann (or Hurwitz with ``q``) zeta function at a real point."""
    if q is None:
        return float(sps.zeta(x))
    return float(sps.zeta(x, q))
