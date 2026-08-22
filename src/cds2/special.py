"""Special mathematical functions built on scipy.special."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import special as sps

__all__ = [
    "gamma_fn",
    "gammaln",
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "fresnel_c",
    "fresnel_s",
    "beta_fn",
    "betaln",
    "bessel_j0",
    "bessel_j1",
    "bessel_y0",
    "spherical_bessel_j0",
    "spherical_bessel_j1",
    "airy_ai",
    "airy_bi",
    "legendre_pn",
    "elliptic_k",
    "elliptic_e",
    "exp1",
    "hypergeometric_2f1",
    "zeta",
    "bessel_jv",
    "bessel_yv",
    "bessel_iv",
    "bessel_kv",
    "hankel1_fn",
    "struve_h0",
    "struve_h1",
    "chebyshev_tn",
    "chebyshev_un",
    "laguerre_ln",
    "hermite_hn",
    "jacobi_pnab",
    "spherical_harmonic",
    "lambert_w",
    "faddeeva_w",
    "exponential_integral_en",
    "sine_cosine_integrals",
    "binomial_coefficient",
]


def _as_array(x: object) -> NDArray[np.float64]:
    return np.atleast_1d(np.asarray(x, dtype=float))


def gamma_fn(x: object) -> NDArray[np.float64]:
    """Euler gamma function for real arguments."""
    return _as_array(sps.gamma(_as_array(x)))


def gammaln(x: object) -> NDArray[np.float64]:
    """Log-gamma function - numerically stable for large inputs."""
    return _as_array(sps.gammaln(_as_array(x)))


def digamma(x: object) -> NDArray[np.float64]:
    """Digamma (psi) function - logarithmic derivative of gamma."""
    return _as_array(sps.digamma(_as_array(x)))


def fresnel_c(x: object) -> NDArray[np.float64]:
    """Fresnel cosine integral C(x)."""
    return _as_array(sps.fresnel(_as_array(x))[0])


def fresnel_s(x: object) -> NDArray[np.float64]:
    """Fresnel sine integral S(x)."""
    return _as_array(sps.fresnel(_as_array(x))[1])


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


def spherical_bessel_j0(x: object) -> NDArray[np.float64]:
    """Spherical Bessel function of the first kind, order 0."""
    return _as_array(sps.spherical_jn(0, _as_array(x)))


def spherical_bessel_j1(x: object) -> NDArray[np.float64]:
    """Spherical Bessel function of the first kind, order 1."""
    return _as_array(sps.spherical_jn(1, _as_array(x)))


def airy_ai(x: object) -> NDArray[np.float64]:
    """Airy function Ai - solves y'' = x*y, decaying for x > 0."""
    ai, _aip, _bi, _bip = sps.airy(_as_array(x))
    return _as_array(ai)


def airy_bi(x: object) -> NDArray[np.float64]:
    """Airy function Bi - grows exponentially for x > 0."""
    _ai, _aip, bi, _bip = sps.airy(_as_array(x))
    return _as_array(bi)


def legendre_pn(n: int, x: object) -> NDArray[np.float64]:
    """Legendre polynomial P_n evaluated on [-1, 1]."""
    return _as_array(sps.eval_legendre(n, _as_array(x)))


def elliptic_k(m: object) -> NDArray[np.float64]:
    """Complete elliptic integral of the first kind K(m)."""
    return _as_array(sps.ellipk(_as_array(m)))


def elliptic_e(m: object) -> NDArray[np.float64]:
    """Complete elliptic integral of the second kind E(m)."""
    return _as_array(sps.ellipe(_as_array(m)))


def exp1(x: object) -> NDArray[np.float64]:
    """Exponential integral E1(x) = integral from x to inf of e^-t/t."""
    return _as_array(sps.exp1(_as_array(x)))


def hypergeometric_2f1(a: float, b: float, c: float, z: object) -> NDArray[np.float64]:
    """Gauss hypergeometric function 2F1(a, b; c; z)."""
    return _as_array(sps.hyp2f1(a, b, c, _as_array(z)))


def zeta(x: float, q: float | None = None) -> float:
    """Riemann (or Hurwitz with ``q``) zeta function at a real point."""
    if q is None:
        return float(sps.zeta(x))
    return float(sps.zeta(x, q))


# -------------------------------------------------- extended families ----
def bessel_jv(nu: float, x: object) -> NDArray[np.float64]:
    """Bessel function of the first kind of real order ``nu``."""
    return _as_array(sps.jv(nu, _as_array(x)))


def bessel_yv(nu: float, x: object) -> NDArray[np.float64]:
    """Bessel function of the second kind of real order ``nu``."""
    return _as_array(sps.yv(nu, _as_array(x)))


def bessel_iv(nu: float, x: object) -> NDArray[np.float64]:
    """Modified Bessel function of the first kind."""
    return _as_array(sps.iv(nu, _as_array(x)))


def bessel_kv(nu: float, x: object) -> NDArray[np.float64]:
    """Modified Bessel function of the second kind."""
    return _as_array(sps.kv(nu, _as_array(x)))


def hankel1_fn(nu: float, x: object) -> NDArray[np.float64]:
    """Hankel function of the first kind H1_nu (real part)."""
    return _as_array(np.real(sps.hankel1(nu, _as_array(x))))


def struve_h0(x: object) -> NDArray[np.float64]:
    """Struve function H_0."""
    return _as_array(sps.struve(0, _as_array(x)))


def struve_h1(x: object) -> NDArray[np.float64]:
    """Struve function H_1."""
    return _as_array(sps.struve(1, _as_array(x)))


def chebyshev_tn(n: int, x: object) -> NDArray[np.float64]:
    """Chebyshev polynomial T_n on [-1, 1]."""
    return _as_array(sps.eval_chebyt(n, _as_array(x)))


def chebyshev_un(n: int, x: object) -> NDArray[np.float64]:
    """Chebyshev polynomial U_n on [-1, 1]."""
    return _as_array(sps.eval_chebyu(n, _as_array(x)))


def laguerre_ln(n: int, x: object) -> NDArray[np.float64]:
    """Laguerre polynomial L_n on [0, inf)."""
    return _as_array(sps.eval_laguerre(n, _as_array(x)))


def hermite_hn(n: int, x: object) -> NDArray[np.float64]:
    """Physicists' Hermite polynomial H_n."""
    return _as_array(sps.eval_hermite(n, _as_array(x)))


def jacobi_pnab(n: int, alpha: float, beta_param: float, x: object) -> NDArray[np.float64]:
    """Jacobi polynomial P_n^(alpha, beta) on [-1, 1]."""
    return _as_array(sps.eval_jacobi(n, alpha, beta_param, _as_array(x)))


def spherical_harmonic(m: int, l_degree: int, theta: object, phi: object) -> NDArray[np.complex128]:
    """Spherical harmonic Y_l^m at polar ``theta`` and azimuth ``phi``."""
    values = sps.sph_harm_y(
        l_degree,
        m,
        np.asarray(theta, dtype=float),
        np.asarray(phi, dtype=float),
    )
    return np.atleast_1d(np.asarray(values, dtype=np.complex128))


def lambert_w(x: object) -> NDArray[np.float64]:
    """Lambert W function solving W*exp(W) = x (principal branch)."""
    return _as_array(np.real(sps.lambertw(_as_array(x))))


def faddeeva_w(z_real: object, z_imag: object) -> NDArray[np.complex128]:
    """Faddeeva function w(z) for z = z_real + i*z_imag."""
    argument = np.asarray(z_real, dtype=float) + 1j * np.asarray(z_imag, dtype=float)
    return np.atleast_1d(np.asarray(sps.wofz(argument), dtype=np.complex128))


def exponential_integral_en(n_order: int, x: object) -> NDArray[np.float64]:
    """Generalized exponential integral E_n(x)."""
    return _as_array(sps.expn(n_order, _as_array(x)))


def sine_cosine_integrals(x: object) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (Si(x), Ci(x)) - sine and cosine integrals."""
    si_values, ci_values = sps.sici(_as_array(x))
    return _as_array(si_values), _as_array(ci_values)


def binomial_coefficient(n: int, k: int) -> float:
    """Number of ways to choose ``k`` items from ``n``."""
    return float(sps.comb(n, k, exact=True))
