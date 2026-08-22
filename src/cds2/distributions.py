"""Probability distribution helpers built on scipy.stats."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sps

__all__ = [
    "student_t_pdf",
    "student_t_cdf",
    "student_t_ppf",
    "chi2_pdf",
    "chi2_cdf",
    "chi2_ppf",
    "f_pdf",
    "f_cdf",
    "f_ppf",
    "exponential_pdf",
    "exponential_cdf",
    "exponential_ppf",
    "uniform_pdf",
    "uniform_cdf",
    "uniform_ppf",
    "lognormal_pdf",
    "lognormal_cdf",
    "lognormal_ppf",
    "poisson_pmf",
    "poisson_cdf",
    "poisson_ppf",
    "binomial_pmf",
    "binomial_cdf",
    "binomial_ppf",
]


def _as_array(x: object) -> NDArray[np.float64]:
    return np.atleast_1d(np.asarray(x, dtype=float))


def _wrap(values: object) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float)


# ------------------------------------------------------- Student's t ----
def student_t_pdf(x: object, df: float) -> NDArray[np.float64]:
    """Student's t density with ``df`` degrees of freedom."""
    return _wrap(sps.t.pdf(_as_array(x), df=df))


def student_t_cdf(x: object, df: float) -> NDArray[np.float64]:
    """Student's t cumulative distribution function."""
    return _wrap(sps.t.cdf(_as_array(x), df=df))


def student_t_ppf(q: object, df: float) -> NDArray[np.float64]:
    """Student's t quantile function."""
    return _wrap(sps.t.ppf(_as_array(q), df=df))


# ------------------------------------------------------ chi-squared ----
def chi2_pdf(x: object, df: float) -> NDArray[np.float64]:
    """Chi-squared density."""
    return _wrap(sps.chi2.pdf(_as_array(x), df=df))


def chi2_cdf(x: object, df: float) -> NDArray[np.float64]:
    """Chi-squared cumulative distribution function."""
    return _wrap(sps.chi2.cdf(_as_array(x), df=df))


def chi2_ppf(q: object, df: float) -> NDArray[np.float64]:
    """Chi-squared quantile function."""
    return _wrap(sps.chi2.ppf(_as_array(q), df=df))


# ---------------------------------------------------------------- F ----
def f_pdf(x: object, dfn: float, dfd: float) -> NDArray[np.float64]:
    """F distribution density."""
    return _wrap(sps.f.pdf(_as_array(x), dfn=dfn, dfd=dfd))


def f_cdf(x: object, dfn: float, dfd: float) -> NDArray[np.float64]:
    """F cumulative distribution function."""
    return _wrap(sps.f.cdf(_as_array(x), dfn=dfn, dfd=dfd))


def f_ppf(q: object, dfn: float, dfd: float) -> NDArray[np.float64]:
    """F quantile function."""
    return _wrap(sps.f.ppf(_as_array(q), dfn=dfn, dfd=dfd))


# -------------------------------------------------------- exponential ----
def exponential_pdf(x: object, scale: float = 1.0) -> NDArray[np.float64]:
    """Exponential density with mean ``scale``."""
    return _wrap(sps.expon.pdf(_as_array(x), scale=scale))


def exponential_cdf(x: object, scale: float = 1.0) -> NDArray[np.float64]:
    """Exponential cumulative distribution function."""
    return _wrap(sps.expon.cdf(_as_array(x), scale=scale))


def exponential_ppf(q: object, scale: float = 1.0) -> NDArray[np.float64]:
    """Exponential quantile function."""
    return _wrap(sps.expon.ppf(_as_array(q), scale=scale))


# ------------------------------------------------------------ uniform ----
def uniform_pdf(x: object, loc: float = 0.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Continuous uniform density on [loc, loc + scale]."""
    return _wrap(sps.uniform.pdf(_as_array(x), loc=loc, scale=scale))


def uniform_cdf(x: object, loc: float = 0.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Uniform cumulative distribution function."""
    return _wrap(sps.uniform.cdf(_as_array(x), loc=loc, scale=scale))


def uniform_ppf(q: object, loc: float = 0.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Uniform quantile function."""
    return _wrap(sps.uniform.ppf(_as_array(q), loc=loc, scale=scale))


# --------------------------------------------------------- lognormal ----
def lognormal_pdf(x: object, sigma: float = 1.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Lognormal density of exp(N(0, sigma^2)) scaled by ``scale``."""
    return _wrap(sps.lognorm.pdf(_as_array(x), s=sigma, scale=scale))


def lognormal_cdf(x: object, sigma: float = 1.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Lognormal cumulative distribution function."""
    return _wrap(sps.lognorm.cdf(_as_array(x), s=sigma, scale=scale))


def lognormal_ppf(q: object, sigma: float = 1.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Lognormal quantile function."""
    return _wrap(sps.lognorm.ppf(_as_array(q), s=sigma, scale=scale))


# ------------------------------------------------------------ poisson ----
def poisson_pmf(k: object, mu: float) -> NDArray[np.float64]:
    """Poisson probability mass at integer counts ``k``."""
    return _wrap(sps.poisson.pmf(np.atleast_1d(np.asarray(k)), mu))


def poisson_cdf(k: object, mu: float) -> NDArray[np.float64]:
    """Poisson cumulative mass up to ``k``."""
    return _wrap(sps.poisson.cdf(np.atleast_1d(np.asarray(k)), mu))


def poisson_ppf(q: object, mu: float) -> NDArray[np.float64]:
    """Poisson quantile (smallest count with CDF >= q)."""
    return _wrap(sps.poisson.ppf(_as_array(q), mu))


# ----------------------------------------------------------- binomial ----
def binomial_pmf(k: object, n: int, p: float) -> NDArray[np.float64]:
    """Binomial probability mass for ``n`` trials with success odds ``p``."""
    return _wrap(sps.binom.pmf(np.atleast_1d(np.asarray(k)), n, p))


def binomial_cdf(k: object, n: int, p: float) -> NDArray[np.float64]:
    """Binomial cumulative mass up to ``k``."""
    return _wrap(sps.binom.cdf(np.atleast_1d(np.asarray(k)), n, p))


def binomial_ppf(q: object, n: int, p: float) -> NDArray[np.float64]:
    """Binomial quantile function."""
    return _wrap(sps.binom.ppf(_as_array(q), n, p))
