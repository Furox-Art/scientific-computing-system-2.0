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
    "gamma_pdf",
    "gamma_cdf",
    "gamma_ppf",
    "beta_pdf",
    "beta_cdf",
    "beta_ppf",
    "weibull_pdf",
    "weibull_cdf",
    "weibull_ppf",
    "cauchy_pdf",
    "cauchy_cdf",
    "cauchy_ppf",
    "laplace_pdf",
    "laplace_cdf",
    "laplace_ppf",
    "gumbel_pdf",
    "gumbel_cdf",
    "gumbel_ppf",
    "pareto_pdf",
    "pareto_cdf",
    "pareto_ppf",
    "rayleigh_pdf",
    "rayleigh_cdf",
    "rayleigh_ppf",
    "geometric_pmf",
    "geometric_cdf",
    "geometric_ppf",
    "negative_binomial_pmf",
    "negative_binomial_cdf",
    "negative_binomial_ppf",
    "hypergeometric_pmf",
    "hypergeometric_cdf",
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


# --------------------------------------------------------------- gamma ----
def gamma_pdf(x: object, a: float, scale: float = 1.0) -> NDArray[np.float64]:
    """Gamma density with shape ``a`` and ``scale``."""
    return _wrap(sps.gamma.pdf(_as_array(x), a=a, scale=scale))


def gamma_cdf(x: object, a: float, scale: float = 1.0) -> NDArray[np.float64]:
    """Gamma cumulative distribution function."""
    return _wrap(sps.gamma.cdf(_as_array(x), a=a, scale=scale))


def gamma_ppf(q: object, a: float, scale: float = 1.0) -> NDArray[np.float64]:
    """Gamma quantile function."""
    return _wrap(sps.gamma.ppf(_as_array(q), a=a, scale=scale))


# ---------------------------------------------------------------- beta ----
def beta_pdf(x: object, a: float, b: float) -> NDArray[np.float64]:
    """Beta density on [0, 1] with shape parameters ``a``, ``b``."""
    return _wrap(sps.beta.pdf(_as_array(x), a=a, b=b))


def beta_cdf(x: object, a: float, b: float) -> NDArray[np.float64]:
    """Beta cumulative distribution function."""
    return _wrap(sps.beta.cdf(_as_array(x), a=a, b=b))


def beta_ppf(q: object, a: float, b: float) -> NDArray[np.float64]:
    """Beta quantile function."""
    return _wrap(sps.beta.ppf(_as_array(q), a=a, b=b))


# ------------------------------------------------------------- weibull ----
def weibull_pdf(x: object, c: float, scale: float = 1.0) -> NDArray[np.float64]:
    """Weibull density with shape ``c``."""
    return _wrap(sps.weibull_min.pdf(_as_array(x), c=c, scale=scale))


def weibull_cdf(x: object, c: float, scale: float = 1.0) -> NDArray[np.float64]:
    """Weibull cumulative distribution function."""
    return _wrap(sps.weibull_min.cdf(_as_array(x), c=c, scale=scale))


def weibull_ppf(q: object, c: float, scale: float = 1.0) -> NDArray[np.float64]:
    """Weibull quantile function."""
    return _wrap(sps.weibull_min.ppf(_as_array(q), c=c, scale=scale))


# -------------------------------------------------------------- cauchy ----
def cauchy_pdf(x: object, loc: float = 0.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Cauchy density - heavy-tailed, no finite moments."""
    return _wrap(sps.cauchy.pdf(_as_array(x), loc=loc, scale=scale))


def cauchy_cdf(x: object, loc: float = 0.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Cauchy cumulative distribution function."""
    return _wrap(sps.cauchy.cdf(_as_array(x), loc=loc, scale=scale))


def cauchy_ppf(q: object, loc: float = 0.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Cauchy quantile function."""
    return _wrap(sps.cauchy.ppf(_as_array(q), loc=loc, scale=scale))


# -------------------------------------------------------------- laplace ----
def laplace_pdf(x: object, loc: float = 0.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Laplace (double exponential) density."""
    return _wrap(sps.laplace.pdf(_as_array(x), loc=loc, scale=scale))


def laplace_cdf(x: object, loc: float = 0.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Laplace cumulative distribution function."""
    return _wrap(sps.laplace.cdf(_as_array(x), loc=loc, scale=scale))


def laplace_ppf(q: object, loc: float = 0.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Laplace quantile function."""
    return _wrap(sps.laplace.ppf(_as_array(q), loc=loc, scale=scale))


# --------------------------------------------------------------- gumbel ----
def gumbel_pdf(x: object, loc: float = 0.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Gumbel extreme-value density for maxima."""
    return _wrap(sps.gumbel_r.pdf(_as_array(x), loc=loc, scale=scale))


def gumbel_cdf(x: object, loc: float = 0.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Gumbel cumulative distribution function."""
    return _wrap(sps.gumbel_r.cdf(_as_array(x), loc=loc, scale=scale))


def gumbel_ppf(q: object, loc: float = 0.0, scale: float = 1.0) -> NDArray[np.float64]:
    """Gumbel quantile function."""
    return _wrap(sps.gumbel_r.ppf(_as_array(q), loc=loc, scale=scale))


# --------------------------------------------------------------- pareto ----
def pareto_pdf(x: object, b: float, scale: float = 1.0) -> NDArray[np.float64]:
    """Pareto power-law density with tail index ``b``."""
    return _wrap(sps.pareto.pdf(_as_array(x), b=b, scale=scale))


def pareto_cdf(x: object, b: float, scale: float = 1.0) -> NDArray[np.float64]:
    """Pareto cumulative distribution function."""
    return _wrap(sps.pareto.cdf(_as_array(x), b=b, scale=scale))


def pareto_ppf(q: object, b: float, scale: float = 1.0) -> NDArray[np.float64]:
    """Pareto quantile function."""
    return _wrap(sps.pareto.ppf(_as_array(q), b=b, scale=scale))


# ------------------------------------------------------------- rayleigh ----
def rayleigh_pdf(x: object, scale: float = 1.0) -> NDArray[np.float64]:
    """Rayleigh density - magnitude of 2-D Gaussian noise."""
    return _wrap(sps.rayleigh.pdf(_as_array(x), scale=scale))


def rayleigh_cdf(x: object, scale: float = 1.0) -> NDArray[np.float64]:
    """Rayleigh cumulative distribution function."""
    return _wrap(sps.rayleigh.cdf(_as_array(x), scale=scale))


def rayleigh_ppf(q: object, scale: float = 1.0) -> NDArray[np.float64]:
    """Rayleigh quantile function."""
    return _wrap(sps.rayleigh.ppf(_as_array(q), scale=scale))


# ------------------------------------------------------------ geometric ----
def geometric_pmf(k: object, p: float) -> NDArray[np.float64]:
    """Geometric mass - trials until first success."""
    return _wrap(sps.geom.pmf(np.atleast_1d(np.asarray(k)), p))


def geometric_cdf(k: object, p: float) -> NDArray[np.float64]:
    """Geometric cumulative mass."""
    return _wrap(sps.geom.cdf(np.atleast_1d(np.asarray(k)), p))


def geometric_ppf(q: object, p: float) -> NDArray[np.float64]:
    """Geometric quantile function."""
    return _wrap(sps.geom.ppf(_as_array(q), p))


# ------------------------------------------------------ negative binomial ----
def negative_binomial_pmf(k: object, n_failures: float, p: float) -> NDArray[np.float64]:
    """Negative-binomial mass - successes before ``n_failures`` failures."""
    return _wrap(sps.nbinom.pmf(np.atleast_1d(np.asarray(k)), n_failures, p))


def negative_binomial_cdf(k: object, n_failures: float, p: float) -> NDArray[np.float64]:
    """Negative-binomial cumulative mass."""
    return _wrap(sps.nbinom.cdf(np.atleast_1d(np.asarray(k)), n_failures, p))


def negative_binomial_ppf(q: object, n_failures: float, p: float) -> NDArray[np.float64]:
    """Negative-binomial quantile function."""
    return _wrap(sps.nbinom.ppf(_as_array(q), n_failures, p))


# ------------------------------------------------------- hypergeometric ----
def hypergeometric_pmf(k: object, ngood: int, nbad: int, nsample: int) -> NDArray[np.float64]:
    """Hypergeometric mass - draws without replacement."""
    population = ngood + nbad
    return _wrap(sps.hypergeom.pmf(np.atleast_1d(np.asarray(k)), population, ngood, nsample))


def hypergeometric_cdf(k: object, ngood: int, nbad: int, nsample: int) -> NDArray[np.float64]:
    """Hypergeometric cumulative mass."""
    population = ngood + nbad
    return _wrap(sps.hypergeom.cdf(np.atleast_1d(np.asarray(k)), population, ngood, nsample))
