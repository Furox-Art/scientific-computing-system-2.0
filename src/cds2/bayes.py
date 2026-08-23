"""Bayesian inference: conjugate updates, posterior sampling and naive Bayes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

__all__ = [
    "BetaPosterior",
    "NormalNormalPosterior",
    "GammaPoissonPosterior",
    "NaiveBayes",
    "beta_binomial_update",
    "normal_normal_update",
    "gamma_poisson_update",
    "posterior_interval",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BetaPosterior:
    """Beta posterior over a Bernoulli success probability."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def mode(self) -> float:
        total = self.alpha + self.beta
        if total <= 2:
            return self.mean
        return (self.alpha - 1.0) / (total - 2.0)


@dataclass(frozen=True)
class NormalNormalPosterior:
    """Conjugate normal posterior over a normal mean with known sigma."""

    mean: float
    precision: float


@dataclass(frozen=True)
class GammaPoissonPosterior:
    """Gamma posterior over a Poisson rate."""

    shape: float
    rate: float

    @property
    def mean(self) -> float:
        return self.shape / self.rate


def beta_binomial_update(
    successes: int | Sequence[int],
    failures: int | Sequence[int],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> BetaPosterior:
    """Conjugate Beta-Binomial update; sequences accumulate batch evidence."""
    alpha = prior_alpha + int(np.sum(np.asarray(successes, dtype=int)))
    beta = prior_beta + int(np.sum(np.asarray(failures, dtype=int)))
    if prior_alpha <= 0 or prior_beta <= 0 or alpha < 0 or beta < 0:
        msg = "prior parameters must be positive"
        raise ValueError(msg)
    return BetaPosterior(alpha=float(alpha), beta=float(beta))


def normal_normal_update(
    observations: Sequence[float],
    known_sigma: float,
    prior_mean: float = 0.0,
    prior_sigma: float = 10.0,
) -> NormalNormalPosterior:
    """Conjugate normal posterior for a mean with a known observation sigma."""
    if known_sigma <= 0 or prior_sigma <= 0:
        msg = "sigmas must be positive"
        raise ValueError(msg)
    data = np.asarray(observations, dtype=float)
    n = data.size
    likelihood_precision = n / known_sigma**2
    prior_precision = 1.0 / prior_sigma**2
    posterior_precision = prior_precision + likelihood_precision
    weighted_mean = (
        prior_mean * prior_precision + float(data.mean()) * likelihood_precision
    ) / posterior_precision
    return NormalNormalPosterior(mean=weighted_mean, precision=posterior_precision)


def gamma_poisson_update(
    event_counts: int | Sequence[int],
    exposure: float = 1.0,
    prior_shape: float = 1.0,
    prior_rate: float = 1.0,
) -> GammaPoissonPosterior:
    """Conjugate Gamma-Poisson update over an event rate per unit exposure."""
    if prior_shape <= 0 or prior_rate <= 0:
        msg = "prior parameters must be positive"
        raise ValueError(msg)
    if exposure <= 0:
        msg = "exposure must be positive"
        raise ValueError(msg)
    total_events = int(np.sum(np.asarray(event_counts, dtype=int)))
    return GammaPoissonPosterior(shape=prior_shape + total_events, rate=prior_rate + exposure)


def posterior_interval(
    distribution: object,
    credibility: float = 0.95,
    sample_size: int = 20_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Equal-tailed credible interval from any scipy frozen distribution.

    Falls back to sampling when the inverse CDF is unavailable.
    """
    if not 0.0 < credibility < 1.0:
        msg = "credibility must lie strictly between 0 and 1"
        raise ValueError(msg)
    ppf = getattr(distribution, "ppf", None)
    if callable(ppf):
        lower_q = (1.0 - credibility) / 2.0
        lower = float(ppf(lower_q))
        upper = float(ppf(1.0 - lower_q))
        return lower, upper
    rvs = getattr(distribution, "rvs", None)
    if not callable(rvs):
        msg = "distribution supports neither ppf nor rvs"
        raise TypeError(msg)
    samples = np.asarray(rvs(size=sample_size, random_state=seed), dtype=float)
    tail = (1.0 - credibility) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return float(lower), float(upper)


class NaiveBayes:
    """Gaussian naive-Bayes classifier fit by class-conditional statistics."""

    def __init__(self) -> None:
        self.classes_: FloatArray | None = None
        self._means: dict[float, FloatArray] = {}
        self._stds: dict[float, FloatArray] = {}
        self._log_priors: dict[float, float] = {}

    def fit(self, features: Sequence[Sequence[float]], labels: Sequence[object]) -> NaiveBayes:
        matrix = np.asarray(features, dtype=float)
        targets = np.asarray(labels)
        if matrix.ndim != 2 or targets.ndim != 1 or matrix.shape[0] != targets.size:
            msg = "features must be 2-D and labels 1-D with matching lengths"
            raise ValueError(msg)
        self.classes_ = np.unique(targets).astype(float)
        count = targets.size
        for cls in self.classes_.tolist():
            mask = targets == cls
            members = matrix[mask]
            self._means[cls] = members.mean(axis=0)
            stds = members.std(axis=0, ddof=0)
            self._stds[cls] = np.where(stds == 0, 1e-9, stds)
            self._log_priors[cls] = float(np.log(mask.sum() / count))
        return self

    def predict_log_proba(self, features: Sequence[Sequence[float]]) -> FloatArray:
        """Log joint probabilities log P(class) + sum log P(feature|class)."""
        if self.classes_ is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        matrix = np.asarray(features, dtype=float)
        scores = np.empty((matrix.shape[0], self.classes_.size))
        for column, cls in enumerate(self.classes_.tolist()):
            gaussian = sp_stats.norm(loc=self._means[cls], scale=self._stds[cls])
            joint = gaussian.logpdf(matrix).sum(axis=1) + self._log_priors[cls]
            scores[:, column] = joint
        return scores

    def predict(self, features: Sequence[Sequence[float]]) -> FloatArray:
        """Most probable class label per row."""
        log_scores = self.predict_log_proba(features)
        assert self.classes_ is not None
        winners = np.argmax(log_scores, axis=1)
        return np.asarray(self.classes_[winners], dtype=float)


def bayes_factor(
    log_likelihood_h1: float,
    log_likelihood_h0: float,
) -> float:
    """Bayes factor BF10 from two maximized log-likelihoods."""
    return float(np.exp(log_likelihood_h1 - log_likelihood_h0))


def metropolis_posterior(
    log_prior: Callable[[FloatArray], float],
    log_likelihood: Callable[[FloatArray], float],
    initial: Sequence[float],
    n_samples: int = 5_000,
    burn_in: int = 500,
    proposal_scale: float = 0.5,
    seed: int | None = None,
) -> FloatArray:
    """Random-walk Metropolis draws from ``log_prior + log_likelihood``."""
    from cds2.montecarlo import metropolis_hastings

    def log_posterior(state: FloatArray) -> float:
        prior_value = log_prior(state)
        if np.isneginf(prior_value):
            return float("-inf")
        return prior_value + log_likelihood(state)

    result = metropolis_hastings(
        log_posterior,
        initial=np.asarray(initial, dtype=float),
        n_samples=n_samples,
        burn_in=burn_in,
        proposal_scale=proposal_scale,
        seed=seed,
    )
    return result.samples
