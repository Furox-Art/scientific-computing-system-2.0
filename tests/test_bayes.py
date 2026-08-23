"""Tests for cds2.bayes."""

import numpy as np
import pytest
from scipy import stats as sp_stats

from cds2 import bayes


class TestBetaBinomial:
    def test_uniform_prior_mean(self) -> None:
        posterior = bayes.beta_binomial_update(successes=7, failures=3)
        assert posterior.alpha == 8.0
        assert posterior.beta == 4.0
        assert posterior.mean == pytest.approx(8.0 / 12.0)

    def test_mode(self) -> None:
        posterior = bayes.beta_binomial_update(successes=7, failures=3)
        assert posterior.mode == pytest.approx(7.0 / 10.0)

    def test_batch_updates(self) -> None:
        single = bayes.beta_binomial_update([2, 3], [1, 2], prior_alpha=2.0, prior_beta=2.0)
        assert (single.alpha, single.beta) == (7.0, 5.0)

    def test_invalid_prior_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            bayes.beta_binomial_update(1, 1, prior_alpha=0.0)


class TestNormalNormal:
    def test_posterior_shrinks_toward_data(self) -> None:
        posterior = bayes.normal_normal_update(
            [5.0, 5.2, 4.8], known_sigma=1.0, prior_mean=0.0, prior_sigma=10.0
        )
        assert posterior.mean == pytest.approx(15.0 / 3.01)

    def test_strong_prior_dominates_single_point(self) -> None:
        posterior = bayes.normal_normal_update(
            [100.0], known_sigma=10.0, prior_mean=0.0, prior_sigma=0.001
        )
        assert posterior.mean < 1.0

    def test_nonpositive_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            bayes.normal_normal_update([1.0], known_sigma=0.0)


class TestGammaPoisson:
    def test_update(self) -> None:
        posterior = bayes.gamma_poisson_update([3, 4], exposure=2.0)
        assert posterior.shape == 8.0
        assert posterior.rate == 3.0
        assert posterior.mean == pytest.approx(8.0 / 3.0)

    def test_invalid_exposure(self) -> None:
        with pytest.raises(ValueError, match="exposure"):
            bayes.gamma_poisson_update(1, exposure=0.0)


class TestPosteriorInterval:
    def test_beta_interval_brackets_mean(self) -> None:
        distribution = sp_stats.beta(8.0, 4.0)
        low, high = bayes.posterior_interval(distribution)
        assert low < 8.0 / 12.0 < high

    def test_credibility_width_grows(self) -> None:
        distribution = sp_stats.norm(loc=0.0, scale=1.0)
        narrow = bayes.posterior_interval(distribution, credibility=0.5)
        wide = bayes.posterior_interval(distribution, credibility=0.99)
        assert wide[1] - wide[0] > narrow[1] - narrow[0]

    def test_invalid_credibility(self) -> None:
        with pytest.raises(ValueError, match="credibility"):
            bayes.posterior_interval(sp_stats.norm(), credibility=1.5)

    def test_sampling_fallback(self) -> None:
        class SampleOnly:
            def rvs(self, size: int, random_state: int) -> np.ndarray:
                rng = np.random.default_rng(random_state)
                return rng.normal(size=size)

        low, high = bayes.posterior_interval(SampleOnly(), seed=1)
        assert low < high

    def test_unsupported_distribution_raises(self) -> None:
        class Empty:
            pass

        with pytest.raises(TypeError, match="ppf nor rvs"):
            bayes.posterior_interval(Empty())


class TestNaiveBayes:
    @pytest.fixture()
    def model(self) -> bayes.NaiveBayes:
        features = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]]
        labels = [0, 0, 0, 1, 1, 1]
        return bayes.NaiveBayes().fit(features, labels)

    def test_separable_classes(self, model: bayes.NaiveBayes) -> None:
        predictions = model.predict([[0.05, 0.05], [5.05, 5.05]])
        assert list(predictions) == [0.0, 1.0]

    def test_log_proba_shape(self, model: bayes.NaiveBayes) -> None:
        scores = model.predict_log_proba([[1.0, 1.0]])
        assert scores.shape == (1, 2)

    def test_unfitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            bayes.NaiveBayes().predict([[0.0]])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="matching lengths"):
            bayes.NaiveBayes().fit([[0.0]], [0, 1])


class TestBayesFactor:
    def test_equal_likelihoods_is_one(self) -> None:
        assert bayes.bayes_factor(-3.0, -3.0) == pytest.approx(1.0)

    def test_evidence_favors_h1(self) -> None:
        assert bayes.bayes_factor(-2.0, -6.0) > 1.0


class TestMetropolisPosterior:
    def test_recovers_normal_location(self) -> None:
        rng = np.random.default_rng(11)
        data = rng.normal(loc=2.0, scale=1.0, size=60)

        samples = bayes.metropolis_posterior(
            log_prior=lambda state: 0.0,
            log_likelihood=lambda state: float(
                sp_stats.norm.logpdf(data, loc=state[0], scale=1.0).sum()
            ),
            initial=[0.0],
            n_samples=4000,
            burn_in=500,
            proposal_scale=0.4,
            seed=3,
        )
        assert abs(float(samples[:, 0].mean()) - 2.0) < 0.25

    def test_rejects_outside_prior(self) -> None:
        def prior(state: np.ndarray) -> float:
            return 0.0 if state[0] > 0 else float("-inf")

        samples = bayes.metropolis_posterior(
            log_prior=prior,
            log_likelihood=lambda state: -float(state[0] ** 2),
            initial=[1.0],
            n_samples=200,
            seed=5,
        )
        assert np.all(samples[:, 0] > 0)


class TestCoverageEdges:
    def test_beta_mode_falls_back_to_mean_for_tiny_alpha(self) -> None:
        posterior = bayes.BetaPosterior(alpha=1.0, beta=0.5)
        assert posterior.mode == posterior.mean

    def test_gamma_poisson_invalid_prior(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            bayes.gamma_poisson_update(1, prior_shape=-1.0)
