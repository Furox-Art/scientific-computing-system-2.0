"""Tests for the Metropolis-Hastings sampler."""

import numpy as np
import pytest

from cds2.montecarlo import metropolis_hastings


class TestMetropolisHastings:
    def test_recovers_gaussian_moments(self) -> None:
        result = metropolis_hastings(
            lambda v: -0.5 * ((v[0] - 3.0) / 2.0) ** 2,
            initial=[0.0],
            n_samples=20_000,
            burn_in=2_000,
            proposal_scale=1.5,
            seed=42,
        )
        assert result.samples.shape == (20_000, 1)
        assert float(np.mean(result.samples[:, 0])) == pytest.approx(3.0, abs=0.1)
        assert float(np.std(result.samples[:, 0])) == pytest.approx(2.0, abs=0.1)

    def test_acceptance_rate_sane(self) -> None:
        result = metropolis_hastings(
            lambda v: -0.5 * v[0] ** 2,
            [0.0],
            n_samples=2000,
            proposal_scale=1.0,
            seed=1,
        )
        assert 0.1 < result.acceptance_rate <= 1.0

    def test_seeded_reproducible(self) -> None:
        def target(v):
            return -abs(v[0])

        first = metropolis_hastings(target, [0.0], n_samples=500, seed=7)
        second = metropolis_hastings(target, [0.0], n_samples=500, seed=7)
        assert np.array_equal(first.samples, second.samples)
        assert first.acceptance_rate == second.acceptance_rate

    def test_burn_in_and_thinning_control_output_size(self) -> None:
        result = metropolis_hastings(
            lambda v: -0.5 * v[0] ** 2,
            [0.0],
            n_samples=100,
            burn_in=300,
            thin=4,
            seed=3,
        )
        assert result.samples.shape[0] == 100

    def test_vector_state_supported(self) -> None:
        result = metropolis_hastings(
            lambda v: -0.5 * (v[0] ** 2 + v[1] ** 2),
            [1.0, -1.0],
            n_samples=5000,
            seed=5,
        )
        assert result.samples.shape == (5000, 2)
        assert abs(float(np.mean(result.samples))) < 0.15

    def test_invalid_arguments_raise(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            metropolis_hastings(lambda v: 0.0, [0.0], n_samples=0, thin=-1)
