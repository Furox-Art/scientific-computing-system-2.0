"""Tests for bootstrap and permutation-based inference."""

import numpy as np
import pytest

from cds2 import stats


class TestBootstrapCI:
    def test_interval_covers_true_mean(self) -> None:
        sample = np.random.default_rng(0).normal(loc=10.0, scale=2.0, size=400)
        result = stats.bootstrap_ci(sample, seed=1)
        assert result.ci_low < 10.0 < result.ci_high
        assert result.estimate == pytest.approx(float(np.mean(sample)))

    def test_standard_error_matches_theory(self) -> None:
        sample = np.random.default_rng(2).normal(scale=4.0, size=900)
        theoretical = 4.0 / np.sqrt(900)
        result = stats.bootstrap_ci(sample, n_resamples=5000, seed=3)
        assert result.standard_error == pytest.approx(theoretical, rel=0.25)

    def test_custom_statistic_median(self) -> None:
        sample = np.arange(1.0, 101.0)
        result = stats.bootstrap_ci(sample, statistic=np.median, n_resamples=2000, seed=5)
        assert result.ci_low <= 50.5 <= result.ci_high

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            stats.bootstrap_ci([1.0, 2.0], confidence=1.5)

    def test_seeded_reproducible(self) -> None:
        data = np.random.default_rng(7).normal(size=120)
        first = stats.bootstrap_ci(data, n_resamples=800, seed=11)
        second = stats.bootstrap_ci(data, n_resamples=800, seed=11)
        assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


class TestPermutationTest:
    def test_detects_shift(self) -> None:
        group_a = np.random.default_rng(0).normal(0.0, 1.0, size=60)
        group_b = np.random.default_rng(1).normal(0.8, 1.0, size=60)
        result = stats.permutation_test(group_a, group_b, n_permutations=4000, seed=2)
        assert result.p_value < 0.01
        assert result.statistic == pytest.approx(float(group_a.mean() - group_b.mean()))

    def test_identical_distributions_not_rejected(self) -> None:
        values = np.random.default_rng(3).normal(size=50)
        result = stats.permutation_test(values, values.copy(), n_permutations=2000, seed=4)
        assert result.p_value > 0.3

    def test_p_value_floor_with_plus_one_correction(self) -> None:
        left = np.zeros(30)
        right = np.ones(30)
        result = stats.permutation_test(left, right, n_permutations=999, seed=6)
        assert result.p_value == pytest.approx(1.0 / 1000.0)

    def test_symmetric_in_group_order(self) -> None:
        group_a = np.random.default_rng(8).normal(size=40)
        group_b = np.random.default_rng(9).normal(0.5, size=40)
        forward = stats.permutation_test(group_a, group_b, n_permutations=3000, seed=10)
        backward = stats.permutation_test(group_b, group_a, n_permutations=3000, seed=10)
        assert abs(forward.statistic + backward.statistic) < 1e-12
        assert forward.p_value == pytest.approx(backward.p_value, abs=0.01)
