"""Tests for cds2.stats."""

import numpy as np
import pytest

from cds2 import stats


class TestDescribe:
    def test_known_sample(self) -> None:
        result = stats.describe([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.n == 5
        assert result.mean == pytest.approx(3.0)
        assert result.median == pytest.approx(3.0)
        assert result.minimum == pytest.approx(1.0)
        assert result.maximum == pytest.approx(5.0)

    def test_std_matches_numpy(self) -> None:
        values = [1.5, 2.5, 7.0]
        assert stats.describe(values).std == pytest.approx(np.std(values, ddof=1))


class TestTTests:
    def test_one_sample_significant(self) -> None:
        sample = [10.1, 9.9, 10.0, 10.2, 10.0]
        result = stats.t_test(sample, popmean=8.0)
        assert result.p_value < 0.001

    def test_one_sample_null_true(self) -> None:
        rng_values = np.random.default_rng(42).normal(loc=5.0, scale=1.0, size=200)
        assert stats.t_test(rng_values, popmean=5.0).p_value > 0.05

    def test_independent_shifted(self) -> None:
        result = stats.independent_t_test([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
        assert result.statistic < -5
        assert result.p_value < 0.01

    def test_welch_differs_from_pooled(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [10.0, 20.0, 30.0, 40.0]
        pooled = stats.independent_t_test(a, b, equal_var=True)
        welch = stats.independent_t_test(a, b, equal_var=False)
        assert pooled.statistic != pytest.approx(welch.statistic)

    def test_paired_detects_change(self) -> None:
        before = [5.0, 6.0, 7.0, 8.0]
        after = [7.0, 8.0, 9.0, 10.0]
        assert stats.paired_t_test(before, after).p_value < 0.001


class TestGroupTests:
    def test_anova_significant(self) -> None:
        groups = [[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]]
        result = stats.anova(*groups)
        assert result.statistic > 20
        assert result.p_value < 0.05

    def test_kruskal(self) -> None:
        result = stats.kruskal_wallis([1, 2, 3], [10, 11, 12])
        assert result.p_value < 0.05

    def test_mann_whitney(self) -> None:
        result = stats.mann_whitney_u([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert result.statistic == 0
        assert result.p_value < 0.01

    def test_wilcoxon(self) -> None:
        before = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        after = [value + 1.0 for value in before]
        result = stats.wilcoxon_signed_rank(before, after)
        assert result.statistic == pytest.approx(0.0)
        assert result.p_value < 0.05

    def test_levene_equal_variances(self) -> None:
        a = np.random.default_rng(1).normal(scale=1.0, size=100)
        b = np.random.default_rng(2).normal(scale=1.0, size=100)
        assert stats.levene_test(a, b).p_value > 0.01

    def test_single_group_raises(self) -> None:
        with pytest.raises(ValueError, match="two"):
            stats.anova([1.0])


class TestCorrelations:
    def test_pearson_perfect_linear(self) -> None:
        result = stats.pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8])
        assert result.r == pytest.approx(1.0)

    def test_spearman_monotone_nonlinear(self) -> None:
        result = stats.spearman_correlation([1, 2, 3, 4], [1, 8, 27, 64])
        assert result.r == pytest.approx(1.0)

    def test_kendall_positive(self) -> None:
        assert stats.kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]).r > 0.99


class TestCategorical:
    def test_chi_square_associated(self) -> None:
        table = [[30, 10], [5, 25]]
        result = stats.chi_square_independence(table)
        assert result.p_value < 0.01

    def test_cramers_v_bounds(self) -> None:
        value = stats.cramers_v([[30, 10], [5, 25]])
        assert 0.0 <= value <= 1.0

    def test_cramers_v_rejects_vector(self) -> None:
        with pytest.raises(ValueError):
            stats.cramers_v([[50, 50]])


class TestEffectSizes:
    def test_cohens_d_sign_and_magnitude(self) -> None:
        values_a = np.random.default_rng(7).normal(10.0, 2.0, 100)
        values_b = np.random.default_rng(8).normal(12.0, 2.0, 100)
        d = stats.cohens_d(values_a, values_b)
        assert d < -1.0

    def test_cohens_d_zero_for_identical_means(self) -> None:
        values = [4.0, 5.0, 6.0, 7.0]
        assert stats.cohens_d(values, list(values)) == pytest.approx(0.0)

    def test_eta_squared_monotone(self) -> None:
        small = stats.eta_squared_from_f(1.0, 2, 27)
        large = stats.eta_squared_from_f(10.0, 2, 27)
        assert 0.0 < small < large < 1.0


class TestHelpers:
    def test_normality_gaussian_passes(self) -> None:
        data = np.random.default_rng(11).normal(size=300)
        assert stats.normality_test(data).p_value > 0.01

    def test_percentile_median(self) -> None:
        assert stats.percentile([1, 2, 3, 4, 5], 50) == pytest.approx(3.0)

    def test_percentile_multiple(self) -> None:
        values = stats.percentile(range(101), [25, 75])
        assert values == [pytest.approx(25.0), pytest.approx(75.0)]

    def test_z_scores_standardized(self) -> None:
        z = stats.z_scores([2.0, 4.0, 6.0, 8.0])
        assert abs(z.mean()) < 1e-12
        assert z.std(ddof=1) == pytest.approx(1.0)

    def test_z_scores_constant_raises(self) -> None:
        with pytest.raises(ValueError, match="constant"):
            stats.z_scores([5.0, 5.0, 5.0])

    def test_norm_functions_roundtrip(self) -> None:
        assert stats.norm_cdf(0.0) == pytest.approx(0.5)
        assert stats.norm_pdf(0.0) == pytest.approx(0.3989422804014327)
        assert stats.norm_ppf(0.975) == pytest.approx(1.959963984540054, rel=1e-6)
