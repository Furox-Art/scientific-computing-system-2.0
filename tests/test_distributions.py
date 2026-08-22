"""Tests for cds2.distributions."""

import numpy as np
import pytest

from cds2 import distributions as dist


class TestStudentT:
    def test_symmetric_about_zero(self) -> None:
        assert float(dist.student_t_cdf(0.0, 10)[0]) == pytest.approx(0.5)
        assert dist.student_t_cdf(-2.0, 10)[0] < 0.5 < dist.student_t_cdf(2.0, 10)[0]

    def test_ppf_roundtrip(self) -> None:
        quantile = dist.student_t_ppf(0.975, 12)
        assert float(dist.student_t_cdf(quantile, 12)[0]) == pytest.approx(0.975)

    def test_pdf_integrates_to_one(self) -> None:
        grid = np.linspace(-40.0, 40.0, 4001)
        area = float(np.trapezoid(dist.student_t_pdf(grid, 3), grid))
        assert area == pytest.approx(1.0, abs=1e-3)


class TestChiSquareAndF:
    def test_chi2_mean_is_df(self) -> None:
        df = 6.0
        grid = np.linspace(0.0, 60.0, 20001)
        mean_estimate = float(np.trapezoid(grid * dist.chi2_pdf(grid, df), grid))
        assert mean_estimate == pytest.approx(df, abs=1e-2)

    def test_chi2_ppf_critical_value(self) -> None:
        assert dist.chi2_ppf(0.95, 1)[0] == pytest.approx(3.841458820694124, rel=1e-6)

    def test_f_cdf_bounds(self) -> None:
        assert float(dist.f_cdf(0.0, 5, 10)[0]) >= 0.0
        assert float(dist.f_cdf(1000.0, 5, 10)[0]) == pytest.approx(1.0)


class TestContinuousDistributions:
    def test_exponential_cdf_closed_form(self) -> None:
        x = np.array([0.5, 1.0, 2.0])
        expected = 1.0 - np.exp(-x / 2.0)
        assert np.allclose(dist.exponential_cdf(x, scale=2.0), expected)

    def test_exponential_ppf_inverse(self) -> None:
        quantile = dist.exponential_ppf(0.6321205588285577, scale=1.0)
        assert quantile[0] == pytest.approx(1.0, rel=1e-9)

    def test_uniform_flat_density(self) -> None:
        inside = dist.uniform_pdf([0.25, 0.75], loc=0.0, scale=1.0)
        outside = dist.uniform_pdf([-1.0, 2.0], loc=0.0, scale=1.0)
        assert np.allclose(inside, 1.0)
        assert np.allclose(outside, 0.0)

    def test_uniform_ppf_endpoints(self) -> None:
        values = dist.uniform_ppf([0.0, 1.0])
        assert values[0] == pytest.approx(0.0)
        assert values[1] == pytest.approx(1.0)

    def test_lognormal_median(self) -> None:
        median = dist.lognormal_cdf(1.0, sigma=0.5, scale=1.0)
        assert median[0] == pytest.approx(0.5)


class TestDiscreteDistributions:
    def test_poisson_pmf_sums_to_one(self) -> None:
        counts = np.arange(0, 40)
        total = float(np.sum(dist.poisson_pmf(counts, mu=4.5)))
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_poisson_cdf_matches_pmf_cumsum(self) -> None:
        counts = np.arange(0, 15)
        cumulative = dist.poisson_cdf(counts, mu=3.0)
        assert np.allclose(cumulative, np.cumsum(dist.poisson_pmf(counts, mu=3.0)))

    def test_binomial_pmf_known_values(self) -> None:
        masses = dist.binomial_pmf([0, 1, 2], n=2, p=0.5)
        assert np.allclose(masses, [0.25, 0.5, 0.25])

    def test_binomial_cdf_and_ppf(self) -> None:
        assert dist.binomial_cdf(1, n=4, p=0.5)[0] == pytest.approx(0.3125)
        assert dist.binomial_ppf(0.5, n=4, p=0.5)[0] == pytest.approx(2.0)


class TestRemainingDistributionPaths:
    def test_chi2_cdf_median_region(self) -> None:
        assert dist.chi2_cdf(6.0, df=6)[0] == pytest.approx(0.5768, rel=1e-4)

    def test_f_pdf_peak_shape(self) -> None:
        peak = dist.f_pdf(1.0, dfn=5, dfd=10)[0]
        tail = dist.f_pdf(50.0, dfn=5, dfd=10)[0]
        assert peak > tail > 0

    def test_f_ppf_roundtrip(self) -> None:
        quantile = dist.f_ppf(0.95, dfn=4, dfd=8)
        assert dist.f_cdf(quantile, dfn=4, dfd=8)[0] == pytest.approx(0.95)

    def test_exponential_pdf_at_zero(self) -> None:
        assert dist.exponential_pdf(0.0, scale=2.0)[0] == pytest.approx(0.5)

    def test_uniform_cdf_linear(self) -> None:
        values = np.array([-1.0, 0.5, 3.0])
        expected = np.clip(values, 0.0, 1.0)
        assert np.allclose(dist.uniform_cdf(values), expected)

    def test_lognormal_pdf_positive_support(self) -> None:
        assert dist.lognormal_pdf(-1.0, sigma=1.0)[0] == pytest.approx(0.0)
        positive = dist.lognormal_pdf(1.0, sigma=1.0)[0]
        assert positive == pytest.approx(0.3989422804014327, rel=1e-9)

    def test_lognormal_ppf_median(self) -> None:
        assert dist.lognormal_ppf(0.5, sigma=0.5, scale=1.0)[0] == pytest.approx(1.0)

    def test_poisson_ppf_low_tail(self) -> None:
        assert dist.poisson_ppf(0.001, mu=3.0)[0] == pytest.approx(0.0)
