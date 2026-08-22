"""Tests for the v3.0.0 surface expansion."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from cds2 import distributions as dist
from cds2 import signals, sparse, special, stats


# ------------------------------------------------- new distributions ----
class TestExpandedDistributions:
    def test_gamma_mean_matches_shape(self) -> None:
        grid = np.linspace(1e-6, 40.0, 20001)
        mean_estimate = float(np.trapezoid(grid * dist.gamma_pdf(grid, a=3.0), grid))
        assert mean_estimate == pytest.approx(3.0, abs=1e-2)

    def test_gamma_cdf_ppf_roundtrip(self) -> None:
        quantile = dist.gamma_ppf(0.7, a=2.5)
        assert dist.gamma_cdf(quantile, a=2.5)[0] == pytest.approx(0.7)

    def test_beta_uniform_prior(self) -> None:
        assert np.allclose(dist.beta_pdf([0.4], a=1.0, b=1.0), 1.0)

    def test_beta_ppf_median_symmetric(self) -> None:
        assert dist.beta_ppf(0.5, a=3.0, b=3.0)[0] == pytest.approx(0.5)

    @pytest.mark.parametrize("shape", [0.5, 1.0, 3.0])
    def test_weibull_pdf_support(self, shape: float) -> None:
        assert dist.weibull_pdf(-1.0, c=shape)[0] == pytest.approx(0.0)
        assert dist.weibull_pdf(1.0, c=shape)[0] > 0.0

    def test_cauchy_heavy_tail(self) -> None:
        central = dist.cauchy_pdf(0.0)[0]
        far = dist.cauchy_pdf(100.0)[0]
        assert central > far > 0.0

    def test_laplace_peaks_at_location(self) -> None:
        assert dist.laplace_pdf(0.0)[0] == pytest.approx(0.5)
        assert dist.laplace_pdf(1.0, loc=1.0)[0] == pytest.approx(0.5)

    def test_gumbel_cdf_closed_form(self) -> None:
        x_value = 0.5
        expected = float(np.exp(-np.exp(-x_value)))
        assert dist.gumbel_cdf(x_value)[0] == pytest.approx(expected)

    def test_pareto_support_starts_at_scale(self) -> None:
        assert dist.pareto_pdf(0.5, b=2.0)[0] == pytest.approx(0.0)
        assert dist.pareto_pdf(2.0, b=2.0)[0] > 0.0

    def test_rayleigh_cdf_closed_form(self) -> None:
        x_value = 1.5
        expected = 1.0 - float(np.exp(-(x_value**2) / 2.0))
        assert dist.rayleigh_cdf(x_value)[0] == pytest.approx(expected)

    def test_geometric_pmf_known(self) -> None:
        masses = dist.geometric_pmf([1, 2], p=0.5)
        assert np.allclose(masses, [0.5, 0.25])

    def test_negative_binomial_sums_to_one(self) -> None:
        counts = np.arange(0, 300)
        total = float(np.sum(dist.negative_binomial_pmf(counts, n_failures=3, p=0.4)))
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_hypergeometric_no_replacement(self) -> None:
        masses = dist.hypergeometric_pmf([0, 1, 2, 3], ngood=3, nbad=4, nsample=3)
        assert float(np.sum(masses)) == pytest.approx(1.0)
        assert masses[3] > 0


# --------------------------------------------------- expanded specials ----
class TestSpecialRoundTwo:
    def test_bessel_orders_match_j0_j1(self) -> None:
        x_value = 1.3
        assert special.bessel_jv(0, x_value)[0] == pytest.approx(special.bessel_j0(x_value)[0])
        assert special.bessel_jv(1, x_value)[0] == pytest.approx(special.bessel_j1(x_value)[0])

    def test_modified_bessels_positive(self) -> None:
        values = special.bessel_iv(0, [1.0, 2.0])
        assert np.all(values[values != 0] > 0)

    def test_kv_singular_at_zero(self) -> None:
        assert np.isinf(special.bessel_kv(0, 0.0)[0])

    def test_hankel1_real_part_equals_bessel(self) -> None:
        value = 2.5
        real_hankel = special.hankel1_fn(0, value)[0]
        assert real_hankel == pytest.approx(special.bessel_j0(value)[0], abs=1e-10)

    def test_struve_values(self) -> None:
        assert special.struve_h0(0.0)[0] == pytest.approx(0.0)
        assert special.struve_h1(1.0)[0] > 0.0
        assert abs(special.struve_h1(-1.0)[0]) == pytest.approx(special.struve_h1(1.0)[0])

    def test_chebyshev_recurrence_values(self) -> None:
        x_value = 0.7
        assert special.chebyshev_tn(2, [x_value])[0] == pytest.approx(2 * x_value**2 - 1)
        assert special.chebyshev_un(1, [x_value])[0] == pytest.approx(2 * x_value)

    def test_orthogonal_polynomial_families(self) -> None:
        assert special.laguerre_ln(0, [5.0])[0] == pytest.approx(1.0)
        assert special.hermite_hn(1, [3.0])[0] == pytest.approx(6.0)
        assert special.jacobi_pnab(0, 0.5, 0.5, [0.25])[0] == pytest.approx(1.0)

    def test_spherical_harmonic_constant_m0(self) -> None:
        value = special.spherical_harmonic(0, 0, theta=0.0, phi=0.0)
        assert abs(value[0]) == pytest.approx(np.sqrt(1.0 / (4 * np.pi)))

    def test_lambert_w_solution_property(self) -> None:
        w_value = special.lambert_w(1.0)[0]
        assert w_value * np.exp(w_value) == pytest.approx(1.0)

    def test_faddeeva_w_zero_is_one(self) -> None:
        result = special.faddeeva_w(0.0, 0.0)
        assert result[0].real == pytest.approx(1.0)
        assert abs(result[0]) > 0

    def test_exponential_integral_en_two(self) -> None:
        assert special.exponential_integral_en(1, 1.0)[0] == pytest.approx(special.exp1(1.0)[0])

    def test_sine_cosine_integrals_tuple(self) -> None:
        si_value, ci_value = special.sine_cosine_integrals(1.0)
        assert si_value.size == ci_value.size == 1

    def test_binomial_coefficient(self) -> None:
        assert special.binomial_coefficient(10, 3) == 120.0


class TestSparseAndSignalAdditions:
    def test_sparse_eye_diag_kron(self) -> None:
        identity = sparse.sparse_eye(3)
        assert identity.shape == (3, 3)
        diagonal = sparse.sparse_diag([1.0, 2.0, 3.0])
        assert diagonal.diagonal().tolist() == [1.0, 2.0, 3.0]
        kron_product = sparse.sparse_kron(np.eye(2), np.array([[1.0, 2.0]]))
        assert kron_product.shape == (2, 4)

    def test_one_norm_est_identity(self) -> None:
        value = sparse.one_norm_est(sparse.sparse_eye(4))
        assert value == pytest.approx(1.0)

    def test_stft_shapes_and_coherence_range(self) -> None:
        time_values = np.arange(1024) / 128.0
        signal_a = np.sin(2 * np.pi * 10.0 * time_values)
        signal_b = np.sin(2 * np.pi * 10.0 * time_values + 0.5)
        times, freqs, zxx = signals.stft(signal_a, fs=128.0, nperseg=256)
        assert zxx.shape == (len(freqs), len(times))
        coherence = signals.coherence(signal_a, signal_b, fs=128.0, nperseg=256)
        assert float(coherence.power.max()) <= 1.0 + 1e-9


class TestStatsMatrixAdditions:
    def test_covariance_matrix_matches_numpy(self) -> None:
        data = np.random.default_rng(0).normal(size=(60, 3))
        result = stats.covariance_matrix(data)
        assert np.allclose(result, np.cov(data, rowvar=False, ddof=1))

    def test_correlation_matrix_diagonal_is_one(self) -> None:
        data = np.random.default_rng(1).normal(size=(50, 4))
        matrix = stats.correlation_matrix(data)
        assert np.allclose(np.diag(matrix), 1.0)

    def test_multivariate_normal_logpdf_standard(self) -> None:
        log_density = stats.multivariate_normal_logpdf(
            [0.0, 0.0], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]
        )
        expected = -np.log(2 * np.pi)
        assert log_density == pytest.approx(expected, rel=1e-10)

    def test_solve_cg_linear_operator_still_supported(self) -> None:
        operator = LinearOperator((2, 2), matvec=lambda v: 2.0 * v)
        result = sparse.solve_cg(operator, [1.0, 1.0])
        assert np.allclose(result.x, [0.5, 0.5], atol=1e-8)


class TestDistributionReturnPaths:
    def test_beta_cdf_symmetry(self) -> None:
        assert dist.beta_cdf(0.5, a=2.0, b=2.0)[0] == pytest.approx(0.5)

    def test_weibull_cdf_ppf_roundtrip(self) -> None:
        quantile = dist.weibull_ppf(0.8, c=1.5)
        assert dist.weibull_cdf(quantile, c=1.5)[0] == pytest.approx(0.8)

    def test_cauchy_cdf_quartiles(self) -> None:
        assert dist.cauchy_cdf(1.0)[0] == pytest.approx(0.75)

    def test_cauchy_ppf_quartile(self) -> None:
        assert dist.cauchy_ppf(0.25)[0] == pytest.approx(-1.0)

    def test_laplace_cdf_left_branch(self) -> None:
        assert dist.laplace_cdf(-2.0, scale=1.0)[0] == pytest.approx(0.5 * float(np.exp(-2.0)))

    def test_laplace_ppf_lower_half(self) -> None:
        assert dist.laplace_ppf(0.25, scale=2.0)[0] == pytest.approx(2.0 * np.log(0.5) + 0.0)

    def test_gumbel_pdf_at_zero(self) -> None:
        assert dist.gumbel_pdf(0.0)[0] == pytest.approx(float(np.exp(-1.0)))

    def test_gumbel_ppf_standard(self) -> None:
        quantile_level = 1.0 - float(np.exp(-1.0))
        expected = -float(np.log(-np.log(quantile_level)))
        assert dist.gumbel_ppf(quantile_level)[0] == pytest.approx(expected)

    def test_pareto_cdf_one_at_large(self) -> None:
        assert dist.pareto_cdf(1000.0, b=2.0)[0] > 0.9999

    def test_pareto_ppf_median(self) -> None:
        assert dist.pareto_ppf(0.5, b=2.0)[0] == pytest.approx(float(np.sqrt(2.0)))

    def test_rayleigh_pdf_peak(self) -> None:
        assert dist.rayleigh_pdf(1.0)[0] == pytest.approx(float(np.exp(-0.5)))

    def test_rayleigh_ppf_roundtrip(self) -> None:
        quantile = dist.rayleigh_ppf(0.9)
        assert dist.rayleigh_cdf(quantile)[0] == pytest.approx(0.9)

    def test_geometric_cdf(self) -> None:
        assert dist.geometric_cdf([2], p=0.5)[0] == pytest.approx(0.75)

    def test_geometric_ppf(self) -> None:
        assert dist.geometric_ppf(0.75, p=0.5)[0] == pytest.approx(2.0)

    def test_negative_binomial_cdf_matches_cumsum(self) -> None:
        counts = np.arange(0, 40)
        cumulative = dist.negative_binomial_cdf(counts, n_failures=2, p=0.6)
        assert np.allclose(cumulative, np.cumsum(dist.negative_binomial_pmf(counts, 2, 0.6)))

    def test_negative_binomial_ppf_low(self) -> None:
        assert dist.negative_binomial_ppf(0.01, n_failures=2, p=0.6)[0] == pytest.approx(0.0)

    def test_hypergeometric_cdf_starts_below_one(self) -> None:
        value = dist.hypergeometric_cdf([1], ngood=3, nbad=4, nsample=3)
        assert value[0] == pytest.approx(float(np.sum(dist.hypergeometric_pmf([0, 1], 3, 4, 3))))


class TestSparseMultiBandAndSpecialYv:
    def test_sparse_diag_with_offsets(self) -> None:
        band_matrix = sparse.sparse_diag([np.ones(3), 2.0 * np.ones(4)], offsets=[-1, 0])
        dense_view = band_matrix.toarray()
        assert np.allclose(np.diag(dense_view), 2.0)
        assert np.allclose(np.diag(dense_view, -1), 1.0)

    def test_bessel_yv_finite_for_positive_args(self) -> None:
        values = special.bessel_yv(0, [1.0, 2.0])
        assert np.all(np.isfinite(values))
