"""Tests for guided_fit.compare_models: AIC/BIC model comparison.

Reference values are recomputed independently from NumPy polynomial
least-squares fits (not from the production curve_fit path), so the
Gaussian log-likelihood, parameter counting (curve parameters + error
variance) and Akaike weights are pinned against the textbook formulas.
"""

from __future__ import annotations

import numpy as np
import pytest

from cds2.guided_fit import FitDataset, compare_models


def _linear_dataset(n: int = 24, seed: int = 5, noise: float = 0.05) -> FitDataset:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, n)
    y = 2.0 * x + 1.0 + rng.normal(scale=noise, size=n)
    return FitDataset("linear", x, y)


def _reference_aic_bic(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[float, float]:
    """Textbook Gaussian AIC/BIC from an independent polyfit solution."""
    coefficients = np.polyfit(x, y, degree)
    residual = y - np.polyval(coefficients, x)
    n = x.size
    rss = float(residual @ residual)
    log_likelihood = -0.5 * n * (np.log(2.0 * np.pi * rss / n) + 1.0)
    k = degree + 1 + 1  # curve parameters + estimated error variance
    return 2.0 * k - 2.0 * log_likelihood, k * np.log(n) - 2.0 * log_likelihood


class TestRanking:
    def test_linear_data_prefers_linear(self) -> None:
        dataset = _linear_dataset()
        results = compare_models(dataset)
        assert results[0].model == "linear"
        assert min(results, key=lambda item: item.bic).model == "linear"

    def test_linear_aic_bic_match_independent_reference(self) -> None:
        dataset = _linear_dataset()
        linear = next(item for item in compare_models(dataset) if item.model == "linear")
        expected_aic, expected_bic = _reference_aic_bic(dataset.x, dataset.y, 1)
        assert linear.aic == pytest.approx(expected_aic, abs=1e-4)
        assert linear.bic == pytest.approx(expected_bic, abs=1e-4)
        assert linear.n_params == 3
        assert linear.n_points == dataset.x.size

    def test_quadratic_data_prefers_quadratic(self) -> None:
        rng = np.random.default_rng(9)
        x = np.linspace(-3.0, 3.0, 30)
        y = x * x + rng.normal(scale=0.05, size=30)
        results = compare_models(FitDataset("bend", x, y))
        assert results[0].model == "quadratic"
        linear = next(item for item in results if item.model == "linear")
        assert linear.delta_bic > 10.0

    def test_quadratic_aic_matches_independent_reference(self) -> None:
        rng = np.random.default_rng(9)
        x = np.linspace(-3.0, 3.0, 30)
        y = x * x + rng.normal(scale=0.05, size=30)
        quadratic = next(
            item for item in compare_models(FitDataset("bend", x, y)) if item.model == "quadratic"
        )
        expected_aic, _ = _reference_aic_bic(x, y, 2)
        assert quadratic.aic == pytest.approx(expected_aic, abs=1e-4)
        assert quadratic.n_params == 4

    def test_sorted_deltas_and_weights(self) -> None:
        results = compare_models(_linear_dataset())
        aics = [item.aic for item in results]
        assert aics == sorted(aics)
        assert results[0].delta_aic == 0.0
        assert min(item.delta_bic for item in results) == 0.0
        assert all(item.delta_aic >= 0.0 and item.delta_bic >= 0.0 for item in results)
        assert sum(item.aic_weight for item in results) == pytest.approx(1.0)
        assert results[0].aic_weight == max(item.aic_weight for item in results)
        # Akaike weight formula pinned against the deltas.
        raw = np.exp(-0.5 * np.array([item.delta_aic for item in results]))
        np.testing.assert_allclose(
            [item.aic_weight for item in results], raw / raw.sum(), rtol=1e-12
        )

    def test_logistic_parameter_count_includes_variance(self) -> None:
        rng = np.random.default_rng(21)
        x = np.linspace(0.0, 10.0, 30)
        y = 2.0 + 3.0 / (1.0 + np.exp(-1.2 * (x - 5.0))) + rng.normal(scale=0.02, size=30)
        results = compare_models(FitDataset("sigmoid", x, y))
        logistic = next(item for item in results if item.model == "logistic")
        assert logistic.n_params == 5


class TestCandidates:
    def test_single_candidate_has_unit_weight(self) -> None:
        rng = np.random.default_rng(13)
        x = np.linspace(0.5, 5.0, 20)
        y = 3.0 * np.sqrt(x) + rng.normal(scale=0.02, size=20)
        results = compare_models(
            FitDataset("root", x, y),
            exclude=("linear", "quadratic", "exponential", "logistic"),
        )
        assert len(results) == 1
        assert results[0].model == "power"
        assert results[0].delta_aic == 0.0
        assert results[0].delta_bic == 0.0
        assert results[0].aic_weight == pytest.approx(1.0)

    def test_unfittable_power_is_skipped_for_nonpositive_x(self) -> None:
        rng = np.random.default_rng(15)
        x = np.linspace(-2.0, 2.0, 24)
        y = 2.0 * x + rng.normal(scale=0.05, size=24)
        results = compare_models(FitDataset("neg", x, y))
        assert "power" not in {item.model for item in results}
        assert len(results) >= 2

    def test_all_candidates_excluded_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one candidate"):
            compare_models(
                _linear_dataset(),
                exclude=("linear", "quadratic", "exponential", "power", "logistic"),
            )

    def test_no_fittable_candidate_raises_runtime_error(self) -> None:
        rng = np.random.default_rng(17)
        x = np.linspace(-2.0, -0.5, 20)
        y = rng.normal(size=20)
        with pytest.raises(RuntimeError, match="no supported model"):
            compare_models(
                FitDataset("neg", x, y), exclude=("linear", "quadratic", "exponential", "logistic")
            )

    def test_constant_data_perfect_fits_are_skipped(self) -> None:
        x = np.linspace(0.0, 5.0, 20)
        y = np.full(20, 3.0)
        results = compare_models(FitDataset("flat", x, y))
        models = {item.model for item in results}
        # linear and logistic reproduce the constant exactly (rmse == 0):
        # their Gaussian likelihood is unbounded, so they are skipped rather
        # than ranked first with a spuriously perfect AIC. power needs x > 0.
        assert "linear" not in models
        assert "logistic" not in models
        assert "power" not in models


class TestPolicies:
    def test_weighted_likelihood_reference_with_sigma(self) -> None:
        dataset = _linear_dataset()
        sigma = np.full(dataset.x.size, 0.5)
        weighted = FitDataset("weighted", dataset.x, dataset.y, sigma)
        linear = next(item for item in compare_models(weighted) if item.model == "linear")
        coefficients = np.polyfit(dataset.x, dataset.y, 1)
        residual = dataset.y - np.polyval(coefficients, dataset.x)
        z = residual / sigma
        expected_log_likelihood = -0.5 * float(np.sum(z * z + np.log(2.0 * np.pi * sigma**2)))
        assert linear.log_likelihood == pytest.approx(expected_log_likelihood, abs=1e-4)
        assert linear.aic == pytest.approx(2.0 * 3 - 2.0 * expected_log_likelihood, abs=1e-4)
        # rmse stays in data units, not weighted units.
        assert linear.rmse == pytest.approx(float(np.sqrt(np.mean(residual**2))), abs=1e-6)

    def test_missing_policy_drop_reduces_points(self) -> None:
        dataset = _linear_dataset()
        y = dataset.y.copy()
        y[3] = np.nan
        dirty = FitDataset("dirty", dataset.x, y)
        results = compare_models(dirty, missing_policy="drop")
        assert all(item.n_points == dataset.x.size - 1 for item in results)

    def test_outlier_exclude_refits_on_retained_points(self) -> None:
        dataset = _linear_dataset(n=24)
        y = dataset.y.copy()
        y[5] += 50.0
        y[17] -= 50.0
        dirty = FitDataset("outliers", dataset.x, y)
        results = compare_models(dirty, outlier_policy="exclude")
        assert results
        assert all(item.n_points <= dataset.x.size for item in results)
        assert results[0].n_points < dataset.x.size

    def test_outlier_exclude_keeps_fit_when_too_few_points_remain(self) -> None:
        # Five points, two gross outliers: the linear fit flags both, leaving
        # three retained points (< 4), so the original fit must be kept.
        x = np.arange(5.0)
        y = np.array([0.0, 1.0, 2.0, 100.0, -100.0])
        results = compare_models(FitDataset("tiny", x, y), outlier_policy="exclude")
        assert results
        assert any(item.n_points == 5 for item in results)

    def test_outlier_exclude_on_clean_data_keeps_all_points(self) -> None:
        results = compare_models(_linear_dataset(noise=0.01), outlier_policy="exclude")
        assert all(item.n_points == 24 for item in results)

    def test_invalid_policies_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing_policy"):
            compare_models(_linear_dataset(), missing_policy="ask")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="outlier_policy"):
            compare_models(_linear_dataset(), outlier_policy="ask")  # type: ignore[arg-type]
