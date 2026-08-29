"""Tests for cds2.bayesopt."""

import numpy as np
import pytest

from cds2.bayesopt import (
    GaussianProcess,
    OptimizeResult,
    bayes_opt,
    expected_improvement,
    upper_confidence_bound,
)


class TestGaussianProcess:
    def test_fit_predict_at_train(self) -> None:
        x_train = np.array([[0.0], [1.0], [2.0]])
        y_train = np.array([0.0, 1.0, 4.0])
        gp = GaussianProcess(length_scale=1.0, sigma_f=1.0, noise=1e-6)
        gp.fit(x_train, y_train)
        mu, std = gp.predict(x_train)
        assert np.allclose(mu, y_train, atol=1e-4)
        assert np.all(std < 0.1)

    def test_predict_shape(self) -> None:
        rng = np.random.default_rng(0)
        x_train = rng.uniform(-1, 1, size=(8, 2))
        y_train = np.sin(x_train[:, 0]) + np.cos(x_train[:, 1])
        gp = GaussianProcess(length_scale=1.0)
        gp.fit(x_train, y_train)
        x_star = rng.uniform(-1, 1, size=(5, 2))
        mu, std = gp.predict(x_star)
        assert mu.shape == (5,)
        assert std.shape == (5,)
        assert np.all(std >= 0)

    def test_not_fitted_raises(self) -> None:
        gp = GaussianProcess()
        with pytest.raises(RuntimeError, match="not fitted"):
            gp.predict([[0.0]])


class TestAcquisition:
    def test_expected_improvement_zero_sigma(self) -> None:
        mu = np.array([0.0, 1.0])
        sigma = np.array([0.0, 0.0])
        ei = expected_improvement(mu, sigma, best_f=0.0)
        assert np.allclose(ei, 0.0)

    def test_expected_improvement_positive(self) -> None:
        # mu lower than best => positive improvement
        mu = np.array([0.0])
        sigma = np.array([1.0])
        ei = expected_improvement(mu, sigma, best_f=1.0)
        assert float(ei[0]) > 0.0
        # mu much larger than best => near zero
        ei2 = expected_improvement(np.array([10.0]), np.array([0.5]), best_f=0.0)
        assert float(ei2[0]) < 1e-4

    def test_upper_confidence_bound(self) -> None:
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5, 1.0])
        ucb = upper_confidence_bound(mu, sigma, kappa=2.0)
        assert ucb[0] == pytest.approx(1.0 + 2.0 * 0.5)
        assert ucb[1] == pytest.approx(2.0 + 2.0 * 1.0)
        # larger sigma -> larger ucb when kappa>0
        ucb2 = upper_confidence_bound(np.array([0.0]), np.array([2.0]), kappa=1.0)
        ucb3 = upper_confidence_bound(np.array([0.0]), np.array([1.0]), kappa=1.0)
        assert float(ucb2[0]) > float(ucb3[0])


class TestBayesOpt:
    def test_finds_minimum_1d_quadratic_ei(self) -> None:
        def quad(x: np.ndarray) -> float:
            return float((x[0] - 2.0) ** 2)

        result = bayes_opt(quad, bounds=[(0.0, 4.0)], n_init=5, n_iter=20, acquisition="ei", seed=0)
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (1,)
        assert result.fun == pytest.approx(0.0, abs=0.05)
        assert abs(float(result.x[0]) - 2.0) < 0.2
        assert result.success
        assert result.n_iterations == 20
        assert result.xs.shape[0] == 25
        assert result.ys.shape[0] == 25

    def test_finds_minimum_1d_quadratic_ucb(self) -> None:
        def quad(x: np.ndarray) -> float:
            return float((x[0] + 1.5) ** 2)

        result = bayes_opt(
            quad, bounds=[(-4.0, 2.0)], n_init=5, n_iter=20, acquisition="ucb", seed=1
        )
        assert result.fun == pytest.approx(0.0, abs=0.1)
        assert abs(float(result.x[0]) + 1.5) < 0.3

    def test_bounds_respected(self) -> None:
        def sphere(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        result = bayes_opt(sphere, bounds=bounds, n_init=4, n_iter=10, acquisition="ei", seed=2)
        assert np.all(result.xs >= -1.0 - 1e-9)
        assert np.all(result.xs <= 1.0 + 1e-9)
        assert np.all(result.x >= -1.0 - 1e-9)
        assert np.all(result.x <= 1.0 + 1e-9)

    def test_reproducible_with_seed(self) -> None:
        def f(x: np.ndarray) -> float:
            return float(np.sin(3 * x[0]) + 0.1 * x[0] ** 2)

        r1 = bayes_opt(f, bounds=[(-5.0, 5.0)], n_init=5, n_iter=10, acquisition="ei", seed=42)
        r2 = bayes_opt(f, bounds=[(-5.0, 5.0)], n_init=5, n_iter=10, acquisition="ei", seed=42)
        assert np.allclose(r1.x, r2.x)
        assert r1.fun == pytest.approx(r2.fun)
        assert np.allclose(r1.xs, r2.xs)
        assert np.allclose(r1.ys, r2.ys)
