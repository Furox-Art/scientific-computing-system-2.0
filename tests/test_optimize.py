"""Tests for cds2.optimize."""

import numpy as np
import pytest

from cds2 import optimize


class TestMinimize:
    def test_quadratic_bowl(self) -> None:
        result = optimize.minimize(lambda v: (v[0] - 2) ** 2 + (v[1] + 1) ** 2, x0=[0.0, 0.0])
        assert np.allclose(result.x, [2.0, -1.0], atol=1e-4)
        assert result.success
        assert result.fun < 1e-10

    def test_scalar_bounded(self) -> None:
        result = optimize.minimize_scalar(lambda x: (x - 3.5) ** 2, bounds=(0.0, 10.0))
        assert result.x == pytest.approx(3.5, abs=1e-3)

    def test_nelder_mead_method_switch(self) -> None:
        result = optimize.minimize(lambda v: (v[0] + 5) ** 2, x0=[0.0], method="Nelder-Mead")
        assert result.x[0] == pytest.approx(-5.0, abs=1e-4)


class TestRoots:
    def test_brentq_sqrt_two(self) -> None:
        root = optimize.find_root_scalar(lambda x: x * x - 2.0, 1.0, 2.0)
        assert root == pytest.approx(np.sqrt(2.0))

    def test_newton_with_derivative(self) -> None:
        root = optimize.newton_root(lambda x: x**2 - 9.0, x0=1.0, fprime=lambda x: 2 * x)
        assert root == pytest.approx(3.0)

    def test_system_root(self) -> None:
        system = lambda v: [v[0] + v[1] - 3.0, v[0] - v[1] - 1.0]  # noqa: E731
        result = optimize.root(system, x0=[0.0, 0.0])
        assert result.success
        assert np.allclose(result.x, [2.0, 1.0])


class TestLinprog:
    def test_simple_lp(self) -> None:
        result = optimize.linprog(
            c=[-1.0, -2.0],
            A_ub=[[1.0, 1.0]],
            b_ub=[4.0],
            bounds=[(0.0, None), (0.0, None)],
        )
        assert result.success
        assert result.fun == pytest.approx(-8.0)

    def test_infeasible_lp(self) -> None:
        result = optimize.linprog(
            c=[1.0],
            A_ub=[[1.0]],
            b_ub=[-10.0],
            bounds=[(0.0, None)],
        )
        assert not result.success


class TestLeastSquaresAndFit:
    def test_least_squares_linear_params(self) -> None:
        residuals = lambda p: p[0] * np.array([1.0, 2.0, 3.0]) - np.array([2.0, 4.0, 6.0])  # noqa: E731
        result = optimize.least_squares(residuals, x0=[0.0])
        assert result.x[0] == pytest.approx(2.0, rel=1e-6)
        assert result.success

    def test_curve_fit_exponential(self) -> None:
        x = np.linspace(0.0, 2.0, 25)
        y = 3.0 * np.exp(-1.7 * x)

        def model(t, amp, decay):
            return amp * np.exp(-decay * t)

        fit = optimize.curve_fit(model, x, y, p0=[1.0, 1.0])
        assert fit.params[0] == pytest.approx(3.0, rel=1e-4)
        assert fit.params[1] == pytest.approx(1.7, rel=1e-3)


class TestDifferentialEvolution:
    def test_finds_global_minimum_of_bowl(self) -> None:
        result = optimize.differential_evolution(
            lambda v: (v[0] - 3.0) ** 2 + (v[1] + 2.0) ** 2 + 7.0,
            bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            seed=1,
            maxiter=200,
        )
        assert np.allclose(result.x, [3.0, -2.0], atol=1e-4)
        assert result.fun == pytest.approx(7.0, abs=1e-8)
        assert result.success
        assert result.n_evaluations > 0

    def test_beats_local_minimum_in_multimodal(self) -> None:
        def bumpy(v):
            return np.sin(3.0 * v[0]) + 0.1 * v[0] ** 2

        result = optimize.differential_evolution(
            bumpy,
            bounds=[(-5.0, 5.0)],
            seed=2,
            maxiter=300,
        )
        grid = np.linspace(-5.0, 5.0, 4001)
        reference = float(np.min(bumpy(grid.reshape(-1, 1))))
        assert result.fun <= reference + 1e-3
