"""Tests for cds2.integrate."""

import numpy as np
import pytest

from cds2 import integrate


class TestQuad:
    def test_polynomial_exact(self) -> None:
        result = integrate.quad(lambda x: 3.0 * x**2, 0.0, 2.0)
        assert result.value == pytest.approx(8.0, rel=1e-10)
        assert result.error < 1e-8

    def test_sin_half_period(self) -> None:
        assert integrate.quad(np.sin, 0.0, np.pi).value == pytest.approx(2.0)


class TestMultidimensional:
    def test_2d_box_volume(self) -> None:
        result = integrate.integrate_2d(lambda x, y: 1.0, 0.0, 2.0, 0.0, 3.0)
        assert result.value == pytest.approx(6.0, rel=1e-8)

    def test_2d_product_function(self) -> None:
        result = integrate.integrate_2d(lambda x, y: x * y, 0.0, 1.0, 0.0, 1.0)
        assert result.value == pytest.approx(0.25, rel=1e-8)

    def test_3d_unit_cube(self) -> None:
        result = integrate.integrate_3d(lambda x, y, z: x + y + z, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        assert result.value == pytest.approx(1.5, rel=1e-8)


class TestNewtonianRules:
    def test_trapezoid_linear_exact(self) -> None:
        y = np.linspace(0.0, 4.0, 21)
        assert integrate.trapezoid(y, dx=y[1] - y[0]) == pytest.approx(8.0, rel=1e-12)

    def test_simpson_quadratic_near_exact(self) -> None:
        x = np.linspace(0.0, 2.0, 101)
        assert integrate.simpson(x**2, x=x) == pytest.approx(8.0 / 3.0, rel=1e-10)

    def test_cumulative_shape_and_endpoints(self) -> None:
        y = np.ones(50)
        running = integrate.cumulative_trapezoid(y, dx=0.5)
        assert len(running) == 50
        assert running[0] == pytest.approx(0.0)
        assert running[-1] == pytest.approx(24.5)


class TestOdeSolvers:
    def test_exponential_growth(self) -> None:
        result = integrate.solve_ivp(lambda t, y: y, (0.0, 1.0), [1.0])
        assert result.success
        assert result.y[0][-1] == pytest.approx(np.e, rel=1e-4)

    def test_backward_integration(self) -> None:
        result = integrate.solve_ivp(lambda t, y: y, (1.0, 0.0), [np.e], rtol=1e-9, atol=1e-12)
        assert result.success
        assert result.y[0][-1] == pytest.approx(1.0, rel=1e-6)

    def test_harmonic_oscillator_energy(self) -> None:
        rhs = lambda _t, state: [state[1], -state[0]]  # noqa: E731
        result = integrate.solve_ivp(rhs, (0.0, 2 * np.pi), [1.0, 0.0], max_step=0.01)
        amplitude = result.y[0][-1]
        assert amplitude == pytest.approx(1.0, abs=1e-3)

    def test_t_eval_grid(self) -> None:
        grid = np.linspace(0.0, 1.0, 11)
        result = integrate.solve_ivp(lambda t, y: y, (0.0, 1.0), [1.0], t_eval=grid)
        expected = np.exp(grid)
        assert np.allclose(result.y[0], expected, rtol=1e-4)


class TestOdeEvents:
    def test_terminal_event_at_exact_time(self) -> None:
        def hit_one(_t: float, state: np.ndarray) -> float:
            return state[0] - 1.0

        hit_one.terminal = True
        result = integrate.solve_ivp(lambda t, y: y, (0.0, 5.0), [0.1], events=[hit_one])
        assert result.t_events is not None and result.y_events is not None
        assert len(result.t_events[0]) == 1
        assert result.t_events[0][0] == pytest.approx(np.log(10.0), rel=1e-4)
        assert result.y_events[0][0][0] == pytest.approx(1.0, rel=1e-8)

    def test_nonterminal_event_records_all_crossings(self) -> None:
        def cross_zero(_t: float, state: np.ndarray) -> float:
            return state[1]

        rhs = lambda _t, state: [state[1], -state[0]]  # noqa: E731
        result = integrate.solve_ivp(
            rhs,
            (0.0, 4 * np.pi),
            [1.0, 0.0],
            events=[cross_zero],
            max_step=0.01,
        )
        crossings = len(result.t_events[0])
        assert crossings >= 3

    def test_no_events_keeps_fields_none(self) -> None:
        result = integrate.solve_ivp(lambda t, y: y, (0.0, 1.0), [1.0])
        assert result.t_events is None
        assert result.y_events is None


class TestStiffSolvers:
    def test_bdf_solves_stiff_decay_system(self) -> None:
        def stiff(t: float, y: np.ndarray) -> list[float]:
            return [-1000.0 * (y[0] - np.exp(-t)) - np.exp(-t)]

        result = integrate.solve_ivp(
            lambda t, y: stiff(t, y),
            (0.0, 1.0),
            [1.0],
            method="BDF",
            rtol=1e-6,
        )
        assert result.success
        # u = y - exp(-t) satisfies u' = -1000u with u(0) = 0, so y = exp(-t)
        assert result.y[0][-1] == pytest.approx(np.exp(-1.0), rel=1e-4)


class TestBVPSolver:
    def test_parabola_unique_solution(self) -> None:
        mesh = np.linspace(0.0, 1.0, 21)
        solution = integrate.solve_bvp(
            lambda _t, y: np.vstack([y[1], -2.0 * np.ones_like(y[1])]),
            lambda ya, yb: np.array([ya[0], yb[0]]),
            mesh,
            np.zeros((2, mesh.size)),
        )
        assert solution.success
        expected = mesh * (1.0 - mesh)
        assert float(np.max(np.abs(solution.y[0] - expected))) < 1e-7

    def test_linear_gradient_problem(self) -> None:
        mesh = np.linspace(0.0, 1.0, 15)
        solution = integrate.solve_bvp(
            lambda _t, y: np.vstack([y[1], np.zeros_like(y[1])]),
            lambda ya, yb: np.array([ya[0] - 2.0, yb[0] - 5.0]),
            mesh,
            np.zeros((2, mesh.size)),
        )
        assert solution.success
        assert np.allclose(solution.y[0], 2.0 + 3.0 * solution.x, atol=1e-7)
