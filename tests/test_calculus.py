"""Tests for cds2.calculus."""

import numpy as np
import pytest

from cds2 import calculus


class TestDerivative:
    def test_cubic_exact(self) -> None:
        value = calculus.derivative(lambda x: x**3, 2.0)
        assert value == pytest.approx(12.0, rel=1e-6)

    def test_sin(self) -> None:
        value = calculus.derivative(np.sin, 0.7)
        assert value == pytest.approx(np.cos(0.7), rel=1e-7)

    def test_forward_and_backward_less_accurate_but_close(self) -> None:
        forward = calculus.derivative(lambda x: np.exp(x), 1.0, method="forward")
        backward = calculus.derivative(lambda x: np.exp(x), 1.0, method="backward")
        assert forward == pytest.approx(np.e, rel=1e-4)
        assert backward == pytest.approx(np.e, rel=1e-4)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            calculus.derivative(np.sin, 1.0, method="spectral")

    def test_explicit_step(self) -> None:
        value = calculus.derivative(lambda x: x**2, 3.0, step=1e-5)
        assert value == pytest.approx(6.0, rel=1e-8)


class TestComplexStep:
    def test_machine_precision_gradient(self) -> None:
        gradient = calculus.complex_step_gradient(lambda v: np.sum(v**3 + 2.0 * v), [2.0, -1.0])
        assert gradient[0] == pytest.approx(14.0, rel=1e-12)
        assert gradient[1] == pytest.approx(5.0, rel=1e-12)


class TestJacobian:
    def test_linear_map_exact(self) -> None:
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = calculus.jacobian(lambda v: matrix @ v, [5.0, 6.0])
        assert np.allclose(result, matrix)

    def test_nonlinear_shape_and_values(self) -> None:
        def field(v):
            return np.array([v[0] ** 2, np.sin(v[1])])

        result = calculus.jacobian(field, [3.0, 0.5])
        expected = np.array([[6.0, 0.0], [0.0, np.cos(0.5)]])
        assert result.shape == (2, 2)
        assert np.allclose(result, expected, rtol=1e-6)

    def test_scalar_output_becomes_row_vector(self) -> None:
        result = calculus.jacobian(lambda v: np.dot(v, [1.0, 10.0]), [1.0, 2.0])
        assert np.allclose(result, [[1.0, 10.0]])


class TestHessian:
    def test_diagonal_quadratic(self) -> None:
        result = calculus.hessian(
            lambda v: 3.0 * v[0] ** 2 + 5.0 * v[1] ** 2, [1.0, 2.0], step=1e-4
        )
        assert result[0, 0] == pytest.approx(6.0, rel=1e-5)
        assert result[1, 1] == pytest.approx(10.0, rel=1e-5)
        assert abs(result[0, 1]) < 1e-6

    def test_mixed_partials_symmetric(self) -> None:
        def bowl(v):
            return v[0] ** 2 * v[1] + v[1] ** 3

        result = calculus.hessian(bowl, [2.0, 1.0], step=1e-4)
        assert result.shape == (2, 2)
        assert result[0, 1] == pytest.approx(result[1, 0], rel=1e-6)
        assert result[0, 1] == pytest.approx(4.0, rel=1e-3)
