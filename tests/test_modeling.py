"""Tests for cds2.modeling symbolic engine."""

from __future__ import annotations

import pytest

from cds2.modeling import (
    Add,
    Constant,
    Divide,
    MathModel,
    Negate,
    Subtract,
    Variable,
    diff,
    evaluate,
    iter_postorder,
    simplify,
    substitute,
    symbol,
    to_latex,
    to_string,
)


@pytest.fixture()
def x() -> Variable:
    return symbol("x")


class TestExpressionBuilding:
    def test_operator_overloads_build_tree(self, x: Variable) -> None:
        expression = 3 * x**2 - x / 2 + 1
        assert isinstance(expression, Add)
        assert expression.variables() == {"x"}

    def test_radd_and_rsub(self, x: Variable) -> None:
        assert isinstance(2 + x, Add)
        assert isinstance(2 - x, Subtract)
        assert isinstance(2 / x, Divide)

    def test_negate(self, x: Variable) -> None:
        assert isinstance(-x, Negate)

    def test_invalid_variable_name(self) -> None:
        with pytest.raises(ValueError, match="identifier"):
            symbol("not valid!")

    def test_wrap_rejects_bool_and_text(self, x: Variable) -> None:
        with pytest.raises(TypeError):
            x + True
        with pytest.raises(TypeError):
            x + "text"


class TestEvaluate:
    def test_quadratic(self, x: Variable) -> None:
        expression = x**2 + 3 * x + 5
        assert evaluate(expression, {"x": 2.0}) == pytest.approx(15.0)

    def test_division_and_negation(self, x: Variable) -> None:
        expression = -x / 4 + 1
        assert evaluate(expression, {"x": 8.0}) == pytest.approx(-1.0)

    def test_unbound_variable_raises(self, x: Variable) -> None:
        with pytest.raises(KeyError, match="unbound"):
            evaluate(x, {})


class TestDiff:
    def test_quadratic_rule(self, x: Variable) -> None:
        derivative = simplify(diff(x**2 + 3 * x + 5, "x"))
        assert evaluate(derivative, {"x": 10.0}) == pytest.approx(23.0)

    def test_product_rule(self, x: Variable) -> None:
        derivative = simplify(diff(x * x, "x"))
        assert evaluate(derivative, {"x": 3.0}) == pytest.approx(6.0)

    def test_quotient_rule(self, x: Variable) -> None:
        derivative = diff(x / 2, "x")
        assert evaluate(simplify(derivative), {"x": 7.0}) == pytest.approx(0.5)

    def test_power_chain_constant_exponent(self, x: Variable) -> None:
        derivative = simplify(diff(x**3, "x"))
        assert evaluate(derivative, {"x": 2.0}) == pytest.approx(12.0)

    def test_constant_base_exponential(self, x: Variable) -> None:
        import math

        derivative = diff(2**x, "x")
        expected = math.log(2.0) * 2.0**1.5
        assert evaluate(derivative, {"x": 1.5}) == pytest.approx(expected, rel=1e-12)

    def test_general_power_raises(self, x: Variable) -> None:
        with pytest.raises(NotImplementedError, match="logarithms"):
            diff(x**x, "x")

    def test_negation_chain(self, x: Variable) -> None:
        assert evaluate(simplify(diff(-x, "x")), {"x": 5.0}) == pytest.approx(-1.0)


class TestSimplify:
    def test_zero_add(self, x: Variable) -> None:
        assert to_string(simplify(x + Constant(0))) == "x"

    def test_zero_multiply(self, x: Variable) -> None:
        assert isinstance(simplify(x * Constant(0)), Constant)

    def test_one_multiply(self, x: Variable) -> None:
        assert to_string(simplify(x * Constant(1))) == "x"

    def test_subtract_self(self, x: Variable) -> None:
        assert isinstance(simplify(x - x), Constant)

    def test_divide_self(self, x: Variable) -> None:
        assert isinstance(simplify(x / x), Constant)

    def test_power_one_is_identity(self, x: Variable) -> None:
        assert to_string(simplify(x**1)) == "x"

    def test_double_negation(self, x: Variable) -> None:
        assert to_string(simplify(-(-x))) == "x"

    def test_constant_folding(self) -> None:
        folded = simplify(Constant(2) * Constant(5))
        assert folded.value == pytest.approx(10.0)

    def test_idempotent(self, x: Variable) -> None:
        once = simplify(x + Constant(0))
        assert to_string(simplify(once)) == to_string(once)


class TestSubstituteAndRender:
    def test_substitute_value(self, x: Variable) -> None:
        result = substitute(x**2, {"x": 3.0})
        assert isinstance(result, Constant)

    def test_substitute_expression(self) -> None:
        x = symbol("x")
        y = symbol("y")
        result = substitute(x + 1, {"x": y * 2})
        assert evaluate(result, {"y": 2.0}) == pytest.approx(5.0)

    def test_to_latex_fraction(self, x: Variable) -> None:
        latex_text = to_latex(x / 2)
        assert "\\frac" in latex_text

    def test_to_latex_power(self, x: Variable) -> None:
        latex_text = to_latex((x + 1) ** 2)
        assert "^{2}" in latex_text

    def test_postorder_visits_children_first(self, x: Variable) -> None:
        expression = x + 1
        visited = [type(node).__name__ for node in iter_postorder(expression)]
        assert visited[-1] == "Add"


class TestMathModel:
    def test_solve_equation_newton(self, x: Variable) -> None:
        model = MathModel.from_formula(x**2 + x, Constant(6.0))
        root = model.solve_equation("x", target_value=0.0, known={}, initial_guess=3.0)
        assert root == pytest.approx(2.0, abs=1e-8)

    def test_residual_zero_at_solution(self, x: Variable) -> None:
        model = MathModel.from_formula(2 * x, Constant(10.0))
        assert model.residual({"x": 5.0}) == pytest.approx(0.0)

    def test_fit_parameters_converges(self, x: Variable) -> None:
        model = MathModel.from_formula(x * symbol("k"), Constant(0.0))
        fitted = model.fit_parameters(
            {"k": 1.0},
            [({"x": 1.0}, 3.0), ({"x": 2.0}, 6.0)],
            learning_rate=0.05,
            epochs=400,
        )
        assert fitted["k"] == pytest.approx(3.0, abs=1e-3)
