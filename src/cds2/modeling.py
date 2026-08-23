"""Symbolic mathematics: expression trees, differentiation, LaTeX export."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping

__all__ = [
    "Expression",
    "Constant",
    "Variable",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Power",
    "Negate",
    "symbol",
    "diff",
    "simplify",
    "substitute",
    "evaluate",
    "to_string",
    "iter_postorder",
    "to_latex",
    "MathModel",
]


class Expression:
    """Base class of the symbolic expression tree."""

    def __add__(self, other: object) -> Expression:
        return Add(self, _wrap(other))

    def __radd__(self, other: object) -> Expression:
        return Add(_wrap(other), self)

    def __sub__(self, other: object) -> Expression:
        return Subtract(self, _wrap(other))

    def __rsub__(self, other: object) -> Expression:
        return Subtract(_wrap(other), self)

    def __mul__(self, other: object) -> Expression:
        return Multiply(self, _wrap(other))

    def __rmul__(self, other: object) -> Expression:
        return Multiply(_wrap(other), self)

    def __truediv__(self, other: object) -> Expression:
        return Divide(self, _wrap(other))

    def __rtruediv__(self, other: object) -> Expression:
        return Divide(_wrap(other), self)

    def __pow__(self, exponent: object) -> Expression:
        return Power(self, _wrap(exponent))

    def __rpow__(self, base: object) -> Expression:
        return Power(_wrap(base), self)

    def __neg__(self) -> Expression:
        return Negate(self)

    def variables(self) -> set[str]:
        raise NotImplementedError

    def _latex(self) -> str:
        raise NotImplementedError


def _wrap(value: object) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid symbol operand")
    if isinstance(value, (int, float)):
        return Constant(float(value))
    raise TypeError(f"cannot wrap {type(value).__name__} into an expression")


class Constant(Expression):
    """A fixed numeric literal."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def variables(self) -> set[str]:
        return set()

    def _latex(self) -> str:
        text = f"{self.value:g}"
        return text


class Variable(Expression):
    """A named free variable."""

    def __init__(self, name: str) -> None:
        if not name.isidentifier():
            msg = f"variable name must be an identifier: {name!r}"
            raise ValueError(msg)
        self.name = name

    def variables(self) -> set[str]:
        return {self.name}

    def _latex(self) -> str:
        return self.name


def symbol(name: str) -> Variable:
    """Shorthand factory for a named :class:`Variable`."""
    return Variable(name)


class BinaryOp(Expression):
    """Base for binary operators."""

    operator_symbol = "?"
    latex_operator = r"\operatorname{?}"

    def __init__(self, left: Expression, right: Expression) -> None:
        self.left = left
        self.right = right

    def variables(self) -> set[str]:
        return self.left.variables() | self.right.variables()


class Add(BinaryOp):
    operator_symbol = "+"
    latex_operator = "+"

    def _latex(self) -> str:
        return f"{self.left._latex()} + {self.right._latex()}"


class Subtract(BinaryOp):
    operator_symbol = "-"
    latex_operator = "-"

    def _latex(self) -> str:
        return f"{self.left._latex()} - {self.right._latex()}"


class Multiply(BinaryOp):
    operator_symbol = "*"
    latex_operator = r"\cdot"

    def _latex(self) -> str:
        return rf"{_parenthesize_add(self.left)} \cdot {_parenthesize_add(self.right)}"


class Divide(BinaryOp):
    operator_symbol = "/"
    latex_operator = "/"

    def _latex(self) -> str:
        return rf"\frac{{{self.left._latex()}}}{{{self.right._latex()}}}"


class Power(BinaryOp):
    operator_symbol = "**"

    def _latex(self) -> str:
        base = (
            f"({self.left._latex()})"
            if isinstance(self.left, (Add, Subtract))
            else self.left._latex()
        )
        return f"{base}^{{{self.right._latex()}}}"


class Negate(Expression):
    """Unary negation."""

    def __init__(self, operand: Expression) -> None:
        self.operand = operand

    def variables(self) -> set[str]:
        return self.operand.variables()

    def _latex(self) -> str:
        inner = (
            f"({self.operand._latex()})"
            if isinstance(self.operand, (Add, Subtract))
            else self.operand._latex()
        )
        return f"-{inner}"


def _parenthesize_add(expr: Expression) -> str:
    text = expr._latex()
    if isinstance(expr, (Add, Subtract)):
        return f"({text})"
    return text


# ------------------------------------------------------------ operations ----
def diff(expression: Expression, variable: str) -> Expression:
    """Symbolic partial derivative of ``expression`` w.r.t. ``variable``."""
    if isinstance(expression, Constant):
        return Constant(0.0)
    if isinstance(expression, Variable):
        return Constant(1.0 if expression.name == variable else 0.0)
    if isinstance(expression, Negate):
        return Negate(diff(expression.operand, variable))
    if isinstance(expression, Add):
        return Add(diff(expression.left, variable), diff(expression.right, variable))
    if isinstance(expression, Subtract):
        return Subtract(diff(expression.left, variable), diff(expression.right, variable))
    if isinstance(expression, Multiply):
        return Add(
            Multiply(diff(expression.left, variable), expression.right),
            Multiply(expression.left, diff(expression.right, variable)),
        )
    if isinstance(expression, Divide):
        numerator = Subtract(
            Multiply(diff(expression.left, variable), expression.right),
            Multiply(expression.left, diff(expression.right, variable)),
        )
        return Divide(numerator, Power(expression.right, Constant(2.0)))
    if isinstance(expression, Power):
        if isinstance(expression.right, Constant):
            reduced_power = Power(expression.left, Constant(expression.right.value - 1.0))
            return Multiply(
                Multiply(expression.right, reduced_power),
                diff(expression.left, variable),
            )
        if isinstance(expression.left, Constant):
            log_term = Multiply(Constant(math.log(expression.left.value)), expression)
            return Multiply(log_term, diff(expression.right, variable))
        raise NotImplementedError("d/dx u(x)^v(x): take logarithms or provide a numeric base")
    raise NotImplementedError(f"differentiation not defined for {type(expression).__name__}")


def simplify(expression: Expression, max_rounds: int = 8) -> Expression:
    """Apply algebraic simplification rules until stable."""
    current = expression
    for _ in range(max_rounds):
        simplified = _simplify_once(current)
        if _signature(simplified) == _signature(current):
            return simplified
        current = simplified
    return current


def _simplify_once(expression: Expression) -> Expression:
    if isinstance(expression, (Constant, Variable)):
        return expression
    if isinstance(expression, Negate):
        operand = _simplify_once(expression.operand)
        if isinstance(operand, Constant):
            return Constant(-operand.value)
        if isinstance(operand, Negate):
            return operand.operand
        return Negate(operand)

    if not isinstance(expression, BinaryOp):
        return expression
    left = _simplify_once(expression.left)
    right = _simplify_once(expression.right)

    if isinstance(left, Constant) and isinstance(right, Constant):
        rebuilt = _rebuild(type(expression), left, right)
        return Constant(evaluate(rebuilt, {}))

    if isinstance(expression, Add):
        if isinstance(left, Constant) and left.value == 0.0:
            return right
        if isinstance(right, Constant) and right.value == 0.0:
            return left
    if isinstance(expression, Subtract):
        if isinstance(right, Constant) and right.value == 0.0:
            return left
        if _signature(left) == _signature(right):
            return Constant(0.0)
    if isinstance(expression, Multiply):
        if (isinstance(left, Constant) and left.value == 0.0) or (
            isinstance(right, Constant) and right.value == 0.0
        ):
            return Constant(0.0)
        if isinstance(left, Constant) and left.value == 1.0:
            return right
        if isinstance(right, Constant) and right.value == 1.0:
            return left
    if isinstance(expression, Divide):
        if isinstance(left, Constant) and left.value == 0.0:
            return Constant(0.0)
        if isinstance(right, Constant) and right.value == 1.0:
            return left
        if _signature(left) == _signature(right):
            return Constant(1.0)
    if isinstance(expression, Power):
        if isinstance(right, Constant) and right.value == 1.0:
            return left
        if isinstance(right, Constant) and right.value == 0.0:
            return Constant(1.0)

    return _rebuild(type(expression), left, right)


def _rebuild(operator_type: type[BinaryOp], left: Expression, right: Expression) -> Expression:
    return operator_type(left, right)


def _signature(expression: Expression) -> str:
    try:
        return to_string(expression)
    except NotImplementedError:
        return f"#unknown-{id(expression)}"


def to_string(expression: Expression) -> str:
    """Plain-text rendering used for structural comparison and printing."""
    if isinstance(expression, Constant):
        return f"#({expression.value:g})"
    if isinstance(expression, Variable):
        return expression.name
    if isinstance(expression, Negate):
        return f"(~{to_string(expression.operand)})"
    if isinstance(expression, BinaryOp):
        inner = f"{to_string(expression.left)} {expression.operator_symbol} {to_string(expression.right)}"
        return f"({inner})"
    raise NotImplementedError(type(expression).__name__)


def substitute(expression: Expression, replacements: Mapping[str, object]) -> Expression:
    """Replace variables with values or other expressions."""

    def walk(node: Expression) -> Expression:
        if isinstance(node, Constant):
            return node
        if isinstance(node, Variable):
            if node.name in replacements:
                return _wrap(replacements[node.name])
            return node
        if isinstance(node, Negate):
            return Negate(walk(node.operand))
        if isinstance(node, BinaryOp):
            return _rebuild(type(node), walk(node.left), walk(node.right))
        raise NotImplementedError(type(node).__name__)

    return simplify(walk(expression))


def evaluate(expression: Expression, environment: dict[str, float]) -> float:
    """Numerically evaluate the expression under a variable binding."""

    def walk(node: Expression) -> float:
        if isinstance(node, Constant):
            return node.value
        if isinstance(node, Variable):
            if node.name not in environment:
                msg = f"unbound variable: {node.name}"
                raise KeyError(msg)
            return float(environment[node.name])
        if isinstance(node, Negate):
            return -walk(node.operand)
        if isinstance(node, Add):
            return float(walk(node.left) + walk(node.right))
        if isinstance(node, Subtract):
            return float(walk(node.left) - walk(node.right))
        if isinstance(node, Multiply):
            return float(walk(node.left) * walk(node.right))
        if isinstance(node, Divide):
            return float(walk(node.left) / walk(node.right))
        if isinstance(node, Power):
            return float(walk(node.left) ** walk(node.right))
        raise NotImplementedError(type(node).__name__)

    return walk(expression)


def iter_postorder(expression: Expression) -> Iterator[Expression]:
    """Yield expression nodes depth-first, children before parents."""
    if isinstance(expression, Negate):
        yield from iter_postorder(expression.operand)
    elif isinstance(expression, BinaryOp):
        yield from iter_postorder(expression.left)
        yield from iter_postorder(expression.right)
    yield expression


def to_latex(expression: Expression) -> str:
    """Render the expression as a LaTeX fragment."""
    return expression._latex()


class MathModel:
    """Equation-centric modelling helper built on the expression tree."""

    def __init__(self, equation_lhs: Expression, equation_rhs: Expression) -> None:
        self.equation = simplify(Subtract(equation_lhs, equation_rhs))

    @classmethod
    def from_formula(cls, lhs: Expression, rhs: Expression) -> MathModel:
        return cls(lhs, rhs)

    def residual(self, bindings: dict[str, float]) -> float:
        """Evaluate LHS - RHS under complete variable bindings."""
        substituted = substitute(self.equation, dict(bindings))
        return evaluate(substituted, {})

    def solve_equation(
        self,
        unknown: str,
        target_value: float,
        known: dict[str, float],
        initial_guess: float = 1.0,
    ) -> float:
        """Solve for ``unknown`` so that LHS equals ``target_value`` (Newton)."""

        def root_function(value: float) -> float:
            bindings = dict(known)
            bindings[unknown] = float(value)
            try:
                substituted = substitute(self.equation, bindings)
                return evaluate(substituted, {})
            except ZeroDivisionError:
                return float("inf")

        guess = initial_guess
        for _ in range(60):
            function_value = root_function(guess) - target_value
            step = 1e-7 * max(1.0, abs(guess))
            derivative_value = (root_function(guess + step) - root_function(guess - step)) / (
                2 * step
            )
            if abs(derivative_value) < 1e-14:
                break
            next_guess = guess - function_value / derivative_value
            if abs(next_guess - guess) < 1e-12 * max(1.0, abs(guess)):
                guess = next_guess
                break
            guess = next_guess
        return float(guess)

    def fit_parameters(
        self,
        parameters: dict[str, float],
        observations: list[tuple[dict[str, float], float]],
        learning_rate: float = 1e-3,
        epochs: int = 500,
    ) -> dict[str, float]:
        """Grid-free least-squares refinement of free parameters by GD."""
        current = dict(parameters)
        for _ in range(epochs):
            gradient: dict[str, float] = {name: 0.0 for name in current}
            for known, target in observations:
                predicted_bindings = {**known, **current}
                substituted = substitute(self.equation, predicted_bindings)
                error = evaluate(substituted, {}) - target
                for name in current:
                    probe = dict(predicted_bindings)
                    probe[name] += 1e-6
                    bumped = evaluate(substitute(self.equation, probe), {})
                    gradient[name] += 2.0 * error * (bumped - evaluate(substituted, {})) / 1e-6
            for name in current:
                current[name] -= learning_rate * gradient[name] / len(observations)
        return current
