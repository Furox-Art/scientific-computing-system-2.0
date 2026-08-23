"""Reverse-mode scalar autograd engine (micrograd-style)."""

from __future__ import annotations

import math
from collections.abc import Callable

__all__ = ["Value"]


class Value:
    """A scalar that records the operations producing it for backprop."""

    def __init__(
        self,
        data: float,
        parents: tuple[Value, ...] = (),
        op: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents
        self._op = op
        self._backward: Callable[[], None] = lambda: None

    def __add__(self, other: object) -> Value:
        other_value = _coerce(other)
        result = Value(self.data + other_value.data, (self, other_value), "+")

        def propagate() -> None:
            self.grad += result.grad
            other_value.grad += result.grad

        result._backward = propagate
        return result

    def __radd__(self, other: object) -> Value:
        return self.__add__(other)

    def __mul__(self, other: object) -> Value:
        other_value = _coerce(other)
        result = Value(self.data * other_value.data, (self, other_value), "*")

        def propagate() -> None:
            self.grad += other_value.data * result.grad
            other_value.grad += self.data * result.grad

        result._backward = propagate
        return result

    def __rmul__(self, other: object) -> Value:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> Value:
        other_value = _coerce(other)
        return self * other_value**-1.0

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: object) -> Value:
        other_value = _coerce(other)
        return self + (-other_value)

    def __rsub__(self, other: object) -> Value:
        other_value = _coerce(other)
        return other_value + (-self)

    def __pow__(self, exponent: float) -> Value:
        if isinstance(exponent, Value):
            raise TypeError("Value supports only constant exponents")
        result = Value(self.data**exponent, (self,), f"**{exponent}")

        def propagate() -> None:
            self.grad += exponent * self.data ** (exponent - 1.0) * result.grad

        result._backward = propagate
        return result

    def exp(self) -> Value:
        result = Value(math.exp(self.data), (self,), "exp")

        def propagate() -> None:
            self.grad += result.data * result.grad

        result._backward = propagate
        return result

    def tanh(self) -> Value:
        result = Value(math.tanh(self.data), (self,), "tanh")

        def propagate() -> None:
            self.grad += (1.0 - result.data**2) * result.grad

        result._backward = propagate
        return result

    def relu(self) -> Value:
        result = Value(max(0.0, self.data), (self,), "relu")

        def propagate() -> None:
            self.grad += (self.data > 0) * result.grad

        result._backward = propagate
        return result

    def backward(self) -> None:
        """Backpropagate gradients through every ancestor operation."""
        ordered: list[Value] = []
        seen: set[int] = set()

        def visit(node: Value) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node._parents:
                visit(parent)
            ordered.append(node)

        visit(self)
        self.grad = 1.0
        for node in reversed(ordered):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:g}, grad={self.grad:g})"


def _coerce(other: object) -> Value:
    if isinstance(other, Value):
        return other
    if isinstance(other, (int, float)):
        return Value(float(other))
    raise TypeError(f"unsupported operand: {type(other).__name__}")
