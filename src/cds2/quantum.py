"""Quantum circuit simulation on dense statevectors."""

from __future__ import annotations

import cmath
import math

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GATES",
    "QuantumCircuit",
]

ComplexVector = NDArray[np.complex128]


def _single_qubit_matrix(name: str, theta: float | None = None) -> NDArray[np.complex128]:
    matrix: NDArray[np.complex128]
    if name == "I":
        matrix = np.eye(2, dtype=np.complex128)
    elif name == "X":
        matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    elif name == "Y":
        matrix = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    elif name == "Z":
        matrix = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    elif name == "H":
        matrix = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    elif name == "S":
        matrix = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    elif name == "T":
        matrix = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=np.complex128)
    else:
        half = theta / 2 if theta is not None else 0.0
        if name == "RX":
            matrix = np.array(
                [
                    [math.cos(half), -1j * math.sin(half)],
                    [-1j * math.sin(half), math.cos(half)],
                ],
                dtype=np.complex128,
            )
        elif name == "RY":
            matrix = np.array(
                [[math.cos(half), -math.sin(half)], [math.sin(half), math.cos(half)]],
                dtype=np.complex128,
            )
        elif name == "RZ":
            matrix = np.array(
                [[np.exp(-1j * half), 0], [0, np.exp(1j * half)]],
                dtype=np.complex128,
            )
        else:
            raise ValueError(f"unknown single-qubit gate: {name!r}")
    return matrix


GATES: dict[str, NDArray[np.complex128]] = {
    name: _single_qubit_matrix(name) for name in ("I", "X", "Y", "Z", "H", "S", "T")
}


class QuantumCircuit:
    """Dense statevector simulator over ``n`` qubits (little-endian ordering)."""

    def __init__(self, n_qubits: int) -> None:
        if not 1 <= n_qubits <= 16:
            msg = "n_qubits must be between 1 and 16"
            raise ValueError(msg)
        self.n_qubits = n_qubits
        self._state: ComplexVector | None = None
        self.reset()

    def reset(self) -> QuantumCircuit:
        """Return to the all-zero computational basis state."""
        self._state = np.zeros(2**self.n_qubits, dtype=np.complex128)
        self._state[0] = 1.0
        return self

    def _require_state(self) -> ComplexVector:
        if self._state is None:
            self.reset()
        assert self._state is not None
        return self._state

    def _validate_target(self, qubit: int) -> int:
        if not 0 <= qubit < self.n_qubits:
            msg = f"qubit index out of range: {qubit}"
            raise ValueError(msg)
        return qubit

    def _apply_single(self, matrix: NDArray[np.complex128], target: int) -> None:
        state = self._require_state()
        step = 1 << target
        for base in range(0, state.size, step * 2):
            for offset in range(step):
                low = base + offset
                high = low + step
                low_amplitude = state[low]
                high_amplitude = state[high]
                state[low] = matrix[0, 0] * low_amplitude + matrix[0, 1] * high_amplitude
                state[high] = matrix[1, 0] * low_amplitude + matrix[1, 1] * high_amplitude

    def gate(self, name: str, qubit: int, theta: float | None = None) -> QuantumCircuit:
        """Apply a named single-qubit gate (optionally parameterized)."""
        self._apply_single(_single_qubit_matrix(name, theta), self._validate_target(qubit))
        return self

    def x(self, qubit: int) -> QuantumCircuit:
        """Pauli-X bit flip."""
        return self.gate("X", qubit)

    def y(self, qubit: int) -> QuantumCircuit:
        """Pauli-Y."""
        return self.gate("Y", qubit)

    def z(self, qubit: int) -> QuantumCircuit:
        """Pauli-Z phase flip."""
        return self.gate("Z", qubit)

    def h(self, qubit: int) -> QuantumCircuit:
        """Hadamard superposition gate."""
        return self.gate("H", qubit)

    def s(self, qubit: int) -> QuantumCircuit:
        """Phase gate S."""
        return self.gate("S", qubit)

    def t(self, qubit: int) -> QuantumCircuit:
        """T (pi/8) gate."""
        return self.gate("T", qubit)

    def rx(self, qubit: int, theta: float) -> QuantumCircuit:
        """Rotation about the X axis."""
        return self.gate("RX", qubit, theta)

    def ry(self, qubit: int, theta: float) -> QuantumCircuit:
        """Rotation about the Y axis."""
        return self.gate("RY", qubit, theta)

    def rz(self, qubit: int, theta: float) -> QuantumCircuit:
        """Rotation about the Z axis."""
        return self.gate("RZ", qubit, theta)

    def cnot(self, control: int, target: int) -> QuantumCircuit:
        """Controlled-NOT between two distinct qubits."""
        control_index = self._validate_target(control)
        target_index = self._validate_target(target)
        if control_index == target_index:
            msg = "control and target must differ"
            raise ValueError(msg)
        state = self._require_state()
        control_mask = 1 << control_index
        target_mask = 1 << target_index
        for basis in range(state.size):
            # outer test guarantees the target bit is clear, so partner is
            # always the larger index here
            if basis & control_mask and not basis & target_mask:
                partner = basis | target_mask
                state[basis], state[partner] = state[partner], state[basis]
        return self

    def cz(self, control: int, target: int) -> QuantumCircuit:
        """Controlled-Z phase kick on the |11> amplitude."""
        control_index = self._validate_target(control)
        target_index = self._validate_target(target)
        if control_index == target_index:
            msg = "control and target must differ"
            raise ValueError(msg)
        state = self._require_state()
        mask = (1 << control_index) | (1 << target_index)
        for basis in range(state.size):
            if basis & mask == mask:
                state[basis] *= -1.0
        return self

    def swap(self, first: int, second: int) -> QuantumCircuit:
        """Exchange two qubit registers via three CNOTs."""
        first_index = self._validate_target(first)
        second_index = self._validate_target(second)
        if first_index == second_index:
            return self
        return (
            self.cnot(first_index, second_index)
            .cnot(second_index, first_index)
            .cnot(first_index, second_index)
        )

    def statevector(self) -> ComplexVector:
        """Copy of the current amplitude vector."""
        copied: ComplexVector = self._require_state().copy()
        return copied

    def probabilities(self) -> NDArray[np.float64]:
        """Measurement probabilities over all computational basis states."""
        magnitudes: NDArray[np.float64] = np.asarray(
            np.abs(self._require_state()) ** 2, dtype=float
        )
        return magnitudes

    def measure(self, shots: int = 1000, seed: int | None = None) -> dict[int, int]:
        """Sample ``shots`` computational-basis outcomes with a seeded RNG."""
        if shots < 1:
            msg = "shots must be at least 1"
            raise ValueError(msg)
        rng_values = np.random.default_rng(seed)
        draws = rng_values.choice(self.probabilities().size, size=shots, p=self.probabilities())
        counts: dict[int, int] = {}
        for outcome in draws.tolist():
            counts[int(outcome)] = counts.get(int(outcome), 0) + 1
        return dict(sorted(counts.items()))
