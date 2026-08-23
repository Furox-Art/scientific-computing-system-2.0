"""Tests for cds2.quantum statevector simulation."""

from __future__ import annotations

import numpy as np
import pytest

from cds2.quantum import GATES, QuantumCircuit


class TestSingleQubitGates:
    def test_x_flips_zero(self) -> None:
        circuit = QuantumCircuit(1).x(0)
        assert np.allclose(circuit.probabilities(), [0.0, 1.0])

    def test_h_creates_equal_superposition(self) -> None:
        circuit = QuantumCircuit(1).h(0)
        assert np.allclose(circuit.probabilities(), [0.5, 0.5])

    def test_z_phase_flip_visible_after_h(self) -> None:
        circuit = QuantumCircuit(1).h(0).z(0).h(0)
        assert circuit.probabilities()[1] == pytest.approx(1.0)

    def test_rx_pi_matches_x_up_to_phase(self) -> None:
        circuit = QuantumCircuit(1).rx(0, np.pi)
        amplitudes = circuit.statevector()
        assert abs(amplitudes[0]) == pytest.approx(0.0, abs=1e-12)
        assert abs(amplitudes[1]) == pytest.approx(1.0)

    def test_ry_half_pi_equal_mix(self) -> None:
        circuit = QuantumCircuit(1).ry(0, np.pi / 2)
        assert np.allclose(circuit.probabilities(), [0.5, 0.5])

    def test_s_and_t_are_phases(self) -> None:
        circuit = QuantumCircuit(1).h(0).s(0)
        state = circuit.statevector()
        assert state[1] == pytest.approx(1j / np.sqrt(2))
        circuit.t(0)
        # phases accumulate but stay invisible to measurement probabilities
        assert circuit.probabilities()[1] == pytest.approx(0.5)
        rotated = circuit.statevector()[1]
        assert abs(rotated) == pytest.approx(1 / np.sqrt(2))

    def test_reset_returns_to_zero(self) -> None:
        circuit = QuantumCircuit(1).x(0).reset()
        assert circuit.probabilities()[0] == pytest.approx(1.0)

    def test_gate_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            QuantumCircuit(1).x(5)

    def test_unknown_gate_name(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            QuantumCircuit(1).gate("W", 0)

    def test_invalid_qubit_count(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 16"):
            QuantumCircuit(0)


class TestTwoQubitGates:
    def test_cnot_entangles_with_hadamard(self) -> None:
        circuit = QuantumCircuit(2).h(0).cnot(0, 1)
        probabilities = circuit.probabilities()
        assert np.allclose(probabilities, [0.5, 0.0, 0.0, 0.5])

    def test_cnot_on_classical_bits(self) -> None:
        circuit = QuantumCircuit(2).x(0).cnot(0, 1)
        assert circuit.probabilities()[3] == pytest.approx(1.0)

    def test_cnot_requires_distinct_qubits(self) -> None:
        with pytest.raises(ValueError, match="differ"):
            QuantumCircuit(2).cnot(0, 0)

    def test_cz_phase_only_on_eleven(self) -> None:
        circuit = QuantumCircuit(2).x(0).x(1).cz(0, 1)
        state = circuit.statevector()
        assert state[3] == pytest.approx(-1.0)

    def test_swap_exchanges_states(self) -> None:
        circuit = QuantumCircuit(2).x(0).swap(0, 1)
        assert circuit.probabilities()[2] == pytest.approx(1.0)

    def test_swap_same_qubit_is_noop(self) -> None:
        circuit = QuantumCircuit(2).x(0).swap(0, 0)
        assert circuit.probabilities()[1] == pytest.approx(1.0)


class TestMeasurement:
    def test_deterministic_state_always_measured(self) -> None:
        counts = QuantumCircuit(2).x(1).measure(shots=500, seed=1)
        assert counts == {2: 500}

    def test_bell_state_uniform_counts(self) -> None:
        counts = QuantumCircuit(2).h(0).cnot(0, 1).measure(shots=4000, seed=42)
        assert set(counts) == {0, 3}
        for outcome in counts.values():
            assert outcome == pytest.approx(2000, rel=0.15)

    def test_invalid_shots_raise(self) -> None:
        with pytest.raises(ValueError, match="shots"):
            QuantumCircuit(1).measure(shots=0)

    def test_gates_table_has_constants(self) -> None:
        assert set(GATES) == {"I", "X", "Y", "Z", "H", "S", "T"}
        assert np.allclose(GATES["X"] @ GATES["X"], GATES["I"])
