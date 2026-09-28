"""Numerical-stability and scientific-invariant stress tests.

Coverage tests show that code paths execute; these checks target properties
that must remain true under difficult scales, conditioning, and long operation
sequences.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats
from scipy.linalg import hilbert

from cds2 import epidemiology, linalg, stats
from cds2.quantum import QuantumCircuit


def test_solve_has_small_backward_error_on_hilbert_system() -> None:
    """An ill-conditioned solve should still have a small backward residual."""
    matrix = hilbert(10)
    expected = np.ones(10)
    rhs = matrix @ expected

    solved = np.asarray(linalg.solve(matrix, rhs), dtype=float)
    residual = matrix @ solved - rhs
    denominator = np.linalg.norm(matrix) * np.linalg.norm(solved) + np.linalg.norm(rhs)
    backward_error = np.linalg.norm(residual) / denominator

    # A Hilbert system is deliberately ill-conditioned, so comparing the
    # recovered x directly with ones would confuse conditioning with solver
    # correctness. Backward error is the appropriate stability diagnostic.
    assert np.isfinite(backward_error)
    assert backward_error < 1e-12


def test_t_test_is_stable_under_large_common_translation() -> None:
    """Adding the same large constant must not materially change inference."""
    rng = np.random.default_rng(2026)
    group_a = rng.normal(loc=0.0, scale=1.0, size=500)
    group_b = rng.normal(loc=0.18, scale=1.1, size=550)
    offset = 1.0e7

    base = stats.independent_t_test(group_a, group_b, equal_var=False)
    shifted = stats.independent_t_test(group_a + offset, group_b + offset, equal_var=False)
    ref_t, ref_p = scipy_stats.ttest_ind(group_a + offset, group_b + offset, equal_var=False)

    assert shifted.statistic == pytest.approx(ref_t, rel=2e-7, abs=2e-7)
    assert shifted.p_value == pytest.approx(ref_p, rel=2e-7, abs=2e-7)
    assert shifted.statistic == pytest.approx(base.statistic, rel=2e-7, abs=2e-7)
    assert shifted.p_value == pytest.approx(base.p_value, rel=2e-7, abs=2e-7)


def test_deep_quantum_circuit_preserves_state_norm() -> None:
    """Hundreds of unitary gates must not accumulate meaningful norm drift."""
    rng = np.random.default_rng(17)
    circuit = QuantumCircuit(4)

    for step in range(400):
        qubit = int(rng.integers(0, 4))
        angle = float(rng.uniform(-8.0 * np.pi, 8.0 * np.pi))
        selector = step % 7
        if selector == 0:
            circuit.h(qubit)
        elif selector == 1:
            circuit.rx(qubit, angle)
        elif selector == 2:
            circuit.ry(qubit, angle)
        elif selector == 3:
            circuit.rz(qubit, angle)
        elif selector == 4:
            target = (qubit + 1) % 4
            circuit.cnot(qubit, target)
        elif selector == 5:
            target = (qubit + 2) % 4
            circuit.cz(qubit, target)
        else:
            target = (qubit + 1) % 4
            circuit.swap(qubit, target)

    state = circuit.statevector()
    probabilities = circuit.probabilities()
    assert np.vdot(state, state).real == pytest.approx(1.0, abs=2e-12)
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=2e-12)
    assert np.all(probabilities >= 0.0)


def test_sir_conserves_population_at_large_scale() -> None:
    """Compartment conservation should survive large population magnitudes."""
    population = 1.0e9
    result = epidemiology.simulate_sir(
        population=population,
        beta=0.31,
        gamma=0.11,
        days=240,
        i0=25_000.0,
        steps_per_day=16,
    )
    total = result.susceptible + result.infected + result.recovered

    assert np.all(np.isfinite(total))
    assert np.allclose(total, population, rtol=5e-13, atol=1e-3)
    assert float(result.susceptible.min()) >= -1e-6
    assert float(result.infected.min()) >= -1e-6
    assert float(result.recovered.min()) >= -1e-6
