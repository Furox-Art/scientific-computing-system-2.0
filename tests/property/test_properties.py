"""Property-based tests for cds2 using hypothesis.

These verify algebraic/structural invariants (``det(A) == det(A.T)``,
``pagerank`` sums to 1, etc.) rather than point values. Install with
``pip install cds2[test]`` which pulls in hypothesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import cds2  # noqa: E402

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def square_matrix(draw: st.DrawFn, min_size: int = 1, max_size: int = 8) -> np.ndarray:
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    rows = []
    for _ in range(n):
        row = draw(
            st.lists(
                st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        )
        rows.append(row)
    return np.array(rows, dtype=float)


@st.composite
def symmetric_matrix(draw: st.DrawFn, max_size: int = 8) -> np.ndarray:
    A = draw(square_matrix(max_size=max_size))
    return (A + A.T) / 2.0


@st.composite
def positive_definite(draw: st.DrawFn, max_size: int = 6) -> np.ndarray:
    A = draw(square_matrix(max_size=max_size))
    return A @ A.T + np.eye(A.shape[0]) * 0.1


@st.composite
def probability_vector(draw: st.DrawFn, max_size: int = 8) -> np.ndarray:
    n = draw(st.integers(min_value=2, max_value=max_size))
    x = np.abs(np.array(draw(st.lists(st.floats(-5.0, 5.0), min_size=n, max_size=n))))
    s = x.sum()
    assume(s > 0)
    return x / s


# ---------------------------------------------------------------------------
# linalg properties
# ---------------------------------------------------------------------------


class TestLinalgProperties:
    @given(A=square_matrix(max_size=6))
    @settings(max_examples=50, deadline=None)
    def test_det_transpose(self, A: np.ndarray) -> None:
        # det(A) == det(A.T) exactly; use atol for near-singular matrices.
        assert np.isclose(cds2.linalg.det(A), cds2.linalg.det(A.T), rtol=1e-6, atol=1e-9)

    @given(A=positive_definite(max_size=5))
    @settings(max_examples=30, deadline=None)
    def test_cholesky_round_trip(self, A: np.ndarray) -> None:
        L = cds2.linalg.cholesky(A)
        assert np.allclose(L @ L.T, A, atol=1e-6)

    @given(A=square_matrix(max_size=5))
    @settings(max_examples=30, deadline=None)
    def test_solve_round_trip(self, A: np.ndarray) -> None:
        assume(np.abs(cds2.linalg.det(A)) > 1e-3)
        x_true = np.ones(A.shape[0])
        b = A @ x_true
        x = cds2.linalg.solve(A, b)
        assert np.allclose(x, x_true, atol=1e-4)

    @given(A=symmetric_matrix(max_size=5))
    @settings(max_examples=20, deadline=None)
    def test_eigh_symmetry(self, A: np.ndarray) -> None:
        res = cds2.linalg.eigh(A)
        assert np.allclose(
            A @ res.eigenvectors, res.eigenvectors @ np.diag(res.eigenvalues), atol=1e-4
        )


# ---------------------------------------------------------------------------
# stats properties
# ---------------------------------------------------------------------------


class TestStatsProperties:
    @given(data=st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=10, max_size=100))
    @settings(max_examples=30, deadline=None)
    def test_describe_bounds(self, data: list[float]) -> None:
        d = cds2.stats.describe(np.array(data))
        assert d.minimum <= d.mean <= d.maximum or np.isclose(d.minimum, d.maximum)
        assert d.std >= 0.0

    @given(p=probability_vector())
    @settings(max_examples=20, deadline=None)
    def test_entropy_nonneg(self, p: np.ndarray) -> None:
        assume(np.all(p > 0))
        assert cds2.infotheory.entropy(p) >= 0.0


# ---------------------------------------------------------------------------
# graph properties
# ---------------------------------------------------------------------------


class TestGraphProperties:
    @given(n=st.integers(2, 20), density=st.floats(0.2, 0.8))
    @settings(max_examples=20, deadline=None)
    def test_pagerank_sums_to_one(self, n: int, density: float) -> None:
        rng = np.random.default_rng(0)
        A = rng.random((n, n)) < density
        A = A.astype(float)
        A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
        scores = cds2.graph.pagerank(A)
        assert np.isclose(scores.sum(), 1.0, atol=1e-6)
        assert np.all(scores >= 0.0)


# ---------------------------------------------------------------------------
# calculus properties
# ---------------------------------------------------------------------------


class TestCalculusProperties:
    @given(x=st.floats(-10.0, 10.0))
    @settings(max_examples=50, deadline=None)
    def test_derivative_sin(self, x: float) -> None:
        d = cds2.calculus.derivative(np.sin, x)
        assert np.isclose(d, np.cos(x), atol=1e-5)

    @given(x=st.floats(-10.0, 10.0))
    @settings(max_examples=50, deadline=None)
    def test_derivative_exp(self, x: float) -> None:
        d = cds2.calculus.derivative(np.exp, x)
        assert np.isclose(d, np.exp(x), rtol=1e-5)
