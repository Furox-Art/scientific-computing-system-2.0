"""Oracle tests: cds2.linalg vs NumPy ground truth.

On each of 100 random systems we assert the cds2 result matches NumPy to
tight tolerance. This catches silent numerical regressions that point tests
miss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import cds2  # noqa: E402


class TestLinalgOracles:
    """Verify cds2.linalg against NumPy."""

    @pytest.mark.parametrize("seed", range(20))
    def test_solve_vs_numpy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = rng.integers(2, 30)
        A = rng.normal(size=(n, n)) + n * np.eye(n)
        b = rng.normal(size=n)
        x_cds2 = cds2.linalg.solve(A, b)
        x_np = np.linalg.solve(A, b)
        assert np.allclose(x_cds2, x_np, atol=1e-6)

    @pytest.mark.parametrize("seed", range(20))
    def test_det_vs_numpy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = rng.integers(2, 20)
        A = rng.normal(size=(n, n))
        assert np.isclose(cds2.linalg.det(A), np.linalg.det(A), rtol=1e-6)

    @pytest.mark.parametrize("seed", range(20))
    def test_inv_vs_numpy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = rng.integers(2, 20)
        A = rng.normal(size=(n, n)) + n * np.eye(n)
        Ainv_cds2 = cds2.linalg.inv(A)
        Ainv_np = np.linalg.inv(A)
        assert np.allclose(Ainv_cds2, Ainv_np, atol=1e-6)

    @pytest.mark.parametrize("seed", range(20))
    def test_eigh_vs_numpy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = rng.integers(2, 20)
        M = rng.normal(size=(n, n))
        A = (M + M.T) / 2.0
        res = cds2.linalg.eigh(A)
        w_cds2, v_cds2 = res.eigenvalues, res.eigenvectors
        w_np, v_np = np.linalg.eigh(A)
        assert np.allclose(w_cds2, w_np, atol=1e-6)
        # Eigenvectors match up to sign per column.
        assert np.allclose(np.abs(v_cds2), np.abs(v_np), atol=1e-6)

    @pytest.mark.parametrize("seed", range(20))
    def test_svd_vs_numpy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        m = rng.integers(5, 15)
        n = rng.integers(3, m)  # m >= n so shapes line up
        A = rng.normal(size=(m, n))
        # cds2.linalg.svd defaults to full_matrices=True; compare on the
        # economical reconstruction U[:, :n] @ diag(s) @ Vh == A.
        res = cds2.linalg.svd(A)
        U_c, s_c, Vh_c = res.u, res.s, res.vh
        U_np, s_np, Vh_np = np.linalg.svd(A, full_matrices=False)
        assert np.allclose(s_c, s_np, atol=1e-6)
        assert np.allclose(U_c[:, :n] @ np.diag(s_c) @ Vh_c, A, atol=1e-6)
