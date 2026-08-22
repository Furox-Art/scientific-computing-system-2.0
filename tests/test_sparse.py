"""Tests for cds2.sparse iterative solvers."""

import numpy as np
import pytest

from cds2 import sparse


@pytest.fixture()
def poisson_system() -> tuple[np.ndarray, np.ndarray]:
    n = 200
    main_diag = 2.0 * np.ones(n)
    off_diag = -np.ones(n - 1)
    matrix = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
    rhs = np.arange(1.0, n + 1.0)
    return matrix, rhs


class TestIterativeSolvers:
    def test_cg_converges_to_direct_solution(self, poisson_system) -> None:
        matrix, rhs = poisson_system
        result = sparse.solve_cg(matrix, rhs)
        reference = np.linalg.solve(matrix, rhs)
        assert result.converged
        assert np.allclose(result.x, reference, atol=1e-6)

    def test_gmres(self, poisson_system) -> None:
        matrix, rhs = poisson_system
        result = sparse.solve_gmres(matrix, rhs, rtol=1e-12, restart=60)
        reference = np.linalg.solve(matrix, rhs)
        assert result.converged
        assert np.max(np.abs(result.x - reference)) < 1e-5

    def test_bicgstab(self, poisson_system) -> None:
        matrix, rhs = poisson_system
        result = sparse.solve_bicgstab(matrix, rhs)
        reference = np.linalg.solve(matrix, rhs)
        assert result.converged
        assert np.max(np.abs(result.x - reference)) < 1e-3

    def test_tight_tolerance_improves_accuracy(self, poisson_system) -> None:
        matrix, rhs = poisson_system
        loose = sparse.solve_cg(matrix, rhs, rtol=1e-3)
        tight = sparse.solve_cg(matrix, rhs, rtol=1e-12)
        reference = np.linalg.solve(matrix, rhs)
        tight_error = float(np.max(np.abs(tight.x - reference)))
        loose_error = float(np.max(np.abs(loose.x - reference)))
        assert tight_error <= loose_error


class TestEigenAndSVD:
    def test_largest_eigenpairs_match_numpy(self) -> None:
        rng_values = np.random.default_rng(0)
        symmetric = rng_values.normal(size=(80, 80))
        symmetric = (symmetric + symmetric.T) / 2.0
        result = sparse.largest_eigenpairs(symmetric, k=3)
        expected = np.sort(np.linalg.eigvalsh(symmetric))[::-1][:3]
        assert np.allclose(sorted(result.eigenvalues, reverse=True), expected, rtol=1e-6)

    def test_smallest_eigenpairs_sorted_ascending(self) -> None:
        diagonal = np.diag([1.0, 5.0, 9.0, 13.0])
        result = sparse.smallest_eigenpairs(diagonal, k=2)
        assert np.allclose(result.eigenvalues, [1.0, 5.0], atol=1e-8)

    def test_truncated_svd_recovers_top_singvals(self) -> None:
        rng_values = np.random.default_rng(1)
        matrix = rng_values.normal(size=(120, 40))
        result = sparse.truncated_svd(matrix, k=4)
        full_singulars = np.linalg.svd(matrix, compute_uv=False)
        assert np.allclose(result.s, full_singulars[:4], rtol=1e-6)
        assert result.u.shape == (120, 4)
        assert result.vt.shape == (4, 40)
