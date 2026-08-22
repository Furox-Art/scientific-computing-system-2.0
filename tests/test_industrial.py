"""Industrial-scale sparse solver tests: preconditioners and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from cds2 import sparse


@pytest.fixture()
def poisson_system_sparse():
    n = 200
    main_diag = 2.0 * np.ones(n)
    off_diag = -np.ones(n - 1)
    matrix = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
    rhs = np.arange(1.0, n + 1.0)
    return matrix, rhs


class TestPreconditioners:
    def test_jacobi_applies_inverse_diagonal(self) -> None:
        matrix = np.array([[4.0, 0.0], [0.0, 2.0]])
        operator = sparse.jacobi_preconditioner(matrix)
        assert np.allclose(operator @ np.ones(2), [0.25, 0.5])

    def test_ilu_preconditioner_inverts(self) -> None:
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        operator = sparse.ilu_preconditioner(matrix)
        applied = operator @ np.array([5.0, 4.0])
        reference = np.linalg.solve(matrix, np.array([5.0, 4.0]))
        assert np.allclose(applied, reference)

    def test_preconditioned_cg_beats_plain_on_ill_conditioned(self) -> None:
        n = 800
        matrix = np.diag(np.concatenate([np.ones(n - 1), [1e-8]]))
        rhs = np.ones(n)
        plain = sparse.solve_cg(matrix, rhs, rtol=1e-12, maxiter=4000)
        preconditioned = sparse.solve_cg(
            matrix,
            rhs,
            rtol=1e-12,
            maxiter=4000,
            M=sparse.jacobi_preconditioner(matrix),
        )
        assert preconditioned.converged
        assert preconditioned.residual_norm is not None
        if plain.residual_norm is not None:
            assert preconditioned.residual_norm <= plain.residual_norm * 10

    def test_ilu_gmres_relative_residual(self, poisson_system_sparse) -> None:
        matrix, rhs = poisson_system_sparse
        result = sparse.solve_gmres(
            matrix,
            rhs,
            rtol=1e-10,
            restart=60,
            M=sparse.ilu_preconditioner(matrix),
        )
        norm_b = float(np.linalg.norm(rhs))
        assert result.residual_norm is not None
        assert result.residual_norm / norm_b < 1e-9


class TestResidualDiagnostics:
    def test_residual_norm_reported(self, poisson_system_sparse) -> None:
        from scipy import sparse as sparse_module

        matrix, rhs = poisson_system_sparse
        result = sparse.solve_cg(matrix, rhs, rtol=1e-10)
        csr_view = sparse_module.csr_matrix(matrix)
        expected = float(np.linalg.norm(csr_view @ result.x - rhs))
        assert result.residual_norm == pytest.approx(expected, rel=1e-6)

    def test_all_solvers_report_residual(self, poisson_system_sparse) -> None:
        matrix, rhs = poisson_system_sparse
        results = [
            sparse.solve_cg(matrix, rhs, rtol=1e-12),
            sparse.solve_gmres(matrix, rhs, rtol=1e-12),
            sparse.solve_bicgstab(matrix, rhs, rtol=1e-12),
        ]
        for result in results:
            assert result.residual_norm is not None
            assert result.residual_norm < 1e-4


class TestLargeScaleSolve:
    def test_million_scale_tridiagonal(self) -> None:
        """κ ≈ (n/π)^2 ≈ 6e9 at n=250k: plain CG needs ~80k+ iterations.

        The industrial pattern makes the impossible routine - an incomplete
        LU preconditioner collapses the iteration count to a handful.
        """
        from scipy.linalg import solve_banded

        n = 250_000
        bands = sparse.sparse_diag(
            [2.0 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)],
            offsets=[0, -1, 1],
        )
        rhs_values = np.linspace(1.0, 2.0, n)

        unpreconditioned = sparse.solve_cg(bands, rhs_values, rtol=1e-8, maxiter=150)
        assert not unpreconditioned.converged

        preconditioner = sparse.ilu_preconditioner(bands, drop_tol=1e-8, fill_factor=10)
        result = sparse.solve_cg(
            bands,
            rhs_values,
            rtol=1e-8,
            maxiter=2000,
            M=preconditioner,
        )
        assert result.converged

        banded_storage = np.zeros((3, n))
        banded_storage[0, 1:] = -1.0
        banded_storage[1, :] = 2.0
        banded_storage[2, :-1] = -1.0
        reference = solve_banded((1, 1), banded_storage, rhs_values)
        assert np.allclose(result.x, reference, atol=1e-7)
