"""Tests for cds2.linalg."""

import numpy as np
import pytest

from cds2 import linalg


class TestSolve:
    def test_known_system(self) -> None:
        x = linalg.solve([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])
        assert np.allclose(x, [2.0, 3.0])

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            linalg.solve([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1.0, 2.0])

    def test_singular_raises(self) -> None:
        with pytest.raises(np.linalg.LinAlgError):
            linalg.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


class TestDetInv:
    def test_det_identity(self) -> None:
        assert linalg.det([[1, 0], [0, 1]]) == pytest.approx(1.0)

    def test_det_swap_rows(self) -> None:
        assert linalg.det([[0.0, 1.0], [1.0, 0.0]]) == pytest.approx(-1.0)

    def test_inv_roundtrip(self) -> None:
        a = np.array([[4.0, 7.0], [2.0, 6.0]])
        assert np.allclose(a @ linalg.inv(a), np.eye(2))

    def test_pinv_on_singular(self) -> None:
        a = [[1.0, 2.0], [2.0, 4.0]]
        product = a @ linalg.pinv(a) @ a
        assert np.allclose(product, a)


class TestEig:
    def test_symmetric_eigenpairs(self) -> None:
        result = linalg.eigh([[2.0, 0.0], [0.0, 5.0]])
        assert np.allclose(sorted(result.eigenvalues), [2.0, 5.0])

    def test_general_eig(self) -> None:
        result = linalg.eig([[2.0]])
        assert result.eigenvalues[0] == pytest.approx(2.0)


class TestSVD:
    def test_reconstruction(self) -> None:
        a = np.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
        decomp = linalg.svd(a)
        rebuilt = decomp.u[:, :2] @ np.diag(decomp.s) @ decomp.vh
        assert np.allclose(rebuilt, a)

    def test_singular_values_sorted(self) -> None:
        decomp = linalg.svd([[1.0, 2.0], [3.0, 4.0]])
        assert np.all(np.diff(decomp.s) <= 0)


class TestNormTrace:
    def test_vector_norm(self) -> None:
        assert linalg.norm([3.0, 4.0]) == pytest.approx(5.0)

    def test_frobenius_default(self) -> None:
        assert linalg.norm([[3.0, 4.0]]) == pytest.approx(5.0)

    def test_l1(self) -> None:
        assert linalg.norm([-3.0, 4.0], ord=1) == pytest.approx(7.0)

    def test_trace(self) -> None:
        assert linalg.trace([[1.0, 2.0], [3.0, 4.0]]) == pytest.approx(5.0)

    def test_norm_rejects_3d(self) -> None:
        with pytest.raises(ValueError, match="1-D vector or 2-D"):
            linalg.norm(np.zeros((2, 2, 2)))


class TestMisc:
    def test_matrix_power(self) -> None:
        a = [[2.0]]
        assert linalg.matrix_power(a, 3)[0, 0] == pytest.approx(8.0)
        assert np.allclose(linalg.matrix_power([[1.0, 2.0], [3.0, 4.0]], 0), np.eye(2))

    def test_rank_full_and_deficient(self) -> None:
        assert linalg.rank([[1.0, 0.0], [0.0, 1.0]]) == 2
        assert linalg.rank([[1.0, 2.0], [2.0, 4.0]]) == 1

    def test_cond_well_vs_ill(self) -> None:
        assert linalg.cond(np.eye(3)) == pytest.approx(1.0)
        ill = np.diag([1.0, 1e-10])
        assert linalg.cond(ill) > 1e9

    def test_cholesky_reconstruction(self) -> None:
        a = [[25.0, 15.0], [15.0, 18.0]]
        lower = linalg.cholesky(a)
        assert np.allclose(lower @ lower.T, a)

    def test_lstsq_overdetermined(self) -> None:
        x = np.linspace(0.0, 1.0, 20)
        y = 2.0 * x + 1.0
        result = linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y)
        assert np.allclose(result.solution, [2.0, 1.0])
        assert result.rank == 2
