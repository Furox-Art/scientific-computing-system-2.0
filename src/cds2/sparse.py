"""Large-scale sparse linear algebra built on scipy.sparse.linalg."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import (
    LinearOperator,
    bicgstab,
    cg,
    eigsh,
    gmres,
    svds,
)

__all__ = [
    "IterativeSolveResult",
    "EigenpairsResult",
    "TruncatedSVDResult",
    "solve_cg",
    "solve_gmres",
    "solve_bicgstab",
    "largest_eigenpairs",
    "smallest_eigenpairs",
    "truncated_svd",
    "sparse_eye",
    "sparse_diag",
    "sparse_kron",
    "one_norm_est",
]


@dataclass(frozen=True)
class IterativeSolveResult:
    """Solution of an iterative linear solver."""

    x: NDArray[np.float64]
    converged: bool
    iterations_info: int


@dataclass(frozen=True)
class EigenpairsResult:
    """Eigenvalues and matching eigenvectors (columns)."""

    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]


@dataclass(frozen=True)
class TruncatedSVDResult:
    """Truncated SVD ``A ~ U diag(s) Vt``."""

    u: NDArray[np.float64]
    s: NDArray[np.float64]
    vt: NDArray[np.float64]


def _as_matrix(A: object) -> object:
    if isinstance(A, LinearOperator):
        return A
    return sparse.csr_matrix(A)


def _finish(x: np.ndarray, info: int) -> IterativeSolveResult:
    return IterativeSolveResult(
        x=np.asarray(x, dtype=float),
        converged=bool(info == 0),
        iterations_info=int(info),
    )


def solve_cg(
    A: object,
    b: object,
    rtol: float = 1e-8,
    maxiter: int | None = None,
) -> IterativeSolveResult:
    """Conjugate-gradient solve for symmetric positive-definite systems."""
    solution, info = cg(_as_matrix(A), np.asarray(b, dtype=float), rtol=rtol, maxiter=maxiter)
    return _finish(solution, info)


def solve_gmres(
    A: object,
    b: object,
    rtol: float = 1e-8,
    maxiter: int | None = None,
    restart: int | None = None,
) -> IterativeSolveResult:
    """Generalized minimal residual solve for general square systems."""
    kwargs: dict[str, object] = {"rtol": rtol, "maxiter": maxiter}
    if restart is not None:
        kwargs["restart"] = restart
    solution, info = gmres(_as_matrix(A), np.asarray(b, dtype=float), **kwargs)
    return _finish(solution, info)


def solve_bicgstab(
    A: object,
    b: object,
    rtol: float = 1e-8,
    maxiter: int | None = None,
) -> IterativeSolveResult:
    """Biconjugate gradient stabilized solve - often faster on nonsymmetric PDEs."""
    solution, info = bicgstab(_as_matrix(A), np.asarray(b, dtype=float), rtol=rtol, maxiter=maxiter)
    return _finish(solution, info)


def largest_eigenpairs(A: object, k: int = 6, seed: int | None = None) -> EigenpairsResult:
    """Algebraically largest eigenpairs of a symmetric matrix (Lanczos)."""
    values, vectors = eigsh(_as_matrix(A), k=k, which="LA")
    order = np.argsort(values)[::-1]
    return EigenpairsResult(
        eigenvalues=np.asarray(values[order]), eigenvectors=np.asarray(vectors[:, order])
    )


def smallest_eigenpairs(A: object, k: int = 6, seed: int | None = None) -> EigenpairsResult:
    """Smallest algebraic eigenpairs of a symmetric matrix/operator."""
    values, vectors = eigsh(_as_matrix(A), k=k, which="SM")
    order = np.argsort(values)
    return EigenpairsResult(
        eigenvalues=np.asarray(values[order]), eigenvectors=np.asarray(vectors[:, order])
    )


def truncated_svd(A: object, k: int = 6, seed: int | None = None) -> TruncatedSVDResult:
    """Rank-``k`` truncated singular value decomposition."""
    u, s, vt = svds(sparse.csr_matrix(A), k=k, solver="arpack", rng=seed)
    order = np.argsort(s)[::-1]
    return TruncatedSVDResult(
        u=np.asarray(u[:, order]), s=np.asarray(s[order]), vt=np.asarray(vt[order, :])
    )


def sparse_eye(n: int) -> sparse.csr_matrix:
    """Identity matrix in CSR form."""
    return sparse.eye(n, format="csr")


def sparse_diag(values: Any, offsets: Sequence[int] | None = None) -> sparse.csr_matrix:
    """Diagonal (or multi-offset band) matrix from ``values``."""
    if offsets is None:
        return sparse.diags(np.asarray(values, dtype=float), offsets=0, format="csr")
    bands = [np.asarray(band, dtype=float) for band in values]
    return sparse.diags(bands, list(offsets), format="csr")


def sparse_kron(a: object, b: object) -> sparse.csr_matrix:
    """Kronecker product of two sparse matrices."""
    return sparse.kron(sparse.csr_matrix(a), sparse.csr_matrix(b), format="csr")


def one_norm_est(A: object, seed: int | None = None) -> float:
    """Lower bound estimate of the 1-norm of a square sparse matrix."""
    from scipy.sparse.linalg import onenormest

    value = onenormest(sparse.csr_matrix(A))
    return float(value)
