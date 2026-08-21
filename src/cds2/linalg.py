"""Dense linear algebra built on NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "EigenResult",
    "SVDResult",
    "LeastSquaresResult",
    "solve",
    "det",
    "inv",
    "pinv",
    "eig",
    "eigh",
    "svd",
    "norm",
    "trace",
    "matrix_power",
    "rank",
    "cond",
    "cholesky",
    "lstsq",
]


@dataclass(frozen=True)
class EigenResult:
    """Eigen-decomposition of a square matrix."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass(frozen=True)
class SVDResult:
    """Singular-value decomposition ``A = U diag(s) Vh``."""

    u: np.ndarray
    s: np.ndarray
    vh: np.ndarray


@dataclass(frozen=True)
class LeastSquaresResult:
    """Solution of a (possibly over-determined) least-squares problem."""

    solution: np.ndarray
    residuals: np.ndarray
    rank: int
    singular_values: np.ndarray


def _as_matrix(a: object, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        msg = f"{name} must be a square 2-D array"
        raise ValueError(msg)
    return arr


def solve(a: object, b: object) -> np.ndarray:
    """Solve the linear system ``a x = b`` for a square matrix ``a``."""
    a_arr = _as_matrix(a, "a")
    b_arr = np.asarray(b, dtype=float)
    return np.linalg.solve(a_arr, b_arr)


def det(a: object) -> float:
    """Determinant of a square matrix."""
    return float(np.linalg.det(_as_matrix(a, "a")))


def inv(a: object) -> np.ndarray:
    """Inverse of a square, non-singular matrix."""
    return np.linalg.inv(_as_matrix(a, "a"))


def pinv(a: object, rcond: float | None = None) -> np.ndarray:
    """Moore-Penrose pseudo-inverse."""
    arr = np.asarray(a, dtype=float)
    kwargs: dict[str, float] = {} if rcond is None else {"rcond": rcond}
    return np.linalg.pinv(arr, **kwargs)


def eig(a: object) -> EigenResult:
    """General eigen-decomposition of a square matrix."""
    values, vectors = np.linalg.eig(_as_matrix(a, "a"))
    return EigenResult(eigenvalues=values, eigenvectors=vectors)


def eigh(a: object) -> EigenResult:
    """Eigen-decomposition of a real symmetric (or Hermitian) matrix."""
    values, vectors = np.linalg.eigh(np.asarray(a, dtype=float))
    return EigenResult(eigenvalues=values, eigenvectors=vectors)


def svd(a: object, full_matrices: bool = True) -> SVDResult:
    """Singular-value decomposition of an arbitrary 2-D array."""
    u, s, vh = np.linalg.svd(np.asarray(a, dtype=float), full_matrices=full_matrices)
    return SVDResult(u=u, s=s, vh=vh)


def norm(x: object, ord: float | str | None = None) -> float:  # noqa: A002
    """Vector or matrix norm (Frobenius by default for matrices)."""
    value = np.asarray(x, dtype=float)
    if value.ndim not in (1, 2):
        msg = "norm expects a 1-D vector or 2-D matrix"
        raise ValueError(msg)
    return float(np.linalg.norm(value, ord=ord))


def trace(a: object) -> float:
    """Sum of the diagonal elements."""
    return float(np.trace(_as_matrix(a, "a")))


def matrix_power(a: object, n: int) -> np.ndarray:
    """Integer power of a square matrix."""
    return np.linalg.matrix_power(_as_matrix(a, "a"), n)


def rank(a: object, tol: float | None = None) -> int:
    """Numerical rank of a matrix."""
    arr = np.asarray(a, dtype=float)
    kwargs: dict[str, float] = {} if tol is None else {"tol": tol}
    return int(np.linalg.matrix_rank(arr, **kwargs))


def cond(a: object) -> float:
    """Condition number (2-norm ratio of extreme singular values)."""
    return float(np.linalg.cond(_as_matrix(a, "a")))


def cholesky(a: object) -> np.ndarray:
    """Lower-triangular Cholesky factor of a positive-definite matrix."""
    return np.linalg.cholesky(_as_matrix(a, "a"))


def lstsq(a: object, b: object, rcond: float | None = None) -> LeastSquaresResult:
    """Least-squares solution of ``a x = b``."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.ndim != 2:
        msg = "a must be a 2-D array"
        raise ValueError(msg)
    solution, residuals, rank_value, singular = np.linalg.lstsq(a_arr, b_arr, rcond=rcond)
    return LeastSquaresResult(
        solution=solution,
        residuals=residuals,
        rank=int(rank_value),
        singular_values=singular,
    )
