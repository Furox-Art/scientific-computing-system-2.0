"""Dense linear algebra with typed dataclass results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg as sla

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
    "expm",
    "logm",
    "sqrtm",
]


@dataclass(frozen=True)
class EigenResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass(frozen=True)
class SVDResult:
    u: np.ndarray
    s: np.ndarray
    vh: np.ndarray


@dataclass(frozen=True)
class LeastSquaresResult:
    solution: np.ndarray
    residuals: np.ndarray
    rank: int
    singular_values: np.ndarray


def _as_numeric(a: object) -> np.ndarray:
    raw = np.asarray(a)
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    return np.asarray(a, dtype=dtype)


def _as_matrix(a: object, name: str) -> np.ndarray:
    arr = _as_numeric(a)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D array")
    return arr


def solve(a: object, b: object) -> np.ndarray:
    a_arr = _as_matrix(a, "a")
    b_arr = _as_numeric(b)
    return np.asarray(np.linalg.solve(a_arr, b_arr))


def det(a: object) -> float | complex:
    value = np.linalg.det(_as_matrix(a, "a"))
    return complex(value) if np.iscomplexobj(value) else float(value)


def inv(a: object) -> np.ndarray:
    return np.asarray(np.linalg.inv(_as_matrix(a, "a")))


def pinv(a: object, rcond: float | None = None) -> np.ndarray:
    arr = _as_numeric(a)
    return np.asarray(np.linalg.pinv(arr) if rcond is None else np.linalg.pinv(arr, rcond=rcond))


def eig(a: object) -> EigenResult:
    values, vectors = np.linalg.eig(_as_matrix(a, "a"))
    return EigenResult(eigenvalues=np.asarray(values), eigenvectors=np.asarray(vectors))


def eigh(a: object) -> EigenResult:
    values, vectors = np.linalg.eigh(_as_matrix(a, "a"))
    return EigenResult(eigenvalues=np.asarray(values), eigenvectors=np.asarray(vectors))


def svd(a: object, full_matrices: bool = True) -> SVDResult:
    arr = _as_numeric(a)
    if arr.ndim != 2:
        raise ValueError("a must be a 2-D array")
    u, s, vh = np.linalg.svd(arr, full_matrices=full_matrices)
    return SVDResult(u=np.asarray(u), s=np.asarray(s), vh=np.asarray(vh))


def norm(x: object, ord: float | Literal["fro", "nuc"] | None = None) -> float:  # noqa: A002
    value = _as_numeric(x)
    if value.ndim not in (1, 2):
        raise ValueError("norm expects a 1-D vector or 2-D matrix")
    return float(np.linalg.norm(value, ord=ord))


def trace(a: object) -> float | complex:
    value = np.trace(_as_matrix(a, "a"))
    return complex(value) if np.iscomplexobj(value) else float(value)


def matrix_power(a: object, n: int) -> np.ndarray:
    return np.asarray(np.linalg.matrix_power(_as_matrix(a, "a"), n))


def rank(a: object, tol: float | None = None) -> int:
    arr = _as_numeric(a)
    return int(np.linalg.matrix_rank(arr) if tol is None else np.linalg.matrix_rank(arr, tol=tol))


def cond(a: object) -> float:
    return float(np.linalg.cond(_as_matrix(a, "a")))


def cholesky(a: object) -> np.ndarray:
    return np.asarray(np.linalg.cholesky(_as_matrix(a, "a")))


def lstsq(a: object, b: object, rcond: float | None = None) -> LeastSquaresResult:
    a_arr = _as_numeric(a)
    b_arr = _as_numeric(b)
    if a_arr.ndim != 2:
        raise ValueError("a must be a 2-D array")
    solution, residuals, rank_value, singular = np.linalg.lstsq(a_arr, b_arr, rcond=rcond)
    return LeastSquaresResult(
        solution=np.asarray(solution),
        residuals=np.asarray(residuals),
        rank=int(rank_value),
        singular_values=np.asarray(singular),
    )


def expm(a: object) -> np.ndarray:
    return np.asarray(sla.expm(_as_matrix(a, "a")))


def logm(a: object) -> np.ndarray:
    return np.asarray(sla.logm(_as_matrix(a, "a")))


def sqrtm(a: object) -> np.ndarray:
    return np.asarray(sla.sqrtm(_as_matrix(a, "a")))
