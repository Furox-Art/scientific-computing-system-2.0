"""GPU-accelerated linear algebra via CuPy.

Each function mirrors its CPU counterpart in ``cds2.linalg`` but runs on the
GPU. Results are synced back to host memory as NumPy arrays on return.
"""

from __future__ import annotations

from typing import Any

from . import _ensure_cupy

__all__ = ["cholesky", "eigh", "solve", "svd"]


def solve(A: Any, b: Any) -> Any:
    """Solve ``A x = b`` on the GPU.

    Args:
        A: (n, n) array-like.
        b: (n,) or (n, k) array-like.
    Returns:
        NumPy array with the solution.
    """
    cp = _ensure_cupy()
    A_gpu = cp.asarray(A)
    b_gpu = cp.asarray(b)
    x_gpu = cp.linalg.solve(A_gpu, b_gpu)
    return cp.asnumpy(x_gpu)


def eigh(A: Any) -> tuple[Any, Any]:
    """Symmetric eigen-decomposition on the GPU.

    Args:
        A: (n, n) symmetric array-like.
    Returns:
        ``(w, v)`` where ``w`` are eigenvalues and ``v`` is the eigenvector
        matrix, both as NumPy arrays.
    """
    cp = _ensure_cupy()
    w, v = cp.linalg.eigh(cp.asarray(A))
    return cp.asnumpy(w), cp.asnumpy(v)


def svd(A: Any, full_matrices: bool = True) -> tuple[Any, Any, Any]:
    """Singular value decomposition on the GPU.

    Args:
        A: (m, n) array-like.
        full_matrices: if True, return full U and Vh.
    Returns:
        ``(U, s, Vh)`` as NumPy arrays.
    """
    cp = _ensure_cupy()
    U, s, Vh = cp.linalg.svd(cp.asarray(A), full_matrices=full_matrices)
    return cp.asnumpy(U), cp.asnumpy(s), cp.asnumpy(Vh)


def cholesky(A: Any) -> Any:
    """Cholesky decomposition on the GPU.

    Args:
        A: (n, n) positive-definite array-like.
    Returns:
        Lower-triangular Cholesky factor as a NumPy array.
    """
    cp = _ensure_cupy()
    L = cp.linalg.cholesky(cp.asarray(A))
    return cp.asnumpy(L)
