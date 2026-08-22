"""Spectral graph theory: Laplacians, Fiedler vectors, spectral clustering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .ml import KMeans

__all__ = [
    "laplacian",
    "fiedler_vector",
    "algebraic_connectivity",
    "spectral_cluster",
]

FloatArray = NDArray[np.float64]


def laplacian(adj: object, normalized: bool = False) -> sparse.csr_matrix:
    """Graph Laplacian ``L = D - W`` (combinatorial or symmetric-normalized)."""
    matrix = sparse.coo_matrix(adj)
    n = matrix.shape[0]
    binary = sparse.csr_matrix(matrix)
    binary.data = np.ones_like(binary.data)
    degree = np.asarray(binary.sum(axis=1)).ravel().astype(float)
    degree_matrix = sparse.diags(degree)
    combinatorial = (degree_matrix - matrix).tocsr()
    if not normalized:
        return combinatorial
    inverse_sqrt = 1.0 / np.sqrt(np.where(degree == 0.0, 1.0, degree))
    scaling = sparse.diags(inverse_sqrt)
    return (scaling @ combinatorial @ scaling).tocsr()


def _bottom_eigenvectors(
    laplacian_matrix: sparse.spmatrix, k: int
) -> tuple[FloatArray, FloatArray]:
    from scipy.sparse.linalg import eigsh

    values, vectors = eigsh(
        sparse.csr_matrix(laplacian_matrix).astype(float),
        k=k,
        sigma=-1e-6,
        which="LM",
    )
    order = np.argsort(values)
    return np.asarray(values[order]), np.asarray(vectors[:, order])


def fiedler_vector(adj: object) -> FloatArray:
    """Eigenvector of the second-smallest Laplacian eigenvalue.

    Sign encodes the best bipartition; smoothness over the graph makes it a
    canonical 1-D embedding.
    """
    laplacian_matrix = laplacian(adj)
    n = laplacian_matrix.shape[0]
    if n < 3:
        msg = "Fiedler vector needs at least 3 nodes"
        raise ValueError(msg)
    _values, vectors = _bottom_eigenvectors(laplacian_matrix, k=2)
    return vectors[:, 1]


def algebraic_connectivity(adj: object) -> float:
    """Second-smallest Laplacian eigenvalue - global connectivity measure."""
    laplacian_matrix = laplacian(adj)
    n = laplacian_matrix.shape[0]
    if n < 2:
        msg = "algebraic connectivity needs at least 2 nodes"
        raise ValueError(msg)
    values, _vectors = _bottom_eigenvectors(laplacian_matrix, k=2)
    return float(max(values[1], 0.0))


def spectral_cluster(
    adj: object,
    n_clusters: int,
    seed: int | None = None,
    normalized: bool = True,
) -> NDArray[np.int64]:
    """Cluster graph nodes by k-means on the bottom Laplacian eigenvectors."""
    laplacian_matrix = laplacian(adj, normalized=normalized)
    n = laplacian_matrix.shape[0]
    if not 1 <= n_clusters <= n:
        msg = "n_clusters must be between 1 and the number of nodes"
        raise ValueError(msg)
    _values, embedding = _bottom_eigenvectors(laplacian_matrix, k=n_clusters)
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    unit_embedding = embedding / np.where(row_norms == 0.0, 1.0, row_norms)
    clusterer = KMeans(n_clusters=n_clusters, seed=seed).fit(unit_embedding)
    return clusterer.labels_
