"""Graph algorithms built on scipy.sparse.csgraph, plus PageRank."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import (
    bellman_ford,
    minimum_spanning_tree,
)
from scipy.sparse.csgraph import (
    connected_components as cs_connected_components,
)
from scipy.sparse.csgraph import (
    dijkstra as cs_dijkstra,
)
from scipy.sparse.csgraph import (
    floyd_warshall as cs_floyd_warshall,
)

__all__ = [
    "DegreeResult",
    "ComponentResult",
    "ShortestPaths",
    "from_edges",
    "degree",
    "connected_components",
    "shortest_paths",
    "single_source_shortest_paths",
    "minimum_spanning_forest",
    "floyd_warshall_paths",
    "bellman_ford_paths",
    "pagerank",
    "topological_order",
]

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


class ShortestPaths(NamedTuple):
    method: str
    distances: FloatArray


@dataclass(frozen=True)
class DegreeResult:
    """In/out/total degrees of each node."""

    in_degree: IntArray
    out_degree: IntArray
    total: IntArray


@dataclass(frozen=True)
class ComponentResult:
    """Connected-component labeling of a graph."""

    count: int
    labels: IntArray


def from_edges(
    n: int,
    edges: list[tuple[int, int]] | list[tuple[int, int, float]],
    weighted: bool = False,
    directed: bool = True,
) -> sparse.csr_matrix:
    """Build a CSR adjacency matrix from an edge list.

    Unweighted edges get weight 1. Undirected graphs mirror every edge.
    """
    if n <= 0:
        msg = "n must be positive"
        raise ValueError(msg)
    if not edges:
        return sparse.csr_matrix((n, n), dtype=float)
    rows = [int(e[0]) for e in edges]
    cols = [int(e[1]) for e in edges]
    data = [float(e[2]) for e in edges] if weighted else [1.0] * len(edges)
    for u, v in zip(rows, cols, strict=False):
        if not (0 <= u < n and 0 <= v < n):
            msg = f"edge ({u}, {v}) references a node outside 0..{n - 1}"
            raise ValueError(msg)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    if not directed:
        adj = adj.maximum(adj.T)
    return sparse.csr_matrix(adj)


def degree(adj: object, directed: bool = True) -> DegreeResult:
    """Node degrees from a CSR/COO adjacency matrix."""
    matrix = sparse.csr_matrix(adj)
    out_degree = (
        np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
        if _is_unweighted(matrix)
        else np.asarray((matrix != 0).sum(axis=1)).ravel().astype(np.int64)
    )
    in_degree = (
        np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)
        if _is_unweighted(matrix)
        else np.asarray((matrix != 0).sum(axis=0)).ravel().astype(np.int64)
    )
    total = in_degree + out_degree if directed else out_degree.copy()
    if not directed:
        in_degree = out_degree.copy()
        total = out_degree.copy()
    return DegreeResult(in_degree=in_degree, out_degree=out_degree, total=total)


def _is_unweighted(matrix: sparse.csr_matrix) -> bool:
    return bool(np.all(np.asarray(matrix.data) == 1.0))


def connected_components(
    adj: object, directed: bool = False, connection: str = "weak"
) -> ComponentResult:
    """Label the connected components of a graph."""
    count, labels = cs_connected_components(
        sparse.csr_matrix(adj), directed=directed, connection=connection
    )
    return ComponentResult(count=int(count), labels=np.asarray(labels, dtype=np.int64))


def shortest_paths(
    adj: object, indices: IntArray | None = None, method: str = "dijkstra", unweighted: bool = False
) -> ShortestPaths:
    """Pairwise shortest paths from ``indices`` to all nodes (Dijkstra or Bellman-Ford)."""
    matrix = sparse.csr_matrix(adj)
    source = None if indices is None else np.asarray(indices, dtype=np.int32)
    if method == "dijkstra":
        dist = cs_dijkstra(matrix, indices=source, unweighted=unweighted)
    elif method == "bellman_ford":
        dist = bellman_ford(matrix, indices=source)
    else:
        msg = f"unsupported shortest-path method: {method!r}"
        raise ValueError(msg)
    return ShortestPaths(method=method, distances=np.asarray(dist))


def single_source_shortest_paths(adj: object, source: int, method: str = "dijkstra") -> FloatArray:
    """Distance vector from one node to all others."""
    result = shortest_paths(adj, indices=np.array([source]), method=method)
    return result.distances[0]


def floyd_warshall_paths(adj: object) -> ShortestPaths:
    """All-pairs shortest paths via the Floyd-Warshall algorithm."""
    dist = cs_floyd_warshall(sparse.csr_matrix(adj))
    return ShortestPaths(method="floyd_warshall", distances=np.asarray(dist))


def bellman_ford_paths(adj: object, indices: IntArray | None = None) -> ShortestPaths:
    """Shortest paths tolerating negative weights (no negative cycles)."""
    return shortest_paths(adj, indices=indices, method="bellman_ford")


def minimum_spanning_forest(adj: object) -> sparse.csr_matrix:
    """Minimum spanning tree/forest of an undirected weighted graph."""
    symmetric = sparse.csr_matrix(adj).maximum(sparse.csr_matrix(adj).T)
    upper = sparse.triu(symmetric, format="csr")
    return sparse.csr_matrix(minimum_spanning_tree(upper))


def pagerank(
    adj: object, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-10
) -> FloatArray:
    """PageRank scores via power iteration on the random-surfer distribution."""
    matrix = sparse.coo_matrix(adj).tocsr()
    n = matrix.shape[0]
    if n == 0:
        return np.zeros(0)
    if not (0.0 < damping < 1.0):
        msg = "damping must be strictly between 0 and 1"
        raise ValueError(msg)
    out_degree = np.asarray((matrix != 0).sum(axis=1)).ravel().astype(float)
    dangling = out_degree == 0
    weights = matrix.multiply(1.0 / np.where(dangling, 1.0, out_degree)[:, None]).tocsr()
    rank_vec = np.full(n, 1.0 / n)
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        follow = damping * (weights.T @ rank_vec)
        dangling_mass = float(rank_vec[dangling].sum())
        new_rank = follow + damping * dangling_mass / n + teleport
        delta = np.abs(new_rank - rank_vec).max()
        rank_vec = new_rank / new_rank.sum()
        if delta < tol:
            break
    return rank_vec


def topological_order(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """Kahn topological order of a DAG; raises ValueError when a cycle exists."""
    if n <= 0:
        msg = "n must be positive"
        raise ValueError(msg)
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
    indegree = np.zeros(n, dtype=np.int64)
    for u, v in edges:
        adjacency[u].append(v)
        indegree[v] += 1
    queue: deque[int] = deque(int(i) for i in np.flatnonzero(indegree == 0))
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adjacency[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    if len(order) != n:
        msg = "cycle detected: graph is not a DAG"
        raise ValueError(msg)
    return order
