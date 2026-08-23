"""Graph algorithms built on scipy.sparse.csgraph, plus PageRank."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

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
    "CentralityResult",
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
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
]

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

try:
    from cds2 import _fast_pagerank as _pr_kernel  # type: ignore[attr-defined]

    _HAS_PR_KERNEL = True
except ImportError:  # pragma: no cover - exercised on pure-Python builds
    _pr_kernel = None
    _HAS_PR_KERNEL = False


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


@dataclass(frozen=True)
class CentralityResult:
    """Per-node centrality scores."""

    scores: FloatArray


def from_edges(
    n: int,
    edges: Sequence[tuple[int, int] | tuple[int, int, float]],
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
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for edge in edges:
        rows.append(int(edge[0]))
        cols.append(int(edge[1]))
        data.append(float(edge[2]) if weighted and len(edge) > 2 else 1.0)
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
        dist: Any = cs_dijkstra(matrix, indices=source, unweighted=unweighted)
    elif method == "bellman_ford":
        dist = bellman_ford(matrix, indices=source)
    else:
        msg = f"unsupported shortest-path method: {method!r}"
        raise ValueError(msg)
    return ShortestPaths(method=method, distances=np.asarray(dist, dtype=float))


def single_source_shortest_paths(adj: object, source: int, method: str = "dijkstra") -> FloatArray:
    """Distance vector from one node to all others."""
    result = shortest_paths(adj, indices=np.array([source]), method=method)
    return np.asarray(result.distances[0], dtype=float)


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
    coo = sparse.coo_matrix(adj)
    n = coo.shape[0]
    if n == 0:
        return np.zeros(0)
    if not (0.0 < damping < 1.0):
        msg = "damping must be strictly between 0 and 1"
        raise ValueError(msg)
    source_nodes = coo.row.astype(np.int64, copy=False)
    weights = coo.data.astype(np.float64, copy=False)
    out_degree = np.bincount(source_nodes, minlength=n).astype(float)

    # Transposed normalized CSR built with plain numpy: row j lists the
    # incoming links of node j (sources), so one sweep per iteration runs
    # the whole follow step - no scipy transpose/copy round-trips.
    normalized = weights / np.where(out_degree == 0.0, 1.0, out_degree)[source_nodes]
    order = np.argsort(coo.col, kind="stable")
    sorted_targets = coo.col[order]
    follow_indices = source_nodes[order]
    follow_data = np.ascontiguousarray(normalized[order])
    follow_indptr = np.searchsorted(sorted_targets, np.arange(n + 1)).astype(np.int64)
    dangling_indices = np.flatnonzero(out_degree == 0).astype(np.int64)

    if _HAS_PR_KERNEL and _pr_kernel is not None:
        rank_buffer, _iterations = _pr_kernel.iterate(
            follow_indptr,
            follow_indices,
            follow_data,
            n,
            damping,
            dangling_indices,
            max_iter,
            tol,
        )
        rank_vec = np.frombuffer(rank_buffer, dtype=np.float64).copy()
        rank_vec = np.frombuffer(rank_buffer, dtype=np.float64).copy()
        return np.asarray(rank_vec / rank_vec.sum(), dtype=float)

    rank_vec = np.full(n, 1.0 / n)
    follow_matrix = sparse.csr_matrix((follow_data, follow_indices, follow_indptr), shape=(n, n))
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        dangling_mass = float(rank_vec.take(dangling_indices).sum())
        new_rank = damping * (follow_matrix @ rank_vec)
        new_rank += damping * dangling_mass / n
        new_rank += teleport
        delta = float(np.abs(new_rank - rank_vec).max())
        rank_vec = new_rank
        if delta < tol:
            break
    return np.asarray(rank_vec / rank_vec.sum(), dtype=float)


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


def betweenness_centrality(adj: object, normalized: bool = True) -> FloatArray:
    """Brandes betweenness centrality on the unweighted successor graph.

    Directed graphs use n=1 pairs; undirected graphs divide by two.
    """
    matrix = sparse.csr_matrix(adj)
    n = matrix.shape[0]
    if n == 0:
        return np.zeros(0)
    successors_list: list[list[int]] = [[] for _ in range(n)]
    coo = matrix.tocoo()
    for u, v in zip(coo.row.tolist(), coo.col.tolist(), strict=False):
        if u != v and matrix[u, v] != 0:
            successors_list[u].append(v)

    betweenness = np.zeros(n)
    for source in range(n):
        stack: deque[int] = deque()
        predecessors: list[list[int]] = [[] for _ in range(n)]
        path_counts = np.zeros(n)
        path_counts[source] = 1.0
        distances = np.full(n, -1.0)
        distances[source] = 0.0
        visit_queue: deque[int] = deque([source])
        while visit_queue:
            node = visit_queue.popleft()
            stack.append(node)
            for neighbor in successors_list[node]:
                if distances[neighbor] < 0.0:
                    distances[neighbor] = distances[node] + 1.0
                    visit_queue.append(neighbor)
                if distances[neighbor] == distances[node] + 1.0:
                    path_counts[neighbor] += path_counts[node]
                    predecessors[neighbor].append(node)
        dependencies = np.zeros(n)
        while stack:
            target = stack.pop()
            for predecessor in predecessors[target]:
                share = path_counts[predecessor] / path_counts[target]
                dependencies[predecessor] += share * (1.0 + dependencies[target])
            if target != source:
                betweenness[target] += dependencies[target]

    scale_denominator = (n - 1) * (n - 2) if n > 2 else 1
    if normalized:
        betweenness /= scale_denominator
    betweenness /= 2.0 if not directed_flag(matrix) else 1.0
    return np.asarray(betweenness, dtype=float)


def directed_flag(matrix: sparse.csr_matrix) -> bool:
    """True when the matrix is not symmetric."""
    difference = abs(matrix - matrix.T)
    return float(difference.sum()) > 1e-12


def closeness_centrality(adj: object, wf_improved: bool = True) -> FloatArray:
    """Wasserman-Faust closeness centrality from single-source distances."""
    matrix = sparse.csr_matrix(adj)
    n = matrix.shape[0]
    if n == 0:
        return np.zeros(0)
    distances = cs_dijkstra(matrix, unweighted=_is_unweighted(matrix))
    scores = np.zeros(n)
    for node in range(n):
        row = distances[node]
        reachable = row[np.isfinite(row) & (row > 0)]
        total = float(reachable.sum())
        if total > 0:
            raw_score = reachable.size / total
            if wf_improved and reachable.size < n - 1:
                raw_score *= reachable.size / (n - 1)
            scores[node] = raw_score
    return scores


def eigenvector_centrality(
    adj: object, max_iter: int = 200, tol: float = 1e-10, directed: bool | None = None
) -> FloatArray:
    """Eigenvector centrality of the largest (Perron) eigenvalue.

    Power iteration with an oscillation guard: bipartite graphs that fail to
    settle fall back to a Lanczos solve for the dominant eigenvector.
    """
    coo = sparse.coo_matrix(adj)
    n = coo.shape[0]
    if n == 0:
        return np.zeros(0)
    is_directed = directed if directed is not None else directed_flag(sparse.csr_matrix(adj))
    matrix = sparse.csr_matrix(coo)
    if is_directed:
        matrix = matrix.maximum(matrix.T).tocsr()
    vector = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new_vector = np.asarray(matrix @ vector, dtype=float).ravel()
        norm_value = float(np.linalg.norm(new_vector, 1))
        if norm_value == 0:
            return np.zeros(n)
        new_vector /= norm_value
        delta = float(np.abs(new_vector - vector).max())
        vector = new_vector
        if delta < tol:
            break
    else:
        from cds2.sparse import largest_eigenpairs

        result = largest_eigenpairs(matrix, k=1)
        vector = np.abs(np.asarray(result.eigenvectors[:, 0], dtype=float)).ravel()
        total = float(vector.sum())
        if total == 0:  # pragma: no cover - defensive: Lanczos returns a unit vector
            return np.zeros(n)
        return np.asarray(vector / total, dtype=float)
    total = vector.sum()
    return np.asarray(vector / total, dtype=float)
