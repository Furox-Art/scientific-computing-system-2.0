"""C-kernel dispatch success paths under a pure-Python build.

The compiled ``_fast_kmeans`` / ``_fast_pagerank`` kernels do not exist in
CDS_PURE builds, so the import-success branch and the kernel dispatch code
never run here (CI covers them with compiled kernels). These tests inject a
faithful NumPy stub kernel into ``sys.modules``, reload the module exactly
like ``test_coverage_gaps.py`` does for the ImportError branch, and verify
the dispatch produces the same numbers as the pure path. State is restored
in ``finally`` so no other test sees the stub.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import cds2
from cds2 import graph as graph_module
from cds2 import ml as ml_module


def _stub_lloyd(
    points: np.ndarray, centers: np.ndarray, max_iter: int, tol: float
) -> tuple[bytes, bytes, int]:
    """Plain NumPy Lloyd iteration returning the C kernel's buffer contract."""
    centers = np.array(centers, dtype=np.float64, copy=True)
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        labels = cdist(points, centers, "sqeuclidean").argmin(axis=1).astype(np.int64)
        new_centers = centers.copy()
        for k in range(centers.shape[0]):
            members = points[labels == k]
            if members.size:
                new_centers[k] = members.mean(axis=0)
        shift = float(np.abs(new_centers - centers).max())
        centers = new_centers
        if shift < tol:
            break
    labels = cdist(points, centers, "sqeuclidean").argmin(axis=1).astype(np.int64)
    return labels.tobytes(), np.ascontiguousarray(centers).tobytes(), iterations


def _stub_pagerank_iterate(
    follow_indptr: np.ndarray,
    follow_indices: np.ndarray,
    follow_data: np.ndarray,
    n: int,
    damping: float,
    dangling_indices: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[bytes, int]:
    """Power iteration mirroring the pure fallback, returning a rank buffer."""
    from scipy import sparse

    follow_matrix = sparse.csr_matrix((follow_data, follow_indices, follow_indptr), shape=(n, n))
    rank = np.full(n, 1.0 / n)
    teleport = (1.0 - damping) / n
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        dangling_mass = float(rank.take(dangling_indices).sum())
        new_rank = damping * (follow_matrix @ rank)
        new_rank += damping * dangling_mass / n
        new_rank += teleport
        delta = float(np.abs(new_rank - rank).max())
        rank = np.asarray(new_rank, dtype=np.float64)
        if delta < tol:
            break
    return rank.tobytes(), iterations


class TestKmeansKernelDispatch:
    def test_kernel_success_path_matches_pure_lloyd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rng = np.random.default_rng(0)
        points = np.vstack([rng.normal(-5, 0.5, (30, 2)), rng.normal(5, 0.5, (30, 2))])
        pure = ml_module.KMeans(n_clusters=2, seed=3).fit(points)

        stub = types.ModuleType("cds2._fast_kmeans")
        stub.lloyd = _stub_lloyd  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "cds2._fast_kmeans", stub)
        monkeypatch.setattr(cds2, "_fast_kmeans", stub, raising=False)
        reloaded = importlib.reload(ml_module)
        try:
            assert reloaded._HAS_C_KERNEL is True
            model = reloaded.KMeans(n_clusters=2, seed=3).fit(points)
            assert model.inertia_ > 0
            assert set(np.unique(model.labels_)) == {0, 1}
            np.testing.assert_allclose(
                np.sort(model.cluster_centers_, axis=0),
                np.sort(pure.cluster_centers_, axis=0),
                atol=1e-8,
            )
        finally:
            monkeypatch.undo()
            importlib.reload(ml_module)
        assert ml_module._HAS_C_KERNEL is False


class TestPagerankKernelDispatch:
    def test_kernel_success_path_matches_pure_iteration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adjacency = [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        pure = graph_module.pagerank(adjacency)

        stub = types.ModuleType("cds2._fast_pagerank")
        stub.iterate = _stub_pagerank_iterate  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "cds2._fast_pagerank", stub)
        monkeypatch.setattr(cds2, "_fast_pagerank", stub, raising=False)
        reloaded = importlib.reload(graph_module)
        try:
            assert reloaded._HAS_PR_KERNEL is True
            ranks = reloaded.pagerank(adjacency)
            np.testing.assert_allclose(ranks, pure, atol=1e-10)
            assert float(ranks.sum()) == pytest.approx(1.0)
        finally:
            monkeypatch.undo()
            importlib.reload(graph_module)
        assert graph_module._HAS_PR_KERNEL is False
