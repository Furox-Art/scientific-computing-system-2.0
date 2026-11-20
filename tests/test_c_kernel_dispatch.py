"""C-kernel dispatch success paths for pure and compiled builds.

The compiled ``_fast_kmeans`` / ``_fast_pagerank`` kernels exist only in
compiled builds; CDS_PURE builds fall back to NumPy. These tests exercise
both dispatch branches in either environment: a faithful NumPy stub kernel
injected into ``sys.modules`` (reloading the module exactly like
``test_coverage_gaps.py``) drives the kernel branch, and forcing the
dispatch flag off drives the pure branch against an independent reference.
Restoration is asserted against the environment's original state rather
than a hard-coded ``False``, so the tests pass with or without compiled
kernels. State is restored in ``finally`` so no other test sees the stub.
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


def _kmeans_points() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(-5, 0.5, (30, 2)), rng.normal(5, 0.5, (30, 2))])


def _dense_pagerank(
    adjacency: list[list[float]], damping: float = 0.85, max_iter: int = 100, tol: float = 1e-10
) -> np.ndarray:
    """Dense power iteration independent of the production CSR assembly."""
    weights = np.asarray(adjacency, dtype=np.float64)
    n = weights.shape[0]
    out_degree = weights.sum(axis=1)
    follow = weights / np.where(out_degree == 0.0, 1.0, out_degree)[:, None]
    dangling = np.flatnonzero(out_degree == 0.0)
    rank = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new_rank = damping * (follow.T @ rank)
        new_rank += damping * float(rank[dangling].sum()) / n
        new_rank += (1.0 - damping) / n
        delta = float(np.abs(new_rank - rank).max())
        rank = new_rank
        if delta < tol:
            break
    return rank / float(rank.sum())


class TestKmeansKernelDispatch:
    def test_kernel_success_path_matches_pure_lloyd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original_flag = ml_module._HAS_C_KERNEL
        original_kernel = ml_module._c_kernel
        points = _kmeans_points()
        # Force the pure branch for the reference so the comparison is
        # kernel-vs-pure in compiled builds too, not kernel-vs-kernel.
        monkeypatch.setattr(ml_module, "_HAS_C_KERNEL", False)
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
        assert ml_module._HAS_C_KERNEL is original_flag
        assert ml_module._c_kernel is original_kernel

    def test_pure_path_matches_independent_lloyd_when_kernel_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        points = _kmeans_points()
        init = ml_module.KMeans(n_clusters=2, seed=3)._kmeans_pp_init(
            points, 2, np.random.default_rng(3)
        )
        _labels_buffer, centers_buffer, _iterations = _stub_lloyd(points, init, 300, 1e-8)
        expected_centers = np.frombuffer(centers_buffer, dtype=np.float64).reshape(2, 2)

        def _forbidden_lloyd(*args: object, **kwargs: object) -> None:
            raise AssertionError("the C kernel must not run on the pure path")

        stub = types.ModuleType("cds2._fast_kmeans")
        stub.lloyd = _forbidden_lloyd  # type: ignore[attr-defined]
        monkeypatch.setattr(ml_module, "_HAS_C_KERNEL", False)
        monkeypatch.setattr(ml_module, "_c_kernel", stub, raising=False)
        fitted = ml_module.KMeans(n_clusters=2, seed=3).fit(points)
        assert fitted.inertia_ > 0
        assert set(np.unique(fitted.labels_)) == {0, 1}
        np.testing.assert_allclose(
            np.sort(fitted.cluster_centers_, axis=0),
            np.sort(expected_centers, axis=0),
            atol=1e-8,
        )


class TestPagerankKernelDispatch:
    def test_kernel_success_path_matches_pure_iteration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original_flag = graph_module._HAS_PR_KERNEL
        original_kernel = graph_module._pr_kernel
        adjacency = [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        # Pure reference with the kernel branch forced off (see KMeans above).
        monkeypatch.setattr(graph_module, "_HAS_PR_KERNEL", False)
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
        assert graph_module._HAS_PR_KERNEL is original_flag
        assert graph_module._pr_kernel is original_kernel

    def test_pure_path_matches_dense_reference_when_kernel_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adjacency = [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        def _forbidden_iterate(*args: object, **kwargs: object) -> None:
            raise AssertionError("the C kernel must not run on the pure path")

        stub = types.ModuleType("cds2._fast_pagerank")
        stub.iterate = _forbidden_iterate  # type: ignore[attr-defined]
        monkeypatch.setattr(graph_module, "_HAS_PR_KERNEL", False)
        monkeypatch.setattr(graph_module, "_pr_kernel", stub, raising=False)
        ranks = graph_module.pagerank(adjacency)
        np.testing.assert_allclose(ranks, _dense_pagerank(adjacency), atol=1e-8)
        assert float(ranks.sum()) == pytest.approx(1.0)
