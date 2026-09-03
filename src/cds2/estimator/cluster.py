"""K-Means estimator with sklearn-compatible API."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import BaseEstimator

__all__ = ["KMeansSKL"]


class KMeansSKL(BaseEstimator):
    """K-Means clustering with the sklearn estimator interface.

    Wraps ``cds2.ml.KMeans`` (or the C kernel ``cds2._fast_kmeans`` when
    available) and exposes ``fit``, ``predict`` and ``fit_predict``.

    Args:
        n_clusters: number of clusters (k).
        max_iter: maximum Lloyd iterations.
        tol: convergence tolerance on centroid shift.
        seed: random seed for centroid initialization.
    """

    _fit_params = ["n_clusters", "max_iter", "tol", "seed"]

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        seed: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.labels_: np.ndarray | None = None
        self.cluster_centers_: np.ndarray | None = None
        self.inertia_: float = 0.0

    def fit(self, X: Any, y: Any = None) -> KMeansSKL:
        X = self._check_X(X)
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        idx = rng.choice(n, size=self.n_clusters, replace=False)
        centers = X[idx].copy()

        for _ in range(self.max_iter):
            # Assignment step.
            dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
            labels = np.argmin(dists, axis=1)
            # Update step.
            new_centers = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            # Handle empty clusters.
            for k in range(self.n_clusters):
                if np.isnan(new_centers[k]).any():
                    new_centers[k] = X[rng.integers(n)]
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(np.sum((X - centers[labels]) ** 2))
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("model not fitted")
        X = self._check_X(X)
        dists = np.linalg.norm(X[:, None] - self.cluster_centers_[None, :], axis=2)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        self.fit(X)
        return self.labels_
