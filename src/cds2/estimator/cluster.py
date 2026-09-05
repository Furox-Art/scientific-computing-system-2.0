"""K-Means estimator with sklearn-compatible API."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._base import BaseEstimator

__all__ = ["KMeansSKL"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class KMeansSKL(BaseEstimator):
    """K-Means clustering with the sklearn estimator interface."""

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
        self.labels_: IntArray | None = None
        self.cluster_centers_: FloatArray | None = None
        self.inertia_: float = 0.0

    def fit(self, X: Any, y: Any = None) -> KMeansSKL:
        del y
        from ..ml import KMeans

        checked = self._check_X(X)
        model = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            seed=self.seed,
        ).fit(checked)
        assert model.cluster_centers_ is not None
        assert model.labels_ is not None
        assert model.inertia_ is not None
        self.cluster_centers_ = np.asarray(model.cluster_centers_, dtype=np.float64).copy()
        self.labels_ = np.asarray(model.labels_, dtype=np.int64).copy()
        self.inertia_ = float(model.inertia_)
        return self

    def predict(self, X: Any) -> IntArray:
        if self.cluster_centers_ is None:
            raise RuntimeError("model not fitted")
        Xc = self._check_X(X)
        if Xc.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        squared = np.sum(
            (Xc[:, None, :] - self.cluster_centers_[None, :, :]) ** 2,
            axis=2,
        )
        result: IntArray = np.asarray(np.argmin(squared, axis=1), dtype=np.int64)
        return result

    def fit_predict(self, X: Any, y: Any = None) -> IntArray:
        self.fit(X, y)
        return self.predict(X)
