"""PCA estimator with sklearn-compatible API."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import BaseEstimator

__all__ = ["PCASKL"]


class PCASKL(BaseEstimator):
    """Principal component analysis via SVD.

    Mirrors ``sklearn.decomposition.PCA`` with ``fit``, ``transform`` and
    ``fit_transform``.

    Args:
        n_components: number of components to keep (<= n_features).
    """

    _fit_params = ["n_components"]

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None

    def fit(self, X: Any, y: Any = None) -> PCASKL:
        X = self._check_X(X)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        n_components = self.n_components or min(X.shape)
        self.components_ = Vt[:n_components]
        n_samples = X.shape[0]
        self.explained_variance_ = (s[:n_components] ** 2) / (n_samples - 1)
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("model not fitted")
        Xc = self._check_X(X)
        transformed: np.ndarray = (Xc - self.mean_) @ self.components_.T
        return transformed

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
