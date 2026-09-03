"""Base class for cds2 estimators with sklearn-compatible API.

Provides ``get_params`` / ``set_params`` and a ``_check_X_y`` helper so the
concrete estimators only implement ``fit`` / ``predict`` / ``score``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class BaseEstimator:
    """Minimal sklearn-compatible base class.

    Concrete subclasses set ``_fit_params`` — the list of constructor kwargs
    that ``get_params`` / ``set_params`` should expose (sklearn convention).
    """

    _fit_params: list[str] = []

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor kwargs that define this estimator."""
        return {name: getattr(self, name) for name in self._fit_params}

    def set_params(self, **params: object) -> BaseEstimator:
        """Set constructor kwargs; returns self for chaining."""
        for key, value in params.items():
            if key not in self._fit_params:
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        return self

    @staticmethod
    def _check_X_y(X: Any, y: Any | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Coerce inputs to float64 arrays, validating shape."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if y is not None:
            y = np.asarray(y, dtype=float)
            if y.ndim != 1:
                raise ValueError("y must be 1-D")
            if y.shape[0] != X.shape[0]:
                raise ValueError("X and y must have the same number of samples")
        return X, y

    @staticmethod
    def _check_X(X: Any) -> np.ndarray:
        arr: np.ndarray = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        return arr
