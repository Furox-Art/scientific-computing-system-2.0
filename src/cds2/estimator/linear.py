"""Linear regression estimators with sklearn-compatible API."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import BaseEstimator

__all__ = ["LinearRegressionGD", "RidgeSGD"]


class LinearRegressionGD(BaseEstimator):
    """Ordinary least squares via batch gradient descent.

    Mirrors ``sklearn.linear_model.LinearRegression`` (no intercept penalty,
    analytic normal equations not used — this is the iterative version).

    Args:
        learning_rate: step size for gradient descent.
        max_iter: maximum iterations.
        tol: convergence tolerance on coefficient change.
    """

    _fit_params = ["learning_rate", "max_iter", "tol"]

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: Any, y: Any) -> LinearRegressionGD:
        X, y = self._check_X_y(X, y)
        n, d = X.shape
        self.coef_ = np.zeros(d)
        self.intercept_ = float(np.mean(y))
        y_centered = y - self.intercept_
        for _ in range(self.max_iter):
            residual = X @ self.coef_ - y_centered
            grad = (2.0 / n) * (X.T @ residual)
            new_coef = self.coef_ - self.learning_rate * grad
            if np.linalg.norm(new_coef - self.coef_) < self.tol:
                break
            self.coef_ = new_coef
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model not fitted")
        X = self._check_X(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X: Any, y: Any) -> float:
        """R² coefficient of determination."""
        y_pred = self.predict(X)
        ss_res = float(np.sum((np.asarray(y) - y_pred) ** 2))
        ss_tot = float(np.sum((np.asarray(y) - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class RidgeSGD(BaseEstimator):
    """L2-regularized linear regression (ridge) via gradient descent.

    Args:
        alpha: regularization strength (>= 0).
        learning_rate: step size.
        max_iter: maximum iterations.
        tol: convergence tolerance.
    """

    _fit_params = ["alpha", "learning_rate", "max_iter", "tol"]

    def __init__(
        self,
        alpha: float = 1.0,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: Any, y: Any) -> RidgeSGD:
        X, y = self._check_X_y(X, y)
        n, d = X.shape
        self.coef_ = np.zeros(d)
        self.intercept_ = float(np.mean(y))
        y_centered = y - self.intercept_
        for _ in range(self.max_iter):
            residual = X @ self.coef_ - y_centered
            grad = (2.0 / n) * (X.T @ residual) + 2.0 * self.alpha * self.coef_
            new_coef = self.coef_ - self.learning_rate * grad
            if np.linalg.norm(new_coef - self.coef_) < self.tol:
                break
            self.coef_ = new_coef
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model not fitted")
        X = self._check_X(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X: Any, y: Any) -> float:
        y_pred = self.predict(X)
        ss_res = float(np.sum((np.asarray(y) - y_pred) ** 2))
        ss_tot = float(np.sum((np.asarray(y) - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
