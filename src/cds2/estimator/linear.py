"""Linear regression estimators with sklearn-compatible API."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import BaseEstimator

__all__ = ["LinearRegressionGD", "RidgeSGD"]


def _validate_gd_hyperparameters(learning_rate: float, max_iter: int, tol: float) -> None:
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be a positive finite number")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be a non-negative finite number")


class LinearRegressionGD(BaseEstimator):
    """Ordinary least squares via batch gradient descent.

    Mirrors ``sklearn.linear_model.LinearRegression`` (no intercept penalty,
    analytic normal equations not used — this is the iterative version).
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
        _validate_gd_hyperparameters(self.learning_rate, self.max_iter, self.tol)
        X, checked_y = self._check_X_y(X, y)
        assert checked_y is not None
        n, d = X.shape
        x_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(checked_y))
        x_centered = X - x_mean
        y_centered = checked_y - y_mean
        coef = np.zeros(d)
        for _ in range(self.max_iter):
            residual = x_centered @ coef - y_centered
            grad = (2.0 / n) * (x_centered.T @ residual)
            new_coef = coef - self.learning_rate * grad
            if not bool(np.all(np.isfinite(new_coef))):
                raise FloatingPointError("gradient descent diverged to non-finite coefficients")
            delta = float(np.linalg.norm(new_coef - coef))
            coef = new_coef
            if delta < self.tol:
                break
        self.coef_ = coef
        self.intercept_ = y_mean - float(x_mean @ coef)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model not fitted")
        Xc = self._check_X(X)
        result: np.ndarray = Xc @ self.coef_ + self.intercept_
        return result

    def score(self, X: Any, y: Any) -> float:
        y_pred = self.predict(X)
        truth = np.asarray(y, dtype=float)
        if truth.ndim != 1 or truth.shape != y_pred.shape:
            raise ValueError("y must be 1-D and match the number of predictions")
        ss_res = float(np.sum((truth - y_pred) ** 2))
        ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class RidgeSGD(BaseEstimator):
    """L2-regularized linear regression (ridge) via gradient descent."""

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
        _validate_gd_hyperparameters(self.learning_rate, self.max_iter, self.tol)
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be a non-negative finite number")
        X, checked_y = self._check_X_y(X, y)
        assert checked_y is not None
        n, d = X.shape
        x_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(checked_y))
        x_centered = X - x_mean
        y_centered = checked_y - y_mean
        coef = np.zeros(d)
        for _ in range(self.max_iter):
            residual = x_centered @ coef - y_centered
            grad = (2.0 / n) * (x_centered.T @ residual) + 2.0 * self.alpha * coef
            new_coef = coef - self.learning_rate * grad
            if not bool(np.all(np.isfinite(new_coef))):
                raise FloatingPointError("gradient descent diverged to non-finite coefficients")
            delta = float(np.linalg.norm(new_coef - coef))
            coef = new_coef
            if delta < self.tol:
                break
        self.coef_ = coef
        self.intercept_ = y_mean - float(x_mean @ coef)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model not fitted")
        Xc = self._check_X(X)
        result: np.ndarray = Xc @ self.coef_ + self.intercept_
        return result

    def score(self, X: Any, y: Any) -> float:
        y_pred = self.predict(X)
        truth = np.asarray(y, dtype=float)
        if truth.ndim != 1 or truth.shape != y_pred.shape:
            raise ValueError("y must be 1-D and match the number of predictions")
        ss_res = float(np.sum((truth - y_pred) ** 2))
        ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
