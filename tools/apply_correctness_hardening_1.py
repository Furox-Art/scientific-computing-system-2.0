from __future__ import annotations

from pathlib import Path


def replace_between(path: str, start: str, end: str, replacement: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    left = text.index(start)
    right = text.index(end, left)
    target.write_text(text[:left] + replacement + text[right:], encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"expected one match in {path!r}, got {text.count(old)}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


Path("src/cds2/estimator/_base.py").write_text('''"""Base class for cds2 estimators with sklearn-compatible API.

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
        del deep
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
        """Coerce inputs to float64 arrays, validating shape and finiteness."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if not bool(np.all(np.isfinite(X))):
            raise ValueError("X must contain only finite values")
        if y is not None:
            y = np.asarray(y, dtype=float)
            if y.ndim != 1:
                raise ValueError("y must be 1-D")
            if y.shape[0] != X.shape[0]:
                raise ValueError("X and y must have the same number of samples")
            if not bool(np.all(np.isfinite(y))):
                raise ValueError("y must contain only finite values")
        return X, y

    @staticmethod
    def _check_X(X: Any) -> np.ndarray:
        arr: np.ndarray = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError("X must contain only finite values")
        return arr
''', encoding="utf-8")

Path("src/cds2/estimator/linear.py").write_text('''"""Linear regression estimators with sklearn-compatible API."""

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
''', encoding="utf-8")

Path("src/cds2/estimator/cluster.py").write_text('''"""K-Means estimator with sklearn-compatible API."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import BaseEstimator

__all__ = ["KMeansSKL"]


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
        self.labels_: np.ndarray | None = None
        self.cluster_centers_: np.ndarray | None = None
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
        self.cluster_centers_ = model.cluster_centers_.copy()
        self.labels_ = model.labels_.copy()
        self.inertia_ = float(model.inertia_)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("model not fitted")
        Xc = self._check_X(X)
        if Xc.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        squared = np.sum((Xc[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=2)
        labels: np.ndarray = np.argmin(squared, axis=1)
        return labels

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_.copy()
''', encoding="utf-8")

Path("src/cds2/linalg.py").write_text('''"""Dense linear algebra with typed dataclass results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg as sla

__all__ = [
    "EigenResult",
    "SVDResult",
    "LeastSquaresResult",
    "solve",
    "det",
    "inv",
    "pinv",
    "eig",
    "eigh",
    "svd",
    "norm",
    "trace",
    "matrix_power",
    "rank",
    "cond",
    "cholesky",
    "lstsq",
    "expm",
    "logm",
    "sqrtm",
]


@dataclass(frozen=True)
class EigenResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass(frozen=True)
class SVDResult:
    u: np.ndarray
    s: np.ndarray
    vh: np.ndarray


@dataclass(frozen=True)
class LeastSquaresResult:
    solution: np.ndarray
    residuals: np.ndarray
    rank: int
    singular_values: np.ndarray


def _as_numeric(a: object) -> np.ndarray:
    raw = np.asarray(a)
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    return np.asarray(a, dtype=dtype)


def _as_matrix(a: object, name: str) -> np.ndarray:
    arr = _as_numeric(a)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D array")
    return arr


def solve(a: object, b: object) -> np.ndarray:
    a_arr = _as_matrix(a, "a")
    b_arr = _as_numeric(b)
    return np.asarray(np.linalg.solve(a_arr, b_arr))


def det(a: object) -> float | complex:
    value = np.linalg.det(_as_matrix(a, "a"))
    return complex(value) if np.iscomplexobj(value) else float(value)


def inv(a: object) -> np.ndarray:
    return np.asarray(np.linalg.inv(_as_matrix(a, "a")))


def pinv(a: object, rcond: float | None = None) -> np.ndarray:
    arr = _as_numeric(a)
    return np.asarray(np.linalg.pinv(arr) if rcond is None else np.linalg.pinv(arr, rcond=rcond))


def eig(a: object) -> EigenResult:
    values, vectors = np.linalg.eig(_as_matrix(a, "a"))
    return EigenResult(eigenvalues=np.asarray(values), eigenvectors=np.asarray(vectors))


def eigh(a: object) -> EigenResult:
    values, vectors = np.linalg.eigh(_as_matrix(a, "a"))
    return EigenResult(eigenvalues=np.asarray(values), eigenvectors=np.asarray(vectors))


def svd(a: object, full_matrices: bool = True) -> SVDResult:
    arr = _as_numeric(a)
    if arr.ndim != 2:
        raise ValueError("a must be a 2-D array")
    u, s, vh = np.linalg.svd(arr, full_matrices=full_matrices)
    return SVDResult(u=np.asarray(u), s=np.asarray(s), vh=np.asarray(vh))


def norm(x: object, ord: float | Literal["fro", "nuc"] | None = None) -> float:  # noqa: A002
    value = _as_numeric(x)
    if value.ndim not in (1, 2):
        raise ValueError("norm expects a 1-D vector or 2-D matrix")
    return float(np.linalg.norm(value, ord=ord))


def trace(a: object) -> float | complex:
    value = np.trace(_as_matrix(a, "a"))
    return complex(value) if np.iscomplexobj(value) else float(value)


def matrix_power(a: object, n: int) -> np.ndarray:
    return np.asarray(np.linalg.matrix_power(_as_matrix(a, "a"), n))


def rank(a: object, tol: float | None = None) -> int:
    arr = _as_numeric(a)
    return int(np.linalg.matrix_rank(arr) if tol is None else np.linalg.matrix_rank(arr, tol=tol))


def cond(a: object) -> float:
    return float(np.linalg.cond(_as_matrix(a, "a")))


def cholesky(a: object) -> np.ndarray:
    return np.asarray(np.linalg.cholesky(_as_matrix(a, "a")))


def lstsq(a: object, b: object, rcond: float | None = None) -> LeastSquaresResult:
    a_arr = _as_numeric(a)
    b_arr = _as_numeric(b)
    if a_arr.ndim != 2:
        raise ValueError("a must be a 2-D array")
    solution, residuals, rank_value, singular = np.linalg.lstsq(a_arr, b_arr, rcond=rcond)
    return LeastSquaresResult(
        solution=np.asarray(solution),
        residuals=np.asarray(residuals),
        rank=int(rank_value),
        singular_values=np.asarray(singular),
    )


def expm(a: object) -> np.ndarray:
    return np.asarray(sla.expm(_as_matrix(a, "a")))


def logm(a: object) -> np.ndarray:
    return np.asarray(sla.logm(_as_matrix(a, "a")))


def sqrtm(a: object) -> np.ndarray:
    return np.asarray(sla.sqrtm(_as_matrix(a, "a")))
''', encoding="utf-8")

replace_between(
    "src/cds2/ml.py",
    "def train_test_split(\n",
    "\n\nclass LinearRegression:",
    '''def train_test_split(
    *arrays: object,
    test_size: float = 0.25,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], ...]:
    """Split arrays into non-empty train/test partitions along axis 0."""
    if not arrays:
        raise ValueError("at least one array is required")
    prepared = [np.asarray(a) for a in arrays]
    if any(a.ndim == 0 for a in prepared):
        raise ValueError("arrays must have at least one dimension")
    n = prepared[0].shape[0]
    for a in prepared:
        if a.shape[0] != n:
            raise ValueError("all arrays must share the same first dimension")
    if n < 2:
        raise ValueError("at least two samples are required")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(indices)
    split = int(np.floor(n * (1 - test_size)))
    split = min(max(split, 1), n - 1)
    train_idx, test_idx = indices[:split], indices[split:]
    parts: list[NDArray[np.float64]] = []
    for a in prepared:
        parts.append(a[train_idx])
        parts.append(a[test_idx])
    return tuple(parts)
''',
)

replace_between(
    "src/cds2/ml.py",
    "class LogisticRegression:\n",
    "\n\nclass KMeans:",
    '''class LogisticRegression:
    """Binary logistic regression trained by gradient descent on BCE loss."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 2000,
        l2: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.coef_: FloatArray | None = None
        self.intercept_: float = 0.0

    def fit(self, x: object, y: object) -> LogisticRegression:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive finite number")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if not np.isfinite(self.l2) or self.l2 < 0.0:
            raise ValueError("l2 must be a non-negative finite number")
        features = np.asarray(x, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("x must contain a non-empty 2-D feature matrix")
        if not bool(np.all(np.isfinite(features))):
            raise ValueError("x must contain only finite values")
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or target.shape[0] != features.shape[0]:
            raise ValueError("y must be 1-D and match the number of samples")
        if not bool(np.all(np.isfinite(target))) or not bool(np.all(np.isin(target, [0.0, 1.0]))):
            raise ValueError("y must contain only binary labels 0 and 1")
        n_samples, n_features = features.shape
        weights = np.zeros(n_features)
        bias = 0.0
        for _ in range(self.max_iter):
            logits = features @ weights + bias
            probs = _sigmoid(logits)
            error = probs - target
            grad_w = (features.T @ error) / n_samples + self.l2 * weights
            grad_b = float(np.mean(error))
            weights -= self.learning_rate * grad_w
            bias -= self.learning_rate * grad_b
            if not bool(np.all(np.isfinite(weights))) or not np.isfinite(bias):
                raise FloatingPointError("gradient descent diverged to non-finite parameters")
        self.coef_ = np.asarray(weights, dtype=np.float64)
        self.intercept_ = bias
        return self

    def predict_proba(self, x: object) -> FloatArray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        features = np.asarray(x, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if features.ndim != 2 or features.shape[1] != self.coef_.size:
            raise ValueError("x has a different number of features than the fitted data")
        if not bool(np.all(np.isfinite(features))):
            raise ValueError("x must contain only finite values")
        return _sigmoid(features @ self.coef_ + self.intercept_)

    def predict(self, x: object, threshold: float = 0.5) -> NDArray[np.int64]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return (self.predict_proba(x) >= threshold).astype(np.int64)

    def score(self, x: object, y: object) -> float:
        return accuracy_score(y, self.predict(x))
''',
)

replace_between(
    "src/cds2/ml.py",
    "class KMeans:\n",
    "\n\nclass PCA:",
    '''class KMeans:
    """Lloyd's k-means with k-means++ seeding."""

    def __init__(
        self, n_clusters: int = 8, max_iter: int = 300, tol: float = 1e-8, seed: int | None = None
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.cluster_centers_: FloatArray | None = None
        self.labels_: NDArray[np.int64] | None = None
        self.inertia_: float | None = None

    def fit(self, x: object) -> KMeans:
        points = np.asarray(x, dtype=float)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("x must contain a non-empty 2-D feature matrix")
        if not bool(np.all(np.isfinite(points))):
            raise ValueError("x must contain only finite values")
        n_samples = points.shape[0]
        if not 1 <= self.n_clusters <= n_samples:
            raise ValueError("n_clusters must be between 1 and number of samples")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if not np.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError("tol must be a non-negative finite number")
        rng = np.random.default_rng(self.seed)
        centers = self._kmeans_pp_init(points, self.n_clusters, rng)
        if _HAS_C_KERNEL and _c_kernel is not None:
            _labels, centers = self._run_c_lloyd(points, centers)
        else:
            _labels, centers = self._run_numpy_lloyd(points, centers)
        distances = cdist(points, centers, "sqeuclidean")
        labels = distances.argmin(axis=1).astype(np.int64)
        inertia = float(distances[np.arange(n_samples), labels].sum())
        self.cluster_centers_ = np.asarray(centers, dtype=np.float64)
        self.labels_ = labels
        self.inertia_ = inertia
        return self

    def _run_c_lloyd(
        self, points: FloatArray, centers: FloatArray
    ) -> tuple[NDArray[np.int64], FloatArray]:
        points_contiguous = np.ascontiguousarray(points, dtype=np.float64)
        centers_contiguous = np.ascontiguousarray(centers, dtype=np.float64)
        labels_buffer, centers_buffer, _iterations = _c_kernel.lloyd(
            points_contiguous,
            centers_contiguous,
            self.max_iter,
            self.tol,
        )
        labels = np.frombuffer(labels_buffer, dtype=np.int64).copy()
        fitted_centers = (
            np.frombuffer(centers_buffer, dtype=np.float64)
            .copy()
            .reshape(self.n_clusters, points.shape[1])
        )
        return labels, fitted_centers

    def _run_numpy_lloyd(
        self, points: FloatArray, centers: FloatArray
    ) -> tuple[NDArray[np.int64], FloatArray]:
        n_samples = points.shape[0]
        sample_indices = np.arange(n_samples)
        for _ in range(self.max_iter):
            distances = cdist(points, centers, "sqeuclidean")
            new_labels = distances.argmin(axis=1).astype(np.int64)
            counts = np.bincount(new_labels, minlength=self.n_clusters).astype(float)
            assignment = sp_sparse.csr_matrix(
                (np.ones(n_samples), (new_labels, sample_indices)),
                shape=(self.n_clusters, n_samples),
            )
            sums = np.asarray(assignment @ points)
            if (counts == 0).any():
                min_distances = distances[np.arange(n_samples), new_labels].copy()
                for cluster_index in np.flatnonzero(counts == 0):
                    candidates = np.flatnonzero(counts[new_labels] > 1.0)
                    farthest = int(candidates[np.argmax(min_distances[candidates])])
                    old_cluster = int(new_labels[farthest])
                    sums[old_cluster] -= points[farthest]
                    counts[old_cluster] -= 1.0
                    sums[cluster_index] = points[farthest]
                    counts[cluster_index] = 1.0
                    new_labels[farthest] = cluster_index
                    min_distances[farthest] = -1.0
            new_centers = sums / counts[:, None]
            shift = float(np.abs(new_centers - centers).max())
            centers = np.asarray(new_centers, dtype=np.float64)
            if shift < self.tol:
                break
        final_distances = cdist(points, centers, "sqeuclidean")
        labels = final_distances.argmin(axis=1).astype(np.int64)
        return labels, centers

    def predict(self, x: object) -> NDArray[np.int64]:
        if self.cluster_centers_ is None:
            raise RuntimeError("model is not fitted")
        points = np.asarray(x, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("x has a different number of features than the fitted data")
        if not bool(np.all(np.isfinite(points))):
            raise ValueError("x must contain only finite values")
        distances = cdist(points, self.cluster_centers_, "sqeuclidean")
        return np.asarray(distances.argmin(axis=1), dtype=np.int64)

    @staticmethod
    def _kmeans_pp_init(
        points: FloatArray, n_clusters: int, rng: np.random.Generator
    ) -> FloatArray:
        n_samples = points.shape[0]
        chosen = [int(rng.integers(n_samples))]
        closest = cdist(points, points[chosen], "sqeuclidean").ravel()
        while len(chosen) < n_clusters:
            total = float(closest.sum())
            if total == 0.0:
                taken = set(chosen)
                candidates = [i for i in range(n_samples) if i not in taken]
                next_index = int(rng.choice(candidates)) if candidates else 0
            else:
                next_index = int(rng.choice(n_samples, p=closest / total))
            chosen.append(next_index)
            closest = np.minimum(
                closest, cdist(points, points[next_index : next_index + 1], "sqeuclidean").ravel()
            )
        return np.asarray(points[chosen], dtype=float)
''',
)

replace_between(
    "src/cds2/ml.py",
    "class PCA:\n",
    "\n\nclass KNeighborsClassifier:",
    '''class PCA:
    """Principal component analysis via SVD of the centered data."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.components_: FloatArray | None = None
        self.explained_variance_ratio_: FloatArray | None = None
        self.mean_: FloatArray | None = None

    def fit(self, x: object) -> PCA:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("PCA expects a non-empty 2-D array of shape (n_samples, n_features)")
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError("x must contain only finite values")
        k = self.n_components
        if not 1 <= k <= min(arr.shape):
            raise ValueError("n_components must be between 1 and min(n_samples, n_features)")
        self.mean_ = np.asarray(arr.mean(axis=0), dtype=np.float64)
        centered = arr - self.mean_
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)
        variance = s**2 / max(arr.shape[0] - 1, 1)
        total_variance = float(variance.sum())
        self.components_ = np.asarray(vt[:k], dtype=np.float64)
        self.explained_variance_ratio_ = (
            np.zeros(k, dtype=np.float64)
            if total_variance == 0.0
            else np.asarray(variance[:k] / total_variance, dtype=np.float64)
        )
        return self

    def transform(self, x: object) -> FloatArray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("model is not fitted")
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.mean_.size:
            raise ValueError("x has a different number of features than the fitted data")
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError("x must contain only finite values")
        return np.asarray((arr - self.mean_) @ self.components_.T, dtype=np.float64)

    def fit_transform(self, x: object) -> FloatArray:
        return self.fit(x).transform(x)
''',
)

replace_between(
    "src/cds2/ml.py",
    "class KNeighborsClassifier:\n",
    "\n\ndef make_regression_data(",
    '''class KNeighborsClassifier:
    """k-nearest-neighbour classifier backed by a KD-tree."""

    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors
        self._tree: cKDTree | None = None
        self._targets: NDArray[np.int64] | None = None
        self._n_features: int | None = None

    def fit(self, x: object, y: object) -> KNeighborsClassifier:
        points = np.asarray(x, dtype=float)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("x must contain a non-empty 2-D feature matrix")
        if not bool(np.all(np.isfinite(points))):
            raise ValueError("x must contain only finite values")
        targets = np.asarray(y, dtype=np.int64)
        if targets.ndim != 1 or targets.shape[0] != points.shape[0]:
            raise ValueError("y must be 1-D and match the number of samples")
        if not 1 <= self.n_neighbors <= points.shape[0]:
            raise ValueError("n_neighbors must be between 1 and number of samples")
        self._tree = cKDTree(points)
        self._targets = targets.copy()
        self._n_features = points.shape[1]
        return self

    def predict(self, x: object) -> NDArray[np.int64]:
        if self._tree is None or self._targets is None or self._n_features is None:
            raise RuntimeError("model is not fitted")
        queries = np.asarray(x, dtype=float)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or queries.shape[1] != self._n_features:
            raise ValueError("x has a different number of features than the fitted data")
        if not bool(np.all(np.isfinite(queries))):
            raise ValueError("x must contain only finite values")
        _distances, indices = self._tree.query(queries, k=self.n_neighbors)
        if self.n_neighbors == 1:
            indices = np.asarray(indices, dtype=np.int64).reshape(-1, 1)
        else:
            indices = np.asarray(indices, dtype=np.int64)
        predictions = np.zeros(queries.shape[0], dtype=np.int64)
        for row_index, neighbor_indices in enumerate(indices):
            votes = self._targets[neighbor_indices]
            values, counts = np.unique(votes, return_counts=True)
            predictions[row_index] = values[counts.argmax()]
        return predictions
''',
)

replace_between(
    "src/cds2/ml.py",
    "def accuracy_score(y_true: object, y_pred: object) -> float:\n",
    "\n\ndef make_regression_data(" if False else "\n\ndef mean_squared_error(",
    '''def _classification_arrays(
    y_true: object, y_pred: object
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or predicted.ndim != 1:
        raise ValueError("classification targets must be 1-D")
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if truth.size == 0:
        raise ValueError("classification targets must not be empty")
    return truth, predicted


def accuracy_score(y_true: object, y_pred: object) -> float:
    truth, predicted = _classification_arrays(y_true, y_pred)
    return float(np.mean(truth == predicted))


def _binary_counts(
    y_true: NDArray[np.int64], y_pred: NDArray[np.int64], pos_label: int
) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fp = int(np.sum((y_pred == pos_label) & (y_true != pos_label)))
    fn = int(np.sum((y_pred != pos_label) & (y_true == pos_label)))
    tn = int(np.sum((y_pred != pos_label) & (y_true != pos_label)))
    return tp, fp, fn, tn


def _labels_for_average(
    truth: NDArray[np.int64], predicted: NDArray[np.int64], average: str
) -> NDArray[np.int64]:
    if average != "macro":
        raise ValueError("average must be 'binary' or 'macro'")
    return np.union1d(truth, predicted).astype(np.int64, copy=False)


def precision_score(
    y_true: object,
    y_pred: object,
    average: str = "binary",
    pos_label: int = 1,
) -> float:
    truth, predicted = _classification_arrays(y_true, y_pred)
    if average == "binary":
        tp, fp, _fn, _tn = _binary_counts(truth, predicted, pos_label)
        return tp / (tp + fp) if tp + fp else 0.0
    labels = _labels_for_average(truth, predicted, average)
    scores = []
    for label in labels:
        tp, fp, _fn, _tn = _binary_counts(truth, predicted, int(label))
        scores.append(tp / (tp + fp) if tp + fp else 0.0)
    return float(np.mean(scores))


def recall_score(
    y_true: object,
    y_pred: object,
    average: str = "binary",
    pos_label: int = 1,
) -> float:
    truth, predicted = _classification_arrays(y_true, y_pred)
    if average == "binary":
        tp, _fp, fn, _tn = _binary_counts(truth, predicted, pos_label)
        return tp / (tp + fn) if tp + fn else 0.0
    labels = _labels_for_average(truth, predicted, average)
    scores = []
    for label in labels:
        tp, _fp, fn, _tn = _binary_counts(truth, predicted, int(label))
        scores.append(tp / (tp + fn) if tp + fn else 0.0)
    return float(np.mean(scores))


def f1_score(
    y_true: object,
    y_pred: object,
    average: str = "binary",
    pos_label: int = 1,
) -> float:
    truth, predicted = _classification_arrays(y_true, y_pred)
    if average == "binary":
        tp, fp, fn, _tn = _binary_counts(truth, predicted, pos_label)
        denom = 2 * tp + fp + fn
        return 2 * tp / denom if denom else 0.0
    labels = _labels_for_average(truth, predicted, average)
    scores = []
    for label in labels:
        tp, fp, fn, _tn = _binary_counts(truth, predicted, int(label))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def confusion_matrix(
    y_true: object,
    y_pred: object,
    labels: list[int] | None = None,
) -> NDArray[np.int64]:
    truth, predicted = _classification_arrays(y_true, y_pred)
    unique = (
        sorted(set(map(int, np.unique(np.concatenate([truth, predicted])))))
        if labels is None
        else list(labels)
    )
    if len(set(unique)) != len(unique):
        raise ValueError("labels must be unique")
    lookup = {label: index for index, label in enumerate(unique)}
    if any(int(value) not in lookup for value in np.concatenate([truth, predicted])):
        raise ValueError("labels must include every value present in y_true and y_pred")
    matrix = np.zeros((len(unique), len(unique)), dtype=np.int64)
    for t, p in zip(truth, predicted, strict=True):
        matrix[lookup[int(t)], lookup[int(p)]] += 1
    return matrix
''',
)

replace_between(
    "src/cds2/ml.py",
    "def mean_squared_error(y_true: object, y_pred: object) -> float:\n",
    "\n\ndef r2_score(",
    '''def _regression_arrays(y_true: object, y_pred: object) -> tuple[FloatArray, FloatArray]:
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if truth.ndim != 1 or predicted.ndim != 1:
        raise ValueError("regression targets must be 1-D")
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if truth.size == 0:
        raise ValueError("regression targets must not be empty")
    if not bool(np.all(np.isfinite(truth))) or not bool(np.all(np.isfinite(predicted))):
        raise ValueError("regression targets must contain only finite values")
    return np.asarray(truth, dtype=np.float64), np.asarray(predicted, dtype=np.float64)


def mean_squared_error(y_true: object, y_pred: object) -> float:
    truth, predicted = _regression_arrays(y_true, y_pred)
    return float(np.mean((truth - predicted) ** 2))


def root_mean_squared_error(y_true: object, y_pred: object) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: object, y_pred: object) -> float:
    truth, predicted = _regression_arrays(y_true, y_pred)
    return float(np.mean(np.abs(truth - predicted)))
''',
)

replace_between(
    "src/cds2/ml.py",
    "def r2_score(y_true: object, y_pred: object) -> float:\n",
    "\n" if False else "__NO_END_MARKER__",
    "",
) if False else None
replace_once(
    "src/cds2/ml.py",
    '''def r2_score(y_true: object, y_pred: object) -> float:
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    ss_res = float(((truth - predicted) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0
''',
    '''def r2_score(y_true: object, y_pred: object) -> float:
    truth, predicted = _regression_arrays(y_true, y_pred)
    ss_res = float(((truth - predicted) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0
''',
)

replace_between(
    "src/cds2/graph.py",
    "def pagerank(\n",
    "\n\ndef topological_order(",
    '''def pagerank(
    adj: object, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-10
) -> FloatArray:
    """PageRank scores for a finite square graph with non-negative weights."""
    coo = sparse.coo_matrix(adj, dtype=np.float64)
    if coo.shape[0] != coo.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if not np.isfinite(damping) or not (0.0 < damping < 1.0):
        raise ValueError("damping must be strictly between 0 and 1")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive finite number")
    coo.sum_duplicates()
    weights = np.asarray(coo.data, dtype=np.float64)
    if not bool(np.all(np.isfinite(weights))):
        raise ValueError("edge weights must be finite")
    if bool(np.any(weights < 0.0)):
        raise ValueError("PageRank requires non-negative edge weights")
    n = coo.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    source_nodes = coo.row.astype(np.int64, copy=False)
    out_degree = np.bincount(source_nodes, weights=weights, minlength=n).astype(float)
    normalized = weights / np.where(out_degree == 0.0, 1.0, out_degree)[source_nodes]
    order = np.argsort(coo.col, kind="stable")
    sorted_targets = coo.col[order]
    follow_indices = np.ascontiguousarray(source_nodes[order], dtype=np.int64)
    follow_data = np.ascontiguousarray(normalized[order], dtype=np.float64)
    follow_indptr = np.ascontiguousarray(
        np.searchsorted(sorted_targets, np.arange(n + 1)).astype(np.int64)
    )
    dangling_indices = np.ascontiguousarray(np.flatnonzero(out_degree == 0), dtype=np.int64)

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
    else:
        rank_vec = np.full(n, 1.0 / n)
        follow_matrix = sparse.csr_matrix(
            (follow_data, follow_indices, follow_indptr), shape=(n, n)
        )
        teleport = (1.0 - damping) / n
        for _ in range(max_iter):
            dangling_mass = float(rank_vec.take(dangling_indices).sum())
            new_rank = damping * (follow_matrix @ rank_vec)
            new_rank += damping * dangling_mass / n
            new_rank += teleport
            delta = float(np.abs(new_rank - rank_vec).max())
            rank_vec = np.asarray(new_rank, dtype=np.float64)
            if delta < tol:
                break
    total = float(rank_vec.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("PageRank iteration produced an invalid probability vector")
    return np.asarray(rank_vec / total, dtype=float)
''',
)

Path("setup.py").write_text('''"""Build script for optional compiled accelerators.

Only the native kernels that are wired into public runtime paths are built:
``cds2._fast_kmeans`` and ``cds2._fast_pagerank``. Keeping unused native
extensions out of wheels reduces maintenance and memory-safety attack surface.
"""

import os
import sys

from setuptools import Extension, setup

extra_compile_args: list[str] = ["-O3"]
extra_link_args: list[str] = []

if os.environ.get("CDS_NO_OPENMP") != "1":
    if sys.platform == "linux":
        extra_compile_args += ["-fopenmp"]
        extra_link_args += ["-fopenmp"]
        if os.uname().machine.startswith("aarch64"):
            extra_compile_args += ["-march=armv8-a+fp+simd"]
    elif sys.platform == "darwin" and os.environ.get("CDS_WITH_LIBOMP") == "1":
        extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
        extra_link_args += ["-lomp"]
        if os.uname().machine.startswith("arm64"):
            extra_compile_args += ["-march=armv8-a+fp+simd"]

extensions: list[Extension] = []
if os.environ.get("CDS_PURE") != "1":
    common = dict(
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        optional=True,
    )
    extensions.extend(
        [
            Extension("cds2._fast_kmeans", sources=["src/cds2/src/_fast_kmeans.c"], **common),
            Extension("cds2._fast_pagerank", sources=["src/cds2/src/_fast_pagerank.c"], **common),
        ]
    )

setup(ext_modules=extensions)
''', encoding="utf-8")

for obsolete in (
    "src/cds2/src/_fast_linop.c",
    "src/cds2/src/_fast_integrate.c",
    "src/cds2/src/_fast_signal.c",
):
    Path(obsolete).unlink()

Path("src/cds2/src/_fast_kmeans.c").write_text(r'''/* Safe Lloyd iteration kernel for cds2.ml.KMeans. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static PyObject *lloyd(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *points_obj, *centers_obj;
    int max_iter;
    double tol;
    if (!PyArg_ParseTuple(args, "OOid", &points_obj, &centers_obj, &max_iter, &tol)) return NULL;

    Py_buffer points_view, centers_view;
    if (PyObject_GetBuffer(points_obj, &points_view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;
    if (PyObject_GetBuffer(centers_obj, &centers_view,
                           PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&points_view);
        return NULL;
    }
    if (points_view.format == NULL || points_view.ndim != 2 || strcmp(points_view.format, "d") != 0) {
        PyErr_SetString(PyExc_ValueError, "points must be a C-contiguous float64 (n, d) array");
        goto fail_views;
    }
    if (centers_view.format == NULL || centers_view.ndim != 2 || strcmp(centers_view.format, "d") != 0 ||
        centers_view.shape[1] != points_view.shape[1]) {
        PyErr_SetString(PyExc_ValueError, "centers must be a C-contiguous float64 (k, d) array matching the point dimension");
        goto fail_views;
    }

    const Py_ssize_t n = points_view.shape[0];
    const Py_ssize_t d = points_view.shape[1];
    const Py_ssize_t k = centers_view.shape[0];
    const double *points = (const double *)points_view.buf;
    double *centers = (double *)centers_view.buf;
    if (n <= 0 || k <= 0 || k > n || d <= 0) {
        PyErr_SetString(PyExc_ValueError, "need 1 <= k <= n and d >= 1");
        goto fail_views;
    }
    if (n > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "too many samples for the compiled kernel");
        goto fail_views;
    }
    if (max_iter < 1) {
        PyErr_SetString(PyExc_ValueError, "max_iter must be at least 1");
        goto fail_views;
    }
    if (!isfinite(tol) || tol < 0.0) {
        PyErr_SetString(PyExc_ValueError, "tol must be a non-negative finite number");
        goto fail_views;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        for (Py_ssize_t t = 0; t < d; t++) {
            if (!isfinite(points[i * d + t])) {
                PyErr_SetString(PyExc_ValueError, "points must contain only finite values");
                goto fail_views;
            }
        }
    }
    for (Py_ssize_t j = 0; j < k; j++) {
        for (Py_ssize_t t = 0; t < d; t++) {
            if (!isfinite(centers[j * d + t])) {
                PyErr_SetString(PyExc_ValueError, "centers must contain only finite values");
                goto fail_views;
            }
        }
    }

    size_t kd;
    if (!checked_mul_size((size_t)k, (size_t)d, &kd)) {
        PyErr_NoMemory();
        goto fail_views;
    }
    long long *labels = PyMem_Malloc((size_t)n * sizeof(long long));
    double *dmin = PyMem_Malloc((size_t)n * sizeof(double));
    double *sums = PyMem_Malloc(kd * sizeof(double));
    double *counts = PyMem_Malloc((size_t)k * sizeof(double));
    double *next_centers = PyMem_Malloc(kd * sizeof(double));
    if (labels == NULL || dmin == NULL || sums == NULL || counts == NULL || next_centers == NULL) {
        PyErr_NoMemory();
        goto fail_alloc;
    }

    int iterations = 0;
    const int row_count = (int)n;
    Py_BEGIN_ALLOW_THREADS
    for (int iter = 0; iter < max_iter; iter++) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < row_count; i++) {
            const double *row = points + (Py_ssize_t)i * d;
            double best_dist = DBL_MAX;
            Py_ssize_t best_index = 0;
            for (Py_ssize_t j = 0; j < k; j++) {
                const double *center_row = centers + j * d;
                double dist = 0.0;
                for (Py_ssize_t t = 0; t < d; t++) {
                    const double diff = row[t] - center_row[t];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_index = j;
                }
            }
            labels[i] = (long long)best_index;
            dmin[i] = best_dist;
        }

        memset(sums, 0, kd * sizeof(double));
        memset(counts, 0, (size_t)k * sizeof(double));
        for (Py_ssize_t i = 0; i < n; i++) {
            const Py_ssize_t label = (Py_ssize_t)labels[i];
            double *sum_row = sums + label * d;
            const double *row = points + i * d;
            for (Py_ssize_t t = 0; t < d; t++) sum_row[t] += row[t];
            counts[label] += 1.0;
        }

        for (Py_ssize_t j = 0; j < k; j++) {
            if (counts[j] > 0.0) continue;
            Py_ssize_t farthest = -1;
            double farthest_dist = -1.0;
            for (Py_ssize_t i = 0; i < n; i++) {
                const Py_ssize_t old = (Py_ssize_t)labels[i];
                if (counts[old] > 1.0 && dmin[i] > farthest_dist) {
                    farthest_dist = dmin[i];
                    farthest = i;
                }
            }
            if (farthest < 0) continue;
            const Py_ssize_t old = (Py_ssize_t)labels[farthest];
            for (Py_ssize_t t = 0; t < d; t++) {
                sums[old * d + t] -= points[farthest * d + t];
                sums[j * d + t] = points[farthest * d + t];
            }
            counts[old] -= 1.0;
            counts[j] = 1.0;
            labels[farthest] = (long long)j;
            dmin[farthest] = -1.0;
        }

        double shift = 0.0;
        for (Py_ssize_t j = 0; j < k; j++) {
            double *center_row = centers + j * d;
            double *next_row = next_centers + j * d;
            const double *sum_row = sums + j * d;
            for (Py_ssize_t t = 0; t < d; t++) {
                next_row[t] = sum_row[t] / counts[j];
                const double diff = fabs(next_row[t] - center_row[t]);
                if (diff > shift) shift = diff;
            }
        }
        memcpy(centers, next_centers, kd * sizeof(double));
        iterations = iter + 1;
        if (shift < tol) break;
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < row_count; i++) {
        const double *row = points + (Py_ssize_t)i * d;
        double best_dist = DBL_MAX;
        Py_ssize_t best_index = 0;
        for (Py_ssize_t j = 0; j < k; j++) {
            const double *center_row = centers + j * d;
            double dist = 0.0;
            for (Py_ssize_t t = 0; t < d; t++) {
                const double diff = row[t] - center_row[t];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_index = j;
            }
        }
        labels[i] = (long long)best_index;
    }
    Py_END_ALLOW_THREADS

    PyObject *labels_bytes = PyBytes_FromStringAndSize((const char *)labels, n * (Py_ssize_t)sizeof(long long));
    PyObject *centers_bytes = PyBytes_FromStringAndSize((const char *)centers, k * d * (Py_ssize_t)sizeof(double));
    PyMem_Free(labels); PyMem_Free(dmin); PyMem_Free(sums); PyMem_Free(counts); PyMem_Free(next_centers);
    PyBuffer_Release(&points_view); PyBuffer_Release(&centers_view);
    if (labels_bytes == NULL || centers_bytes == NULL) {
        Py_XDECREF(labels_bytes); Py_XDECREF(centers_bytes); return NULL;
    }
    return Py_BuildValue("NNi", labels_bytes, centers_bytes, iterations);

fail_alloc:
    PyMem_Free(labels); PyMem_Free(dmin); PyMem_Free(sums); PyMem_Free(counts); PyMem_Free(next_centers);
fail_views:
    PyBuffer_Release(&points_view); PyBuffer_Release(&centers_view); return NULL;
}

static PyMethodDef methods[] = {
    {"lloyd", lloyd, METH_VARARGS, "lloyd(points, centers_init, max_iter, tol) -> (labels_i64, centers_f64, iterations)"},
    {NULL, NULL, 0, NULL},
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "cds2._fast_kmeans", "Compiled Lloyd kernel.", -1, methods};
PyMODINIT_FUNC PyInit__fast_kmeans(void) { return PyModule_Create(&module); }
''', encoding="utf-8")

Path("src/cds2/src/_fast_pagerank.c").write_text(r'''/* Hardened PageRank power-iteration kernel for cds2.graph. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int require_buffer(PyObject *obj, Py_buffer *view, int want_double, const char *name) {
    if (PyObject_GetBuffer(obj, view, PyBUF_CONTIG_RO | PyBUF_FORMAT) < 0) return 0;
    int format_ok = want_double
        ? (view->format != NULL && strcmp(view->format, "d") == 0)
        : (view->format != NULL && (strcmp(view->format, "q") == 0 || strcmp(view->format, "l") == 0));
    if (!format_ok || view->ndim != 1) {
        PyErr_Format(PyExc_ValueError, "%s must be a contiguous 1-D %s array", name,
                     want_double ? "float64" : "int64");
        PyBuffer_Release(view);
        return 0;
    }
    return 1;
}

static PyObject *iterate(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *indptr_obj, *indices_obj, *data_obj, *dangling_obj;
    int n_nodes, max_iter;
    double damping, tol;
    if (!PyArg_ParseTuple(args, "OOOidOid", &indptr_obj, &indices_obj, &data_obj,
                          &n_nodes, &damping, &dangling_obj, &max_iter, &tol)) return NULL;

    Py_buffer indptr_view, indices_view, data_view, dangling_view;
    if (!require_buffer(indptr_obj, &indptr_view, 0, "indptr")) return NULL;
    if (!require_buffer(indices_obj, &indices_view, 0, "indices")) { PyBuffer_Release(&indptr_view); return NULL; }
    if (!require_buffer(data_obj, &data_view, 1, "data")) { PyBuffer_Release(&indptr_view); PyBuffer_Release(&indices_view); return NULL; }
    if (!require_buffer(dangling_obj, &dangling_view, 0, "dangling")) {
        PyBuffer_Release(&indptr_view); PyBuffer_Release(&indices_view); PyBuffer_Release(&data_view); return NULL;
    }

    if (n_nodes <= 0 || indptr_view.shape[0] != (Py_ssize_t)n_nodes + 1) {
        PyErr_SetString(PyExc_ValueError, "indptr must have exactly n + 1 entries"); goto fail_views;
    }
    if (!isfinite(damping) || !(0.0 < damping && damping < 1.0)) {
        PyErr_SetString(PyExc_ValueError, "damping must be strictly between 0 and 1"); goto fail_views;
    }
    if (max_iter < 1) { PyErr_SetString(PyExc_ValueError, "max_iter must be at least 1"); goto fail_views; }
    if (!isfinite(tol) || tol <= 0.0) { PyErr_SetString(PyExc_ValueError, "tol must be a positive finite number"); goto fail_views; }

    const int64_t *indptr = (const int64_t *)indptr_view.buf;
    const int64_t *indices = (const int64_t *)indices_view.buf;
    const double *weights = (const double *)data_view.buf;
    const int64_t *dangling = (const int64_t *)dangling_view.buf;
    const Py_ssize_t nnz = indices_view.shape[0];
    if (data_view.shape[0] != nnz) { PyErr_SetString(PyExc_ValueError, "indices and data must have the same length"); goto fail_views; }
    if (indptr[0] != 0 || indptr[n_nodes] != (int64_t)nnz) {
        PyErr_SetString(PyExc_ValueError, "indptr must start at 0 and end at nnz"); goto fail_views;
    }
    for (int j = 0; j < n_nodes; j++) {
        if (indptr[j] < 0 || indptr[j] > indptr[j + 1] || indptr[j + 1] > (int64_t)nnz) {
            PyErr_SetString(PyExc_ValueError, "indptr must be monotone and bounded by nnz"); goto fail_views;
        }
    }
    for (Py_ssize_t p = 0; p < nnz; p++) {
        if (indices[p] < 0 || indices[p] >= n_nodes) {
            PyErr_SetString(PyExc_ValueError, "indices contain a node outside 0..n-1"); goto fail_views;
        }
        if (!isfinite(weights[p]) || weights[p] < 0.0) {
            PyErr_SetString(PyExc_ValueError, "data must contain finite non-negative weights"); goto fail_views;
        }
    }
    const Py_ssize_t n_dangling = dangling_view.shape[0];
    for (Py_ssize_t p = 0; p < n_dangling; p++) {
        if (dangling[p] < 0 || dangling[p] >= n_nodes) {
            PyErr_SetString(PyExc_ValueError, "dangling contains a node outside 0..n-1"); goto fail_views;
        }
    }

    double *rank = PyMem_Malloc((size_t)n_nodes * sizeof(double));
    double *next = PyMem_Malloc((size_t)n_nodes * sizeof(double));
    if (rank == NULL || next == NULL) { PyErr_NoMemory(); goto fail_alloc; }
    const double initial = 1.0 / (double)n_nodes;
    for (int j = 0; j < n_nodes; j++) rank[j] = initial;
    const double teleport = (1.0 - damping) / (double)n_nodes;
    int iterations = 0;

    Py_BEGIN_ALLOW_THREADS
    for (int iter = 0; iter < max_iter; iter++) {
        double dangling_mass = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : dangling_mass)
#endif
        for (Py_ssize_t p = 0; p < n_dangling; p++) dangling_mass += rank[dangling[p]];
        const double uniform_add = damping * dangling_mass / (double)n_nodes + teleport;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < n_nodes; j++) {
            double acc = 0.0;
            for (int64_t p = indptr[j]; p < indptr[j + 1]; p++) acc += weights[p] * rank[indices[p]];
            next[j] = damping * acc + uniform_add;
        }
        double delta = 0.0;
        for (int j = 0; j < n_nodes; j++) {
            const double diff = fabs(next[j] - rank[j]);
            if (diff > delta) delta = diff;
        }
        double *tmp = rank; rank = next; next = tmp;
        iterations = iter + 1;
        if (delta < tol) break;
    }
    Py_END_ALLOW_THREADS

    PyObject *rank_bytes = PyBytes_FromStringAndSize((const char *)rank,
        (Py_ssize_t)n_nodes * (Py_ssize_t)sizeof(double));
    PyMem_Free(rank); PyMem_Free(next);
    PyBuffer_Release(&indptr_view); PyBuffer_Release(&indices_view);
    PyBuffer_Release(&data_view); PyBuffer_Release(&dangling_view);
    if (rank_bytes == NULL) return NULL;
    return Py_BuildValue("Ni", rank_bytes, iterations);

fail_alloc:
    PyMem_Free(rank); PyMem_Free(next);
fail_views:
    PyBuffer_Release(&indptr_view); PyBuffer_Release(&indices_view);
    PyBuffer_Release(&data_view); PyBuffer_Release(&dangling_view); return NULL;
}

static PyMethodDef methods[] = {
    {"iterate", iterate, METH_VARARGS, "iterate(indptr, indices, data, n, damping, dangling, max_iter, tol)"},
    {NULL, NULL, 0, NULL},
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "cds2._fast_pagerank", "Compiled PageRank kernel.", -1, methods};
PyMODINIT_FUNC PyInit__fast_pagerank(void) { return PyModule_Create(&module); }
''', encoding="utf-8")

replace_once(
    "tests/test_coverage_gaps.py",
    '''        assert f1 == pytest.approx(2 * precision * recall / (precision + recall))
''',
    '''        assert f1 == pytest.approx(0.6)
''',
)

replace_between(
    "tests/test_estimator.py",
    "    def test_empty_cluster_gets_reseeded(self) -> None:\n",
    "\n    def test_fit_loop_exhausts_without_converging",
    '''    def test_empty_cluster_gets_reseeded(self) -> None:
        X = np.zeros((6, 2))
        est = KMeansSKL(n_clusters=2, max_iter=2, tol=0.0, seed=0).fit(X)
        assert est.cluster_centers_ is not None
        assert est.labels_ is not None
        assert np.isfinite(est.cluster_centers_).all()
''',
)

Path("tests/test_correctness_hardening.py").write_text('''"""Regression tests for correctness and native-boundary hardening."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from cds2 import graph, linalg, ml
from cds2.estimator import KMeansSKL, LinearRegressionGD, RidgeSGD


def test_gradient_descent_regression_handles_shifted_features() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(loc=[50.0, -20.0], scale=[2.0, 3.0], size=(400, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 7.5
    model = LinearRegressionGD(learning_rate=0.02, max_iter=20_000, tol=1e-12).fit(X, y)
    assert model.coef_ == pytest.approx([3.0, -2.0], abs=2e-3)
    assert model.intercept_ == pytest.approx(7.5, abs=0.1)
    assert model.score(X, y) > 0.999999


def test_ridge_intercept_is_not_regularized_or_shift_biased() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(loc=100.0, scale=1.0, size=(300, 2))
    y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + 12.0
    model = RidgeSGD(alpha=1e-6, learning_rate=0.03, max_iter=20_000, tol=1e-12).fit(X, y)
    assert model.coef_ == pytest.approx([1.5, -0.5], abs=3e-3)
    assert model.intercept_ == pytest.approx(12.0, abs=0.5)


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegressionGD(learning_rate=0.0),
        LinearRegressionGD(max_iter=0),
        LinearRegressionGD(tol=-1.0),
        RidgeSGD(alpha=-1.0),
    ],
)
def test_gradient_descent_rejects_invalid_hyperparameters(estimator) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        estimator.fit(np.ones((4, 1)), np.arange(4.0))


def test_macro_f1_is_mean_of_per_class_f1() -> None:
    truth = [0, 1, 2, 2]
    predicted = [0, 2, 2, 2]
    assert ml.f1_score(truth, predicted, average="macro") == pytest.approx(0.6)


def test_macro_metrics_include_predicted_only_labels() -> None:
    truth = [0, 0, 1, 1]
    predicted = [0, 2, 1, 2]
    assert ml.precision_score(truth, predicted, average="macro") == pytest.approx(2.0 / 3.0)
    assert ml.recall_score(truth, predicted, average="macro") == pytest.approx(1.0 / 3.0)
    assert ml.f1_score(truth, predicted, average="macro") == pytest.approx(4.0 / 9.0)


def test_metrics_reject_broadcasting_and_invalid_average() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ml.mean_squared_error([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="same shape"):
        ml.accuracy_score([0, 1], [0])
    with pytest.raises(ValueError, match="average"):
        ml.f1_score([0, 1], [0, 1], average="weighted")


def test_complex_matrix_functions_preserve_imaginary_components() -> None:
    root = linalg.sqrtm([[-1.0]])
    assert np.iscomplexobj(root)
    assert np.allclose(root @ root, [[-1.0]], atol=1e-10)
    logarithm = linalg.logm([[-1.0]])
    assert np.iscomplexobj(logarithm)
    assert logarithm[0, 0].imag == pytest.approx(np.pi, abs=1e-10)
    solution = linalg.solve([[1.0 + 1.0j]], [2.0j])
    assert solution[0] == pytest.approx(1.0 + 1.0j)


def test_kmeans_validation_and_final_labels_match_centers() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        ml.KMeans(n_clusters=2, max_iter=0).fit(np.ones((4, 2)))
    with pytest.raises(ValueError, match="finite"):
        ml.KMeans(n_clusters=2).fit([[0.0], [np.nan]])
    rng = np.random.default_rng(3)
    points = np.vstack([rng.normal(-4.0, 0.3, (20, 2)), rng.normal(4.0, 0.3, (20, 2))])
    model = ml.KMeans(n_clusters=2, seed=1).fit(points)
    assert model.labels_ is not None
    assert np.array_equal(model.labels_, model.predict(points))


def test_kmeans_sklearn_wrapper_uses_consistent_core() -> None:
    points = np.array([[0.0], [0.0], [10.0], [10.0]])
    wrapped = KMeansSKL(n_clusters=2, seed=0).fit(points)
    assert wrapped.labels_ is not None
    assert np.array_equal(wrapped.labels_, wrapped.predict(points))


def test_pca_constant_data_has_finite_zero_explained_variance() -> None:
    fitted = ml.PCA(n_components=2).fit(np.ones((5, 3)))
    assert fitted.explained_variance_ratio_ is not None
    assert np.array_equal(fitted.explained_variance_ratio_, np.zeros(2))


def test_knn_rejects_mismatched_targets() -> None:
    with pytest.raises(ValueError, match="match"):
        ml.KNeighborsClassifier(1).fit([[0.0], [1.0]], [0])


def test_pagerank_rejects_invalid_graph_inputs() -> None:
    with pytest.raises(ValueError, match="square"):
        graph.pagerank(np.ones((2, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        graph.pagerank([[0.0, -1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        graph.pagerank([[0.0, np.nan], [1.0, 0.0]])
    with pytest.raises(ValueError, match="max_iter"):
        graph.pagerank(np.eye(2), max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        graph.pagerank(np.eye(2), tol=0.0)


def test_native_kmeans_boundary_guards_when_extension_available() -> None:
    try:
        kernel = importlib.import_module("cds2._fast_kmeans")
    except ImportError:
        pytest.skip("compiled k-means extension is not installed")
    points = np.ascontiguousarray([[0.0], [1.0]], dtype=np.float64)
    centers = np.ascontiguousarray([[0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="max_iter"):
        kernel.lloyd(points, centers, 0, 1e-8)


def test_native_pagerank_boundary_guards_when_extension_available() -> None:
    try:
        kernel = importlib.import_module("cds2._fast_pagerank")
    except ImportError:
        pytest.skip("compiled PageRank extension is not installed")
    indptr = np.ascontiguousarray([0, 2, 1], dtype=np.int64)
    indices = np.ascontiguousarray([0], dtype=np.int64)
    data = np.ascontiguousarray([1.0], dtype=np.float64)
    dangling = np.ascontiguousarray([], dtype=np.int64)
    with pytest.raises(ValueError):
        kernel.iterate(indptr, indices, data, 2, 0.85, dangling, 20, 1e-10)
''', encoding="utf-8")

print("Applied correctness hardening wave 1")
