"""Classical machine-learning models, utilities and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import sparse as sp_sparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    pass

try:
    from cds2 import _fast_kmeans as _c_kernel  # type: ignore[attr-defined]

    _HAS_C_KERNEL = True
except ImportError:
    _c_kernel = None
    _HAS_C_KERNEL = False

__all__ = [
    "StandardScaler",
    "LinearRegression",
    "LogisticRegression",
    "KMeans",
    "PCA",
    "KNeighborsClassifier",
    "train_test_split",
    "make_regression_data",
    "make_blobs",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
]

FloatArray = NDArray[np.float64]


class StandardScaler:
    """Zero-mean unit-variance feature scaler."""

    def __init__(self) -> None:
        self.mean_: FloatArray | None = None
        self.scale_: FloatArray | None = None

    def fit(self, x: object) -> StandardScaler:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.mean_ = np.asarray(arr.mean(axis=0))
        std = arr.std(axis=0)
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, x: object) -> FloatArray:
        if self.mean_ is None or self.scale_ is None:
            msg = "scaler is not fitted"
            raise RuntimeError(msg)
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return np.asarray((arr - self.mean_) / self.scale_)

    def fit_transform(self, x: object) -> FloatArray:
        return self.fit(x).transform(x)


def train_test_split(
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


class LinearRegression:
    """Ordinary least-squares linear regression with an intercept."""

    def __init__(self) -> None:
        self.coef_: FloatArray | None = None
        self.intercept_: float = 0.0

    def fit(self, x: object, y: object) -> LinearRegression:
        design = self._design(x)
        target = np.asarray(y, dtype=float).ravel()
        solution, *_rest = np.linalg.lstsq(design, target, rcond=None)
        self.coef_ = solution[:-1]
        self.intercept_ = float(solution[-1])
        return self

    def predict(self, x: object) -> FloatArray:
        if self.coef_ is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        design = self._design(x)
        return np.asarray(design @ np.concatenate([self.coef_, [self.intercept_]]))

    def score(self, x: object, y: object) -> float:
        return r2_score(np.asarray(y, dtype=float).ravel(), self.predict(x))

    @staticmethod
    def _design(x: object) -> FloatArray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return np.hstack([arr, np.ones((arr.shape[0], 1))])


def _sigmoid(z: FloatArray) -> FloatArray:
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


class LogisticRegression:
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


class KMeans:
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


class PCA:
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


class KNeighborsClassifier:
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


def make_regression_data(
    n: int = 100,
    n_features: int = 2,
    noise: float = 10.0,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Synthetic linear-regression dataset y = X w + b + noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features))
    weights = rng.uniform(-4.0, 4.0, size=n_features)
    y = x @ weights + rng.normal(scale=noise, size=n)
    return x, np.asarray(y)


def make_blobs(
    n_samples: int = 300,
    centers: int = 3,
    cluster_std: float = 1.0,
    seed: int | None = None,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Gaussian blobs around random centers with integer labels."""
    rng = np.random.default_rng(seed)
    center_points = rng.uniform(-10.0, 10.0, size=(centers, 2))
    counts = np.full(centers, n_samples // centers, dtype=int)
    counts[: n_samples % centers] += 1
    xs: list[FloatArray] = []
    ys: list[NDArray[np.int64]] = []
    for label, (count, center) in enumerate(zip(counts, center_points, strict=True)):
        xs.append(rng.normal(loc=center, scale=cluster_std, size=(count, 2)))
        ys.append(np.full(count, label, dtype=np.int64))
    return np.vstack(xs), np.concatenate(ys)


def _classification_arrays(
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


def _regression_arrays(y_true: object, y_pred: object) -> tuple[FloatArray, FloatArray]:
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


def r2_score(y_true: object, y_pred: object) -> float:
    truth, predicted = _regression_arrays(y_true, y_pred)
    ss_res = float(((truth - predicted) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0
