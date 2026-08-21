"""Classical machine-learning models, utilities and metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

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
    """Split arrays into train/test partitions along axis 0."""
    prepared = [np.asarray(a) for a in arrays]
    n = prepared[0].shape[0]
    for a in prepared:
        if a.shape[0] != n:
            msg = "all arrays must share the same first dimension"
            raise ValueError(msg)
    if not 0 < test_size < 1:
        msg = "test_size must be in (0, 1)"
        raise ValueError(msg)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(indices)
    split = int(np.floor(n * (1 - test_size)))
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
        features = np.asarray(x, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        target = np.asarray(y, dtype=float).ravel()
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
        self.coef_ = weights
        self.intercept_ = bias
        return self

    def predict_proba(self, x: object) -> FloatArray:
        if self.coef_ is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        features = np.asarray(x, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        return _sigmoid(features @ self.coef_ + self.intercept_)

    def predict(self, x: object, threshold: float = 0.5) -> NDArray[np.int64]:
        return (self.predict_proba(x) >= threshold).astype(np.int64)

    def score(self, x: object, y: object) -> float:
        return accuracy_score(np.asarray(y), self.predict(x))


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
        if not 1 <= self.n_clusters <= points.shape[0]:
            msg = "n_clusters must be between 1 and number of samples"
            raise ValueError(msg)
        rng = np.random.default_rng(self.seed)
        centers = self._kmeans_pp_init(points, self.n_clusters, rng)
        labels = np.zeros(points.shape[0], dtype=np.int64)
        for _ in range(self.max_iter):
            distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for k in range(self.n_clusters):
                members = points[new_labels == k]
                if members.size:
                    new_centers[k] = members.mean(axis=0)
                else:
                    farthest = distances.min(axis=1).argmax()
                    new_centers[k] = points[farthest]
            shift = np.abs(new_centers - centers).max()
            centers = new_centers
            labels = new_labels
            if shift < self.tol:
                break
        inertia = float(((points - centers[labels]) ** 2).sum())
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        return self

    def predict(self, x: object) -> NDArray[np.int64]:
        if self.cluster_centers_ is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        points = np.asarray(x, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        distances = ((points[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

    @staticmethod
    def _kmeans_pp_init(
        points: FloatArray, n_clusters: int, rng: np.random.Generator
    ) -> FloatArray:
        n_samples = points.shape[0]
        chosen = [int(rng.integers(n_samples))]
        while len(chosen) < n_clusters:
            center_stack = points[np.asarray(chosen)]
            distances = (
                ((points[:, None, :] - center_stack[None, :, :]) ** 2).sum(axis=2).min(axis=1)
            )
            total = float(distances.sum())
            if total == 0.0:
                candidates = [i for i in range(n_samples) if i not in chosen]
                next_index = int(rng.choice(candidates)) if candidates else 0
            else:
                next_index = int(rng.choice(n_samples, p=distances / total))
            chosen.append(next_index)
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
        if arr.ndim != 2:
            msg = "PCA expects a 2-D array of shape (n_samples, n_features)"
            raise ValueError(msg)
        self.mean_ = arr.mean(axis=0)
        centered = arr - self.mean_
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)
        variance = s**2 / max(arr.shape[0] - 1, 1)
        total_variance = variance.sum()
        k = self.n_components
        if not 1 <= k <= arr.shape[1]:
            msg = "n_components must be between 1 and n_features"
            raise ValueError(msg)
        self.components_ = vt[:k]
        self.explained_variance_ratio_ = np.asarray(variance[:k] / total_variance)
        return self

    def transform(self, x: object) -> FloatArray:
        if self.components_ is None or self.mean_ is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        arr = np.asarray(x, dtype=float)
        return np.asarray((arr - self.mean_) @ self.components_.T)

    def fit_transform(self, x: object) -> FloatArray:
        return self.fit(x).transform(x)


class KNeighborsClassifier:
    """k-nearest-neighbour classifier backed by a KD-tree."""

    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors
        self._tree: cKDTree | None = None
        self._targets: NDArray[np.int64] | None = None

    def fit(self, x: object, y: object) -> KNeighborsClassifier:
        points = np.asarray(x, dtype=float)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        targets = np.asarray(y, dtype=np.int64).ravel()
        if not 1 <= self.n_neighbors <= points.shape[0]:
            msg = "n_neighbors must be between 1 and number of samples"
            raise ValueError(msg)
        self._tree = cKDTree(points)
        self._targets = targets
        return self

    def predict(self, x: object) -> NDArray[np.int64]:
        if self._tree is None or self._targets is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        queries = np.asarray(x, dtype=float)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        _distances, indices = self._tree.query(queries, k=self.n_neighbors)
        indices = np.atleast_2d(indices)
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


def accuracy_score(y_true: object, y_pred: object) -> float:
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    return float(np.mean(truth == predicted))


def _binary_counts(
    y_true: NDArray[np.int64], y_pred: NDArray[np.int64], pos_label: int
) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fp = int(np.sum((y_pred == pos_label) & (y_true != pos_label)))
    fn = int(np.sum((y_pred != pos_label) & (y_true == pos_label)))
    tn = int(np.sum((y_pred != pos_label) & (y_true != pos_label)))
    return tp, fp, fn, tn


def precision_score(
    y_true: object,
    y_pred: object,
    average: str = "binary",
    pos_label: int = 1,
) -> float:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if average == "binary":
        tp, fp, _fn, _tn = _binary_counts(truth, predicted, pos_label)
        return tp / (tp + fp) if tp + fp else 0.0
    scores: list[float] = []
    for label in np.unique(truth):
        tp, fp, _fn, _tn = _binary_counts(truth, predicted, int(label))
        scores.append(tp / (tp + fp) if tp + fp else 0.0)
    return float(np.mean(scores))


def recall_score(
    y_true: object,
    y_pred: object,
    average: str = "binary",
    pos_label: int = 1,
) -> float:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if average == "binary":
        tp, _fp, fn, _tn = _binary_counts(truth, predicted, pos_label)
        return tp / (tp + fn) if tp + fn else 0.0
    scores: list[float] = []
    for label in np.unique(truth):
        tp, _fp, fn, _tn = _binary_counts(truth, predicted, int(label))
        scores.append(tp / (tp + fn) if tp + fn else 0.0)
    return float(np.mean(scores))


def f1_score(
    y_true: object,
    y_pred: object,
    average: str = "binary",
    pos_label: int = 1,
) -> float:
    p = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    r = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    return 2 * p * r / (p + r) if p + r else 0.0


def confusion_matrix(
    y_true: object,
    y_pred: object,
    labels: list[int] | None = None,
) -> NDArray[np.int64]:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    unique = (
        sorted(set(map(int, np.unique(np.concatenate([truth, predicted])))))
        if labels is None
        else list(labels)
    )
    lookup = {label: index for index, label in enumerate(unique)}
    matrix = np.zeros((len(unique), len(unique)), dtype=np.int64)
    for t, p in zip(truth, predicted, strict=True):
        matrix[lookup[int(t)], lookup[int(p)]] += 1
    return matrix


def mean_squared_error(y_true: object, y_pred: object) -> float:
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return float(np.mean((truth - predicted) ** 2))


def root_mean_squared_error(y_true: object, y_pred: object) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: object, y_pred: object) -> float:
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(truth - predicted)))


def r2_score(y_true: object, y_pred: object) -> float:
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    ss_res = float(((truth - predicted) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0
