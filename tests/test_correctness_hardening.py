"""Regression tests for correctness and native-boundary hardening."""

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
