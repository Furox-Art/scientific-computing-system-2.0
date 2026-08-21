"""Tests for cds2.ml."""

import numpy as np
import pytest

from cds2 import ml


class TestStandardScaler:
    def test_zero_mean_unit_variance(self) -> None:
        x = np.random.default_rng(0).normal(loc=10.0, scale=5.0, size=(100, 3))
        scaled = ml.StandardScaler().fit(x).transform(x)
        assert np.allclose(scaled.mean(axis=0), 0.0)
        assert np.allclose(scaled.std(axis=0), 1.0)

    def test_constant_column_survives(self) -> None:
        x = np.array([[1.0, 5.0], [2.0, 5.0]])
        scaler = ml.StandardScaler().fit(x)
        assert np.allclose(scaler.transform(x)[:, 1], 0.0)

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            ml.StandardScaler().transform([[1.0]])


class TestTrainTestSplit:
    def test_partition_sizes(self) -> None:
        x = np.arange(40.0).reshape(20, 2)
        x_train, x_test = ml.train_test_split(x, test_size=0.25, seed=1)
        assert len(x_train) == 15
        assert len(x_test) == 5

    def test_no_overlap(self) -> None:
        x = np.arange(30.0).reshape(30, 1)
        a_train, a_test, b_train, b_test = ml.train_test_split(x, x * 2, seed=2)
        assert np.allclose(a_train * 2, b_train)
        union = np.sort(np.concatenate([a_test.ravel(), a_train.ravel()]))
        assert len(set(union)) == 30

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError, match="test_size"):
            ml.train_test_split(np.zeros((5, 1)), test_size=1.5)


class TestLinearRegression:
    def test_recovers_line(self) -> None:
        x, y = ml.make_regression_data(n=200, n_features=2, noise=0.0, seed=5)
        model = ml.LinearRegression().fit(x, y)
        assert model.score(x, y) > 0.9999

    def test_predict_shape_1d_input(self) -> None:
        model = ml.LinearRegression().fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert np.allclose(model.predict([4.0]), [8.0])

    def test_unfitted_predict_raises(self) -> None:
        with pytest.raises(RuntimeError):
            ml.LinearRegression().predict([1.0])


class TestLogisticRegression:
    def test_separable_classes(self) -> None:
        rng_values = np.random.default_rng(9)
        class_a = rng_values.normal(-2.0, 0.7, size=(60, 2))
        class_b = rng_values.normal(+2.0, 0.7, size=(60, 2))
        x = np.vstack([class_a, class_b])
        y = np.array([0] * 60 + [1] * 60)
        model = ml.LogisticRegression(learning_rate=0.3, max_iter=3000).fit(x, y)
        assert model.score(x, y) > 0.95

    def test_proba_range_and_threshold(self) -> None:
        x = [[-1.0], [-0.8], [0.8], [1.0]]
        y = [0, 0, 1, 1]
        model = ml.LogisticRegression(max_iter=1500).fit(x, y)
        probs = model.predict_proba(x)
        assert np.all((probs >= 0) & (probs <= 1))
        assert list(model.predict(x)) == [0, 0, 1, 1]


class TestKMeans:
    def test_separates_three_blobs(self) -> None:
        x, labels_true = ml.make_blobs(n_samples=300, centers=3, cluster_std=0.6, seed=12)
        kmeans = ml.KMeans(n_clusters=3, seed=42).fit(x)
        agreement = max(
            np.mean(kmeans.labels_ == labels_true),
            np.mean(kmeans.labels_ == (labels_true + 1) % 3),
            np.mean(kmeans.labels_ == (labels_true + 2) % 3),
        )
        assert agreement > 0.9
        assert kmeans.inertia_ > 0

    def test_predict_assigns_to_nearest_center(self) -> None:
        x, _y = ml.make_blobs(n_samples=90, centers=2, cluster_std=0.5, seed=13)
        kmeans = ml.KMeans(n_clusters=2, seed=1).fit(x)
        point = kmeans.cluster_centers_[0].reshape(1, -1)
        assert kmeans.predict(point)[0] in (0, 1)

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="n_clusters"):
            ml.KMeans(n_clusters=99).fit(np.ones((5, 2)))


class TestPCA:
    def test_first_component_dominates(self) -> None:
        t_values = np.linspace(0.0, 10.0, 200)
        x = np.column_stack([t_values, 2.0 * t_values]) + 0.01 * np.random.default_rng(4).normal(
            size=(200, 2)
        )
        pca = ml.PCA(n_components=2).fit(x)
        assert pca.explained_variance_ratio_[0] > 0.999

    def test_transform_reduces_dimensions(self) -> None:
        x = np.random.default_rng(6).normal(size=(50, 5))
        transformed = ml.PCA(n_components=2).fit_transform(x)
        assert transformed.shape == (50, 2)


class TestKNN:
    def test_perfectly_separable(self) -> None:
        x = np.array([[0.0], [0.2], [0.4], [9.0], [9.2], [9.4]])
        y = np.array([0, 0, 0, 1, 1, 1])
        knn = ml.KNeighborsClassifier(n_neighbors=3).fit(x, y)
        predictions = knn.predict(np.array([[0.1], [9.1]]))
        assert list(predictions) == [0, 1]


class TestDataGenerators:
    def test_regression_shapes(self) -> None:
        x, y = ml.make_regression_data(n=50, n_features=3, seed=0)
        assert x.shape == (50, 3)
        assert y.shape == (50,)

    def test_blobs_labels(self) -> None:
        _x, y = ml.make_blobs(n_samples=95, centers=4, seed=1)
        assert set(np.unique(y)) == {0, 1, 2, 3}


class TestMetrics:
    def test_accuracy(self) -> None:
        assert ml.accuracy_score([1, 0, 1, 1], [1, 0, 1, 0]) == pytest.approx(0.75)

    def test_precision_recall_f1_binary(self) -> None:
        truth = [1, 1, 0, 0]
        pred = [1, 0, 0, 0]
        assert ml.precision_score(truth, pred) == pytest.approx(1.0)
        assert ml.recall_score(truth, pred) == pytest.approx(0.5)
        expected_f1 = 2 * 1.0 * 0.5 / 1.5
        assert ml.f1_score(truth, pred) == pytest.approx(expected_f1)

    def test_confusion_matrix_layout(self) -> None:
        cm = ml.confusion_matrix([0, 1, 1, 0], [0, 1, 0, 1])
        assert cm.tolist() == [[1, 1], [1, 1]]

    def test_mse_rmse_mae(self) -> None:
        errors_truth = [0.0, 0.0]
        errors_pred = [3.0, -1.0]
        assert ml.mean_squared_error(errors_truth, errors_pred) == pytest.approx(5.0)
        assert ml.root_mean_squared_error(errors_truth, errors_pred) == pytest.approx(np.sqrt(5.0))
        assert ml.mean_absolute_error(errors_truth, errors_pred) == pytest.approx(2.0)

    def test_r2_perfect_and_worse_than_mean(self) -> None:
        truth = np.array([1.0, 2.0, 3.0])
        assert ml.r2_score(truth, truth) == pytest.approx(1.0)
        assert ml.r2_score(truth, np.full(3, 2.0)) == pytest.approx(0.0)
