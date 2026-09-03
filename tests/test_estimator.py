"""Tests for cds2.estimator sklearn-compatible estimators."""

import numpy as np
import pytest

from cds2.estimator import (
    PCASKL,
    KMeansSKL,
    LinearRegressionGD,
    RidgeSGD,
)
from cds2.estimator._base import BaseEstimator


@pytest.fixture
def regression_data():  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.0 + rng.normal(scale=0.01, size=60)
    return X, y


class TestBaseEstimator:
    def test_get_and_set_params(self) -> None:
        est = LinearRegressionGD(learning_rate=0.05, max_iter=10)
        assert est.get_params() == {"learning_rate": 0.05, "max_iter": 10, "tol": 1e-6}
        est.set_params(max_iter=20)
        assert est.max_iter == 20

    def test_set_unknown_param_rejected(self) -> None:
        with pytest.raises(ValueError):
            LinearRegressionGD().set_params(nope=1)

    def test_check_X_y_without_y(self) -> None:
        X, y = BaseEstimator._check_X_y(np.zeros((4, 2)))
        assert X.shape == (4, 2)
        assert y is None

    def test_check_X_y_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError):
            BaseEstimator._check_X_y(np.zeros((4,)))
        with pytest.raises(ValueError):
            BaseEstimator._check_X_y(np.zeros((4, 2)), y=np.zeros((4, 2)))
        with pytest.raises(ValueError):
            BaseEstimator._check_X_y(np.zeros((4, 2)), y=np.zeros((3,)))

    def test_check_X_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError):
            BaseEstimator._check_X(np.zeros((4,)))


class TestLinearRegressionGD:
    def test_fit_recovers_coefficients(self, regression_data) -> None:  # type: ignore[no-untyped-def]
        X, y = regression_data
        est = LinearRegressionGD(learning_rate=0.1, max_iter=5000).fit(X, y)
        assert est.coef_ is not None
        assert est.coef_ == pytest.approx([3.0, -2.0], abs=0.1)

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            LinearRegressionGD().predict(np.zeros((2, 2)))

    def test_score_is_high_on_linear_data(self, regression_data) -> None:  # type: ignore[no-untyped-def]
        X, y = regression_data
        est = LinearRegressionGD(learning_rate=0.1, max_iter=5000).fit(X, y)
        assert est.score(X, y) > 0.98

    def test_fit_loop_exhausts_without_converging(self, regression_data) -> None:  # type: ignore[no-untyped-def]
        X, y = regression_data
        # tol=0.0 can never trigger the early break, so the loop runs max_iter.
        est = LinearRegressionGD(max_iter=2, tol=0.0).fit(X, y)
        assert est.coef_ is not None

    def test_score_zero_variance_target(self) -> None:
        X = np.ones((10, 2))
        y = np.ones(10)
        est = LinearRegressionGD(max_iter=5).fit(X, y)
        assert est.score(X, y) == 0.0


class TestRidgeSGD:
    def test_fit_and_predict(self, regression_data) -> None:  # type: ignore[no-untyped-def]
        X, y = regression_data
        est = RidgeSGD(alpha=0.01, learning_rate=0.1, max_iter=5000).fit(X, y)
        pred = est.predict(X)
        assert pred.shape == (60,)
        assert est.score(X, y) > 0.98

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            RidgeSGD().predict(np.zeros((2, 2)))

    def test_score_zero_variance_target(self) -> None:
        X = np.ones((10, 2))
        y = np.ones(10)
        est = RidgeSGD(max_iter=5).fit(X, y)
        assert est.score(X, y) == 0.0

    def test_fit_loop_exhausts_without_converging(self, regression_data) -> None:  # type: ignore[no-untyped-def]
        X, y = regression_data
        est = RidgeSGD(max_iter=2, tol=0.0).fit(X, y)
        assert est.coef_ is not None


class TestPCASKL:
    def test_fit_transform_shapes(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 5))
        est = PCASKL(n_components=2).fit(X)
        assert est.components_ is not None
        assert est.components_.shape == (2, 5)
        Z = est.transform(X)
        assert Z.shape == (40, 2)

    def test_fit_transform_shortcut(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 4))
        Z = PCASKL(n_components=3).fit_transform(X)
        assert Z.shape == (30, 3)

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            PCASKL().transform(np.zeros((4, 4)))


class TestKMeansSKL:
    def test_fit_predict_two_blobs(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(loc=-5.0, scale=0.5, size=(20, 2))
        b = rng.normal(loc=5.0, scale=0.5, size=(20, 2))
        X = np.vstack([a, b])
        est = KMeansSKL(n_clusters=2, seed=0).fit(X)
        assert est.cluster_centers_ is not None
        assert est.cluster_centers_.shape == (2, 2)
        assert est.labels_ is not None
        assert set(np.unique(est.labels_)) == {0, 1}
        assert est.inertia_ > 0.0
        pred = est.predict(X)
        assert pred.shape == (40,)
        both = est.fit_predict(X)
        assert both.shape == (40,)

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            KMeansSKL().predict(np.zeros((4, 2)))

    def test_empty_cluster_gets_reseeded(self) -> None:
        # All points identical -> every cluster but one is empty -> reseed path.
        X = np.zeros((6, 2))
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            est = KMeansSKL(n_clusters=2, max_iter=2, tol=0.0, seed=0).fit(X)
        assert est.cluster_centers_ is not None
        assert est.labels_ is not None

    def test_fit_loop_exhausts_without_converging(self) -> None:
        rng = np.random.default_rng(4)
        X = rng.normal(size=(20, 2))
        est = KMeansSKL(n_clusters=2, max_iter=2, tol=0.0, seed=0).fit(X)
        assert est.cluster_centers_ is not None
