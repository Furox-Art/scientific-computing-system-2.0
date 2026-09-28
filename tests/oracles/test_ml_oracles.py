"""Independent machine-learning oracle tests against scikit-learn.

These checks compare CDS2 outputs with a second implementation rather than
repeating the same equations in the test.  They intentionally exercise mildly
ill-conditioned data and compare predictions/subspaces, which are the
scientifically meaningful quantities when coefficients or component signs are
not unique.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from cds2 import ml


@pytest.mark.parametrize("seed", range(6))
def test_linear_regression_predictions_match_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(300, 5))
    # Add a nearly collinear feature so this checks the solved model rather
    # than only an easy orthogonal design.
    x[:, 4] = 0.999 * x[:, 0] - 0.15 * x[:, 1] + rng.normal(scale=1e-4, size=x.shape[0])
    coefficients = np.array([1.5, -2.0, 0.3, 4.0, 0.7])
    y = x @ coefficients + 2.25 + rng.normal(scale=0.02, size=x.shape[0])

    cds = ml.LinearRegression().fit(x, y)
    reference = SklearnLinearRegression().fit(x, y)

    # Individual coefficients can move substantially in a nearly collinear
    # system while representing the same fitted hyperplane. Predictions are
    # the invariant quantity to compare.
    assert np.allclose(cds.predict(x), reference.predict(x), rtol=2e-9, atol=2e-9)
    assert cds.score(x, y) == pytest.approx(reference.score(x, y), abs=1e-11)


@pytest.mark.parametrize("seed", range(4))
def test_standard_scaler_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.lognormal(mean=2.0, sigma=1.4, size=(250, 6))
    x[:, -1] = 17.0  # constant-column edge case

    cds = ml.StandardScaler().fit(x)
    reference = SklearnStandardScaler().fit(x)

    assert np.allclose(cds.mean_, reference.mean_, rtol=1e-12, atol=1e-12)
    assert np.allclose(cds.scale_, reference.scale_, rtol=1e-12, atol=1e-12)
    assert np.allclose(cds.transform(x), reference.transform(x), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_pca_subspace_and_variance_match_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(400, 3))
    mixing = rng.normal(size=(3, 8))
    x = latent @ mixing + 0.03 * rng.normal(size=(400, 8))

    cds = ml.PCA(n_components=3).fit(x)
    reference = SklearnPCA(n_components=3, svd_solver="full").fit(x)

    assert cds.components_ is not None
    assert cds.explained_variance_ratio_ is not None
    assert np.allclose(
        cds.explained_variance_ratio_, reference.explained_variance_ratio_, rtol=1e-10, atol=1e-12
    )

    # PCA vector signs are arbitrary. Compare the projection operators for the
    # learned three-dimensional subspace instead of raw component signs.
    cds_projection = cds.components_.T @ cds.components_
    reference_projection = reference.components_.T @ reference.components_
    assert np.allclose(cds_projection, reference_projection, rtol=1e-9, atol=1e-9)
