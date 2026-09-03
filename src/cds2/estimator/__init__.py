"""cds2.estimator — scikit-learn compatible estimators.

Each class follows the sklearn estimator protocol (``fit``, ``predict``,
``score``, ``get_params``, ``set_params``) so they can be dropped into
sklearn pipelines and cross-validation. scikit-learn is a soft dependency:
the module imports lazily and skips tests when sklearn is absent.

Classes::

    LinearRegressionGD   — ordinary least squares via gradient descent
    RidgeSGD             — L2-regularized linear regression
    KMeansSKL            — K-Means with sklearn-compatible API
    PCASKL               — PCA via SVD
"""

from __future__ import annotations

__all__ = [
    "KMeansSKL",
    "LinearRegressionGD",
    "PCASKL",
    "RidgeSGD",
]

from .cluster import KMeansSKL
from .decomposition import PCASKL
from .linear import LinearRegressionGD, RidgeSGD
