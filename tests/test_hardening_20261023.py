"""Regressions for degenerate statistics, spline boundary conditions, and 1-D grids."""
from __future__ import annotations

import numpy as np
import pytest

from cds2 import interpolate, stats


def test_t_test_rejects_constant_sample() -> None:
    with pytest.raises(ValueError, match="constant sample"):
        stats.t_test([2.0, 2.0, 2.0])


def test_independent_t_test_rejects_two_constant_samples() -> None:
    with pytest.raises(ValueError, match="both samples are constant"):
        stats.independent_t_test([1.0, 1.0], [2.0, 2.0])


def test_anova_rejects_all_constant_groups() -> None:
    with pytest.raises(ValueError, match="every group is constant"):
        stats.anova([1.0, 1.0], [2.0, 2.0])


def test_cubic_spline_uses_natural_boundary_conditions() -> None:
    spline = interpolate.cubic_spline([0.0, 1.0, 2.0], [1.0, 3.0, 2.0])
    assert spline(0.0, 2) == pytest.approx(0.0, abs=1e-12)
    assert spline(2.0, 2) == pytest.approx(0.0, abs=1e-12)


def test_regular_grid_interp_supports_one_dimensional_axes() -> None:
    assert interpolate.regular_grid_interp(
        [np.array([0.0, 0.5, 1.0])],
        np.array([0.0, 5.0, 10.0]),
        np.array([0.25, 0.75]),
    ) == pytest.approx([2.5, 7.5])


def test_percentile_accepts_tuples_and_arrays() -> None:
    assert stats.percentile([1, 2, 3, 4, 5], (25, 75)) == pytest.approx([2.0, 4.0])
    assert stats.percentile([1, 2, 3, 4, 5], np.array([50.0])) == pytest.approx([3.0])


def test_covariance_matrix_keeps_single_column_two_dimensional() -> None:
    result = stats.covariance_matrix([[1.0], [2.0], [3.0]])
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(1.0)


def test_multivariate_normal_logpdf_validates_arguments() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        stats.multivariate_normal_logpdf([[0.0]], [0.0], [[1.0]])
    with pytest.raises(ValueError, match="same length"):
        stats.multivariate_normal_logpdf([0.0], [0.0, 1.0], [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="square matrix"):
        stats.multivariate_normal_logpdf([0.0, 1.0], [0.0, 1.0], [[1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        stats.multivariate_normal_logpdf([0.0, 0.0], [0.0, 0.0], [[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        stats.multivariate_normal_logpdf([0.0], [0.0], [[0.0]])
