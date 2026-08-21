"""Tests for cds2.interpolate."""

import numpy as np
import pytest

from cds2 import interpolate


class TestOneDimensional:
    def test_linear_passes_through_knots(self) -> None:
        x_known = [0.0, 1.0, 2.0]
        y_known = [0.0, 2.0, 4.0]
        result = interpolate.linear_interp([0.5, 1.0, 1.5], x_known, y_known)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_cubic_spline_exact_at_knots(self) -> None:
        spline = interpolate.cubic_spline([0.0, 1.0, 2.0], [1.0, 3.0, 2.0])
        assert np.allclose(spline([0.0, 1.0, 2.0]), [1.0, 3.0, 2.0])

    def test_cubic_spline_smooth_derivative(self) -> None:
        spline = interpolate.cubic_spline([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 1.0])
        left = spline(1.0 - 1e-6)
        right = spline(1.0 + 1e-6)
        assert abs(right - left) < 1e-4

    def test_pchip_monotone(self) -> None:
        pchip = interpolate.pchip_interpolator([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])
        grid = np.linspace(0.0, 3.0, 50)
        values = pchip(grid)
        assert np.all(np.diff(values) >= 0)

    def test_lagrange_exact(self) -> None:
        poly = interpolate.lagrange_poly([0.0, 1.0, 2.0], [1.0, 3.0, 7.0])
        assert np.allclose(poly([0.0, 1.0, 2.0]), [1.0, 3.0, 7.0])


class TestMultidimensional:
    def test_griddata_linear_plane(self) -> None:
        points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        values = [0.0, 1.0, 1.0, 2.0]
        result = interpolate.grid_interp(points, values, [(0.5, 0.5)])
        assert result[0] == pytest.approx(1.0)

    def test_griddata_nearest(self) -> None:
        points = [(0.0,), (10.0,)]
        result = interpolate.grid_interp(points, [1.0, 9.0], [(1.2,)], method="nearest")
        assert result[0] == pytest.approx(1.0)

    def test_regular_grid_bilinear(self) -> None:
        axes_x = [0.0, 1.0]
        axes_y = [0.0, 1.0]
        values = [[0.0, 1.0], [1.0, 2.0]]
        result = interpolate.regular_grid_interp([axes_x, axes_y], values, [(0.25, 0.75)])
        assert result[0] == pytest.approx(1.0)
