"""Tests for cds2.geometry."""

import numpy as np
import pytest

from cds2 import geometry

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]


class TestConvexHull:
    def test_interior_point_excluded(self) -> None:
        points = [*TRIANGLE, (0.3, 0.3)]
        result = geometry.convex_hull(points)
        assert len(result.vertices) == 3
        assert result.indices is not None and result.indices.size == 3

    def test_few_points(self) -> None:
        result = geometry.convex_hull([(0.0, 0.0), (1.0, 1.0)])
        assert result.vertices.shape == (2, 2)

    def test_bad_input_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            geometry.convex_hull([(0.0,)])


class TestHullAreaAndPolygon:
    def test_triangle_area(self) -> None:
        assert geometry.hull_area(TRIANGLE) == pytest.approx(0.5)

    def test_square_polygon_area(self) -> None:
        assert geometry.polygon_area(SQUARE) == pytest.approx(1.0)

    def test_square_perimeter(self) -> None:
        assert geometry.polygon_perimeter(SQUARE) == pytest.approx(4.0)

    def test_degenerate_hull_area_zero(self) -> None:
        assert geometry.hull_area([(0.0, 0.0), (1.0, 1.0)]) == 0.0


class TestDistances:
    def test_pairwise_matrix_symmetric(self) -> None:
        distances = geometry.point_cloud_distances([(0.0, 0.0), (3.0, 4.0)])
        assert distances[0, 1] == pytest.approx(5.0)
        assert distances[0, 0] == pytest.approx(0.0)

    def test_cross_distances(self) -> None:
        a = [(0.0, 0.0)]
        b = [(1.0, 0.0), (0.0, 2.0)]
        distances = geometry.point_cloud_distances(a, b)
        assert distances.shape == (1, 2)

    def test_closest_pair(self) -> None:
        i, j, distance = geometry.closest_pair([(0.0, 0.0), (5.0, 5.0), (1.0, 1.0), (9.0, 9.0)])
        assert {i, j} == {0, 2}
        assert distance == pytest.approx(np.sqrt(2))

    def test_closest_pair_needs_two(self) -> None:
        with pytest.raises(ValueError, match="two"):
            geometry.closest_pair([(0.0, 0.0)])


class TestContainment:
    def test_inside_outside_boundary(self) -> None:
        assert geometry.point_in_polygon((0.5, 0.5), SQUARE)
        assert not geometry.point_in_polygon((1.5, 0.5), SQUARE)

    def test_concave_polygon(self) -> None:
        notch = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (0.5, 0.5), (-2.0, 2.0)]
        assert geometry.point_in_polygon((-1.0, 0.0), notch)
        assert geometry.point_in_polygon((1.8, 1.2), notch)
        assert not geometry.point_in_polygon((1.2, 1.8), notch)


class TestIntersections:
    def test_segments_cross(self) -> None:
        assert geometry.segments_intersect([(0.0, 0.0), (2.0, 2.0)], [(0.0, 2.0), (2.0, 0.0)])

    def test_segments_disjoint(self) -> None:
        assert not geometry.segments_intersect([(0.0, 0.0), (1.0, 0.0)], [(0.0, 1.0), (1.0, 1.0)])

    def test_touching_endpoint(self) -> None:
        assert geometry.segments_intersect([(0.0, 0.0), (1.0, 1.0)], [(1.0, 1.0), (2.0, 0.0)])

    def test_line_intersection_point(self) -> None:
        x_value, y_value = geometry.line_intersection(
            [(0.0, 0.0), (2.0, 2.0)], [(0.0, 2.0), (2.0, 0.0)]
        )
        assert x_value == pytest.approx(1.0)
        assert y_value == pytest.approx(1.0)

    def test_parallel_lines_raise(self) -> None:
        with pytest.raises(ValueError, match="parallel"):
            geometry.line_intersection([(0.0, 0.0), (1.0, 0.0)], [(0.0, 1.0), (1.0, 1.0)])


class TestTransforms:
    def test_centroid(self) -> None:
        cx, cy = geometry.centroid([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)])
        assert (cx, cy) == pytest.approx((2 / 3, 2 / 3))

    def test_rotation_quarter_turn(self) -> None:
        rotated = geometry.rotate_points([(1.0, 0.0)], np.pi / 2)
        assert rotated[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert rotated[0, 1] == pytest.approx(1.0)

    def test_rotation_around_pivot(self) -> None:
        rotated = geometry.rotate_points([(2.0, 0.0)], np.pi, pivot=(1.0, 0.0))
        assert rotated[0, 0] == pytest.approx(0.0, abs=1e-12)


class TestCoverageEdges:
    def test_closest_pair_index_ordering_swap(self) -> None:
        i, j, distance = geometry.closest_pair([(0.0, 0.0), (0.1, 0.0), (5.0, 5.0)])
        assert (i, j) == (0, 1)
        assert distance == pytest.approx(0.1)

    def test_ring_rejects_two_vertices(self) -> None:
        with pytest.raises(ValueError, match="three"):
            geometry.polygon_area([(0.0, 0.0), (1.0, 1.0)])
        with pytest.raises(ValueError, match="three"):
            geometry.point_in_polygon((0.0, 0.0), [(0.0, 0.0), (1.0, 1.0)])

    def test_collinear_touching_segments(self) -> None:
        base = [(0.0, 0.0), (2.0, 0.0)]
        assert geometry.segments_intersect(base, [(2.0, 0.0), (3.0, 0.0)])
        assert geometry.segments_intersect(base, [(-1.0, 0.0), (1.0, 0.0)])
        assert geometry.segments_intersect(base, [(1.5, 0.0), (3.0, 0.0)])
        assert geometry.segments_intersect([(3.0, 0.0), (4.0, 0.0)], [(0.0, 0.0), (3.5, 0.0)])


class TestIntersectionCoverageEdges:
    def test_closest_pair_tie_breaks_to_lower_index(self) -> None:
        points = [(0.0, 0.0), (3.0, 3.0), (3.04, 3.0), (3.06, 3.0)]
        i, j, distance = geometry.closest_pair(points)
        assert {i, j} == {2, 3}
        assert distance == pytest.approx(0.02)

    def test_collinear_containment_via_third_orientation(self) -> None:
        long_segment = [(0.0, 0.0), (4.0, 0.0)]
        assert geometry.segments_intersect([(2.0, 0.0), (1.0, 5.0)], long_segment)
        assert geometry.segments_intersect([(1.0, 5.0), (2.0, 0.0)], long_segment)
