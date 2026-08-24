"""Tests for cds2.spatial."""

import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from cds2 import spatial

TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
CLUSTER_VALUES = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
CHECKER_VALUES = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def _block_weight() -> np.ndarray:
    weights = np.zeros((6, 6))
    for block in (range(3), range(3, 6)):
        index = np.asarray(list(block))
        weights[np.ix_(index, index)] = 1.0
    np.fill_diagonal(weights, 0.0)
    return weights / weights.sum(axis=1, keepdims=True)


def _path_weight(size: int = 6) -> np.ndarray:
    weights = np.zeros((size, size))
    for i in range(size - 1):
        weights[i, i + 1] = 1.0
        weights[i + 1, i] = 1.0
    return weights / weights.sum(axis=1, keepdims=True)


class TestBuildWeightMatrix:
    def test_rows_sum_to_one_for_connected_nodes(self) -> None:
        square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        weights = spatial.build_weight_matrix(square, cutoff=1.2)
        assert np.allclose(weights.sum(axis=1), 1.0)
        expected = 0.5 * (np.ones((4, 4)) - np.eye(4)) - 0.5 * np.array(
            [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
        )
        assert np.allclose(weights, expected)

    def test_default_cutoff_is_mean_nearest_neighbour_distance(self) -> None:
        weights = spatial.build_weight_matrix(TRIANGLE)
        assert np.allclose(weights, [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def test_large_cutoff_connects_everything(self) -> None:
        weights = spatial.build_weight_matrix(TRIANGLE, cutoff=10.0)
        assert np.allclose(weights, 0.5 * (np.ones((3, 3)) - np.eye(3)))

    def test_diagonal_stays_zero_and_support_is_symmetric(self) -> None:
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (5.0, 0.0)]
        weights = spatial.build_weight_matrix(points, cutoff=1.5)
        assert not np.any(np.diag(weights))
        support = weights != 0.0
        assert np.array_equal(support, support.T)

    def test_isolated_node_gets_zero_row_and_column(self) -> None:
        points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (100.0, 100.0)]
        weights = spatial.build_weight_matrix(points, cutoff=1.5)
        assert np.all(weights[3] == 0.0)
        assert np.all(weights[:, 3] == 0.0)
        assert weights[:3, :3].sum() == pytest.approx(3.0)

    def test_zero_cutoff_raises(self) -> None:
        with pytest.raises(ValueError, match="cutoff must be positive"):
            spatial.build_weight_matrix(TRIANGLE, cutoff=0.0)

    def test_negative_cutoff_raises(self) -> None:
        with pytest.raises(ValueError, match="cutoff must be positive"):
            spatial.build_weight_matrix(TRIANGLE, cutoff=-1.0)

    def test_too_few_points_raise(self) -> None:
        with pytest.raises(ValueError, match="three rows"):
            spatial.build_weight_matrix([(0.0, 0.0), (1.0, 1.0)])

    def test_wrong_shape_points_raise(self) -> None:
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            spatial.build_weight_matrix([(0.0,)])


class TestMoransI:
    def test_perfectly_clustered_values_give_index_one(self) -> None:
        result = spatial.morans_i(CLUSTER_VALUES, _block_weight())
        assert result.index == pytest.approx(1.0)

    def test_checkerboard_values_give_negative_index(self) -> None:
        result = spatial.morans_i(CHECKER_VALUES, _path_weight())
        assert result.index == pytest.approx(-1.0)

    def test_grid_regions_versus_column_alternation(self) -> None:
        grid = [(float(x), float(y)) for y in range(3) for x in range(2)]
        weights = spatial.build_weight_matrix(grid, cutoff=1.5)
        columns = spatial.morans_i([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], weights)
        rows = spatial.morans_i([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], weights)
        assert columns.index < 0.0
        assert rows.index > 0.2
        assert rows.index > columns.index

    def test_hand_computed_three_point_example(self) -> None:
        weights = spatial.build_weight_matrix(TRIANGLE)
        result = spatial.morans_i([0.0, 1.0, 2.0], weights)
        assert result.index == pytest.approx(-0.75)
        assert result.expected == pytest.approx(-0.5)
        assert result.z_score == pytest.approx(-0.25 / math.sqrt(1.0 / 3.0))

    def test_expected_equals_negative_one_over_n_minus_one(self) -> None:
        result = spatial.morans_i([1.0, 2.0, 3.0, 4.0], _path_weight(4))
        assert result.expected == pytest.approx(-1.0 / 3.0)

    def test_z_score_matches_documented_approximation(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        result = spatial.morans_i(values, _path_weight(4))
        assert result.z_score == pytest.approx(
            (result.index - result.expected) / math.sqrt(1.0 / len(values))
        )

    def test_result_is_frozen(self) -> None:
        result = spatial.MoranResult(index=0.5, expected=-0.5, z_score=1.0)
        with pytest.raises(FrozenInstanceError):
            result.index = 9.9


class TestGearysC:
    def test_clustered_values_give_low_c(self) -> None:
        result = spatial.gearys_c(CLUSTER_VALUES, _block_weight())
        assert result.c == pytest.approx(0.0)

    def test_checkerboard_values_give_high_c(self) -> None:
        result = spatial.gearys_c(CHECKER_VALUES, _path_weight())
        assert result.c == pytest.approx(5.0 / 3.0)

    def test_inverse_relationship_with_moran(self) -> None:
        clustered_geary = spatial.gearys_c(CLUSTER_VALUES, _block_weight())
        clustered_moran = spatial.morans_i(CLUSTER_VALUES, _block_weight())
        assert clustered_geary.c < 0.1
        assert clustered_moran.index > 0.9
        checker_geary = spatial.gearys_c(CHECKER_VALUES, _path_weight())
        checker_moran = spatial.morans_i(CHECKER_VALUES, _path_weight())
        assert checker_geary.c > 1.5
        assert checker_moran.index < 0.0

    def test_hand_computed_three_point_example(self) -> None:
        weights = spatial.build_weight_matrix(TRIANGLE)
        result = spatial.gearys_c([0.0, 1.0, 2.0], weights)
        assert result.c == pytest.approx(1.25)

    def test_expected_defaults_to_one(self) -> None:
        result = spatial.GearyResult(c=0.5, z_score=2.0)
        assert result.expected == 1.0

    def test_z_score_matches_documented_approximation(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        result = spatial.gearys_c(values, _path_weight(4))
        assert result.z_score == pytest.approx((result.c - 1.0) / math.sqrt(1.0 / 8))

    def test_result_is_frozen(self) -> None:
        result = spatial.GearyResult(c=0.5, z_score=2.0)
        with pytest.raises(FrozenInstanceError):
            result.c = 9.9


@pytest.mark.parametrize("statistic", [spatial.morans_i, spatial.gearys_c])
class TestStatisticValidation:
    def test_size_mismatch_raises(
        self,
        statistic: Callable[[list[float], np.ndarray], object],
    ) -> None:
        with pytest.raises(ValueError, match="sizes differ"):
            statistic([0.0, 1.0, 2.0], np.zeros((4, 4)))

    def test_non_square_matrix_raises(
        self,
        statistic: Callable[[list[float], np.ndarray], object],
    ) -> None:
        with pytest.raises(ValueError, match="sizes differ"):
            statistic([0.0, 1.0, 2.0], np.zeros((3, 4)))

    def test_too_few_rows_raises(
        self,
        statistic: Callable[[list[float], np.ndarray], object],
    ) -> None:
        with pytest.raises(ValueError, match="three rows"):
            statistic([0.0, 1.0], np.eye(2))

    def test_link_free_matrix_raises(
        self,
        statistic: Callable[[list[float], np.ndarray], object],
    ) -> None:
        with pytest.raises(ValueError, match="no links"):
            statistic([0.0, 1.0, 2.0, 3.0], np.zeros((4, 4)))

    def test_constant_values_raise(
        self,
        statistic: Callable[[list[float], np.ndarray], object],
    ) -> None:
        with pytest.raises(ValueError, match="constant"):
            statistic([1.0, 1.0, 1.0, 1.0], _path_weight(4))


class TestNearestNeighborIndex:
    def test_clustered_points_have_ratio_below_one(self) -> None:
        points = [
            (0.0, 0.0),
            (0.01, 0.0),
            (0.0, 0.01),
            (10.0, 10.0),
            (10.01, 10.0),
            (10.0, 10.01),
        ]
        result = spatial.nearest_neighbor_index(points)
        assert result.pattern == "clustered"
        assert result.ratio < 0.95

    def test_regular_grid_has_ratio_above_one(self) -> None:
        grid = [(float(x), float(y)) for y in range(8) for x in range(8)]
        result = spatial.nearest_neighbor_index(grid)
        assert result.pattern == "dispersed"
        assert result.ratio > 1.05
        assert result.ratio == pytest.approx(16.0 / 7.0)

    def test_uniform_sample_reads_as_random(self) -> None:
        rng = np.random.default_rng(42)
        points = rng.uniform(0.0, 1.0, size=(200, 2))
        result = spatial.nearest_neighbor_index(points)
        assert result.pattern == "random"
        assert 0.95 < result.ratio < 1.05

    def test_bounding_box_area_and_observed_mean(self) -> None:
        result = spatial.nearest_neighbor_index(TRIANGLE)
        assert result.observed_mean == pytest.approx(1.0)
        assert result.expected_mean == pytest.approx(0.5 / math.sqrt(3.0 / 1.0))

    def test_area_override_scales_expected_mean(self) -> None:
        base = spatial.nearest_neighbor_index(TRIANGLE)
        overridden = spatial.nearest_neighbor_index(TRIANGLE, area=12.0)
        assert overridden.observed_mean == pytest.approx(base.observed_mean)
        assert overridden.expected_mean == pytest.approx(0.5 / math.sqrt(3.0 / 12.0))
        assert overridden.ratio == pytest.approx(base.ratio / math.sqrt(12.0))

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError, match="area must be positive"):
            spatial.nearest_neighbor_index(TRIANGLE, area=0.0)

    def test_negative_area_raises(self) -> None:
        with pytest.raises(ValueError, match="area must be positive"):
            spatial.nearest_neighbor_index(TRIANGLE, area=-5.0)

    def test_degenerate_bounding_box_raises(self) -> None:
        collinear = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        with pytest.raises(ValueError, match="area must be positive"):
            spatial.nearest_neighbor_index(collinear)

    def test_too_few_points_raise(self) -> None:
        with pytest.raises(ValueError, match="three rows"):
            spatial.nearest_neighbor_index([(0.0, 0.0), (1.0, 1.0)])

    def test_result_is_frozen(self) -> None:
        result = spatial.NearestNeighborIndex(
            observed_mean=1.0,
            expected_mean=1.0,
            ratio=1.0,
            pattern="random",
        )
        with pytest.raises(FrozenInstanceError):
            result.pattern = "clustered"
