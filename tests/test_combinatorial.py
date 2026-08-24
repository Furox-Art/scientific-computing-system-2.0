"""Tests for cds2.combinatorial."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from cds2 import combinatorial as comb

DIST_4 = np.array(
    [
        [0.0, 10.0, 15.0, 20.0],
        [10.0, 0.0, 35.0, 25.0],
        [15.0, 35.0, 0.0, 30.0],
        [20.0, 25.0, 30.0, 0.0],
    ]
)

ASSIGN_3X3 = np.array(
    [
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0],
    ]
)


def _is_subsequence(candidate: str, source: str) -> bool:
    iterator = iter(source)
    return all(character in iterator for character in candidate)


class TestDistanceMatrixValidation:
    def test_asymmetric_shape_rejected(self) -> None:
        bad = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]])
        with pytest.raises(ValueError, match="square with zero diagonal"):
            comb.nearest_neighbor_tsp(bad)

    def test_negative_entry_rejected(self) -> None:
        bad = np.array([[0.0, -1.0], [-1.0, 0.0]])
        with pytest.raises(ValueError, match="non-negative entries"):
            comb.nearest_neighbor_tsp(bad)

    def test_nonzero_diagonal_rejected(self) -> None:
        bad = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(ValueError, match="zero diagonal"):
            comb.two_opt([0, 1, 0], bad)

    def test_too_small_matrix_rejected(self) -> None:
        with pytest.raises(ValueError, match="distance matrix must be square"):
            comb._as_distance_matrix([[0.0]])


class TestNearestNeighbor:
    def test_known_matrix_matches_hand_tour(self) -> None:
        result = comb.nearest_neighbor_tsp(DIST_4)
        assert result.tour.tolist() == [0, 1, 3, 2, 0]
        assert result.cost == pytest.approx(80.0)
        assert result.improved_from is None
        assert len(result.tour) == DIST_4.shape[0] + 1

    def test_explicit_start_city(self) -> None:
        result = comb.nearest_neighbor_tsp(DIST_4, start=2)
        assert result.tour.tolist() == [2, 0, 1, 3, 2]
        assert result.cost == pytest.approx(80.0)

    def test_start_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="start city out of range"):
            comb.nearest_neighbor_tsp(DIST_4, start=4)

    def test_negative_start_raises(self) -> None:
        with pytest.raises(ValueError, match="start city out of range"):
            comb.nearest_neighbor_tsp(DIST_4, start=-1)


class TestTwoOpt:
    def test_improves_crossing_tour(self) -> None:
        suboptimal = [0, 1, 2, 3, 0]
        assert comb._tour_cost(np.asarray(suboptimal), DIST_4) == pytest.approx(95.0)
        result = comb.two_opt(suboptimal, DIST_4)
        assert result.cost == pytest.approx(80.0)
        assert result.improved_from == pytest.approx(95.0)
        assert result.tour[0] == result.tour[-1]

    def test_already_optimal_tour_reports_no_improvement(self) -> None:
        optimal = [0, 1, 3, 2, 0]
        result = comb.two_opt(optimal, DIST_4)
        assert result.cost == pytest.approx(80.0)
        assert result.tour.tolist() == optimal
        assert result.improved_from is None

    def test_endpoint_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="start and end at the same city"):
            comb.two_opt([0, 1, 2, 3], DIST_4)

    def test_out_of_range_city_raises(self) -> None:
        with pytest.raises(ValueError, match="outside the distance matrix"):
            comb.two_opt([0, 1, 2, 9, 0], DIST_4)


class TestKnapsack:
    def test_classic_example_with_reconstruction(self) -> None:
        result = comb.knapsack_01([2, 3, 4, 5], [3, 4, 5, 6], 5)
        assert result.total_value == pytest.approx(7.0)
        assert result.chosen_indices.tolist() == [0, 1]
        assert result.total_weight == pytest.approx(5.0)

    def test_float_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="integer-valued for exact DP"):
            comb.knapsack_01([2.5, 3.0], [3.0, 4.0], 5)

    def test_float_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="integer-valued for exact DP"):
            comb.knapsack_01([2, 3], [3, 4], 5.5)

    def test_zero_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity must be positive"):
            comb.knapsack_01([2], [3], 0)

    def test_negative_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity must be positive"):
            comb.knapsack_01([2], [3], -3)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal-length non-empty"):
            comb.knapsack_01([1, 2], [5], 10)

    def test_empty_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="equal-length non-empty"):
            comb.knapsack_01([], [], 10)

    def test_nothing_fits_yields_empty_choice(self) -> None:
        result = comb.knapsack_01([10], [7], 5)
        assert result.total_value == pytest.approx(0.0)
        assert result.chosen_indices.tolist() == []
        assert result.total_weight == pytest.approx(0.0)


class TestAssignment:
    def test_known_square_matrix_matches_scipy(self) -> None:
        result = comb.assign_min_cost(ASSIGN_3X3)
        rows, columns = linear_sum_assignment(ASSIGN_3X3)
        direct_cost = float(ASSIGN_3X3[rows, columns].sum())
        assert result.cost == pytest.approx(direct_cost)
        assert result.cost == pytest.approx(5.0)
        assert result.row_to_col.tolist() == columns.tolist()

    def test_rectangular_two_by_three(self) -> None:
        matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = comb.assign_min_cost(matrix)
        assert result.cost == pytest.approx(6.0)
        assert result.row_to_col.tolist() == [0, 1]
        assert len(result.row_to_col) == 2

    def test_non_2d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            comb.assign_min_cost([1.0, 2.0, 3.0])


class TestLCS:
    def test_known_strings_yield_valid_lcs(self) -> None:
        result = comb.longest_common_subsequence("ABCBDAB", "BDCABA")
        assert len(result) == 4
        assert _is_subsequence(result, "ABCBDAB")
        assert _is_subsequence(result, "BDCABA")

    def test_tie_breaking_pair_still_subsequence(self) -> None:
        result = comb.longest_common_subsequence("ab", "ba")
        assert len(result) == 1
        assert result in {"a", "b"}

    def test_identical_strings(self) -> None:
        assert comb.longest_common_subsequence("abc", "abc") == "abc"

    def test_disjoint_strings(self) -> None:
        assert comb.longest_common_subsequence("abc", "xyz") == ""

    def test_empty_first_argument(self) -> None:
        assert comb.longest_common_subsequence("", "abc") == ""

    def test_empty_second_argument(self) -> None:
        assert comb.longest_common_subsequence("abc", "") == ""

    def test_both_arguments_empty(self) -> None:
        assert comb.longest_common_subsequence("", "") == ""
