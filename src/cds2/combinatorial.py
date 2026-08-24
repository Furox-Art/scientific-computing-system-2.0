"""Classic discrete optimization: TSP heuristics, 0/1 knapsack, assignment and LCS."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

__all__ = [
    "AssignmentResult",
    "KnapsackResult",
    "TourResult",
    "assign_min_cost",
    "knapsack_01",
    "longest_common_subsequence",
    "nearest_neighbor_tsp",
    "two_opt",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_EPSILON = 1e-12


@dataclass(frozen=True)
class TourResult:
    """Closed tour (first city repeated at the end) with its total cost."""

    tour: IntArray
    cost: float
    improved_from: float | None = None


@dataclass(frozen=True)
class KnapsackResult:
    """Best 0/1 knapsack packing with the ascending indices of chosen items."""

    total_value: float
    chosen_indices: IntArray
    total_weight: float


@dataclass(frozen=True)
class AssignmentResult:
    """Optimal assignment cost together with the row-to-column mapping."""

    cost: float
    row_to_col: IntArray


def _as_distance_matrix(matrix: FloatArray | Sequence[Sequence[float]]) -> FloatArray:
    """Validate and coerce ``matrix`` into a square non-negative distance matrix."""
    array: FloatArray = np.asarray(matrix, dtype=float)
    if (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
        or array.shape[0] < 2
        or bool((array < 0).any())
        or bool(array.diagonal().any())
    ):
        msg = "distance matrix must be square with zero diagonal and non-negative entries"
        raise ValueError(msg)
    return array


def _tour_cost(nodes: IntArray, distance_matrix: FloatArray) -> float:
    """Total length of the closed tour stored in ``nodes``."""
    legs = (float(distance_matrix[int(nodes[i]), int(nodes[i + 1])]) for i in range(nodes.size - 1))
    return float(sum(legs))


def nearest_neighbor_tsp(
    distance_matrix: FloatArray | Sequence[Sequence[float]],
    start: int = 0,
) -> TourResult:
    """Greedy nearest-neighbour heuristic for the travelling-salesman problem."""
    matrix = _as_distance_matrix(distance_matrix)
    city_count = matrix.shape[0]
    if not 0 <= start < city_count:
        msg = "start city out of range"
        raise ValueError(msg)

    visited = np.zeros(city_count, dtype=bool)
    visited[start] = True
    order = [start]
    current = start
    total = 0.0
    for _ in range(city_count - 1):
        best_city = -1
        best_distance = math.inf
        for candidate in range(city_count):
            if visited[candidate]:
                continue
            candidate_distance = float(matrix[current, candidate])
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_city = candidate
        visited[best_city] = True
        order.append(best_city)
        total += best_distance
        current = best_city
    total += float(matrix[order[-1], start])
    order.append(start)

    tour: IntArray = np.asarray(order, dtype=np.int64)
    return TourResult(tour=tour, cost=float(total))


def two_opt(
    tour: IntArray | Sequence[int],
    distance_matrix: FloatArray | Sequence[Sequence[float]],
) -> TourResult:
    """Improve a closed tour with 2-opt edge exchanges until no swap helps."""
    matrix = _as_distance_matrix(distance_matrix)
    city_count = matrix.shape[0]
    nodes: IntArray = np.asarray(tour, dtype=np.int64)
    if nodes[0] != nodes[-1]:
        msg = "tour must start and end at the same city"
        raise ValueError(msg)
    if int(nodes.min()) < 0 or int(nodes.max()) >= city_count:
        msg = "tour contains city indices outside the distance matrix"
        raise ValueError(msg)

    initial_cost = _tour_cost(nodes, matrix)
    best_cost = initial_cost
    improved = True
    while improved:
        improved = False
        for i in range(1, city_count - 1):
            for j in range(i + 1, city_count):
                removed = float(matrix[int(nodes[i - 1]), int(nodes[i])]) + float(
                    matrix[int(nodes[j]), int(nodes[j + 1])]
                )
                added = float(matrix[int(nodes[i - 1]), int(nodes[j])]) + float(
                    matrix[int(nodes[i]), int(nodes[j + 1])]
                )
                if added - removed < -_EPSILON:
                    nodes[i : j + 1] = nodes[i : j + 1][::-1]
                    best_cost += added - removed
                    improved = True
                    break
            if improved:
                break

    return TourResult(
        tour=nodes,
        cost=best_cost,
        improved_from=initial_cost if best_cost < initial_cost else None,
    )


def knapsack_01(
    weights: Sequence[float] | FloatArray,
    values: Sequence[float] | FloatArray,
    capacity: float,
) -> KnapsackResult:
    """Exact 0/1 knapsack via dynamic programming over integer capacity units."""
    weight_array = np.asarray(weights, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if (
        weight_array.ndim != 1
        or value_array.ndim != 1
        or weight_array.size != value_array.size
        or weight_array.size == 0
    ):
        msg = "weights and values must be equal-length non-empty"
        raise ValueError(msg)
    capacity_value = float(capacity)
    weight_values = [float(item) for item in weight_array.tolist()]
    value_values = [float(item) for item in value_array.tolist()]
    if not all(item.is_integer() for item in (*weight_values, capacity_value)):
        msg = "weights and capacity must be integer-valued for exact DP"
        raise ValueError(msg)

    capacity_units = int(round(capacity_value))
    if capacity_units <= 0:
        msg = "capacity must be positive"
        raise ValueError(msg)

    item_count = weight_array.size
    integer_weights = [int(round(item)) for item in weight_values]
    table = np.zeros((item_count + 1, capacity_units + 1), dtype=float)
    for item in range(1, item_count + 1):
        item_weight = integer_weights[item - 1]
        item_value = value_values[item - 1]
        for remaining in range(capacity_units + 1):
            best = float(table[item - 1, remaining])
            if item_weight <= remaining:
                candidate = float(table[item - 1, remaining - item_weight]) + item_value
                if candidate > best:
                    best = candidate
            table[item, remaining] = best

    chosen: list[int] = []
    remaining = capacity_units
    for item in range(item_count, 0, -1):
        if float(table[item, remaining]) != float(table[item - 1, remaining]):
            chosen.append(item - 1)
            remaining -= integer_weights[item - 1]
    chosen.reverse()

    return KnapsackResult(
        total_value=float(table[item_count, capacity_units]),
        chosen_indices=np.asarray(chosen, dtype=np.int64),
        total_weight=float(sum(weight_values[index] for index in chosen)),
    )


def assign_min_cost(
    cost_matrix: FloatArray | Sequence[Sequence[float]],
) -> AssignmentResult:
    """Minimum-cost bipartite assignment over a (possibly rectangular) matrix."""
    matrix: FloatArray = np.asarray(cost_matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        msg = "cost matrix must be 2-D with at least one row and one column"
        raise ValueError(msg)
    rows, columns = linear_sum_assignment(matrix)
    mapping: IntArray = np.asarray(columns, dtype=np.int64)
    total = float(np.asarray(matrix[rows, columns], dtype=float).sum())
    return AssignmentResult(cost=total, row_to_col=mapping)


def longest_common_subsequence(a: str, b: str) -> str:
    """One longest common subsequence of ``a`` and ``b`` via dynamic programming."""
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    characters: list[str] = []
    i, j = len(a), len(b)
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            characters.append(a[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(characters))
