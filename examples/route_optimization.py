"""Case study 21: Route optimization on a delivery map.

Twelve depots are placed on a map; a nearest-neighbor tour is built and
refined with 2-opt, then a courier-assignment problem is solved optimally
and the classic knapsack decides which parcels fit the evening van.
"""

from __future__ import annotations

import numpy as np

import cds2


def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    differences = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((differences**2).sum(axis=-1))


def main() -> None:
    rng = np.random.default_rng(13)
    coords = rng.uniform(0.0, 100.0, size=(12, 2))
    distances = build_distance_matrix(coords)

    greedy = cds2.combinatorial.nearest_neighbor_tsp(distances, start=0)
    improved = cds2.combinatorial.two_opt(greedy.tour, distances)
    gain = 100.0 * (1.0 - improved.cost / greedy.cost)

    print("== Delivery route (12 stops) ==")
    print(f"nearest neighbor : {greedy.cost:.1f} km")
    print(f"after 2-opt      : {improved.cost:.1f} km  ({gain:.1f}% shorter)")
    print(f"tour             : {improved.tour.tolist()}")

    print("\n== Courier assignment (3 couriers x 3 zones) ==")
    cost_matrix = [
        [40.0, 55.0, 62.0],
        [48.0, 41.0, 58.0],
        [60.0, 52.0, 45.0],
    ]
    assignment = cds2.combinatorial.assign_min_cost(cost_matrix)
    print(f"minimum total hours : {assignment.cost:.0f}")
    for row, col in enumerate(assignment.row_to_col.tolist()):
        print(f"  courier {row + 1} -> zone {col + 1}")

    print("\n== Evening van packing (capacity 12 kg) ==")
    weights = [4.0, 4.5, 3.0, 5.0, 1.5, 2.5]
    values = [9.0, 10.0, 7.0, 11.0, 3.0, 6.0]
    knapsack = cds2.combinatorial.knapsack_01(
        [int(w * 10) for w in weights], [int(v * 10) for v in values], capacity=120
    )
    chosen = knapsack.chosen_indices.tolist()
    print(f"parcels taken       : {chosen}")
    print(f"total weight        : {sum(weights[i] for i in chosen):.1f} kg")
    print(f"total value         : {knapsack.total_value / 10:.1f}")

    common = cds2.combinatorial.longest_common_subsequence(
        "delivery-route-north", "reliable-route-south"
    )
    print(f"\nLCS of route names  : {common!r}")


if __name__ == "__main__":
    main()
