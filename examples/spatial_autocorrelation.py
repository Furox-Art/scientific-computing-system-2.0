"""Case study 22: Spatial autocorrelation of sensor readings.

Sixteen monitoring stations are laid over a grid; a smooth pollution field
plus local hotspots is measured three ways - Moran's I for global
clustering, Geary's C as its sensitive complement and the Clark-Evans
nearest-neighbor index for point-pattern structure.
"""

from __future__ import annotations

import numpy as np

import cds2


def main() -> None:
    rng = np.random.default_rng(17)
    grid = np.linspace(0.0, 10.0, 4)
    x_coords, y_coords = np.meshgrid(grid, grid)
    points = np.column_stack([x_coords.ravel(), y_coords.ravel()])

    base_field = 2.0 * (x_coords + y_coords) / 20.0
    hotspot = np.exp(-((x_coords - 6.7) ** 2 + (y_coords - 3.3) ** 2) / 4.0)
    readings = (
        40.0 + 50.0 * (base_field + hotspot).ravel() + rng.normal(scale=1.5, size=points.shape[0])
    )

    print("== Sensor field ==")
    print(
        f"stations: {points.shape[0]}, readings in "
        f"[{readings.min():.1f}, {readings.max():.1f}] ug/m^3"
    )

    weights = cds2.spatial.build_weight_matrix(points, cutoff=4.5)
    links_per_node = (weights > 0).sum(axis=1)
    print(f"weight rows sum to 1: {np.allclose(weights.sum(axis=1), 1.0)}")
    print(f"neighbours per node : min {links_per_node.min()}, max {links_per_node.max()}")

    moran = cds2.spatial.morans_i(readings, weights)
    geary = cds2.spatial.gearys_c(readings, weights)
    print("\n== Global autocorrelation ==")
    print(f"Moran's I  : {moran.index:.3f} (expected {moran.expected:.3f}, z={moran.z_score:.2f})")
    print(f"Geary's C  : {geary.c:.3f} (expected {geary.expected:.1f}, z={geary.z_score:.2f})")

    shuffled = rng.permutation(readings)
    moran_null = cds2.spatial.morans_i(shuffled, weights)
    print("\n== Randomisation control ==")
    print(f"Moran's I on shuffled readings: {moran_null.index:.3f}")

    clustered_points = np.vstack(
        [
            rng.normal(loc=[3.0, 3.0], scale=0.4, size=(12, 2)),
            rng.normal(loc=[8.0, 8.0], scale=0.4, size=(12, 2)),
        ]
    )
    nni_clustered = cds2.spatial.nearest_neighbor_index(clustered_points)
    nni_grid = cds2.spatial.nearest_neighbor_index(points)
    print("\n== Point patterns (Clark-Evans) ==")
    print(f"two-hotspot layout : ratio {nni_clustered.ratio:.2f} -> {nni_clustered.pattern}")
    print(f"regular grid       : ratio {nni_grid.ratio:.2f} -> {nni_grid.pattern}")


if __name__ == "__main__":
    main()
