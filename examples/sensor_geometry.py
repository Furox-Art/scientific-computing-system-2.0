"""Case study 11: Computational geometry of a sensor field.

A set of ground sensors is analysed geometrically: the convex hull bounds
the monitored region, closest-pair search flags near-duplicate placements,
and a point-in-polygon test decides whether a target lies inside coverage.
"""

from __future__ import annotations

import numpy as np

import cds2

SENSORS = [
    (0.0, 0.0),
    (10.0, 1.0),
    (11.5, 6.5),
    (7.0, 12.0),
    (2.0, 11.0),
    (-1.5, 5.5),
    (4.8, 6.1),  # interior relay
    (9.9, 1.3),  # nearly duplicates sensor 1
]


def main() -> None:
    hull = cds2.geometry.convex_hull(SENSORS)
    area = cds2.geometry.polygon_area(hull.vertices.tolist())
    perimeter = cds2.geometry.polygon_perimeter(hull.vertices.tolist())
    print("== Sensor-field geometry ==")
    print(f"hull vertices          : {len(hull.vertices)} of {len(SENSORS)} sensors")
    print(f"covered area           : {area:.2f} m^2")
    print(f"hull perimeter         : {perimeter:.2f} m")

    i, j, gap = cds2.geometry.closest_pair(SENSORS)
    print(f"closest pair           : sensors {i}-{j} at {gap:.2f} m  <- redundancy flag")

    centroid_x, centroid_y = cds2.geometry.centroid(
        [hull.vertices[k] for k in range(len(hull.vertices))]
    )
    ring = hull.vertices.tolist()
    inside_hull = cds2.geometry.point_in_polygon((centroid_x, centroid_y), ring)
    inside_target = cds2.geometry.point_in_polygon((10.8, 5.9), ring)
    print(f"centroid ({centroid_x:.2f}, {centroid_y:.2f}) inside hull: {inside_hull}")
    print(f"target (10.80, 5.90) inside hull: {inside_target}")

    rotated = cds2.geometry.rotate_points(ring, np.pi / 4, pivot=(5.0, 6.0))
    print(f"45-deg rotated hull x-range: [{rotated[:, 0].min():.2f}, {rotated[:, 0].max():.2f}]")


if __name__ == "__main__":
    main()
