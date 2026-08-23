"""Computational geometry: hulls, distances, polygons and intersections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import spatial

__all__ = [
    "HullResult",
    "convex_hull",
    "hull_area",
    "point_cloud_distances",
    "closest_pair",
    "point_in_polygon",
    "polygon_area",
    "polygon_perimeter",
    "segments_intersect",
    "line_intersection",
    "centroid",
    "rotate_points",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class HullResult:
    """Convex hull of a 2-D point set."""

    vertices: FloatArray
    indices: IntArray | None = None


def _as_points(points: Sequence[Sequence[float]] | FloatArray) -> FloatArray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 1:
        msg = "points must be a non-empty (n, 2) array"
        raise ValueError(msg)
    return array


def convex_hull(points: Sequence[Sequence[float]] | FloatArray) -> HullResult:
    """Counter-clockwise convex hull of a 2-D point set via scipy.spatial."""
    pts = _as_points(points)
    if pts.shape[0] < 3:
        unique = np.unique(pts, axis=0)
        return HullResult(vertices=unique)
    hull = spatial.ConvexHull(pts)
    ordered = hull.vertices.tolist()
    return HullResult(vertices=pts[ordered], indices=np.asarray(ordered, dtype=np.int64))


def hull_area(points: Sequence[Sequence[float]] | FloatArray) -> float:
    """Area enclosed by the convex hull of a 2-D point set."""
    result = convex_hull(points)
    if result.vertices.shape[0] < 3:
        return 0.0
    return polygon_area(result.vertices)


def point_cloud_distances(
    a: Sequence[Sequence[float]] | FloatArray,
    b: Sequence[Sequence[float]] | FloatArray | None = None,
) -> FloatArray:
    """Pairwise Euclidean distances; B defaults to A itself."""
    left = _as_points(a)
    right = left if b is None else _as_points(b)
    differences = left[:, None, :] - right[None, :, :]
    return np.asarray(np.sqrt((differences**2).sum(axis=-1)), dtype=float)


def closest_pair(points: Sequence[Sequence[float]] | FloatArray) -> tuple[int, int, float]:
    """Indices and distance of the closest pair of distinct points."""
    pts = _as_points(points)
    n = pts.shape[0]
    if n < 2:
        msg = "need at least two points"
        raise ValueError(msg)
    tree = spatial.cKDTree(pts)
    # Column 0 is each point's zero self-distance; column 1 is its nearest
    # distinct neighbour. Symmetry means the argmin row is the lower index,
    # so no post-hoc ordering of the returned pair is required.
    neighbour_distances, neighbour_indices = tree.query(pts, k=2)
    best_row = int(np.argmin(neighbour_distances[:, 1]))
    return (
        best_row,
        int(neighbour_indices[best_row, 1]),
        float(neighbour_distances[best_row, 1]),
    )


def _polygon_ring(ring: Sequence[Sequence[float]] | FloatArray) -> FloatArray:
    ring_array = _as_points(ring)
    if ring_array.shape[0] < 3:
        msg = "a polygon needs at least three vertices"
        raise ValueError(msg)
    return ring_array


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting containment test for one point against a closed ring."""
    px, py = float(point[0]), float(point[1])
    ring = _polygon_ring(polygon)
    inside = False
    count = ring.shape[0]
    for i in range(count):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % count]
        crosses = (y1 > py) != (y2 > py)
        if crosses:
            x_at_y = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_at_y:
                inside = not inside
    return inside


def polygon_area(ring: Sequence[Sequence[float]] | FloatArray) -> float:
    """Unsigned shoelace area of a simple polygon."""
    vertices = _polygon_ring(ring)
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    shifted_x = np.roll(x_coords, -1)
    shifted_y = np.roll(y_coords, -1)
    signed = float(np.sum(x_coords * shifted_y - shifted_x * y_coords))
    return abs(signed) / 2.0


def polygon_perimeter(ring: Sequence[Sequence[float]] | FloatArray) -> float:
    """Total edge length of a polygon ring (closing edge included)."""
    vertices = _polygon_ring(ring)
    rolled = np.roll(vertices, -1, axis=0)
    return float(np.linalg.norm(rolled - vertices, axis=1).sum())


def segments_intersect(
    segment_a: Sequence[tuple[float, float]],
    segment_b: Sequence[tuple[float, float]],
) -> bool:
    """Orientation-based intersection test between two 2-D segments."""
    p1, p2 = (np.asarray(v, dtype=float) for v in segment_a)
    q1, q2 = (np.asarray(v, dtype=float) for v in segment_b)

    def orientation(u: FloatArray, v: FloatArray, w: FloatArray) -> float:
        cross = (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
        return float(np.sign(cross))

    def on_segment(u: FloatArray, v: FloatArray, w: FloatArray) -> bool:
        within_x = min(u[0], w[0]) <= v[0] <= max(u[0], w[0])
        within_y = min(u[1], w[1]) <= v[1] <= max(u[1], w[1])
        return bool(within_x and within_y)

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)
    if o1 != o2 and o3 != o4 and o1 != 0.0 and o2 != 0.0 and o3 != 0.0 and o4 != 0.0:
        return True
    if o1 == 0.0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0.0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0.0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0.0 and on_segment(q1, p2, q2):
        return True
    return False


def line_intersection(
    line_a: Sequence[tuple[float, float]],
    line_b: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Intersection point of two infinite lines through the given point pairs."""
    (p, r_end), (q, s_end) = (
        (np.asarray(v, dtype=float) for v in line_a),
        (np.asarray(v, dtype=float) for v in line_b),
    )
    direction_r = r_end - p
    direction_s = s_end - q
    denominator = direction_r[0] * direction_s[1] - direction_r[1] * direction_s[0]
    if abs(denominator) < 1e-15:
        msg = "lines are parallel or collinear"
        raise ValueError(msg)
    difference = q - p
    t = (difference[0] * direction_s[1] - difference[1] * direction_s[0]) / denominator
    point = p + t * direction_r
    return float(point[0]), float(point[1])


def centroid(points: Sequence[Sequence[float]] | FloatArray) -> tuple[float, float]:
    """Arithmetic mean position of a point set."""
    pts = _as_points(points)
    mean = pts.mean(axis=0)
    return float(mean[0]), float(mean[1])


def rotate_points(
    points: Sequence[Sequence[float]] | FloatArray,
    angle_rad: float,
    pivot: Sequence[float] | None = None,
) -> FloatArray:
    """Rotate a 2-D point set by ``angle_rad`` around ``pivot`` (default origin)."""
    pts = _as_points(points)
    center = np.zeros(2) if pivot is None else np.asarray(pivot, dtype=float)[:2]
    cos_a, sin_a = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    shifted = pts - center
    return np.asarray((rotation @ shifted.T).T + center, dtype=float)
