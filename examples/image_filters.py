"""Case study 17: Image filtering pipeline.

A synthetic 64x64 scene - bright square on a gradient - is blurred with a
Gaussian kernel, edge-detected with Sobel operators, pooled down and
cleaned up with binary morphology.
"""

from __future__ import annotations

import numpy as np

import cds2


def build_scene(size: int = 64) -> np.ndarray:
    x_coords, y_coords = np.meshgrid(np.arange(size), np.arange(size))
    scene = 40.0 + 60.0 * (x_coords + y_coords) / (2.0 * size)
    scene[16:48, 20:44] = 230.0
    rng = np.random.default_rng(3)
    return scene + rng.normal(scale=4.0, size=scene.shape)


def main() -> None:
    scene = build_scene()
    print("== Synthetic scene ==")
    print(f"size {scene.shape[0]}x{scene.shape[1]}, range [{scene.min():.0f}, {scene.max():.0f}]")

    kernel = cds2.image.gaussian_kernel(size=7, sigma=1.6)
    print(f"\ngaussian kernel sums to {kernel.sum():.6f}")
    blurred = cds2.image.gaussian_blur(scene, sigma=1.6, size=7)
    noise_proxy = float(blurred.std() / scene.std())
    print(f"std after blur: {blurred.std():.1f} ({noise_proxy:.0%} of original)")

    edges = cds2.image.sobel_edges(blurred)
    strong_edges = int((edges.magnitude > 120).sum())
    print(f"strong edge pixels (>120): {strong_edges}")

    pooled = cds2.image.downsample(edges.magnitude, factor=4, method="max")
    print(f"max-pooled edge map: {pooled.shape[0]}x{pooled.shape[1]}")

    binary = cds2.image.binarize(edges.magnitude, threshold=120)
    eroded = cds2.image.erode(binary, structure_size=3)
    dilated = cds2.image.dilate(binary, structure_size=3)
    print("\n== Binary morphology on edge mask ==")
    print(f"edge pixels : {int(binary.sum())}")
    print(f"after erode : {int(eroded.sum())}")
    print(f"after dilate: {int(dilated.sum())}")


if __name__ == "__main__":
    main()
