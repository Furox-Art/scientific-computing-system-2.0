"""Grayscale image processing: convolution, filtering and morphology."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EdgeResult",
    "binarize",
    "convolve2d",
    "dilate",
    "downsample",
    "erode",
    "gaussian_blur",
    "gaussian_kernel",
    "sobel_edges",
]

FloatArray = NDArray[np.float64]

ImageLike = Sequence[Sequence[float]] | FloatArray


def _as_image(img: ImageLike) -> FloatArray:
    array = np.asarray(img, dtype=float)
    if array.ndim != 2 or array.size == 0:
        msg = "image must be a non-empty 2-D array"
        raise ValueError(msg)
    return array


def _as_square_kernel(kernel: ImageLike) -> FloatArray:
    array = np.asarray(kernel, dtype=float)
    side_ok = array.ndim == 2 and array.shape[0] == array.shape[1]
    if not side_ok or array.shape[0] % 2 == 0 or array.shape[0] < 1:
        msg = "kernel must be square with an odd side"
        raise ValueError(msg)
    return array


def _sliding_dot(padded: FloatArray, kernel: FloatArray) -> FloatArray:
    side = kernel.shape[0]
    windows = np.lib.stride_tricks.sliding_window_view(padded, (side, side))
    return np.asarray(np.einsum("ijkl,kl->ij", windows, kernel), dtype=float)


def convolve2d(image: ImageLike, kernel: ImageLike, mode: str = "same") -> FloatArray:
    """True 2-D convolution (kernel flipped) with zero padding in ``same`` mode."""
    img = _as_image(image)
    ker = _as_square_kernel(kernel)
    if mode not in ("same", "valid"):
        msg = "mode must be same or valid"
        raise ValueError(msg)
    side = ker.shape[0]
    pad = side // 2
    flipped = np.ascontiguousarray(ker[::-1, ::-1])
    if mode == "same":
        padded = np.pad(img, pad, mode="constant", constant_values=0.0)
    else:
        padded = img
    return _sliding_dot(padded, flipped)


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> FloatArray:
    """Separable Gaussian kernel normalized to unit sum."""
    if size < 3 or size % 2 == 0:
        msg = "kernel size must be odd and at least 3"
        raise ValueError(msg)
    if sigma <= 0:
        msg = "sigma must be positive"
        raise ValueError(msg)
    offsets = np.arange(size, dtype=float) - (size - 1) / 2
    profile = np.exp(-(offsets**2) / (2.0 * sigma**2))
    profile /= profile.sum()
    return np.asarray(np.outer(profile, profile), dtype=float)


def gaussian_blur(image: ImageLike, sigma: float = 1.0, size: int = 5) -> FloatArray:
    """Smooth ``image`` by convolving with a ``size`` x ``size`` Gaussian kernel."""
    return convolve2d(image, gaussian_kernel(size=size, sigma=sigma), mode="same")


@dataclass(frozen=True)
class EdgeResult:
    """Gradient magnitude and orientation returned by :func:`sobel_edges`."""

    magnitude: FloatArray
    direction: FloatArray


def _correlate2d(img: FloatArray, kernel: FloatArray) -> FloatArray:
    """Cross-correlation (no kernel flip) with zero padding, same output shape."""
    pad = kernel.shape[0] // 2
    padded = np.pad(img, pad, mode="constant", constant_values=0.0)
    return _sliding_dot(padded, kernel)


def sobel_edges(image: ImageLike) -> EdgeResult:
    """Sobel gradient magnitude and direction via correlation (no flip)."""
    img = _as_image(image)
    sobel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    sobel_y = sobel_x.T.copy()
    gx = _correlate2d(img, sobel_x)
    gy = _correlate2d(img, sobel_y)
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    return EdgeResult(
        magnitude=np.asarray(magnitude, dtype=float),
        direction=np.asarray(direction, dtype=float),
    )


def downsample(image: ImageLike, factor: int = 2, method: str = "mean") -> FloatArray:
    """Reduce resolution by non-overlapping ``factor`` blocks after cropping."""
    img = _as_image(image)
    if method not in ("mean", "max", "min"):
        msg = "method must be mean, max or min"
        raise ValueError(msg)
    if factor < 2:
        msg = "factor must be at least 2"
        raise ValueError(msg)
    height = (img.shape[0] // factor) * factor
    width = (img.shape[1] // factor) * factor
    blocks = img[:height, :width].reshape(height // factor, factor, width // factor, factor)
    if method == "mean":
        reduced = blocks.mean(axis=(1, 3))
    elif method == "max":
        reduced = blocks.max(axis=(1, 3))
    else:
        reduced = blocks.min(axis=(1, 3))
    return np.asarray(reduced, dtype=float)


def binarize(image: ImageLike, threshold: float) -> FloatArray:
    """Threshold an image into a 0/1 mask of pixels strictly above ``threshold``."""
    img = _as_image(image)
    return np.asarray((img > threshold).astype(float), dtype=float)


def _as_binary(binary: ImageLike) -> FloatArray:
    arr = _as_image(binary)
    if not bool(np.all((arr == 0.0) | (arr == 1.0))):
        msg = "binary image must contain only 0 and 1"
        raise ValueError(msg)
    return arr


def _as_structure_size(structure_size: int) -> int:
    if structure_size < 3 or structure_size % 2 == 0:
        msg = "structure size must be odd and at least 3"
        raise ValueError(msg)
    return structure_size


def erode(binary: ImageLike, structure_size: int = 3) -> FloatArray:
    """Morphological erosion as a min filter over a square window."""
    arr = _as_binary(binary)
    side = _as_structure_size(structure_size)
    pad = side // 2
    padded = np.pad(arr, pad, mode="constant", constant_values=1.0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (side, side))
    return np.asarray(windows.min(axis=(-2, -1)), dtype=float)


def dilate(binary: ImageLike, structure_size: int = 3) -> FloatArray:
    """Morphological dilation as a max filter over a square window."""
    arr = _as_binary(binary)
    side = _as_structure_size(structure_size)
    pad = side // 2
    padded = np.pad(arr, pad, mode="constant", constant_values=0.0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (side, side))
    return np.asarray(windows.max(axis=(-2, -1)), dtype=float)
