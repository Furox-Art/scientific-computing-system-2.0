"""Pure NumPy Haar wavelet transforms: decomposition, reconstruction, denoising."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DWTResult",
    "DenoiseResult",
    "haar_dwt",
    "haar_idwt",
    "dwt_levels",
    "idwt_levels",
    "wavelet_denoise",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _as_signal(values: Sequence[float] | FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        msg = "values must be a 1-D series of at least two points"
        raise ValueError(msg)
    return array


def haar_dwt(
    signal: Sequence[float] | FloatArray, normalize: bool = True
) -> tuple[FloatArray, FloatArray]:
    """Single-level Haar decomposition into pairwise averages and differences."""
    series = _as_signal(signal)
    if series.size % 2 != 0:
        msg = "signal length must be even"
        raise ValueError(msg)
    pairs = series.reshape(-1, 2)
    scale = np.sqrt(2.0) if normalize else 2.0
    approx = np.asarray((pairs[:, 0] + pairs[:, 1]) / scale)
    detail = np.asarray((pairs[:, 0] - pairs[:, 1]) / scale)
    return approx, detail


def haar_idwt(
    approx: Sequence[float] | FloatArray,
    detail: Sequence[float] | FloatArray,
    normalize: bool = True,
) -> FloatArray:
    """Single-level Haar inverse transform restoring the paired samples."""
    averages = np.asarray(approx, dtype=float)
    differences = np.asarray(detail, dtype=float)
    if averages.ndim != 1 or differences.ndim != 1 or averages.size < 1 or differences.size < 1:
        msg = "approximation and detail must be non-empty 1-D sequences"
        raise ValueError(msg)
    if averages.shape != differences.shape:
        msg = "approximation and detail must have the same length"
        raise ValueError(msg)
    scale = np.sqrt(2.0) if normalize else 1.0
    restored = np.empty(averages.size * 2, dtype=float)
    restored[0::2] = (averages + differences) / scale
    restored[1::2] = (averages - differences) / scale
    return restored


@dataclass(frozen=True)
class DWTResult:
    """Multi-level Haar decomposition of a finite series."""

    approximations: list[FloatArray]
    details: list[FloatArray]
    original_length: int


def dwt_levels(
    signal: Sequence[float] | FloatArray, levels: int, normalize: bool = True
) -> DWTResult:
    """Multi-level Haar decomposition applying ``haar_dwt`` to each approximation."""
    if levels < 1:
        msg = "levels must be at least 1"
        raise ValueError(msg)
    series = _as_signal(signal)
    if series.size < 2**levels:
        msg = "signal too short"
        raise ValueError(msg)
    approximations: list[FloatArray] = []
    details: list[FloatArray] = []
    current = series
    for _ in range(levels):
        current, detail = haar_dwt(current, normalize=normalize)
        approximations.append(current)
        details.append(detail)
    return DWTResult(
        approximations=approximations, details=details, original_length=int(series.size)
    )


def idwt_levels(result: DWTResult, normalize: bool = True) -> FloatArray:
    """Reconstruct the original signal from a :class:`DWTResult`."""
    current = result.approximations[-1]
    for detail in reversed(result.details):
        current = haar_idwt(current, detail, normalize=normalize)
    return current


@dataclass(frozen=True)
class DenoiseResult:
    """Outcome of wavelet shrinkage denoising."""

    data: FloatArray
    thresholded_counts: int
    levels: int


def wavelet_denoise(
    signal: Sequence[float] | FloatArray,
    threshold_factor: float = 3.0,
    levels: int = 2,
    normalize: bool = True,
) -> DenoiseResult:
    """Zero detail coefficients below ``threshold_factor`` standard deviations per level."""
    decomposition = dwt_levels(signal, levels, normalize=normalize)
    filtered: list[FloatArray] = []
    zeroed = 0
    for detail in decomposition.details:
        sigma = float(np.median(np.abs(detail)) / 0.6745)
        cutoff = threshold_factor * sigma
        keep = np.abs(detail) >= cutoff
        zeroed += int((~keep).sum())
        filtered.append(np.asarray(np.where(keep, detail, 0.0)))
    reconstructed = DWTResult(
        approximations=decomposition.approximations,
        details=filtered,
        original_length=decomposition.original_length,
    )
    return DenoiseResult(
        data=idwt_levels(reconstructed, normalize=normalize),
        thresholded_counts=zeroed,
        levels=levels,
    )
