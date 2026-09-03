"""GPU-accelerated signal processing via CuPy.

Each function mirrors its CPU counterpart in ``cds2.signals``.
"""

from __future__ import annotations

from typing import Any

from . import _ensure_cupy

__all__ = ["fft", "ifft", "power_spectrum", "rfft"]


def rfft(x: Any, n: int | None = None) -> Any:
    """Real-valued FFT on the GPU.

    Args:
        x: 1-D array-like.
        n: optional FFT length (zero-padded or truncated).
    Returns:
        Complex spectrum as a NumPy array.
    """
    cp = _ensure_cupy()
    X = cp.fft.rfft(cp.asarray(x), n=n)
    return cp.asnumpy(X)


def fft(x: Any, n: int | None = None) -> Any:
    """Complex FFT on the GPU."""
    cp = _ensure_cupy()
    X = cp.fft.fft(cp.asarray(x), n=n)
    return cp.asnumpy(X)


def ifft(X: Any, n: int | None = None) -> Any:
    """Inverse FFT on the GPU."""
    cp = _ensure_cupy()
    x = cp.fft.ifft(cp.asarray(X), n=n)
    return cp.asnumpy(x)


def power_spectrum(x: Any, fs: float = 1.0) -> tuple[Any, Any]:
    """One-sided power spectrum on the GPU.

    Args:
        x: 1-D array-like.
        fs: sampling frequency.
    Returns:
        ``(freqs, psd)`` as NumPy arrays.
    """
    cp = _ensure_cupy()
    X = cp.fft.rfft(cp.asarray(x))
    freqs = cp.fft.rfftfreq(len(x), d=1.0 / fs)
    psd = (cp.abs(X) ** 2) / (fs * len(x))
    psd[1:-1] *= 2.0
    return cp.asnumpy(freqs), cp.asnumpy(psd)
