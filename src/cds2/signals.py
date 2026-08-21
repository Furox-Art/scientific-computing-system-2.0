"""Signal processing built on scipy.signal and numpy.fft."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

__all__ = [
    "Spectrum",
    "SpectrogramResult",
    "PeakResult",
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fftfreq",
    "power_spectrum",
    "welch_spectrum",
    "spectrogram",
    "butter_lowpass",
    "butter_highpass",
    "butter_bandpass",
    "filter_signal",
    "moving_average",
    "convolve_signals",
    "correlate_signals",
    "find_peaks",
    "envelope",
    "resample_signal",
    "detrend_signal",
]


@dataclass(frozen=True)
class Spectrum:
    """Power spectral density estimate."""

    frequencies: np.ndarray
    power: np.ndarray


@dataclass(frozen=True)
class SpectrogramResult:
    """Short-time Fourier transform magnitude spectrogram."""

    times: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray


@dataclass(frozen=True)
class PeakResult:
    """Indices and amplitudes of detected peaks."""

    indices: np.ndarray
    heights: np.ndarray


def fft(x: object) -> np.ndarray:
    """Full complex discrete Fourier transform."""
    return np.asarray(np.fft.fft(np.asarray(x)))


def ifft(x: object) -> np.ndarray:
    """Inverse discrete Fourier transform."""
    return np.asarray(np.fft.ifft(np.asarray(x)))


def rfft(x: object) -> np.ndarray:
    """Real-input FFT returning the non-negative frequency half."""
    return np.asarray(np.fft.rfft(np.asarray(x)))


def irfft(x: object, n: int | None = None) -> np.ndarray:
    """Inverse real FFT back to a time-domain signal."""
    return np.asarray(np.fft.irfft(np.asarray(x), n=n))


def fftfreq(n: int, dt: float = 1.0) -> np.ndarray:
    """FFT sample frequencies for a window of ``n`` samples with spacing ``dt``."""
    return np.asarray(np.fft.fftfreq(n, d=dt))


def power_spectrum(x: object, fs: float) -> Spectrum:
    """Periodogram-based power spectral density of a uniformly sampled signal."""
    freqs, psd = sps.periodogram(np.asarray(x, dtype=float), fs=fs)
    return Spectrum(frequencies=np.asarray(freqs), power=np.asarray(psd))


def welch_spectrum(x: object, fs: float, nperseg: int = 256) -> Spectrum:
    """Welch-averaged power spectral density (lower variance than periodogram)."""
    freqs, psd = sps.welch(np.asarray(x, dtype=float), fs=fs, nperseg=min(nperseg, len(x)))
    return Spectrum(frequencies=np.asarray(freqs), power=np.asarray(psd))


def spectrogram(x: object, fs: float, nperseg: int = 256) -> SpectrogramResult:
    """STFT magnitude squared over sliding windows."""
    freqs, times, sxx = sps.spectrogram(
        np.asarray(x, dtype=float),
        fs=fs,
        nperseg=min(nperseg, len(x)),
    )
    return SpectrogramResult(
        times=np.asarray(times), frequencies=np.asarray(freqs), power=np.asarray(sxx)
    )


def butter_lowpass(cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    """Second-order-section coefficients of a Butterworth low-pass filter."""
    nyquist = fs / 2.0
    sos = sps.butter(order, cutoff / nyquist, btype="low", output="sos")
    return np.asarray(sos)


def butter_highpass(cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    """Second-order-section coefficients of a Butterworth high-pass filter."""
    nyquist = fs / 2.0
    sos = sps.butter(order, cutoff / nyquist, btype="high", output="sos")
    return np.asarray(sos)


def butter_bandpass(low: float, high: float, fs: float, order: int = 5) -> np.ndarray:
    """Second-order-section coefficients of a Butterworth band-pass filter."""
    nyquist = fs / 2.0
    sos = sps.butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")
    return np.asarray(sos)


def filter_signal(x: object, sos: object) -> np.ndarray:
    """Zero-phase forward-backward filtering with second-order sections."""
    filtered = sps.sosfiltfilt(sos, np.asarray(x, dtype=float))
    return np.asarray(filtered)


def moving_average(x: object, window: int) -> np.ndarray:
    """Centered moving average with edge-preserving 'same' convolution."""
    if window < 1:
        msg = "window must be at least 1"
        raise ValueError(msg)
    kernel = np.ones(window) / window
    return np.asarray(np.convolve(np.asarray(x, dtype=float), kernel, mode="same"))


def convolve_signals(a: object, b: object, mode: str = "same") -> np.ndarray:
    """Linear convolution of two sequences."""
    return np.asarray(
        sps.convolve(np.asarray(a, dtype=float), np.asarray(b, dtype=float), mode=mode)
    )


def correlate_signals(a: object, b: object, mode: str = "same") -> np.ndarray:
    """Cross-correlation of two sequences."""
    return np.asarray(
        sps.correlate(np.asarray(a, dtype=float), np.asarray(b, dtype=float), mode=mode)
    )


def find_peaks(
    x: object,
    height: float | None = None,
    distance: int | None = None,
    prominence: float | None = None,
) -> PeakResult:
    """Local maxima detection with optional height/distance/prominence filters."""
    values = np.asarray(x, dtype=float)
    kwargs: dict[str, float] = {}
    if height is not None:
        kwargs["height"] = height
    if distance is not None:
        kwargs["distance"] = distance
    if prominence is not None:
        kwargs["prominence"] = prominence
    indices, _properties = sps.find_peaks(values, **kwargs)
    return PeakResult(indices=np.asarray(indices), heights=values[indices])


def envelope(x: object) -> np.ndarray:
    """Analytic-signal amplitude envelope via the Hilbert transform."""
    analytic = sps.hilbert(np.asarray(x, dtype=float))
    return np.abs(analytic)


def resample_signal(x: object, num: int) -> np.ndarray:
    """Fourier-method resampling to exactly ``num`` samples."""
    resampled = sps.resample(np.asarray(x, dtype=float), num)
    return np.asarray(resampled)


def detrend_signal(x: object) -> np.ndarray:
    """Remove the best-fit linear trend from a signal."""
    detrended = sps.detrend(np.asarray(x, dtype=float))
    return np.asarray(detrended)
