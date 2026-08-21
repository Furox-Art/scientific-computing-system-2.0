"""Tests for cds2.signals."""

import numpy as np
import pytest

from cds2 import signals


def _sine(fs: float, freq: float, n: int = 1024) -> np.ndarray:
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * freq * t)


class TestFFT:
    def test_roundtrip(self) -> None:
        x = _sine(256.0, 32.0)
        assert np.allclose(signals.ifft(signals.fft(x)).real, x)

    def test_rfft_peak_location(self) -> None:
        spectrum = signals.rfft(_sine(256.0, 64.0))
        expected_bin = int(64.0 * 1024 / 256)
        assert int(np.argmax(np.abs(spectrum))) == expected_bin

    def test_irfft_roundtrip(self) -> None:
        x = _sine(128.0, 16.0)
        assert np.allclose(signals.irfft(signals.rfft(x), n=len(x)), x)

    def test_fftfreq_spacing(self) -> None:
        freqs = signals.fftfreq(8, dt=0.5)
        assert freqs[1] == pytest.approx(0.25)


class TestSpectra:
    def test_periodogram_finds_tone(self) -> None:
        spec = signals.power_spectrum(_sine(256.0, 50.0), fs=256.0)
        assert float(spec.frequencies[np.argmax(spec.power)]) == pytest.approx(50.0)

    def test_welch_finds_tone(self) -> None:
        spec = signals.welch_spectrum(_sine(512.0, 100.0), fs=512.0)
        assert float(spec.frequencies[np.argmax(spec.power)]) == pytest.approx(100.0)

    def test_spectrogram_shapes(self) -> None:
        result = signals.spectrogram(_sine(256.0, 40.0, n=512), fs=256.0, nperseg=128)
        assert result.power.shape[0] == len(result.frequencies)
        assert result.power.shape[1] == len(result.times)


class TestFilters:
    def test_lowpass_attenuates_high(self) -> None:
        fs = 1000.0
        sos = signals.butter_lowpass(50.0, fs, order=6)
        slow = signals.filter_signal(_sine(fs, 5.0), sos)
        fast = signals.filter_signal(_sine(fs, 300.0), sos)
        assert np.std(slow[100:-100]) > 10 * np.std(fast[100:-100])

    def test_highpass_removes_dc_drift(self) -> None:
        fs = 200.0
        t = np.arange(2000) / fs
        drifted = np.sin(2 * np.pi * 20.0 * t) + np.linspace(0.0, 3.0, 2000)
        cleaned = signals.filter_signal(drifted, signals.butter_highpass(10.0, fs))
        assert abs(np.mean(cleaned)) < 0.05

    def test_bandpass_isolates_midband(self) -> None:
        fs = 500.0
        t = np.arange(1500) / fs
        mixed = (
            np.sin(2 * np.pi * 5.0 * t)
            + np.sin(2 * np.pi * 40.0 * t)
            + np.sin(2 * np.pi * 200.0 * t)
        )
        band = signals.filter_signal(mixed, signals.butter_bandpass(20.0, 90.0, fs))
        spec = signals.power_spectrum(band[200:], fs=fs)
        peak = float(spec.frequencies[np.argmax(spec.power)])
        assert 30.0 <= peak <= 60.0


class TestOperations:
    def test_moving_average_smooths_noise(self) -> None:
        rng_values = np.random.default_rng(3).normal(size=500)
        smoothed = signals.moving_average(rng_values, window=25)
        assert np.std(smoothed[25:-25]) < np.std(rng_values[25:-25])

    def test_convolve_delta_identity(self) -> None:
        a = [1.0, 2.0, 3.0]
        delta = [0.0, 1.0, 0.0]
        assert np.allclose(signals.convolve_signals(a, delta), a)

    def test_correlate_self_peak(self) -> None:
        pattern = np.sin(np.linspace(0.0, 3 * np.pi, 30))
        corr = signals.correlate_signals(pattern, pattern)
        assert int(np.argmax(corr)) == len(pattern) // 2

    def test_find_peaks_two_bumps(self) -> None:
        y = np.concatenate([np.zeros(20), [1.0], np.zeros(19), [2.0], np.zeros(20)])
        peaks = signals.find_peaks(y, height=0.5, distance=5)
        assert len(peaks.indices) == 2
        assert max(peaks.heights) == pytest.approx(2.0)

    def test_envelope_matches_amplitude(self) -> None:
        tone = _sine(256.0, 10.0, n=512)
        env = signals.envelope(tone)
        assert np.median(env[len(env) // 4 : -len(env) // 4]) == pytest.approx(1.0, rel=0.05)

    def test_resample_length(self) -> None:
        out = signals.resample_signal(_sine(100.0, 10.0, n=400), num=100)
        assert len(out) == 100

    def test_detrend_removes_slope(self) -> None:
        ramp = np.linspace(0.0, 10.0, 200)
        flat = signals.detrend_signal(ramp)
        assert np.max(np.abs(flat)) < 1e-9
