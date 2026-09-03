"""Oracle tests: cds2.signals vs SciPy ground truth."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal as sp_signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import cds2  # noqa: E402


class TestSignalOracles:
    """Verify cds2.signals against SciPy."""

    @pytest.mark.parametrize("seed", range(10))
    def test_welch_vs_scipy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=4096)
        fs = 1000.0
        res = cds2.signals.welch_spectrum(x, fs=fs)
        freqs_c, psd_c = res.frequencies, res.power
        freqs_s, psd_s = sp_signal.welch(x, fs=fs)
        assert np.allclose(freqs_c, freqs_s, atol=1e-6)
        assert np.allclose(psd_c, psd_s, rtol=1e-4)

    @pytest.mark.parametrize("seed", range(10))
    def test_fft_vs_numpy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=1024)
        X_c = cds2.signals.fft(x)
        X_np = np.fft.fft(x)
        assert np.allclose(X_c, X_np, atol=1e-10)

    @pytest.mark.parametrize("seed", range(10))
    def test_rfft_vs_numpy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=1024)
        X_c = cds2.signals.rfft(x)
        X_np = np.fft.rfft(x)
        assert np.allclose(X_c, X_np, atol=1e-10)
