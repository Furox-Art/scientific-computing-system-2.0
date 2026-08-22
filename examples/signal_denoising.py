"""Case study 1: signal denoising with Butterworth filtering and PSD.

Simulates a 50 Hz measurement corrupted by 60 Hz mains interference and
broadband noise, designs a band-stop-free low-pass chain, and verifies the
recovery through Welch spectra.
"""

from __future__ import annotations

import numpy as np

import cds2


def main() -> None:
    fs = 1000.0
    time = np.arange(8000) / fs
    clean = np.sin(2 * np.pi * 50.0 * time)
    noisy = clean + 0.8 * np.sin(2 * np.pi * 210.0 * time)
    rng_values = np.random.default_rng(7)
    measured = noisy + rng_values.normal(scale=0.4, size=noisy.size)

    filtered = cds2.signals.filter_signal(measured, cds2.signals.butter_lowpass(120.0, fs, order=6))

    clean_spectrum = cds2.signals.welch_spectrum(clean[500:-500], fs=fs)
    raw_spectrum = cds2.signals.welch_spectrum(measured[500:-500], fs=fs)
    filtered_spectrum = cds2.signals.welch_spectrum(filtered[500:-500], fs=fs)

    def tone_power(spectrum: cds2.signals.Spectrum, target_hz: float) -> float:
        index = int(np.argmin(np.abs(spectrum.frequencies - target_hz)))
        return float(spectrum.power[index])

    residual = clean[500:-500] - cds2.signals.moving_average(filtered[500:-500], window=1)

    print("== Signal denoising ==")
    print(
        f"50 Hz tone preserved : {tone_power(filtered_spectrum, 50.0):.4f} "
        f"(clean {tone_power(clean_spectrum, 50.0):.4f})"
    )
    print(
        f"210 Hz interference  : {tone_power(raw_spectrum, 210.0):.4f} -> "
        f"{tone_power(filtered_spectrum, 210.0):.6f}"
    )
    print(f"RMS residual vs clean: {float(np.std(residual)):.4f}")


if __name__ == "__main__":
    main()
