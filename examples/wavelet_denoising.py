"""Case study 15: Wavelet denoising of a noisy tone.

Two low-frequency tones buried in Gaussian noise are decomposed with the
Haar wavelet; MAD-thresholded detail coefficients suppress the noise while
the reconstruction keeps the tonal structure intact.
"""

from __future__ import annotations

import numpy as np

import cds2


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal**2)))


def main() -> None:
    rng = np.random.default_rng(21)
    t_values = np.linspace(0.0, 1.0, 512)
    clean = np.sin(2.0 * np.pi * 5.0 * t_values) + 0.4 * np.sin(2.0 * np.pi * 11.0 * t_values)
    noise = rng.normal(scale=0.5, size=t_values.size)
    noisy = clean + noise

    print("== Wavelet denoising of a noisy two-tone ==")
    print(f"noise-only RMS        : {rms(noise):.3f}")

    decomposition = cds2.wavelets.dwt_levels(noisy, levels=4)
    print("detail energy per level:")
    for level, detail in enumerate(decomposition.details, start=1):
        energy = float(np.sum(detail**2))
        bar = "#" * int(energy * 2)
        print(f"  level {level}: {energy:8.2f} |{bar}")

    denoised = cds2.wavelets.wavelet_denoise(noisy, threshold_factor=3.0, levels=4)
    raw_error = rms(noisy - clean)
    denoised_error = rms(denoised.data - clean)
    reduction = 100.0 * (1.0 - denoised_error / raw_error)

    print(f"\nRMS error before      : {raw_error:.3f}")
    print(f"RMS error after       : {denoised_error:.3f}")
    print(f"error reduction       : {reduction:.1f}%")
    print(f"coefficients zeroed   : {denoised.thresholded_counts}")

    roundtrip = cds2.wavelets.haar_idwt(*cds2.wavelets.haar_dwt(clean))
    print(f"transform is lossless : {np.allclose(roundtrip, clean)}")


if __name__ == "__main__":
    main()
