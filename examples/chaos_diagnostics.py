"""Case study 8: Detecting chaos in a measured time series.

A laboratory-style series is analysed with the standard nonlinear-dynamics
toolchain: delay embedding, false nearest neighbours for the embedding
dimension, Rosenstein's Lyapunov exponent and the correlation dimension.
Chaotic and periodic regimes of the logistic map serve as ground truth.
"""

from __future__ import annotations

import numpy as np

import cds2


def describe(label: str, series: np.ndarray) -> None:
    fnn_low = cds2.chaos.false_nearest_fraction(series, dimension=1)
    fnn_high = cds2.chaos.false_nearest_fraction(series, dimension=4)
    lyap = cds2.chaos.largest_lyapunov_exponent(series, embedding_dimension=5).exponent
    dim = cds2.chaos.correlation_dimension(series, embedding_dimension=5).dimension

    print(f"-- {label}")
    print(f"   false-nearest fraction  d=1: {fnn_low:.3f}   d=4: {fnn_high:.3f}")
    print(f"   largest Lyapunov exponent   : {lyap:+.3f}")
    print(f"   correlation dimension       : {dim:.3f}")


def main() -> None:
    chaotic = cds2.chaos.logistic_map(3.99, length=800, x0=0.2, seed=11)
    periodic = cds2.chaos.logistic_map(3.20, length=800, x0=0.2, seed=12)
    noise = np.random.default_rng(3).normal(size=800)

    print("== Chaos diagnostics ==")
    describe("logistic map r=3.99 (chaotic)", chaotic)
    describe("logistic map r=3.20 (period-2)", periodic)
    describe("white noise (infinite dimensional)", noise)

    params, states = cds2.chaos.bifurcation_scan(
        lambda r, x: r * x * (1.0 - x),
        np.linspace(2.8, 4.0, num=120),
        iterations=400,
        last_values=60,
        seed=5,
    )
    spread = float(np.std(states))
    print(f"\nBifurcation scan: {params.size} asymptotic states, std={spread:.3f}")


if __name__ == "__main__":
    main()
