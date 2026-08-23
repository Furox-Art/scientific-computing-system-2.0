"""Case study 7: Information-theoretic dependence discovery.

Two sensors observe a shared hidden signal plus independent noise. The
joint histogram is used to estimate mutual information, showing how MI
detects the coupling even when Pearson correlation is diluted by noise.
"""

from __future__ import annotations

import numpy as np

import cds2


def joint_table(x: np.ndarray, y: np.ndarray, bins: int = 12) -> np.ndarray:
    x_edges = np.histogram_bin_edges(x, bins=bins)
    y_edges = np.histogram_bin_edges(y, bins=bins)
    table, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    return table / table.sum()


def main() -> None:
    rng = np.random.default_rng(7)
    hidden = rng.normal(size=4000)
    coupled = 0.8 * hidden + 0.6 * rng.normal(size=4000)
    independent = rng.normal(size=4000)

    print("== Mutual information vs correlation ==")
    for label, series in (("coupled", coupled), ("independent", independent)):
        table = joint_table(hidden, series)
        mi = cds2.infotheory.mutual_information(table)
        nmi = cds2.infotheory.normalized_mutual_information(table)
        r = cds2.stats.pearson_correlation(hidden.tolist(), series.tolist())
        print(f"{label:12s}  MI={mi:.3f} bits  NMI={nmi:.3f}  Pearson r={r.r:.3f}")

    alphabet = "ACGTACGTGATTACAGGCAT"
    counts = [alphabet.count(base) / len(alphabet) for base in "ACGT"]
    print(f"\nDNA alphabet entropy      : {cds2.infotheory.entropy(counts):.3f} bits")
    print(
        f"Permutation entropy sine  : "
        f"{cds2.infotheory.permutation_entropy(np.sin(np.linspace(0, 40 * np.pi, 2000)), order=4):.3f}"
    )


if __name__ == "__main__":
    main()
