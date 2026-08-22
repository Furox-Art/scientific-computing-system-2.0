"""Case study 4: citation-network analysis - PageRank and spectral clusters.

Builds a small directed citation graph between research topics, ranks the
influential nodes with the compiled PageRank kernel, undirected-clusters
the collaboration graph spectrally, and validates the DAG structure of
prerequisite chains topologically.
"""

from __future__ import annotations

import numpy as np

import cds2

TOPICS = [
    "linear-algebra",
    "probability",
    "optimization",
    "statistics",
    "machine-learning",
    "signal-processing",
    "control-theory",
    "quantum-computing",
]
CITATIONS = [
    (0, 3),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 4),
    (3, 6),
    (2, 5),
    (5, 7),
    (0, 7),
    (4, 7),
    (6, 2),
    (1, 5),
    (3, 2),
]
COLLABORATIONS = [
    (i, j) for i in range(8) for j in range(i + 1, 8) if (i, j) in CITATIONS or (j, i) in CITATIONS
]


def main() -> None:
    citation_graph = cds2.graph.from_edges(len(TOPICS), CITATIONS, directed=True)
    scores = cds2.graph.pagerank(citation_graph)

    print("== Citation network ==")
    ranking = np.argsort(scores)[::-1]
    for rank, node in enumerate(ranking[:4], start=1):
        print(f"  {rank}. {TOPICS[node]:<18} score={scores[node]:.4f}")

    collaboration_graph = cds2.graph.from_edges(len(TOPICS), COLLABORATIONS, directed=False)
    labels = cds2.spectral.spectral_cluster(collaboration_graph, n_clusters=3, seed=11)
    groups: dict[int, list[str]] = {}
    for node, label_value in zip(range(len(TOPICS)), labels, strict=True):
        groups.setdefault(int(label_value), []).append(TOPICS[node])

    print("== Spectral collaboration clusters ==")
    for label_value in sorted(groups):
        members = ", ".join(sorted(groups[label_value]))
        print(f"  cluster {label_value}: {members}")

    order = cds2.graph.topological_order(len(TOPICS), CITATIONS)
    positions = {node: index for index, node in enumerate(order)}
    valid_prerequisites = all(positions[before] < positions[after] for before, after in CITATIONS)
    print(f"== Prerequisite DAG consistent : {valid_prerequisites} ==")
    print(
        f"== Algebraic connectivity      : "
        f"{cds2.spectral.algebraic_connectivity(cds2.graph.from_edges(8, COLLABORATIONS, directed=False)):.4f} =="
    )


if __name__ == "__main__":
    main()
