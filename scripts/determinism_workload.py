"""Deterministic CDS2 workload sampled by the determinism matrix."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "src"
    if source_dir.is_dir():
        sys.path.insert(0, str(source_dir))

    from cds2 import graph as cds_graph
    from cds2 import sparse as cds_sparse
    from cds2 import spectral, viz
    from cds2.knowledge import Concept, KnowledgeGraph, Notebook, Relation, search
    from cds2.ml import KMeans
    from cds2.montecarlo import pi_estimate
    from cds2.stats import describe

    graph = KnowledgeGraph()
    for name, summary in [
        ("Alpha", "quantum entanglement links particles"),
        ("Beta", "classical mechanics of particles"),
        ("Gamma", "telescope optics and spectroscopy"),
        ("Delta", "quantum tunneling in semiconductors"),
    ]:
        graph.add_concept(Concept(name=name, summary=summary))
    for source, target, label in [
        ("Alpha", "Gamma", "measured-by"),
        ("Beta", "Alpha", "extends"),
        ("Delta", "Gamma", "sampled-with"),
    ]:
        graph.add_relation(Relation(source=source, target=target, label=label))

    notebook = Notebook()
    notebook.add("Lab log", "quantum tunneling observed today", tags=["spectroscopy"])
    notebook.add("Optics lab", "telescope reading", tags=["optics"])

    summary = describe([3.0, 1.0, 2.0, 5.0, 4.0, 2.5])
    weighted_path = np.zeros((9, 9))
    for index in range(8):
        weight = 1.0 + 0.1 * index
        weighted_path[index, index + 1] = weight
        weighted_path[index + 1, index] = weight
    fiedler = spectral.fiedler_vector(weighted_path, seed=20261024)
    smallest = cds_sparse.smallest_eigenpairs(weighted_path, k=2, seed=20261024)
    norm_matrix = np.arange(36.0).reshape(6, 6) % 7.0
    norm_matrix[:, 0] = 0.0
    pagerank_adj = np.zeros((64, 64))
    for index in range(24):
        pagerank_adj[index, (7 * index + 3) % 64] = 1.0 + 0.1 * (index % 5)
        pagerank_adj[index, (11 * index + 5) % 64] = 0.5 + 0.05 * (index % 3)
    pagerank = cds_graph.pagerank(pagerank_adj, max_iter=100, tol=1e-13)
    # Integer grid coordinates keep every squared distance and cluster sum
    # exactly representable, so this payload isolates the compiled Lloyd
    # kernel from floating-point rounding noise.
    kmeans_points = np.array(
        [[index % 7, (index * 3) % 5] for index in range(42)]
        + [[index % 5 + 9, (index * 2) % 6 + 7] for index in range(38)],
        dtype=float,
    )
    kmeans = KMeans(n_clusters=3, max_iter=100, tol=1e-10, seed=20261024).fit(kmeans_points)
    payload = {
        "search": search("quantum particles telescope", graph, notebook),
        "neighbors": graph.neighbors("Alpha"),
        "path": graph.shortest_path("Beta", "Gamma"),
        "describe": {
            "n": summary.n,
            "mean": summary.mean,
            "std": summary.std,
            "minimum": summary.minimum,
            "q25": summary.q25,
            "median": summary.median,
            "q75": summary.q75,
            "maximum": summary.maximum,
            "skewness": summary.skewness,
            "kurtosis": summary.kurtosis,
        },
        "pi": pi_estimate(n=2_000, seed=20261024),
        "spectral": {
            "fiedler": [round(float(value), 6) for value in fiedler],
            "labels": spectral.spectral_cluster(weighted_path, 2, seed=20261024).tolist(),
        },
        "sparse": {
            "eigenvalues": [round(float(value), 6) for value in smallest.eigenvalues],
            "one_norm": round(float(cds_sparse.one_norm_est(norm_matrix, seed=20261024)), 6),
        },
        "pagerank": {
            "head": [round(float(value), 12) for value in pagerank[:4]],
            "sum": round(float(pagerank.sum()), 12),
            "argmax": int(pagerank.argmax()),
        },
        "kmeans": {
            "labels": np.asarray(kmeans.labels_, dtype=np.int64).tolist(),
            "centers": [
                [round(float(value), 9) for value in row]
                for row in np.asarray(kmeans.cluster_centers_)
            ],
            "inertia": round(float(kmeans.inertia_), 6),
        },
    }
    if not payload["search"] or not payload["neighbors"]:
        raise SystemExit("workload produced empty search results; it is vacuous")
    if not payload["spectral"]["fiedler"] or not payload["sparse"]["eigenvalues"]:
        raise SystemExit("workload produced empty numeric results; it is vacuous")
    if abs(payload["pagerank"]["sum"] - 1.0) > 1e-9:
        raise SystemExit("pagerank scores are not normalized; the workload is vacuous")
    if len(set(payload["kmeans"]["labels"])) != 3:
        raise SystemExit("kmeans did not assign every cluster; the workload is vacuous")
    artifact_dir = os.environ.get("DETERMINISM_ARTIFACT_DIR")
    if artifact_dir:
        artifacts = Path(artifact_dir)
        for suffix in (".pdf", ".svg"):
            viz.plot_series(
                [3.0, 1.0, 2.0, 5.0, 4.0, 2.5],
                title="Determinism",
                xlabel="x",
                ylabel="y",
                save=artifacts / f"series{suffix}",
            )
        artifacts.joinpath("payload.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        for name in ("payload.json", "series.pdf", "series.svg"):
            if artifacts.joinpath(name).stat().st_size == 0:
                raise SystemExit(f"artifact {name} is empty")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
