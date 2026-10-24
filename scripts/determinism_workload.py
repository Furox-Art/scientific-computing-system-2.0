"""Deterministic CDS2 workload sampled by the determinism matrix."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "src"
    if source_dir.is_dir():
        sys.path.insert(0, str(source_dir))

    from cds2.knowledge import Concept, KnowledgeGraph, Notebook, Relation, search
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
    }
    if not payload["search"] or not payload["neighbors"]:
        raise SystemExit("workload produced empty search results; it is vacuous")
    artifact_dir = os.environ.get("DETERMINISM_ARTIFACT_DIR")
    if artifact_dir:
        Path(artifact_dir, "payload.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
