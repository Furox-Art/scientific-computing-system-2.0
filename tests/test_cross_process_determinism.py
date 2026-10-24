"""Cross-process determinism: identical inputs must produce identical bytes.

The workload runs in two fresh interpreters whose only difference is
``PYTHONHASHSEED``. If knowledge-graph traversal, keyword search or the
descriptive statistics leaked dict/set iteration order into their results,
the two runs would differ.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CHILD = r"""
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
sys.path.insert(0, str(repo_root / "src"))

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
print(json.dumps(payload, ensure_ascii=False))
"""


@cache
def _run_with_hash_seed(seed: str) -> str:
    env = {
        **os.environ,
        "PYTHONHASHSEED": seed,
        "PYTHONIOENCODING": "utf-8",
    }
    completed = subprocess.run(
        [sys.executable, "-c", CHILD, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_workload_is_byte_identical_across_hash_seeds() -> None:
    first = _run_with_hash_seed("0")
    second = _run_with_hash_seed("12345")
    assert first.strip()
    assert first.strip() == second.strip()


def test_workload_exercises_non_empty_results() -> None:
    payload = json.loads(_run_with_hash_seed("0"))
    assert payload["search"], "keyword search produced no hits; the workload is vacuous"
    assert payload["neighbors"], "graph traversal produced no neighbours"
    assert payload["path"] == ["Beta", "Alpha", "Gamma"]
    assert payload["describe"]["n"] == 6
    assert payload["pi"] > 0.0
