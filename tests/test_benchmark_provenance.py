"""Tests for provenance-rich benchmark reporting."""

from __future__ import annotations

import json

from benchmarks.research_run import collect_provenance, run_research_benchmarks


def test_collect_provenance_contains_reproducibility_fields() -> None:
    provenance = collect_provenance(command=["--quick", "--core-only"])

    assert provenance["schema_version"] == 1
    assert provenance["command"] == ["--quick", "--core-only"]
    assert provenance["python"]["version"]
    assert provenance["python"]["implementation"]
    assert provenance["operating_system"]["system"]
    assert "logical_cpu_count" in provenance["hardware"]
    assert "package_versions" in provenance
    assert "numpy_configuration" in provenance
    assert set(provenance["threading_environment"]).issubset(
        {
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        }
    )


def test_research_run_writes_protocol_and_provenance(tmp_path) -> None:  # type: ignore[no-untyped-def]
    results = run_research_benchmarks(
        tmp_path,
        quick=True,
        benchmarks=["solve_small"],
        include_optional=False,
        command=["--quick", "--only", "solve_small"],
    )

    assert len(results) == 1
    results_payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    provenance_payload = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))

    assert results_payload["protocol"]["quick_mode"] is True
    assert results_payload["protocol"]["selected_benchmarks"] == ["solve_small"]
    assert results_payload["protocol"]["optional_benchmarks_enabled"] is False
    assert results_payload["provenance"] == provenance_payload
    assert results_payload["results"][0]["name"].startswith("solve 8x8")
