"""Tests for the benchmark suite (tiny sizes, isolated output)."""

from __future__ import annotations

import json

from benchmarks.run_benchmarks import CORE_BENCHMARKS, BenchResult, format_table, run_all


class TestRunAll:
    def test_core_run_returns_results(self) -> None:
        results = run_all(quick=True, include_optional=False)
        assert len(results) == len(CORE_BENCHMARKS)
        assert all(isinstance(result, BenchResult) for result in results)

    def test_optional_missing_library_is_skipped(self) -> None:
        results = run_all(quick=True, benchmarks=["networkx_pagerank"])
        networkx_installed = True
        try:
            import networkx  # noqa: F401
        except ImportError:
            networkx_installed = False
        if networkx_installed:
            assert results
        else:
            assert results == []

    def test_selected_subset(self) -> None:
        results = run_all(quick=True, benchmarks=["mc_pi"], include_optional=False)
        assert [result.name for result in results] == ["mc-pi n=200000"]

    def test_json_output_isolated_to_tmp(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        report = tmp_path / "results.json"
        run_all(output_dir=tmp_path, quick=True, benchmarks=["solve_small"], include_optional=False)
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert "environment" in payload
        entry = payload["results"][0]
        assert {
            "name",
            "baseline_library",
            "baseline_seconds",
            "cds2_seconds",
            "ratio_cds2_over_baseline",
        } <= set(entry)


class TestReporting:
    def test_ratio_property(self) -> None:
        result = BenchResult(
            name="demo", baseline_library="numpy", baseline_seconds=2.0, cds2_seconds=0.5
        )
        assert result.ratio == 0.25

    def test_format_table_alignment(self) -> None:
        table = format_table(
            [
                BenchResult(
                    name="demo", baseline_library="numpy", baseline_seconds=1.0, cds2_seconds=0.5
                )
            ]
        )
        assert "demo" in table
        assert "numpy" in table
        assert "0.50x" in table
