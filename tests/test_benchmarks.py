"""Tests for the benchmark suite (tiny sizes, isolated output)."""

from __future__ import annotations

import json

import benchmarks.run_benchmarks as run_benchmarks
from benchmarks.run_benchmarks import (
    CORE_BENCHMARKS,
    BenchResult,
    format_table,
    run_all,
)


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


class TestInterleavedRace:
    """The race measures both sides under the same machine state.

    Release run 42435498797 reported eight "regressions" that were run-to-run
    noise: the two sides used to be timed back to back, so a drift of the CPU
    state between the phases moved the ratio by tens of percent.
    """

    @staticmethod
    def _fake_clock(monkeypatch, durations: list[float]) -> None:
        """Make ``perf_counter`` advance by the given durations in order."""
        remaining = list(durations)

        class Clock:
            now = 0.0

            def __call__(self) -> float:
                if remaining:
                    self.now += remaining.pop(0)
                return self.now

        monkeypatch.setattr(run_benchmarks.time, "perf_counter", Clock())

    def test_the_two_sides_are_measured_alternately(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        order: list[str] = []

        def baseline() -> None:
            order.append("baseline")

        def contender() -> None:
            order.append("contender")

        self._fake_clock(monkeypatch, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        run_benchmarks._race("bench", "numpy", baseline, contender, 3, 3)
        # One untimed warm-up pair, then three measured pairs.
        assert order == ["baseline", "contender"] * 4

    def test_the_warm_up_is_not_timed(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        calls = {"n": 0}

        def side() -> None:
            calls["n"] += 1

        self._fake_clock(monkeypatch, [0.0, 1.0] * 6)
        run_benchmarks._race("bench", "numpy", side, side, 3, 3)
        # Two warm-up calls plus three measured rounds on each side.
        assert calls["n"] == 8

    def test_each_side_reports_its_median(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # Baseline rounds take 1.0, 5.0, 1.0 s (one disturbed round); contender
        # rounds take 2.0, 2.0, 2.0 s. The median must ignore the disturbance.
        self._fake_clock(monkeypatch, [0.0, 1.0, 0.0, 2.0, 0.0, 5.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0])
        result = run_benchmarks._race("bench", "numpy", lambda: None, lambda: None, 3, 3)
        assert result.baseline_seconds == 1.0
        assert result.cds2_seconds == 2.0
        assert result.ratio == 2.0

    def test_unequal_repeat_counts_stay_interleaved(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        order: list[str] = []

        def baseline() -> None:
            order.append("baseline")

        def contender() -> None:
            order.append("contender")

        self._fake_clock(monkeypatch, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        result = run_benchmarks._race("bench", "numpy", baseline, contender, 3, 2)
        assert order == ["baseline", "contender"] + [
            "baseline",
            "contender",
            "baseline",
            "contender",
            "baseline",
        ]
        assert result.ratio == 1.0


class TestRecordFlag:
    """CI records the run it just measured, from the same invocation.

    The release workflow used to run the quick suite twice -- once for its table
    and once through an inline copy of the recording logic -- so the release paid
    double and two copies of the logic could drift apart.
    """

    def test_record_appends_one_run_to_the_given_history_dir(self, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
        code = run_benchmarks.main(
            [
                "--quick",
                "--core-only",
                "--only",
                "solve_small",
                "--record",
                "--history-dir",
                str(tmp_path),
            ]
        )
        assert code == 0
        runs = sorted(tmp_path.glob("*.jsonl"))
        assert len(runs) == 1
        assert "recorded 1 results" in capsys.readouterr().out

    def test_without_the_flag_nothing_is_recorded(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        code = run_benchmarks.main(
            ["--quick", "--core-only", "--only", "solve_small", "--history-dir", str(tmp_path)]
        )
        assert code == 0
        assert not list(tmp_path.glob("*.jsonl"))
