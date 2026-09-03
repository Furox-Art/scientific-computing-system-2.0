"""Tests for cds2.prof profiling, benchmark history and regression gates."""

from datetime import date

import pandas as pd
import pytest

from cds2 import prof
from cds2.prof import gates
from cds2.prof.gates import GateResult, RegressionGate
from cds2.prof.history import BenchEntry, BenchHistory


def _work() -> int:
    total = 0
    for i in range(1000):
        total += i
    return total


class TestProfile:
    def test_profile_returns_statistics(self) -> None:
        result = prof.profile(_work, repeats=3)
        assert result.name == "_work"
        assert result.repeats == 3
        assert result.wall_min <= result.wall_median <= result.wall_max
        assert result.wall_min >= 0.0
        assert result.cpu_min <= result.cpu_median <= result.cpu_max
        assert result.peak_rss_mb >= 0.0

    def test_profile_summary_mentions_name(self) -> None:
        result = prof.profile(_work, repeats=1)
        assert "_work" in result.summary()

    def test_profile_rejects_zero_repeats(self) -> None:
        with pytest.raises(ValueError):
            prof.profile(_work, repeats=0)

    def test_timed_decorator_returns_value_and_profiles(self) -> None:
        @prof.timed
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5
        last = add.last_profile  # type: ignore[attr-defined]
        assert last.repeats == 1
        assert last.wall_median >= 0.0


class TestBenchHistory:
    def test_append_and_load_roundtrip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        path = hist.append(
            [
                BenchEntry(
                    name="solve", baseline_library="scipy", baseline_seconds=1.0, cds2_seconds=0.9
                )
            ],
            run_id="run-1",
        )
        assert path.exists()
        df = hist.load_range()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "solve"
        assert df.iloc[0]["ratio"] == pytest.approx(0.9)

    def test_append_accepts_plain_dicts(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        hist.append(
            [
                {
                    "name": "solve",
                    "baseline_library": "scipy",
                    "baseline_seconds": 2.0,
                    "cds2_seconds": 1.0,
                }
            ],
            run_id="run-1",
        )
        df = hist.load_range()
        assert df.iloc[0]["ratio"] == pytest.approx(0.5)

    def test_empty_history_loads_empty_frame(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        df = hist.load_range()
        assert df.empty
        assert "ratio" in df.columns

    def test_date_objects_filter_range(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        hist.append(
            [
                BenchEntry(
                    name="solve", baseline_library="s", baseline_seconds=1.0, cds2_seconds=1.0
                )
            ],
            run_id="run-1",
        )
        assert len(hist.load_range(start=date(2000, 1, 1), end=date(2100, 1, 1))) == 1
        assert hist.load_range(start=date(2100, 1, 1)).empty
        assert hist.load_range(end=date(2000, 1, 1)).empty

    def test_bad_lines_are_skipped(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        p = hist.append(
            [
                BenchEntry(
                    name="solve", baseline_library="s", baseline_seconds=1.0, cds2_seconds=1.0
                )
            ],
            run_id="run-1",
        )
        with p.open("a", encoding="utf-8") as fh:
            fh.write("not json\n\n")
        assert len(hist.load_range()) == 1

    def test_invalid_run_id_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        with pytest.raises(ValueError):
            hist.append([], run_id="!!!")

    def test_zero_baseline_ratio_is_inf(self) -> None:
        entry = BenchEntry(name="x", baseline_library="s", baseline_seconds=0.0, cds2_seconds=1.0)
        assert entry.ratio == float("inf")


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestRegressionGate:
    def test_empty_frame_passes_vacuously(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        gate = RegressionGate(history=BenchHistory(history_dir=tmp_path))
        assert gate.check_latest(pd.DataFrame()) == []

    def test_frame_without_name_column_passes_vacuously(self) -> None:
        gate = RegressionGate()
        df = _frame([{"timestamp": "2026-01-01T00:00:00", "ratio": 9.0}])
        assert gate.check_latest(df) == []

    def test_stable_benchmark_passes(self) -> None:
        gate = RegressionGate(tolerance=0.10)
        df = _frame(
            [
                {"name": "solve", "timestamp": "2026-01-01T00:00:00", "ratio": 1.0},
                {"name": "solve", "timestamp": "2026-01-02T00:00:00", "ratio": 1.0},
            ]
        )
        results = gate.check_latest(df)
        assert len(results) == 1
        assert results[0].passed
        assert "PASS" in str(results[0])

    def test_regressed_benchmark_fails_and_raises(self) -> None:
        gate = RegressionGate(tolerance=0.10)
        df = _frame(
            [
                {"name": "solve", "timestamp": "2026-01-01T00:00:00", "ratio": 1.0},
                {"name": "solve", "timestamp": "2026-01-02T00:00:00", "ratio": 5.0},
            ]
        )
        results = gate.check_latest(df)
        assert not results[0].passed
        assert "FAIL" in str(results[0])
        with pytest.raises(AssertionError):
            gate.assert_latest(df)

    def test_name_filter_selects_benchmark(self) -> None:
        gate = RegressionGate(tolerance=0.10)
        df = _frame(
            [
                {"name": "solve", "timestamp": "2026-01-01T00:00:00", "ratio": 1.0},
                {"name": "other", "timestamp": "2026-01-01T00:00:00", "ratio": 5.0},
            ]
        )
        assert [r.name for r in gate.check_latest(df, name="solve")] == ["solve"]

    def test_check_latest_loads_from_history(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        hist.append(
            [
                BenchEntry(
                    name="solve", baseline_library="s", baseline_seconds=1.0, cds2_seconds=1.0
                )
            ],
            run_id="run-1",
        )
        gate = RegressionGate(history=hist)
        results = gate.check_latest()
        assert len(results) == 1
        assert results[0].passed

    def test_assert_latest_passes_on_stable(self) -> None:
        gate = RegressionGate(tolerance=0.10)
        df = _frame(
            [
                {"name": "solve", "timestamp": "2026-01-01T00:00:00", "ratio": 1.0},
                {"name": "solve", "timestamp": "2026-01-02T00:00:00", "ratio": 1.0},
            ]
        )
        assert gate.assert_latest(df) is None

    def test_entries_with_explicit_ratio_kept(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        hist.append(
            [
                {
                    "name": "solve",
                    "baseline_library": "s",
                    "baseline_seconds": 1.0,
                    "cds2_seconds": 2.0,
                    "ratio": 9.99,
                }
            ],
            run_id="run-1",
        )
        df = hist.load_range()
        assert df.iloc[0]["ratio"] == pytest.approx(9.99)

    def test_git_commit_unknown_when_git_missing(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import subprocess

        from cds2.prof import history as history_mod

        def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise FileNotFoundError("no git")

        monkeypatch.setattr(subprocess, "run", _boom)
        assert history_mod._git_commit() == "unknown"

    def test_invalid_parameters_rejected(self) -> None:
        with pytest.raises(ValueError):
            RegressionGate(tolerance=-0.1)
        with pytest.raises(ValueError):
            RegressionGate(lookback=0)

    def test_empty_series_band_is_tolerance_floor(self) -> None:
        gate = RegressionGate(tolerance=0.10)
        assert gate._band(pd.Series(dtype=float)) == pytest.approx(1.10)

    def test_gate_result_str(self) -> None:
        assert "PASS" in str(GateResult(name="a", latest_ratio=1.0, limit=1.1, passed=True))

    def test_pytest_plugin_hooks(self) -> None:
        seen: dict = {}

        class Parser:
            def addoption(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                seen.update(kwargs)

        gates.pytest_addoption(Parser())  # type: ignore[arg-type]
        assert seen["default"] is False

        class Config:
            def __init__(self, flag: bool):
                self.flag = flag

            def getoption(self, name: str) -> bool:  # type: ignore[no-untyped-def]
                return self.flag

        class Item:
            def __init__(self, keywords=None) -> None:  # type: ignore[no-untyped-def]
                self.keywords = keywords if keywords is not None else {"regression": True}
                self.markers: list = []

            def add_marker(self, marker) -> None:  # type: ignore[no-untyped-def]
                self.markers.append(marker)

        items = [Item(), Item(), Item(keywords={"other": True})]
        gates.pytest_collection_modifyitems(Config(False), items)  # type: ignore[arg-type]
        assert len(items[0].markers) == 1
        assert len(items[1].markers) == 1
        assert items[2].markers == []
        items = [Item()]
        gates.pytest_collection_modifyitems(Config(True), items)  # type: ignore[arg-type]
        assert items[0].markers == []


class TestGpuSoftDependency:
    def test_gpu_reports_unavailable_without_cupy(self) -> None:
        from cds2 import gpu

        assert gpu.is_available() is False

    def test_ensure_cupy_raises_helpful_error(self) -> None:
        from cds2 import gpu
        from cds2.gpu import linalg as gpu_linalg
        from cds2.gpu import montecarlo as gpu_montecarlo
        from cds2.gpu import signal as gpu_signal

        with pytest.raises(RuntimeError, match="CuPy"):
            gpu.cupy()
        # synchronize() is a no-op when CuPy is absent
        assert gpu.synchronize() is None
        with pytest.raises(RuntimeError, match="CuPy"):
            gpu_linalg.solve([[1.0]], [1.0])
        with pytest.raises(RuntimeError, match="CuPy"):
            gpu_signal.fft([1.0, 2.0])
        with pytest.raises(RuntimeError, match="CuPy"):
            gpu_montecarlo.pi_estimate(n_samples=100)
