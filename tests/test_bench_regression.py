"""Tests for cds2.bench.regression report generation."""

import pytest

from cds2.bench.regression import RegressionReport, run_regression_check
from cds2.prof.history import BenchEntry, BenchHistory


class TestRunRegressionCheck:
    def test_empty_history_passes_vacuously(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        report = run_regression_check(history=hist)
        assert report.passed is True
        assert report.results == []
        assert "ALL PASS" in report.to_table()

    def test_stable_history_passes(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        hist.append(
            [
                BenchEntry(
                    name="solve", baseline_library="s", baseline_seconds=1.0, cds2_seconds=1.0
                )
            ],
            run_id="run-1",
        )
        report = run_regression_check(history=hist)
        assert report.passed is True
        assert len(report.results) == 1
        table = report.to_table()
        assert "solve" in table
        assert "PASS" in table

    def test_regressed_history_fails(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        hist.append(
            [
                BenchEntry(
                    name="solve", baseline_library="s", baseline_seconds=1.0, cds2_seconds=1.0
                )
            ],
            run_id="run-1",
        )
        hist.append(
            [
                BenchEntry(
                    name="solve", baseline_library="s", baseline_seconds=1.0, cds2_seconds=5.0
                )
            ],
            run_id="run-2",
        )
        report = run_regression_check(history=hist)
        assert report.passed is False
        assert "FAIL" in report.to_table()
        assert "FAILURES DETECTED" in report.to_table()

    def test_a_first_run_is_labelled_baseline_not_pass(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """No prior run means no comparison; the table must not imply one.

        Release run 42435498797 is the real case: a table that printed a limit
        next to PASS made a vacuous check indistinguishable from a measured one.
        """
        hist = BenchHistory(history_dir=tmp_path)
        hist.append(
            [
                BenchEntry(
                    name="solve", baseline_library="s", baseline_seconds=1.0, cds2_seconds=0.4
                )
            ],
            run_id="run-1",
        )
        report = run_regression_check(history=hist)
        table = report.to_table()
        assert report.passed is True
        assert report.results[0].baseline is True
        assert "baseline" in table
        assert "FAIL" not in table

    def test_a_compared_row_is_labelled_pass(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        for run_id in ("run-1", "run-2"):
            hist.append(
                [
                    BenchEntry(
                        name="solve", baseline_library="s", baseline_seconds=1.0, cds2_seconds=1.0
                    )
                ],
                run_id=run_id,
            )
        report = run_regression_check(history=hist)
        assert report.results[0].baseline is False
        assert "PASS" in report.to_table()

    def test_report_table_lists_each_result(self) -> None:
        report = RegressionReport(results=[], passed=True)
        assert "ALL PASS" in report.to_table()

    def test_the_table_reports_the_measured_noise(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hist = BenchHistory(history_dir=tmp_path)
        for run_id, noise in (("run-1", 0.0), ("run-2", 0.12)):
            hist.append(
                [
                    BenchEntry(
                        name="solve",
                        baseline_library="s",
                        baseline_seconds=1.0,
                        cds2_seconds=1.0,
                        noise=noise,
                    )
                ],
                run_id=run_id,
            )
        report = run_regression_check(history=hist)
        assert report.results[0].noise == pytest.approx(0.12)
        assert "noise" in report.to_table()
