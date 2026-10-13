"""Tests for the committed benchmark snapshot's provenance marker.

``benchmarks/results.json`` is a derived artifact: a measurement of one old
release on one machine, kept in the repository because ``docs/benchmarks.md``
and ``scripts/make_promo.py`` read it. Without an explicit, machine-checkable
marker it is indistinguishable from a current claim, so the marker and the
regeneration commands are asserted here rather than left to review.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = REPO_ROOT / "benchmarks" / "results.json"


def _payload() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


class TestSnapshotMarker:
    def test_snapshot_declares_itself_generated(self) -> None:
        snapshot = _payload()["_snapshot"]
        assert snapshot["kind"] == "generated"
        assert snapshot["generator"] == "benchmarks/run_benchmarks.py"

    def test_snapshot_records_the_version_it_measured(self) -> None:
        snapshot = _payload()["_snapshot"]
        environment = _payload()["environment"]
        assert snapshot["recorded_cds2_version"] == environment["versions"]["cds2"]
        assert snapshot["recorded_commit"] == environment["git_commit"]

    def test_snapshot_does_not_claim_to_be_current(self) -> None:
        assert _payload()["_snapshot"]["is_current_release"] is False

    def test_snapshot_names_the_regeneration_commands(self) -> None:
        commands = _payload()["_snapshot"]["regenerate"]
        assert isinstance(commands, list) and commands
        assert any("research_run" in command for command in commands)
        assert any("run_benchmarks" in command for command in commands)

    def test_snapshot_points_at_the_fresh_output_directory(self) -> None:
        assert _payload()["_snapshot"]["fresh_output"] == "benchmarks/artifacts/latest/"

    def test_fresh_output_directory_is_git_ignored(self) -> None:
        ignored = subprocess.run(
            ["git", "check-ignore", "benchmarks/artifacts/latest/results.json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert ignored.returncode == 0, ignored.stderr

    def test_marker_does_not_break_the_existing_consumers(self) -> None:
        payload = _payload()
        assert payload["environment"]["versions"]["cds2"]
        assert payload["results"]
        for row in payload["results"]:
            assert {
                "name",
                "baseline_library",
                "baseline_seconds",
                "cds2_seconds",
                "ratio_cds2_over_baseline",
            } <= set(row)

    def test_ratios_still_match_the_absolute_timings(self) -> None:
        # make_promo.py recomputes ratios from the absolute timings, so the
        # committed ratio column and the timings must stay consistent. The
        # writer rounds the ratio from the *unrounded* timings while storing
        # 6-decimal timings, so the two can differ by one unit in the last
        # place; anything larger means the table was hand-edited.
        for row in _payload()["results"]:
            expected = row["cds2_seconds"] / row["baseline_seconds"]
            assert row["ratio_cds2_over_baseline"] == pytest.approx(expected, abs=2e-3)


class TestSnapshotDocumentation:
    def test_docs_point_at_the_regeneration_path(self) -> None:
        text = (REPO_ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
        assert "benchmarks/artifacts/latest" in text
        assert "is_current_release" in text

    def test_docs_state_the_snapshot_is_not_current(self) -> None:
        text = (REPO_ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
        assert "not a claim about any current release" in text
        assert "is_current_release: false" in text


class TestSnapshotIsReadableByTheModule:
    def test_results_json_is_valid_json(self) -> None:
        assert sys.version_info >= (3, 10)
        assert isinstance(_payload(), dict)
