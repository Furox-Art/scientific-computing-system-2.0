"""Workflow-graph coverage tests.

Publishing is driven by two registries with two entry points (the automatic
``src/cds2/_version.py`` bump and manual ``workflow_dispatch``), and the gap
between them is invisible until a release is actually attempted. These tests
assert the contract in ``scripts/check_workflows.py`` against the live
workflow graph, so the next divergence fails in CI rather than at publish time.

They also lock the CI side of that contract: the workflow that guards
publishing must not filter its triggers by path, or a change to a publish-path
file would stop re-running the tests that guard it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_workflows  # noqa: E402


class TestWorkflowGraph:
    def test_the_audit_reports_no_coverage_problems(self) -> None:
        assert check_workflows.coverage_problems() == []

    def test_the_version_bump_dispatches_every_publish_workflow(self) -> None:
        workflows = check_workflows._workflow_texts()
        publishes = {
            name
            for name, text in workflows.items()
            if check_workflows._registries(text)
        }
        dispatched = check_workflows._dispatched(workflows[check_workflows.BUMP_WORKFLOW])
        assert publishes, "no publish workflow found"
        assert publishes <= dispatched

    def test_every_publish_workflow_runs_its_own_guard(self) -> None:
        workflows = check_workflows._workflow_texts()
        for name, text in workflows.items():
            if name == check_workflows.CI_WORKFLOW:
                continue
            for registry in check_workflows._registries(text):
                guard = check_workflows.REGISTRY_GUARDS[registry]
                assert check_workflows._runs_guard(text, guard), (
                    f"{name} publishes to {registry} without running {guard}"
                )

    def test_ci_does_not_filter_publish_path_changes_away(self) -> None:
        text = check_workflows._workflow_texts()[check_workflows.CI_WORKFLOW]
        for event in ("push", "pull_request"):
            assert check_workflows._path_filters(text, event) == [], (
                f"{check_workflows.CI_WORKFLOW} filters on.{event} by path"
            )


class TestWorkflowGraphParsers:
    """The parsers must not be fooled by prose that talks about publishing."""

    def test_run_commands_include_block_scalars(self) -> None:
        text = (
            "jobs:\n"
            "  a:\n"
            "    steps:\n"
            "      - run: |\n"
            "          set -euo pipefail\n"
            "          npm test\n"
            "      - run: echo inline\n"
        )
        assert check_workflows._run_commands(text) == [
            "set -euo pipefail",
            "npm test",
            "echo inline",
        ]

    def test_a_step_named_after_a_publish_is_not_a_publish(self) -> None:
        text = "      - name: Dispatch npm publish workflow on main\n"
        assert check_workflows._registries(text) == set()

    def test_uses_lines_are_publish_markers(self) -> None:
        pin = "dc37677b2e1c63e2034f94d8a5b11f265b73ba33"
        text = f"        uses: pypa/gh-action-pypi-publish@{pin}\n"
        assert check_workflows._registries(text) == {"pypi"}

    def test_path_filters_read_nested_lists(self) -> None:
        text = (
            "on:\n"
            "  push:\n"
            "    branches: [main]\n"
            "    paths:\n"
            '      - "src/cds2/_version.py"\n'
        )
        assert check_workflows._path_filters(text, "push") == ["src/cds2/_version.py"]
        assert check_workflows._path_filters(text, "pull_request") == []

    def test_runs_guard_matches_command_prefixes(self) -> None:
        text = "      - run: pytest --cov=cds2 --cov-fail-under=100\n"
        assert check_workflows._runs_guard(text, "pytest")
        assert not check_workflows._runs_guard(text, "npm test")


class TestWorkflowAuditScript:
    def test_script_entry_point_exits_zero_on_a_sound_graph(self) -> None:
        assert check_workflows.main() == 0
