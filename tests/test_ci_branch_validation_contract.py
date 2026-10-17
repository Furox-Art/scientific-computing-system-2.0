"""CI may validate current development work without modifying protected main."""

from pathlib import Path

CI_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tests.yml"


def test_current_development_branch_triggers_full_ci_on_push() -> None:
    document = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "branches: [main, dev/parallel-hardening-20261016]" in document


def test_ci_can_be_dispatched_and_reused_without_a_pull_request() -> None:
    document = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "\n  workflow_dispatch:\n" in document
    assert "\n  workflow_call:\n" in document
    assert "\npermissions:\n  contents: read\n" in document
