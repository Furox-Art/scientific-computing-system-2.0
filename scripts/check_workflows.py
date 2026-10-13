"""Workflow-graph audit: assert every publish path is covered.

Deliberately text-based rather than YAML-based, exactly like
``tools/consistency_audit.py``: PyYAML is not a declared dependency of this
repository, so an audit that imported it could not run in the CI job that
installs only ``.[dev]``.

The audit derives a graph from ``.github/workflows/*.yml``:

* which workflows publish, and to which registry (the
  ``pypa/gh-action-pypi-publish`` action -> pypi, an ``npm publish`` command
  -> npm);
* which workflows dispatch which, via ``gh workflow run <file>.yml``;
* which paths each trigger filters on.

Both publish markers are matched in executable positions only -- ``run:``
bodies for shell commands, ``uses:`` lines for actions -- because every
publish workflow also talks about "npm publish" in prose, and a step *named*
"Dispatch npm publish workflow" must not make the dispatcher look like a
publisher.

It then asserts the coverage contract documented in ``docs/release.md``:

1. the automatic version-bump path fans out to *every* publish workflow, so
   one ``src/cds2/_version.py`` bump reaches both registries and no registry
   is silently skipped;
2. no publish workflow is orphaned -- each one must be reachable from the
   automatic bump path, and the bump path must not dispatch anything that
   does not publish;
3. the bump trigger must actually watch the version file;
4. the CI workflow that guards publishing must not filter its triggers by
   path, so a change to a publish-path file still re-runs the guarding tests
   -- and it costs nothing extra, because no new job is added;
5. every publish workflow re-runs those guards itself, so publishing never
   depends on some other run having gone green first.

Run standalone (``python scripts/check_workflows.py``) or through the pytest
wrapper in ``tests/test_workflow_graph.py``, which the ``test`` and
``coverage`` CI jobs already collect.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"

# Turns a version bump into a release; the single automatic entry point.
BUMP_WORKFLOW = "release-on-version-bump.yml"
# The workflow whose job is to guard every change, publish changes included.
CI_WORKFLOW = "tests.yml"
# The file whose change on main must fan out to every publish workflow.
VERSION_FILE = "src/cds2/_version.py"

# Publish marker -> registry, so a new publish workflow joins the contract
# without anyone having to remember to register it.
PUBLISH_ACTIONS = {"pypa/gh-action-pypi-publish": "pypi"}
PUBLISH_COMMANDS = {"npm publish": "npm"}

# The guard each registry's publisher must run before uploading anything.
REGISTRY_GUARDS = {"pypi": "pytest", "npm": "npm test"}

DISPATCH_RE = re.compile(r"gh workflow run ([A-Za-z0-9_.\-]+\.yml)")
USES_RE = re.compile(r"^\s*uses:\s*([A-Za-z0-9_.\-/]+)@", re.MULTILINE)
RUN_RE = re.compile(r"^(\s*)(?:-\s*)?run:\s*(.*)$")


def _workflow_texts() -> dict[str, str]:
    """Every workflow file as text, keyed by file name."""
    paths = sorted(WORKFLOW_DIR.glob("*.yml"))
    return {path.name: path.read_text(encoding="utf-8") for path in paths}


def _dispatched(text: str) -> set[str]:
    """Workflow files this workflow starts with ``gh workflow run``."""
    return set(DISPATCH_RE.findall(text))


def _run_commands(text: str) -> list[str]:
    """Every shell command a workflow executes, block scalars included.

    Only ``run:`` bodies are returned. A step *named* "Dispatch npm publish
    workflow on main" is a ``name:``, not a ``run:``, so it is never mistaken
    for an actual publish.
    """
    lines = text.splitlines()
    commands: list[str] = []
    index = 0
    while index < len(lines):
        match = RUN_RE.match(lines[index])
        if match is None:
            index += 1
            continue
        indent, inline = match.group(1), match.group(2).strip()
        index += 1
        if inline.startswith("|") or inline.startswith(">"):
            folded = inline.lstrip("|>-")
            if folded.strip():
                commands.append(folded.strip())
            while index < len(lines):
                line = lines[index]
                if line.strip() and not line.startswith(f"{indent} "):
                    break
                if line.strip():
                    commands.append(line.strip())
                index += 1
        elif inline:
            commands.append(inline)
    return commands


def _uses_actions(text: str) -> set[str]:
    """Actions referenced by ``uses:`` lines, version pin stripped."""
    return set(USES_RE.findall(text))


def _registries(text: str) -> set[str]:
    """Registries this workflow actually publishes to."""
    registries = {
        registry for action, registry in PUBLISH_ACTIONS.items() if action in _uses_actions(text)
    }
    for command in _run_commands(text):
        words = command.split()
        for marker, registry in PUBLISH_COMMANDS.items():
            if words[: len(marker.split())] == marker.split():
                registries.add(registry)
    return registries


def _runs_guard(text: str, guard: str) -> bool:
    """True when the workflow executes *guard* as a command."""
    wanted = guard.split()
    return any(command.split()[: len(wanted)] == wanted for command in _run_commands(text))


def _block(lines: list[str], header: str, indent: str) -> list[str]:
    """Body of the block introduced by ``indent + header + ':'``.

    Indentation-based, mirroring ``tools/consistency_audit._iter_jobs``.
    """
    prefix = f"{indent}{header}:"
    start = -1
    for index, line in enumerate(lines):
        if line == prefix:
            start = index + 1
            break
    if start < 0:
        return []
    body: list[str] = []
    deeper = f"{indent} "
    for line in lines[start:]:
        if line.strip() and not line.startswith(deeper):
            break
        body.append(line)
    return body


def _on_lines(text: str) -> list[str]:
    """Lines of the top-level ``on:`` block (empty when there is none)."""
    return _block(text.splitlines(), "on", "")


def _path_filters(text: str, event: str) -> list[str]:
    """Path filters declared for ``on.<event>`` (empty when unfiltered)."""
    event_block = _block(_on_lines(text), event, "  ")
    filters: list[str] = []
    for key in ("paths", "paths-ignore"):
        for line in _block(event_block, key, "    "):
            entry = line.strip()
            if entry.startswith("- "):
                filters.append(entry[2:].strip().strip("\"'"))
    return filters


def coverage_problems() -> list[str]:
    """Violations of the publish-path coverage contract; empty when sound."""
    problems: list[str] = []
    workflows = _workflow_texts()

    publishes = {
        name: sorted(_registries(text)) for name, text in workflows.items() if _registries(text)
    }
    if not publishes:
        return ["no publish workflow found: the audit would silently cover nothing"]

    if BUMP_WORKFLOW not in workflows:
        return [f"{BUMP_WORKFLOW} is missing: a version bump would never publish"]

    dispatched = _dispatched(workflows[BUMP_WORKFLOW])

    # 1 + 2. One bump reaches every registry, and no publisher is orphaned.
    for name in sorted(publishes):
        if name not in dispatched:
            problems.append(
                f"{BUMP_WORKFLOW} does not dispatch {name}: a version bump would "
                f"silently skip publishing to {', '.join(publishes[name])}"
            )
    for name in sorted(dispatched):
        if name not in publishes:
            problems.append(f"{BUMP_WORKFLOW} dispatches {name}, which publishes nothing")

    # 3. Without the version file in the trigger paths the fan-out never fires.
    if VERSION_FILE not in _path_filters(workflows[BUMP_WORKFLOW], "push"):
        problems.append(f"{BUMP_WORKFLOW} does not trigger on {VERSION_FILE}")

    if CI_WORKFLOW not in workflows:
        problems.append(f"{CI_WORKFLOW} is missing: nothing guards the publish path")
        return problems

    # 4. A path filter here is how publish-path changes stop being tested.
    for event in ("push", "pull_request"):
        filters = _path_filters(workflows[CI_WORKFLOW], event)
        if filters:
            problems.append(
                f"{CI_WORKFLOW} filters on.{event} by path ({', '.join(filters)}): a "
                "publish-path change would skip the tests that guard publishing"
            )

    # 5. Publishers re-run their own guards rather than trusting another run.
    for name in sorted(publishes):
        if name == CI_WORKFLOW:
            continue
        text = workflows[name]
        for registry in publishes[name]:
            guard = REGISTRY_GUARDS[registry]
            if not _runs_guard(text, guard):
                problems.append(
                    f"{name} publishes to {registry} without running its guard "
                    f"({guard}) first"
                )

    return problems


def main() -> int:
    """Print the audit result; non-zero when the graph is not sound."""
    problems = coverage_problems()
    if problems:
        print("WORKFLOW COVERAGE PROBLEMS:")
        for problem in problems:
            print(f" - {problem}")
        return 1
    print("all publish paths are covered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
