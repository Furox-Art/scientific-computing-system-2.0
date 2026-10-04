"""Consistency audit: repo vs package vs docs vs PyPI.

Runs as a required CI gate. Checks are split by dependency so a transient
network failure can never block a merge:

* Offline structural checks (repo, package, docs tree) are fatal. They are
  deterministic and must hold on every commit.
* Remote checks (PyPI, deployed docs site) are reported as warnings. They are
  advisory because the repository is intentionally *ahead* of PyPI between a
  version bump and its publication, and because an external outage is not a
  defect in this repository.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

issues: list[str] = []
warnings: list[str] = []


def _iter_jobs(workflow_text: str):
    """Yield (job_name, job_body) for each top-level job in a workflow.

    Deliberately indentation-based rather than YAML-based: the caller only needs
    the job's own text, and this keeps the audit importable without PyYAML, which
    is not a declared dependency of the repository.
    """
    lines = workflow_text.splitlines()
    starts = [
        (index, line.split(":", 1)[0].strip())
        for index, line in enumerate(lines)
        if line.startswith("  ") and not line.startswith("   ") and line.rstrip().endswith(":")
    ]
    for position, (index, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        yield name, "\n".join(lines[index:end])


# 1. Every source module wired into __init__
#
# `__init__`, `_version` and `__main__` are excluded from the public module
# inventory. `__main__` is the `python -m cds2` shim, not an API surface: it has
# no public functions to document, and importing it from `__init__` would execute
# the CLI (it raises SystemExit) on a bare `import cds2`. Requiring it to be
# bound in `__init__` would be actively harmful.
_NON_API_MODULES = {"__init__", "_version", "__main__"}
src_modules = sorted(
    p.stem for p in (ROOT / "src" / "cds2").glob("*.py") if p.stem not in _NON_API_MODULES
)
init_text = (ROOT / "src" / "cds2" / "__init__.py").read_text(encoding="utf-8")
block = init_text[
    init_text.index("from . import (") : init_text.index(")", init_text.index("from . import ("))
]
for mod in src_modules:
    if f"    {mod}," not in block:
        issues.append(f"module {mod} missing from __init__ from-import block")
    if f'"{mod}",' not in init_text:
        issues.append(f"module {mod} missing from __init__ __all__")

# 2. Every __all__ entry resolves to a real attribute
import cds2  # noqa: E402  (path set up above)

missing_attrs = [name for name in cds2.__all__ if not hasattr(cds2, name)]
if missing_attrs:
    issues.append(f"__all__ entries without attributes: {missing_attrs}")

# 3. Version lockstep inside the repository (offline, fatal)
data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
pyproject_version = data["project"]["version"]
from cds2._version import __version__  # noqa: E402

changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
top_entry = re.search(r"## \[v([\d.]+)\]", changelog)
changelog_version = top_entry.group(1) if top_entry else "?"
versions = {
    "pyproject": pyproject_version,
    "_version": __version__,
    "CHANGELOG": changelog_version,
}
if len(set(versions.values())) != 1:
    issues.append(f"version mismatch: {versions}")

# The npm package version must track the Python release version.
package_json = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
if package_json.get("version") != pyproject_version:
    issues.append(
        f"package.json version {package_json.get('version')!r} "
        f"does not match pyproject.toml {pyproject_version!r}"
    )

# 4. Every mkdocs api page exists and every src module has an api page
mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
api_refs = re.findall(r"api/([\w_]+)\.md", mkdocs)
for ref in sorted(set(api_refs)):
    if not (ROOT / "docs" / "api" / f"{ref}.md").exists():
        issues.append(f"mkdocs references api/{ref}.md which does not exist")
uncovered = [m for m in src_modules if m not in set(api_refs)]
if uncovered:
    issues.append(f"modules without an api page: {uncovered}")

# 5. README documents every module. Accepts either a backticked `cds2.<module>`
# reference or a markdown link to docs/api/<module>.md, so the check does not
# depend on the README using one particular table layout.
readme = (ROOT / "README.md").read_text(encoding="utf-8")
undocumented_readme = [
    m for m in src_modules if f"`cds2.{m}`" not in readme and f"api/{m}.md" not in readme
]
if undocumented_readme:
    issues.append(f"modules absent from README: {undocumented_readme}")


def _fetch(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "cds2-consistency-audit"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


# 6. PyPI agreement (remote, advisory). The repository legitimately runs ahead
# of PyPI between a version bump and its publication: `release-on-version-bump`
# only fires after the bump merges, and publishing then takes minutes. Treating
# "repo ahead of PyPI" as an error would deadlock every future release, so only a
# repo that has fallen *behind* a published release is flagged.
try:
    pypi_version = json.loads(_fetch("https://pypi.org/pypi/scientific-computing-system-2.0/json"))[
        "info"
    ]["version"]
    if pypi_version != pyproject_version:
        message = f"PyPI version {pypi_version} != repository version {pyproject_version}"
        if tuple(map(int, pypi_version.split("."))) > tuple(map(int, pyproject_version.split("."))):
            issues.append(f"{message} (repository is behind a published release)")
        else:
            warnings.append(f"{message} (expected until the pending release is published)")
except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
    warnings.append(f"could not read PyPI metadata ({type(exc).__name__}: {exc})")

# 7. Docs site live (remote, advisory)
try:
    _fetch("https://furox-art.github.io/scientific-computing-system-2.0/")
except (urllib.error.URLError, OSError) as exc:
    warnings.append(f"could not reach the deployed docs site ({type(exc).__name__}: {exc})")

# 8. GitHub Action pins.
#
# Every `uses:` must be a full 40-character commit SHA, and that SHA must
# actually exist upstream. This is not theoretical: release.yml shipped pins for
# `pypa/cibuildwheel` and `softprops/action-gh-release` that no upstream commit
# matched. GitHub Actions resolves those at job start, so every run that reached
# the step failed before the job body executed -- and because the only affected
# steps are in the release path, the breakage was invisible until a release was
# attempted.
#
# The upstream existence check is fatal only when GitHub definitively reports the
# commit as missing. Transport errors and rate limiting produce a warning, so a
# network blip cannot fail a build that is actually fine.
WORKFLOW_DIR = ROOT / ".github" / "workflows"
USES_RE = re.compile(r"uses:\s*([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)@([^\s#]+)")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# Actions that mutate the repository through the REST/git API and therefore
# require `contents: write` on the calling job. A release job granted only
# `contents: read` publishes successfully and then fails, which is a confusing
# failure mode worth ruling out structurally.
REPO_WRITER_ACTIONS = ("softprops/action-gh-release",)

action_pins: dict[tuple[str, str], set[str]] = {}
unpinned: list[str] = []
for path in sorted(WORKFLOW_DIR.glob("*.yml")):
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = USES_RE.search(line)
        if not match:
            continue
        action, ref = match.groups()
        if not SHA_RE.match(ref):
            unpinned.append(f"{path.name}:{lineno} {action}@{ref} is not a full commit SHA")
            continue
        action_pins.setdefault((action, ref), set()).add(path.name)

for entry in unpinned:
    issues.append(f"action is not pinned to a full commit SHA -> {entry}")


def _action_commit_exists(action: str, sha: str) -> bool | None:
    """True/False if GitHub answered, None if the check could not be made."""
    url = f"https://api.github.com/repos/{action}/commits/{sha}"
    headers = {"User-Agent": "cds2-consistency-audit", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return 200 <= response.status < 300
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                # Rate limited or forbidden: we do not know the answer.
                last_error = exc
                time.sleep(2**attempt)
                continue
            # 404 / 422 are GitHub stating the commit does not exist.
            return False
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    warnings.append(f"could not verify {action}@{sha[:12]} ({type(last_error).__name__})")
    return None


verified_actions = 0
for (action, sha), files in sorted(action_pins.items()):
    exists = _action_commit_exists(action, sha)
    if exists is None:
        continue
    verified_actions += 1
    if not exists:
        issues.append(
            f"action pin does not exist upstream: {action}@{sha} "
            f"(used in {', '.join(sorted(files))})"
        )

# Release-writing actions must sit in a job with contents: write.
for path in sorted(WORKFLOW_DIR.glob("*.yml")):
    text = path.read_text(encoding="utf-8")
    for action in REPO_WRITER_ACTIONS:
        if f"{action}@" not in text:
            continue
        for job_name, body in _iter_jobs(text):
            if f"{action}@" in body and "contents: write" not in body:
                issues.append(
                    f"{path.name}: job '{job_name}' uses {action} but does not grant "
                    "contents: write"
                )

print(f"source modules : {len(src_modules)}")
print(f"public exports : {len(cds2.__all__)}")
print(f"versions       : {versions}")
print(f"api pages      : {len(set(api_refs))}")
print(f"action pins    : {len(action_pins)} checked, {verified_actions} verified upstream")

for warning in warnings:
    print(f" - WARNING: {warning}")

if issues:
    print("\nINCONSISTENCIES FOUND:")
    for issue in issues:
        print(f" - {issue}")
    raise SystemExit(1)

print("\nALL CONSISTENCY CHECKS PASSED")
