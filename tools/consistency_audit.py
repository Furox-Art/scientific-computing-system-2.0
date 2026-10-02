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
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

issues: list[str] = []
warnings: list[str] = []

# 1. Every source module wired into __init__
src_modules = sorted(
    p.stem for p in (ROOT / "src" / "cds2").glob("*.py") if p.stem not in {"__init__", "_version"}
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

print(f"source modules : {len(src_modules)}")
print(f"public exports : {len(cds2.__all__)}")
print(f"versions       : {versions}")
print(f"api pages      : {len(set(api_refs))}")

for warning in warnings:
    print(f" - WARNING: {warning}")

if issues:
    print("\nINCONSISTENCIES FOUND:")
    for issue in issues:
        print(f" - {issue}")
    raise SystemExit(1)

print("\nALL CONSISTENCY CHECKS PASSED")
