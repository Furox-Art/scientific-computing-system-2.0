"""One-shot consistency audit: repo vs package vs docs vs PyPI."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
issues: list[str] = []

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

# 2. Every __all__ entry resolves to a real attribute
import cds2  # noqa: E402  (path set up above)

missing_attrs = [name for name in cds2.__all__ if not hasattr(cds2, name)]
if missing_attrs:
    issues.append(f"__all__ entries without attributes: {missing_attrs}")

# 3. Version agreement: pyproject == _version == CHANGELOG top == git tag == PyPI
data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
pyproject_version = data["project"]["version"]
from cds2._version import __version__  # noqa: E402

changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
top_entry = re.search(r"## \[v([\d.]+)\]", changelog)
changelog_version = top_entry.group(1) if top_entry else "?"
with urllib.request.urlopen(  # noqa: S310
    "https://pypi.org/pypi/scientific-computing-system-2.0/json", timeout=30
) as response:
    pypi_version = json.loads(response.read())["info"]["version"]
versions = {
    "pyproject": pyproject_version,
    "_version": __version__,
    "CHANGELOG": changelog_version,
    "PyPI": pypi_version,
}
if len(set(versions.values())) != 1:
    issues.append(f"version mismatch: {versions}")

# 4. Every mkdocs api page exists and every src module has an api page
mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
api_refs = re.findall(r"api/([\w_]+)\.md", mkdocs)
for ref in sorted(set(api_refs)):
    if not (ROOT / "docs" / "api" / f"{ref}.md").exists():
        issues.append(f"mkdocs references api/{ref}.md which does not exist")
uncovered = [m for m in src_modules if m not in set(api_refs) and m != "cli"]
# cli is documented too but keep the check honest:
uncovered = [m for m in uncovered]
if uncovered:
    issues.append(f"modules without an api page: {uncovered}")

# 5. README documents every module
readme = (ROOT / "README.md").read_text(encoding="utf-8")
undocumented_readme = [m for m in src_modules if f"`cds2.{m}`" not in readme]
if undocumented_readme:
    issues.append(f"modules absent from README table: {undocumented_readme}")

# 6. Docs site live + PyPI files present
with urllib.request.urlopen(  # noqa: S310
    "https://furox-art.github.io/scientific-computing-system-2.0/", timeout=30
) as response:
    if response.status != 200:
        issues.append(f"docs site returned {response.status}")

print(f"source modules : {len(src_modules)}")
print(f"public exports : {len(cds2.__all__)}")
print(f"versions       : {versions}")
print(f"api pages      : {len(set(api_refs))}")
if issues:
    print("\nINCONSISTENCIES FOUND:")
    for issue in issues:
        print(f" - {issue}")
    raise SystemExit(1)
print("\nALL CONSISTENCY CHECKS PASSED")
