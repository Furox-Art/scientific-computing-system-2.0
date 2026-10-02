"""Release metadata consistency: version lockstep and CHANGELOG coverage."""

import json
import re
from pathlib import Path

import cds2

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "version not found in pyproject.toml"
    return match.group(1)


def _changelog_versions() -> list[str]:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    return re.findall(r"^## \[v([^\]]+)\]", text, re.MULTILINE)


def _package_json_version() -> str:
    data = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    version = data.get("version")
    assert isinstance(version, str), "version not found in package.json"
    return version


def _citation_cff_version() -> str:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'^version:\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "version not found in CITATION.cff"
    return match.group(1)


def test_package_version_matches_pyproject() -> None:
    assert cds2.__version__ == _pyproject_version()


def test_changelog_covers_current_version() -> None:
    assert _changelog_versions()[0] == _pyproject_version()


def test_npm_package_version_matches_pyproject() -> None:
    assert _package_json_version() == _pyproject_version()


def test_citation_version_matches_pyproject() -> None:
    assert _citation_cff_version() == _pyproject_version()
