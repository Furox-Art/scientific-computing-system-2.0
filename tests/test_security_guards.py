"""Security guardrails: fail closed if dangerous primitives ever enter src/cds2.

The shipped package runs locally on user data. These tests pin the audit
result that no dynamic code execution, unsafe deserialisation or shell
invocation exists anywhere under ``src/cds2`` (Python modules and the C
accelerator sources alike).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "cds2"
SCANNED_SUFFIXES = {".py", ".c"}

BANNED_PATTERNS = {
    r"\beval\s*\(": "eval() executes arbitrary code",
    r"\bexec\s*\(": "exec() executes arbitrary code",
    r"\bpickle\b": "pickle deserialisation executes arbitrary code on load",
    r"\byaml\.load\s*\(": "yaml.load() without SafeLoader executes arbitrary code",
    r"\bshell\s*=\s*True": "shell=True enables shell-injection via PATH/arguments",
}


def _scan() -> list[tuple[str, int, str, str]]:
    hits: list[tuple[str, int, str, str]] = []
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern, reason in BANNED_PATTERNS.items():
                if re.search(pattern, line):
                    hits.append((str(path.relative_to(ROOT)), lineno, pattern, reason))
    return hits


def _scanned_files() -> list[Path]:
    return [p for p in SRC.rglob("*") if p.suffix in SCANNED_SUFFIXES and p.is_file()]


def test_guard_scan_covers_all_shipped_sources() -> None:
    files = _scanned_files()
    assert len(files) > 50
    names = {path.name for path in files}
    assert {"cli.py", "io.py", "guided_fit.py", "_fast_pagerank.c", "_fast_kmeans.c"} <= names


def test_no_banned_execution_primitives_in_shipped_sources() -> None:
    hits = _scan()
    assert not hits, f"banned patterns present in src/cds2: {hits!r}"
