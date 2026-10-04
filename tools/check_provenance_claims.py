"""Guard against unverified provenance claims in the documentation.

Why this exists
---------------
The docs previously described only npm's missing attestation, which implied the
PyPI side was attested. Measurement showed otherwise: **neither** registry serves
a provenance attestation for any published release, so any affirmative
"attested"/"signed provenance" wording was false.

Two failure modes are guarded, in both directions:

1. **False affirmative.** A doc claims an attestation exists (or implies one)
   without a live check backing it. Rejected.
2. **Stale negative.** A doc correctly says "no attestation today", but the
   registry later starts serving one (someone registers the trusted publisher).
   The negative claim is then itself wrong. Caught by probing the live endpoints
   and failing when they stop returning 404.

Usage
-----
    python tools/check_provenance_claims.py            # text + live probe
    python tools/check_provenance_claims.py --offline  # text only

Exit codes: 0 clean, 1 a claim problem was found.

Not wired into CI yet: adding it requires a one-line step in an existing
workflow, which is outside the documentation file ownership this change was made
under. Wire it as::

    - name: Provenance claim guard
      run: python tools/check_provenance_claims.py
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Documentation this guard is allowed to police.
#
# CHANGELOG.md is listed for existence checking but deliberately excluded from the
# affirmative scan: it is a historical record written in the past tense, so it
# necessarily *describes* claims that were once wrong ("which implied the PyPI
# side was attested"). Scanning it would flag the very corrections this guard
# exists to encourage. Current-state truth lives in the files below.
SCANNED_DOCS = (
    "README.md",
    "SECURITY.md",
    "docs/npm.md",
    "docs/supply-chain.md",
    "docs/release.md",
)

WATCHED_DOCS = SCANNED_DOCS + ("CHANGELOG.md",)

PACKAGE = "scientific-computing-system-2.0"

# Endpoints that return 404 while no attestation exists.
ATTESTATION_ENDPOINTS = (
    f"https://pypi.org/integrity/{PACKAGE}/",
    f"https://pypi.org/integrity/{PACKAGE}/5.2.6/",
    f"https://registry.npmjs.org/-/npm/v1/attestations/{PACKAGE}@5.2.6",
)

# Phrases that assert or imply a published artifact carries provenance. Each must
# be negated in context ("no attestation", "not attested", "cannot mint"). The
# guard matches the affirmative form only.
FORBIDDEN_AFFIRMATIVE = (
    r"(?<!no )(?<!not )\bis attested\b",
    r"(?<!no )(?<!not )\bare attested\b",
    r"(?<!no )(?<!not )\bwas attested\b",
    r"(?<!no )(?<!not )\bprovenance[- ]attested\b",
    r"(?<!no )(?<!not )\bsigned provenance\b",
    r"(?<!no )(?<!not )\bwith provenance\b",
    r"(?<!no )(?<!not )\bcarries an attestation\b",
    r"(?<!no )(?<!not )\bcarries a Sigstore attestation\b",
    r"(?<!no )(?<!not )\bverified provenance\b",
    r"\bPEP 740 attestation (?:is|was) (?:present|available)\b",
)

# The honest negative must be stated in these files, and it is checked
# semantically rather than by exact phrasing: the file must discuss
# "attestation" and must negate it somewhere ("no ... attestation",
# "not ... attested", "no signed provenance"). Keying on a fixed sentence was
# tried first and proved fragile, because an unrelated heading could satisfy it
# while the actual claim was removed.
REQUIRED_NEGATIVE_FILES = ("SECURITY.md", "docs/supply-chain.md", "README.md")

NEGATED_ATTESTATION = re.compile(
    r"(?:no|not|never|without|cannot|can't)\b[^.]{0,80}?\b(?:attestation|attested|provenance)\b",
    re.IGNORECASE,
)


def _negated_context(text: str, match: re.Match[str]) -> bool:
    """True when a preceding negation governs the match.

    Looks back over the same sentence for a negator, so "no attestation" and
    "do not treat it as an attested build" are not flagged while "is attested"
    on its own is.
    """
    start = text.rfind(".", 0, match.start()) + 1
    window = text[start : match.start()].lower()
    return any(
        token in window
        for token in (
            "no ",
            "not ",
            "never",
            "without",
            "cannot",
            "can't",
            "do not",
            "doesn't",
            "absent",
            "returns `404`",
            "pending",
        )
    )


# A quoted span is a citation, not the document's own assertion. CHANGELOG.md has
# to be able to quote the phrases this guard forbids (that is how a regression
# gets described), so inline code and quoted text are blanked before scanning.
QUOTED_SPANS = (
    re.compile(r"`[^`]*`"),
    re.compile(r"\"[^\"\n]{0,120}\""),
    re.compile(r"'[^'\n]{0,120}'"),
)


def _mask_quoted(text: str) -> str:
    """Replace quoted spans with spaces, preserving offsets and line numbers."""
    masked = text
    for pattern in QUOTED_SPANS:
        masked = pattern.sub(lambda m: " " * len(m.group(0)), masked)
    return masked


def scan_texts() -> list[str]:
    """Return one problem string per unverified affirmative provenance claim."""
    problems: list[str] = []
    for rel in WATCHED_DOCS:
        if not (ROOT / rel).exists():
            problems.append(f"{rel}: file listed in the guard is missing")
    for rel in SCANNED_DOCS:
        path = ROOT / rel
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8")
        text = _mask_quoted(raw)
        for pattern in FORBIDDEN_AFFIRMATIVE:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if _negated_context(text, match):
                    continue
                line = raw[: match.start()].count("\n") + 1
                snippet = raw[max(0, match.start() - 40) : match.end() + 40].replace("\n", " ")
                problems.append(
                    f"{rel}:{line}: affirmative provenance claim not backed by a "
                    f"live check -> ...{snippet}..."
                )
    for rel in REQUIRED_NEGATIVE_FILES:
        path = ROOT / rel
        if not path.exists():
            problems.append(f"{rel}: required statement of the negative is missing")
            continue
        text = _mask_quoted(path.read_text(encoding="utf-8"))
        if "attestation" not in text.lower() and "attested" not in text.lower():
            problems.append(f"{rel}: must discuss whether an attestation exists")
        if not NEGATED_ATTESTATION.search(text):
            problems.append(
                f"{rel}: must state plainly that no published release carries an "
                "attestation on either registry"
            )
    return problems


def probe(url: str, timeout: float = 20.0) -> int | None:
    """Return the HTTP status for ``url``, or None when unreachable."""
    request = urllib.request.Request(url, headers={"user-agent": "provenance-claim-guard"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def scan_live() -> tuple[list[str], list[str]]:
    """Cross-check the documented negative against the live registries.

    Returns ``(problems, notes)``. Unreachable endpoints are reported as notes,
    never as failures, so an offline run cannot produce a false alarm.
    """
    problems: list[str] = []
    notes: list[str] = []
    for url in ATTESTATION_ENDPOINTS:
        status = probe(url)
        if status is None:
            notes.append(f"skipped (unreachable): {url}")
            continue
        if status == 404:
            notes.append(f"404 as documented: {url}")
        elif status == 200:
            problems.append(
                f"{url} now returns 200: an attestation EXISTS. The documented "
                "no-attestation statement is stale and must be updated."
            )
        else:
            notes.append(f"unexpected status {status} (not treated as failure): {url}")
    return problems, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the live registry probe (text checks only)",
    )
    args = parser.parse_args(argv)

    problems = scan_texts()
    notes: list[str] = []

    if args.offline:
        notes.append("live probe skipped (--offline)")
    else:
        live_problems, notes = scan_live()
        problems.extend(live_problems)

    for note in notes:
        print(f"note: {note}")

    if problems:
        print(f"\n{len(problems)} provenance claim problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("\nprovenance claims OK: no unverified attestations asserted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
