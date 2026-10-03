"""Guard: release tooling must be pinned, and the pin must fit our metadata.

`twine check --strict` is the gate that decides whether a built artifact is
publishable, so its verdict must not depend on which version happened to resolve
on the day CI ran.

The specific hazard this pins down: twine <= 6.2.0 carries a frozen list of
valid ``Metadata-Version`` values that stops at 2.4, so it rejects a perfectly
valid ``Metadata-Version: 2.5`` artifact with

    InvalidDistribution: '2.5' is not a valid metadata version

even though ``packaging`` knows 2.5. twine 7.0.0 dropped that hardcoded list and
delegates to ``packaging.metadata.parse_email``, so it tracks ``packaging``
instead of lagging behind it.

Our build backend currently emits ``Metadata-Version: 2.4``, but both ``build``
and the ``setuptools>=77`` build requirement float, so the emitted version can
advance without any change in this repository. These checks therefore assert
three things:

1. ``build`` and ``twine`` are pinned with ``==`` in ``constraints/release-tools.txt``.
2. Every workflow that runs ``twine check`` installs tooling through that
   constraints file, never floating.
3. The pinned twine can validate the metadata version we emit, i.e. its ceiling
   is at least :data:`EMITTED_METADATA_VERSION`.

``EMITTED_METADATA_VERSION`` is asserted empirically by the ``package`` CI job,
which performs a real ``python -m build`` followed by
``python -m twine check --strict`` with these exact pins.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / "constraints" / "release-tools.txt"
WORKFLOWS = ROOT / ".github" / "workflows"

# Measured from a real `python -m build` of this repository (setuptools 84.0.0):
# both the wheel and the sdist declare Metadata-Version 2.4.
EMITTED_METADATA_VERSION = (2, 4)

# Highest Metadata-Version a given twine major can validate.
#   * < 7.0.0 hardcodes a list ending at 2.4 (twine/package.py).
#   * >= 7.0.0 delegates to packaging.metadata.parse_email and therefore tracks
#     whatever packaging knows, with no ceiling of its own.
TWINE_FROZEN_METADATA_CEILING = (2, 4)
PACKAGING_TRACKED_TWINE = (7, 0, 0)


def _parse_version(text: str) -> tuple[int, ...]:
    parts = re.split(r"[.\-+]", text.strip())
    out: list[int] = []
    for part in parts:
        if not part.isdigit():
            break
        out.append(int(part))
    return tuple(out)


def _pinned_versions() -> dict[str, tuple[int, ...]]:
    text = CONSTRAINTS.read_text(encoding="utf-8")
    pins: dict[str, tuple[int, ...]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([^\s]+)", line)
        assert match is not None, f"release tooling constraint is not pinned with '==': {line!r}"
        pins[match.group(1).lower()] = _parse_version(match.group(2))
    return pins


def _twine_metadata_ceiling(twine: tuple[int, ...]) -> tuple[int, ...] | None:
    """Highest Metadata-Version this twine can validate; None means unbounded."""
    if twine >= PACKAGING_TRACKED_TWINE:
        return None
    return TWINE_FROZEN_METADATA_CEILING


def test_release_tooling_constraints_exist_and_are_pinned() -> None:
    assert CONSTRAINTS.is_file(), f"missing {CONSTRAINTS.relative_to(ROOT)}"
    pins = _pinned_versions()
    for tool in ("build", "twine"):
        assert tool in pins, f"{tool} is not pinned in {CONSTRAINTS.relative_to(ROOT)}"
        assert pins[tool], f"{tool} pin did not parse into a version"


def test_pinned_twine_can_validate_the_emitted_metadata_version() -> None:
    twine = _pinned_versions()["twine"]
    ceiling = _twine_metadata_ceiling(twine)
    if ceiling is None:
        return  # tracks packaging, so it validates any version packaging knows
    assert EMITTED_METADATA_VERSION <= ceiling, (
        f"twine {'.'.join(map(str, twine))} can only validate Metadata-Version "
        f"{'.'.join(map(str, ceiling))}, but this build emits "
        f"{'.'.join(map(str, EMITTED_METADATA_VERSION))}. "
        "Pin twine >= 7.0.0, which delegates to packaging.metadata.parse_email."
    )


def test_pinned_twine_is_not_a_frozen_metadata_list() -> None:
    """Pin explicitly on a packaging-tracking twine, not merely a passing one.

    twine 6.2.0 would pass today's Metadata-Version 2.4 artifacts, so the ceiling
    check alone does not catch a downgrade. This asserts the pin directly.
    """
    twine = _pinned_versions()["twine"]
    assert twine >= PACKAGING_TRACKED_TWINE, (
        f"twine is pinned to {'.'.join(map(str, twine))}, which carries the frozen "
        "Metadata-Version list ending at 2.4. It will reject a valid 2.5 artifact. "
        "Pin twine >= 7.0.0."
    )


def test_every_twine_check_workflow_installs_pinned_tooling() -> None:
    offenders: list[str] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if "twine check" not in text:
            continue
        # Find the install step that provisions build/twine in this workflow.
        installs = [
            line
            for line in text.splitlines()
            if re.search(r"pip install", line) and re.search(r"\b(build|twine)\b", line)
        ]
        if not installs:
            offenders.append(f"{path.name}: runs `twine check` but never installs tooling")
            continue
        for line in installs:
            if "constraints/release-tools.txt" not in line:
                offenders.append(
                    f"{path.name}: installs build/twine without "
                    f"constraints/release-tools.txt -> {line.strip()!r}"
                )
    assert not offenders, "twine is not pinned where it is used:\n  " + "\n  ".join(offenders)


def test_twine_check_stays_strict() -> None:
    """`--strict` must not be dropped; it is what surfaces metadata errors."""
    checked = 0
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "twine check" not in line:
                continue
            checked += 1
            assert "--strict" in line, f"{path.name}: twine check lost --strict -> {line.strip()!r}"
    assert checked, "expected at least one `twine check` invocation to audit"
