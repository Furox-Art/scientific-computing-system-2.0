# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 5.x | yes |
| 4.x | yes |
| < 4.0 | no - please upgrade |

The full list of published releases lives on the
[releases page](https://github.com/Furox-Art/scientific-computing-system-2.0/releases),
on [PyPI](https://pypi.org/project/scientific-computing-system-2.0/), and on
[npm](https://www.npmjs.com/package/scientific-computing-system-2.0).
Note: tag `v4.3.0` was never published to GitHub Releases or PyPI, so the
newest installable 4.x artifacts are `4.2.0`.

## Supply chain: no signed provenance today

**No published release of this project carries a provenance attestation, on
either registry.** Verified against the live endpoints:

- `https://pypi.org/integrity/scientific-computing-system-2.0/5.2.6/` → **404**
  (no PEP 740 attestation; also 404 project-wide and for 5.2.5)
- `https://registry.npmjs.org/-/npm/v1/attestations/scientific-computing-system-2.0@5.2.6` → **404**
  (no Sigstore attestation; also 404 for 5.2.5)

So do not describe any release as attested, verified or provenance-backed.

Three things are routinely conflated, so this policy states them apart:

- **Digests are integrity, not origin.** PyPI publishes `sha256` and
  `blake2b_256` per file; npm publishes `dist.integrity` (`sha512`). A digest
  detects tampering with the bytes. It does not identify the builder, so it is
  not provenance.
- **npm's `dist.signatures` are registry transport signatures, not build
  provenance.** They are signed by npm and prove the tarball came from npm. They
  say nothing about which source commit or build runner produced it. They must
  never be presented as provenance.
- **A PEP 740 or Sigstore attestation is the missing piece.** It binds the
  artifact digest to the workflow identity and source commit via a transparency
  log. Absent on both channels today.

What a consumer can do now: pin and verify digests
(`pip install --require-hashes`, and `npm view <pkg>@<version> dist.integrity`).

What the project does guarantee, enforced by CI on every push rather than by
assertion: a bit-reproducible pure-Python wheel (`reproducible-build` job pins
`SOURCE_DATE_EPOCH` and compares two builds' sha256), an sdist that genuinely
carries the native accelerators (`sdist-native` job builds a wheel from the
sdist and asserts both compiled modules are present and importable), version
lockstep across `package.json`, `src/cds2/_version.py` and `pyproject.toml`, and
cross-platform install-and-run smoke tests. These constrain the build; they do
not attest it.

**Fix path, pending not in place.** Provenance requires registry configuration,
not code changes: a PyPI Trusted Publisher (Owner `Furox-Art`, Repository
`scientific-computing-system-2.0`, Workflow `release.yml`, Environment `pypi`)
and an npmjs.com trusted publisher (Owner `Furox-Art`, Repository
`scientific-computing-system-2.0`, Workflow filename `npm-publish.yml`,
Environment `npm`). Both publish workflows are already structured for OIDC with
`id-token: write`, but both retain a token fallback, and a token upload cannot
mint an attestation.

Full detail, including how to reproduce every check:
[docs/supply-chain.md](https://furox-art.github.io/scientific-computing-system-2.0/supply-chain/).

## npm launcher shim

The npm package `scientific-computing-system-2.0` is a Node launcher shim that
spawns `python -m cds2.cli`; it contains no Python and does not vendor the
scientific stack. It executes whatever `python` resolves to on your `PATH`, so it
grants no privilege beyond the interpreter you already trust. Two npm versions
are unusable and cannot be repaired, because npm versions are immutable: the
stale `2.0.0`, which shipped a broken entry point before this repository took
over the name, and therefore install `5.2.5` or later.

## Reporting a vulnerability

Please report security issues privately via
[GitHub security advisories](https://github.com/Furox-Art/scientific-computing-system-2.0/security/advisories/new)
rather than public issues. You will receive a response within a week.

The attack surface is intentionally small: cds2 runs locally on NumPy arrays,
reads only the files you pass it, and makes no network calls. The one
exception to "no subprocesses" is benchmark provenance: `cds2.prof.history`
runs `git rev-parse --short HEAD` with a fixed argument vector (no shell) to
stamp the commit into benchmark history, and falls back to `"unknown"` when
git is unavailable.
