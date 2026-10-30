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

## Supply chain: PyPI is attested, and npm from 5.4.5

Verified against the live registries on 2026-10-22; npm re-checked on 2026-10-30:

- **PyPI: attested.** Every file of every release from `5.1.0` through `5.4.5`
  (330 files) carries a PEP 740 provenance attestation. The per-file endpoint
  `https://pypi.org/integrity/scientific-computing-system-2.0/<version>/<file>/provenance`
  returns `200`. The project-wide URL
  `https://pypi.org/integrity/scientific-computing-system-2.0/` returns `404`
  even when per-file attestations exist, so it is not the endpoint to check.
- **npm: attested from `5.4.5`.** `5.4.5`, published on 2026-10-23, is the first
  npm version whose attestation endpoint
  `https://registry.npmjs.org/-/npm/v1/attestations/scientific-computing-system-2.0@5.4.5`
  returns `200` (SLSA provenance v1). Every earlier npm version, including
  `5.4.4`, returns `404` and has no `attestations` field; those uploads used
  `NPM_TOKEN`, and a token upload cannot mint a Sigstore attestation.

So `5.4.5` and later npm releases may be described as attested; never describe
an npm release before `5.4.5` as attested, verified or provenance-backed.

Three things are routinely conflated, so this policy states them apart:

- **Digests are integrity, not origin.** PyPI publishes `sha256` and
  `blake2b_256` per file; npm publishes `dist.integrity` (`sha512`). A digest
  detects tampering with the bytes. It does not identify the builder, so it is
  not provenance.
- **npm's `dist.signatures` are registry transport signatures, not build
  provenance.** They are signed by npm and prove the tarball came from npm. They
  say nothing about which source commit or build runner produced it. They must
  never be presented as provenance.
- **A PEP 740 or Sigstore attestation binds the artifact digest to the workflow
  identity and source commit via a transparency log.** Present on PyPI since
  `5.1.0` and on npm since `5.4.5`.

What a consumer can do now: fetch the PyPI provenance bundle for the exact file
you install, and pin and verify digests (`pip install --require-hashes`, and
`npm view <pkg>@<version> dist.integrity`).

What the project does guarantee, enforced by CI on every push rather than by
assertion: a bit-reproducible pure-Python wheel (`reproducible-build` job pins
`SOURCE_DATE_EPOCH` and compares two builds' sha256), an sdist that genuinely
carries the native accelerators (`sdist-native` job builds a wheel from the
sdist and asserts both compiled modules are present and importable), version
lockstep across `package.json`, `src/cds2/_version.py`, `pyproject.toml`,
`CITATION.cff`, `codemeta.json` and the conda recipe, and cross-platform
install-and-run smoke tests. These constrain the build; they do not replace the
PEP 740 attestation, which is now present for PyPI.

**Fix path.** PyPI Trusted Publishing is registered, and the `5.1.0`+ releases
were uploaded through it, which is why their attestations exist. The npm
trusted publisher (Owner `Furox-Art`, Repository
`scientific-computing-system-2.0`, Workflow filename `npm-publish.yml`,
Environment `npm`) is registered as well, and `5.4.5` was the first release
published through it; the npm uploads before `5.4.5` used `NPM_TOKEN` and carry
no attestation, which cannot be changed after the fact.

Full detail, including how to reproduce every check:
[docs/supply-chain.md](https://furox-art.github.io/scientific-computing-system-2.0/supply-chain/).

## npm launcher shim

The npm package `scientific-computing-system-2.0` is a Node launcher shim that
spawns `python -m cds2.cli`; it contains no Python and does not vendor the
scientific stack. It executes whatever `python` resolves to on your `PATH`, so it
grants no privilege beyond the interpreter you already trust. One npm version
is unusable and cannot be repaired, because npm versions are immutable: the
stale `2.0.0`, which shipped a broken entry point before this repository took
over the name. Install `5.2.5` or later.

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
