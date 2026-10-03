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

## npm package provenance

The npm package `scientific-computing-system-2.0` is a Node launcher shim that
spawns `python -m cds2.cli`; it contains no Python and does not vendor the
scientific stack. Treat it as untrusted launcher code with one supply-chain
caveat worth stating plainly:

- **The published `5.2.5` has no Sigstore provenance attestation.** The
  registry's attestation endpoint returns `404` and the version document has no
  `attestations` field, because it was published through the long-lived
  automation-token path, which cannot mint an attestation. The `dist.signatures`
  values that are present are npm's own ECDSA registry signatures, not build
  provenance. Pin by integrity instead:
  `npm view scientific-computing-system-2.0@5.2.5 dist.integrity`.
- **A stale `2.0.0` remains on the registry.** It shipped a broken entry point
  and npm versions are immutable, so it cannot be repaired. Install `5.2.5` or
  later.
- **The library itself is the PyPI package.** The npm shim executes whatever
  `python` resolves to on your `PATH`; it grants no additional privilege beyond
  the Python interpreter you already trust.

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
