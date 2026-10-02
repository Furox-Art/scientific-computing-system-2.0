# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 5.x | yes |
| 4.x | yes |
| < 4.0 | no - please upgrade |

The full list of published releases lives on the
[releases page](https://github.com/Furox-Art/scientific-computing-system-2.0/releases)
and on [PyPI](https://pypi.org/project/scientific-computing-system-2.0/).
Note: tag `v4.3.0` was never published to GitHub Releases or PyPI, so the
newest installable 4.x artifacts are `4.2.0`.

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
