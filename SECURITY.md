# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 5.x | yes |
| 4.x | yes |
| < 4.0 | no - the project was renamed; please upgrade |

## Reporting a vulnerability

Please report security issues privately via
[GitHub security advisories](https://github.com/Furox-Art/scientific-computing-system-2.0/security/advisories/new)
rather than public issues. You will receive a response within a week.

The attack surface is intentionally small: cds2 runs locally on NumPy arrays,
reads only the files you pass it, and makes no network calls.
