# Release process

Releases are fully automated from git tags. The `Release` workflow tests,
builds distributions for every supported platform and publishes to PyPI.

**Trusted Publishing is the intended credential, but it is not the only path and
it is not currently proven in place.** The publish job requests `id-token: write`
and runs `pypa/gh-action-pypi-publish` (trusted publishing) first, but it also
retains a `PYPI_API_TOKEN` fallback for when trusted publishing fails. A token
upload cannot mint a PEP 740 attestation, which is why
`pypi.org/integrity/<project>/<version>/` currently returns `404` for every
published release. Until the publisher below is registered and the
trusted-publishing path is the one that actually succeeds, no release carries
provenance. See [Supply chain and provenance](supply-chain.md).

## One-time PyPI setup (per distribution name) - required for provenance

Without this, uploads fall back to a token and carry no attestation.

1. Sign in at [pypi.org](https://pypi.org) and open
   **Account management -> Publishing**.
2. Add a **new pending publisher** with:
   - Owner: `Furox-Art`
   - Repository: `scientific-computing-system-2.0`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. Create the matching **`pypi` environment** in the GitHub repository
   settings if it does not exist yet.
4. Confirm afterwards that `pypi.org/integrity/scientific-computing-system-2.0/`
   stops returning `404`.

The equivalent npmjs.com trusted publisher (Owner `Furox-Art`, Repository
`scientific-computing-system-2.0`, Workflow filename `npm-publish.yml`,
Environment `npm`) is required for npm provenance; see
[npm launcher shim](npm.md).

## Cutting a release

1. Update `CHANGELOG.md` with a new `## [vX.Y.Z] - YYYY-MM-DD` section.
2. Mirror the version in both places that must stay in lockstep:
   - `pyproject.toml` -> `version = "X.Y.Z"`
   - `src/cds2/_version.py` -> `__version__ = "X.Y.Z"`
3. Commit, push to `main`, then tag and push the tag:

   ```bash
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```

4. The workflow then runs:
   - full CI (lint, strict mypy, 100% coverage gate, 12 OS/Python matrix)
   - sdist + pure-Python fallback wheel build (`CDS_PURE=1`)
   - compiled wheels via cibuildwheel for Linux/macOS/Windows on CPython
     3.10-3.13
   - PyPI upload and a GitHub Release with generated notes.

## Manual local publish (fallback)

If Trusted Publishing is unavailable, build locally and upload with a token:

```bash
CDS_PURE=1 python -m build --sdist && CDS_PURE=1 python -m build --wheel
python -m build --wheel            # native wheel with compiled kernels
python -m twine check dist/*
python -m twine upload dist/*
```

Never commit tokens; pass them through the environment only. Be aware that a
token upload, like the workflow's own token fallback, produces **no provenance
attestation** - so a release cut this way carries digests only.

## Notes

- A version already uploaded to PyPI can never be re-uploaded under the same
  filename; bump the version instead of re-tagging.
- The import root stays `cds2` and the console script stays `cds2`
  regardless of the distribution name.
