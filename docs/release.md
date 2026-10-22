# Release process

Releases are automated from `main`. A version bump is the single entry point;
tags and GitHub Releases are created by the workflow after PyPI accepts the
upload, never by hand.

## Cutting a release

1. Update `CHANGELOG.md` with a new `## [vX.Y.Z] - YYYY-MM-DD` section.
2. Update every version surface (all are checked against each other):
   - `pyproject.toml` -> `version = "X.Y.Z"`
   - `src/cds2/_version.py` -> `__version__ = "X.Y.Z"`
   - `package.json` -> `"version": "X.Y.Z"`
   - `CITATION.cff` -> `version: "X.Y.Z"`
3. Commit and merge to `main`. No tag is pushed by hand.

A merge that changes `src/cds2/_version.py` runs
`release-on-version-bump.yml`, which dispatches `release.yml`. That workflow:

- refuses to run anywhere except `main`;
- re-verifies version lockstep (including `CHANGELOG.md` and `CITATION.cff`);
- runs lint, format, strict mypy, the 100% coverage gate, property tests and
  the benchmark regression gate on Python 3.12;
- builds the sdist and pure-Python fallback wheel (`CDS_PURE=1`), plus
  compiled wheels with cibuildwheel for Linux/macOS/Windows on CPython
  3.10-3.13;
- publishes to PyPI;
- smoke-tests the published wheel on all three operating systems, using the
  artifact built in the run (never a registry download);
- creates the `vX.Y.Z` tag and a GitHub Release carrying the same wheel and
  sdist files that were published to PyPI.

The repository's `CI` workflow also runs on the version-bump push, so the full
OS/Python matrix gates the change before or alongside the release run.

### npm follows a successful PyPI release

`npm-publish.yml` does not run in parallel with the Python release. It is
triggered by `workflow_run` when `Release` completes, and its gate reads that
run's `publish` job: npm proceeds only when PyPI actually published. A failed
or skipped Python release therefore cannot leave an npm launcher pointing at a
version that PyPI does not serve.

| Registry | Workflow | Trigger | Credential |
|---|---|---|---|
| PyPI | `release.yml` | dispatched by the version bump | trusted publishing first, `PYPI_API_TOKEN` fallback |
| npm | `npm-publish.yml` | `workflow_run` after a successful `Release` | `NPM_TOKEN` (no npm trusted publisher is registered yet) |

**Provenance status.** PyPI Trusted Publishing is registered and working: every
file of every release from `5.1.0` through `5.4.5` carries a PEP 740
attestation (the per-file `pypi.org/integrity/.../<file>/provenance` endpoint
returns `200`; the directory URL returns `404` even when the files are
attested). The `PYPI_API_TOKEN` fallback remains for recovery, but a token
upload mints no attestation, and the attestations above prove the trusted path
is what actually publishes. The npm launcher still uses `NPM_TOKEN` because no
npm trusted publisher is registered; a token cannot mint a Sigstore
attestation, so npm releases carry none.

## Trusted-publisher setup

PyPI is done. npm is pending:

1. Sign in at [npmjs.com](https://www.npmjs.com) and open the package settings
   for `scientific-computing-system-2.0`.
2. Add a **trusted publisher** with:
   - Owner: `Furox-Art`
   - Repository: `scientific-computing-system-2.0`
   - Workflow filename: `npm-publish.yml`
   - Environment: `npm`
3. Confirm afterwards that
   `https://registry.npmjs.org/-/npm/v1/attestations/scientific-computing-system-2.0@<version>`
   returns `200` for the next release (it returns `404` for `5.4.4`).
4. Dispatch `npm-publish.yml` without `use_token_fallback` to publish through
   OIDC; see [npm launcher shim](npm.md).

## Recovery

- **PyPI published, tag or GitHub Release missing.** Re-dispatch `Release`
  from `main` while `main` still declares that version (before the next bump).
  The publish job rebuilds the wheel and sdist and compares them with the files
  PyPI serves (name, size, sha256). Only an exact match skips the upload; any
  difference fails instead of silently substituting bytes. The tag and GitHub
  Release steps then run, accepting an existing tag only when it is on `main`
  and declares the same version.
- **PyPI published, npm missing or failed.** Dispatch `npm-publish.yml` with
  `use_token_fallback=true`. It refuses to overwrite an existing npm version
  and verifies the published version afterwards.
- **Manual full release.** Dispatch `Release`. When it succeeds, the same
  `workflow_run` rule publishes npm; dispatching `npm-publish.yml` directly
  publishes npm alone.

## Manual local publish (fallback)

If both trusted publishing and the API token are unavailable, build locally
and upload with your own token:

```bash
CDS_PURE=1 python -m build --sdist && CDS_PURE=1 python -m build --wheel
python -m build --wheel            # native wheel with compiled kernels
python -m twine check dist/*
python -m twine upload dist/*
```

Never commit tokens; pass them through the environment only. A token upload
produces **no provenance attestation**.

## Workflow-graph check

`scripts/check_workflows.py` reads the workflow files and fails if the
automatic path stops reaching a publisher (directly or through
`workflow_run`), if a dispatched workflow leads to no publisher, if the bump
trigger stops watching `src/cds2/_version.py`, if CI starts filtering
publish-path changes, or if a publisher stops running its own guard (`pytest`
for PyPI, `npm test` for npm) before uploading:

```bash
python scripts/check_workflows.py
```

It runs on every commit through `tests/test_workflow_graph.py`.

## Notes

- A version already uploaded to PyPI can never be re-uploaded under the same
  filename; bump the version instead of re-tagging.
- The import root stays `cds2` and the console script stays `cds2` regardless
  of the distribution name.
