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

### One bump, both registries

`src/cds2/_version.py` is the single automatic trigger. Pushing a commit that
changes only that file on `main` runs `release-on-version-bump.yml`, which
dispatches **both** publish workflows:

| Registry | Workflow | Credential path selected by the bump |
|---|---|---|
| PyPI | `release.yml` | trusted publishing, `PYPI_API_TOKEN` fallback |
| npm | `npm-publish.yml` | `use_token_fallback=true` (the `NPM_TOKEN` path) |

This coverage is enforced, not assumed. `scripts/check_workflows.py` reads the
workflow graph and fails if the bump path stops dispatching a publish
workflow, if a publish workflow becomes unreachable, if the bump trigger stops
watching `src/cds2/_version.py`, or if a publisher stops running its own guard
(`pytest` for PyPI, `npm test` for npm) before uploading:

```bash
python scripts/check_workflows.py
```

It runs on every commit through `tests/test_workflow_graph.py`, so CI picks it
up with no extra job.

The same script also locks the CI side of the contract: `tests.yml` must not
filter `push`/`pull_request` by path. A path filter there is exactly how a
change to `release.yml`, `npm-publish.yml`, `package.json` or `index.js` would
stop re-running the tests that guard publishing.

### Manual dispatch is the deliberate exception

Both `release.yml` and `npm-publish.yml` also accept `workflow_dispatch`, and
dispatching either one publishes **only that registry**. That asymmetry with
the bump path is intentional (an escape hatch for recovery, and manual
re-dispatches are idempotent because both publishers refuse an
already-published version), but it means a manual publish must be dispatched
twice, once per registry. Do not "fix" it by adding a `push` trigger to
`npm-publish.yml`: a version bump touches `package.json` too, which would fire
both that trigger and the bump dispatch and publish the same version twice.


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
