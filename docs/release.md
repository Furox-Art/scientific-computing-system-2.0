# Release process

Releases are fully automated from git tags. The `Release` workflow tests,
builds distributions for every supported platform and publishes to PyPI via
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) - no API
tokens are stored in the repository.

## One-time PyPI setup (per distribution name)

1. Sign in at [pypi.org](https://pypi.org) and open
   **Account management -> Publishing**.
2. Add a **new pending publisher** with:
   - Owner: `Furox-Art`
   - Repository: `scientific-computing-system-2.0`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. Create the matching **`pypi` environment** in the GitHub repository
   settings if it does not exist yet.

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

Never commit tokens; pass them through the environment only.

## Notes

- A version already uploaded to PyPI can never be re-uploaded under the same
  filename; bump the version instead of re-tagging.
- The import root stays `cds2` and the console script stays `cds2`
  regardless of the distribution name.
