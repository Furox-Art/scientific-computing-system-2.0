## [Unreleased]

### Fixed

- **Documentation claimed a provenance guarantee the project does not have.**
  PR #37 added `docs/npm.md` and a SECURITY.md supply-chain section on the
  assumption that the PyPI side carried a PEP 740 attestation while npm did not.
  **Measurement shows neither registry serves one.** Verified live:

  | Endpoint | Result |
  |---|---|
  | `pypi.org/integrity/scientific-computing-system-2.0/5.2.6/` | **404** |
  | `pypi.org/integrity/scientific-computing-system-2.0/` (project-wide) | **404** |
  | `pypi.org/integrity/scientific-computing-system-2.0/5.2.5/` | **404** |
  | `registry.npmjs.org/-/npm/v1/attestations/scientific-computing-system-2.0@5.2.6` | **404** |
  | `registry.npmjs.org/-/npm/v1/attestations/scientific-computing-system-2.0@5.2.5` | **404** |

  The old wording discussed only npm's missing attestation, which implied the
  PyPI side was attested. Every such statement is corrected, and the three things
  that were being conflated are now separated explicitly:

  - a **digest** (`sha256`/`blake2b_256` on PyPI, `dist.integrity` `sha512` on
    npm) detects tampering with the bytes but proves nothing about origin;
  - npm's **`dist.signatures`** are registry transport signatures proving the
    tarball came from npm, *not* build provenance;
  - a **PEP 740 / Sigstore attestation** is the missing piece, and is absent on
    both channels.

  New page `docs/supply-chain.md` carries the verified table, the distinction,
  the digest values a consumer can pin, the build properties this repository
  actually does guarantee, and the fix path. `SECURITY.md`, `docs/npm.md` and
  `README.md` link to it and no longer imply provenance on either channel.
- **`docs/release.md` claimed "no API tokens are stored in the repository".**
  False as written: `release.yml` keeps a `PYPI_API_TOKEN` fallback for when
  trusted publishing fails. The page now says trusted publishing is the intended
  credential *and* that a token fallback exists, which is also the reason no
  attestation is produced. The one-time publisher setup is now marked as
  required for provenance, with a verification step.
- **`python -m cds2` documentation made consistent.** `src/cds2/__main__.py` now
  exists (5.2.6), so `python -m cds2 info` and `python -m cds2 --help` both exit
  0 - verified. `docs/npm.md` stated the shim target without mentioning the
  equivalent module invocation; it now notes all three forms work.

### Added

- **`tools/check_provenance_claims.py`**, a guard against unverified
  provenance claims in both directions: it rejects affirmative "attested" /
  "signed provenance" wording that no live check backs, and it re-probes the
  attestation endpoints so a correct-today "no attestation" statement is itself
  flagged as stale once a trusted publisher is registered. Unreachable endpoints
  are reported as notes, never as failures, so an offline run cannot raise a
  false alarm. Negative-controlled in both directions: injecting
  "is attested and carries signed provenance" fails the text scan, and pointing
  an endpoint at a URL that returns 200 fails the staleness check.

  Not wired into CI in this change: that needs a step added to an existing
  workflow, which is outside this change's file ownership. Wire it as
  `python tools/check_provenance_claims.py`.

### Not changed

- No version bump, no tag, no release, no publish. `5.2.6` remains the live
  version and was not touched.

### Fixed

- **Two pinned GitHub Actions pointed at commits that do not exist upstream.**
  `release.yml` pinned `pypa/cibuildwheel@cfbec09` and
  `softprops/action-gh-release@5113cdc`; the GitHub API returns
  `No commit found for SHA` (HTTP 422) for both. GitHub Actions resolves a
  `uses:` pin before the job body runs, so any workflow reaching either step
  failed at action resolution. Because both sit in the release path, the breakage
  was invisible until a release was attempted. Replaced with the commits the
  upstream tags actually point at: `pypa/cibuildwheel@v4.2` →
  `e090b81e30c4d855ea63bf4b6e59204c09a101ae` and
  `softprops/action-gh-release@v3` → `efb35369e0ad2afab669f228072c1b0d510eae64`.
- `tools/consistency_audit.py` now verifies every `uses:` pin: each must be a full
  40-character commit SHA, and that SHA must resolve to a real commit in the
  upstream repository. It also requires any job using a repository-writing action
  such as `softprops/action-gh-release` to hold `contents: write`, which rules out
  a release that publishes successfully and then reports failure. A definitive
  "no such commit" fails the audit; a network or rate-limit problem is reported as
  a warning so a transient blip cannot fail a healthy build.

### Notes

- All ten distinct action pins across the five workflows were checked. The other
  eight resolve correctly and match the version named in their trailing comment.
- The `pypi` publish job keeps least privilege (`contents: read` plus
  `id-token: write` for trusted publishing); only the `github_release` job holds
  `contents: write`, which is what creating a tag and release requires.

- **`python -m cds2` now works.** `src/cds2/__main__.py` did not exist, so the
  module invocation always failed with `No module named cds2.__main__`. It is the
  conventional way to run a package and costs three lines, so it is now supported
  rather than documented as a papercut. `python -m cds2 info` and
  `python -m cds2 --help` both exit 0. As a side benefit the module entry point
  does not emit the `runpy` double-import RuntimeWarning that `python -m cds2.cli`
  triggers on every call, because `cds2/__init__.py` imports `cli` eagerly.
- **The CLI fuzz harness now actually exercises the CLI.** It spawned
  `python -m cds2` while no `__main__` module existed, so every invocation exited
  1 without running anything, and the assertion `returncode in (0, 1, 2)` was
  satisfied by that failure. Exit status alone cannot carry the property the
  harness claims to test, because Python exits `1` both for a handled validation
  error and for an uncaught exception. The assertion now also requires a
  non-negative return code (no fatal signal), no traceback on stderr, and no
  missing-module error, and a dedicated canary asserts that `python -m cds2 info`
  exits 0 and prints the distribution banner.

### Changed

- The CLI fuzz subprocess timeout is 30 s. A cold `import cds2` was measured at a
  median of 2.1 s with a worst observed cost of 7.7 s, which is what made the
  previous 5.0 s timeout fail the macOS / Python 3.10 leg. 30 s is roughly 4x the
  worst legitimate cost and still bounds a genuine hang: a control that sleeps for
  60 s is caught at exactly 30 s.
- The README no longer states that `python -m cds2` does not work; it documents
  the equivalence with the `cds2` console script.

## [v5.2.6] - 2026-10-03

Patch release whose main purpose is to let the PyPI project page pick up the
corrected README: the distribution long description comes from `README.md` at
build time, and only a new release can change it, so the 5.2.5 project page still
shows the previous 2,041-character description. PyPI release descriptions are
immutable, so this could not be fixed in place.

This is **not** a documentation-only release. The range also contains the
security hardening of #28, which changes shipped runtime behavior; it is
described under "Security" rather than folded into the documentation notes.

### Security

Hardening from #28, with new guard coverage:

- `prof/history` never creates directories in `__init__`; installed layouts fall
  back to a per-user data directory instead of `parents[3]` inside the
  interpreter.
- `guided_fit.rerun_manifest` confines `source_path` to the manifest directory
  and the current working directory by default, with
  `allow_outside_run_dir` / `trusted_roots` as an explicit opt-out and a new
  `--allow-outside-run-dir` CLI flag. Relative sources resolve run-dir first,
  then the working directory; a trusted-but-missing source raises
  `FileNotFoundError` naming the paths tried.
- `_fast_pagerank` requires `PyBUF_C_CONTIGUOUS` and additionally checks
  `PyBuffer_IsContiguous`.
- New `tests/test_security_guards.py` fails the build if `eval`, `exec`,
  `pickle`, unsafe `yaml.load` or `shell=True` ever appear under `src/cds2`,
  scanning the Python modules and the C accelerator sources alike.

### Fixed

- **npm publishing can run before a trusted publisher exists.**
  `npm-publish.yml` gained an explicit `use_token_fallback` dispatch input
  (boolean, default `false`). When set, it publishes with the repository's
  `NPM_TOKEN` as `NODE_AUTH_TOKEN` and omits `--provenance`, because a
  long-lived automation token cannot mint a Sigstore attestation; OIDC trusted
  publishing with provenance stays the default. Requesting the token mode with
  an empty secret fails closed.
- An explicit `use_token_fallback=false` was previously rejected as "must be a
  boolean" instead of selecting the OIDC path.
- npm publish now tolerates registry propagation delay after uploading, and the
  npm documentation reflects published reality rather than intent.

### Changed

- **All GitHub Actions are pinned to full commit SHAs**, every job declares
  least-privilege `permissions`, and CI tool versions (`ruff`, `mypy`) are pinned
  through `constraints/ci-tools.txt`. Unpinned tooling had turned `main` red
  repeatedly.
- `setup.py` now fails closed if the C accelerator sources are missing, instead
  of silently producing a capability-less wheel; CI builds the wheel *from the
  sdist* and asserts both accelerators are present and importable.
- Builds are pinned to `SOURCE_DATE_EPOCH`, and CI asserts the wheel is
  bit-identical across two builds.
- A new `MANIFEST.in` gives the sdist full scope (C sources, the complete test
  tree, docs, examples, benchmarks, tooling) while excluding CI workflows.
- The npm tarball is restricted by a `files` allowlist; CI asserts the packed
  contents match it exactly.
- `consistency_audit` is a real gate rather than advisory. It previously died
  with `ModuleNotFoundError` because the job never installed the package.
- `tools/consistency_audit.py` no longer treats a repository legitimately ahead
  of PyPI as a fatal error, which would have deadlocked every release once `main`
  required pull requests.
- `guided_fit` is bound in `src/cds2/__init__.py` and listed in `__all__`, so
  `cds2.guided_fit` resolves after a bare `import cds2`. Public exports go from
  534 to 535.

### Documentation

- **The README now explains the whole repository.** It was 747 words for a
  package with 46 flat modules and 6 subpackages: 46 modules received 78 words
  between them, all ten CLI commands were not listed (2 of 10 appeared), the
  `guided-fit` workflow was a bare link, all 20 scripts in `examples/` went
  unmentioned, the five install extras were undocumented, and the optional C
  kernels were not mentioned at all. It is now ~1,510 words with a 54-row
  module table (module, what it does, real entry points), all ten subcommands,
  a guided-fit section, the extras table, the C-kernel fallback caveat, and links
  to the twelve docs pages and six repository files that had none.
  Every entry point named in the table was checked against the module's `__all__`
  rather than written from memory.
- **npm section cut from 283 to ~60 words.** It accounted for ~38% of the README
  for an install path the README itself calls unnecessary. The detail now lives
  in `docs/npm.md`, which the README links.
- **Hero figure counts are generated, and the counting logic was wrong.**
  `scripts/make_promo.py` counted `glob("*.py")`, which silently ignored the six
  subpackages (`array_api`, `bench`, `estimator`, `gpu`, `nlp`, `prof`) and so
  under-reported the surface. It now counts flat modules *plus* directories that
  carry an `__init__.py`, which excludes `src/cds2/src` (C sources, not a
  package). Sub-facts are derived too: the API-page count comes from `docs/api`,
  and the test count from `pytest --collect-only`.

### Fixed

- **Stale numbers in the README and hero image, all corrected against the tree:**

  | Claim | Was | Now |
  |---|---|---|
  | hero "public functions" | 534 | **535 public exports** - and relabelled, because `__all__` mixes functions, classes, submodules and data, so a "functions" label was structurally wrong |
  | hero "tests" | 1,724 | **1,805** |
  | "API reference for all 46 modules" | 46 | **47 pages** (the count `tools/consistency_audit.py` reports) |
  | "46 importable modules" | 46 | **52 importable names** (46 flat + 6 subpackages) |

  No benchmark-speedup claim was reintroduced; the previous audit had already
  removed the "10-100x faster" line and it stays gone.
- **Deprecation notice was technically true but misleading.** The README said
  `cds2.special` and `cds2.distributions` warn "on import". Because
  `__init__.py` imports both eagerly, a bare `import cds2` emits **both**
  warnings whether or not you touch either module - verified. The README now
  says so.
- **`python -m cds2` does not work and the README implied otherwise.** There is
  no `cds2.__main__`; the correct invocations are the `cds2` console script or
  `python -m cds2.cli`. Now stated explicitly, because `docs/npm.md` tells npm
  users that the shim runs `python -m cds2.cli` and the difference is easy to
  trip over.
- **Benchmark provenance is disclosed instead of glossed.** The README described
  the results as "roughly at parity on wrapper-only calls". They are in fact from
  **cds2 3.0.0, commit `6ea5a02`, 2026-08-22**, while the release is 5.2.5, and
  **4 of the 13 cases are losses**: `dataframe summary` 1.82x, `solve 800x800`
  1.22x, `solve 8x8 x300` 1.17x, `welch` 1.03x. The README now states the stale
  commit, names the four losses, notes that the pandas row computes strictly
  more, and keeps the one clear win (PageRank at 0.15x vs NetworkX) in proportion.
- **`cds2.nlp` was missing from the README** entirely, despite having an API page.

npm publishing is live for this repository and `5.2.5` is now on the npm
registry. This reverses an earlier owner decision, and the reversal is recorded
here deliberately rather than applied silently.

### Fixed

- **False publish failure: the post-publish visibility check no longer
  misreports a successful publish.** Run 37120536500 published `5.2.5` and then
  failed, logging `npm registry metadata not visible yet (attempt 1/6)` through
  `(attempt 6/6)` and finally `ERROR: ... was not visible on the registry after
  publishing`. The version *was* published and is served by the registry today.
  Two independent causes, both fixed:

  1. **Budget far shorter than the propagation window.** `npm publish` returns
     when the registry's *write* path accepts the upload, while the *read* path
     is served by a CDN that propagates asynchronously. ~95s was observed for a
     sibling repository; the check allowed ~60s (6 attempts, flat `sleep 10`).
  2. **Stale cached reads.** The packument is served with
     `Cache-Control: public, max-age=300`, and `npm view` resolves through
     npm's on-disk HTTP cache, so retries inside that TTL could keep replaying
     the pre-publish document even after the CDN had it.

  `npm-publish.yml` now delegates to `scripts/wait-for-npm-visibility.mjs`: 12
  attempts, 5s base backoff capped at 45s (a ~7.5 minute budget, wider than both
  the 95s propagation window and the 300s cache TTL), reading the registry
  directly with `Cache-Control: no-cache` plus a cache-busting query parameter.
  A direct read is used instead of `npm view` because it (a) defeats both cache
  layers, (b) distinguishes a 404 ("not propagated yet") from a 5xx or transport
  error, which `npm view` collapses into one exit code, and (c) avoids paying npm
  CLI startup on every attempt. First match wins. On budget exhaustion the
  script emits `::warning::` with a re-check hint and then exits non-zero, so a
  version that genuinely never appeared still fails closed - a slow-but-successful
  publish is never reported as a hard error, and a missing publish is never
  reported as a success.

  Regression test `scripts/check-npm-visibility-poll.test.mjs` (wired into
  `npm test`, and therefore into the `npm-validate` CI job) asserts **both**
  directions: a 404-then-success sequence resolves and stops polling on the first
  match, and a version that never appears still fails after the full budget. It
  also pins the budget above the propagation window and cache TTL, bounds the
  backoff, and requires every read to defeat caching. Four mutations were
  verified to break it: shrinking the budget to the old 6x10s shape, making the
  poll always succeed, removing the `::warning::` marker, and removing the
  cache-buster.

### Documentation

- **npm is documented as published.** The README previously stated that npm was
  discontinued and carried `"private": true`; that was already false on `main`
  and is now corrected. `README.md` documents the npm package as what it
  actually is - a Node launcher shim whose whole tarball is five files and about
  8 KB, which spawns `python -m cds2.cli` and therefore still requires the PyPI
  package on `PATH`. PyPI remains the install path; the conda recipe in
  `packaging/conda/` is in-repo only and no conda package is published.
- **Supply chain stated without overclaiming.** The published `5.2.5` carries
  **no** Sigstore provenance attestation: `/-/npm/v1/attestations/...@5.2.5`
  returns `404` and the version document has no `attestations` field, because it
  was published with the long-lived automation-token path that cannot mint one.
  The `dist.signatures` values that are present are npm's own ECDSA registry
  signatures, not build provenance, and the docs say so rather than implying an
  attested build. The npm badge was added only after confirming the registry
  `latest` dist-tag and the published version list match PyPI (`5.2.5` on both).

### Reverted

- **npm publishing re-enabled.** Commit `4c8cd18` ("docs: mark this as the NumPy
  build and stop npm publishing") disabled npm publishing and added
  `"private": true` to `package.json` as a guard. Both are reverted: the npm
  registry is a distribution channel again for all owned repositories, and
  `"private": true` is removed so `npm publish` is not blocked by npm itself.
- **The disabled-marker guard removed.** The `npm publish` step that echoed
  `npm publishing is disabled; PyPI is the install path` and exited 0 is gone,
  replaced by a real publish.

### Added

- **Token publish mode.** `npm-publish.yml` now takes an explicit
  `workflow_dispatch` input `use_token_fallback` (boolean, default `false`).
  When set to `true` it publishes with the repository's long-lived `NPM_TOKEN`
  secret as `NODE_AUTH_TOKEN` and runs `npm publish --access public` **without**
  `--provenance`. This supersedes the earlier decision, recorded just below, that
  a long-lived token is never used at all. It exists because no npmjs.com trusted
  publisher is registered yet, so OIDC-only publishing cannot succeed today.
  OIDC trusted publishing remains the default and the preferred path for every
  future release.
- OIDC trusted publishing (`id-token: write`, `npm publish --provenance`) gated
  by a new `npm` environment. This is the default path and the one to keep using
  once an npmjs.com trusted publisher is configured.
- One version bump now updates both registries: `release-on-version-bump.yml`
  dispatches `npm-publish.yml` alongside `release.yml`. npm publishing stays in
  its own workflow so a failed npm publish cannot fail a completed PyPI release.
- Publish-path gates: JS entry-point syntax checks plus `npm test`, an exact
  `npm pack` tarball allowlist assertion, version lockstep across
  `package.json` / `src/cds2/_version.py` / `pyproject.toml`, an idempotent
  registry version existence check, and post-publish registry verification.
- A publish-workflow contract test (`scripts/check-npm-publish-contract.mjs`,
  wired into `npm test`) asserts both publish modes, that the token path omits
  `--provenance` and supplies `NODE_AUTH_TOKEN`, that a requested mode without a
  credential fails closed, that an explicit `use_token_fallback=false` is
  accepted, and that the npm version floor is compared numerically rather than by
  regex.

### Changed

- **Provenance is conditional on the credential, not on the flag.** A long-lived
  automation token cannot mint a Sigstore attestation, so the token path does not
  pass `--provenance` and the verification step states plainly that no attestation
  exists on that path. Publishing via token is therefore a weaker guarantee than
  publishing via OIDC, and this should be treated as temporary.

### Notes

- **No provenance on the token path.** Anyone installing npm `5.2.5` published
  through the token mode gets no Sigstore attestation. That is expected and is
  not a defect, but it is the reason to register the trusted publisher and return
  to OIDC. Verified against the registry: the attestation endpoint for
  `scientific-computing-system-2.0@5.2.5` returns `404`, and the published
  version document has no `attestations` field.
- **The npm version floor is numeric.** It is asserted with a numeric semver
  comparison, not a regular expression. A pattern such as
  `^11\.(5[1-9]|[6-9][0-9])\.|^1[2-9]\.` is wrong: `[6-9][0-9]` only covers
  `11.60`-`11.99`, so it rejects `npm 11.19.0` — the version bundled with Node
  24 — and would also reject `11.5.2` through `11.59.99`. The contract test fails
  the build if a regex gate reappears.
- The npm registry holds `2.0.0` and `5.2.5`, with `latest` resolving to
  `5.2.5`. `2.0.0` shipped a broken entry point (a JavaScript syntax error plus
  an `scs2.cli` module target that does not exist; the import root is `cds2`).
  npm versions are immutable, so `2.0.0` cannot be repaired in place. Install
  `5.2.5` or later.
- PyPI `5.2.5` and npm `5.2.5` are both published and both serve the same
  version string, so the version-lockstep gate in `npm-publish.yml` holds.
- Publishing an npm package from this repository means the same Python project
  is distributed on npm here as well as under its other published names. This is
  a second npm package name for one project, which is a consequence of the
  decision to re-enable npm publishing and is recorded here rather than left
  implicit. It is now actually the case, not a plan: npm `5.2.5` is live. What
  ships to npm is a Node launcher shim of five files (about 8 KB unpacked) that
  spawns `python -m cds2.cli`; it contains no Python and does not vendor the
  scientific stack, so PyPI remains the install path and npm is optional.
- `4c8cd18` also added `codemeta.json` and a conda-forge recipe
  (`packaging/conda/meta.yaml`) while stating this repository is not a second
  product. Those files are unrelated to npm publishing and are left in place.

## [v5.2.5] - 2026-09-29

PyPI discoverability metadata refresh patch. No public API or numerical behavior changes.

### Changed

- Expanded PyPI keywords and classifiers for scientific Python, Bayesian inference, reinforcement learning, information theory, numerical methods, and related discovery terms.
- Kept runtime behavior unchanged.

### Fixed

- Removed one redundant static type cast caught by the current mypy toolchain.
- Applied the repository's formatter to the recipe code examples so release CI remains green.

## [v5.2.4] - 2026-09-05

PyPI presentation patch: the README hero image now uses an absolute raw GitHub URL so it renders correctly in the package description.

### Fixed

- Updated the README hero image source to an absolute `raw.githubusercontent.com` URL compatible with PyPI rendering.
- Published the README presentation fix as a distinct patch release because existing PyPI release descriptions are immutable.

## [v5.2.3] - 2026-09-05

Scientific-correctness and reproducibility hardening patch: the full PR #19 audit is now included in the published package.

### Changed

- Guided-fit reproducibility is hardened for duplicate dataset identities, manifest reruns, plot-output naming, outlier diagnostics and raw-input hashing.
- Numerical and input validation is strengthened across statistics, reliability, Monte Carlo/MCMC, epidemiology, linear algebra, graph, machine-learning and estimator paths.
- Reliability fitting handles censored data and availability edge cases more defensively.
- Native-extension fallbacks and validation behavior are hardened while preserving pure-Python fallback operation.

### Fixed

- Removed a redundant unreachable guided-fit manifest hash fallback instead of retaining dead defensive code.
- Removed the remaining invalid-regex-escape test warning without changing runtime behavior.
- Added focused regression coverage for scientific edge cases identified by the correctness audit.

### Validation

- PR #19 final CI passed Ruff lint/format, strict mypy, 1703 tests at 100.00% coverage, property tests, package/installed-wheel/CLI smoke on Ubuntu/Windows/macOS, the Python 3.10-3.13 OS matrix, and the benchmark regression gate.

## [v5.2.2] - 2026-09-05

Guided-fit reporting patch: long PDF reports are now paginated without silently truncating scientific results or report text.

### Changed

- Guided-fit PDF reports wrap long lines and automatically span as many pages as required.
- The previous fixed 12,000-character PDF report slice has been removed.
- Regression coverage verifies that content from the beginning, middle and end of long reports is preserved.

### Fixed

- Prevented long guided-fit PDF reports from silently dropping content beyond the previous single-page character limit.

## [v5.2.1] - 2026-09-05

Release-integrity patch: the cross-platform installed-wheel CLI validation merged after v5.2.0 is now part of a distinct release commit, and the release pipeline is fail-closed against tag/PyPI drift.

### Changed

- Installed-wheel CLI smoke validation is retained on Ubuntu, Windows and macOS for `cds2 info`, `guided-fit`, fit/residual artifacts and `guided-fit-rerun`.
- Release preflight now requires `main`, enforces lockstep package/changelog metadata, and refuses an existing PyPI version or Git tag instead of silently reusing it.
- After publication, the exact public PyPI version is clean-installed and CLI-smoke-tested on Ubuntu, Windows and macOS before the GitHub Release is created.
- GitHub Release creation now verifies that the resulting tag resolves to the exact workflow commit.

### Fixed

- Prevented a release-integrity failure mode where an already-published PyPI version could be skipped while a GitHub release/tag was created from newer repository code.

## [v5.2.0] - 2026-09-05

Guided-fit scientific validation release: stronger diagnostics, rerun stability checks and dataset-specific guidance extend the 5.1 workflow without changing its user-controlled decision model.

### Added

- Dedicated residual diagnostic plots in PNG and PDF for every guided fit.
- Quantified detected-outlier influence using the estimated percentage RMSE reduction from a diagnostic refit; detected points are still never removed silently.
- Manifest rerun stability checks that flag material changes in input hashes, fit RMSE, parameters or reliability labels.
- Dataset-specific model recommendations when one shared model is materially weaker for part of a multi-dataset analysis.
- Real-world scientific validation tests using the scikit-learn Diabetes and Linnerud datasets as independent packaged test data.

### Changed

- Installed-wheel CLI smoke tests now require residual plot artifacts in addition to fit plots and reproducibility manifests.
- Guided-fit CLI output, README and API documentation now expose outlier influence, residual diagnostics, stability warnings and separate-model recommendations.
- Supported package version advances from 5.1.0 to 5.2.0.

## [v5.1.0] - 2026-09-05

Guided scientific fitting release: user-controlled model recommendation, validated fitting workflows, reproducibility records, reporting and a fully tested command-line interface join the library.

### Added

- **`cds2.guided_fit`** - recommends one candidate model while leaving final model choice to the user; supports multiple datasets, measurement uncertainty, explicit missing-data policies and user-approved outlier handling.
- Repeated cross-validation, independent numerical cross-checks, parameter uncertainty, 95% confidence intervals and reliability labels.
- Matplotlib PNG/PDF fit diagnostics, reproducibility manifests and PDF/HTML/Markdown reports.
- **CLI** - `cds2 guided-fit` and `cds2 guided-fit-rerun`, with interactive and non-interactive operation.
- Release/package CI smoke-tests the installed wheel through fit -> artifacts/manifest -> rerun.

### Changed

- `cds2.optimize.curve_fit` exposes fit diagnostics and forwards weighting, bounds, solver method and Jacobian controls to SciPy.
- Supported package version advances from 5.0.0 to 5.1.0.

## [v5.0.0] - 2026-09-03

Performance, GPU, testing and ecosystem release: profiling/benchmark
infrastructure, compiled C kernels, an optional GPU backend,
property-based/fuzz/oracle tests, NumPy Array API compliance and
scikit-learn estimator interfaces join the library.

### Added

- **`cds2.prof`** - wall-clock, memory and CPU-time profiling with a
  `@timed` decorator, append-only JSONL benchmark history and
  tolerance-based regression gates (`pytest --regression`).
- **`cds2.bench`** - CLI report generator for CI regression runs.
- **C kernel extensions** (`src/cds2/src/`) - `solve_triangular` and
  `eigh_tridiag` (`_fast_linop`), RK4 step and batched trapezoid rule
  (`_fast_integrate`), 1-D convolution and SOS IIR filtering
  (`_fast_signal`); OpenMP on Linux, ARM64 NEON paths, serial fallback
  elsewhere.
- **`cds2.gpu`** (optional `gpu` extra) - lazy CuPy backend for linalg
  (solve, eigh, SVD, Cholesky), signal (FFT family, power spectrum)
  and Monte Carlo (pi estimate, MC integration, Metropolis-Hastings).
- **`cds2.array_api`** - NumPy Array API 2023.12 compliant namespace
  (reductions, elementwise math, FFT, linalg).
- **`cds2.estimator`** - scikit-learn compatible estimators
  (LinearRegressionGD, RidgeSGD, KMeansSKL, PCASKL).
- **Testing** - hypothesis property-based tests (`tests/property/`),
  CLI and API fuzz tests (`tests/fuzz/`), SciPy/NumPy ground-truth
  oracle comparisons (`tests/oracles/`); CI gains property-tests and
  regression-gate jobs.

### Changed

- Version 4.3.0 -> 5.0.0; new optional extras `test` (hypothesis,
  scikit-learn) and `gpu` (cupy-cuda12x).
- Test count ~1200 -> 1596; 100% blended coverage, mypy strict and
  ruff clean maintained.

## [v4.3.0] - 2026-09-03

Domain and audit release: Bayesian optimization and SDE ensembles join
the library, PDE and data-analysis ports land, facade modules are
documented and thin wrappers deprecated.

### Added

- **`cds2.bayesopt`** - Gaussian Process surrogate, expected
  improvement and UCB acquisition, Bayesian optimization loop.
- **`cds2.sde`** - Euler-Maruyama and Milstein SDE ensembles with
  ensemble statistics.
- **`cds2.pde`** - heat/wave 1-D/2-D solvers (FTCS/leapfrog,
  CFL-guarded, Dirichlet/Neumann) ported from v1 and accelerated.
- **`cds2.data_analysis`** - DataSet/DataFrame bridge with
  describe/summarize, group-by and NaN-aware helpers.
- **`tools/consistency_audit.py`** - automated facade-vs-native audit
  used to keep wrapper docs honest.

### Changed

- Docs: facade vs real-capability guidance added; `cds2.special` and
  `cds2.distributions` documented as convenience re-exports kept for
  their typed dataclass DX.
- `cds2.interpolate`: deprecated `scipy.interpolate.lagrange`
  replaced with `BarycentricInterpolator`.

### Fixed

- CLI polynomial solver checked the leading coefficient instead of
  the trailing one, accepting degenerate input.
- Three wrong results on weighted graphs corrected; oracle tests
  against networkx added.
- Extension modules marked `optional=True` so compiler-less installs
  fall back to pure NumPy/SciPy.

## [v4.2.0] - 2026-08-24

Ten-domain expansion release built with parallel agent orchestration:
wavelets, epidemiology, image processing, genetics, reliability, finance,
text analysis, game theory, combinatorial optimization and spatial
statistics join the library.

### Added

- **`cds2.wavelets`** - Haar DWT/IDWT, multi-level decomposition and
  MAD-thresholded wavelet denoising.
- **`cds2.epidemiology`** - SIR/SEIR RK4 simulation with conservation
  guarantees, herd-immunity threshold, final-size fixed-point solver.
- **`cds2.image`** - 2-D convolution (same/valid), Gaussian blur, Sobel edge
  detection with direction, mean/max/min pooling, binary morphology.
- **`cds2.genetics`** - GC content, Hamming distance, k-mers, reverse
  complement, Needleman-Wunsch global alignment with traceback, ORF finder
  with standard translation table.
- **`cds2.reliability`** - Kaplan-Meier survival curves with censoring,
  Weibull MLE fit, MTBF/availability, composite bathtub hazard model.
- **`cds2.finance`** - log/simple returns, Sharpe/Sortino ratios, maximum
  drawdown tracking, Black-Scholes pricing with greeks, historical and
  Monte Carlo VaR.
- **`cds2.text`** - tokenization, smoothed TF-IDF matrix, cosine/Jaccard
  similarity, top-k term summaries.
- **`cds2.game_theory`** - strictly dominated action elimination, pure Nash
  enumeration, zero-sum minimax via linear programming, iterated prisoner's
  dilemma tournaments with five classic strategies.
- **`cds2.combinatorial`** - nearest-neighbor TSP + 2-opt improvement, 0/1
  knapsack DP with reconstruction, optimal assignment wrapper, LCS.
- **`cds2.spatial`** - row-standardized weight builder, Moran's I and
  Geary's C autocorrelation with z-scores, Clark-Evans nearest-neighbor
  index.

### Changed

- Module count 32 -> 42 (+ CLI), export surface ~470 names.
- Test count 927 -> ~1200; 100% blended coverage, mypy strict and ruff clean
  maintained across all 48 source files.
## [v4.1.0] - 2026-08-24

Industrial-grade expansion: statistical process control, design of
experiments and graph community detection join the library, plus three new
benchmark races and project governance docs.

### Added

- **`cds2.quality`** - Shewhart X-bar chart (A2/R-bar limits), EWMA chart
  with time-varying limits, two-sided CUSUM, attribute p-chart for variable
  lot sizes, and Cp/Cpk capability indices with normal-tail defective PPM.
- **`cds2.design`** - full factorial designs, 2^k-p fractional factorials
  via defining-relation generators ("D=ABC"), Latin hypercube sampling,
  face-centred/rotatable central composite designs and coded-to-physical
  factor mapping.
- **`cds2.graph`** additions: seeded label-propagation community detection
  (`detect_communities`) returning Newman modularity (`modularity`) for
  partition quality.
- Benchmarks: entropy vs hand-rolled numpy, Latin hypercube vs
  scipy.stats.qmc, PSO vs scipy differential evolution.
- Governance: CONTRIBUTING.md, SECURITY.md and Dependabot configuration.

### Changed

- Test count 866 -> 940+; 100% blended coverage, mypy strict and ruff clean
  maintained across all 38 source files.
## [v4.0.0] - 2026-08-24

Identity and discovery release: the project is renamed to
**scientific-computing-system-2.0** (repository, PyPI distribution and CLI
branding) and six new domain modules join the library. Module count
24 -> 30 (+ CLI), export surface ~350 names.

### Added

- **`cds2.infotheory`** - Shannon/joint/conditional entropy, KL and
  Jensen-Shannon divergence, cross entropy, (normalized) mutual information
  and Bandt-Pompe permutation entropy.
- **`cds2.chaos`** - Takens delay embedding, false nearest neighbours,
  Rosenstein largest Lyapunov exponent, Grassberger-Procaccia correlation
  dimension, sample entropy, R/S Hurst exponent, logistic map iteration and
  generic bifurcation scans.
- **`cds2.bayes`** - Beta-Binomial / Normal-Normal / Gamma-Poisson conjugate
  updates, credible intervals with sampling fallback, Gaussian naive Bayes,
  Bayes factors and Metropolis posterior draws.
- **`cds2.metaheuristics`** - real-coded genetic algorithm (tournament
  selection, blend crossover, elitism), particle swarm optimization and
  simulated annealing with exponential cooling.
- **`cds2.geometry`** - convex hulls, closest pair via KD-tree, point-in-
  polygon ray casting, shoelace area/perimeter, segment intersection tests,
  infinite-line intersections, centroids and rotations.
- **`cds2.rl`** - Bernoulli multi-armed bandits (epsilon-greedy with decay,
  UCB1), tabular Q-learning and a deterministic grid-world environment.
- **`cds2.modeling`** additions: symbolic polynomial integration
  (`integrate`) and exact polynomial root finding (`polynomial_coefficients`,
  `solve_polynomial`).
- **`cds2.graph`** additions: Brandes betweenness centrality, Wasserman-Faust
  closeness centrality and Lanczos-backed eigenvector centrality with an
  oscillation guard for bipartite graphs.
- **`cds2.scientific`** additions: `convert_units` / `list_units` covering
  length, mass, time, energy, pressure, angle and temperature.

### Changed

- Package identity: PyPI distribution and repository renamed from
  `cognitive-discovery-system-v2` to `scientific-computing-system-2.0`.
  Import root remains `cds2`; the CLI remains `cds2`.
- Test count 856; 100% blended coverage, mypy strict and ruff clean
  maintained across all 36 source files.
- Tooling: pre-commit configuration and a cibuildwheel wheel-building
  workflow added; GitHub repository renamed.

## [v3.3.0] - 2026-08-23

Heritage-completion release: every module from the v1.x line now has a
home in cds2. Module count 17 -> 24 (+ CLI), export surface ~330 names.

### Added

- **`cds2.modeling`** - symbolic mathematics: expression trees with full
  operator overloading, symbolic differentiation (product/quotient/chain
  for constant exponents and bases), algebraic simplification rules,
  substitution, LaTeX export, Newton equation solving and least-squares
  parameter fitting via `MathModel`.
- **`cds2.quantum`** - dense statevector circuit simulator: X/Y/Z/H/S/T
  gates, RX/RY/RZ rotations, CNOT/CZ/SWAP, probabilities and seeded
  measurement sampling (up to 16 qubits).
- **`cds2.knowledge`** - `KnowledgeGraph` (typed relations, BFS shortest
  paths, transitive closure, cycle detection), `Notebook` with tags and
  concept links, and ranked `search` across concepts/relations/notes.
- **`cds2.scientific`** - CODATA physical constants plus mechanics,
  electromagnetism, thermodynamics and relativity formula helpers.
- **`cds2.hypothesis`** - heuristic hypothesis generation: trend,
  periodicity, outlier and pairwise-correlation hypotheses with confidence
  scores and domain tags.
- **`cds2.nlp`** - educational NLP toolkit: micrograd-style scalar
  autograd, trainable BPE tokenizer, scaled dot-product + multi-head
  attention, and a deterministic mini-GPT forward pass with sampling.

### Changed

- Test count 453 -> 643; 100% blended coverage and mypy strict maintained
  across all 30 source files.

## [v3.2.1] - 2026-08-22

Packaging fix: restore the PEP 561 py.typed marker that went missing from
the working tree, so type checkers pick up cds2 inline annotations again
in installed distributions. Full audit otherwise clean: 453 tests at 100%
blended coverage, mypy strict zero errors, ruff clean, docs strict build,
all four examples verified, wheel contents confirmed (py.typed + both C
kernels).

## [v3.2.0] - 2026-08-22

Multicore-kernel release: OpenMP-accelerated C kernels on Linux wheels,
GIL released during all hot loops, plus an industrial computing guide.

### Added

- **OpenMP parallel C kernels** (Linux wheels): KMeans assignment sweep
  and PageRank power iteration fan across cores via pragma-guided loops;
  both kernels drop the GIL during compute so embedded Python threads
  keep running. Windows/macOS wheels build the serial kernel silently.
- **docs/industrial.md** - end-to-end guide: preconditioned six-figure
  solves, process-parallel Monte Carlo, streaming statistics, out-of-core
  CSV processing.

### Changed

- setup.py enables -fopenmp only on Linux toolchains (MSVC legacy OpenMP
  rejects the kernel loop shapes; Apple clang needs external libomp).
- Sparse iterative results gained residual_norm diagnostics earlier this
  cycle are now exercised on every solver in the test suite.
# Changelog

All notable changes to **scientific-computing-system-2.0** will be documented in
this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [v3.1.0] - 2026-08-22

Industrial-tier release: preconditioned sparse solvers at six-figure scale,
process-parallel Monte Carlo and constant-memory streaming statistics.

### Added

- **`cds2.sparse.jacobi_preconditioner` / `ilu_preconditioner`** - SuperLU
  ILU wrapped as LinearOperator; all iterative solvers accept ``M``.
  Showcase: a 250k-unknown system with condition ~6e9 stalls plain CG yet
  converges routinely under ILU preconditioning.
- **Solver diagnostics** - iterative results report true residual norm.
- **`cds2.montecarlo.parallel_mc_integrate`** - process-parallel chunked
  integration with independent per-worker seeds.
- **`cds2.stats.StreamingStats`** - Welford incremental mean/variance,
  vectorized pushes + pairwise merge, constant memory.
- **`cds2.io.iter_csv`** - chunked reader generator for out-of-core data.

### Changed

- Test count 431 -> 453; gates held throughout.

## [v3.0.0] - 2026-08-22

Flagship surface expansion.

### Added

- **`cds2.distributions`** - 57 functions across 19 probability families
  (t, chi2, F, exponential, uniform, lognormal, Poisson, binomial, gamma,
  beta, Weibull, Cauchy, Laplace, Gumbel, Pareto, Rayleigh, geometric,
  negative-binomial, hypergeometric), each with pdf/pmf + cdf (+ ppf).
- **Special functions doubled to 42** - digamma, Fresnel C/S, Airy Ai/Bi,
  Legendre Pn, elliptic K/E, exp1, hypergeometric 2F1, spherical Bessels,
  Bessel jv/yv/iv/kv, Hankel1, Struve H0/H1, Chebyshev T/U, Laguerre,
  Hermite, Jacobi, spherical harmonics, Lambert W, Faddeeva w, E_n, Si/Ci.
- **`examples/`** - four runnable end-to-end case studies: signal
  denoising, Michaelis-Menten fitting, Bayesian MCMC inference,
  citation-network PageRank + spectral clustering.

## [v2.6.0] - 2026-08-22

Scientific-depth release closing the biggest coverage gaps of the platform:
a full distributions module, a 2x special-functions expansion and runnable
case studies.

### Added

- **`cds2.distributions`**: 24 functions across eight probability
  distributions (Student-t, chi-squared, F, exponential, uniform,
  lognormal, Poisson, binomial) with pdf/pmf, cdf and ppf for each.
- **Special functions doubled**: digamma, Fresnel integrals, Airy Ai/Bi,
  Legendre polynomials, complete elliptic integrals K/E, exponential
  integral E1, Gauss hypergeometric 2F1, spherical Bessels j0/j1.
- **`examples/`**: four runnable end-to-end case studies:
  - signal denoising (Butterworth + Welch verification)
  - Michaelis-Menten experiment fitting with residual inference
  - Bayesian sensor-bias MCMC vs analytic conjugate posterior
  - citation-network PageRank + spectral clustering + DAG validation

### Changed

- Module count 16 -> 17; flat exports ~213; test count grew again with
  distribution and special-function suites.

## [v2.5.1] - 2026-08-22

Quality-gate release: strict typing and full coverage are now enforced by CI.
No API or behavior changes; 265 -> 344 tests.

### Added

- **mypy `--strict` gate**: zero errors across all 19 source files,
  enforced by a dedicated CI job (`types`). SciPy/Matplotlib/openpyxl
  handled via documented module overrides; pandas typed through stubs.
- **100% blended coverage gate**: statement + branch coverage at 100%
  enforced via `--cov-fail-under=100` on the reference cell. Coverage rose
  from 88%: ~80 gap-closing tests including forced NumPy-fallback KMeans
  paths, pagerank fallback arcs, empty-cluster rescue and every validation
  guard.

### Changed

- dev extras gained openpyxl, pyarrow, pandas-stubs and a numpy<2.5 cap
  (dev-only; mypy target 3.10 cannot parse numpy>=2.5 stubs).
- `pacf` dropped an always-true guard found while chasing the last branch.

## [v2.5.0] - 2026-08-22

Flagship release: two new scientific modules plus four major capability
upgrades across the platform. Module count now 16 + CLI, ~165 flat exports,
265 tests.

### Added

- **`cds2.sparse`**: large-scale sparse linear algebra: conjugate gradient,
  GMRES (with restart) and BiCGSTAB iterative solvers, Lanczos eigenpairs
  (`largest_eigenpairs` / `smallest_eigenpairs`) and truncated SVD.
- **`cds2.spectral`**: spectral graph theory: combinatorial and
  normalized Laplacians, Fiedler vectors, algebraic connectivity and
  spectral clustering (eigendecomposition embedding + k-means).
- **`cds2.montecarlo.metropolis_hastings`**: seeded random-walk MH sampler
  with burn-in, thinning and acceptance-rate diagnostics.
- **`cds2.optimize.minimize_constrained`**: SLSQP-based constrained
  minimization with SciPy-dict equality/inequality constraints.
- **`cds2.calculus.propagate_error`**: first-order uncertainty propagation
  through arbitrary functions via the Jacobian.
- **`cds2.integrate.solve_bvp`**: two-point boundary value problems via
  4th-order collocation.

### Changed

- `largest_eigenpairs` selects algebraically largest eigenvalues (LA).
- 28 new tests this cycle (236 -> 265).

## [v2.4.0] - 2026-08-22

Scientific-computing depth release: scattered-data RBF interpolation, ODE
event detection, stiff solvers and global optimization.

### Added

- **`cds2.interpolate.rbf_interp`**: radial-basis-function interpolation
  for scattered N-D data (thin-plate-spline default), with `smoothing` for
  approximating fits and `neighbors` kNN mode for large problems.
- **ODE events**: `cds2.integrate.solve_ivp` now accepts `events`
  (zero-crossing callables, `terminal = True` honored) and returns
  `t_events` / `y_events` on the result.
- **Stiff-solver documentation path**: `method="Radau" | "BDF" | "LSODA"`
  documented and tested with a stiff decay system.
- **`cds2.optimize.differential_evolution`**: stochastic global minimizer
  over box constraints returning `GlobalResult` (x, fun, nit, nfev).
- 9 new tests (236 total).

## [v2.3.0] - 2026-08-22

Scientific-surface expansion: two new modules and modern resampling-based
inference.

### Added

- **`cds2.calculus`**: numerical differentiation: `derivative` (central /
  forward / backward with adaptive steps), `complex_step_gradient`
  (machine-precision gradients via the complex-step trick), `jacobian`
  (finite-difference, R^n -> R^m) and `hessian` (central differences with
  exact-symmetric mixed partials).
- **`cds2.special`**: special functions: gamma/gammaln, erf/erfc/erfinv,
  beta/betaln, Bessel j0/j1/y0, Riemann-Hurwitz zeta.
- **`cds2.stats.bootstrap_ci`**: percentile bootstrap confidence intervals
  for arbitrary statistics, seeded and vectorized.
- **`cds2.stats.permutation_test`**: two-sided permutation test on mean
  differences with the +1 corrected p-value.
- **`cds2.linalg.expm/logm/sqrtm`**: matrix exponential, logarithm and
  principal square root.
- 40+ new tests (227 total); docs pages for both new modules.

### Changed

- Flat exports grew to ~150 symbols; module count now 14 plus the CLI.

## [v2.2.0] - 2026-08-22

Second compiled-acceleration release: PageRank joins the C kernel family,
and the specialist win column widens.

### Added

- **`cds2._fast_pagerank`**: C extension running the full power iteration
  over transposed-CSR arrays (buffer protocol, no build-time deps). The
  Python side now builds that structure with plain vectorized NumPy
  (argsort + searchsorted) instead of SciPy multiply/transpose round-trips.
  Graceful SciPy-sparse fallback retained.

### Changed

- **PageRank vs NetworkX: 1.35x slower -> 0.18x: about 5x faster** than
  NetworkX on the benchmark graph.
- KMeans vs scikit-learn improved further with kernel-path tuning:
  0.79x -> **0.72x faster**.
- Benchmark fairness fix: the PageRank race no longer times graph
  construction inside the cds2 contender loop.
- Scoreboard after this release - wins: linreg 0.74x, kmeans 0.72x,
  pagerank 0.18x, mc-pi 0.77x, minimize 0.81x, t-test 0.83x vs the
  underlying libraries; parity everywhere else; honest premiums only where
  cds2 returns strictly more information (describe quartiles, dataframe
  nulls/uniques).

## [v2.1.0] - 2026-08-21

The compiled-acceleration release: cds2 now ships its own C kernels where
they beat the wrapped libraries, with graceful NumPy fallback everywhere.

### Added

- **`cds2._fast_kmeans`**: a from-scratch C extension (buffer-protocol API,
  no build-time NumPy headers) implementing the Lloyd iteration loop:
  fused assignment/update, empty-cluster relocation, convergence tracking.
- **Compiled-wheel release pipeline**: cibuildwheel builds wheels for
  Linux/Windows/macOS x Python 3.10-3.13; a pure-Python fallback wheel and
  sdist are published alongside so compiler-less installs keep working.
- `KMeans._run_c_lloyd` / `_run_numpy_lloyd` split: identical results either
  way, chosen automatically by kernel availability.
- Benchmarks page note distinguishing wrapper-parity rows from
  more-information rows.

### Changed

- Build backend hatchling -> setuptools to declare the optional extension;
  `CDS_PURE=1` skips compilation entirely.
- KMeans vs scikit-learn: **1.91x slower -> 0.79x faster** (C kernel).
- PageRank: transposed-CSR matvec prepared once, dangling mass via `take`,
  per-iteration renormalization removed. 2.29x -> ~1.35x vs NetworkX on the
  benchmark graph.
- `io.summarize`: vectorized aggregation, single notna pass.
- `stats.describe`: single-pass manual moments matching scipy exactly.

## [v2.0.0] - 2026-08-21

The first release of the v2 generation: a full rewrite on top of the scientific
Python stack. The pure-Python algorithms proven in
[scientific-computing-system](https://github.com/Furox-Art/scientific-computing-system)
(v1.x) form the foundation; v2 rebuilds them on NumPy/SciPy for speed and adds
new domain modules on top.

### Added

- **Accelerated core** (NumPy / SciPy backed):
  - `cds2.linalg`: solve, det, inv, pinv, eig/eigh, SVD, least squares,
    cholesky, norms, trace, matrix power, rank, condition number
  - `cds2.stats`: descriptive statistics, t-tests (one-sample, independent,
    Welch, paired), ANOVA, Kruskal-Wallis, Mann-Whitney U, Wilcoxon,
    normality test, Pearson/Spearman/Kendall correlations, chi-square
    independence, effect sizes (Cohen's d, eta-squared, Cramer's V),
    percentiles, z-scores, normal pdf/cdf/ppf
  - `cds2.optimize`: minimize, scalar minimization, root finding
    (brentq/newton/systems), linear programming, nonlinear least squares,
    curve fitting
  - `cds2.integrate`: quad/dblquad/triple integration, ODE solving via
    `solve_ivp`, trapezoid/simpson rules, cumulative integration
  - `cds2.interpolate`: linear/cubic/pchip interpolation, Lagrange
    polynomials, scattered-data gridding, regular-grid interpolation
  - `cds2.signals`: FFT family, periodogram/Welch/spectrogram, Butterworth
    low/high/band-pass filters, peak finding, convolution/correlation,
    Hilbert envelope, resampling, detrending
  - `cds2.montecarlo`: seeded pi estimation, 1-D Monte Carlo integration,
    expectation estimation, hit-or-miss area estimation
  - `cds2.graph`: adjacency builders, connected components, Dijkstra /
    Bellman-Ford / Floyd-Warshall shortest paths, minimum spanning tree,
    degrees, topological order, and **PageRank** (power iteration)
- **New domain modules**:
  - `cds2.ml`: LinearRegression, LogisticRegression, KMeans (k-means++
    seeding), PCA, KNeighborsClassifier, StandardScaler, train/test split,
    synthetic data generators, classification/regression metrics
  - `cds2.timeseries`: moving average, exponential smoothing, differencing,
    classical seasonal decomposition, ACF/PACF, Ljung-Box test
  - `cds2.viz`: matplotlib helpers: series, histogram, scatter, heatmap,
    spectrum, regression overlay, confusion matrix
  - `cds2.io`: pandas-backed CSV/JSON readers/writers plus optional
    Excel/Parquet bridges and a DataFrame summarizer
- **CLI**: `cds2 info | stats | integrate | linsolve | plot`
- **CI**: GitHub Actions matrix (Ubuntu/Windows/macOS x Python 3.10-3.13)
  with ruff + pytest

### Changed

- Runtime dependencies are now explicit: numpy, scipy, pandas, matplotlib.
  The zero-dependency philosophy of v1.x is intentionally retired in favor of
  a faster, richer stack. The v1 line remains maintained separately at its own
  repository.
