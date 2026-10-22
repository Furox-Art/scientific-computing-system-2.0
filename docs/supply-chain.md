# Supply chain and provenance

**PyPI releases are attested; npm releases are not.** This page states exactly
what exists, how to check it, and what has to happen before npm catches up. It
is deliberately explicit about the difference between a digest, a registry
transport signature, and a build provenance attestation, because conflating
them is how supply-chain claims go wrong.

## Verified state as of 5.4.5

Checked directly against the live registries on 2026-10-22. PyPI provenance is
per file: the directory URL returns `404` even when every file is attested, so
the per-file endpoint is the one that answers the question.

| Check | Endpoint | Result |
|---|---|---|
| PyPI PEP 740 attestations, 5.4.5 wheel (pure) | `https://pypi.org/integrity/scientific-computing-system-2.0/5.4.5/scientific_computing_system_2_0-5.4.5-py3-none-any.whl/provenance` | **200** |
| PyPI PEP 740 attestations, 5.4.5 sdist | `https://pypi.org/integrity/scientific-computing-system-2.0/5.4.5/scientific_computing_system_2_0-5.4.5.tar.gz/provenance` | **200** |
| PyPI PEP 740 attestations, 5.1.0-5.4.5, every file | per-file `/provenance` endpoint | **200** (330 files) |
| PyPI PEP 740 attestations, 5.0.0 | per-file `/provenance` endpoint | **404** |
| PyPI PEP 740 attestations, project-wide directory | `https://pypi.org/integrity/scientific-computing-system-2.0/` | **404** |
| npm Sigstore attestations, 5.4.4 (latest npm) | `https://registry.npmjs.org/-/npm/v1/attestations/scientific-computing-system-2.0@5.4.4` | **404** |

Published versions: PyPI `4.0.0, 4.2.0, 5.0.0, 5.1.0, 5.2.0-5.2.6, 5.3.0, 5.4.0-5.4.5`
(latest `5.4.5`); npm `2.0.0, 5.2.5, 5.2.6, 5.3.0, 5.4.0-5.4.4` (latest `5.4.4`;
`5.4.5` has not been published to npm).

You can reproduce every row above with `curl -o /dev/null -w '%{http_code}\n' <url>`.

## Three different things, routinely confused

**1. A digest — integrity, not proof of origin.**
Every artifact carries a content hash. PyPI publishes `sha256` and `blake2b_256`
per file; npm publishes `dist.integrity` (a `sha512`) and a legacy
`dist.shasum` (`sha1`). A digest lets you detect that the bytes you received are
the bytes the registry published. It says nothing about *who* built them, so it
is not provenance: an attacker who can replace the artifact can replace the
digest they advertise.

**2. `dist.signatures` on npm — a registry transport signature, NOT build
provenance.**
npm attaches its own ECDSA signatures to every tarball. They prove the tarball
came from npm's registry and was not altered in transit or at rest on npm's
side. They are signed by npm, **not** by the build, and they say nothing about
which source commit, runner or build environment produced the bytes. Do not
present them as provenance. Present on every published npm version.

**3. A provenance attestation — signed evidence about the build, present on PyPI.**
A PyPI PEP 740 attestation or an npm Sigstore attestation binds the artifact
digest to the identity of the workflow and the source commit, via a public
transparency log. That is the thing people mean by "verified build". **Every
PyPI file from 5.1.0 through 5.4.5 carries one; no npm release does.**

## What this repository does guarantee

These are build-integrity properties enforced by CI on every push. They are real
and were verified against the workflow definitions, not inherited from prose:

- **Reproducible pure-Python wheel.** The `reproducible-build` job builds the
  wheel twice with `SOURCE_DATE_EPOCH` pinned to the commit timestamp and
  asserts the two `sha256` sums are identical, so the build is bit-reproducible
  on a fixed toolchain.
- **The sdist actually ships the native accelerators.** The `sdist-native` job
  builds an sdist, then builds a native wheel *from that sdist*, and asserts
  `cds2/_fast_kmeans` and `cds2/_fast_pagerank` are present and importable. A
  sdist that silently omitted the C sources could not pass.
- **Version lockstep.** `package.json`, `src/cds2/_version.py`, `pyproject.toml`,
  `CITATION.cff`, `codemeta.json` and the conda recipe must agree, checked by a
  release-metadata test, so the two registries cannot drift.
- **Cross-platform install smoke.** Published wheels are installed and their CLI
  exercised on Linux, macOS and Windows before release.
- **Immutable history.** Release jobs fail closed if the version already exists
  on a registry, and both registries make published versions immutable.

What these do **not** give you: they are not a substitute for a provenance
attestation. They constrain the build; the PEP 740 attestation is what binds the
published PyPI artifact to the workflow identity and source commit.

## What a consumer can check today

Verify PyPI provenance for the exact file you install, and pin by digest. For
PyPI, `pip install --require-hashes` against the `sha256` values in the
release's file list; for npm, compare `dist.integrity` with what your lockfile
pins:

```bash
# PyPI: the per-file attestation bundle for 5.4.5's sdist
curl -sSf -o provenance.json \
  https://pypi.org/integrity/scientific-computing-system-2.0/5.4.5/scientific_computing_system_2_0-5.4.5.tar.gz/provenance
python -m json.tool provenance.json | head

# what the registry currently serves
pip download scientific-computing-system-2.0==5.4.5 --no-deps -d /tmp/x
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" /tmp/x/<file>

npm view scientific-computing-system-2.0@5.4.4 dist.integrity
```

The sdist digest for `5.4.5` is
`ca21009208debb4e576ad06cac80ba0dd716389563c907e96e56e04e60f34c81`.

## npm provenance is still pending

PyPI is done: the Trusted Publisher is registered under *Account management →
Publishing* (Owner `Furox-Art`, Repository `scientific-computing-system-2.0`,
Workflow `release.yml`, Environment `pypi`), and every file from `5.1.0`
through `5.4.5` was uploaded through it, which is why the attestations exist.
The `PYPI_API_TOKEN` fallback remains for recovery only; a token upload mints no
attestation.

npm is **pending, not in place**:

- A trusted publisher must be registered on npmjs.com for Owner `Furox-Art`,
  Repository `scientific-computing-system-2.0`, Workflow filename
  `npm-publish.yml`, Environment `npm`. Until then npm publishes use
  `NPM_TOKEN`; a classic automation token cannot mint a Sigstore attestation,
  which is why the token path omits `--provenance`. Once registered, dispatch
  `npm-publish.yml` without `use_token_fallback` to publish through OIDC.

In short: npm needs **registry configuration, not workflow flags.** Until then,
treat npm releases as digest-verifiable only, and do not describe them as
attested.

## npm-specific notes

The npm package is a thin Node launcher, not a second distribution of the
library: it contains no Python and calls into the PyPI-installed package. See
[npm launcher shim](npm.md). One npm version is unusable and cannot be
repaired because npm versions are immutable: `2.0.0` shipped a broken entry
point, and the stale channel predates this repository taking over the name.
Install `5.2.5` or later; the latest npm version is `5.4.4`.