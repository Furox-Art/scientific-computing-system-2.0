# Supply chain and provenance

**There is no signed build provenance for any published release, on either
registry.** This page states exactly what exists, how to check it, and what has
to happen before an attestation appears. It is deliberately explicit about the
difference between a digest, a registry transport signature, and a build
provenance attestation, because conflating them is how supply-chain claims go
wrong.

## Verified state as of 5.2.6

Checked directly against the live registries:

| Check | Endpoint | Result |
|---|---|---|
| PyPI PEP 740 attestations, 5.2.6 | `https://pypi.org/integrity/scientific-computing-system-2.0/5.2.6/` | **404** |
| PyPI PEP 740 attestations, project-wide | `https://pypi.org/integrity/scientific-computing-system-2.0/` | **404** |
| PyPI PEP 740 attestations, 5.2.5 | `https://pypi.org/integrity/scientific-computing-system-2.0/5.2.5/` | **404** |
| npm Sigstore attestations, 5.2.6 | `https://registry.npmjs.org/-/npm/v1/attestations/scientific-computing-system-2.0@5.2.6` | **404** |
| npm Sigstore attestations, 5.2.5 | `https://registry.npmjs.org/-/npm/v1/attestations/scientific-computing-system-2.0@5.2.5` | **404** |

Published versions: PyPI `4.0.0, 4.2.0, 5.0.0, 5.1.0, 5.2.0-5.2.6` (latest `5.2.6`);
npm `2.0.0, 5.2.5, 5.2.6` (latest `5.2.6`).

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
present them as provenance. Present on `2.0.0`, `5.2.5` and `5.2.6`.

**3. A provenance attestation — signed evidence about the build, and absent here.**
A PyPI PEP 740 attestation or an npm Sigstore attestation binds the artifact
digest to the identity of the workflow and the source commit, via a public
transparency log. That is the thing people mean by "verified build". **This
project does not have it on either channel today**, as the 404s above show.

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
- **Version lockstep.** `package.json`, `src/cds2/_version.py` and
  `pyproject.toml` must agree, checked by a release-metadata test, so the two
  registries cannot drift.
- **Cross-platform install smoke.** Published wheels are installed and their CLI
  exercised on Linux, macOS and Windows before release.
- **Immutable history.** Release jobs fail closed if the version already exists
  on a registry, and both registries make published versions immutable.

What these do **not** give you: they do not tell you who built the artifact.
They constrain the build; they do not attest it.

## What a consumer can check today

Pin by digest. For PyPI, `pip install --require-hashes` against the `sha256`
values in the release's file list; for npm, compare `dist.integrity` with what
your lockfile pins:

```bash
# what the registry currently serves
pip download scientific-computing-system-2.0==5.2.6 --no-deps -d /tmp/x
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" /tmp/x/<file>

npm view scientific-computing-system-2.0@5.2.6 dist.integrity
```

The sdist digest for `5.2.6` is
`0f6ecdf1843793ae6320d746136de2c81585611e50d9ba35a1fa401232873622`.

## What has to happen before provenance appears

Both are **pending, not in place**:

- **PyPI.** A Trusted Publisher must be registered on pypi.org under
  *Account management → Publishing* with Owner `Furox-Art`, Repository
  `scientific-computing-system-2.0`, Workflow `release.yml`, Environment
  `pypi`. The `release.yml` publish step is already structured for it
  (`pypa/gh-action-pypi-publish` with `id-token: write` and the `pypi`
  environment), but it has a `PYPI_API_TOKEN` fallback, and a token upload
  cannot mint an attestation. Until the publisher is registered *and* the
  trusted-publishing path is the one that actually succeeds,
  `pypi.org/integrity/...` will keep returning 404.
- **npm.** A trusted publisher must be registered on npmjs.com for Owner
  `Furox-Art`, Repository `scientific-computing-system-2.0`, Workflow filename
  `npm-publish.yml`, Environment `npm`. `npm-publish.yml` also has an opt-in
  long-lived-token mode, which deliberately omits `--provenance`, because a
  classic automation token cannot mint a Sigstore attestation. OIDC trusted
  publishing with `--provenance` is the default path and the one to keep using.

In short: **fix the registry configuration, not the flags.** The workflow flags
are already correct; the credentials that make them meaningful are not yet
registered. Until then, treat every release as digest-verifiable only.

## npm-specific notes

The npm package is a thin Node launcher, not a second distribution of the
library: it contains no Python and calls into the PyPI-installed package. See
[npm launcher shim](npm.md). Two npm versions are unusable and cannot be
repaired because npm versions are immutable: `2.0.0` shipped a broken entry
point, and the stale channel predates this repository taking over the name.
Install `5.2.5` or later.