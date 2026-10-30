# npm launcher shim

**PyPI is the install path for this library.** The npm package
[`scientific-computing-system-2.0`](https://www.npmjs.com/package/scientific-computing-system-2.0)
is an optional Node launcher, not a second distribution of the code.

## What is actually in the npm tarball

The published `5.4.5` tarball is about 9 KB compressed (about 24 KB unpacked)
and contains five files:

| File | Role |
|---|---|
| `index.js` | spawns `python -m cds2.cli` |
| `bin/scs2.js` | the `scs2` executable shim |
| `package.json` | metadata, version lockstep with PyPI |
| `README.md`, `LICENSE` | documentation and licence |

There is no Python in it, and it does not vendor NumPy, SciPy, pandas or
matplotlib. That is the important consequence: **`npm install` alone gives you
nothing scientific.** The shim invokes whichever `python` is on your `PATH` and
delegates everything to the PyPI package.

```bash
pip install scientific-computing-system-2.0   # required
npm install -g scientific-computing-system-2.0 # optional
scs2 stats 1,2,3,4,5                           # from the PyPI install
```

If you only want the command-line tool, stop after the `pip install`: it already
provides a `cds2` console script, which is the same entry point the shim calls.
`python -m cds2` also works and is equivalent, since `src/cds2/__main__.py`
exists.

## Supply chain

**Signing is version-dependent.** `5.4.5`, published on 2026-10-23, is the
first npm version with a Sigstore attestation: its attestation endpoint returns
`200` with a SLSA provenance v1 bundle. Every earlier npm version, including
`5.4.4`, returns `404` and has no `attestations` field; do not treat those as
attested builds. (PyPI is a different story: every file from `5.1.0` through
`5.4.5` carries a PEP 740 attestation; see
[Supply chain and provenance](supply-chain.md).)

- **Why the split.** `npm-publish.yml` has an opt-in long-lived-token mode which
  deliberately omits `--provenance`, because a classic automation token cannot
  mint a Sigstore attestation. That path produced every npm version before
  `5.4.5`; `5.4.5` was published through npm's OIDC trusted publisher. See
  [Supply chain and provenance](supply-chain.md) for the full picture.
- **What `dist.signatures` is, and is not.** npm attaches its own ECDSA registry
  signatures to every tarball. They prove the tarball came from npm; they are
  **not** Sigstore build provenance and are not evidence about which source
  commit or runner produced it.
- **What you can check.** Pin `dist.integrity` (a `sha512`):
  `npm view scientific-computing-system-2.0@5.4.5 dist.integrity`. That is a
  digest, so it detects tampering with the bytes but proves nothing about origin.
- **A stale `2.0.0` is still on the registry.** It shipped a broken entry point
  (a JavaScript syntax error, and it targeted an `scs2.cli` module that does not
  exist - the import root is `cds2`). npm versions are immutable, so it cannot be
  repaired in place. Use `5.2.5` or later.

## Two names, one project

Publishing from this repository means the same project is reachable on npm under
this name as well as on PyPI. That is a deliberate, recorded consequence of
re-enabling npm publishing rather than an accident; it does not fork the code,
and the npm package tracks the PyPI version exactly. The version-lockstep gate in
`npm-publish.yml` fails the release if `package.json`, `src/cds2/_version.py` and
`pyproject.toml` ever disagree.

## Reproducing the publish verification locally

Post-publish verification polls the registry until the new version is visible,
because `npm publish` returns on the write path while the CDN-backed read path
propagates asynchronously. The same check runs locally against a fake registry:

```bash
node scripts/check-npm-visibility-poll.test.mjs
```

See the release notes in `CHANGELOG.md` for the propagation-window and caching
details behind that poll.
