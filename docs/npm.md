# npm launcher shim

**PyPI is the install path for this library.** The npm package
[`scientific-computing-system-2.0`](https://www.npmjs.com/package/scientific-computing-system-2.0)
is an optional Node launcher, not a second distribution of the code.

## What is actually in the npm tarball

The published `5.2.5` tarball is about 8 KB unpacked and contains five files:

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

## Supply chain

- **No provenance attestation on `5.2.5`.** The npm attestation endpoint for
  this version returns `404`, and the published version document has no
  `attestations` field. It was published with a long-lived automation token,
  which cannot mint a Sigstore attestation, so `--provenance` was deliberately
  not passed. Do not treat it as an attested build.
- **What the `dist.signatures` field is.** npm attaches its own ECDSA registry
  signatures to every published tarball. They prove the tarball came from npm;
  they are *not* Sigstore build provenance and are not evidence about how the
  release was built.
- **Pin by integrity.** `npm view scientific-computing-system-2.0@5.2.5 dist.integrity`
  returns the published `sha512` hash.
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