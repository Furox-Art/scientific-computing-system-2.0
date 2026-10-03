# scientific-computing-system-2.0  
  
<p align="center">  
  <img src="https://raw.githubusercontent.com/Furox-Art/scientific-computing-system-2.0/main/docs/assets/promo_hero.png" alt="scientific-computing-system-2.0 scientific computing platform" width="100%">  
</p>  
  
[![CI](https://github.com/Furox-Art/scientific-computing-system-2.0/actions/workflows/tests.yml/badge.svg)](https://github.com/Furox-Art/scientific-computing-system-2.0/actions/workflows/tests.yml)  
[![PyPI](https://img.shields.io/pypi/v/scientific-computing-system-2.0)](https://pypi.org/project/scientific-computing-system-2.0/)  
[![Python](https://img.shields.io/pypi/pyversions/scientific-computing-system-2.0)](https://pypi.org/project/scientific-computing-system-2.0/)  
[![npm](https://img.shields.io/npm/v/scientific-computing-system-2.0)](https://www.npmjs.com/package/scientific-computing-system-2.0)  
  
This is the NumPy build of [scientific-computing-system](https://github.com/Furox-Art/scientific-computing-system), not a separate product. Install this when you want NumPy, SciPy, pandas, and matplotlib. Install the other package when you want readable pure Python and no runtime dependencies.  
  
PyPI is the install path. A conda recipe lives in `packaging/conda/` for local builds; no conda package is published. The npm package is a launcher shim, not a second distribution of the library (see [npm](#npm-launcher-shim-optional) below).  
  
## Installation  
  
Requires Python 3.10+:  
  
```bash  
pip install scientific-computing-system-2.0  
```  
  
Full guide with a ten-minute tour:  
<https://furox-art.github.io/scientific-computing-system-2.0/getting-started/>  
  
## Quick start  
  
Every output below was executed against `scientific-computing-system-2.0==5.2.5`:  
  
```python
import cds2

x = cds2.linalg.solve([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])
print(x)  # [2. 3.]
print(cds2.montecarlo.pi_estimate(n=100_000, seed=42))  # 3.13776
print(cds2.infotheory.entropy([0.25, 0.25, 0.25, 0.25]))  # 2.0
```  
  
Command line:  
  
```bash  
cds2 stats 1,2,3,4,5  
cds2 info  
```  
  
API reference for all 46 modules:  
<https://furox-art.github.io/scientific-computing-system-2.0/>  
  
## What changed from v1  
  
- **Performance**: Vectorized NumPy/SciPy instead of pure-Python loops. Measured results are published with full provenance in [docs/benchmarks.md](docs/benchmarks.md): roughly at parity with the underlying libraries on wrapper-only calls, faster where a dedicated implementation applies (for example PageRank vs NetworkX). No blanket speedup factor is claimed.
- **Coverage**: More algorithms, more edge cases handled  
- **Testing**: Property-based tests, oracle comparisons against reference implementations  
- **Documentation**: Actually complete, with examples that run  
  
## Common scientific Python use cases

- Accelerated **linear algebra, statistics, optimization, integration, and interpolation**.
- **Bayesian inference**, Monte Carlo, uncertainty analysis, and reproducible model fitting.
- **Signal processing, time-series analysis, graphs and PageRank**.
- **Machine learning, reinforcement learning, information theory, and computational geometry**.
- **PDE/SDE solvers**, scientific visualization, pandas-backed I/O, and research-quality validation.

## npm launcher shim (optional)

The npm package `scientific-computing-system-2.0` is published from this
repository and tracks the same version as PyPI; both currently serve `5.2.5`.
It is a **thin Node launcher, not a second copy of the library**. Its entire
tarball is five small files (`index.js`, `bin/scs2.js`, `package.json`,
`README.md`, `LICENSE`, about 8 KB unpacked) and it contains no Python code. It
spawns `python -m cds2.cli`, so:

- you still install the library from PyPI - the npm package does not vendor
  NumPy, SciPy, pandas or matplotlib;
- the `scs2` command only works if that Python package is installed and
  importable on your `PATH`;
- for most people `pip install scientific-computing-system-2.0` plus the `cds2`
  console script is the whole story, and the npm package is unnecessary.

```bash
npm install -g scientific-computing-system-2.0   # optional
scs2 stats 1,2,3,4,5
```

Supply chain, stated precisely:

- **No provenance attestation.** The published `5.2.5` carries no Sigstore
  attestation: the registry's attestation endpoint
  (`/-/npm/v1/attestations/scientific-computing-system-2.0@5.2.5`) returns
  `404 Not found`, and the version document has no `attestations` field. It was
  published through the long-lived automation-token path, which cannot mint an
  attestation, so `--provenance` was deliberately not passed. The `dist.signatures`
  values that *are* present are npm's own ECDSA registry signatures proving the
  tarball came from npm; they are not Sigstore build provenance. Do not treat this
  package as provenance-attested.
- **Two names, one project.** This repository publishes the same project on npm
  under this name as well as on PyPI, and npm already carries an older `2.0.0`
  from before this repository took over the name. npm versions are immutable, so
  `2.0.0` cannot be repaired; install `5.2.5` or later.
- **Verification.** `npm view scientific-computing-system-2.0@5.2.5 dist.integrity`
  returns the published `sha512` integrity hash you can pin against.

## Modules

46 importable modules, each with an API reference page. Core numerical stack,
discovery and modelling, then applied domains
(`*` = [deprecated](https://furox-art.github.io/scientific-computing-system-2.0/deprecated/)):

- Core: [`cds2.linalg`](docs/api/linalg.md), [`cds2.stats`](docs/api/stats.md), [`cds2.optimize`](docs/api/optimize.md), [`cds2.integrate`](docs/api/integrate.md), [`cds2.interpolate`](docs/api/interpolate.md), [`cds2.signals`](docs/api/signals.md), [`cds2.montecarlo`](docs/api/montecarlo.md), [`cds2.graph`](docs/api/graph.md), [`cds2.ml`](docs/api/ml.md), [`cds2.timeseries`](docs/api/timeseries.md), [`cds2.viz`](docs/api/viz.md), [`cds2.io`](docs/api/io.md), [`cds2.calculus`](docs/api/calculus.md), [`cds2.special`](docs/api/special.md)`*`, [`cds2.sparse`](docs/api/sparse.md), [`cds2.spectral`](docs/api/spectral.md), [`cds2.distributions`](docs/api/distributions.md)`*`
- Discovery and modelling: [`cds2.infotheory`](docs/api/infotheory.md), [`cds2.chaos`](docs/api/chaos.md), [`cds2.bayes`](docs/api/bayes.md), [`cds2.metaheuristics`](docs/api/metaheuristics.md), [`cds2.geometry`](docs/api/geometry.md), [`cds2.rl`](docs/api/rl.md), [`cds2.quality`](docs/api/quality.md), [`cds2.design`](docs/api/design.md), [`cds2.modeling`](docs/api/modeling.md), [`cds2.hypothesis`](docs/api/hypothesis.md), [`cds2.knowledge`](docs/api/knowledge.md), [`cds2.scientific`](docs/api/scientific.md), [`cds2.quantum`](docs/api/quantum.md), [`cds2.cli`](docs/api/cli.md)
- Applied domains: [`cds2.bayesopt`](docs/api/bayesopt.md), [`cds2.combinatorial`](docs/api/combinatorial.md), [`cds2.data_analysis`](docs/api/data_analysis.md), [`cds2.epidemiology`](docs/api/epidemiology.md), [`cds2.finance`](docs/api/finance.md), [`cds2.game_theory`](docs/api/game_theory.md), [`cds2.genetics`](docs/api/genetics.md), [`cds2.guided_fit`](docs/api/guided_fit.md), [`cds2.image`](docs/api/image.md), [`cds2.pde`](docs/api/pde.md), [`cds2.reliability`](docs/api/reliability.md), [`cds2.sde`](docs/api/sde.md), [`cds2.spatial`](docs/api/spatial.md), [`cds2.text`](docs/api/text.md), [`cds2.wavelets`](docs/api/wavelets.md)

## The philosophy  
  
Scientific code should be boring. Not boring to write-boring to read. You should be able to look at a function and know exactly what it does, what it expects, and what it returns. No magic, no hidden state, no "just trust the library."  
  
Every function here has type hints, docstrings with examples, and tests that verify the math against known results. If the documentation and the code disagree, the code is wrong.  
  
## Deprecated modules  
  
`cds2.special` and `cds2.distributions` are deprecated since 4.3.0 and emit a  
`DeprecationWarning` on import. Use `scipy.special` and `scipy.stats` directly  
instead. Details:  
<https://furox-art.github.io/scientific-computing-system-2.0/deprecated/>  
  
## License  
  
MIT. 
