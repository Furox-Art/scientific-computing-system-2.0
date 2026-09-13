# scientific-computing-system-2.0

<p align="center">
  <img src="https://raw.githubusercontent.com/Furox-Art/scientific-computing-system-2.0/main/docs/assets/promo_hero.png" alt="scientific-computing-system-2.0 scientific computing platform" width="100%">
</p>

[![CI](https://github.com/Furox-Art/scientific-computing-system-2.0/actions/workflows/tests.yml/badge.svg)](https://github.com/Furox-Art/scientific-computing-system-2.0/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/scientific-computing-system-2.0)](https://pypi.org/project/scientific-computing-system-2.0/)
[![Python](https://img.shields.io/pypi/pyversions/scientific-computing-system-2.0)](https://pypi.org/project/scientific-computing-system-2.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

**CDS2** is a scientific-computing platform built on NumPy, SciPy, pandas, and matplotlib. It combines a uniform high-level API with CDS-native numerical methods, compiled acceleration where it is useful, reproducible scientific workflows, and explicit validation practices.

It is designed for researchers and developers who want more than a collection of wrappers: model fitting with diagnostics and replayable manifests, numerical and stochastic methods across scientific domains, benchmarked native kernels, strict CI, and documentation that separates convenience APIs from original CDS capability.

> **CDS2 is not a replacement for NumPy/SciPy.** Thin wrappers are identified as such and upstream libraries are preferred when CDS2 adds no scientific or workflow value.

## Why CDS2?

- **Broad scientific coverage** — linear algebra, statistics, optimization, integration, signals, time series, Monte Carlo, graphs, ML, Bayesian methods, PDE/SDE methods, chaos, epidemiology, reliability, design of experiments, scientific constants, and more.
- **Reproducible fitting workflows** — `guided-fit` records model/preprocessing choices, uncertainty information, validation metrics, plots, reports, and a rerunnable manifest.
- **Native capability where it matters** — CDS-specific algorithms and compiled C kernels coexist with clearly labeled convenience wrappers.
- **Scientific quality gates** — strict typing, property-based tests, numerical regression checks, and an enforced 100% coverage gate in CI.
- **Cross-platform verification** — CI tests Linux, Windows, and macOS across Python 3.10–3.13.
- **Research-oriented documentation** — validation strategy, reproducibility expectations, benchmark methodology, case-study standards, and citation metadata are part of the repository.

## Installation

```bash
pip install scientific-computing-system-2.0
```

From source:

```bash
git clone https://github.com/Furox-Art/scientific-computing-system-2.0.git
cd scientific-computing-system-2.0
pip install -e .[dev]
```

## Quick start

```python
import numpy as np
import cds2

# Linear algebra
A = [[3.0, 1.0], [1.0, 2.0]]
b = [9.0, 8.0]
x = cds2.linalg.solve(A, b)

# Statistics
result = cds2.stats.independent_t_test(
    [1, 2, 3, 4, 5],
    [3, 4, 5, 6, 7],
)

# Optimization
opt = cds2.optimize.minimize(
    lambda v: (v[0] - 2) ** 2 + (v[1] + 1) ** 2,
    x0=[0.0, 0.0],
)

# Signal analysis
freqs, psd = cds2.signals.power_spectrum(
    np.sin(np.linspace(0, 100, 1024)),
    fs=256.0,
)

# Reproducible stochastic optimization
pso = cds2.metaheuristics.pso_minimize(
    lambda v: (v[0] - 3) ** 2,
    [(-10, 10)],
    seed=1,
)
```

## Scientific capability

CDS2 groups its functionality into several layers rather than pretending every module contributes the same kind of value.

| Area | Representative modules | Examples |
|---|---|---|
| Numerical core | `linalg`, `calculus`, `integrate`, `interpolate`, `sparse` | linear systems, derivatives, quadrature, ODEs, iterative solvers |
| Statistics & inference | `stats`, `distributions`, `bayes`, `bayesopt` | hypothesis tests, effect sizes, conjugate updates, Bayesian optimization |
| Signals & dynamics | `signals`, `spectral`, `wavelets`, `chaos`, `timeseries` | PSD/Welch, filtering, spectral clustering, entropy, nonlinear diagnostics |
| Modeling & simulation | `pde`, `sde`, `epidemiology`, `reliability`, `design` | PDE solvers, stochastic ensembles, SIR/SEIR, survival analysis, DOE |
| Machine learning | `ml`, `nlp`, `text` | regression, classification, clustering, PCA, tokenization, attention |
| Graphs & optimization | `graph`, `optimize`, `metaheuristics`, `combinatorial`, `game_theory` | PageRank, shortest paths, PSO/GA/SA, assignment, minimax |
| Scientific domains | `scientific`, `quantum`, `genetics`, `spatial`, `finance`, `quality` | constants/formulas, statevectors, sequence tools, spatial statistics, risk, SPC |
| Reproducible workflows | `guided_fit`, `cli`, `io`, `viz` | model fitting, manifests, reruns, reports, plots, data I/O |

See the [modules documentation](docs/modules.md) for the full API surface.

## Native capability vs convenience wrappers

Some modules intentionally provide a smaller, typed interface over NumPy/SciPy/pandas. Others contain CDS-native algorithms or workflow logic.

- **Use upstream directly** when CDS2 only mirrors a mature upstream function and adds no meaningful workflow value.
- **Use CDS2 convenience APIs** when a consistent return type or integration with the rest of CDS2 is useful.
- **Use CDS2-native modules** for capabilities implemented by CDS2 itself, including native ML components, stochastic methods, scientific workflows, and accelerated kernels.

This distinction is documented rather than hidden. The project does not treat a thin wrapper as new numerical research.

## Guided scientific fitting

Interactive workflow:

```bash
cds2 guided-fit data.csv --x time --y response
```

Multiple datasets with measurement uncertainty:

```bash
cds2 guided-fit experiment-a.csv experiment-b.csv \
  --x time --y response --sigma uncertainty \
  --report pdf --output-dir guided-fit-results
```

Replay a completed analysis:

```bash
cds2 guided-fit-rerun guided-fit-results/guided_fit_manifest.json
```

`guided-fit` supports linear, quadratic, exponential, power, and logistic models. It can report RMSE, held-out validation RMSE, R² where defined, parameter uncertainty, fit/residual plots, and an overall reliability verdict. Model choice, missing-data treatment, outlier handling, and report generation remain explicit user-facing choices.

## Validation and reproducibility

High line coverage alone is not considered scientific validation. CDS2 now documents a stronger evidence model built around:

- analytical invariants and known closed-form cases,
- independent implementation oracles,
- property-based testing,
- numerical stress and edge-case testing,
- justified tolerances,
- reproducible seeds/configuration,
- provenance for external data,
- replayable end-to-end workflows.

Read:

- [Validation & reproducibility](docs/validation-and-reproducibility.md)
- [Research-readiness checklist](docs/research-readiness.md)
- [Reproducible case-study standard](docs/case-studies.md)

## Benchmarks

CDS2 ships an executable benchmark suite and a CI performance-regression gate. The current repository scoreboard includes the following reference measurements:

| Race | Baseline | CDS2 / baseline |
|---|---|---:|
| PageRank 400 nodes, 2400 edges | NetworkX | **0.18x** |
| K-Means 4000×2, k=8 | scikit-learn | **0.72x** |
| Linear regression 20000×10 | scikit-learn | **0.74x** |
| Monte Carlo π, n=2M | hand-vectorized NumPy | **0.77x** |
| solve / eigh / rfft / Welch / minimize | NumPy / SciPy | ~1.00x |

The PageRank and K-Means hot paths include from-scratch C extensions (`cds2._fast_pagerank`, `cds2._fast_kmeans`) with pure-Python fallback packaging.

These measurements are **reference-environment results, not universal speed claims**. Hardware, BLAS/LAPACK backend, compiler, dependency versions, workload size, warm-up, and run-to-run variance matter.

```bash
python benchmarks/run_benchmarks.py
python benchmarks/run_benchmarks.py --quick
```

See [benchmark results](docs/benchmarks.md) and the [benchmark methodology](docs/benchmark-methodology.md).

## CI and engineering controls

The main CI pipeline includes:

- Ruff lint and format checks,
- strict `mypy`,
- a 100% blended coverage gate,
- Linux / Windows / macOS package smoke tests,
- Python 3.10 / 3.11 / 3.12 / 3.13 test matrix,
- property-based tests with Hypothesis,
- benchmark regression checks,
- built-wheel installation and CLI smoke tests.

These controls improve software reliability; they do not replace domain-specific scientific review.

## Examples

Runnable examples live in [`examples/`](examples/). Existing examples include nonlinear experiment fitting, epidemic simulation, Bayesian inference, chaos diagnostics, graph analysis, metaheuristic optimization, document similarity, image processing, portfolio risk, and reinforcement learning.

For substantial examples, the project uses the [case-study standard](docs/case-studies.md): explicit question, provenance, assumptions, validation target, uncertainty, diagnostics, and reproduction instructions.

## Relationship to the pure-Python CDS line

The independent [scientific-computing-system](https://github.com/Furox-Art/scientific-computing-system) project keeps a zero-runtime-dependency pure-Python core.

CDS2 makes the opposite engineering tradeoff: it uses the scientific Python ecosystem to gain broader domain coverage, optimized backends, richer workflows, and optional acceleration.

## Documentation

- [Project documentation](https://furox-art.github.io/scientific-computing-system-2.0/)
- [Getting started](docs/getting-started.md)
- [Modules](docs/modules.md)
- [Examples](docs/examples.md)
- [Benchmarks](docs/benchmarks.md)
- [Validation & reproducibility](docs/validation-and-reproducibility.md)

## Development

```bash
pip install -e .[dev]
pytest
ruff check .
mypy
```

## Citation

Research users can cite the software using [`CITATION.cff`](CITATION.cff). A DOI is intentionally not claimed here unless one is actually registered for a release.

## License

MIT — see [LICENSE](LICENSE).
