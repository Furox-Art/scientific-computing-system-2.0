# cognitive-discovery-system-v2

[![CI](https://github.com/Furox88/cognitive-discovery-system-v2/actions/workflows/tests.yml/badge.svg)](https://github.com/Furox88/cognitive-discovery-system-v2/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/cognitive-discovery-system-v2)](https://pypi.org/project/cognitive-discovery-system-v2/)
[![Python](https://img.shields.io/pypi/pyversions/cognitive-discovery-system-v2)](https://pypi.org/project/cognitive-discovery-system-v2/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

**CDS v2** is a scientific computing platform built on the scientific Python
stack — NumPy, SciPy, pandas and matplotlib. The algorithms proven in the
pure-Python [cognitive-discovery-system](https://github.com/Furox88/cognitive-discovery-system)
(v1.x) form its foundation; v2 rebuilds them for speed and adds new domain
modules on top.

## Installation

```bash
pip install cognitive-discovery-system-v2
```

From source:

```bash
git clone https://github.com/Furox88/cognitive-discovery-system-v2.git
cd cognitive-discovery-system-v2
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
r = cds2.stats.independent_t_test([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])

# Optimization
res = cds2.optimize.minimize(lambda v: (v[0] - 2) ** 2 + (v[1] + 1) ** 2, x0=[0.0, 0.0])
print(res.x)  # ~ [2.0, -1.0]

# Signals
freqs, psd = cds2.signals.power_spectrum(np.sin(np.linspace(0, 100, 1024)), fs=256.0)

# Graphs with PageRank
adj = cds2.graph.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)], directed=True)
scores = cds2.graph.pagerank(adj)
```

## Modules

| Module | Built on | Highlights |
|---|---|---|
| `cds2.linalg` | NumPy | solve, det, inv, pinv, eig/eigh, SVD, least squares, cholesky, cond |
| `cds2.stats` | scipy.stats | t-tests, ANOVA, non-parametrics, correlations, chi-square, effect sizes, normal dist helpers |
| `cds2.optimize` | scipy.optimize | minimize, roots (brentq/newton/system), linprog, least squares, curve fit |
| `cds2.integrate` | scipy.integrate | quad, 2-D/3-D integration, ODE solvers, trapezoid/simpson |
| `cds2.interpolate` | scipy.interpolate | linear/cubic/pchip, lagrange, griddata, regular grids |
| `cds2.signals` | scipy.signal | FFT, PSD/welch/spectrogram, Butterworth filters, peaks, envelope |
| `cds2.montecarlo` | NumPy Generator | pi estimate, MC integration/expectation, hit-or-miss (all seedable) |
| `cds2.graph` | scipy.sparse.csgraph | components, Dijkstra/Bellman-Ford/Floyd-Warshall, MST, topological order, PageRank |
| `cds2.ml` | NumPy/SciPy | LinearRegression, LogisticRegression, KMeans++, PCA, KNN, metrics, data generators |
| `cds2.timeseries` | pandas | moving average, EWM, differencing, seasonal decomposition, ACF/PACF, Ljung-Box |
| `cds2.viz` | matplotlib | series/histogram/scatter/heatmap/spectrum/regression/confusion plots |
| `cds2.io` | pandas | CSV/JSON read-write, optional Excel/Parquet bridges, DataFrame summaries |

## CLI

```bash
cds2 info
cds2 stats 1,2,3,4,5
cds2 integrate sin --a 0 --b 3.14159
cds2 linsolve --a "3,1;1,2" --b "9,8"
cds2 plot 1,3,2,5,4 --file out.png
```

## Relationship to CDS v1.x

The original zero-dependency pure-Python line lives at
[Furox88/cognitive-discovery-system](https://github.com/Furox88/cognitive-discovery-system)
and remains available. v2 is an independent project that trades that
constraint for the speed and breadth of the scientific Python ecosystem.

## Development

```bash
pip install -e .[dev]
pytest            # run the test suite
ruff check .      # lint
```

## License

MIT — see [LICENSE](LICENSE).
