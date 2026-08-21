# Modules overview

CDS v2 ships twelve importable modules plus a CLI. Every heavy computation is
delegated to NumPy/SciPy; pandas carries tabular data; matplotlib renders
figures.

| Module | Built on | Purpose |
|---|---|---|
| [`cds2.linalg`](api/linalg.md) | numpy.linalg | solve, det, inv, pinv, eig/eigh, SVD, least squares, cholesky, norms |
| [`cds2.stats`](api/stats.md) | scipy.stats | t-tests, ANOVA, non-parametrics, correlations, chi-square, effect sizes |
| [`cds2.optimize`](api/optimize.md) | scipy.optimize | minimize, root finding, linear programming, curve fitting |
| [`cds2.integrate`](api/integrate.md) | scipy.integrate | quad, multi-dimensional integration, ODE solvers, Newton-Cotes rules |
| [`cds2.interpolate`](api/interpolate.md) | scipy.interpolate | linear/cubic/pchip splines, lagrange, scattered and regular grids |
| [`cds2.signals`](api/signals.md) | scipy.signal | FFT family, PSD/welch/spectrogram, Butterworth filters, peaks, envelope |
| [`cds2.montecarlo`](api/montecarlo.md) | numpy.random | seeded pi estimate, MC integration/expectation, hit-or-miss |
| [`cds2.graph`](api/graph.md) | scipy.sparse.csgraph | components, shortest paths, MST, degrees, topological order, PageRank |
| [`cds2.ml`](api/ml.md) | numpy + cKDTree | linear/logistic regression, k-means++, PCA, kNN, metrics, generators |
| [`cds2.timeseries`](api/timeseries.md) | pandas | moving average, EWM, decomposition, ACF/PACF, Ljung-Box |
| [`cds2.viz`](api/viz.md) | matplotlib | series, histogram, scatter, heatmap, spectrum, regression, confusion |
| [`cds2.io`](api/io.md) | pandas | CSV/JSON read-write, optional Excel/Parquet bridges, DataFrame summaries |
| [`cds2.cli`](api/cli.md) | argparse | `cds2` console entry point |

## Import styles

Flat re-exports live on the package root for the most common helpers:

```python
import cds2

cds2.pagerank
cds2.pi_estimate
cds2.minimize
```

Each module can also be imported directly:

```python
from cds2.graph import pagerank
```

## Versioning

Static semantic versioning: `cds2.__version__` mirrors the git tag. The v1.x
zero-dependency line is maintained separately in its own repository.
