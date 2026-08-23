# Modules overview

cds2 ships thirty importable modules plus a CLI. Every heavy computation is
delegated to NumPy/SciPy; pandas carries tabular data; matplotlib renders
figures.

## Core scientific stack

| Module | Built on | Purpose |
|---|---|---|
| [`cds2.linalg`](api/linalg.md) | numpy.linalg | solve, det, inv, pinv, eig/eigh, SVD, least squares, cholesky, norms |
| [`cds2.stats`](api/stats.md) | scipy.stats | t-tests, ANOVA, non-parametrics, correlations, chi-square, effect sizes |
| [`cds2.optimize`](api/optimize.md) | scipy.optimize | minimize, root finding, linear programming, curve fitting |
| [`cds2.integrate`](api/integrate.md) | scipy.integrate | quad, multi-dimensional integration, ODE solvers, Newton-Cotes rules |
| [`cds2.interpolate`](api/interpolate.md) | scipy.interpolate | linear/cubic/pchip splines, lagrange, scattered and regular grids |
| [`cds2.signals`](api/signals.md) | scipy.signal | FFT family, PSD/welch/spectrogram, Butterworth filters, peaks, envelope |
| [`cds2.montecarlo`](api/montecarlo.md) | numpy.random | seeded pi estimate, MC integration/expectation, Metropolis-Hastings |
| [`cds2.graph`](api/graph.md) | scipy.sparse.csgraph | components, shortest paths, MST, PageRank, centrality measures |
| [`cds2.ml`](api/ml.md) | numpy + cKDTree | linear/logistic regression, k-means++, PCA, kNN, metrics, generators |
| [`cds2.timeseries`](api/timeseries.md) | pandas | moving average, EWM, decomposition, ACF/PACF, Ljung-Box |
| [`cds2.viz`](api/viz.md) | matplotlib | series, histogram, scatter, heatmap, spectrum, regression, confusion |
| [`cds2.io`](api/io.md) | pandas | CSV/JSON read-write, optional Excel/Parquet bridges, DataFrame summaries |
| [`cds2.calculus`](api/calculus.md) | NumPy/SciPy | derivative, complex-step gradient, jacobian, hessian, error propagation |
| [`cds2.special`](api/special.md) | scipy.special | gamma, erf, Bessels, elliptics, orthogonal polynomials, zeta |
| [`cds2.sparse`](api/sparse.md) | scipy.sparse.linalg | CG/GMRES/BiCGSTAB, Lanczos eigenpairs, truncated SVD |
| [`cds2.spectral`](api/spectral.md) | scipy.sparse | Laplacians, Fiedler vector, algebraic connectivity, spectral clustering |
| [`cds2.distributions`](api/distributions.md) | scipy.stats | pdf/cdf/ppf for twenty-plus distributions |

## Discovery and modelling

| Module | Built on | Purpose |
|---|---|---|
| [`cds2.infotheory`](api/infotheory.md) | NumPy | entropy, KL / Jensen-Shannon divergence, mutual information, permutation entropy |
| [`cds2.chaos`](api/chaos.md) | NumPy | delay embedding, Lyapunov exponents, correlation dimension, Hurst exponent |
| [`cds2.bayes`](api/bayes.md) | scipy.stats | conjugate updates, credible intervals, naive Bayes, Metropolis posteriors |
| [`cds2.metaheuristics`](api/metaheuristics.md) | NumPy | genetic algorithm, particle swarm optimization, simulated annealing |
| [`cds2.geometry`](api/geometry.md) | scipy.spatial | convex hull, closest pair, polygons, point-in-polygon, intersections |
| [`cds2.rl`](api/rl.md) | NumPy | bandits (epsilon-greedy, UCB1), tabular Q-learning, grid world |
| [`cds2.modeling`](api/modeling.md) | pure Python | expression trees, symbolic diff/integral, polynomial solving, MathModel |
| [`cds2.hypothesis`](api/hypothesis.md) | pure Python | heuristic hypothesis generation with confidence scores |
| [`cds2.knowledge`](api/knowledge.md) | pure Python | knowledge graph, notebook, ranked search |
| [`cds2.scientific`](api/scientific.md) | pure Python | physical constants, formulas and unit conversion |
| [`cds2.quantum`](api/quantum.md) | NumPy | statevector circuit simulator up to 16 qubits |
| [`cds2.nlp`](api/nlp.md) | NumPy | scalar autograd, BPE tokenizer, attention, mini-GPT forward pass |
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
