# Changelog

All notable changes to **cognitive-discovery-system-v2** will be documented in
this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [v2.5.0] - 2026-08-22

Flagship release: two new scientific modules plus four major capability
upgrades across the platform. Module count now 16 + CLI, ~165 flat exports,
265 tests.

### Added

- **`cds2.sparse`** — large-scale sparse linear algebra: conjugate gradient,
  GMRES (with restart) and BiCGSTAB iterative solvers, Lanczos eigenpairs
  (`largest_eigenpairs` / `smallest_eigenpairs`) and truncated SVD.
- **`cds2.spectral`** — spectral graph theory: combinatorial and
  normalized Laplacians, Fiedler vectors, algebraic connectivity and
  spectral clustering (eigendecomposition embedding + k-means).
- **`cds2.montecarlo.metropolis_hastings`** — seeded random-walk MH sampler
  with burn-in, thinning and acceptance-rate diagnostics.
- **`cds2.optimize.minimize_constrained`** — SLSQP-based constrained
  minimization with SciPy-dict equality/inequality constraints.
- **`cds2.calculus.propagate_error`** — first-order uncertainty propagation
  through arbitrary functions via the Jacobian.
- **`cds2.integrate.solve_bvp`** — two-point boundary value problems via
  4th-order collocation.

### Changed

- `largest_eigenpairs` selects algebraically largest eigenvalues (LA).
- 28 new tests this cycle (236 -> 265).

## [v2.4.0] - 2026-08-22

Scientific-computing depth release: scattered-data RBF interpolation, ODE
event detection, stiff solvers and global optimization.

### Added

- **`cds2.interpolate.rbf_interp`** — radial-basis-function interpolation
  for scattered N-D data (thin-plate-spline default), with `smoothing` for
  approximating fits and `neighbors` kNN mode for large problems.
- **ODE events** — `cds2.integrate.solve_ivp` now accepts `events`
  (zero-crossing callables, `terminal = True` honored) and returns
  `t_events` / `y_events` on the result.
- **Stiff-solver documentation path** — `method="Radau" | "BDF" | "LSODA"`
  documented and tested with a stiff decay system.
- **`cds2.optimize.differential_evolution`** — stochastic global minimizer
  over box constraints returning `GlobalResult` (x, fun, nit, nfev).
- 9 new tests (236 total).

## [v2.3.0] - 2026-08-22

Scientific-surface expansion: two new modules and modern resampling-based
inference.

### Added

- **`cds2.calculus`** — numerical differentiation: `derivative` (central /
  forward / backward with adaptive steps), `complex_step_gradient`
  (machine-precision gradients via the complex-step trick), `jacobian`
  (finite-difference, R^n -> R^m) and `hessian` (central differences with
  exact-symmetric mixed partials).
- **`cds2.special`** — special functions: gamma/gammaln, erf/erfc/erfinv,
  beta/betaln, Bessel j0/j1/y0, Riemann-Hurwitz zeta.
- **`cds2.stats.bootstrap_ci`** — percentile bootstrap confidence intervals
  for arbitrary statistics, seeded and vectorized.
- **`cds2.stats.permutation_test`** — two-sided permutation test on mean
  differences with the +1 corrected p-value.
- **`cds2.linalg.expm/logm/sqrtm`** — matrix exponential, logarithm and
  principal square root.
- 40+ new tests (227 total); docs pages for both new modules.

### Changed

- Flat exports grew to ~150 symbols; module count now 14 plus the CLI.

## [v2.2.0] - 2026-08-22

Second compiled-acceleration release: PageRank joins the C kernel family,
and the specialist win column widens.

### Added

- **`cds2._fast_pagerank`** — C extension running the full power iteration
  over transposed-CSR arrays (buffer protocol, no build-time deps). The
  Python side now builds that structure with plain vectorized NumPy
  (argsort + searchsorted) instead of SciPy multiply/transpose round-trips.
  Graceful SciPy-sparse fallback retained.

### Changed

- **PageRank vs NetworkX: 1.35x slower -> 0.18x — about 5x faster** than
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

- **`cds2._fast_kmeans`** — a from-scratch C extension (buffer-protocol API,
  no build-time NumPy headers) implementing the Lloyd iteration loop:
  fused assignment/update, empty-cluster relocation, convergence tracking.
- **Compiled-wheel release pipeline** — cibuildwheel builds wheels for
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
[cognitive-discovery-system](https://github.com/Furox88/cognitive-discovery-system)
(v1.x) form the foundation; v2 rebuilds them on NumPy/SciPy for speed and adds
new domain modules on top.

### Added

- **Accelerated core** (NumPy / SciPy backed):
  - `cds2.linalg` — solve, det, inv, pinv, eig/eigh, SVD, least squares,
    cholesky, norms, trace, matrix power, rank, condition number
  - `cds2.stats` — descriptive statistics, t-tests (one-sample, independent,
    Welch, paired), ANOVA, Kruskal-Wallis, Mann-Whitney U, Wilcoxon,
    normality test, Pearson/Spearman/Kendall correlations, chi-square
    independence, effect sizes (Cohen's d, eta-squared, Cramer's V),
    percentiles, z-scores, normal pdf/cdf/ppf
  - `cds2.optimize` — minimize, scalar minimization, root finding
    (brentq/newton/systems), linear programming, nonlinear least squares,
    curve fitting
  - `cds2.integrate` — quad/dblquad/triple integration, ODE solving via
    `solve_ivp`, trapezoid/simpson rules, cumulative integration
  - `cds2.interpolate` — linear/cubic/pchip interpolation, Lagrange
    polynomials, scattered-data gridding, regular-grid interpolation
  - `cds2.signals` — FFT family, periodogram/Welch/spectrogram, Butterworth
    low/high/band-pass filters, peak finding, convolution/correlation,
    Hilbert envelope, resampling, detrending
  - `cds2.montecarlo` — seeded pi estimation, 1-D Monte Carlo integration,
    expectation estimation, hit-or-miss area estimation
  - `cds2.graph` — adjacency builders, connected components, Dijkstra /
    Bellman-Ford / Floyd-Warshall shortest paths, minimum spanning tree,
    degrees, topological order, and **PageRank** (power iteration)
- **New domain modules**:
  - `cds2.ml` — LinearRegression, LogisticRegression, KMeans (k-means++
    seeding), PCA, KNeighborsClassifier, StandardScaler, train/test split,
    synthetic data generators, classification/regression metrics
  - `cds2.timeseries` — moving average, exponential smoothing, differencing,
    classical seasonal decomposition, ACF/PACF, Ljung-Box test
  - `cds2.viz` — matplotlib helpers: series, histogram, scatter, heatmap,
    spectrum, regression overlay, confusion matrix
  - `cds2.io` — pandas-backed CSV/JSON readers/writers plus optional
    Excel/Parquet bridges and a DataFrame summarizer
- **CLI**: `cds2 info | stats | integrate | linsolve | plot`
- **CI**: GitHub Actions matrix (Ubuntu/Windows/macOS x Python 3.10-3.13)
  with ruff + pytest

### Changed

- Runtime dependencies are now explicit: numpy, scipy, pandas, matplotlib.
  The zero-dependency philosophy of v1.x is intentionally retired in favor of
  a faster, richer stack. The v1 line remains maintained separately at its own
  repository.
