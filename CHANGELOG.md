# Changelog

All notable changes to **cognitive-discovery-system-v2** will be documented in
this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

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
