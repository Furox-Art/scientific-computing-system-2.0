## [v5.2.3] - 2026-09-05

Scientific-correctness and reproducibility hardening patch: the full PR #19 audit is now included in the published package.

### Changed

- Guided-fit reproducibility is hardened for duplicate dataset identities, manifest reruns, plot-output naming, outlier diagnostics and raw-input hashing.
- Numerical and input validation is strengthened across statistics, reliability, Monte Carlo/MCMC, epidemiology, linear algebra, graph, machine-learning and estimator paths.
- Reliability fitting handles censored data and availability edge cases more defensively.
- Native-extension fallbacks and validation behavior are hardened while preserving pure-Python fallback operation.

### Fixed

- Removed a redundant unreachable guided-fit manifest hash fallback instead of retaining dead defensive code.
- Removed the remaining invalid-regex-escape test warning without changing runtime behavior.
- Added focused regression coverage for scientific edge cases identified by the correctness audit.

### Validation

- PR #19 final CI passed Ruff lint/format, strict mypy, 1703 tests at 100.00% coverage, property tests, package/installed-wheel/CLI smoke on Ubuntu/Windows/macOS, the Python 3.10-3.13 OS matrix, and the benchmark regression gate.

## [v5.2.2] - 2026-09-05

Guided-fit reporting patch: long PDF reports are now paginated without silently truncating scientific results or report text.

### Changed

- Guided-fit PDF reports wrap long lines and automatically span as many pages as required.
- The previous fixed 12,000-character PDF report slice has been removed.
- Regression coverage verifies that content from the beginning, middle and end of long reports is preserved.

### Fixed

- Prevented long guided-fit PDF reports from silently dropping content beyond the previous single-page character limit.

## [v5.2.1] - 2026-09-05

Release-integrity patch: the cross-platform installed-wheel CLI validation merged after v5.2.0 is now part of a distinct release commit, and the release pipeline is fail-closed against tag/PyPI drift.

### Changed

- Installed-wheel CLI smoke validation is retained on Ubuntu, Windows and macOS for `cds2 info`, `guided-fit`, fit/residual artifacts and `guided-fit-rerun`.
- Release preflight now requires `main`, enforces lockstep package/changelog metadata, and refuses an existing PyPI version or Git tag instead of silently reusing it.
- After publication, the exact public PyPI version is clean-installed and CLI-smoke-tested on Ubuntu, Windows and macOS before the GitHub Release is created.
- GitHub Release creation now verifies that the resulting tag resolves to the exact workflow commit.

### Fixed

- Prevented a release-integrity failure mode where an already-published PyPI version could be skipped while a GitHub release/tag was created from newer repository code.

## [v5.2.0] - 2026-09-05

Guided-fit scientific validation release: stronger diagnostics, rerun stability checks and dataset-specific guidance extend the 5.1 workflow without changing its user-controlled decision model.

### Added

- Dedicated residual diagnostic plots in PNG and PDF for every guided fit.
- Quantified detected-outlier influence using the estimated percentage RMSE reduction from a diagnostic refit; detected points are still never removed silently.
- Manifest rerun stability checks that flag material changes in input hashes, fit RMSE, parameters or reliability labels.
- Dataset-specific model recommendations when one shared model is materially weaker for part of a multi-dataset analysis.
- Real-world scientific validation tests using the scikit-learn Diabetes and Linnerud datasets as independent packaged test data.

### Changed

- Installed-wheel CLI smoke tests now require residual plot artifacts in addition to fit plots and reproducibility manifests.
- Guided-fit CLI output, README and API documentation now expose outlier influence, residual diagnostics, stability warnings and separate-model recommendations.
- Supported package version advances from 5.1.0 to 5.2.0.

## [v5.1.0] - 2026-09-05

Guided scientific fitting release: user-controlled model recommendation, validated fitting workflows, reproducibility records, reporting and a fully tested command-line interface join the library.

### Added

- **`cds2.guided_fit`** - recommends one candidate model while leaving final model choice to the user; supports multiple datasets, measurement uncertainty, explicit missing-data policies and user-approved outlier handling.
- Repeated cross-validation, independent numerical cross-checks, parameter uncertainty, 95% confidence intervals and reliability labels.
- Matplotlib PNG/PDF fit diagnostics, reproducibility manifests and PDF/HTML/Markdown reports.
- **CLI** - `cds2 guided-fit` and `cds2 guided-fit-rerun`, with interactive and non-interactive operation.
- Release/package CI smoke-tests the installed wheel through fit -> artifacts/manifest -> rerun.

### Changed

- `cds2.optimize.curve_fit` exposes fit diagnostics and forwards weighting, bounds, solver method and Jacobian controls to SciPy.
- Supported package version advances from 5.0.0 to 5.1.0.

## [v5.0.0] - 2026-09-03

Performance, GPU, testing and ecosystem release: profiling/benchmark
infrastructure, compiled C kernels, an optional GPU backend,
property-based/fuzz/oracle tests, NumPy Array API compliance and
scikit-learn estimator interfaces join the library.

### Added

- **`cds2.prof`** - wall-clock, memory and CPU-time profiling with a
  `@timed` decorator, append-only JSONL benchmark history and
  tolerance-based regression gates (`pytest --regression`).
- **`cds2.bench`** - CLI report generator for CI regression runs.
- **C kernel extensions** (`src/cds2/src/`) - `solve_triangular` and
  `eigh_tridiag` (`_fast_linop`), RK4 step and batched trapezoid rule
  (`_fast_integrate`), 1-D convolution and SOS IIR filtering
  (`_fast_signal`); OpenMP on Linux, ARM64 NEON paths, serial fallback
  elsewhere.
- **`cds2.gpu`** (optional `gpu` extra) - lazy CuPy backend for linalg
  (solve, eigh, SVD, Cholesky), signal (FFT family, power spectrum)
  and Monte Carlo (pi estimate, MC integration, Metropolis-Hastings).
- **`cds2.array_api`** - NumPy Array API 2023.12 compliant namespace
  (reductions, elementwise math, FFT, linalg).
- **`cds2.estimator`** - scikit-learn compatible estimators
  (LinearRegressionGD, RidgeSGD, KMeansSKL, PCASKL).
- **Testing** - hypothesis property-based tests (`tests/property/`),
  CLI and API fuzz tests (`tests/fuzz/`), SciPy/NumPy ground-truth
  oracle comparisons (`tests/oracles/`); CI gains property-tests and
  regression-gate jobs.

### Changed

- Version 4.3.0 -> 5.0.0; new optional extras `test` (hypothesis,
  scikit-learn) and `gpu` (cupy-cuda12x).
- Test count ~1200 -> 1596; 100% blended coverage, mypy strict and
  ruff clean maintained.

## [v4.3.0] - 2026-09-03

Domain and audit release: Bayesian optimization and SDE ensembles join
the library, PDE and data-analysis ports land, facade modules are
documented and thin wrappers deprecated.

### Added

- **`cds2.bayesopt`** - Gaussian Process surrogate, expected
  improvement and UCB acquisition, Bayesian optimization loop.
- **`cds2.sde`** - Euler-Maruyama and Milstein SDE ensembles with
  ensemble statistics.
- **`cds2.pde`** - heat/wave 1-D/2-D solvers (FTCS/leapfrog,
  CFL-guarded, Dirichlet/Neumann) ported from v1 and accelerated.
- **`cds2.data_analysis`** - DataSet/DataFrame bridge with
  describe/summarize, group-by and NaN-aware helpers.
- **`tools/consistency_audit.py`** - automated facade-vs-native audit
  used to keep wrapper docs honest.

### Changed

- Docs: facade vs real-capability guidance added; `cds2.special` and
  `cds2.distributions` documented as convenience re-exports kept for
  their typed dataclass DX.
- `cds2.interpolate`: deprecated `scipy.interpolate.lagrange`
  replaced with `BarycentricInterpolator`.

### Fixed

- CLI polynomial solver checked the leading coefficient instead of
  the trailing one, accepting degenerate input.
- Three wrong results on weighted graphs corrected; oracle tests
  against networkx added.
- Extension modules marked `optional=True` so compiler-less installs
  fall back to pure NumPy/SciPy.

## [v4.2.0] - 2026-08-24

Ten-domain expansion release built with parallel agent orchestration:
wavelets, epidemiology, image processing, genetics, reliability, finance,
text analysis, game theory, combinatorial optimization and spatial
statistics join the library.

### Added

- **`cds2.wavelets`** - Haar DWT/IDWT, multi-level decomposition and
  MAD-thresholded wavelet denoising.
- **`cds2.epidemiology`** - SIR/SEIR RK4 simulation with conservation
  guarantees, herd-immunity threshold, final-size fixed-point solver.
- **`cds2.image`** - 2-D convolution (same/valid), Gaussian blur, Sobel edge
  detection with direction, mean/max/min pooling, binary morphology.
- **`cds2.genetics`** - GC content, Hamming distance, k-mers, reverse
  complement, Needleman-Wunsch global alignment with traceback, ORF finder
  with standard translation table.
- **`cds2.reliability`** - Kaplan-Meier survival curves with censoring,
  Weibull MLE fit, MTBF/availability, composite bathtub hazard model.
- **`cds2.finance`** - log/simple returns, Sharpe/Sortino ratios, maximum
  drawdown tracking, Black-Scholes pricing with greeks, historical and
  Monte Carlo VaR.
- **`cds2.text`** - tokenization, smoothed TF-IDF matrix, cosine/Jaccard
  similarity, top-k term summaries.
- **`cds2.game_theory`** - strictly dominated action elimination, pure Nash
  enumeration, zero-sum minimax via linear programming, iterated prisoner's
  dilemma tournaments with five classic strategies.
- **`cds2.combinatorial`** - nearest-neighbor TSP + 2-opt improvement, 0/1
  knapsack DP with reconstruction, optimal assignment wrapper, LCS.
- **`cds2.spatial`** - row-standardized weight builder, Moran's I and
  Geary's C autocorrelation with z-scores, Clark-Evans nearest-neighbor
  index.

### Changed

- Module count 32 -> 42 (+ CLI), export surface ~470 names.
- Test count 927 -> ~1200; 100% blended coverage, mypy strict and ruff clean
  maintained across all 48 source files.
## [v4.1.0] - 2026-08-24

Industrial-grade expansion: statistical process control, design of
experiments and graph community detection join the library, plus three new
benchmark races and project governance docs.

### Added

- **`cds2.quality`** - Shewhart X-bar chart (A2/R-bar limits), EWMA chart
  with time-varying limits, two-sided CUSUM, attribute p-chart for variable
  lot sizes, and Cp/Cpk capability indices with normal-tail defective PPM.
- **`cds2.design`** - full factorial designs, 2^k-p fractional factorials
  via defining-relation generators ("D=ABC"), Latin hypercube sampling,
  face-centred/rotatable central composite designs and coded-to-physical
  factor mapping.
- **`cds2.graph`** additions: seeded label-propagation community detection
  (`detect_communities`) returning Newman modularity (`modularity`) for
  partition quality.
- Benchmarks: entropy vs hand-rolled numpy, Latin hypercube vs
  scipy.stats.qmc, PSO vs scipy differential evolution.
- Governance: CONTRIBUTING.md, SECURITY.md and Dependabot configuration.

### Changed

- Test count 866 -> 940+; 100% blended coverage, mypy strict and ruff clean
  maintained across all 38 source files.
## [v4.0.0] - 2026-08-24

Identity and discovery release: the project is renamed to
**scientific-computing-system-2.0** (repository, PyPI distribution and CLI
branding) and six new domain modules join the library. Module count
24 -> 30 (+ CLI), export surface ~350 names.

### Added

- **`cds2.infotheory`** - Shannon/joint/conditional entropy, KL and
  Jensen-Shannon divergence, cross entropy, (normalized) mutual information
  and Bandt-Pompe permutation entropy.
- **`cds2.chaos`** - Takens delay embedding, false nearest neighbours,
  Rosenstein largest Lyapunov exponent, Grassberger-Procaccia correlation
  dimension, sample entropy, R/S Hurst exponent, logistic map iteration and
  generic bifurcation scans.
- **`cds2.bayes`** - Beta-Binomial / Normal-Normal / Gamma-Poisson conjugate
  updates, credible intervals with sampling fallback, Gaussian naive Bayes,
  Bayes factors and Metropolis posterior draws.
- **`cds2.metaheuristics`** - real-coded genetic algorithm (tournament
  selection, blend crossover, elitism), particle swarm optimization and
  simulated annealing with exponential cooling.
- **`cds2.geometry`** - convex hulls, closest pair via KD-tree, point-in-
  polygon ray casting, shoelace area/perimeter, segment intersection tests,
  infinite-line intersections, centroids and rotations.
- **`cds2.rl`** - Bernoulli multi-armed bandits (epsilon-greedy with decay,
  UCB1), tabular Q-learning and a deterministic grid-world environment.
- **`cds2.modeling`** additions: symbolic polynomial integration
  (`integrate`) and exact polynomial root finding (`polynomial_coefficients`,
  `solve_polynomial`).
- **`cds2.graph`** additions: Brandes betweenness centrality, Wasserman-Faust
  closeness centrality and Lanczos-backed eigenvector centrality with an
  oscillation guard for bipartite graphs.
- **`cds2.scientific`** additions: `convert_units` / `list_units` covering
  length, mass, time, energy, pressure, angle and temperature.

### Changed

- Package identity: PyPI distribution and repository renamed from
  `cognitive-discovery-system-v2` to `scientific-computing-system-2.0`.
  Import root remains `cds2`; the CLI remains `cds2`.
- Test count 856; 100% blended coverage, mypy strict and ruff clean
  maintained across all 36 source files.
- Tooling: pre-commit configuration and a cibuildwheel wheel-building
  workflow added; GitHub repository renamed.

## [v3.3.0] - 2026-08-23

Heritage-completion release: every module from the v1.x line now has a
home in cds2. Module count 17 -> 24 (+ CLI), export surface ~330 names.

### Added

- **`cds2.modeling`** - symbolic mathematics: expression trees with full
  operator overloading, symbolic differentiation (product/quotient/chain
  for constant exponents and bases), algebraic simplification rules,
  substitution, LaTeX export, Newton equation solving and least-squares
  parameter fitting via `MathModel`.
- **`cds2.quantum`** - dense statevector circuit simulator: X/Y/Z/H/S/T
  gates, RX/RY/RZ rotations, CNOT/CZ/SWAP, probabilities and seeded
  measurement sampling (up to 16 qubits).
- **`cds2.knowledge`** - `KnowledgeGraph` (typed relations, BFS shortest
  paths, transitive closure, cycle detection), `Notebook` with tags and
  concept links, and ranked `search` across concepts/relations/notes.
- **`cds2.scientific`** - CODATA physical constants plus mechanics,
  electromagnetism, thermodynamics and relativity formula helpers.
- **`cds2.hypothesis`** - heuristic hypothesis generation: trend,
  periodicity, outlier and pairwise-correlation hypotheses with confidence
  scores and domain tags.
- **`cds2.nlp`** - educational NLP toolkit: micrograd-style scalar
  autograd, trainable BPE tokenizer, scaled dot-product + multi-head
  attention, and a deterministic mini-GPT forward pass with sampling.

### Changed

- Test count 453 -> 643; 100% blended coverage and mypy strict maintained
  across all 30 source files.

## [v3.2.1] - 2026-08-22

Packaging fix: restore the PEP 561 py.typed marker that went missing from
the working tree, so type checkers pick up cds2 inline annotations again
in installed distributions. Full audit otherwise clean: 453 tests at 100%
blended coverage, mypy strict zero errors, ruff clean, docs strict build,
all four examples verified, wheel contents confirmed (py.typed + both C
kernels).

## [v3.2.0] - 2026-08-22

Multicore-kernel release: OpenMP-accelerated C kernels on Linux wheels,
GIL released during all hot loops, plus an industrial computing guide.

### Added

- **OpenMP parallel C kernels** (Linux wheels): KMeans assignment sweep
  and PageRank power iteration fan across cores via pragma-guided loops;
  both kernels drop the GIL during compute so embedded Python threads
  keep running. Windows/macOS wheels build the serial kernel silently.
- **docs/industrial.md** - end-to-end guide: preconditioned six-figure
  solves, process-parallel Monte Carlo, streaming statistics, out-of-core
  CSV processing.

### Changed

- setup.py enables -fopenmp only on Linux toolchains (MSVC legacy OpenMP
  rejects the kernel loop shapes; Apple clang needs external libomp).
- Sparse iterative results gained residual_norm diagnostics earlier this
  cycle are now exercised on every solver in the test suite.
# Changelog

All notable changes to **scientific-computing-system-2.0** will be documented in
this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [v3.1.0] - 2026-08-22

Industrial-tier release: preconditioned sparse solvers at six-figure scale,
process-parallel Monte Carlo and constant-memory streaming statistics.

### Added

- **`cds2.sparse.jacobi_preconditioner` / `ilu_preconditioner`** - SuperLU
  ILU wrapped as LinearOperator; all iterative solvers accept ``M``.
  Showcase: a 250k-unknown system with condition ~6e9 stalls plain CG yet
  converges routinely under ILU preconditioning.
- **Solver diagnostics** - iterative results report true residual norm.
- **`cds2.montecarlo.parallel_mc_integrate`** - process-parallel chunked
  integration with independent per-worker seeds.
- **`cds2.stats.StreamingStats`** - Welford incremental mean/variance,
  vectorized pushes + pairwise merge, constant memory.
- **`cds2.io.iter_csv`** - chunked reader generator for out-of-core data.

### Changed

- Test count 431 -> 453; gates held throughout.

## [v3.0.0] - 2026-08-22

Flagship surface expansion.

### Added

- **`cds2.distributions`** - 57 functions across 19 probability families
  (t, chi2, F, exponential, uniform, lognormal, Poisson, binomial, gamma,
  beta, Weibull, Cauchy, Laplace, Gumbel, Pareto, Rayleigh, geometric,
  negative-binomial, hypergeometric), each with pdf/pmf + cdf (+ ppf).
- **Special functions doubled to 42** - digamma, Fresnel C/S, Airy Ai/Bi,
  Legendre Pn, elliptic K/E, exp1, hypergeometric 2F1, spherical Bessels,
  Bessel jv/yv/iv/kv, Hankel1, Struve H0/H1, Chebyshev T/U, Laguerre,
  Hermite, Jacobi, spherical harmonics, Lambert W, Faddeeva w, E_n, Si/Ci.
- **`examples/`** - four runnable end-to-end case studies: signal
  denoising, Michaelis-Menten fitting, Bayesian MCMC inference,
  citation-network PageRank + spectral clustering.

## [v2.6.0] - 2026-08-22

Scientific-depth release closing the biggest coverage gaps of the platform:
a full distributions module, a 2x special-functions expansion and runnable
case studies.

### Added

- **`cds2.distributions`**: 24 functions across eight probability
  distributions (Student-t, chi-squared, F, exponential, uniform,
  lognormal, Poisson, binomial) with pdf/pmf, cdf and ppf for each.
- **Special functions doubled**: digamma, Fresnel integrals, Airy Ai/Bi,
  Legendre polynomials, complete elliptic integrals K/E, exponential
  integral E1, Gauss hypergeometric 2F1, spherical Bessels j0/j1.
- **`examples/`**: four runnable end-to-end case studies:
  - signal denoising (Butterworth + Welch verification)
  - Michaelis-Menten experiment fitting with residual inference
  - Bayesian sensor-bias MCMC vs analytic conjugate posterior
  - citation-network PageRank + spectral clustering + DAG validation

### Changed

- Module count 16 -> 17; flat exports ~213; test count grew again with
  distribution and special-function suites.

## [v2.5.1] - 2026-08-22

Quality-gate release: strict typing and full coverage are now enforced by CI.
No API or behavior changes; 265 -> 344 tests.

### Added

- **mypy `--strict` gate**: zero errors across all 19 source files,
  enforced by a dedicated CI job (`types`). SciPy/Matplotlib/openpyxl
  handled via documented module overrides; pandas typed through stubs.
- **100% blended coverage gate**: statement + branch coverage at 100%
  enforced via `--cov-fail-under=100` on the reference cell. Coverage rose
  from 88%: ~80 gap-closing tests including forced NumPy-fallback KMeans
  paths, pagerank fallback arcs, empty-cluster rescue and every validation
  guard.

### Changed

- dev extras gained openpyxl, pyarrow, pandas-stubs and a numpy<2.5 cap
  (dev-only; mypy target 3.10 cannot parse numpy>=2.5 stubs).
- `pacf` dropped an always-true guard found while chasing the last branch.

## [v2.5.0] - 2026-08-22

Flagship release: two new scientific modules plus four major capability
upgrades across the platform. Module count now 16 + CLI, ~165 flat exports,
265 tests.

### Added

- **`cds2.sparse`**: large-scale sparse linear algebra: conjugate gradient,
  GMRES (with restart) and BiCGSTAB iterative solvers, Lanczos eigenpairs
  (`largest_eigenpairs` / `smallest_eigenpairs`) and truncated SVD.
- **`cds2.spectral`**: spectral graph theory: combinatorial and
  normalized Laplacians, Fiedler vectors, algebraic connectivity and
  spectral clustering (eigendecomposition embedding + k-means).
- **`cds2.montecarlo.metropolis_hastings`**: seeded random-walk MH sampler
  with burn-in, thinning and acceptance-rate diagnostics.
- **`cds2.optimize.minimize_constrained`**: SLSQP-based constrained
  minimization with SciPy-dict equality/inequality constraints.
- **`cds2.calculus.propagate_error`**: first-order uncertainty propagation
  through arbitrary functions via the Jacobian.
- **`cds2.integrate.solve_bvp`**: two-point boundary value problems via
  4th-order collocation.

### Changed

- `largest_eigenpairs` selects algebraically largest eigenvalues (LA).
- 28 new tests this cycle (236 -> 265).

## [v2.4.0] - 2026-08-22

Scientific-computing depth release: scattered-data RBF interpolation, ODE
event detection, stiff solvers and global optimization.

### Added

- **`cds2.interpolate.rbf_interp`**: radial-basis-function interpolation
  for scattered N-D data (thin-plate-spline default), with `smoothing` for
  approximating fits and `neighbors` kNN mode for large problems.
- **ODE events**: `cds2.integrate.solve_ivp` now accepts `events`
  (zero-crossing callables, `terminal = True` honored) and returns
  `t_events` / `y_events` on the result.
- **Stiff-solver documentation path**: `method="Radau" | "BDF" | "LSODA"`
  documented and tested with a stiff decay system.
- **`cds2.optimize.differential_evolution`**: stochastic global minimizer
  over box constraints returning `GlobalResult` (x, fun, nit, nfev).
- 9 new tests (236 total).

## [v2.3.0] - 2026-08-22

Scientific-surface expansion: two new modules and modern resampling-based
inference.

### Added

- **`cds2.calculus`**: numerical differentiation: `derivative` (central /
  forward / backward with adaptive steps), `complex_step_gradient`
  (machine-precision gradients via the complex-step trick), `jacobian`
  (finite-difference, R^n -> R^m) and `hessian` (central differences with
  exact-symmetric mixed partials).
- **`cds2.special`**: special functions: gamma/gammaln, erf/erfc/erfinv,
  beta/betaln, Bessel j0/j1/y0, Riemann-Hurwitz zeta.
- **`cds2.stats.bootstrap_ci`**: percentile bootstrap confidence intervals
  for arbitrary statistics, seeded and vectorized.
- **`cds2.stats.permutation_test`**: two-sided permutation test on mean
  differences with the +1 corrected p-value.
- **`cds2.linalg.expm/logm/sqrtm`**: matrix exponential, logarithm and
  principal square root.
- 40+ new tests (227 total); docs pages for both new modules.

### Changed

- Flat exports grew to ~150 symbols; module count now 14 plus the CLI.

## [v2.2.0] - 2026-08-22

Second compiled-acceleration release: PageRank joins the C kernel family,
and the specialist win column widens.

### Added

- **`cds2._fast_pagerank`**: C extension running the full power iteration
  over transposed-CSR arrays (buffer protocol, no build-time deps). The
  Python side now builds that structure with plain vectorized NumPy
  (argsort + searchsorted) instead of SciPy multiply/transpose round-trips.
  Graceful SciPy-sparse fallback retained.

### Changed

- **PageRank vs NetworkX: 1.35x slower -> 0.18x: about 5x faster** than
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

- **`cds2._fast_kmeans`**: a from-scratch C extension (buffer-protocol API,
  no build-time NumPy headers) implementing the Lloyd iteration loop:
  fused assignment/update, empty-cluster relocation, convergence tracking.
- **Compiled-wheel release pipeline**: cibuildwheel builds wheels for
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
  - `cds2.linalg`: solve, det, inv, pinv, eig/eigh, SVD, least squares,
    cholesky, norms, trace, matrix power, rank, condition number
  - `cds2.stats`: descriptive statistics, t-tests (one-sample, independent,
    Welch, paired), ANOVA, Kruskal-Wallis, Mann-Whitney U, Wilcoxon,
    normality test, Pearson/Spearman/Kendall correlations, chi-square
    independence, effect sizes (Cohen's d, eta-squared, Cramer's V),
    percentiles, z-scores, normal pdf/cdf/ppf
  - `cds2.optimize`: minimize, scalar minimization, root finding
    (brentq/newton/systems), linear programming, nonlinear least squares,
    curve fitting
  - `cds2.integrate`: quad/dblquad/triple integration, ODE solving via
    `solve_ivp`, trapezoid/simpson rules, cumulative integration
  - `cds2.interpolate`: linear/cubic/pchip interpolation, Lagrange
    polynomials, scattered-data gridding, regular-grid interpolation
  - `cds2.signals`: FFT family, periodogram/Welch/spectrogram, Butterworth
    low/high/band-pass filters, peak finding, convolution/correlation,
    Hilbert envelope, resampling, detrending
  - `cds2.montecarlo`: seeded pi estimation, 1-D Monte Carlo integration,
    expectation estimation, hit-or-miss area estimation
  - `cds2.graph`: adjacency builders, connected components, Dijkstra /
    Bellman-Ford / Floyd-Warshall shortest paths, minimum spanning tree,
    degrees, topological order, and **PageRank** (power iteration)
- **New domain modules**:
  - `cds2.ml`: LinearRegression, LogisticRegression, KMeans (k-means++
    seeding), PCA, KNeighborsClassifier, StandardScaler, train/test split,
    synthetic data generators, classification/regression metrics
  - `cds2.timeseries`: moving average, exponential smoothing, differencing,
    classical seasonal decomposition, ACF/PACF, Ljung-Box test
  - `cds2.viz`: matplotlib helpers: series, histogram, scatter, heatmap,
    spectrum, regression overlay, confusion matrix
  - `cds2.io`: pandas-backed CSV/JSON readers/writers plus optional
    Excel/Parquet bridges and a DataFrame summarizer
- **CLI**: `cds2 info | stats | integrate | linsolve | plot`
- **CI**: GitHub Actions matrix (Ubuntu/Windows/macOS x Python 3.10-3.13)
  with ruff + pytest

### Changed

- Runtime dependencies are now explicit: numpy, scipy, pandas, matplotlib.
  The zero-dependency philosophy of v1.x is intentionally retired in favor of
  a faster, richer stack. The v1 line remains maintained separately at its own
  repository.
