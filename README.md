# scientific-computing-system-2.0

<p align="center">
  <img src="https://raw.githubusercontent.com/Furox-Art/scientific-computing-system-2.0/main/docs/assets/promo_hero.png" alt="scientific-computing-system-2.0 scientific computing platform" width="100%">
</p>

[![CI](https://github.com/Furox-Art/scientific-computing-system-2.0/actions/workflows/tests.yml/badge.svg)](https://github.com/Furox-Art/scientific-computing-system-2.0/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/scientific-computing-system-2.0)](https://pypi.org/project/scientific-computing-system-2.0/)
[![Python](https://img.shields.io/pypi/pyversions/scientific-computing-system-2.0)](https://pypi.org/project/scientific-computing-system-2.0/)
[![npm](https://img.shields.io/npm/v/scientific-computing-system-2.0)](https://www.npmjs.com/package/scientific-computing-system-2.0)

This is the **NumPy build** of [scientific-computing-system](https://github.com/Furox-Art/scientific-computing-system), not a separate product: pick this one for NumPy/SciPy/pandas/matplotlib, the other for readable pure Python with zero runtime dependencies.

**52 importable names, 535 public exports, 1,805 tests, 100% branch coverage, MIT.** Every count here is generated from the tree by `scripts/make_promo.py` and CI, not typed by hand.

## Installation

Python 3.10+. [PyPI is the install path](https://pypi.org/project/scientific-computing-system-2.0/):

```bash
pip install scientific-computing-system-2.0
```

| Extra | Adds | Use when |
|---|---|---|
| `[dev]` | pytest, ruff, mypy, hypothesis, scikit-learn, networkx, pandas-stubs | contributing |
| `[test]` | pytest, pytest-cov, hypothesis, scikit-learn | running the suite only |
| `[docs]` | mkdocs, mkdocs-material, mkdocstrings | building the site |
| `[gpu]` | cupy-cuda12x | using the optional `cds2.gpu` backend |
| `[all]` | all of the above | everything |

From a clone: `pip install -e ".[dev]"`.

**Optional C kernels.** `_fast_kmeans` and `_fast_pagerank` are compiled when a
toolchain is available and **fall back silently** to pure NumPy otherwise, so
`CDS_PURE=1 pip install .` is fully functional but slower on those two
operations.

**npm (optional).** A same-named npm package is a **thin Node launcher shim** —
five files, ~8 KB, no Python — that calls `python -m cds2.cli`. You still need
the PyPI install, so most people can skip it: [docs/npm.md](https://furox-art.github.io/scientific-computing-system-2.0/npm/).

**Supply chain: no signed provenance.** No published release carries a
provenance attestation, on **either** registry: `pypi.org/integrity/...` returns
`404` and the npm attestation endpoint returns `404`, verified for the current
`5.2.6` and for `5.2.5`. Pin digests instead. What the project does guarantee
(reproducible wheel, an sdist that really ships the native kernels, version
lockstep, cross-platform smoke) and what a consumer can check:
[Supply chain](https://furox-art.github.io/scientific-computing-system-2.0/supply-chain/).

## Quick start

Executed against `scientific-computing-system-2.0==5.2.5`:

```python
import cds2

print(cds2.linalg.solve([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0]))  # [2. 3.]
print(cds2.montecarlo.pi_estimate(n=100_000, seed=42))  # 3.13776
print(cds2.infotheory.entropy([0.25, 0.25, 0.25, 0.25]))  # 2.0
print(cds2.graph.pagerank(cds2.graph.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])))
# [0.1375043  0.19594362 0.19594362 0.47060846]
```

## Guided fitting: the differentiating feature

`cds2.guided_fit` turns a CSV into a reproducible, inspectable model fit. Unlike
a bare `curve_fit`, the model is **recommended and then confirmed by you**, and
the run is written to a replayable manifest.

```bash
cds2 guided-fit data.csv --x time --y response \
    --model exponential --missing drop --outliers exclude --report pdf
cds2 guided-fit-rerun guided-fit-results/guided_fit_manifest.json
```

Five model families (`linear`, `quadratic`, `exponential`, `power`, `logistic`),
explicit missing-data (`ask`/`drop`/`interpolate`) and outlier (`ask`/`keep`/`exclude`)
policies, PDF/HTML/Markdown reports, and a manifest carrying the input hash so a
rerun proves it used the same data. See [case studies](https://furox-art.github.io/scientific-computing-system-2.0/case-studies/).

## Command line

All ten subcommands, from `cds2 --help`:

```text
cds2 info                      show version information
cds2 stats 1,2,3,4,5           descriptive statistics
cds2 integrate sin --a 0 --b 3.14159
cds2 linsolve --a "3,1;1,2" --b "9,8"
cds2 entropy "0.25,0.25,0.25,0.25"
cds2 units 5 --from-unit km --to-unit mile
cds2 solve --coeffs "1,-5,6"
cds2 plot 1,3,2,5,4 --file out.png
cds2 guided-fit data.csv --x time --y response
cds2 guided-fit-rerun guided-fit-results/guided_fit_manifest.json
```

`python -m cds2` works and is equivalent to the `cds2` console script, so either
invocation is fine:

```bash
cds2 info
python -m cds2 info
```

## Modules

46 flat modules plus 6 subpackages; 47 generated API pages. `*` marks [deprecated](https://furox-art.github.io/scientific-computing-system-2.0/deprecated/) modules.

| Module | What it does | Entry points |
|---|---|---|
| [`cds2.linalg`](https://furox-art.github.io/scientific-computing-system-2.0/api/linalg/) | Dense linear algebra, typed results | `solve`, `det`, `svd`, `eigh`, `lstsq`, `expm` |
| [`cds2.stats`](https://furox-art.github.io/scientific-computing-system-2.0/api/stats/) | Tests, summaries, effect sizes | `independent_t_test`, `permutation_test`, `describe`, `bootstrap_ci` |
| [`cds2.optimize`](https://furox-art.github.io/scientific-computing-system-2.0/api/optimize/) | Optimizers and root finding | `minimize`, `curve_fit`, `find_root_scalar`, `linprog` |
| [`cds2.integrate`](https://furox-art.github.io/scientific-computing-system-2.0/api/integrate/) | Quadrature and ODEs | `quad`, `integrate_2d`, `solve_ivp`, `solve_bvp`, `simpson` |
| [`cds2.interpolate`](https://furox-art.github.io/scientific-computing-system-2.0/api/interpolate/) | Interpolation | `linear_interp`, `cubic_spline`, `pchip_interpolator`, `rbf_interp` |
| [`cds2.signals`](https://furox-art.github.io/scientific-computing-system-2.0/api/signals/) | FFT, spectra, filters | `power_spectrum`, `welch_spectrum`, `butter_lowpass`, `find_peaks` |
| [`cds2.sparse`](https://furox-art.github.io/scientific-computing-system-2.0/api/sparse/) | Sparse solvers, CDS preconditioners | `solve_cg`, `solve_gmres`, `truncated_svd`, `ilu_preconditioner` |
| [`cds2.spectral`](https://furox-art.github.io/scientific-computing-system-2.0/api/spectral/) | Spectral graph theory | `laplacian`, `fiedler_vector`, `algebraic_connectivity`, `spectral_cluster` |
| [`cds2.calculus`](https://furox-art.github.io/scientific-computing-system-2.0/api/calculus/) | Derivatives, Jacobians, Hessians | `derivative`, `jacobian`, `hessian`, `propagate_error` |
| [`cds2.ml`](https://furox-art.github.io/scientific-computing-system-2.0/api/ml/) | Classical ML and metrics | `LinearRegression`, `KMeans`, `PCA`, `KNeighborsClassifier`, `r2_score` |
| [`cds2.timeseries`](https://furox-art.github.io/scientific-computing-system-2.0/api/timeseries/) | pandas-backed series analysis | `seasonal_decompose`, `exponential_smoothing`, `acf`, `ljung_box` |
| [`cds2.montecarlo`](https://furox-art.github.io/scientific-computing-system-2.0/api/montecarlo/) | Seeded estimation | `pi_estimate`, `mc_integrate`, `metropolis_hastings`, `parallel_mc_integrate` |
| [`cds2.graph`](https://furox-art.github.io/scientific-computing-system-2.0/api/graph/) | Graphs and PageRank | `from_edges`, `pagerank`, `connected_components`, `modularity` |
| [`cds2.io`](https://furox-art.github.io/scientific-computing-system-2.0/api/io/) | pandas I/O and summaries | `read_csv`, `write_csv`, `summarize`, `iter_csv` |
| [`cds2.viz`](https://furox-art.github.io/scientific-computing-system-2.0/api/viz/) | Matplotlib helpers | `plot_series`, `plot_heatmap`, `plot_regression`, `plot_confusion_matrix` |
| [`cds2.data_analysis`](https://furox-art.github.io/scientific-computing-system-2.0/api/data_analysis/) | DataFrame bridge | `DataSet`, `to_dataframe`, `from_dataframe` |
| [`cds2.bayes`](https://furox-art.github.io/scientific-computing-system-2.0/api/bayes/) | Conjugate updates, naive Bayes | `beta_binomial_update`, `NaiveBayes`, `posterior_interval` |
| [`cds2.bayesopt`](https://furox-art.github.io/scientific-computing-system-2.0/api/bayesopt/) | GP-surrogate optimization | `bayes_opt`, `GaussianProcess`, `expected_improvement` |
| [`cds2.infotheory`](https://furox-art.github.io/scientific-computing-system-2.0/api/infotheory/) | Entropy and dependence | `entropy`, `mutual_information`, `kl_divergence`, `permutation_entropy` |
| [`cds2.chaos`](https://furox-art.github.io/scientific-computing-system-2.0/api/chaos/) | Nonlinear dynamics | `delay_embed`, `largest_lyapunov_exponent`, `hurst_exponent`, `bifurcation_scan` |
| [`cds2.metaheuristics`](https://furox-art.github.io/scientific-computing-system-2.0/api/metaheuristics/) | Population and annealing | `genetic_minimize`, `pso_minimize`, `simulated_annealing` |
| [`cds2.geometry`](https://furox-art.github.io/scientific-computing-system-2.0/api/geometry/) | Hulls, distances, polygons | `convex_hull`, `closest_pair`, `point_in_polygon`, `segments_intersect` |
| [`cds2.rl`](https://furox-art.github.io/scientific-computing-system-2.0/api/rl/) | Bandits and tabular RL | `ucb1`, `epsilon_greedy`, `q_learn`, `GridWorld` |
| [`cds2.quality`](https://furox-art.github.io/scientific-computing-system-2.0/api/quality/) | Statistical process control | `xbar_chart`, `ewma_chart`, `cusum_chart`, `process_capability` |
| [`cds2.design`](https://furox-art.github.io/scientific-computing-system-2.0/api/design/) | Design of experiments | `full_factorial`, `latin_hypercube`, `central_composite` |
| [`cds2.reliability`](https://furox-art.github.io/scientific-computing-system-2.0/api/reliability/) | Survival analysis | `kaplan_meier`, `weibull_fit`, `mtbf`, `availability` |
| [`cds2.modeling`](https://furox-art.github.io/scientific-computing-system-2.0/api/modeling/) | Symbolic maths, pure Python | `Expression`, `diff`, `integrate`, `solve_polynomial`, `to_latex` |
| [`cds2.hypothesis`](https://furox-art.github.io/scientific-computing-system-2.0/api/hypothesis/) | Heuristic hypothesis generation | `HypothesisEngine`, `Hypothesis`, `Domain` |
| [`cds2.knowledge`](https://furox-art.github.io/scientific-computing-system-2.0/api/knowledge/) | Concept graphs and retrieval | no public names yet |
| [`cds2.scientific`](https://furox-art.github.io/scientific-computing-system-2.0/api/scientific/) | CODATA constants and formulas, CDS-native | `CONSTANTS`, `speed_of_light`, `photon_energy`, `convert_units` |
| [`cds2.quantum`](https://furox-art.github.io/scientific-computing-system-2.0/api/quantum/) | Dense statevector circuits | `QuantumCircuit`, `GATES` |
| [`cds2.genetics`](https://furox-art.github.io/scientific-computing-system-2.0/api/genetics/) | DNA composition and alignment | `gc_content`, `kmer_counts`, `global_align`, `find_orfs` |
| [`cds2.epidemiology`](https://furox-art.github.io/scientific-computing-system-2.0/api/epidemiology/) | SIR/SEIR trajectories | `simulate_sir`, `simulate_seir`, `herd_immunity_threshold` |
| [`cds2.finance`](https://furox-art.github.io/scientific-computing-system-2.0/api/finance/) | Returns, drawdown, risk | `log_returns`, `max_drawdown`, `sharpe_ratio`, `black_scholes` |
| [`cds2.game_theory`](https://furox-art.github.io/scientific-computing-system-2.0/api/game_theory/) | Nash, dominance, zero-sum | `pure_nash_equilibria`, `iterated_elimination`, `zero_sum_mixed` |
| [`cds2.combinatorial`](https://furox-art.github.io/scientific-computing-system-2.0/api/combinatorial/) | TSP, knapsack, assignment | `assign_min_cost`, `knapsack_01`, `nearest_neighbor_tsp`, `two_opt` |
| [`cds2.spatial`](https://furox-art.github.io/scientific-computing-system-2.0/api/spatial/) | Contiguity and point patterns | `build_weight_matrix`, `morans_i`, `gearys_c` |
| [`cds2.text`](https://furox-art.github.io/scientific-computing-system-2.0/api/text/) | Tokenization and similarity | `tokenize`, `tfidf_matrix`, `cosine_similarity`, `summarize_terms` |
| [`cds2.image`](https://furox-art.github.io/scientific-computing-system-2.0/api/image/) | Grayscale filters, morphology | `convolve2d`, `gaussian_blur`, `sobel_edges`, `binarize` |
| [`cds2.wavelets`](https://furox-art.github.io/scientific-computing-system-2.0/api/wavelets/) | Haar decomposition | `haar_dwt`, `haar_idwt`, `dwt_levels`, `wavelet_denoise` |
| [`cds2.pde`](https://furox-art.github.io/scientific-computing-system-2.0/api/pde/) | Vectorised FTCS and leapfrog | `heat_equation_1d`, `wave_equation_1d`, `solve_heat` |
| [`cds2.sde`](https://furox-art.github.io/scientific-computing-system-2.0/api/sde/) | Stochastic differential equations | `sde_euler_maruyama`, `sde_milstein`, `ensemble_stats` |
| [`cds2.guided_fit`](https://furox-art.github.io/scientific-computing-system-2.0/api/guided_fit/) | Guided reproducible fitting | `run_guided_fit`, `recommend_model`, `save_manifest`, `rerun_manifest` |
| [`cds2.cli`](https://furox-art.github.io/scientific-computing-system-2.0/api/cli/) | argparse front end | `main`, the `cds2` console script |
| [`cds2.special`](https://furox-art.github.io/scientific-computing-system-2.0/api/special/)* | scipy.special wrapper | 41 aliases; use scipy directly |
| [`cds2.distributions`](https://furox-art.github.io/scientific-computing-system-2.0/api/distributions/)* | scipy.stats wrapper | 56 pdf/cdf/ppf aliases; use scipy directly |
| `cds2.array_api` | Array API 2023.12 namespace | `matmul`, `cholesky`, `std`, 21 names; no API page |
| [`cds2.nlp`](https://furox-art.github.io/scientific-computing-system-2.0/api/nlp/) | BPE tokenizer, autograd, attention, mini-GPT | `BPETokenizer`, `multi_head_attention`, `TinyGPT` |
| `cds2.estimator` | scikit-learn compatible estimators | `LinearRegressionGD`, `KMeansSKL`, `PCASKL`, `RidgeSGD` |
| `cds2.bench` | Benchmark history and regression | `run_regression_check`, `RegressionReport` |
| `cds2.prof` | Profiling, history, regression gates | `profile`, `timed`, `BenchHistory`, `RegressionGate` |
| `cds2.gpu` | Optional CuPy backend | `is_available`, `synchronize`, `cupy` |

## Examples

20 runnable scripts in [`examples/`](https://github.com/Furox-Art/scientific-computing-system-2.0/tree/main/examples), all executed by the `examples-smoke` CI job on every push. Try [`bayesian_inference.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/bayesian_inference.py) (Metropolis-Hastings vs an analytic conjugate result), [`experiment_fitting.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/experiment_fitting.py) (Michaelis-Menten fit, permutation test on residuals) and [`signal_denoising.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/signal_denoising.py) (Butterworth filter, Welch spectra). Catalogue: [Examples](https://furox-art.github.io/scientific-computing-system-2.0/examples/) · [Recipes](https://furox-art.github.io/scientific-computing-system-2.0/recipes/).

## Performance, measured honestly

`benchmarks/results.json` and [docs/benchmarks.md](https://furox-art.github.io/scientific-computing-system-2.0/benchmarks/) are a real run, but a **stale one: cds2 3.0.0, commit `6ea5a02`, 2026-08-22**, while the release is 5.2.5. Read them as "3.0.0 versus the baseline libraries on one machine", not as a claim about today.

Of the 13 measured cases, **CDS v2 was slower in 4**: `dataframe summary` 1.82x, `solve 800x800` 1.22x, `solve 8x8 x300` 1.17x and `welch` 1.03x. The pandas row returns strictly more information (per-column nulls and uniques), so part of that premium is extra work. The clearest win is PageRank at 0.15x versus NetworkX. There is **no blanket speedup claim**, and [benchmark-methodology.md](https://furox-art.github.io/scientific-computing-system-2.0/benchmark-methodology/) sets out the limits.

## Documentation map

| Page | What is in it |
|---|---|
| [Getting started](https://furox-art.github.io/scientific-computing-system-2.0/getting-started/) | install plus a ten-minute tour, one section per domain |
| [Modules overview](https://furox-art.github.io/scientific-computing-system-2.0/modules/) | every module with its backing library |
| [Examples](https://furox-art.github.io/scientific-computing-system-2.0/examples/) · [Recipes](https://furox-art.github.io/scientific-computing-system-2.0/recipes/) | runnable scripts; copy-paste snippets |
| [Case studies](https://furox-art.github.io/scientific-computing-system-2.0/case-studies/) | reproducible end-to-end workflows |
| [Validation](https://furox-art.github.io/scientific-computing-system-2.0/validation-and-reproducibility/) · [research-readiness](https://furox-art.github.io/scientific-computing-system-2.0/research-readiness/) | what is verified, and how |
| [Industrial computing](https://furox-art.github.io/scientific-computing-system-2.0/industrial/) | I/O and industrial use cases |
| [Benchmarks](https://furox-art.github.io/scientific-computing-system-2.0/benchmarks/) · [methodology](https://furox-art.github.io/scientific-computing-system-2.0/benchmark-methodology/) | measurements and their limits |
| [API reference](https://furox-art.github.io/scientific-computing-system-2.0/) | 47 generated module pages |
| [Deprecated modules](https://furox-art.github.io/scientific-computing-system-2.0/deprecated/) | `cds2.special`, `cds2.distributions` and how to migrate |
| [npm shim](https://furox-art.github.io/scientific-computing-system-2.0/npm/) | what the npm package is, and is not |
| [Release process](https://furox-art.github.io/scientific-computing-system-2.0/release/) | how a release is cut |

## Contributing

```bash
pip install -e ".[dev]"
pytest --cov=cds2 --cov-fail-under=100   # the gate is fail_under=100
mkdocs serve                              # docs at http://127.0.0.1:8000
```

Coverage below 100% fails CI, so new code needs tests on success *and* error paths. Full rules in [CONTRIBUTING.md](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/CONTRIBUTING.md).

## Deprecated modules

`cds2.special` and `cds2.distributions` are deprecated since 4.3.0. Because
`__init__.py` imports both eagerly, a bare `import cds2` emits **both**
`DeprecationWarning`s even if you never touch them:

```text
DeprecationWarning: cds2.special is deprecated since 4.3.0 ...
DeprecationWarning: cds2.distributions is deprecated since 4.3.0 ...
```

They still work; prefer `scipy.special` and `scipy.stats` directly. Migration
notes: [Deprecated modules](https://furox-art.github.io/scientific-computing-system-2.0/deprecated/).

## Project files

[CHANGELOG.md](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/CHANGELOG.md) ·
[CONTRIBUTING.md](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/CONTRIBUTING.md) ·
[SECURITY.md](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/SECURITY.md) ·
[CODE_OF_CONDUCT.md](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/CODE_OF_CONDUCT.md) ·
[CITATION.cff](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/CITATION.cff) ·
[codemeta.json](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/codemeta.json) ·
[LICENSE](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/LICENSE) (MIT)

## License

MIT.
