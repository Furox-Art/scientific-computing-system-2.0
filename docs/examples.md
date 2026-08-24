# Examples & case studies

Runnable end-to-end scripts live in the repository under `examples/`.
Each one prints its analysis; several also save figures.

| Script | Techniques used |
|---|---|
| [`examples/signal_denoising.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/signal_denoising.py) | Butterworth low-pass design, Welch spectra, interference suppression |
| [`examples/experiment_fitting.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/experiment_fitting.py) | Michaelis-Menten curve fit, parameter covariance, Welch t-test + permutation test on residuals |
| [`examples/bayesian_inference.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/bayesian_inference.py) | Metropolis-Hastings MCMC, posterior mean, 95% credible interval vs analytic conjugate result |
| [`examples/graph_analysis.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/graph_analysis.py) | Compiled PageRank kernel, spectral clustering, topological sort, algebraic connectivity |
| [`examples/information_flow.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/information_flow.py) | Mutual information vs Pearson correlation, entropy, permutation entropy (`cds2.infotheory`) |
| [`examples/chaos_diagnostics.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/chaos_diagnostics.py) | Delay embedding, false nearest neighbours, Lyapunov exponent, correlation dimension, bifurcation scan (`cds2.chaos`) |
| [`examples/ab_test_bayes.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/ab_test_bayes.py) | Beta-Binomial conjugate posteriors, P(B>A) via Monte Carlo, uplift CI, naive Bayes (`cds2.bayes`) |
| [`examples/metaheuristic_race.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/metaheuristic_race.py) | Genetic algorithm vs PSO vs simulated annealing vs differential evolution on a multimodal landscape (`cds2.metaheuristics`, `cds2.optimize`) |
| [`examples/sensor_geometry.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/sensor_geometry.py) | Convex hull coverage area, closest-pair redundancy flag, point-in-polygon containment, rotations (`cds2.geometry`) |
| [`examples/reinforcement_learning.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/reinforcement_learning.py) | UCB1 vs epsilon-greedy bandits, tabular Q-learning with greedy path replay (`cds2.rl`) |
| [`examples/wavelet_denoising.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/wavelet_denoising.py) | Haar multi-level decomposition, detail energy profile, MAD-threshold denoising (`cds2.wavelets`) |
| [`examples/epidemic_simulation.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/epidemic_simulation.py) | SIR mitigation scenarios, SEIR incubation lag, herd-immunity vs final-size analytics (`cds2.epidemiology`) |
| [`examples/image_filters.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/image_filters.py) | Gaussian blur, Sobel edges with direction, max-pooling, binary morphology (`cds2.image`) |
| [`examples/sequence_analysis.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/sequence_analysis.py) | GC content, mutation alignment, k-mer ranking, ORF scan, reverse complement (`cds2.genetics`) |
| [`examples/survival_reliability.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/survival_reliability.py) | Censored Kaplan-Meier, Weibull parameter recovery, MTBF/availability, bathtub hazard (`cds2.reliability`) |
| [`examples/portfolio_risk.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/portfolio_risk.py) | Volatility, Sharpe/Sortino, drawdown tracking, historical + Monte Carlo VaR, protective put (`cds2.finance`) |
| [`examples/document_similarity.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/document_similarity.py) | TF-IDF space, pairwise cosine clusters, Jaccard contrast, corpus terms (`cds2.text`) |
| [`examples/game_tournament.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/game_tournament.py) | Pure Nash enumeration, round-robin iterated PD tournament, zero-sum minimax (`cds2.game_theory`) |
| [`examples/route_optimization.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/route_optimization.py) | Nearest-neighbor TSP + 2-opt gain, optimal courier assignment, knapsack packing, LCS (`cds2.combinatorial`) |
| [`examples/spatial_autocorrelation.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/spatial_autocorrelation.py) | Row-standardized weights, Moran's I / Geary's C z-scores, Clark-Evans point patterns (`cds2.spatial`) |

Run any of them from a repository checkout:

```bash
python examples/signal_denoising.py
```

Sample output - Bayesian inference:

```text
MCMC     posterior mean : 1.0874
Analytic posterior mean  : 1.0820
95% credible interval    : [0.5815, 1.5722]
Acceptance rate          : 0.71
```

Sample output - A/B test with conjugate Bayes:

```text
P(B > A)                 : 0.951
95% CI on relative uplift: [-0.033, +0.492]
```

## Suggested learning path

1. Start with the [getting started guide](getting-started.md).
2. Read the [modules overview](modules.md), then skim the API reference.
3. Study each example above alongside its module documentation.
4. Benchmark what you build: see [benchmarks](benchmarks.md) for how cds2
   compares against the underlying stack.
