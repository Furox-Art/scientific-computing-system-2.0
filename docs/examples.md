# Examples & case studies

Runnable end-to-end scripts live in the repository under `examples/`.
Each one prints its analysis; several also save figures.

| Script | Techniques used |
|---|---|
| [`examples/signal_denoising.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/signal_denoising.py) | Butterworth low-pass design, Welch spectra, interference suppression |
| [`examples/experiment_fitting.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/experiment_fitting.py) | Michaelis-Menten curve fit, parameter covariance, Welch t-test + permutation test on residuals |
| [`examples/bayesian_inference.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/bayesian_inference.py) | Metropolis-Hastings MCMC, posterior mean, 95% credible interval vs analytic conjugate result |
| [`examples/graph_analysis.py`](https://github.com/Furox-Art/scientific-computing-system-2.0/blob/main/examples/graph_analysis.py) | Compiled PageRank kernel, spectral clustering, topological sort, algebraic connectivity |

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

## Suggested learning path

1. Start with the [getting started guide](getting-started.md).
2. Read the [modules overview](modules.md), then skim the API reference.
3. Study each example above alongside its module documentation.
4. Benchmark what you build: see [benchmarks](benchmarks.md) for how cds2
   compares against the underlying stack.
