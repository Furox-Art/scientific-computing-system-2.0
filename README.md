# scientific-computing-system-2.0  
  
<p align="center">  
  <img src="https://raw.githubusercontent.com/Furox-Art/scientific-computing-system-2.0/main/docs/assets/promo_hero.png" alt="scientific-computing-system-2.0 scientific computing platform" width="100%">  
</p>  
  
[![CI](https://github.com/Furox-Art/scientific-computing-system-2.0/actions/workflows/tests.yml/badge.svg)](https://github.com/Furox-Art/scientific-computing-system-2.0/actions/workflows/tests.yml)  
[![PyPI](https://img.shields.io/pypi/v/scientific-computing-system-2.0)](https://pypi.org/project/scientific-computing-system-2.0/)  
[![npm](https://img.shields.io/npm/v/scientific-computing-system-2.0)](https://www.npmjs.com/package/scientific-computing-system-2.0)  
[![Python](https://img.shields.io/pypi/pyversions/scientific-computing-system-2.0)](https://pypi.org/project/scientific-computing-system-2.0/)  
  
The original scientific-computing-system was pure Python-no NumPy, no SciPy, no dependencies. It was beautiful and slow and educational.  
  
This is the pragmatic sequel. It uses NumPy, SciPy, pandas, and matplotlib under the hood, but wraps them in a cleaner, more consistent API. You get the speed of optimized C with the readability of modern Python.  
  
## Installation  
  
Requires Python 3.10+:  
  
```bash  
pip install scientific-computing-system-2.0  
```  
  
Full guide with a ten-minute tour:  
<https://furox-art.github.io/scientific-computing-system-2.0/getting-started/>  
  
## Quick start  
  
Every output below was executed against `scientific-computing-system-2.0==5.2.5`:  
  
```python  
import cds2  
  
x = cds2.linalg.solve([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])  
print(x)  # [2. 3.]  
print(cds2.montecarlo.pi_estimate(n=100_000, seed=42))  # 3.13776  
print(cds2.infotheory.entropy([0.25, 0.25, 0.25, 0.25]))  # 2.0  
```  
  
Command line:  
  
```bash  
cds2 stats 1,2,3,4,5  
cds2 info  
```  
  
API reference for all 46 modules:  
<https://furox-art.github.io/scientific-computing-system-2.0/>  
  
## What changed from v1  
  
- **Performance**: Vectorized NumPy/SciPy instead of pure-Python loops. Measured results are published with full provenance in [docs/benchmarks.md](docs/benchmarks.md): roughly at parity with the underlying libraries on wrapper-only calls, faster where a dedicated implementation applies (for example PageRank vs NetworkX). No blanket speedup factor is claimed.
- **Coverage**: More algorithms, more edge cases handled  
- **Testing**: Property-based tests, oracle comparisons against reference implementations  
- **Documentation**: Actually complete, with examples that run  
  
## Common scientific Python use cases

- Accelerated **linear algebra, statistics, optimization, integration, and interpolation**.
- **Bayesian inference**, Monte Carlo, uncertainty analysis, and reproducible model fitting.
- **Signal processing, time-series analysis, graphs and PageRank**.
- **Machine learning, reinforcement learning, information theory, and computational geometry**.
- **PDE/SDE solvers**, scientific visualization, pandas-backed I/O, and research-quality validation.

## The philosophy  
  
Scientific code should be boring. Not boring to write-boring to read. You should be able to look at a function and know exactly what it does, what it expects, and what it returns. No magic, no hidden state, no "just trust the library."  
  
Every function here has type hints, docstrings with examples, and tests that verify the math against known results. If the documentation and the code disagree, the code is wrong.  
  
## Deprecated modules  
  
`cds2.special` and `cds2.distributions` are deprecated since 4.3.0 and emit a  
`DeprecationWarning` on import. Use `scipy.special` and `scipy.stats` directly  
instead. Details:  
<https://furox-art.github.io/scientific-computing-system-2.0/deprecated/>  
  
## License  
  
MIT. 
