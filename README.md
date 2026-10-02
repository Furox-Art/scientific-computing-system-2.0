# scientific-computing-system-2.0  
  
<p align="center">  
  <img src="https://raw.githubusercontent.com/Furox-Art/scientific-computing-system-2.0/main/docs/assets/promo_hero.png" alt="scientific-computing-system-2.0 scientific computing platform" width="100%">  
</p>  
  
[![CI](https://github.com/Furox-Art/scientific-computing-system-2.0/actions/workflows/tests.yml/badge.svg)](https://github.com/Furox-Art/scientific-computing-system-2.0/actions/workflows/tests.yml)  
[![PyPI](https://img.shields.io/pypi/v/scientific-computing-system-2.0)](https://pypi.org/project/scientific-computing-system-2.0/)  
[![npm](https://img.shields.io/npm/v/scientific-computing-system-2.0)](https://www.npmjs.com/package/scientific-computing-system-2.0)  
[![Python](https://img.shields.io/pypi/pyversions/scientific-computing-system-2.0)](https://pypi.org/project/scientific-computing-system-2.0/)  
  
This is the NumPy build of [scientific-computing-system](https://github.com/Furox-Art/scientific-computing-system), not a separate product. Install this when you want NumPy, SciPy, pandas, and matplotlib. Install the other package when you want readable pure Python and no runtime dependencies.  
  
PyPI is the install path. npm is no longer published.  
  
## What changed from v1  
  
- **Performance**: 10-100x faster on real workloads (thanks, NumPy)  
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
  
## License  
  
MIT. 
