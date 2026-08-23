# CDS v2

**scientific-computing-system-2.0** is a scientific computing platform built on
the scientific Python stack - NumPy, SciPy, pandas and matplotlib. The
algorithms proven in the pure-Python
[cognitive-discovery-system](https://github.com/Furox88/cognitive-discovery-system)
(v1.x) form its foundation; v2 rebuilds them for speed and adds new domain
modules on top.

## Highlights

- **Accelerated core** - linear algebra, statistics, optimization,
  integration, interpolation, signal processing and Monte Carlo on top of
  NumPy/SciPy instead of hand-rolled loops.
- **Graphs with PageRank** - components, shortest paths (Dijkstra /
  Bellman-Ford / Floyd-Warshall), minimum spanning forests, topological order.
- **Machine learning** - linear/logistic regression, k-means++, PCA, kNN,
  metrics and synthetic data generators.
- **Time series** - seasonal decomposition, ACF/PACF, Ljung-Box on pandas
  series.
- **Visualization & I/O** - matplotlib helpers and pandas-backed readers.

## Quick start

```bash
pip install scientific-computing-system-2.0
```

```python
import numpy as np
import cds2

# Solve a linear system
x = cds2.linalg.solve([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])

# PageRank over a directed graph
adj = cds2.graph.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
scores = cds2.graph.pagerank(adj)

# Monte Carlo estimate of pi
print(cds2.montecarlo.pi_estimate(n=100_000, seed=42))
```

Continue with the [getting started guide](getting-started.md) or dive into
the [API reference](api/linalg.md).
