# Industrial-scale computing with cds2

How the pieces fit together when problems get big: six-figure unknowns,
multi-gigabyte streams and multi-core kernels.

## 1. Preconditioned iterative solvers at six-figure scale

A 250,000-unknown tridiagonal system has condition number ~6e9. Plain CG
stalls; an incomplete-LU preconditioner makes it routine:

```python
import numpy as np
from cds2 import sparse

n = 250_000
bands = sparse.sparse_diag(
    [2.0 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)],
    offsets=[0, -1, 1],
)
rhs = np.linspace(1.0, 2.0, n)

preconditioner = sparse.ilu_preconditioner(bands, drop_tol=1e-8)
result = sparse.solve_cg(bands, rhs, rtol=1e-8, M=preconditioner)

print(result.converged, result.residual_norm)
```

Every iterative solver (`solve_cg`, `solve_gmres`, `solve_bicgstab`)
accepts ``M`` and reports the true residual norm ``||Ax - b||``.

## 2. Process-parallel Monte Carlo

```python
from cds2.montecarlo import parallel_mc_integrate

def integrand(x):          # must be picklable: defined at module level
    return np.exp(-x * x)

estimate = parallel_mc_integrate(integrand, 0.0, 6.0,
                                 n_total=20_000_000, workers=8, seed=42)
```

Workers receive equal sub-intervals and independent seeds; the estimate is
the sum of per-chunk integrals.

## 3. Constant-memory statistics for streams larger than RAM

```python
import numpy as np
from cds2.stats import StreamingStats

stream = StreamingStats()
for chunk in iter_csv_chunk_source():      # your generator of batches
    stream.push(chunk[["signal"]].to_numpy())

print(stream.mean, stream.standard_deviation, stream.count_value)
```

Welford updates keep memory constant regardless of row count; ``merge``
combines partial accumulators (map-reduce friendly).

## 4. Out-of-core CSV processing

```python
from cds2.io import iter_csv

totals = []
for chunk in iter_csv("measurements.csv", chunksize=250_000):
    totals.append(chunk["value"].sum())
print(sum(totals))
```

## 5. Multicore C kernels

On Linux wheels the KMeans assignment sweep and PageRank power iteration
compile with OpenMP (`-fopenmp`) and drop the GIL during compute, so both
scale across cores exactly when datasets grow. Windows/macOS wheels ship
the serial kernel; set ``CDS_NO_OPENMP=1`` to disable explicitly.

## Why this stack holds up

- Every claim above runs inside the test suite at 100% blended coverage.
- Preconditioned convergence is asserted against direct banded solves.
- Streaming results are asserted equal to batch NumPy statistics.
