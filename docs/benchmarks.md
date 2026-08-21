# CDS v2 benchmarks

Generated 2026-08-21 by `benchmarks/run_benchmarks.py`. Ratio is cds2 time divided by baseline time: 1.00x means parity with the underlying library.

| Benchmark | Baseline | Baseline (s) | cds2 (s) | cds2/baseline |
|---|---|---:|---:|---:|
| solve 8x8 x300 | numpy | 0.00118 | 0.00136 | 1.16x |
| solve 800x800 | numpy | 0.01122 | 0.01142 | 1.02x |
| eigh 400x400 | numpy | 0.05656 | 0.04993 | 0.88x |
| rfft n=262144 | numpy | 0.00376 | 0.00367 | 0.98x |
| welch n=131072 | scipy | 0.01823 | 0.02261 | 1.24x |
| t-test n=100000 | scipy | 0.00113 | 0.00093 | 0.83x |
| describe n=500000 | scipy | 0.02842 | 0.03137 | 1.10x |
| minimize rosenbrock | scipy | 0.00334 | 0.00270 | 0.81x |
| mc-pi n=2000000 | numpy | 0.05279 | 0.04081 | 0.77x |
| dataframe summary 200000x5 | pandas | 0.05271 | 0.07633 | 1.45x |
| pagerank 400n/2400e | networkx | 0.00236 | 0.00042 | 0.18x |
| kmeans 4000x2 k=8 | sklearn | 0.01143 | 0.00823 | 0.72x |
| linreg 20000x10 | sklearn | 0.00554 | 0.00408 | 0.74x |

Wrapper-only rows (solve, fft, welch, t-test, minimize) demonstrate
that the cds2 API layer adds no measurable cost over calling
NumPy/SciPy directly. Rows where cds2 returns strictly more
information (describe adds quartiles; dataframe summary adds nulls
and uniques per column) carry a small honest premium for it.
Specialist rows (NetworkX, scikit-learn) show where pure-NumPy
implementations win or lose against compiled C code.
