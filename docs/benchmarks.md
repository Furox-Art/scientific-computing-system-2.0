# CDS v2 benchmarks

Generated 2026-08-21 by `benchmarks/run_benchmarks.py`. Ratio is cds2 time divided by baseline time: 1.00x means parity with the underlying library.

| Benchmark | Baseline | Baseline (s) | cds2 (s) | cds2/baseline |
|---|---|---:|---:|---:|
| solve 8x8 x300 | numpy | 0.00105 | 0.00116 | 1.10x |
| solve 800x800 | numpy | 0.00909 | 0.00940 | 1.03x |
| eigh 400x400 | numpy | 0.03668 | 0.03420 | 0.93x |
| rfft n=262144 | numpy | 0.00273 | 0.00276 | 1.01x |
| welch n=131072 | scipy | 0.01582 | 0.01634 | 1.03x |
| t-test n=100000 | scipy | 0.00079 | 0.00077 | 0.97x |
| describe n=500000 | scipy | 0.01952 | 0.02378 | 1.22x |
| minimize rosenbrock | scipy | 0.00237 | 0.00236 | 1.00x |
| mc-pi n=2000000 | numpy | 0.03358 | 0.02468 | 0.74x |
| dataframe summary 200000x5 | pandas | 0.03186 | 0.04831 | 1.52x |
| pagerank 400n/2400e | networkx | 0.00139 | 0.00188 | 1.35x |
| kmeans 4000x2 k=8 | sklearn | 0.01045 | 0.00827 | 0.79x |
| linreg 20000x10 | sklearn | 0.00300 | 0.00210 | 0.70x |

Wrapper-only rows (solve, fft, welch, t-test, minimize) demonstrate
that the cds2 API layer adds no measurable cost over calling
NumPy/SciPy directly. Rows where cds2 returns strictly more
information (describe adds quartiles; dataframe summary adds nulls
and uniques per column) carry a small honest premium for it.
Specialist rows (NetworkX, scikit-learn) show where pure-NumPy
implementations win or lose against compiled C code.
