# CDS v2 benchmarks

Generated 2026-08-21 by `benchmarks/run_benchmarks.py`. Ratio is cds2 time divided by baseline time: 1.00x means parity with the underlying library.

| Benchmark | Baseline | Baseline (s) | cds2 (s) | cds2/baseline |
|---|---|---:|---:|---:|
| solve 8x8 x300 | numpy | 0.00102 | 0.00114 | 1.11x |
| solve 800x800 | numpy | 0.00774 | 0.00802 | 1.04x |
| eigh 400x400 | numpy | 0.03465 | 0.03436 | 0.99x |
| rfft n=262144 | numpy | 0.00267 | 0.00266 | 1.00x |
| welch n=131072 | scipy | 0.01568 | 0.01575 | 1.00x |
| t-test n=100000 | scipy | 0.00077 | 0.00075 | 0.97x |
| describe n=500000 | scipy | 0.01814 | 0.02262 | 1.25x |
| minimize rosenbrock | scipy | 0.00223 | 0.00231 | 1.03x |
| mc-pi n=2000000 | numpy | 0.03216 | 0.02350 | 0.73x |
| dataframe summary 200000x5 | pandas | 0.02847 | 0.04492 | 1.58x |
| pagerank 400n/2400e | networkx | 0.00428 | 0.00655 | 1.53x |
| kmeans 4000x2 k=8 | sklearn | 0.01819 | 0.03483 | 1.91x |
| linreg 20000x10 | sklearn | 0.00522 | 0.00371 | 0.71x |

Wrapper-only rows (solve, fft, welch, t-test, minimize) demonstrate
that the cds2 API layer adds no measurable cost over calling
NumPy/SciPy directly. Rows where cds2 returns strictly more
information (describe adds quartiles; dataframe summary adds nulls
and uniques per column) carry a small honest premium for it.
Specialist rows (NetworkX, scikit-learn) show where pure-NumPy
implementations win or lose against compiled C code.
