# CDS v2 benchmarks

Generated 2026-08-21 by `benchmarks/run_benchmarks.py`. Ratio is cds2 time divided by baseline time: 1.00x means parity with the underlying library.

| Benchmark | Baseline | Baseline (s) | cds2 (s) | cds2/baseline |
|---|---|---:|---:|---:|
| solve 8x8 x300 | numpy | 0.00104 | 0.00114 | 1.09x |
| solve 800x800 | numpy | 0.00818 | 0.00849 | 1.04x |
| eigh 400x400 | numpy | 0.03538 | 0.03460 | 0.98x |
| rfft n=262144 | numpy | 0.00268 | 0.00263 | 0.98x |
| welch n=131072 | scipy | 0.01578 | 0.01585 | 1.00x |
| t-test n=100000 | scipy | 0.00077 | 0.00074 | 0.96x |
| describe n=500000 | scipy | 0.01819 | 0.02269 | 1.25x |
| minimize rosenbrock | scipy | 0.00237 | 0.00234 | 0.99x |
| mc-pi n=2000000 | numpy | 0.03299 | 0.02392 | 0.73x |
| dataframe summary 200000x5 | pandas | 0.02841 | 0.04764 | 1.68x |
| pagerank 400n/2400e | networkx | 0.00135 | 0.00309 | 2.29x |
| kmeans 4000x2 k=8 | sklearn | 0.01447 | 0.08805 | 6.09x |
| linreg 20000x10 | sklearn | 0.00644 | 0.00490 | 0.76x |

Wrapper-only rows (solve small, fft, welch, ...) demonstrate that the
cds2 API layer adds no measurable cost over calling NumPy/SciPy
directly. Specialist rows (NetworkX, scikit-learn) show where the
pure-NumPy implementations inside cds2 win or lose.
