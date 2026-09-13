# Benchmark methodology

CDS2 benchmark results are reproducible engineering measurements, not universal performance claims. Timings depend on hardware, BLAS/LAPACK implementation, compiler, operating system, Python version, dependency versions, CPU power policy, and background load.

## Reproducible runner

For research-facing measurements, prefer the provenance-rich wrapper:

```bash
python -m benchmarks.research_run
python -m benchmarks.research_run --quick --core-only
python -m benchmarks.research_run --only solve_large networkx_pagerank
```

By default it writes `results.json` and `provenance.json` under `benchmarks/artifacts/latest/`. The result record includes the benchmark protocol and captures the full Git commit when available, dirty-tree state, Python implementation and executable, operating-system details, architecture, logical CPU count, dependency versions, thread-control environment variables, and NumPy's BLAS/LAPACK/compiler configuration.

The original `benchmarks/run_benchmarks.py` remains the low-level timing suite; `research_run.py` deliberately wraps it instead of maintaining a second implementation of the benchmark cases.

## Minimum metadata

Every published benchmark run should record:

- CDS2 commit SHA and package version,
- operating system and architecture,
- CPU model and logical/physical core count,
- Python version,
- NumPy, SciPy, pandas, scikit-learn, and NetworkX versions when used,
- relevant BLAS/LAPACK backend information,
- compiler information for CDS2 C extensions,
- benchmark parameters and dataset sizes,
- warm-up count,
- measured repetition count,
- statistic reported,
- dispersion such as min/max, IQR, MAD, or confidence interval when useful.

The current low-level suite uses the minimum observed wall-clock time across benchmark-specific repeat counts. That is useful for implementation-overhead comparisons but should not be presented as a population estimate. For publication-quality latency distributions, retain per-repeat samples and report a robust center and dispersion in addition to the minimum.

## Measurement protocol

For short-running operations, use enough repetitions to reduce timer quantization and scheduler noise. Warm up import paths and compiled code before recording measurements. Compare implementations in the same process/environment when possible and alternate execution order if cache effects may bias one side.

Report ratios only together with the absolute timings that produced them. A ratio below `1.0x` means CDS2 was faster in that run; it does not imply CDS2 is universally faster on other machines or problem sizes.

## Correctness before speed

A benchmark result is valid only if compared implementations solve the same problem to an equivalent numerical tolerance. Faster but numerically different output is not a win. Benchmark cases that return additional information should be labeled because the work performed is not identical.

## Performance regression policy

The CI regression gate is intended to catch substantial regressions, not normal cloud-runner noise. Changes that intentionally trade speed for correctness, stability, or richer diagnostics should document the reason and update the baseline in a reviewable commit.

## Publishing results

Generated benchmark artifacts should use a machine-readable format in addition to Markdown when practical. Keep `results.json` and `provenance.json` together so another researcher can determine exactly what code, software stack, and machine context produced the timings.

The current scoreboard is in `docs/benchmarks.md`; the executable suite lives in `benchmarks/run_benchmarks.py`, with research-oriented provenance collection in `benchmarks/research_run.py`.
