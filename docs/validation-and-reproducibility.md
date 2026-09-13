# Validation and reproducibility

CDS2 treats numerical correctness, reproducibility, and benchmark transparency as separate requirements.

## Validation strategy

Important numerical routines should be checked through more than one route whenever a trustworthy independent oracle exists:

1. **Analytical invariants** — exact identities, conservation laws, known closed-form solutions, symmetry constraints, monotonicity, and dimensional sanity checks.
2. **Independent implementation oracles** — compare CDS2 results against established implementations from NumPy, SciPy, scikit-learn, NetworkX, or another domain-appropriate library. Oracle dependencies remain development/test-only unless the runtime feature itself requires them.
3. **Property-based tests** — exercise broad input families, including degeneracies and adversarial edge cases, instead of relying only on hand-picked examples.
4. **Numerical stress tests** — ill-conditioned systems, nearly singular matrices, extreme probabilities, stiff or unstable regimes, boundary values, and scale-separated inputs.
5. **Regression tests** — preserve previously validated numerical behavior while allowing intentional algorithmic changes to be reviewed explicitly.

Passing unit tests is necessary but not sufficient evidence of scientific correctness. Numerical claims should be tied to a stated tolerance, oracle, invariant, or reference result.

## Reproducibility contract

For stochastic routines and workflows, reproducible runs should record or expose:

- random seed / generator state where applicable,
- package version,
- Python version,
- dependency versions,
- input-data identity or checksum when a workflow consumes external data,
- model and preprocessing choices,
- uncertainty assumptions,
- output artifact locations.

`cds2 guided-fit` already writes a reproducibility manifest and `guided-fit-rerun` replays saved analyses. New end-to-end scientific workflows should follow the same pattern rather than introducing opaque state.

## Numerical tolerances

Tests should prefer a tolerance justified by the operation and conditioning instead of a single global epsilon.

- Exact combinatorial/discrete results should use exact equality.
- Well-conditioned floating-point routines should use strict `rtol`/`atol`.
- Iterative solvers should validate both solution error and residual norms.
- Stochastic estimators should validate confidence-aware error bounds rather than exact samples.
- Chaotic trajectories should validate invariants/statistics, not long-horizon pointwise equality.

Any relaxed tolerance should be documented in the test or validation note with the reason it is required.

## Evidence levels

| Level | Meaning |
|---|---|
| API-tested | Expected inputs/outputs and error handling are covered by tests. |
| Property-tested | General invariants are exercised over generated inputs. |
| Oracle-validated | Results are cross-checked against an independent implementation or analytical result. |
| Stress-tested | Pathological numerical regimes and boundary conditions are exercised. |
| Reproducible workflow | Inputs, configuration, seed/state, versions, and outputs can be replayed. |

A module should not be described as scientifically validated merely because it has high line coverage.

## Benchmark reproducibility

Performance numbers are not scientific validation. Benchmark reports should record enough environment information to reproduce the measurement, including CPU, operating system, Python, NumPy/SciPy versions, repetitions, warm-up policy, and summary statistic. Where timing noise is material, report a distribution or robust statistic rather than a single run.

See `docs/benchmarks.md` for the current scoreboard and `benchmarks/run_benchmarks.py` for the executable benchmark suite.
