# Research-readiness checklist

Use this checklist when adding a new scientific algorithm or end-to-end workflow to CDS2.

## Numerical implementation

- [ ] Mathematical definition and domain are documented.
- [ ] Input validation covers invalid shapes, non-finite values, and unsupported regimes where relevant.
- [ ] Numerical stability considerations are documented.
- [ ] Deterministic behavior is guaranteed when the algorithm is deterministic.
- [ ] Randomized algorithms expose an explicit seed or generator.

## Verification

- [ ] At least one analytical invariant, closed-form case, or independent oracle is tested when available.
- [ ] Degenerate and boundary cases are covered.
- [ ] Property-based tests are added for reusable numerical invariants when practical.
- [ ] Iterative methods expose or test convergence/residual information.
- [ ] Tolerances are justified rather than chosen only to make a test pass.

## Reproducibility

- [ ] Random state is recordable.
- [ ] Version and configuration information needed to reproduce outputs is available.
- [ ] External datasets have provenance and immutable identifiers/checksums.
- [ ] Generated reports and plots can be recreated from code.

## Documentation

- [ ] API documentation explains assumptions and failure modes.
- [ ] A minimal runnable example exists.
- [ ] A realistic case study exists for substantial new workflows.
- [ ] Claims about accuracy or speed link to evidence.

## Performance

- [ ] Benchmarks compare equivalent numerical work.
- [ ] Environment and repetition methodology are recorded.
- [ ] Correctness is checked before timing results are accepted.
- [ ] Performance regressions are reviewed explicitly.

A completed checklist is evidence of engineering discipline, not a substitute for independent scientific review. Domain-critical use still requires validation appropriate to the application.
