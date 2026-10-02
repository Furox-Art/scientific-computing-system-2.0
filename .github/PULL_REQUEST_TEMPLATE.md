# Pull request

## What changed and why

<!-- One paragraph: the problem, the fix, the evidence. -->

## Quality gates

<!-- All enforced by CI; check each box only if it actually holds. -->

- [ ] `pytest` passes
- [ ] Coverage stays at 100% (`pytest --cov=cds2 --cov-fail-under=100`)
- [ ] `mypy` strict is clean
- [ ] `ruff check .` and `ruff format --check .` are clean
- [ ] New behavior has tests on both success and error paths
- [ ] Docs updated (`docs/`, `mkdocs.yml` nav, examples where relevant)
- [ ] No version bumps, changelogs, or release metadata in this PR
      (releases are cut separately — see `docs/release.md`)

## Reproduction / verification

<!-- Commands run and their output, e.g. a snippet proving a
     docstring example prints what it claims. -->
