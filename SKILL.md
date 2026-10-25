---
name: scientific-computing-system-2.0
description: "Use when a task needs NumPy/SciPy/pandas/matplotlib-based scientific computing, statistics, optimization, machine learning, information theory, or reproducible curve fitting with replayable manifests. Ships the `cds2` package and CLI; choose the zero-dependency `cds` build when third-party packages are not allowed. Published on PyPI and npm as `scientific-computing-system-2.0`, where 2.0 names the generation, not the version."
argument-hint: "<scientific analysis, dataset, or fitting task>"
metadata:
  version: "5.4.5"
---

# Scientific Computing System 2.0 (`cds2`)

The NumPy build of Scientific Computing System. It is the same scientific
project family as `scientific-computing-system`, but depends on the numerical
stack and offers 47 flat modules plus 6 subpackages, 535 public exports, and
1,926 tests with a 100% branch-coverage gate.

`2.0` in the package and repository name identifies this NumPy generation, not
the version. The current release is **5.4.5**; import and CLI are `cds2`.

## When to use this skill

- The task benefits from NumPy/SciPy/pandas/matplotlib.
- You need statistics, optimization, signal processing, ML, bayes, information
  theory, metaheuristics, geometry, or RL routines in one API.
- You need a reproducible fit: recommended model, explicit missing/outlier
  policy, saved manifest, and a rerun that reports environment drift.

Use `scientific-computing-system` instead when the environment must stay
dependency-free.

## Install

```bash
pip install scientific-computing-system-2.0
# extras: [dev] [test] [docs] [gpu] [all]
```

Python 3.10+. `python -m cds2` is equivalent to the `cds2` console script.

## Python quick start

```python
import cds2

print(cds2.linalg.solve([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0]))
print(cds2.montecarlo.pi_estimate(n=100_000, seed=42))
print(cds2.infotheory.entropy([0.25, 0.25, 0.25, 0.25]))
print(cds2.graph.pagerank(cds2.graph.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])))
```

## Reproducible fitting (`guided-fit`)

```bash
cds2 guided-fit data.csv --x time --y response \
    --model exponential --missing drop --outliers exclude --report pdf
cds2 guided-fit-rerun guided-fit-results/guided_fit_manifest.json
```

Five model families: `linear`, `quadratic`, `exponential`, `power`, `logistic`.
Missing-data policies: `ask`, `drop`, `interpolate`. Outlier policies: `ask`,
`keep`, `exclude`. The manifest stores the input hash, seed, and exact
Python/NumPy/SciPy/pandas/matplotlib versions; the rerun reports material
version or confidence-interval drift instead of silently treating the rerun as
equivalent.

## CLI (`cds2`)

Ten subcommands:

```text
cds2 info
cds2 stats 1,2,3,4,5
cds2 integrate sin --a 0 --b 3.14159
cds2 linsolve --a "3,1;1,2" --b "9,8"
cds2 entropy "0.25,0.25,0.25,0.25"
cds2 units 5 --from-unit km --to-unit mile
cds2 solve --coeffs "1,-5,6"
cds2 plot 1,3,2,5,4 --file out.png
cds2 guided-fit data.csv --x time --y response
cds2 guided-fit-rerun guided-fit-results/guided_fit_manifest.json
```

## Working rules

- Report the exact environment for numerical results; a successful rerun under
  different dependency versions is not automatically equivalent.
- The archived benchmark run in the README is from `cds2 3.0.0`, not the current
  release. Treat it as a historical measurement, not a present-day claim.
- The suite enforces 100% branch coverage. Add tests for new branches.
- Optional C kernels (`_fast_kmeans`, `_fast_pagerank`) are compiled when a
  compiler is available; verify which implementation actually ran.
- Use `scientific-computing-system` for zero-dependency or auditable pure-Python
  implementations.

## References

- Repository: https://github.com/Furox-Art/scientific-computing-system-2.0
- Docs: https://furox-art.github.io/scientific-computing-system-2.0/
- Case studies: https://furox-art.github.io/scientific-computing-system-2.0/case-studies/
- PyPI: https://pypi.org/project/scientific-computing-system-2.0/
