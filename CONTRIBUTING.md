# Contributing to scientific-computing-system-2.0

Thanks for considering a contribution! The project values small, fast,
well-tested numerical APIs.

## Development setup

```bash
git clone https://github.com/Furox-Art/scientific-computing-system-2.0.git
cd scientific-computing-system-2.0
pip install -e .[dev]
pre-commit install          # optional: ruff + hygiene on every commit
```

## Quality gates (all enforced by CI)

| Gate | Command | Requirement |
|---|---|---|
| Tests | `pytest` | all pass |
| Coverage | `pytest --cov=cds2 --cov-fail-under=100` | **100% blended coverage** |
| Types | `mypy` | zero errors, strict mode |
| Lint | `ruff check .` | clean |
| Format | `ruff format --check .` | clean |

A pull request that drops coverage below 100% will fail the gate; tests must
exercise both success and error paths of any new code.

## Conventions

- One domain per module in `src/cds2/`; export its public API via `__all__`.
- Frozen dataclasses for multi-value results; plain functions for single values.
- Docstrings in NumPy style; they feed the mkdocstrings API reference.
- Error handling: `msg = "..."; raise ValueError(msg)` pattern, no bare raises.
- Deterministic randomness through explicit `seed` parameters on
  `numpy.random.default_rng`.
- Heavy lifting belongs to NumPy/SciPy vectorization or a compiled kernel,
  never to Python loops over large arrays.

## Adding a module checklist

1. `src/cds2/<name>.py` with module docstring and `__all__`.
2. Wire it into `src/cds2/__init__.py` (import + re-export + `__all__`).
3. `tests/test_<name>.py` reaching 100% coverage including branches.
4. `docs/api/<name>.md` page plus an entry in `mkdocs.yml` nav and
   `docs/modules.md`.
5. Optional: a runnable `examples/<topic>.py` case study.

## Releasing

See [docs/release.md](docs/release.md) - tags drive automated builds and
PyPI publication.
