name: Feature request
description: Propose a new module, function, or documentation improvement
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: |
        Small, fast, well-tested numerical APIs only — see
        [CONTRIBUTING.md](../../CONTRIBUTING.md) for the quality gates
        (100% coverage, mypy strict, ruff clean).
  - type: textarea
    id: proposal
    attributes:
      label: Proposal
      description: What should be added and which module it belongs in?
    validations:
      required: true
  - type: textarea
    id: api
    attributes:
      label: Proposed API sketch
      description: Function signatures with a short usage example
      placeholder: |
        from cds2 import stats
        stats.trimmed_mean([1.0, 2.0, 100.0], proportion=0.1)
  - type: textarea
    id: reference
    attributes:
      label: Reference results
      description: Textbook, paper, or another implementation the math can be checked against
