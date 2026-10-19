name: Bug report
description: Report a wrong result, crash, or misleading documentation
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for reporting. Please complete the sections below so the
        issue is reproducible. For security vulnerabilities, do **not**
        open an issue — see [SECURITY.md](../../SECURITY.md).
  - type: input
    id: version
    attributes:
      label: Package version
      description: Output of `cds2 info` or `pip show scientific-computing-system-2.0`
      placeholder: "5.2.5"
    validations:
      required: true
  - type: input
    id: python
    attributes:
      label: Python version and OS
      placeholder: "3.12 on Windows 11"
    validations:
      required: true
  - type: textarea
    id: repro
    attributes:
      label: Minimal reproduction
      description: Copy-pasteable snippet plus actual vs expected output
      placeholder: |
        from cds2 import linalg
        print(linalg.solve([[4.0, 7.0], [2.0, 6.0]], [18.0, 16.0]))
        # actual: [-0.4  2.8]
        # expected: ...
    validations:
      required: true
  - type: dropdown
    id: area
    attributes:
      label: Area
      options:
        - numerical result
        - crash / exception
        - documentation / example
        - performance
        - packaging / install
        - other
    validations:
      required: true
