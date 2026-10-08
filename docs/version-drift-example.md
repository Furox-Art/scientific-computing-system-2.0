# NumPy/SciPy version-drift example

This case study answers a question unit tests cannot: **does a saved scientific
result still hold when the numerical stack underneath it changes?**

A single deterministic dataset, a single model (`exponential`) and a single
random seed (`20261006`) are fitted twice — once in one numerical stack and
once in another. The fitted parameters, their 95% confidence intervals and
every scalar diagnostic are then compared side by side, with absolute and
relative deviation reported for each quantity. Environment drift is reported
explicitly, and any measured change that exceeds a materiality threshold is
escalated to a loud `WARNING` instead of being silently rounded away.

## Existing tooling

The drift-detection mechanism is not new; this example packages it end to end.

| Component | Role |
|---|---|
| `src/cds2/guided_fit.py` | `run_guided_fit` records `package_versions` and `data_hashes`; `rerun_manifest` replays a saved manifest and diffs versions, input hashes, CIs, parameters, RMSE and the reliability label against the 5% materiality threshold |
| `scripts/reproduce_guided_fit_dependency_drift.py` | Driver with `baseline` and `rerun` subcommands; writes `baseline-result.json` and `upgraded-rerun.json` |
| `scripts/compare_drift_results.py` | New: compares a baseline/rerun pair, prints the side-by-side table and per-quantity deviation, writes a machine-readable comparison, exits non-zero on material drift |
| `examples/version_drift_fit.py` | New: runnable, self-contained case study over the committed records |
| `examples/data/version-drift/` | New: committed, already-measured baseline/rerun records and manifest |
| `.github/workflows/guided-fit-drift-case.yml` | CI recipe that produces the baseline and rerun in two separate jobs |
| `benchmarks/history/guided_fit_stack_upgrade_20261006.json` | Archived provenance record for the original case |

## Methodology

1. **Fix the invariants.** The dataset is synthetic and deterministic
   (`np.linspace(0, 0.9, 96)` with a closed-form exponential plus two small
   periodic offsets), so no external download and no hidden state is involved.
   The manifest pins model, missing/outlier policy and seed, and records the
   SHA-256 of the scientific input.
2. **Fit once per environment.** The baseline environment writes the manifest.
   Each other environment replays *that same manifest* through the real
   `guided-fit-rerun` code path, so the comparison exercises shipped behaviour
   rather than a re-implementation.
3. **Record the environment.** Python, NumPy, SciPy, pandas and Matplotlib
   versions are captured on both sides.
4. **Measure deviation.** For every parameter and every CI bound, both absolute
   and relative deviation are computed. Whole-result metrics are the L2 norms
   of the CI-bound and parameter-vector differences.
5. **Escalate on materiality.** A relative change above the threshold
   (default 5%) produces a `WARNING` and a non-zero exit status, so the check
   works as a CI gate. Below it, the environment change is still reported but
   the result is declared reproducible.

## Reproduction

```bash
git clone https://github.com/Furox-Art/scientific-computing-system-2.0
cd scientific-computing-system-2.0
git checkout feat/version-drift-example

python -m venv /tmp/env-old && /tmp/env-old/bin/pip install "numpy==1.26.4" "scipy==1.11.4" "pandas==2.2.3" "matplotlib==3.10.8"
python -m venv /tmp/env-new && /tmp/env-new/bin/pip install "numpy==2.3.5" "scipy==1.18.1" "pandas==2.2.3" "matplotlib==3.10.8"

# 1. baseline: old stack creates the manifest
PYTHONPATH=src /tmp/env-old/bin/python scripts/reproduce_guided_fit_dependency_drift.py \
    baseline --output-dir case

# 2. rerun: new stack replays that exact manifest
PYTHONPATH=src /tmp/env-new/bin/python scripts/reproduce_guided_fit_dependency_drift.py \
    rerun --manifest case/guided_fit_manifest.json --output case/upgraded-rerun.json

# 3. compare and gate
PYTHONPATH=src /tmp/env-old/bin/python scripts/compare_drift_results.py \
    --baseline case/baseline-result.json --rerun case/upgraded-rerun.json \
    --output case/drift-comparison.json

# 4. print the committed case study
PYTHONPATH=src /tmp/env-old/bin/python examples/version_drift_fit.py --self-check
```

`rerun` only reads sources inside the manifest directory or the current working
directory. If you move the manifest, keep its CSV beside it or the trusted-path
guard will correctly refuse to read a foreign absolute path.

## Measured result

Both halves below were **actually executed** in their own environment.
Python, pandas and Matplotlib were pinned identically (3.12.3 / 2.2.3 / 3.10.8);
only NumPy and SciPy differ.

| Component | Baseline | Rerun |
|---|---:|---:|
| Python | 3.12.3 | 3.12.3 |
| NumPy | 1.26.4 | 2.3.5 |
| SciPy | 1.11.4 | 1.18.1 |
| pandas | 2.2.3 | 2.2.3 |
| Matplotlib | 3.10.8 | 3.10.8 |

NumPy and SciPy are upgraded together because SciPy 1.18.1 requires
NumPy >= 2.0, so no honest common NumPy pin spans this pair. This is a
**numerical-stack upgrade case**, not a claim that SciPy alone caused any
observed difference.

### Environment drift reported

```text
runtime version changed: numpy 1.26.4 -> 2.3.5
runtime version changed: scipy 1.11.4 -> 1.18.1
```

### Parameters

| Parameter | Baseline | Rerun | Abs deviation | Rel deviation |
|---|---:|---:|---:|---:|
| `a` | 1.644089576846 | 1.644071588798 | 1.7988e-05 | 0.0011% |
| `b` | 0.080969414942 | 0.080970269199 | 8.5426e-07 | 0.0011% |
| `c` | 2.356647052271 | 2.356665048168 | 1.7996e-05 | 0.0008% |

### Confidence intervals

| Bound | Baseline | Rerun | Abs deviation | Rel deviation |
|---|---:|---:|---:|---:|
| `a` lower | -0.361642994648 | -0.361528607327 | 1.14387e-04 | 0.0316% |
| `a` upper | 3.649822148340 | 3.649671784924 | 1.50363e-04 | 0.0041% |
| `b` lower | -0.014297030760 | -0.014295490250 | 1.54051e-06 | 0.0108% |
| `b` upper | 0.176235860643 | 0.176236028648 | 1.68004e-07 | 0.0001% |
| `c` lower | 0.350035518747 | 0.350185883937 | 1.50365e-04 | 0.0430% |
| `c` upper | 4.363258585795 | 4.363144212398 | 1.14373e-04 | 0.0026% |

### Scalar diagnostics

| Quantity | Baseline | Rerun | Abs deviation | Rel deviation |
|---|---:|---:|---:|---:|
| RMSE | 0.000322850056641 | 0.000322850056642 | 1.0453e-15 | 3.2e-10% |
| CV RMSE | 0.000332987990343 | 0.000332988009641 | 1.9298e-11 | 5.8e-06% |
| R² | 0.999920691932623 | 0.999920691932622 | 5.5511e-16 | — |
| Cross-check error | 1.7425e-09 | 1.9166e-09 | 1.7410e-10 | 9.9912% |
| Reliability | reliable | reliable | — | unchanged |

### Aggregate deviation metrics

```text
max_abs_ci_bound_shift            0.000150365189991
relative_ci_l2                    4.67637304143e-05
parameter_l2_shift                2.54588280424e-05
parameter_l2_relative             8.8564516882e-06
rmse_relative_change              3.23766185878e-12
cv_rmse_relative_change           5.79533573121e-08
r_squared_absolute_change         5.55111512313e-16
```

### Verdict

```text
materiality threshold          : 5% relative change
environment drift detected     : True
reliability label              : reliable -> reliable (unchanged)
material drift                 : False
```

## What the numbers mean

- **`max_abs_ci_bound_shift` = 1.504e-04** — the largest movement of any single
  CI bound, in the same units as the parameter. The parameters are of order
  0.08–2.4, so a shift of ~1.5e-04 is roughly five orders of magnitude below
  the parameter itself.
- **`relative_ci_l2` = 4.676e-05 (0.00468%)** — the whole CI block moved by
  about five ten-thousandths of one percent. The 5% threshold is roughly
  **1000× larger** than the measured drift.
- **Parameters moved by ~0.001% each**, and `parameter_l2_relative` is
  8.86e-06. The fitted values agree to about one part in 10⁵.
- **`r_squared` differs only in the last bits** (5.55e-16, i.e. 1–2 ULPs of a
  value very close to 1). This is pure floating-point rounding, not a
  modelling change.
- **`cross_check_error` shows ~10% relative change, but it is not a material
  signal.** Both values are ~1e-09, near the noise floor of the independent
  cross-check fit. A 1.7e-10 absolute change on a 1.7e-09 quantity is still an
  agreement to about one part in ten. Relative change on a quantity this close
  to zero is the wrong diagnostic; this is exactly why the materiality gate is
  driven by CI and parameter vectors, not by this scalar.
- **The reliability label stayed `reliable`.** The scientific conclusion —
  the fitted exponential and its uncertainty — is unchanged.

The intended distinction is therefore preserved: the tool **reports the
environment change** even when the **scientific result is numerically
stable**, and it does not promote a small numerical difference into a
material-drift claim.

## Verified control cases

These were run as part of building this example, to confirm the mechanism
behaves correctly in both directions:

1. **Same-environment replay (expect zero drift).** Replaying the baseline
   manifest in the baseline environment produced `stability_warning=False`,
   empty `stability_details`, and **bit-identical** `params`,
   `confidence_95`, `parameter_std`, `rmse`, `cv_rmse`, `r_squared`,
   `cross_check_error` and `trust`. Zero environment change yields exactly
   zero numerical change, which is the necessary control for the drift
   measurement above.
2. **Forced material drift (expect WARNING).** Running the comparison with
   `--materiality-threshold 1e-5` correctly classified the real measured drift
   as material, printed the `WARNING`, and exited `1`. The escalation path is
   not decorative.

## Honest limitations

- This case is a **NumPy + SciPy stack upgrade**, not an isolated SciPy bump.
  SciPy 1.18.1 requires NumPy >= 2.0, so the pair cannot be separated honestly
  at these pins.
- Only one dataset, one model and one seed are covered. A different model
  (notably `logistic` or `power`) or an ill-conditioned dataset could drift
  more than this well-conditioned case; the threshold behaviour should be
  re-measured rather than assumed to transfer.
- Python, pandas and Matplotlib were held fixed. A simultaneous Python change
  would add a third variable.
- BLAS backend, thread count and CPU were not varied. Those are additional
  real sources of numerical drift and are not exercised here.
- The committed records were produced on WSL Ubuntu with CPython 3.12.3. The
  original CI case (Python 3.12.14) reported a slightly different
  `max_abs_ci_bound_shift` of 1.1748e-04 versus the 1.5037e-04 measured here.
  Both are far below the threshold, but the exact last digits are
  platform- and build-dependent; treat the order of magnitude, not the exact
  digits, as portable.
