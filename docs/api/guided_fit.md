# `cds2.guided_fit`

Guided scientific fitting adds a user-controlled workflow on top of the low-level optimization API.

The workflow:

1. inspects one or more CSV datasets and reports missing values;
2. runs small pilot fits and repeated cross-validation before recommending one model;
3. explains the recommendation with speed, expected fit behaviour and complexity;
4. leaves final model selection to the user;
5. includes measurement uncertainty when a `sigma` column is supplied;
6. detects suspicious residual outliers, quantifies their estimated RMSE effect and asks before excluding them;
7. fits the selected model, calculates parameter uncertainty and 95% confidence intervals;
8. checks held-out performance with repeated 5-fold cross-validation;
9. cross-checks the fitted result with an independent numerical method;
10. produces Matplotlib fit and residual PNG/PDF figures, a reproducibility manifest and an optional PDF/HTML/Markdown report; long PDF reports paginate automatically instead of truncating results;
11. labels the overall result `reliable`, `caution` or `unreliable`;
12. recommends a different model when the selected fit is weak;
13. recommends dataset-specific models when a single common model is materially weaker;
14. compares manifest reruns with the saved analysis and warns when runtime versions, inputs, confidence intervals, fit parameters, RMSE or the reliability label change materially.

## CLI

Interactive use:

```bash
cds2 guided-fit experiment.csv --x time --y response --sigma uncertainty
```

The CLI asks for decisions only when needed: missing-data handling, model selection, outlier exclusion and report format.

Non-interactive/reproducible use:

```bash
cds2 guided-fit experiment.csv \
  --x time \
  --y response \
  --sigma uncertainty \
  --model linear \
  --missing interpolate \
  --outliers keep \
  --report pdf \
  --seed 0 \
  --output-dir results
```

Multiple datasets can be supplied to search for a common model:

```bash
cds2 guided-fit run-a.csv run-b.csv --x time --y response
```

Repeat an analysis from its saved manifest:

```bash
cds2 guided-fit-rerun results/guided_fit_manifest.json
```

The manifest stores the selected model, policies, seed, data hashes, input source metadata and NumPy/SciPy/pandas/Matplotlib/Python versions. Reruns compare those saved results with the new calculation and surface material instability instead of silently replacing the prior result.

### What a drift warning looks like

When the runtime environment or numerical result has moved, the CLI makes that visible instead of silently treating the rerun as equivalent:

```text
warning   rerun differs materially from the saved analysis
detail    runtime version changed: scipy <saved-version> -> <current-version>
detail    confidence intervals changed materially for experiment: <relative-change>
```

The placeholders above are replaced by the versions and measured relative change from the actual rerun. A version drift is reported even when the numerical fit still happens to agree; confidence-interval drift is checked independently from RMSE and parameter drift.

## API

::: cds2.guided_fit
