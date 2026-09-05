from __future__ import annotations

import ast
import textwrap
from pathlib import Path


def replace_function(path: str, name: str, source: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
        ),
        None,
    )
    if node is None or node.end_lineno is None:
        raise RuntimeError(f"function {name!r} not found in {path}")
    lines = text.splitlines(keepends=True)
    replacement = textwrap.dedent(source).strip("\n") + "\n\n"
    lines[node.lineno - 1 : node.end_lineno] = [replacement]
    target.write_text("".join(lines), encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"marker not found in {path}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_all(path: str, old: str, new: str, expected: int) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"expected {expected} markers in {path}, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


# ---------------------------------------------------------------------------
# guided_fit: raw-data reproducibility, duplicate-name safety, statistically
# correct confidence intervals, same-set outlier diagnostics and a genuinely
# separate numerical optimizer for nonlinear cross-checks.
# ---------------------------------------------------------------------------
replace_once(
    "src/cds2/guided_fit.py",
    "from scipy import optimize as spo\n",
    "from scipy import optimize as spo, stats as sps\n",
)

replace_function(
    "src/cds2/guided_fit.py",
    "inspect_dataset",
    r"""
def inspect_dataset(dataset: FitDataset) -> dict[str, object]:
    x = np.asarray(dataset.x)
    y = np.asarray(dataset.y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if dataset.sigma is not None:
        sigma = np.asarray(dataset.sigma)
        if sigma.ndim != 1 or sigma.shape != y.shape:
            raise ValueError("sigma and y must have the same 1-D shape")
    missing_mask = ~np.isfinite(x) | ~np.isfinite(y)
    if dataset.sigma is not None:
        sigma = np.asarray(dataset.sigma, dtype=np.float64)
        missing_mask |= ~np.isfinite(sigma) | (sigma <= 0.0)
    missing_count = int(np.count_nonzero(missing_mask))
    return {
        "name": dataset.name,
        "points": int(y.size),
        "missing": missing_count,
        "suggested_missing_policy": "interpolate" if missing_count else "none",
    }
""",
)

replace_function(
    "src/cds2/guided_fit.py",
    "_prepare",
    r"""
def _prepare(dataset: FitDataset, missing_policy: MissingPolicy) -> FitDataset:
    if missing_policy not in {"drop", "interpolate"}:
        raise ValueError("missing_policy must be 'drop' or 'interpolate'")
    x: FloatArray = np.asarray(dataset.x, dtype=np.float64).copy()
    y: FloatArray = np.asarray(dataset.y, dtype=np.float64).copy()
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    sigma: FloatArray | None = (
        None if dataset.sigma is None else np.asarray(dataset.sigma, dtype=np.float64).copy()
    )
    if sigma is not None and (sigma.ndim != 1 or sigma.shape != y.shape):
        raise ValueError("sigma and y must have the same 1-D shape")

    valid_x = np.isfinite(x)
    valid_y = np.isfinite(y)
    valid_sigma = (
        np.ones_like(valid_y, dtype=bool) if sigma is None else np.isfinite(sigma) & (sigma > 0.0)
    )
    if missing_policy == "drop":
        keep = valid_x & valid_y & valid_sigma
        x, y = x[keep], y[keep]
        sigma = None if sigma is None else sigma[keep]
    else:
        keep = valid_x & valid_sigma
        x, y = x[keep], y[keep]
        sigma = None if sigma is None else sigma[keep]
        good = np.isfinite(y)
        if np.count_nonzero(good) < 2:
            raise ValueError("interpolation needs at least two finite y values")
        order = np.argsort(x, kind="stable")
        sorted_x, sorted_y = x[order], y[order]
        good_sorted = np.isfinite(sorted_y)
        sorted_y[~good_sorted] = np.interp(
            sorted_x[~good_sorted], sorted_x[good_sorted], sorted_y[good_sorted]
        )
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        y = sorted_y[inverse]
    if x.size < 4:
        raise ValueError("at least four usable points are required")
    if not bool(np.all(np.isfinite(x))) or not bool(np.all(np.isfinite(y))):
        raise ValueError("prepared x and y must be finite")
    return FitDataset(dataset.name, x, y, sigma, dataset.source_path)
""",
)

replace_function(
    "src/cds2/guided_fit.py",
    "recommend_model",
    r"""
def recommend_model(
    datasets: tuple[FitDataset, ...],
    *,
    missing_policy: MissingPolicy = "interpolate",
    seed: int = 0,
    max_pilot_points: int = 2000,
    exclude: tuple[ModelName, ...] = (),
) -> ModelRecommendation:
    if not datasets:
        raise ValueError("at least one dataset is required")
    if max_pilot_points < 4:
        raise ValueError("max_pilot_points must be at least 4")
    prepared = tuple(
        _pilot(_prepare(ds, missing_policy), max_pilot_points, seed + i)
        for i, ds in enumerate(datasets)
    )
    candidates = tuple(model for model in MODEL_NAMES if model not in exclude)
    if not candidates:
        raise ValueError("at least one candidate model is required")
    scores: dict[ModelName, float] = {}
    per_dataset: dict[ModelName, list[float]] = {}
    complexity_penalty: dict[ModelName, float] = {
        "linear": 0.0,
        "quadratic": 0.01,
        "exponential": 0.02,
        "power": 0.02,
        "logistic": 0.03,
    }
    for model in candidates:
        values: list[float] = []
        for i, ds in enumerate(prepared):
            try:
                rmse = _cv_rmse(ds, model, seed + 17 * i)
                scale = float(np.std(ds.y)) or 1.0
                values.append(rmse / scale)
            except (ValueError, RuntimeError, FloatingPointError, OverflowError):
                values.append(float("inf"))
        per_dataset[model] = values
        scores[model] = float(np.mean(values)) + complexity_penalty[model]
    model = min(scores, key=scores.__getitem__)
    if not np.isfinite(scores[model]):
        raise RuntimeError("no supported model could be fitted to the data")

    common_warning = False
    separate_models: list[tuple[str, ModelName]] = []
    keys = _dataset_keys(tuple(ds for ds in prepared))
    if len(prepared) > 1:
        for j, dataset in enumerate(prepared):
            best_model = min(
                candidates,
                key=lambda candidate: per_dataset[candidate][j] + complexity_penalty[candidate],
            )
            separate_models.append((keys[j], best_model))
            best_single = per_dataset[best_model][j]
            chosen = per_dataset[model][j]
            if np.isfinite(best_single) and chosen > 1.5 * max(best_single, 1e-12):
                common_warning = True
    speed, accuracy, simplicity = _MODEL_META[model]
    reason = f"lowest cross-validated error among supported models (score={scores[model]:.3g})"
    return ModelRecommendation(
        model,
        reason,
        speed,
        accuracy,
        simplicity,
        scores[model],
        common_warning,
        tuple(separate_models),
    )
""",
)

replace_function(
    "src/cds2/guided_fit.py",
    "_cross_check",
    r'''
def _cross_check(
    model: ModelName,
    x: FloatArray,
    y: FloatArray,
    params: FloatArray,
    sigma: FloatArray | None = None,
) -> float:
    """Cross-check fitted predictions with an independently optimized solution.

    Linear and quadratic models use NumPy's polynomial least-squares solver.
    Nonlinear models use SciPy's derivative-free Powell minimizer on a scalar
    weighted-SSE objective, rather than re-running the same least-squares
    routine from the primary solution.
    """
    f = _MODEL_FUNCS[model]
    primary = np.asarray(f(x, *params), dtype=np.float64)
    if model == "linear":
        weights = None if sigma is None else 1.0 / sigma
        p = np.polyfit(x, y, 1, w=weights)
        secondary = _linear(x, p[0], p[1])
    elif model == "quadratic":
        weights = None if sigma is None else 1.0 / sigma
        p = np.polyfit(x, y, 2, w=weights)
        secondary = _quadratic(x, p[0], p[1], p[2])
    else:
        p0, _ = _initial_guess(model, x, y)
        if np.allclose(p0, params, rtol=1e-6, atol=1e-9):
            p0 = p0.copy()
            p0 += 0.05 * np.maximum(np.abs(p0), 1.0)
            if model == "logistic":
                p0[2] = max(p0[2], 1e-6)

        def objective(candidate: FloatArray) -> float:
            prediction = np.asarray(f(x, *candidate), dtype=np.float64)
            residual = prediction - y
            if sigma is not None:
                residual = residual / sigma
            if not bool(np.all(np.isfinite(residual))):
                return float("inf")
            return float(residual @ residual)

        bounds = None
        if model == "logistic":
            bounds = [(None, None), (None, None), (0.0, None), (None, None)]
        check = spo.minimize(
            objective,
            np.asarray(p0, dtype=np.float64),
            method="Powell",
            bounds=bounds,
            options={"maxiter": 5000, "xtol": 1e-10, "ftol": 1e-10},
        )
        if not check.success or not bool(np.all(np.isfinite(check.x))):
            raise RuntimeError(f"independent cross-check failed: {check.message}")
        secondary = np.asarray(f(x, *check.x), dtype=np.float64)
    return float(np.sqrt(np.mean((primary - secondary) ** 2)))
''',
)

replace_function(
    "src/cds2/guided_fit.py",
    "_hash_dataset",
    r'''
def _hash_dataset(dataset: FitDataset) -> str:
    """Hash the raw scientific input, including array structure and sigma presence."""
    digest = hashlib.sha256()
    for label, values in (("x", dataset.x), ("y", dataset.y), ("sigma", dataset.sigma)):
        digest.update(label.encode("ascii"))
        if values is None:
            digest.update(b"<none>")
            continue
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _dataset_keys(datasets: tuple[FitDataset, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for dataset in datasets:
        counts[dataset.name] = counts.get(dataset.name, 0) + 1
    seen: dict[str, int] = {}
    keys: list[str] = []
    for dataset in datasets:
        if counts[dataset.name] == 1:
            keys.append(dataset.name)
            continue
        ordinal = seen.get(dataset.name, 0) + 1
        seen[dataset.name] = ordinal
        keys.append(f"{dataset.name}#{ordinal}")
    return tuple(keys)


def _dataset_file_stems(datasets: tuple[FitDataset, ...]) -> tuple[str, ...]:
    keys = _dataset_keys(datasets)
    return tuple(key.replace("#", "_") for key in keys)


def _confidence_interval(fit: FitResult) -> FloatArray:
    if fit.parameter_std is None:
        return np.full((fit.params.size, 2), np.nan, dtype=np.float64)
    std = np.asarray(fit.parameter_std, dtype=np.float64)
    if fit.dof is None or fit.dof <= 0:
        return np.full((fit.params.size, 2), np.nan, dtype=np.float64)
    critical = float(sps.t.ppf(0.975, fit.dof))
    return np.column_stack((fit.params - critical * std, fit.params + critical * std))


def _prediction_rmse(model: ModelName, dataset: FitDataset, params: FloatArray) -> float:
    prediction = np.asarray(_MODEL_FUNCS[model](dataset.x, *params), dtype=np.float64)
    residual = dataset.y - prediction
    return float(np.sqrt(np.mean(residual * residual)))
''',
)

replace_function(
    "src/cds2/guided_fit.py",
    "run_guided_fit",
    r"""
def run_guided_fit(
    datasets: tuple[FitDataset, ...],
    model: ModelName,
    *,
    missing_policy: MissingPolicy = "interpolate",
    outlier_policy: OutlierPolicy = "keep",
    seed: int = 0,
) -> GuidedFitResult:
    if not datasets:
        raise ValueError("at least one dataset is required")
    if model not in MODEL_NAMES:
        raise ValueError(f"unsupported model: {model!r}")
    if missing_policy not in {"drop", "interpolate"}:
        raise ValueError("missing_policy must be 'drop' or 'interpolate'")
    if outlier_policy not in {"keep", "exclude"}:
        raise ValueError("outlier_policy must be 'keep' or 'exclude'")

    results: list[DatasetResult] = []
    operations = [
        f"missing-data policy: {missing_policy}",
        f"model selected by user: {model}",
    ]
    for dataset in datasets:
        prepared = _prepare(dataset, missing_policy)
        fit = _fit_arrays(model, prepared.x, prepared.y, prepared.sigma)
        residuals = np.asarray(fit.residuals, dtype=np.float64)
        outlier_indices = _outliers(residuals)
        used = prepared
        outlier_rmse_reduction_pct = 0.0
        if outlier_indices.size:
            keep = np.ones(prepared.x.size, dtype=bool)
            keep[outlier_indices] = False
            sigma = None if prepared.sigma is None else prepared.sigma[keep]
            diagnostic_used = FitDataset(
                prepared.name,
                prepared.x[keep],
                prepared.y[keep],
                sigma,
                prepared.source_path,
            )
            if diagnostic_used.x.size < 4:
                if outlier_policy == "exclude":
                    raise ValueError("excluding detected outliers leaves fewer than four usable points")
            else:
                diagnostic_fit = _fit_arrays(
                    model, diagnostic_used.x, diagnostic_used.y, diagnostic_used.sigma
                )
                baseline_same_set = _prediction_rmse(
                    model,
                    diagnostic_used,
                    np.asarray(fit.params, dtype=np.float64),
                )
                diagnostic_rmse = cast(float, diagnostic_fit.rmse)
                scale = max(abs(baseline_same_set), np.finfo(float).eps)
                outlier_rmse_reduction_pct = (
                    100.0 * (baseline_same_set - diagnostic_rmse) / scale
                )
                if outlier_policy == "exclude":
                    used = diagnostic_used
                    fit = diagnostic_fit

        cv = _cv_rmse(used, model, seed)
        rmse = cast(float, fit.rmse)
        std = np.asarray(fit.parameter_std, dtype=np.float64)
        ci = _confidence_interval(fit)
        cross_error = _cross_check(
            model,
            used.x,
            used.y,
            np.asarray(fit.params, dtype=np.float64),
            used.sigma,
        )
        results.append(
            DatasetResult(
                used.name,
                np.asarray(fit.params, dtype=np.float64),
                std,
                ci,
                float(rmse),
                fit.r_squared,
                cv,
                outlier_indices,
                cross_error,
                int(used.x.size),
                float(np.min(used.x)),
                float(np.max(used.x)),
                float(np.mean(used.y)),
                float(np.std(used.y)),
                float(outlier_rmse_reduction_pct),
            )
        )
    operations.append(f"outlier policy: {outlier_policy}")
    operations.append("outlier influence quantified on the same retained observations")
    operations.append("3x repeated 5-fold cross-validation completed")
    if any(ds.sigma is not None for ds in datasets):
        operations.append("measurement uncertainty used in weighted fitting")
    operations.append("independent numerical cross-check completed with a separate optimizer")
    r2_values = [r.r_squared for r in results if r.r_squared is not None]
    relative_cv = [r.cv_rmse / (r.y_std or 1.0) for r in results]
    max_cross = max(r.cross_check_error for r in results)
    if (
        r2_values
        and min(r2_values) >= 0.9
        and max(relative_cv) <= 0.35
        and max_cross <= max(1e-8, 0.1 * max(r.rmse for r in results))
    ):
        trust: TrustLabel = "reliable"
        comment = "Fit is stable across held-out data and the independent numerical cross-check."
    elif (r2_values and min(r2_values) < 0.5) or max(relative_cv) > 1.0:
        trust = "unreliable"
        comment = "Fit does not generalize well; try another model or inspect data quality."
    else:
        trust = "caution"
        comment = "Fit is usable with caution; inspect residuals, uncertainty and outliers."
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    keys = _dataset_keys(datasets)
    return GuidedFitResult(
        model,
        tuple(results),
        trust,
        comment,
        seed,
        missing_policy,
        outlier_policy,
        tuple(operations),
        versions,
        {key: _hash_dataset(ds) for key, ds in zip(keys, datasets, strict=True)},
    )
""",
)

replace_function(
    "src/cds2/guided_fit.py",
    "plot_result",
    r"""
def plot_result(
    result: GuidedFitResult,
    datasets: tuple[FitDataset, ...],
    output_dir: str | Path,
) -> list[Path]:
    if len(datasets) != len(result.datasets):
        raise ValueError("datasets must match the fitted result count")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    stems = _dataset_file_stems(datasets)
    for ds, item, stem in zip(datasets, result.datasets, stems, strict=True):
        prepared = _prepare(ds, result.missing_policy)
        order = np.argsort(prepared.x)
        y_fit = _MODEL_FUNCS[result.model](prepared.x[order], *item.params)

        fig, ax = plt.subplots()
        ax.scatter(prepared.x, prepared.y, label="data")
        if item.outlier_indices.size:
            ax.scatter(
                prepared.x[item.outlier_indices],
                prepared.y[item.outlier_indices],
                marker="x",
                label="outlier",
            )
        ax.plot(prepared.x[order], y_fit, label=f"{result.model} fit")
        if prepared.sigma is not None:
            ax.errorbar(prepared.x, prepared.y, yerr=prepared.sigma, fmt="none", alpha=0.5)
        ax.set_title(f"{item.name}: {result.trust}")
        ax.legend()
        for suffix in ("png", "pdf"):
            path = target / f"{stem}_fit.{suffix}"
            fig.savefig(path, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

        predictions = np.asarray(
            _MODEL_FUNCS[result.model](prepared.x, *item.params), dtype=np.float64
        )
        residuals = prepared.y - predictions
        residual_fig, residual_ax = plt.subplots()
        residual_ax.scatter(prepared.x, residuals, label="residual")
        if item.outlier_indices.size:
            residual_ax.scatter(
                prepared.x[item.outlier_indices],
                residuals[item.outlier_indices],
                marker="x",
                label="outlier",
            )
        residual_ax.axhline(0.0, linewidth=1.0)
        residual_ax.set_xlabel("x")
        residual_ax.set_ylabel("residual")
        residual_ax.set_title(f"{item.name}: residuals")
        residual_ax.legend()
        for suffix in ("png", "pdf"):
            path = target / f"{stem}_residuals.{suffix}"
            residual_fig.savefig(path, bbox_inches="tight")
            paths.append(path)
        plt.close(residual_fig)
    return paths
""",
)

replace_function(
    "src/cds2/guided_fit.py",
    "manifest_dict",
    r"""
def manifest_dict(
    result: GuidedFitResult,
    datasets: tuple[FitDataset, ...],
    *,
    x_column: str | None = None,
    y_column: str | None = None,
    sigma_column: str | None = None,
) -> dict[str, object]:
    if len(datasets) != len(result.datasets):
        raise ValueError("datasets must match the fitted result count")
    keys = _dataset_keys(datasets)
    return {
        "result": {
            "model": result.model,
            "trust": result.trust,
            "comment": result.comment,
            "seed": result.seed,
            "missing_policy": result.missing_policy,
            "outlier_policy": result.outlier_policy,
            "operations": list(result.operations),
            "package_versions": result.package_versions,
            "data_hashes": result.data_hashes,
            "stability_warning": result.stability_warning,
            "stability_details": list(result.stability_details),
            "datasets": [
                {
                    **{
                        k: v
                        for k, v in asdict(item).items()
                        if k not in {"params", "parameter_std", "confidence_95", "outlier_indices"}
                    },
                    "dataset_key": key,
                    "params": item.params.tolist(),
                    "parameter_std": item.parameter_std.tolist(),
                    "confidence_95": item.confidence_95.tolist(),
                    "outlier_indices": item.outlier_indices.tolist(),
                }
                for key, item in zip(keys, result.datasets, strict=True)
            ],
        },
        "inputs": [
            {
                "name": ds.name,
                "dataset_key": key,
                "source_path": ds.source_path,
                "x_column": x_column,
                "y_column": y_column,
                "sigma_column": sigma_column,
            }
            for key, ds in zip(keys, datasets, strict=True)
        ],
    }
""",
)

replace_function(
    "src/cds2/guided_fit.py",
    "rerun_manifest",
    r"""
def rerun_manifest(path: str | Path) -> GuidedFitResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = payload["result"]
    datasets: list[FitDataset] = []
    for item in payload["inputs"]:
        if not item["source_path"] or not item["x_column"] or not item["y_column"]:
            raise ValueError("manifest does not contain reusable CSV source metadata")
        datasets.append(
            load_csv_dataset(
                item["source_path"],
                item["x_column"],
                item["y_column"],
                item["sigma_column"],
            )
        )
    rerun = run_guided_fit(
        tuple(datasets),
        cfg["model"],
        missing_policy=cfg["missing_policy"],
        outlier_policy=cfg["outlier_policy"],
        seed=int(cfg["seed"]),
    )

    details: list[str] = []
    previous_hashes = cast(dict[str, str], cfg["data_hashes"])
    keys = _dataset_keys(tuple(datasets))
    for index, (key, dataset) in enumerate(zip(keys, datasets, strict=True)):
        digest = rerun.data_hashes[key]
        previous_digest = previous_hashes.get(key)
        if previous_digest is None and len({item.name for item in datasets}) == len(datasets):
            previous_digest = previous_hashes.get(dataset.name)
        if previous_digest != digest:
            details.append(f"input data changed: {key}")

    saved_results = cast(list[dict[str, object]], cfg["datasets"])
    if len(saved_results) != len(rerun.datasets):
        details.append(
            f"dataset count changed: {len(saved_results)} -> {len(rerun.datasets)}"
        )
    for index, item in enumerate(rerun.datasets):
        if index >= len(saved_results):
            break
        previous = saved_results[index]
        key = keys[index]
        old_rmse = float(cast(float, previous["rmse"]))
        rmse_change = abs(item.rmse - old_rmse) / max(abs(old_rmse), 1e-12)
        old_params = np.asarray(cast(list[float], previous["params"]), dtype=np.float64)
        if old_params.shape != item.params.shape:
            details.append(f"parameter shape changed for {key}")
            continue
        param_scale = max(float(np.linalg.norm(old_params)), 1e-12)
        param_change = float(np.linalg.norm(item.params - old_params)) / param_scale
        if max(rmse_change, param_change) > 0.05:
            details.append(
                f"fit changed materially for {key}: "
                f"rmse={rmse_change:.1%}, parameters={param_change:.1%}"
            )

    if cast(str, cfg["trust"]) != rerun.trust:
        details.append(f"reliability label changed: {cfg['trust']} -> {rerun.trust}")

    return replace(rerun, stability_warning=bool(details), stability_details=tuple(details))
""",
)

# ---------------------------------------------------------------------------
# Reliability: correct right-censored Weibull likelihood and valid availability
# for MTTR >= MTBF / zero repair time; tighten scientific input validation.
# ---------------------------------------------------------------------------
replace_once(
    "src/cds2/reliability.py",
    "from scipy import stats as sp_stats\n",
    "from scipy import optimize as sp_optimize, stats as sp_stats\n",
)

replace_function(
    "src/cds2/reliability.py",
    "kaplan_meier",
    r'''
def kaplan_meier(
    durations: Sequence[float] | FloatArray,
    events: Sequence[float] | FloatArray,
) -> KMResult:
    """Kaplan-Meier product-limit estimate; events must be exactly 0 or 1."""
    durations_array = np.asarray(durations, dtype=float)
    events_array = np.asarray(events, dtype=float)
    if (
        durations_array.ndim != 1
        or events_array.ndim != 1
        or durations_array.size == 0
        or durations_array.size != events_array.size
    ):
        raise ValueError("durations and events must be equal-length non-empty series")
    if not bool(np.all(np.isfinite(durations_array))) or not bool(np.all(np.isfinite(events_array))):
        raise ValueError("durations and events must be finite")
    if np.any(durations_array < 0.0):
        raise ValueError("durations must be non-negative")
    if not bool(np.all((events_array == 0.0) | (events_array == 1.0))):
        raise ValueError("events must contain only 0 or 1")
    if not np.any(events_array == 1.0):
        raise ValueError("at least one event is required")
    failure_mask = events_array == 1.0
    times = np.unique(durations_array[failure_mask])
    survival = 1.0
    estimates: list[float] = []
    for time in times.tolist():
        at_risk = int(np.count_nonzero(durations_array >= time))
        deaths = int(np.count_nonzero((durations_array == time) & failure_mask))
        survival *= 1.0 - deaths / at_risk
        estimates.append(survival)
    return KMResult(times=np.asarray(times, dtype=float), survival=np.asarray(estimates, dtype=float))
''',
)

replace_function(
    "src/cds2/reliability.py",
    "weibull_fit",
    r'''
def weibull_fit(
    durations: Sequence[float] | FloatArray,
    failures_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
) -> WeibullFit:
    """Maximum-likelihood two-parameter Weibull fit with optional right censoring."""
    data = np.asarray(durations, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("durations must be a non-empty 1-D series")
    if not bool(np.all(np.isfinite(data))) or np.any(data < 0.0):
        raise ValueError("durations must be finite and non-negative")

    if failures_mask is None:
        failures_only = data[data > 0.0]
        if failures_only.size < 2:
            raise ValueError("at least two positive durations are required")
        shape, _loc, scale = sp_stats.weibull_min.fit(failures_only, floc=0.0)
        return WeibullFit(shape=float(shape), scale=float(scale))

    mask = np.asarray(failures_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape != data.shape:
        raise ValueError("failures_mask must match durations shape")
    failure_times = data[mask]
    if failure_times.size < 2 or np.any(failure_times <= 0.0):
        raise ValueError("at least two positive failure durations are required")

    initial_shape, _loc, initial_scale = sp_stats.weibull_min.fit(failure_times, floc=0.0)

    def negative_log_likelihood(log_params: NDArray[np.float64]) -> float:
        shape = float(np.exp(log_params[0]))
        scale = float(np.exp(log_params[1]))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cumulative_hazard = np.power(data / scale, shape)
            failure_log_density = (
                np.log(shape)
                + (shape - 1.0) * np.log(failure_times)
                - shape * np.log(scale)
            )
            value = float(np.sum(cumulative_hazard) - np.sum(failure_log_density))
        return value if np.isfinite(value) else float("inf")

    result = sp_optimize.minimize(
        negative_log_likelihood,
        np.log(np.array([initial_shape, initial_scale], dtype=np.float64)),
        method="L-BFGS-B",
    )
    if not result.success or not bool(np.all(np.isfinite(result.x))):
        raise RuntimeError(f"censored Weibull fit failed: {result.message}")
    shape, scale = np.exp(result.x)
    return WeibullFit(shape=float(shape), scale=float(scale))
''',
)

replace_function(
    "src/cds2/reliability.py",
    "mtbf",
    r'''
def mtbf(total_operating_time: float, failures: int) -> float:
    """Mean time between failures from cumulative operating time and failure count."""
    if not np.isfinite(total_operating_time) or total_operating_time <= 0.0:
        raise ValueError("total_operating_time must be positive and finite")
    if not isinstance(failures, (int, np.integer)) or isinstance(failures, bool) or failures < 1:
        raise ValueError("failures must be a positive integer")
    return float(total_operating_time / failures)
''',
)

replace_function(
    "src/cds2/reliability.py",
    "availability",
    r'''
def availability(mtbf_value: float, mttr: float) -> float:
    """Steady-state availability ``MTBF / (MTBF + MTTR)``."""
    if not np.isfinite(mtbf_value) or mtbf_value <= 0.0:
        raise ValueError("mtbf must be positive and finite")
    if not np.isfinite(mttr) or mttr < 0.0:
        raise ValueError("mttr must be non-negative and finite")
    return float(mtbf_value / (mtbf_value + mttr))
''',
)

replace_function(
    "src/cds2/reliability.py",
    "weibull_survival",
    r'''
def weibull_survival(
    time_values: Sequence[float] | FloatArray,
    shape: float,
    scale: float,
) -> FloatArray:
    """Weibull reliability function exp(-(t/scale)**shape) evaluated at each time."""
    if not np.isfinite(shape) or shape <= 0.0:
        raise ValueError("shape must be positive and finite")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    times = np.asarray(time_values, dtype=float)
    if not bool(np.all(np.isfinite(times))) or np.any(times < 0.0):
        raise ValueError("time values must be finite and non-negative")
    ratio = np.asarray(times / scale, dtype=float)
    return np.asarray(np.exp(-np.power(ratio, shape)), dtype=float)
''',
)

replace_function(
    "src/cds2/reliability.py",
    "bathtub_curve",
    r'''
def bathtub_curve(
    time_values: Sequence[float] | FloatArray,
    early_rate: float,
    intrinsic_rate: float,
    wearout_rate: float,
    knee_early: float,
    knee_wearout: float,
) -> FloatArray:
    """Bathtub hazard combining early-life decay, intrinsic floor and wear-out growth."""
    parameters = np.asarray(
        [early_rate, intrinsic_rate, wearout_rate, knee_early, knee_wearout], dtype=float
    )
    if not bool(np.all(np.isfinite(parameters))):
        raise ValueError("rates and knees must be finite")
    if min(early_rate, intrinsic_rate, wearout_rate) < 0.0:
        raise ValueError("rates must be non-negative")
    if knee_early <= 0.0 or knee_wearout <= 0.0:
        raise ValueError("knees must be positive")
    times = np.asarray(time_values, dtype=float)
    if not bool(np.all(np.isfinite(times))) or np.any(times < 0.0):
        raise ValueError("time values must be finite and non-negative")
    hazard = (
        intrinsic_rate
        + early_rate * np.exp(-times / knee_early)
        + wearout_rate * np.exp((times - knee_wearout) / knee_wearout)
    )
    return np.asarray(hazard, dtype=float)
''',
)

# ---------------------------------------------------------------------------
# Monte Carlo: fail closed on invalid sample counts/domains, preserve reversed
# integral orientation, enforce hit-or-miss envelope, validate MCMC proposals,
# and distribute exactly n_total samples across parallel workers.
# ---------------------------------------------------------------------------
replace_function(
    "src/cds2/montecarlo.py",
    "pi_estimate",
    r'''
def pi_estimate(n: int = 100_000, seed: int | None = None) -> float:
    """Estimate pi by uniform sampling inside the unit square."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    y = rng.uniform(0.0, 1.0, size=n)
    inside = np.count_nonzero(x * x + y * y <= 1.0)
    return float(4.0 * inside / n)
''',
)

replace_function(
    "src/cds2/montecarlo.py",
    "mc_integrate",
    r'''
def mc_integrate(
    func: Callable[[FloatArray], FloatArray],
    a: float,
    b: float,
    n: int = 100_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo estimate of the oriented integral of ``func`` from a to b."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("integration bounds must be finite")
    if a == b:
        return 0.0
    low, high = (a, b) if a < b else (b, a)
    orientation = 1.0 if a < b else -1.0
    rng = np.random.default_rng(seed)
    samples = rng.uniform(low, high, size=n)
    values = np.asarray(func(samples), dtype=float)
    if values.ndim == 0:
        mean_value = float(values)
    else:
        if values.shape != samples.shape:
            raise ValueError("func must return a scalar or one value per sample")
        if not bool(np.all(np.isfinite(values))):
            raise ValueError("func returned non-finite values")
        mean_value = float(np.mean(values))
    if not np.isfinite(mean_value):
        raise ValueError("func returned a non-finite value")
    return float(orientation * (high - low) * mean_value)
''',
)

replace_function(
    "src/cds2/montecarlo.py",
    "mc_expectation",
    r'''
def mc_expectation(
    func: Callable[[FloatArray], FloatArray],
    sampler: Callable[[np.random.Generator, int], FloatArray],
    n: int = 100_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo expectation E[func(X)] where ``sampler(rng, n)`` draws X."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    samples = np.asarray(sampler(rng, n), dtype=float)
    if samples.ndim == 0 or samples.shape[0] != n:
        raise ValueError("sampler must return n samples")
    if not bool(np.all(np.isfinite(samples))):
        raise ValueError("sampler returned non-finite samples")
    values = np.asarray(func(samples), dtype=float)
    if values.ndim == 0:
        result = float(values)
    else:
        if values.shape != (n,):
            raise ValueError("func must return a scalar or one value per sample")
        if not bool(np.all(np.isfinite(values))):
            raise ValueError("func returned non-finite values")
        result = float(np.mean(values))
    if not np.isfinite(result):
        raise ValueError("func returned a non-finite value")
    return result
''',
)

replace_function(
    "src/cds2/montecarlo.py",
    "hit_or_miss",
    r'''
def hit_or_miss(
    func: Callable[..., object],
    a: float,
    b: float,
    y_max: float,
    n: int = 100_000,
    seed: int | None = None,
) -> float:
    """Hit-or-miss area estimate for ``0 <= func(x) <= y_max`` on ``[a, b]``."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        raise ValueError("hit-or-miss requires finite bounds with b > a")
    if not np.isfinite(y_max) or y_max <= 0.0:
        raise ValueError("y_max must be positive and finite")
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n)
    y = rng.uniform(0.0, y_max, size=n)
    try:
        raw = func(x)
    except (TypeError, ValueError):
        f_values = np.array([func(xi) for xi in x.tolist()], dtype=float)
    else:
        candidate = np.asarray(raw, dtype=float)
        if candidate.ndim == 0:
            f_values = np.full_like(x, float(candidate), dtype=float)
        elif candidate.shape == x.shape:
            f_values = candidate
        else:
            raise ValueError("func must return a scalar or one value per sample")
    if not bool(np.all(np.isfinite(f_values))):
        raise ValueError("func returned non-finite values")
    if np.any(f_values < 0.0) or np.any(f_values > y_max):
        raise ValueError("func values must lie within [0, y_max]")
    hits = int(np.count_nonzero(y <= f_values))
    return float((b - a) * y_max * hits / n)
''',
)

replace_function(
    "src/cds2/montecarlo.py",
    "metropolis_hastings",
    r'''
def metropolis_hastings(
    log_prob: Callable[[FloatArray], float],
    initial: object,
    n_samples: int = 10_000,
    burn_in: int = 1_000,
    proposal_scale: float = 1.0,
    thin: int = 1,
    seed: int | None = None,
) -> MCMCResult:
    """Metropolis-Hastings sampler for an unnormalized log density."""
    if n_samples < 1 or burn_in < 0 or thin < 1:
        raise ValueError("need n_samples >= 1, burn_in >= 0 and thin >= 1")
    if not np.isfinite(proposal_scale) or proposal_scale <= 0.0:
        raise ValueError("proposal_scale must be positive and finite")
    rng = np.random.default_rng(seed)
    current = np.atleast_1d(np.asarray(initial, dtype=float)).copy()
    if current.ndim != 1 or current.size == 0 or not bool(np.all(np.isfinite(current))):
        raise ValueError("initial must be a non-empty finite 1-D state")
    current_log_prob = float(log_prob(current))
    if np.isnan(current_log_prob) or current_log_prob == float("inf"):
        raise ValueError("log_prob(initial) must be finite or -inf")
    total_steps = burn_in + n_samples * thin
    samples = np.empty((n_samples, current.size))
    accepted = 0
    kept = 0
    for step in range(total_steps):
        proposal = current + rng.normal(scale=proposal_scale, size=current.size)
        proposal_log_prob = float(log_prob(proposal))
        if np.isnan(proposal_log_prob) or proposal_log_prob == float("inf"):
            raise ValueError("log_prob must not return NaN or +inf")
        if current_log_prob == float("-inf"):
            accept = proposal_log_prob > float("-inf")
        elif proposal_log_prob == float("-inf"):
            accept = False
        else:
            accept = np.log(rng.random()) < proposal_log_prob - current_log_prob
        if accept:
            current = proposal
            current_log_prob = proposal_log_prob
            accepted += 1
        if step >= burn_in and (step - burn_in) % thin == 0 and kept < n_samples:
            samples[kept] = current
            kept += 1
    return MCMCResult(samples=samples, acceptance_rate=accepted / total_steps)
''',
)

replace_function(
    "src/cds2/montecarlo.py",
    "parallel_mc_integrate",
    r'''
def parallel_mc_integrate(
    func: Callable[..., object],
    a: float,
    b: float,
    n_total: int = 4_000_000,
    workers: int | None = None,
    seed: int | None = None,
) -> float:
    """Chunked Monte Carlo integration using exactly ``n_total`` samples."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    if not isinstance(n_total, (int, np.integer)) or isinstance(n_total, bool) or n_total < 1:
        raise ValueError("n_total must be a positive integer")
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        raise ValueError("b must be greater than a and both bounds must be finite")
    if workers is not None and (
        not isinstance(workers, (int, np.integer)) or isinstance(workers, bool) or workers < 1
    ):
        raise ValueError("workers must be a positive integer or None")
    requested_workers = workers if workers is not None else min(os.cpu_count() or 2, 4)
    worker_count = min(int(requested_workers), int(n_total))
    width = (b - a) / worker_count
    quotient, remainder = divmod(int(n_total), worker_count)
    counts = [quotient + (1 if index < remainder else 0) for index in range(worker_count)]
    if seed is None:
        seeds: list[int | None] = [None] * worker_count
    else:
        children = np.random.SeedSequence(seed).spawn(worker_count)
        seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    jobs = [
        (func, a + index * width, a + (index + 1) * width, counts[index], seeds[index])
        for index in range(worker_count)
    ]
    try:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:  # pragma: no cover
            estimates = list(pool.map(_integrate_chunk, jobs))
    except Exception:  # pragma: no cover - process/pickling fallback
        estimates = [_integrate_chunk(job) for job in jobs]
    return float(np.sum(estimates))
''',
)

# ---------------------------------------------------------------------------
# Statistics: reject non-finite/degenerate inputs instead of silently returning
# NaN, validate paired lengths/normal parameters/resampling counts and matrix
# orientation assumptions.
# ---------------------------------------------------------------------------
replace_function(
    "src/cds2/stats.py",
    "_as_1d",
    r"""
def _as_1d(x: object, *, min_size: int = 1) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size < min_size:
        raise ValueError(f"expected a 1-D numeric sequence with at least {min_size} value(s)")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("sample values must be finite")
    return arr


def _paired_arrays(a: object, b: object, *, min_size: int = 2) -> tuple[np.ndarray, np.ndarray]:
    x = _as_1d(a, min_size=min_size)
    y = _as_1d(b, min_size=min_size)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    return x, y


def _validate_normal(mu: float, sigma: float) -> None:
    if not np.isfinite(mu):
        raise ValueError("mu must be finite")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite")


def _contingency_table(table: object) -> np.ndarray:
    observed = np.asarray(table, dtype=float)
    if observed.ndim != 2 or min(observed.shape, default=0) < 2:
        raise ValueError("contingency table needs at least two rows and two columns")
    if not bool(np.all(np.isfinite(observed))) or np.any(observed < 0.0):
        raise ValueError("contingency counts must be finite and non-negative")
    if float(np.sum(observed)) <= 0.0:
        raise ValueError("contingency table must contain a positive total count")
    return observed


def _matrix_observations(data: object) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("data must be a 2-D matrix with at least two observations")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("data must contain only finite values")
    return values
""",
)

replace_function(
    "src/cds2/stats.py",
    "t_test",
    r'''
def t_test(sample: object, popmean: float = 0.0) -> TestResult:
    """One-sample t-test of the sample mean against ``popmean``."""
    if not np.isfinite(popmean):
        raise ValueError("popmean must be finite")
    result = sps.ttest_1samp(_as_1d(sample, min_size=2), popmean=popmean)
    return TestResult(float(result.statistic), float(result.pvalue))
''',
)

replace_function(
    "src/cds2/stats.py",
    "independent_t_test",
    r'''
def independent_t_test(a: object, b: object, equal_var: bool = True) -> TestResult:
    """Two-sample t-test; ``equal_var=False`` runs Welch's variant."""
    res = sps.ttest_ind(_as_1d(a, min_size=2), _as_1d(b, min_size=2), equal_var=equal_var)
    return TestResult(float(res.statistic), float(res.pvalue))
''',
)

replace_function(
    "src/cds2/stats.py",
    "paired_t_test",
    r'''
def paired_t_test(a: object, b: object) -> TestResult:
    """Paired-samples t-test on two related samples."""
    x, y = _paired_arrays(a, b)
    res = sps.ttest_rel(x, y)
    return TestResult(float(res.statistic), float(res.pvalue))
''',
)

replace_function(
    "src/cds2/stats.py",
    "wilcoxon_signed_rank",
    r'''
def wilcoxon_signed_rank(a: object, b: object) -> TestResult:
    """Wilcoxon signed-rank test for paired samples."""
    x, y = _paired_arrays(a, b)
    res = sps.wilcoxon(x, y)
    return TestResult(float(res.statistic), float(res.pvalue))
''',
)

replace_function(
    "src/cds2/stats.py",
    "normality_test",
    r'''
def normality_test(data: object) -> TestResult:
    """Shapiro-Wilk test of normality; requires at least three observations."""
    res = sps.shapiro(_as_1d(data, min_size=3))
    return TestResult(float(res.statistic), float(res.pvalue))
''',
)

for correlation_name, scipy_name in (
    ("pearson_correlation", "pearsonr"),
    ("spearman_correlation", "spearmanr"),
    ("kendall_tau", "kendalltau"),
):
    replace_function(
        "src/cds2/stats.py",
        correlation_name,
        f'''
def {correlation_name}(x: object, y: object) -> CorrelationResult:
    """Correlation between equal-length finite samples."""
    a, b = _paired_arrays(x, y)
    res = sps.{scipy_name}(a, b)
    return CorrelationResult(r=float(res.statistic), p_value=float(res.pvalue))
''',
    )

replace_function(
    "src/cds2/stats.py",
    "chi_square_independence",
    r'''
def chi_square_independence(table: object) -> TestResult:
    """Chi-square test of independence on a valid contingency table."""
    contingency = _contingency_table(table)
    stat, p_value, _dof, _expected = sps.chi2_contingency(contingency)
    return TestResult(float(stat), float(p_value))
''',
)

replace_function(
    "src/cds2/stats.py",
    "cohens_d",
    r'''
def cohens_d(a: object, b: object) -> float:
    """Cohen's d standardized mean difference with pooled sample SD."""
    x, y = _as_1d(a, min_size=2), _as_1d(b, min_size=2)
    nx, ny = x.size, y.size
    pooled_variance = (
        (nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)
    ) / (nx + ny - 2)
    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        raise ValueError("cohens_d is undefined when pooled variance is zero")
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled_variance))
''',
)

replace_function(
    "src/cds2/stats.py",
    "eta_squared_from_f",
    r'''
def eta_squared_from_f(f_statistic: float, df1: int, df2: int) -> float:
    """Eta-squared effect size derived from a non-negative ANOVA F statistic."""
    if not np.isfinite(f_statistic) or f_statistic < 0.0:
        raise ValueError("f_statistic must be non-negative and finite")
    if df1 <= 0 or df2 <= 0:
        raise ValueError("df1 and df2 must be positive")
    return float((f_statistic * df1) / (f_statistic * df1 + df2))
''',
)

replace_function(
    "src/cds2/stats.py",
    "cramers_v",
    r'''
def cramers_v(table: object) -> float:
    """Cramer's V association strength from a contingency table in [0, 1]."""
    observed = _contingency_table(table)
    stat, _p, _dof, _expected = sps.chi2_contingency(observed)
    n = float(observed.sum())
    denominator = min(observed.shape[0] - 1, observed.shape[1] - 1)
    return float(np.sqrt((stat / n) / denominator))
''',
)

replace_function(
    "src/cds2/stats.py",
    "percentile",
    r'''
def percentile(data: object, q: float | list[float]) -> float | list[float]:
    """Percentile(s) of a sample for ``q`` in [0, 100]."""
    values = _as_1d(data)
    q_array = np.asarray(q, dtype=float)
    if not bool(np.all(np.isfinite(q_array))) or np.any((q_array < 0.0) | (q_array > 100.0)):
        raise ValueError("q must contain values in [0, 100]")
    if isinstance(q, list):
        return [float(v) for v in np.percentile(values, q)]
    return float(np.percentile(values, q))
''',
)

replace_function(
    "src/cds2/stats.py",
    "z_scores",
    r'''
def z_scores(data: object) -> np.ndarray:
    """Standardize a sample to zero mean and unit sample standard deviation."""
    values = _as_1d(data, min_size=2)
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        raise ValueError("z_scores undefined for a constant sample")
    return np.asarray((values - np.mean(values)) / sd)
''',
)

for norm_name, scipy_name in (("norm_pdf", "pdf"), ("norm_cdf", "cdf")):
    replace_function(
        "src/cds2/stats.py",
        norm_name,
        f'''
def {norm_name}(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal distribution helper with validated location and scale."""
    _validate_normal(mu, sigma)
    if not np.isfinite(x):
        raise ValueError("x must be finite")
    return float(sps.norm.{scipy_name}(x, loc=mu, scale=sigma))
''',
    )

replace_function(
    "src/cds2/stats.py",
    "norm_ppf",
    r'''
def norm_ppf(q: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Inverse normal CDF for a probability in [0, 1]."""
    _validate_normal(mu, sigma)
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    return float(sps.norm.ppf(q, loc=mu, scale=sigma))
''',
)

replace_function(
    "src/cds2/stats.py",
    "bootstrap_ci",
    r'''
def bootstrap_ci(
    data: object,
    statistic: Callable[..., float] | None = None,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Percentile bootstrap confidence interval for an arbitrary scalar statistic."""
    values = _as_1d(data)
    if not isinstance(n_resamples, (int, np.integer)) or isinstance(n_resamples, bool) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer >= 2")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    stat_fn = statistic if statistic is not None else np.mean
    n = values.size
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n, size=(n_resamples, n))
    samples = values[resample_indices]
    try:
        estimates = np.asarray(stat_fn(samples, axis=1), dtype=float)
    except TypeError:
        estimates = np.array([float(stat_fn(sample)) for sample in samples], dtype=float)
    if estimates.shape != (n_resamples,) or not bool(np.all(np.isfinite(estimates))):
        raise ValueError("statistic must return one finite scalar per resample")
    point_estimate = float(stat_fn(values))
    if not np.isfinite(point_estimate):
        raise ValueError("statistic returned a non-finite point estimate")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return BootstrapResult(
        estimate=point_estimate,
        standard_error=float(np.std(estimates, ddof=1)),
        ci_low=float(low),
        ci_high=float(high),
    )
''',
)

replace_function(
    "src/cds2/stats.py",
    "permutation_test",
    r'''
def permutation_test(
    a: object,
    b: object,
    n_permutations: int = 10_000,
    seed: int | None = None,
) -> TestResult:
    """Two-sided permutation test on the difference of means."""
    if not isinstance(n_permutations, (int, np.integer)) or isinstance(n_permutations, bool) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")
    group_a = _as_1d(a)
    group_b = _as_1d(b)
    pooled = np.concatenate([group_a, group_b])
    nx = group_a.size
    observed = float(group_a.mean() - group_b.mean())
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((n_permutations, pooled.size)), axis=1)
    shuffled = pooled[order]
    permuted_diffs = shuffled[:, :nx].mean(axis=1) - shuffled[:, nx:].mean(axis=1)
    extreme_count = int(np.count_nonzero(np.abs(permuted_diffs) >= abs(observed)))
    p_value = (extreme_count + 1) / (n_permutations + 1)
    return TestResult(statistic=observed, p_value=float(p_value))
''',
)

replace_function(
    "src/cds2/stats.py",
    "covariance_matrix",
    r'''
def covariance_matrix(data: object) -> np.ndarray:
    """Sample covariance matrix of column-variables (rows = observations)."""
    values = _matrix_observations(data)
    return np.asarray(np.cov(values, rowvar=False, ddof=1))
''',
)

replace_function(
    "src/cds2/stats.py",
    "correlation_matrix",
    r'''
def correlation_matrix(data: object) -> np.ndarray:
    """Pearson correlation matrix of non-constant column-variables."""
    values = _matrix_observations(data)
    if np.any(np.std(values, axis=0) == 0.0):
        raise ValueError("correlation is undefined for constant variables")
    return np.asarray(np.corrcoef(values, rowvar=False))
''',
)

# StreamingStats should not let one NaN poison all future updates.
replace_once(
    "src/cds2/stats.py",
    "        if batch.size == 0:\n            return self\n        batch_count = batch.size\n",
    '        if batch.size == 0:\n            return self\n        if not bool(np.all(np.isfinite(batch))):\n            raise ValueError("chunk must contain only finite values")\n        batch_count = batch.size\n',
)

# ---------------------------------------------------------------------------
# Epidemiology: finite parameter validation, sensible subcritical herd threshold,
# bounded effective-R validation, robust final-size root and gamma=0 R0 semantics.
# ---------------------------------------------------------------------------
replace_all(
    "src/cds2/epidemiology.py",
    "        return self.beta / self.gamma\n",
    "        return math.inf if self.gamma == 0.0 else self.beta / self.gamma\n",
    expected=2,
)

replace_function(
    "src/cds2/epidemiology.py",
    "_validate_common",
    r'''
def _validate_common(
    population: float, days: int, steps_per_day: int, beta: float, gamma: float
) -> None:
    """Shared parameter validation for the compartmental simulators."""
    if not np.isfinite(population) or population <= 0.0:
        raise ValueError("population must be positive and finite")
    if not isinstance(days, (int, np.integer)) or isinstance(days, bool) or days < 1:
        raise ValueError("days must be an integer at least 1")
    if (
        not isinstance(steps_per_day, (int, np.integer))
        or isinstance(steps_per_day, bool)
        or steps_per_day < 1
    ):
        raise ValueError("steps_per_day must be an integer at least 1")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be non-negative and finite")
''',
)

# Initial-state NaN checks are otherwise bypassed by chained comparisons.
replace_once(
    "src/cds2/epidemiology.py",
    '    if not 0.0 <= i0 <= population:\n        msg = "initial infections must lie within the population"\n        raise ValueError(msg)\n',
    '    if not np.isfinite(i0) or not 0.0 <= i0 <= population:\n        msg = "initial infections must lie within the population"\n        raise ValueError(msg)\n',
)
replace_once(
    "src/cds2/epidemiology.py",
    '    if sigma <= 0.0:\n        msg = "sigma must be positive"\n        raise ValueError(msg)\n    if not 0.0 <= i0 <= population:\n',
    '    if not np.isfinite(sigma) or sigma <= 0.0:\n        msg = "sigma must be positive"\n        raise ValueError(msg)\n    if not np.isfinite(i0) or not 0.0 <= i0 <= population:\n',
)
replace_once(
    "src/cds2/epidemiology.py",
    '    if e0 < 0.0:\n        msg = "initial exposures must be non-negative"\n        raise ValueError(msg)\n',
    '    if not np.isfinite(e0) or e0 < 0.0:\n        msg = "initial exposures must be non-negative"\n        raise ValueError(msg)\n',
)

replace_function(
    "src/cds2/epidemiology.py",
    "herd_immunity_threshold",
    r'''
def herd_immunity_threshold(r0: float) -> float:
    """Critical immune fraction; zero when transmission is already subcritical."""
    if not np.isfinite(r0) or r0 <= 0.0:
        raise ValueError("r0 must be positive and finite")
    if r0 <= 1.0:
        return 0.0
    return float(1.0 - 1.0 / r0)
''',
)

replace_function(
    "src/cds2/epidemiology.py",
    "effective_reproduction",
    r'''
def effective_reproduction(r0_value: float, susceptible_fraction: float) -> float:
    """Effective reproduction number ``r0 * susceptible_fraction``."""
    if not np.isfinite(r0_value) or r0_value < 0.0:
        raise ValueError("r0_value must be non-negative and finite")
    if not np.isfinite(susceptible_fraction) or not 0.0 <= susceptible_fraction <= 1.0:
        raise ValueError("susceptible_fraction must be in [0, 1]")
    return float(r0_value * susceptible_fraction)
''',
)

replace_function(
    "src/cds2/epidemiology.py",
    "final_size_iteration",
    r'''
def final_size_iteration(r0: float, tol: float = 1e-10, max_iter: int = 200) -> float:
    """Final epidemic size solving ``z = 1 - exp(-r0*z)`` for the nonzero root."""
    if not np.isfinite(r0) or r0 < 0.0:
        raise ValueError("r0 must be non-negative and finite")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be positive and finite")
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if r0 <= 1.0:
        return 0.0

    def objective(z: float) -> float:
        return z - (1.0 - math.exp(-r0 * z))

    epsilon = min(1e-8, 0.1 * (r0 - 1.0) / r0)
    try:
        return float(
            __import__("scipy").optimize.brentq(
                objective,
                epsilon,
                1.0 - np.finfo(float).eps,
                xtol=tol,
                rtol=max(4.0 * np.finfo(float).eps, tol),
                maxiter=max_iter,
            )
        )
    except RuntimeError as exc:
        raise RuntimeError("final-size iteration did not converge") from exc
''',
)

# ---------------------------------------------------------------------------
# Update old tests that encoded the two corrected reliability semantics.
# ---------------------------------------------------------------------------
replace_once(
    "tests/test_reliability.py",
    """    def test_mask_excludes_censored_entries(self) -> None:\n        data = sp_stats.weibull_min.rvs(c=1.5, scale=1000.0, size=400, random_state=1)\n        padded = np.concatenate([data, [10.0, 9999.0]])\n        mask = np.concatenate([np.ones(400, dtype=bool), [False, False]])\n        masked_fit = rel.weibull_fit(padded, failures_mask=mask)\n        reference = rel.weibull_fit(data)\n        assert masked_fit.shape == pytest.approx(reference.shape, rel=1e-12)\n        assert masked_fit.scale == pytest.approx(reference.scale, rel=1e-12)\n        assert masked_fit.shape == pytest.approx(1.5, rel=0.20)\n""",
    """    def test_mask_uses_right_censored_likelihood(self) -> None:\n        failures = np.array([100.0, 180.0, 250.0, 320.0, 400.0])\n        durations = np.concatenate([failures, [900.0, 1000.0, 1100.0]])\n        mask = np.array([True] * failures.size + [False, False, False])\n        censored_fit = rel.weibull_fit(durations, failures_mask=mask)\n        failure_only = rel.weibull_fit(failures)\n        assert censored_fit.shape > 0.0\n        assert censored_fit.scale > failure_only.scale\n""",
)

replace_once(
    "tests/test_reliability.py",
    """    @pytest.mark.parametrize("mttr", [100.0, 150.0])\n    def test_availability_mttr_not_below_mtbf_raises(self, mttr: float) -> None:\n        with pytest.raises(ValueError, match="mttr must be smaller than mtbf"):\n            rel.availability(100.0, mttr)\n""",
    """    def test_availability_allows_long_or_zero_repair_time(self) -> None:\n        assert rel.availability(100.0, 150.0) == pytest.approx(0.4)\n        assert rel.availability(100.0, 0.0) == pytest.approx(1.0)\n""",
)

replace_once(
    "tests/test_reliability.py",
    """        [(0.0, 1.0), (-1.0, 1.0), (100.0, 0.0), (100.0, -1.0)],\n""",
    """        [(0.0, 1.0), (-1.0, 1.0), (100.0, -1.0)],\n""",
)

# Add oracle/edge-case tests for this hardening wave.
Path("tests/test_correctness_hardening_wave2.py").write_text(
    r'''"""Correctness regression tests for hardening wave 2."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from cds2 import epidemiology as epi
from cds2 import guided_fit as gf
from cds2 import montecarlo as mc
from cds2 import reliability as rel
from cds2 import stats


def _constant_two(x):  # type: ignore[no-untyped-def]
    return np.full_like(x, 2.0, dtype=float)


def test_guided_fit_hashes_raw_inputs_not_only_interpolated_values() -> None:
    x = np.arange(1.0, 7.0)
    with_missing = gf.FitDataset("same", x, np.array([2.0, 4.0, np.nan, 8.0, 10.0, 12.0]))
    explicit = gf.FitDataset("same", x, np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]))
    first = gf.run_guided_fit((with_missing,), "linear", missing_policy="interpolate")
    second = gf.run_guided_fit((explicit,), "linear", missing_policy="interpolate")
    assert first.data_hashes["same"] != second.data_hashes["same"]


def test_guided_fit_duplicate_names_do_not_collide(tmp_path: Path) -> None:
    x = np.linspace(1.0, 8.0, 30)
    datasets = (
        gf.FitDataset("duplicate", x, 2.0 * x + 1.0),
        gf.FitDataset("duplicate", x, 3.0 * x - 2.0),
    )
    result = gf.run_guided_fit(datasets, "linear")
    assert set(result.data_hashes) == {"duplicate#1", "duplicate#2"}
    paths = gf.plot_result(result, datasets, tmp_path)
    assert len(paths) == 8
    assert len({path.name for path in paths}) == 8


def test_guided_fit_duplicate_manifest_reruns_by_position(tmp_path: Path) -> None:
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    x = np.linspace(1.0, 8.0, 30)
    p1 = first_dir / "data.csv"
    p2 = second_dir / "data.csv"
    pd.DataFrame({"x": x, "y": 2.0 * x + 1.0}).to_csv(p1, index=False)
    pd.DataFrame({"x": x, "y": 4.0 * x + 3.0}).to_csv(p2, index=False)
    datasets = (gf.load_csv_dataset(p1, "x", "y"), gf.load_csv_dataset(p2, "x", "y"))
    result = gf.run_guided_fit(datasets, "linear")
    manifest = gf.save_manifest(result, datasets, tmp_path / "manifest.json", x_column="x", y_column="y")
    stable = gf.rerun_manifest(manifest)
    assert stable.stability_warning is False
    pd.DataFrame({"x": x, "y": 8.0 * x - 7.0}).to_csv(p2, index=False)
    changed = gf.rerun_manifest(manifest)
    assert changed.stability_warning is True
    assert any("duplicate#2" in detail for detail in changed.stability_details)


def test_guided_fit_outlier_effect_uses_same_evaluation_set() -> None:
    x = np.linspace(1.0, 10.0, 60)
    y = 2.5 * x + 1.0
    y[30] += 40.0
    result = gf.run_guided_fit((gf.FitDataset("outlier", x, y),), "linear")
    effect = result.datasets[0].outlier_rmse_reduction_pct
    assert 90.0 < effect <= 100.000001


def test_guided_fit_confidence_interval_uses_student_t() -> None:
    rng = np.random.default_rng(5)
    x = np.linspace(0.0, 4.0, 10)
    y = 1.5 * x + 2.0 + rng.normal(scale=0.5, size=x.size)
    item = gf.run_guided_fit((gf.FitDataset("small", x, y),), "linear").datasets[0]
    critical = (item.confidence_95[:, 1] - item.params) / item.parameter_std
    expected = sps.t.ppf(0.975, x.size - 2)
    assert critical.tolist() == pytest.approx([expected, expected], rel=1e-7)


def test_reliability_validates_events_and_censored_weibull() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        rel.kaplan_meier([1.0, 2.0], [1, 2])
    failures = np.array([100.0, 150.0, 220.0, 300.0])
    durations = np.concatenate([failures, [800.0, 900.0]])
    fit = rel.weibull_fit(durations, [True, True, True, True, False, False])
    assert fit.shape > 0.0 and fit.scale > 0.0
    assert rel.availability(100.0, 150.0) == pytest.approx(0.4)
    assert rel.availability(100.0, 0.0) == 1.0


def test_montecarlo_domain_and_count_validation() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        mc.pi_estimate(0)
    with pytest.raises(ValueError, match="positive integer"):
        mc.mc_integrate(lambda x: x, 0.0, 1.0, n=0)
    assert mc.mc_integrate(lambda x: x, 1.0, 0.0, n=200_000, seed=4) == pytest.approx(-0.5, abs=0.005)
    with pytest.raises(ValueError, match="\[0, y_max\]"):
        mc.hit_or_miss(lambda _x: 2.0, 0.0, 1.0, 1.0, n=100, seed=1)
    with pytest.raises(ValueError, match="proposal_scale"):
        mc.metropolis_hastings(lambda v: -0.5 * v[0] ** 2, [0.0], proposal_scale=0.0)


def test_parallel_mc_uses_tiny_exact_budget_without_zero_worker_chunks() -> None:
    assert mc.parallel_mc_integrate(_constant_two, 0.0, 1.0, n_total=3, workers=8, seed=3) == pytest.approx(2.0)


def test_stats_fail_closed_on_undefined_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        stats.describe([1.0, np.nan])
    with pytest.raises(ValueError, match="same length"):
        stats.paired_t_test([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        stats.z_scores([1.0])
    with pytest.raises(ValueError, match="sigma"):
        stats.norm_pdf(0.0, sigma=0.0)
    with pytest.raises(ValueError, match="2-D"):
        stats.covariance_matrix([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="pooled variance"):
        stats.cohens_d([1.0, 1.0], [2.0, 2.0])
    with pytest.raises(ValueError, match="n_resamples"):
        stats.bootstrap_ci([1.0, 2.0], n_resamples=1)
    with pytest.raises(ValueError, match="n_permutations"):
        stats.permutation_test([1.0], [2.0], n_permutations=0)


def test_epidemiology_thresholds_and_validation() -> None:
    assert epi.herd_immunity_threshold(0.5) == 0.0
    with pytest.raises(ValueError, match="susceptible_fraction"):
        epi.effective_reproduction(2.0, 1.1)
    with pytest.raises(ValueError, match="r0_value"):
        epi.effective_reproduction(-1.0, 0.5)
    with pytest.raises(ValueError, match="r0"):
        epi.final_size_iteration(-1.0)
    near_critical = epi.final_size_iteration(1.01)
    assert 0.0 < near_critical < 0.1
    no_recovery = epi.simulate_sir(100.0, 0.2, 0.0, 2)
    assert math.isinf(no_recovery.r0)
''',
    encoding="utf-8",
)

print("Applied correctness hardening wave 2")
