"""Guided, user-controlled scientific model fitting.

The workflow recommends one model, but never silently chooses user-facing
decisions such as model selection, missing-data treatment or outlier removal.
"""

from __future__ import annotations

import hashlib
import json
import platform
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from scipy import optimize as spo
from scipy import stats as sps

from .optimize import FitResult, curve_fit

ModelName = Literal["linear", "quadratic", "exponential", "power", "logistic"]
MissingPolicy = Literal["drop", "interpolate"]
OutlierPolicy = Literal["keep", "exclude"]
TrustLabel = Literal["reliable", "caution", "unreliable"]
FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]
ModelFunc = Callable[..., FloatArray]
Bounds = tuple[Sequence[float] | float, Sequence[float] | float]

MODEL_NAMES: tuple[ModelName, ...] = (
    "linear",
    "quadratic",
    "exponential",
    "power",
    "logistic",
)


@dataclass(frozen=True)
class FitDataset:
    name: str
    x: FloatArray
    y: FloatArray
    sigma: FloatArray | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class ModelRecommendation:
    model: ModelName
    reason: str
    speed: str
    accuracy: str
    simplicity: str
    score: float
    common_model_warning: bool
    separate_models: tuple[tuple[str, ModelName], ...] = ()


@dataclass(frozen=True)
class DatasetResult:
    name: str
    params: FloatArray
    parameter_std: FloatArray
    confidence_95: FloatArray
    rmse: float
    r_squared: float | None
    cv_rmse: float
    outlier_indices: IndexArray
    cross_check_error: float
    n_points: int
    x_min: float
    x_max: float
    y_mean: float
    y_std: float
    outlier_rmse_reduction_pct: float = 0.0


@dataclass(frozen=True)
class GuidedFitResult:
    model: ModelName
    datasets: tuple[DatasetResult, ...]
    trust: TrustLabel
    comment: str
    seed: int
    missing_policy: MissingPolicy
    outlier_policy: OutlierPolicy
    operations: tuple[str, ...]
    package_versions: dict[str, str]
    data_hashes: dict[str, str]
    stability_warning: bool = False
    stability_details: tuple[str, ...] = ()


def _linear(x: FloatArray, a: float, b: float) -> FloatArray:
    return np.asarray(a * x + b, dtype=np.float64)


def _quadratic(x: FloatArray, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * x * x + b * x + c, dtype=np.float64)


def _exponential(x: FloatArray, a: float, b: float, c: float) -> FloatArray:
    values = a * np.exp(np.clip(b * x, -700.0, 700.0)) + c
    return np.asarray(values, dtype=np.float64)


def _power(x: FloatArray, a: float, b: float, c: float) -> FloatArray:
    values = a * np.power(np.maximum(x, np.finfo(float).tiny), b) + c
    return np.asarray(values, dtype=np.float64)


def _logistic(x: FloatArray, low: float, high: float, k: float, x0: float) -> FloatArray:
    values = low + (high - low) / (1.0 + np.exp(np.clip(-k * (x - x0), -700.0, 700.0)))
    return np.asarray(values, dtype=np.float64)


_MODEL_FUNCS: dict[ModelName, ModelFunc] = {
    "linear": _linear,
    "quadratic": _quadratic,
    "exponential": _exponential,
    "power": _power,
    "logistic": _logistic,
}
_MODEL_META = {
    "linear": ("fastest", "best for straight trends", "simplest"),
    "quadratic": ("very fast", "captures one smooth bend", "simple"),
    "exponential": ("fast", "best for growth/decay", "moderate"),
    "power": ("fast", "best for scale-law behaviour", "moderate"),
    "logistic": ("moderate", "best for bounded S-shaped trends", "more complex"),
}


def load_csv_dataset(
    path: str | Path,
    x_column: str,
    y_column: str,
    sigma_column: str | None = None,
) -> FitDataset:
    frame = pd.read_csv(path)
    required = [x_column, y_column] + ([] if sigma_column is None else [sigma_column])
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    sigma = (
        None
        if sigma_column is None
        else np.asarray(frame[sigma_column].to_numpy(dtype=np.float64), dtype=np.float64)
    )
    return FitDataset(
        name=Path(path).stem,
        x=np.asarray(frame[x_column].to_numpy(dtype=np.float64), dtype=np.float64),
        y=np.asarray(frame[y_column].to_numpy(dtype=np.float64), dtype=np.float64),
        sigma=sigma,
        source_path=str(Path(path)),
    )


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


def _initial_guess(model: ModelName, x: FloatArray, y: FloatArray) -> tuple[FloatArray, Bounds]:
    if model == "linear":
        p = np.polyfit(x, y, 1)
        return np.array([p[0], p[1]]), (-np.inf, np.inf)
    if model == "quadratic":
        p = np.polyfit(x, y, 2)
        return np.array([p[0], p[1], p[2]]), (-np.inf, np.inf)
    if model == "exponential":
        return np.array([max(float(np.ptp(y)), 1.0), 0.01, float(np.min(y))]), (
            -np.inf,
            np.inf,
        )
    if model == "power":
        return np.array([1.0, 1.0, float(np.min(y))]), (-np.inf, np.inf)
    return np.array([float(np.min(y)), float(np.max(y)), 1.0, float(np.median(x))]), (
        [-np.inf, -np.inf, 0.0, -np.inf],
        [np.inf, np.inf, np.inf, np.inf],
    )


def _fit_arrays(
    model: ModelName, x: FloatArray, y: FloatArray, sigma: FloatArray | None = None
) -> FitResult:
    if model == "power" and np.any(x <= 0.0):
        raise ValueError("power model requires x > 0")
    p0, bounds = _initial_guess(model, x, y)
    return curve_fit(
        cast(Callable[..., object], _MODEL_FUNCS[model]),
        x.tolist(),
        y.tolist(),
        p0=p0.tolist(),
        sigma=None if sigma is None else sigma.tolist(),
        absolute_sigma=sigma is not None,
        bounds=bounds,
        method=None,
    )


def _pilot(dataset: FitDataset, max_points: int, seed: int) -> FitDataset:
    if dataset.x.size <= max_points:
        return dataset
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(dataset.x.size, size=max_points, replace=False))
    sigma = None if dataset.sigma is None else dataset.sigma[idx]
    return FitDataset(dataset.name, dataset.x[idx], dataset.y[idx], sigma, dataset.source_path)


def _cv_rmse(dataset: FitDataset, model: ModelName, seed: int, folds: int = 5) -> float:
    n = dataset.x.size
    folds = min(max(2, folds), n)
    rng = np.random.default_rng(seed)
    errors: list[float] = []
    for _ in range(3):
        indices = rng.permutation(n)
        for fold in np.array_split(indices, folds):
            train = np.setdiff1d(indices, fold, assume_unique=True)
            if train.size < 4 or fold.size == 0:
                continue
            sigma_train = None if dataset.sigma is None else dataset.sigma[train]
            fit = _fit_arrays(model, dataset.x[train], dataset.y[train], sigma_train)
            pred = np.asarray(_MODEL_FUNCS[model](dataset.x[fold], *fit.params), dtype=np.float64)
            errors.extend((dataset.y[fold] - pred).tolist())
    if not errors:
        return float("inf")
    arr = np.asarray(errors, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


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


def _outliers(residuals: FloatArray) -> IndexArray:
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    if mad == 0.0:
        std = float(np.std(residuals))
        if std == 0.0:
            return np.zeros(0, dtype=np.intp)
        z_score = (residuals - float(np.mean(residuals))) / std
        return np.flatnonzero(np.abs(z_score) > 3.5)
    robust_z = 0.67448975 * (residuals - median) / mad
    return np.flatnonzero(np.abs(robust_z) > 3.5)


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
    reserved = {name for name, count in counts.items() if count == 1}
    seen: dict[str, int] = {}
    used: set[str] = set()
    keys: list[str] = []
    for dataset in datasets:
        name = dataset.name
        if counts[name] == 1:
            key = name
        else:
            ordinal = seen.get(name, 0) + 1
            seen[name] = ordinal
            base = f"{name}#{ordinal}"
            key = base
            suffix = 2
            while key in used or key in reserved:
                key = f"{base}~{suffix}"
                suffix += 1
        used.add(key)
        keys.append(key)
    return tuple(keys)


def _dataset_file_stems(datasets: tuple[FitDataset, ...]) -> tuple[str, ...]:
    def safe(value: str) -> str:
        cleaned = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value
        ).strip(" .")
        return (cleaned or "dataset")[:120]

    keys = _dataset_keys(datasets)
    counts: dict[str, int] = {}
    for dataset in datasets:
        counts[dataset.name] = counts.get(dataset.name, 0) + 1
    reserved = {safe(dataset.name) for dataset in datasets if counts[dataset.name] == 1}
    used: set[str] = set()
    stems: list[str] = []
    for dataset, key in zip(datasets, keys, strict=True):
        base = safe(dataset.name if counts[dataset.name] == 1 else key)
        stem = base
        suffix = 2
        while stem in used or (counts[dataset.name] > 1 and stem in reserved):
            stem = f"{base}__{suffix}"
            suffix += 1
        used.add(stem)
        stems.append(stem)
    return tuple(stems)


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
                    raise ValueError(
                        "excluding detected outliers leaves fewer than four usable points"
                    )
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
                outlier_rmse_reduction_pct = 100.0 * (baseline_same_set - diagnostic_rmse) / scale
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
    rerun_hashes = dict(result.data_hashes)
    if x_column and y_column:
        for key, dataset in zip(keys, datasets, strict=True):
            if dataset.source_path:
                source_dataset = load_csv_dataset(
                    dataset.source_path,
                    x_column,
                    y_column,
                    sigma_column,
                )
                source_dataset = replace(source_dataset, name=dataset.name)
                rerun_hashes[key] = _hash_dataset(source_dataset)
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
            "rerun_data_hashes": rerun_hashes,
            "stability_warning": result.stability_warning,
            "stability_details": list(result.stability_details),
            "datasets": [
                {
                    **{
                        k: v
                        for k, v in asdict(dataset_result).items()
                        if k
                        not in {
                            "params",
                            "parameter_std",
                            "confidence_95",
                            "outlier_indices",
                        }
                    },
                    "dataset_key": key,
                    "params": dataset_result.params.tolist(),
                    "parameter_std": dataset_result.parameter_std.tolist(),
                    "confidence_95": dataset_result.confidence_95.tolist(),
                    "outlier_indices": dataset_result.outlier_indices.tolist(),
                }
                for key, dataset_result in zip(keys, result.datasets, strict=True)
            ],
        },
        "inputs": [
            {
                "name": dataset.name,
                "dataset_key": key,
                "source_path": dataset.source_path,
                "x_column": x_column,
                "y_column": y_column,
                "sigma_column": sigma_column,
            }
            for key, dataset in zip(keys, datasets, strict=True)
        ],
    }


def save_manifest(
    result: GuidedFitResult,
    datasets: tuple[FitDataset, ...],
    path: str | Path,
    *,
    x_column: str | None = None,
    y_column: str | None = None,
    sigma_column: str | None = None,
) -> Path:
    target = Path(path)
    target.write_text(
        json.dumps(
            manifest_dict(
                result,
                datasets,
                x_column=x_column,
                y_column=y_column,
                sigma_column=sigma_column,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def rerun_manifest(path: str | Path) -> GuidedFitResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = payload["result"]
    input_items = cast(list[dict[str, object]], payload["inputs"])
    datasets: list[FitDataset] = []
    for input_item in input_items:
        source_path = cast(str | None, input_item.get("source_path"))
        x_column = cast(str | None, input_item.get("x_column"))
        y_column = cast(str | None, input_item.get("y_column"))
        sigma_column = cast(str | None, input_item.get("sigma_column"))
        if not source_path or not x_column or not y_column:
            raise ValueError("manifest does not contain reusable CSV source metadata")
        loaded = load_csv_dataset(source_path, x_column, y_column, sigma_column)
        datasets.append(replace(loaded, name=str(input_item.get("name") or loaded.name)))

    rerun = run_guided_fit(
        tuple(datasets),
        cfg["model"],
        missing_policy=cfg["missing_policy"],
        outlier_policy=cfg["outlier_policy"],
        seed=int(cfg["seed"]),
    )

    details: list[str] = []
    previous_hashes = cast(
        dict[str, str],
        cfg.get("rerun_data_hashes", cfg["data_hashes"]),
    )
    runtime_keys = _dataset_keys(tuple(datasets))
    saved_keys = tuple(
        str(input_item.get("dataset_key") or runtime_key)
        for input_item, runtime_key in zip(input_items, runtime_keys, strict=True)
    )
    for runtime_key, saved_key, dataset in zip(runtime_keys, saved_keys, datasets, strict=True):
        digest = rerun.data_hashes[runtime_key]
        previous_digest = previous_hashes.get(saved_key)
        if previous_digest is None:
            previous_digest = previous_hashes.get(runtime_key)
        if previous_digest is None and len({value.name for value in datasets}) == len(datasets):
            previous_digest = previous_hashes.get(dataset.name)
        if previous_digest != digest:
            details.append(f"input data changed: {saved_key}")

    saved_results = cast(list[dict[str, object]], cfg["datasets"])
    if len(saved_results) != len(rerun.datasets):
        details.append(f"dataset count changed: {len(saved_results)} -> {len(rerun.datasets)}")
    for index, dataset_result in enumerate(rerun.datasets):
        if index >= len(saved_results):
            break
        previous = saved_results[index]
        key = saved_keys[index] if index < len(saved_keys) else runtime_keys[index]
        old_rmse = float(cast(float, previous["rmse"]))
        rmse_change = abs(dataset_result.rmse - old_rmse) / max(abs(old_rmse), 1e-12)
        old_params = np.asarray(cast(list[float], previous["params"]), dtype=np.float64)
        if old_params.shape != dataset_result.params.shape:
            details.append(f"parameter shape changed for {key}")
            continue
        param_scale = max(float(np.linalg.norm(old_params)), 1e-12)
        param_change = float(np.linalg.norm(dataset_result.params - old_params)) / param_scale
        if max(rmse_change, param_change) > 0.05:
            details.append(
                f"fit changed materially for {key}: "
                f"rmse={rmse_change:.1%}, parameters={param_change:.1%}"
            )

    if cast(str, cfg["trust"]) != rerun.trust:
        details.append(f"reliability label changed: {cfg['trust']} -> {rerun.trust}")

    return replace(
        rerun,
        stability_warning=bool(details),
        stability_details=tuple(details),
    )


def _paginate_report_text(
    text: str,
    *,
    width: int = 100,
    lines_per_page: int = 56,
) -> tuple[str, ...]:
    """Wrap report text into complete PDF pages without dropping content."""
    wrapped_lines: list[str] = []
    for line in text.splitlines():
        chunks = textwrap.wrap(
            line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped_lines.extend(chunks or [""])
    return tuple(
        "\n".join(wrapped_lines[start : start + lines_per_page])
        for start in range(0, len(wrapped_lines), lines_per_page)
    )


def write_report(
    result: GuidedFitResult,
    output_dir: str | Path,
    report_format: Literal["markdown", "html", "pdf"],
) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Guided scientific fit report",
        "",
        f"- Model: {result.model}",
        f"- Reliability: {result.trust}",
        f"- Interpretation: {result.comment}",
        f"- Seed: {result.seed}",
        "",
        "## Operations",
        *[f"- {op}" for op in result.operations],
        "",
        "## Results",
    ]
    for item in result.datasets:
        lines.extend(
            [
                f"### {item.name}",
                f"- Points: {item.n_points}",
                f"- x range: [{item.x_min:.6g}, {item.x_max:.6g}]",
                f"- y mean/std: {item.y_mean:.6g} / {item.y_std:.6g}",
                f"- RMSE: {item.rmse:.6g}",
                f"- CV RMSE: {item.cv_rmse:.6g}",
                f"- R²: {'undefined' if item.r_squared is None else f'{item.r_squared:.6g}'}",
                f"- Cross-check error: {item.cross_check_error:.6g}",
                f"- Outliers detected: {item.outlier_indices.tolist()}",
                f"- Estimated RMSE reduction without detected outliers: {item.outlier_rmse_reduction_pct:.2f}%",
                f"- Parameters: {item.params.tolist()}",
                f"- 95% confidence intervals: {item.confidence_95.tolist()}",
            ]
        )
    text = "\n".join(lines) + "\n"
    if report_format == "markdown":
        path = target_dir / "guided_fit_report.md"
        path.write_text(text, encoding="utf-8")
        return path
    if report_format == "html":
        import html

        path = target_dir / "guided_fit_report.html"
        path.write_text(
            f"<html><body><pre>{html.escape(text)}</pre></body></html>",
            encoding="utf-8",
        )
        return path
    from matplotlib.backends.backend_pdf import PdfPages

    path = target_dir / "guided_fit_report.pdf"
    with PdfPages(path) as pdf:
        for page_text in _paginate_report_text(text):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.05, 0.95, page_text, va="top", family="monospace", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
    return path


__all__ = [
    "DatasetResult",
    "FitDataset",
    "GuidedFitResult",
    "MODEL_NAMES",
    "ModelRecommendation",
    "inspect_dataset",
    "load_csv_dataset",
    "manifest_dict",
    "plot_result",
    "recommend_model",
    "rerun_manifest",
    "run_guided_fit",
    "save_manifest",
    "write_report",
]
