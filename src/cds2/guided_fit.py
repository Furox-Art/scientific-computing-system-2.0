"""Guided, user-controlled scientific model fitting.

The workflow recommends one model, but never silently chooses user-facing
decisions such as model selection, missing-data treatment or outlier removal.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import optimize as spo

from .optimize import curve_fit

ModelName = Literal["linear", "quadratic", "exponential", "power", "logistic"]
MissingPolicy = Literal["drop", "interpolate"]
OutlierPolicy = Literal["keep", "exclude"]
TrustLabel = Literal["reliable", "caution", "unreliable"]

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
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray | None = None
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


@dataclass(frozen=True)
class DatasetResult:
    name: str
    params: np.ndarray
    parameter_std: np.ndarray
    confidence_95: np.ndarray
    rmse: float
    r_squared: float | None
    cv_rmse: float
    outlier_indices: np.ndarray
    cross_check_error: float
    n_points: int
    x_min: float
    x_max: float
    y_mean: float
    y_std: float


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


def _linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def _quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x * x + b * x + c


def _exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(np.clip(b * x, -700.0, 700.0)) + c


def _power(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(np.maximum(x, np.finfo(float).tiny), b) + c


def _logistic(x: np.ndarray, low: float, high: float, k: float, x0: float) -> np.ndarray:
    return low + (high - low) / (
        1.0 + np.exp(np.clip(-k * (x - x0), -700.0, 700.0))
    )


_MODEL_FUNCS = {
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
    sigma = None if sigma_column is None else frame[sigma_column].to_numpy(dtype=float)
    return FitDataset(
        name=Path(path).stem,
        x=frame[x_column].to_numpy(dtype=float),
        y=frame[y_column].to_numpy(dtype=float),
        sigma=sigma,
        source_path=str(Path(path)),
    )


def inspect_dataset(dataset: FitDataset) -> dict[str, object]:
    if dataset.x.shape != dataset.y.shape:
        raise ValueError("x and y must have the same shape")
    if dataset.sigma is not None and dataset.sigma.shape != dataset.y.shape:
        raise ValueError("sigma and y must have the same shape")
    missing_mask = ~np.isfinite(dataset.x) | ~np.isfinite(dataset.y)
    if dataset.sigma is not None:
        missing_mask |= ~np.isfinite(dataset.sigma) | (dataset.sigma <= 0.0)
    missing_count = int(np.count_nonzero(missing_mask))
    return {
        "name": dataset.name,
        "points": int(dataset.y.size),
        "missing": missing_count,
        "suggested_missing_policy": "interpolate" if missing_count else "none",
    }


def _prepare(dataset: FitDataset, missing_policy: MissingPolicy) -> FitDataset:
    x = np.asarray(dataset.x, dtype=float).copy()
    y = np.asarray(dataset.y, dtype=float).copy()
    sigma = None if dataset.sigma is None else np.asarray(dataset.sigma, dtype=float).copy()
    valid_x = np.isfinite(x)
    valid_y = np.isfinite(y)
    valid_sigma = (
        np.ones_like(valid_y, dtype=bool)
        if sigma is None
        else np.isfinite(sigma) & (sigma > 0.0)
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
        order = np.argsort(x)
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
    return FitDataset(dataset.name, x, y, sigma, dataset.source_path)


def _initial_guess(
    model: ModelName, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, tuple[object, object]]:
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
    return np.array(
        [float(np.min(y)), float(np.max(y)), 1.0, float(np.median(x))]
    ), (
        [-np.inf, -np.inf, 0.0, -np.inf],
        [np.inf, np.inf, np.inf, np.inf],
    )


def _fit_arrays(
    model: ModelName, x: np.ndarray, y: np.ndarray, sigma: np.ndarray | None = None
):
    if model == "power" and np.any(x <= 0.0):
        raise ValueError("power model requires x > 0")
    p0, bounds = _initial_guess(model, x, y)
    return curve_fit(
        _MODEL_FUNCS[model],
        x,
        y,
        p0=p0,
        sigma=sigma,
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
            pred = np.asarray(_MODEL_FUNCS[model](dataset.x[fold], *fit.params), dtype=float)
            errors.extend((dataset.y[fold] - pred).tolist())
    if not errors:
        return float("inf")
    arr = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def recommend_model(
    datasets: tuple[FitDataset, ...],
    *,
    missing_policy: MissingPolicy = "interpolate",
    seed: int = 0,
    max_pilot_points: int = 2000,
    exclude: tuple[ModelName, ...] = (),
) -> ModelRecommendation:
    prepared = tuple(
        _pilot(_prepare(ds, missing_policy), max_pilot_points, seed + i)
        for i, ds in enumerate(datasets)
    )
    candidates = tuple(model for model in MODEL_NAMES if model not in exclude)
    if not candidates:
        raise ValueError("at least one candidate model is required")
    scores: dict[ModelName, float] = {}
    per_dataset: dict[ModelName, list[float]] = {}
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
        complexity = {
            "linear": 0.0,
            "quadratic": 0.01,
            "exponential": 0.02,
            "power": 0.02,
            "logistic": 0.03,
        }[model]
        scores[model] = float(np.mean(values)) + complexity
    model = min(scores, key=scores.__getitem__)
    if not np.isfinite(scores[model]):
        raise RuntimeError("no supported model could be fitted to the data")
    common_warning = False
    if len(prepared) > 1:
        for j in range(len(prepared)):
            best_single = min(per_dataset[m][j] for m in candidates)
            chosen = per_dataset[model][j]
            if np.isfinite(best_single) and chosen > 1.5 * max(best_single, 1e-12):
                common_warning = True
                break
    speed, accuracy, simplicity = _MODEL_META[model]
    reason = (
        f"lowest cross-validated error among supported models (score={scores[model]:.3g})"
    )
    return ModelRecommendation(
        model,
        reason,
        speed,
        accuracy,
        simplicity,
        scores[model],
        common_warning,
    )


def _outliers(residuals: np.ndarray) -> np.ndarray:
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    if mad == 0.0:
        std = float(np.std(residuals))
        if std == 0.0:
            return np.zeros(0, dtype=int)
        z_score = (residuals - float(np.mean(residuals))) / std
        return np.flatnonzero(np.abs(z_score) > 3.5)
    robust_z = 0.67448975 * (residuals - median) / mad
    return np.flatnonzero(np.abs(robust_z) > 3.5)


def _cross_check(
    model: ModelName, x: np.ndarray, y: np.ndarray, params: np.ndarray
) -> float:
    f = _MODEL_FUNCS[model]
    primary = np.asarray(f(x, *params), dtype=float)
    if model == "linear":
        p = np.polyfit(x, y, 1)
        secondary = _linear(x, p[0], p[1])
    elif model == "quadratic":
        p = np.polyfit(x, y, 2)
        secondary = _quadratic(x, p[0], p[1], p[2])
    else:
        result = spo.least_squares(
            lambda p: np.asarray(f(x, *p), dtype=float) - y, params
        )
        secondary = np.asarray(f(x, *result.x), dtype=float)
    return float(np.sqrt(np.mean((primary - secondary) ** 2)))


def _hash_dataset(dataset: FitDataset) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(dataset.x).tobytes())
    digest.update(np.ascontiguousarray(dataset.y).tobytes())
    if dataset.sigma is not None:
        digest.update(np.ascontiguousarray(dataset.sigma).tobytes())
    return digest.hexdigest()


def run_guided_fit(
    datasets: tuple[FitDataset, ...],
    model: ModelName,
    *,
    missing_policy: MissingPolicy = "interpolate",
    outlier_policy: OutlierPolicy = "keep",
    seed: int = 0,
) -> GuidedFitResult:
    results: list[DatasetResult] = []
    operations = [
        f"missing-data policy: {missing_policy}",
        f"model selected by user: {model}",
    ]
    for dataset in datasets:
        prepared = _prepare(dataset, missing_policy)
        fit = _fit_arrays(model, prepared.x, prepared.y, prepared.sigma)
        residuals = np.asarray(fit.residuals, dtype=float)
        outlier_indices = _outliers(residuals)
        used = prepared
        if outlier_policy == "exclude" and outlier_indices.size:
            keep = np.ones(prepared.x.size, dtype=bool)
            keep[outlier_indices] = False
            sigma = None if prepared.sigma is None else prepared.sigma[keep]
            used = FitDataset(
                prepared.name,
                prepared.x[keep],
                prepared.y[keep],
                sigma,
                prepared.source_path,
            )
            fit = _fit_arrays(model, used.x, used.y, used.sigma)
        cv = _cv_rmse(used, model, seed)
        std = np.asarray(fit.parameter_std, dtype=float)
        ci = np.column_stack((fit.params - 1.96 * std, fit.params + 1.96 * std))
        cross_error = _cross_check(
            model, used.x, used.y, np.asarray(fit.params, dtype=float)
        )
        results.append(
            DatasetResult(
                used.name,
                np.asarray(fit.params, dtype=float),
                std,
                ci,
                float(fit.rmse),
                fit.r_squared,
                cv,
                outlier_indices,
                cross_error,
                int(used.x.size),
                float(np.min(used.x)),
                float(np.max(used.x)),
                float(np.mean(used.y)),
                float(np.std(used.y)),
            )
        )
    operations.append(f"outlier policy: {outlier_policy}")
    operations.append("3x repeated 5-fold cross-validation completed")
    if any(ds.sigma is not None for ds in datasets):
        operations.append("measurement uncertainty used in weighted fitting")
    operations.append("independent numerical cross-check completed")
    r2_values = [r.r_squared for r in results if r.r_squared is not None]
    relative_cv = [
        r.cv_rmse / (float(np.std(_prepare(ds, missing_policy).y)) or 1.0)
        for r, ds in zip(results, datasets, strict=True)
    ]
    max_cross = max(r.cross_check_error for r in results)
    if (
        r2_values
        and min(r2_values) >= 0.9
        and max(relative_cv) <= 0.35
        and max_cross <= max(1e-8, 0.1 * max(r.rmse for r in results))
    ):
        trust: TrustLabel = "reliable"
        comment = (
            "Fit is stable across held-out data and the independent numerical cross-check."
        )
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
        {ds.name: _hash_dataset(_prepare(ds, missing_policy)) for ds in datasets},
    )


def plot_result(
    result: GuidedFitResult,
    datasets: tuple[FitDataset, ...],
    output_dir: str | Path,
) -> list[Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ds, item in zip(datasets, result.datasets, strict=True):
        prepared = _prepare(ds, result.missing_policy)
        fig, ax = plt.subplots()
        ax.scatter(prepared.x, prepared.y, label="data")
        if item.outlier_indices.size:
            ax.scatter(
                prepared.x[item.outlier_indices],
                prepared.y[item.outlier_indices],
                marker="x",
                label="outlier",
            )
        order = np.argsort(prepared.x)
        y_fit = _MODEL_FUNCS[result.model](prepared.x[order], *item.params)
        ax.plot(prepared.x[order], y_fit, label=f"{result.model} fit")
        if prepared.sigma is not None:
            ax.errorbar(
                prepared.x,
                prepared.y,
                yerr=prepared.sigma,
                fmt="none",
                alpha=0.5,
            )
        ax.set_title(f"{item.name}: {result.trust}")
        ax.legend()
        for suffix in ("png", "pdf"):
            path = target / f"{item.name}_fit.{suffix}"
            fig.savefig(path, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
    return paths


def manifest_dict(
    result: GuidedFitResult,
    datasets: tuple[FitDataset, ...],
    *,
    x_column: str | None = None,
    y_column: str | None = None,
    sigma_column: str | None = None,
) -> dict[str, object]:
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
            "datasets": [
                {
                    **{
                        k: v
                        for k, v in asdict(item).items()
                        if k
                        not in {
                            "params",
                            "parameter_std",
                            "confidence_95",
                            "outlier_indices",
                        }
                    },
                    "params": item.params.tolist(),
                    "parameter_std": item.parameter_std.tolist(),
                    "confidence_95": item.confidence_95.tolist(),
                    "outlier_indices": item.outlier_indices.tolist(),
                }
                for item in result.datasets
            ],
        },
        "inputs": [
            {
                "name": ds.name,
                "source_path": ds.source_path,
                "x_column": x_column,
                "y_column": y_column,
                "sigma_column": sigma_column,
            }
            for ds in datasets
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
    datasets = []
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
    return run_guided_fit(
        tuple(datasets),
        cfg["model"],
        missing_policy=cfg["missing_policy"],
        outlier_policy=cfg["outlier_policy"],
        seed=int(cfg["seed"]),
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
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.05, 0.95, text[:12000], va="top", family="monospace", fontsize=8)
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
