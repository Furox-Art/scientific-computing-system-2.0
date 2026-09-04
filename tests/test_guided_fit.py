"""Tests for guided scientific fitting."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

import cds2.guided_fit as gf

matplotlib.use("Agg")


def _linear_dataset(name: str = "linear", *, outlier: bool = False, sigma: bool = False) -> gf.FitDataset:
    x = np.linspace(1.0, 10.0, 60)
    y = 2.5 * x + 1.0
    if outlier:
        y = y.copy()
        y[30] += 40.0
    uncertainty = np.full_like(x, 0.2) if sigma else None
    return gf.FitDataset(name, x, y, uncertainty)


def test_load_inspect_and_validation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, np.nan, 6.0, 8.0], "s": [0.1] * 4}
    ).to_csv(path, index=False)
    dataset = gf.load_csv_dataset(path, "x", "y", "s")
    info = gf.inspect_dataset(dataset)
    assert info["points"] == 4
    assert info["missing"] == 1
    assert info["suggested_missing_policy"] == "interpolate"
    with pytest.raises(ValueError, match="missing columns"):
        gf.load_csv_dataset(path, "missing", "y")
    with pytest.raises(ValueError, match="same shape"):
        gf.inspect_dataset(gf.FitDataset("bad", np.arange(3.0), np.arange(4.0)))
    with pytest.raises(ValueError, match="sigma"):
        gf.inspect_dataset(gf.FitDataset("bad", np.arange(4.0), np.arange(4.0), np.ones(3)))


def test_prepare_drop_interpolate_and_failures() -> None:
    dataset = gf.FitDataset(
        "missing",
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan]),
        np.array([2.0, np.nan, 6.0, 8.0, 10.0, 12.0, 14.0]),
        np.array([0.2, 0.2, 0.2, 0.0, 0.2, 0.2, 0.2]),
    )
    dropped = gf._prepare(dataset, "drop")
    assert dropped.x.size == 4
    interpolated = gf._prepare(dataset, "interpolate")
    assert np.isfinite(interpolated.y).all()
    with pytest.raises(ValueError, match="two finite"):
        gf._prepare(
            gf.FitDataset(
                "few-y",
                np.arange(5.0),
                np.array([np.nan, np.nan, 1.0, np.nan, np.nan]),
            ),
            "interpolate",
        )
    with pytest.raises(ValueError, match="four usable"):
        gf._prepare(gf.FitDataset("tiny", np.arange(3.0), np.arange(3.0)), "drop")


@pytest.mark.parametrize(
    ("model", "x", "y", "n_params"),
    [
        (
            "linear",
            np.linspace(1.0, 5.0, 30),
            2.0 * np.linspace(1.0, 5.0, 30) + 3.0,
            2,
        ),
        (
            "quadratic",
            np.linspace(-2.0, 2.0, 40),
            1.2 * np.linspace(-2.0, 2.0, 40) ** 2
            + 0.5 * np.linspace(-2.0, 2.0, 40)
            + 2.0,
            3,
        ),
        (
            "exponential",
            np.linspace(0.0, 4.0, 40),
            2.0 * np.exp(0.3 * np.linspace(0.0, 4.0, 40)) + 1.0,
            3,
        ),
        (
            "power",
            np.linspace(1.0, 6.0, 40),
            1.5 * np.linspace(1.0, 6.0, 40) ** 1.7 + 0.2,
            3,
        ),
        (
            "logistic",
            np.linspace(-4.0, 4.0, 50),
            1.0 + 5.0 / (1.0 + np.exp(-1.2 * np.linspace(-4.0, 4.0, 50))),
            4,
        ),
    ],
)
def test_each_supported_model_fits(model, x, y, n_params) -> None:  # type: ignore[no-untyped-def]
    fit = gf._fit_arrays(model, x, y)
    assert fit.params.size == n_params
    assert fit.rmse is not None
    assert fit.rmse < 1e-4


def test_power_rejects_nonpositive_x() -> None:
    with pytest.raises(ValueError, match="x > 0"):
        gf._fit_arrays("power", np.array([0.0, 1.0, 2.0, 3.0]), np.arange(4.0))


def test_pilot_and_recommendation() -> None:
    dataset = _linear_dataset()
    pilot = gf._pilot(dataset, 10, 7)
    assert pilot.x.size == 10
    assert gf._pilot(dataset, 100, 7) is dataset
    rec = gf.recommend_model((dataset,), seed=4, max_pilot_points=20)
    assert rec.model == "linear"
    assert rec.speed == "fastest"
    assert rec.common_model_warning is False


def test_recommendation_handles_candidate_failure_and_multiple_datasets() -> None:
    x = np.linspace(-2.0, 2.0, 40)
    linear = gf.FitDataset("a", x, 2 * x + 1)
    quadratic = gf.FitDataset("b", x, 8 * x**2 + 1)
    rec = gf.recommend_model((linear, quadratic), seed=2)
    assert rec.model in gf.MODEL_NAMES
    assert isinstance(rec.common_model_warning, bool)


def test_outliers_and_cross_checks() -> None:
    assert gf._outliers(np.zeros(10)).size == 0
    residuals = np.zeros(20)
    residuals[-1] = 20.0
    assert gf._outliers(residuals).tolist() == [19]

    x = np.linspace(1.0, 5.0, 30)
    y_linear = 3.0 * x + 2.0
    assert gf._cross_check("linear", x, y_linear, np.array([3.0, 2.0])) < 1e-10
    y_quad = 2.0 * x**2 + x + 1.0
    assert gf._cross_check("quadratic", x, y_quad, np.array([2.0, 1.0, 1.0])) < 1e-10
    y_exp = 2.0 * np.exp(0.2 * x) + 1.0
    assert gf._cross_check("exponential", x, y_exp, np.array([2.0, 0.2, 1.0])) < 1e-8


def test_run_guided_fit_reliable_weighted_and_outlier_exclusion() -> None:
    weighted = _linear_dataset("weighted", sigma=True)
    result = gf.run_guided_fit((weighted,), "linear", outlier_policy="keep", seed=3)
    assert result.trust == "reliable"
    assert result.datasets[0].confidence_95.shape == (2, 2)
    assert result.package_versions["numpy"]
    assert result.data_hashes["weighted"]

    contaminated = _linear_dataset("contaminated", outlier=True)
    kept = gf.run_guided_fit((contaminated,), "linear", outlier_policy="keep")
    excluded = gf.run_guided_fit((contaminated,), "linear", outlier_policy="exclude")
    assert kept.datasets[0].outlier_indices.size >= 1
    assert excluded.datasets[0].n_points < kept.datasets[0].n_points
    assert excluded.datasets[0].rmse < kept.datasets[0].rmse


def test_run_guided_fit_unreliable_or_caution() -> None:
    x = np.linspace(0.0, 8.0, 80)
    y = np.sin(4 * x)
    result = gf.run_guided_fit((gf.FitDataset("oscillatory", x, y),), "linear")
    assert result.trust in {"caution", "unreliable"}
    assert "Fit" in result.comment


def test_plot_manifest_reports_and_rerun(tmp_path) -> None:  # type: ignore[no-untyped-def]
    csv_path = tmp_path / "weighted.csv"
    x = np.linspace(1.0, 8.0, 40)
    frame = pd.DataFrame({"x": x, "y": 4.0 * x + 2.0, "sigma": np.full_like(x, 0.3)})
    frame.to_csv(csv_path, index=False)
    dataset = gf.load_csv_dataset(csv_path, "x", "y", "sigma")
    result = gf.run_guided_fit((dataset,), "linear")

    paths = gf.plot_result(result, (dataset,), tmp_path)
    assert {path.suffix for path in paths} == {".png", ".pdf"}

    manifest = gf.save_manifest(
        result,
        (dataset,),
        tmp_path / "manifest.json",
        x_column="x",
        y_column="y",
        sigma_column="sigma",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["result"]["model"] == "linear"
    assert gf.rerun_manifest(manifest).model == "linear"

    for report_format, suffix in [("markdown", ".md"), ("html", ".html"), ("pdf", ".pdf")]:
        report = gf.write_report(result, tmp_path, report_format)
        assert report.suffix == suffix
        assert report.exists()


def test_manifest_without_source_metadata_cannot_rerun(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dataset = _linear_dataset()
    result = gf.run_guided_fit((dataset,), "linear")
    path = gf.save_manifest(result, (dataset,), tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="reusable CSV"):
        gf.rerun_manifest(path)


def test_additional_branches(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    plain = _linear_dataset("plain")
    assert gf.inspect_dataset(plain)["missing"] == 0

    tiny = gf.FitDataset("tiny-cv", np.arange(4.0), np.arange(4.0))
    assert np.isinf(gf._cv_rmse(tiny, "linear", 0))

    with pytest.raises(ValueError, match="candidate"):
        gf.recommend_model((plain,), exclude=gf.MODEL_NAMES)

    def all_fail(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("fail")

    monkeypatch.setattr(gf, "_cv_rmse", all_fail)
    with pytest.raises(RuntimeError, match="no supported"):
        gf.recommend_model((plain,))

    def fake_cv(dataset, model, seed, folds=5):  # type: ignore[no-untyped-def]
        del seed, folds
        table = {
            "first": {
                "linear": 0.1,
                "quadratic": 0.2,
                "exponential": 0.3,
                "power": 0.4,
                "logistic": 0.5,
            },
            "second": {
                "linear": 10.0,
                "quadratic": 0.1,
                "exponential": 0.3,
                "power": 0.4,
                "logistic": 0.5,
            },
        }
        return table[dataset.name][model]

    monkeypatch.setattr(gf, "_cv_rmse", fake_cv)
    first = gf.FitDataset("first", np.arange(1.0, 9.0), np.arange(1.0, 9.0))
    second = gf.FitDataset("second", np.arange(1.0, 9.0), np.arange(1.0, 9.0))
    rec = gf.recommend_model((first, second))
    assert rec.common_model_warning is True
    monkeypatch.undo()

    contaminated = _linear_dataset("plot-outlier", outlier=True)
    result = gf.run_guided_fit((contaminated,), "linear", outlier_policy="keep")
    paths = gf.plot_result(result, (contaminated,), tmp_path)
    assert len(paths) == 2
