"""Coverage-focused regression tests for scientific correctness guards.

These tests exercise defensive branches added by the correctness audit.  They
are deliberately contract-oriented: invalid scientific inputs must fail closed
rather than silently produce NaN/Inf or ambiguous results.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy

from cds2 import epidemiology as epi
from cds2 import graph, linalg, ml, reliability, stats
from cds2 import guided_fit as gf
from cds2 import montecarlo as mc
from cds2.estimator import KMeansSKL, LinearRegressionGD, RidgeSGD
from cds2.estimator._base import BaseEstimator

# ---------------------------------------------------------------------------
# Estimator input contracts and gradient-descent guards
# ---------------------------------------------------------------------------


def test_base_estimator_check_x_y_rejects_all_invalid_shapes_and_values() -> None:
    with pytest.raises(ValueError, match="2-D"):
        BaseEstimator._check_X_y([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least one"):
        BaseEstimator._check_X_y(np.empty((0, 2)), np.empty(0))
    with pytest.raises(ValueError, match="finite"):
        BaseEstimator._check_X_y([[1.0], [np.nan]], [1.0, 2.0])
    with pytest.raises(ValueError, match="1-D"):
        BaseEstimator._check_X_y([[1.0], [2.0]], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="same number"):
        BaseEstimator._check_X_y([[1.0], [2.0]], [1.0])
    with pytest.raises(ValueError, match="finite"):
        BaseEstimator._check_X_y([[1.0], [2.0]], [1.0, np.inf])
    x, y = BaseEstimator._check_X_y([[1.0]], None)
    assert x.shape == (1, 1) and y is None


def test_base_estimator_check_x_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="2-D"):
        BaseEstimator._check_X([1.0])
    with pytest.raises(ValueError, match="at least one"):
        BaseEstimator._check_X(np.empty((2, 0)))
    with pytest.raises(ValueError, match="finite"):
        BaseEstimator._check_X([[np.inf]])


def test_estimator_parameter_and_prediction_guards() -> None:
    model = LinearRegressionGD()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict([[1.0]])
    fitted = LinearRegressionGD(max_iter=5).fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="1-D"):
        fitted.score([[0.0]], [[0.0]])

    ridge = RidgeSGD()
    with pytest.raises(RuntimeError, match="not fitted"):
        ridge.predict([[1.0]])
    fitted_ridge = RidgeSGD(max_iter=5).fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="1-D"):
        fitted_ridge.score([[0.0]], [[0.0]])


def test_gradient_descent_divergence_is_detected() -> None:
    x = np.array([[1e154], [-1e154], [1e154], [-1e154]])
    y = np.array([1.0, -1.0, 1.0, -1.0])
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="diverged"):
            LinearRegressionGD(learning_rate=1.0, max_iter=5, tol=0.0).fit(x, y)
        with pytest.raises(FloatingPointError, match="diverged"):
            RidgeSGD(alpha=1.0, learning_rate=1.0, max_iter=5, tol=0.0).fit(x, y)


def test_kmeans_sklearn_predict_rejects_wrong_feature_count() -> None:
    fitted = KMeansSKL(n_clusters=2, seed=0).fit([[0.0], [0.1], [9.9], [10.0]])
    with pytest.raises(ValueError, match="different number of features"):
        fitted.predict([[1.0, 2.0]])


# ---------------------------------------------------------------------------
# Linear algebra and graph native-boundary guards
# ---------------------------------------------------------------------------


def test_svd_rejects_non_matrix() -> None:
    with pytest.raises(ValueError, match="2-D"):
        linalg.svd([1.0, 2.0, 3.0])


def test_pagerank_rejects_invalid_native_probability_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenKernel:
        @staticmethod
        def iterate(*args):  # type: ignore[no-untyped-def]
            n = int(args[3])
            return np.zeros(n, dtype=np.float64).tobytes(), 1

    monkeypatch.setattr(graph, "_HAS_PR_KERNEL", True)
    monkeypatch.setattr(graph, "_pr_kernel", BrokenKernel())
    with pytest.raises(FloatingPointError, match="invalid probability"):
        graph.pagerank([[0.0, 1.0], [1.0, 0.0]])


# ---------------------------------------------------------------------------
# Guided-fit validation, reproducibility, cross-check and manifest contracts
# ---------------------------------------------------------------------------


def _linear_dataset(name: str = "data", *, source_path: str | None = None) -> gf.FitDataset:
    x = np.linspace(1.0, 8.0, 20)
    return gf.FitDataset(name, x, 2.0 * x + 1.0, source_path=source_path)


def test_guided_dataset_shape_and_prepare_guards() -> None:
    with pytest.raises(ValueError, match="1-D"):
        gf.inspect_dataset(gf.FitDataset("bad", np.ones((2, 2)), np.ones(4)))
    with pytest.raises(ValueError, match="same shape"):
        gf.inspect_dataset(gf.FitDataset("bad", np.ones(4), np.ones(5)))
    with pytest.raises(ValueError, match="sigma"):
        gf.inspect_dataset(gf.FitDataset("bad", np.ones(4), np.ones(4), np.ones((2, 2))))

    dataset = _linear_dataset()
    with pytest.raises(ValueError, match="missing_policy"):
        gf._prepare(dataset, "invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1-D"):
        gf._prepare(gf.FitDataset("bad", np.ones((2, 2)), np.ones(4)), "drop")
    with pytest.raises(ValueError, match="same shape"):
        gf._prepare(gf.FitDataset("bad", np.ones(4), np.ones(5)), "drop")
    with pytest.raises(ValueError, match="sigma"):
        gf._prepare(gf.FitDataset("bad", np.ones(4), np.ones(4), np.ones(3)), "drop")
    with pytest.raises(ValueError, match="interpolation"):
        gf._prepare(
            gf.FitDataset("bad", np.arange(5.0), np.array([1.0, np.nan, np.nan, np.nan, np.nan])),
            "interpolate",
        )
    with pytest.raises(ValueError, match="four usable"):
        gf._prepare(gf.FitDataset("bad", np.arange(3.0), np.arange(3.0)), "drop")


def test_guided_power_and_recommendation_guards() -> None:
    with pytest.raises(ValueError, match="x > 0"):
        gf._fit_arrays("power", np.arange(4.0), np.arange(4.0))
    with pytest.raises(ValueError, match="at least one dataset"):
        gf.recommend_model(())
    with pytest.raises(ValueError, match="max_pilot_points"):
        gf.recommend_model((_linear_dataset(),), max_pilot_points=3)
    with pytest.raises(ValueError, match="candidate model"):
        gf.recommend_model((_linear_dataset(),), exclude=gf.MODEL_NAMES)


def test_guided_pilot_and_no_cv_error_paths() -> None:
    dataset = _linear_dataset()
    assert gf._pilot(dataset, 100, 1) is dataset
    tiny = gf.FitDataset("tiny", np.arange(4.0), np.arange(4.0))
    assert np.isinf(gf._cv_rmse(tiny, "linear", 0, folds=4))


def test_guided_cross_check_defensive_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    x = np.linspace(1.0, 5.0, 10)
    y = 2.0 * np.exp(0.2 * x) + 1.0
    p0, _ = gf._initial_guess("exponential", x, y)

    calls: list[float] = []

    def fake_minimize(fun, start, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(float(fun(np.full_like(np.asarray(start, dtype=float), 1e308))))
        return SimpleNamespace(success=True, x=np.asarray(start, dtype=float), message="ok")

    monkeypatch.setattr(gf.spo, "minimize", fake_minimize)
    value = gf._cross_check("exponential", x, y, p0, np.ones_like(x))
    assert np.isfinite(value)
    assert np.isinf(calls[0])

    logistic_y = 1.0 + 4.0 / (1.0 + np.exp(-1.2 * (x - 3.0)))
    logistic_p0, _ = gf._initial_guess("logistic", x, logistic_y)
    assert np.isfinite(gf._cross_check("logistic", x, logistic_y, logistic_p0))

    def failed_minimize(*args, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(success=False, x=np.ones(3), message="failed")

    monkeypatch.setattr(gf.spo, "minimize", failed_minimize)
    with pytest.raises(RuntimeError, match="cross-check failed"):
        gf._cross_check("exponential", x, y, np.array([2.0, 0.2, 1.0]))


def test_guided_dataset_key_and_file_stem_fallbacks() -> None:
    x = np.arange(1.0, 6.0)
    datasets = (
        gf.FitDataset("a", x, x),
        gf.FitDataset("a", x, x + 1),
        gf.FitDataset("a#1", x, x + 2),
        gf.FitDataset("../", x, x + 3),
        gf.FitDataset("x" * 150, x, x + 4),
    )
    keys = gf._dataset_keys(datasets)
    assert len(keys) == len(set(keys))
    assert any("~2" in key for key in keys)
    stems = gf._dataset_file_stems(datasets)
    assert len(stems) == len(set(stems))
    assert all("/" not in stem for stem in stems)
    assert max(map(len, stems)) <= 120


def test_guided_confidence_interval_guard_paths() -> None:
    no_std = SimpleNamespace(params=np.array([1.0, 2.0]), parameter_std=None, dof=3)
    assert np.isnan(gf._confidence_interval(no_std)).all()
    no_dof = SimpleNamespace(params=np.array([1.0, 2.0]), parameter_std=np.array([0.1, 0.2]), dof=0)
    assert np.isnan(gf._confidence_interval(no_dof)).all()


def test_guided_run_and_plot_contract_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="at least one dataset"):
        gf.run_guided_fit((), "linear")
    with pytest.raises(ValueError, match="unsupported"):
        gf.run_guided_fit((_linear_dataset(),), "unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing_policy"):
        gf.run_guided_fit((_linear_dataset(),), "linear", missing_policy="ask")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outlier_policy"):
        gf.run_guided_fit((_linear_dataset(),), "linear", outlier_policy="ask")  # type: ignore[arg-type]

    small = gf.FitDataset("small", np.arange(1.0, 6.0), np.arange(1.0, 6.0))
    monkeypatch.setattr(gf, "_outliers", lambda residuals: np.array([0, 1], dtype=np.intp))
    with pytest.raises(ValueError, match="fewer than four"):
        gf.run_guided_fit((small,), "linear", outlier_policy="exclude")

    result = gf.run_guided_fit((_linear_dataset(),), "linear")
    with pytest.raises(ValueError, match="fitted result count"):
        gf.plot_result(result, (), tmp_path)
    with pytest.raises(ValueError, match="fitted result count"):
        gf.manifest_dict(result, ())


def _save_manifest_for_mutation(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.csv"
    x = np.linspace(1.0, 8.0, 20)
    pd.DataFrame({"x": x, "y": 2.0 * x + 1.0}).to_csv(source, index=False)
    dataset = gf.load_csv_dataset(source, "x", "y")
    result = gf.run_guided_fit((dataset,), "linear")
    manifest = gf.save_manifest(
        result,
        (dataset,),
        tmp_path / "manifest.json",
        x_column="x",
        y_column="y",
    )
    return manifest, source


def test_guided_manifest_rerun_defensive_branches(tmp_path: Path) -> None:
    manifest, _source = _save_manifest_for_mutation(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    malformed = tmp_path / "malformed.json"
    malformed_payload = json.loads(json.dumps(payload))
    malformed_payload["inputs"][0]["x_column"] = None
    malformed.write_text(json.dumps(malformed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reusable CSV"):
        gf.rerun_manifest(malformed)

    count_changed = tmp_path / "count.json"
    count_payload = json.loads(json.dumps(payload))
    count_payload["result"]["datasets"] = []
    count_changed.write_text(json.dumps(count_payload), encoding="utf-8")
    count_result = gf.rerun_manifest(count_changed)
    assert any("dataset count changed" in detail for detail in count_result.stability_details)

    shape_changed = tmp_path / "shape.json"
    shape_payload = json.loads(json.dumps(payload))
    shape_payload["result"]["datasets"][0]["params"] = [1.0]
    shape_changed.write_text(json.dumps(shape_payload), encoding="utf-8")
    shape_result = gf.rerun_manifest(shape_changed)
    assert any("parameter shape changed" in detail for detail in shape_result.stability_details)

    material = tmp_path / "material.json"
    material_payload = json.loads(json.dumps(payload))
    material_payload["result"]["datasets"][0]["rmse"] = 100.0
    material_payload["result"]["datasets"][0]["params"] = [100.0, 100.0]
    material_payload["result"]["trust"] = "unreliable"
    material.write_text(json.dumps(material_payload), encoding="utf-8")
    material_result = gf.rerun_manifest(material)
    assert any("fit changed materially" in detail for detail in material_result.stability_details)
    assert any(
        "reliability label changed" in detail for detail in material_result.stability_details
    )


def test_guided_manifest_hash_fallbacks(tmp_path: Path) -> None:
    manifest, _source = _save_manifest_for_mutation(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    key = payload["inputs"][0]["dataset_key"]
    digest = payload["result"]["rerun_data_hashes"].pop(key)
    payload["result"]["rerun_data_hashes"][payload["inputs"][0]["name"]] = digest
    fallback = tmp_path / "fallback.json"
    fallback.write_text(json.dumps(payload), encoding="utf-8")
    assert gf.rerun_manifest(fallback).stability_warning is False


def test_guided_remaining_defensive_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interpolated = gf.FitDataset(
        "interpolated",
        np.arange(5.0),
        np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
    )

    def nonfinite_interp(x, xp, fp):  # type: ignore[no-untyped-def]
        return np.full(np.asarray(x).shape, np.inf, dtype=float)

    monkeypatch.setattr(gf.np, "interp", nonfinite_interp)
    with pytest.raises(ValueError, match="prepared x and y must be finite"):
        gf._prepare(interpolated, "interpolate")
    monkeypatch.undo()

    small = gf.FitDataset("small-keep", np.arange(1.0, 6.0), 2.0 * np.arange(1.0, 6.0) + 1.0)
    monkeypatch.setattr(gf, "_outliers", lambda residuals: np.array([0, 1], dtype=np.intp))
    kept = gf.run_guided_fit((small,), "linear", outlier_policy="keep")
    assert kept.outlier_policy == "keep"
    assert kept.datasets[0].name == "small-keep"
    monkeypatch.undo()

    dataset = _linear_dataset()
    result = gf.run_guided_fit((dataset,), "linear")
    manifest = gf.manifest_dict(result, (dataset,), x_column="x", y_column="y")
    assert manifest["result"]["rerun_data_hashes"] == result.data_hashes

    saved_manifest, _source = _save_manifest_for_mutation(tmp_path)
    payload = json.loads(saved_manifest.read_text(encoding="utf-8"))
    payload["inputs"][0]["dataset_key"] = "saved-alias"
    runtime_fallback = tmp_path / "runtime-fallback.json"
    runtime_fallback.write_text(json.dumps(payload), encoding="utf-8")
    assert gf.rerun_manifest(runtime_fallback).stability_warning is False


# ---------------------------------------------------------------------------
# ML utility/model validation branches
# ---------------------------------------------------------------------------


def test_train_test_split_rejects_invalid_contracts_and_handles_no_shuffle() -> None:
    with pytest.raises(ValueError, match="at least one array"):
        ml.train_test_split()
    with pytest.raises(ValueError, match="at least one dimension"):
        ml.train_test_split(1.0)
    with pytest.raises(ValueError, match="same first dimension"):
        ml.train_test_split(np.arange(3), np.arange(4))
    with pytest.raises(ValueError, match="at least two samples"):
        ml.train_test_split([1.0])
    with pytest.raises(ValueError, match="test_size"):
        ml.train_test_split([1.0, 2.0], test_size=1.0)
    train, test = ml.train_test_split(np.arange(4.0), test_size=0.5, shuffle=False)
    assert train.tolist() == [0.0, 1.0]
    assert test.tolist() == [2.0, 3.0]


@pytest.mark.parametrize(
    "model, message",
    [
        (ml.LogisticRegression(learning_rate=0.0), "learning_rate"),
        (ml.LogisticRegression(max_iter=0), "max_iter"),
        (ml.LogisticRegression(l2=-1.0), "l2"),
    ],
)
def test_logistic_hyperparameter_guards(model: ml.LogisticRegression, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        model.fit([[0.0], [1.0]], [0, 1])


def test_logistic_input_prediction_and_divergence_guards() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ml.LogisticRegression().fit(np.empty((0, 1)), [])
    with pytest.raises(ValueError, match="finite"):
        ml.LogisticRegression().fit([[0.0], [np.nan]], [0, 1])
    with pytest.raises(ValueError, match="1-D"):
        ml.LogisticRegression().fit([[0.0], [1.0]], [[0], [1]])
    with pytest.raises(ValueError, match="binary"):
        ml.LogisticRegression().fit([[0.0], [1.0]], [0, 2])

    unfitted = ml.LogisticRegression()
    with pytest.raises(RuntimeError, match="not fitted"):
        unfitted.predict_proba([[0.0]])

    fitted = ml.LogisticRegression(max_iter=2).fit([[0.0], [1.0]], [0, 1])
    assert fitted.predict_proba([0.5]).shape == (1,)
    with pytest.raises(ValueError, match="different number"):
        fitted.predict_proba([[0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        fitted.predict_proba([[np.inf]])
    with pytest.raises(ValueError, match="threshold"):
        fitted.predict([[0.0]], threshold=2.0)

    huge_x = np.array([[1e308], [-1e308]])
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="diverged"):
            ml.LogisticRegression(learning_rate=1e308, max_iter=2).fit(huge_x, [1, 0])


def test_kmeans_core_validation_empty_cluster_and_identical_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ml.KMeans().fit(np.empty((0, 1)))
    with pytest.raises(ValueError, match="n_clusters"):
        ml.KMeans(n_clusters=3).fit([[0.0], [1.0]])
    with pytest.raises(ValueError, match="tol"):
        ml.KMeans(n_clusters=1, tol=-1.0).fit([[0.0], [1.0]])

    unfitted = ml.KMeans(n_clusters=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        unfitted.predict([[0.0]])

    fitted = ml.KMeans(n_clusters=1, seed=0).fit([[0.0], [1.0]])
    with pytest.raises(ValueError, match="different number"):
        fitted.predict([[0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        fitted.predict([[np.nan]])

    points = np.array([[0.0], [0.0], [10.0], [10.0]])
    centers = np.array([[0.0], [0.0], [10.0]])
    model = ml.KMeans(n_clusters=3, max_iter=2, tol=0.0)
    labels, moved = model._run_numpy_lloyd(points, centers)
    assert labels.shape == (4,) and np.isfinite(moved).all()

    identical = np.zeros((4, 2))
    seeded = ml.KMeans._kmeans_pp_init(identical, 3, np.random.default_rng(1))
    assert seeded.shape == (3, 2)


def test_pca_and_knn_validation_branches() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ml.PCA().fit([1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        ml.PCA(1).fit([[1.0], [np.nan]])
    with pytest.raises(ValueError, match="n_components"):
        ml.PCA(3).fit(np.ones((2, 2)))

    pca = ml.PCA(1)
    with pytest.raises(RuntimeError, match="not fitted"):
        pca.transform([[1.0]])
    pca.fit([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="different number"):
        pca.transform([[1.0]])
    with pytest.raises(ValueError, match="finite"):
        pca.transform([[np.nan, 1.0]])

    with pytest.raises(ValueError, match="non-empty"):
        ml.KNeighborsClassifier(1).fit(np.empty((0, 1)), [])
    with pytest.raises(ValueError, match="finite"):
        ml.KNeighborsClassifier(1).fit([[0.0], [np.nan]], [0, 1])
    with pytest.raises(ValueError, match="n_neighbors"):
        ml.KNeighborsClassifier(3).fit([[0.0], [1.0]], [0, 1])

    knn = ml.KNeighborsClassifier(1)
    with pytest.raises(RuntimeError, match="not fitted"):
        knn.predict([[0.0]])
    knn.fit([[0.0], [1.0]], [0, 1])
    assert knn.predict([0.1]).shape == (1,)
    with pytest.raises(ValueError, match="different number"):
        knn.predict([[0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        knn.predict([[np.inf]])


def test_ml_metric_validation_branches() -> None:
    with pytest.raises(ValueError, match="1-D"):
        ml.accuracy_score([[0, 1]], [[0, 1]])
    with pytest.raises(ValueError, match="not be empty"):
        ml.accuracy_score([], [])
    with pytest.raises(ValueError, match="labels must be unique"):
        ml.confusion_matrix([0, 1], [0, 1], labels=[0, 0, 1])
    with pytest.raises(ValueError, match="include every"):
        ml.confusion_matrix([0, 1], [0, 1], labels=[0])
    with pytest.raises(ValueError, match="1-D"):
        ml.mean_squared_error([[1.0]], [[1.0]])
    with pytest.raises(ValueError, match="not be empty"):
        ml.mean_squared_error([], [])
    with pytest.raises(ValueError, match="finite"):
        ml.mean_squared_error([1.0, np.nan], [1.0, 2.0])
    assert ml.precision_score([0], [0], pos_label=1) == 0.0
    assert ml.recall_score([0], [0], pos_label=1) == 0.0
    assert ml.f1_score([0], [0], pos_label=1) == 0.0


# ---------------------------------------------------------------------------
# Monte Carlo/MCMC fail-closed branches
# ---------------------------------------------------------------------------


def test_monte_carlo_integrator_validation_and_scalar_paths() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        mc.pi_estimate(0)
    with pytest.raises(ValueError, match="positive integer"):
        mc.mc_integrate(lambda x: x, 0.0, 1.0, n=0)
    with pytest.raises(ValueError, match="finite"):
        mc.mc_integrate(lambda x: x, np.nan, 1.0)
    assert mc.mc_integrate(lambda x: x, 2.0, 2.0) == 0.0
    assert mc.mc_integrate(lambda x: np.array(2.0), 1.0, 0.0, n=20, seed=1) == pytest.approx(-2.0)
    with pytest.raises(ValueError, match="one value per sample"):
        mc.mc_integrate(lambda x: np.ones((x.size, 2)), 0.0, 1.0, n=5)
    with pytest.raises(ValueError, match="non-finite values"):
        mc.mc_integrate(lambda x: np.full(x.shape, np.nan), 0.0, 1.0, n=5)
    with pytest.raises(ValueError, match="non-finite value"):
        mc.mc_integrate(lambda x: np.array(np.nan), 0.0, 1.0, n=5)


def test_mc_expectation_validation_branches() -> None:
    def sampler(rng, n):  # type: ignore[no-untyped-def]
        return rng.normal(size=n)

    with pytest.raises(ValueError, match="positive integer"):
        mc.mc_expectation(lambda x: x, sampler, n=0)
    with pytest.raises(ValueError, match="return n samples"):
        mc.mc_expectation(lambda x: x, lambda rng, n: np.array(1.0), n=3)
    with pytest.raises(ValueError, match="non-finite samples"):
        mc.mc_expectation(lambda x: x, lambda rng, n: np.full(n, np.nan), n=3)
    assert mc.mc_expectation(lambda x: np.array(3.0), sampler, n=4, seed=1) == 3.0
    with pytest.raises(ValueError, match="one value per sample"):
        mc.mc_expectation(lambda x: np.ones((x.shape[0], 2)), sampler, n=4)
    with pytest.raises(ValueError, match="non-finite values"):
        mc.mc_expectation(lambda x: np.full(x.shape[0], np.inf), sampler, n=4)
    with pytest.raises(ValueError, match="non-finite value"):
        mc.mc_expectation(lambda x: np.array(np.nan), sampler, n=4)


def test_hit_or_miss_validation_and_scalar_fallback_paths() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        mc.hit_or_miss(lambda x: x, 0.0, 1.0, 1.0, n=0)
    with pytest.raises(ValueError, match="b > a"):
        mc.hit_or_miss(lambda x: x, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="y_max"):
        mc.hit_or_miss(lambda x: x, 0.0, 1.0, 0.0)
    assert 0.0 <= mc.hit_or_miss(lambda x: 0.5, 0.0, 1.0, 1.0, n=20, seed=1) <= 1.0

    def scalar_only(value):  # type: ignore[no-untyped-def]
        if isinstance(value, np.ndarray):
            raise TypeError("scalar only")
        return float(value) / 2.0

    assert 0.0 <= mc.hit_or_miss(scalar_only, 0.0, 1.0, 1.0, n=20, seed=2) <= 1.0
    with pytest.raises(ValueError, match="one value per sample"):
        mc.hit_or_miss(lambda x: np.ones((x.size, 2)), 0.0, 1.0, 1.0, n=5)
    with pytest.raises(ValueError, match="non-finite"):
        mc.hit_or_miss(lambda x: np.full(x.shape, np.nan), 0.0, 1.0, 1.0, n=5)
    with pytest.raises(ValueError, match=r"\[0, y_max\]"):
        mc.hit_or_miss(lambda x: np.full(x.shape, 2.0), 0.0, 1.0, 1.0, n=5)


def test_metropolis_all_validation_and_infinite_density_paths() -> None:
    for kwargs in ({"burn_in": 1.5}, {"thin": True}):
        with pytest.raises(ValueError, match="integer"):
            mc.metropolis_hastings(lambda v: 0.0, [0.0], n_samples=2, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="need n_samples"):
        mc.metropolis_hastings(lambda v: 0.0, [0.0], n_samples=0)
    with pytest.raises(ValueError, match="proposal_scale"):
        mc.metropolis_hastings(lambda v: 0.0, [0.0], n_samples=2, proposal_scale=0.0)
    with pytest.raises(ValueError, match="initial"):
        mc.metropolis_hastings(lambda v: 0.0, [[0.0]], n_samples=2)
    with pytest.raises(ValueError, match="initial"):
        mc.metropolis_hastings(lambda v: np.nan, [0.0], n_samples=2)

    result = mc.metropolis_hastings(
        lambda v: 0.0 if v[0] > 0.0 else float("-inf"),
        [-1.0],
        n_samples=4,
        burn_in=0,
        proposal_scale=2.0,
        seed=1,
    )
    assert result.samples.shape == (4, 1)

    with pytest.raises(ValueError, match="NaN or \+inf"):
        mc.metropolis_hastings(
            lambda v: 0.0 if v[0] == 0.0 else float("inf"),
            [0.0],
            n_samples=2,
            burn_in=0,
            seed=1,
        )


def test_parallel_mc_validation_seeded_and_unseeded() -> None:
    with pytest.raises(ValueError, match="n_total"):
        mc.parallel_mc_integrate(np.square, 0.0, 1.0, n_total=0)
    with pytest.raises(ValueError, match="greater than"):
        mc.parallel_mc_integrate(np.square, 1.0, 0.0, n_total=2)
    with pytest.raises(ValueError, match="workers"):
        mc.parallel_mc_integrate(np.square, 0.0, 1.0, n_total=2, workers=0)
    assert np.isfinite(mc.parallel_mc_integrate(np.square, 0.0, 1.0, n_total=2, workers=5, seed=1))
    assert np.isfinite(mc.parallel_mc_integrate(np.square, 0.0, 1.0, n_total=2, workers=1))


# ---------------------------------------------------------------------------
# Reliability and statistical fail-closed branches
# ---------------------------------------------------------------------------


def test_reliability_validation_and_optimizer_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="equal-length"):
        reliability.kaplan_meier([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        reliability.kaplan_meier([1.0, np.nan], [1.0, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        reliability.kaplan_meier([-1.0, 2.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="only 0 or 1"):
        reliability.kaplan_meier([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="event"):
        reliability.kaplan_meier([1.0, 2.0], [0.0, 0.0])
    assert reliability.KMResult(np.array([1.0]), np.array([0.8])).median is None

    with pytest.raises(ValueError, match="finite and non-negative"):
        reliability.weibull_fit([1.0, np.inf])
    with pytest.raises(ValueError, match="failures_mask"):
        reliability.weibull_fit([1.0, 2.0], [True])
    with pytest.raises(ValueError, match="positive failure"):
        reliability.weibull_fit([0.0, 2.0, 3.0], [True, True, False])

    original = reliability.sp_optimize.minimize
    monkeypatch.setattr(
        reliability.sp_optimize,
        "minimize",
        lambda *a, **k: SimpleNamespace(success=False, x=np.array([0.0, 0.0]), message="bad"),
    )
    with pytest.raises(RuntimeError, match="censored Weibull"):
        reliability.weibull_fit([1.0, 2.0, 3.0], [True, True, False])
    monkeypatch.setattr(reliability.sp_optimize, "minimize", original)

    with pytest.raises(ValueError, match="operating"):
        reliability.mtbf(0.0, 1)
    with pytest.raises(ValueError, match="failure"):
        reliability.mtbf(10.0, True)
    with pytest.raises(ValueError, match="mtbf"):
        reliability.availability(0.0, 1.0)
    with pytest.raises(ValueError, match="mttr"):
        reliability.availability(1.0, np.inf)
    with pytest.raises(ValueError, match="shape"):
        reliability.weibull_survival([1.0], 0.0, 1.0)
    with pytest.raises(ValueError, match="scale"):
        reliability.weibull_survival([1.0], 1.0, 0.0)
    with pytest.raises(ValueError, match="time values"):
        reliability.weibull_survival([-1.0], 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        reliability.bathtub_curve([1.0], np.nan, 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="rates"):
        reliability.bathtub_curve([1.0], -1.0, 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="knees"):
        reliability.bathtub_curve([1.0], 1.0, 1.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="time values"):
        reliability.bathtub_curve([-1.0], 1.0, 1.0, 1.0, 1.0, 1.0)


def test_stats_low_level_validation_branches() -> None:
    with pytest.raises(ValueError, match="finite"):
        stats._as_1d([1.0, np.nan])
    with pytest.raises(ValueError, match="same length"):
        stats._paired_arrays([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="at least 2"):
        stats._paired_arrays([1.0], [1.0])
    with pytest.raises(ValueError, match="mu"):
        stats._validate_normal(np.nan, 1.0)
    with pytest.raises(ValueError, match="sigma"):
        stats._validate_normal(0.0, 0.0)
    with pytest.raises(ValueError, match="two rows"):
        stats._contingency_table([[1.0, 2.0]])
    with pytest.raises(ValueError, match="finite and non-negative"):
        stats._contingency_table([[1.0, -1.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="positive total"):
        stats._contingency_table([[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="2-D matrix"):
        stats._matrix_observations([1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        stats._matrix_observations([[1.0], [np.inf]])


def test_stats_public_validation_branches() -> None:
    with pytest.raises(ValueError, match="popmean"):
        stats.t_test([1.0, 2.0], np.inf)
    with pytest.raises(ValueError, match="at least two groups"):
        stats.anova([1.0])
    with pytest.raises(ValueError, match="at least two groups"):
        stats.kruskal_wallis([1.0])
    with pytest.raises(ValueError, match="at least two groups"):
        stats.levene_test([1.0])
    with pytest.raises(ValueError, match="zero"):
        stats.cohens_d([1.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="f_statistic"):
        stats.eta_squared_from_f(-1.0, 1, 1)
    with pytest.raises(ValueError, match="df1"):
        stats.eta_squared_from_f(1.0, 0, 1)
    with pytest.raises(ValueError, match="q"):
        stats.percentile([1.0], 101.0)
    with pytest.raises(ValueError, match="constant"):
        stats.z_scores([1.0, 1.0])
    with pytest.raises(ValueError, match="x"):
        stats.norm_pdf(np.inf)
    with pytest.raises(ValueError, match="x"):
        stats.norm_cdf(np.nan)
    with pytest.raises(ValueError, match="q"):
        stats.norm_ppf(2.0)


def test_stats_bootstrap_permutation_matrix_and_streaming_guards() -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        stats.bootstrap_ci([1.0, 2.0], n_resamples=1)
    with pytest.raises(ValueError, match="confidence"):
        stats.bootstrap_ci([1.0, 2.0], confidence=1.0)

    def scalar_stat(sample):  # type: ignore[no-untyped-def]
        if np.asarray(sample).ndim != 1:
            raise TypeError("scalar sample only")
        return float(np.mean(sample))

    assert np.isfinite(
        stats.bootstrap_ci([1.0, 2.0, 3.0], scalar_stat, n_resamples=10, seed=1).estimate
    )
    with pytest.raises(ValueError, match="one finite scalar"):
        stats.bootstrap_ci(
            [1.0, 2.0], lambda sample, axis=1: np.array([np.nan, np.nan]), n_resamples=2
        )

    def point_nan_stat(sample, axis=None):  # type: ignore[no-untyped-def]
        if axis is None:
            return float("nan")
        return np.mean(sample, axis=axis)

    with pytest.raises(ValueError, match="point estimate"):
        stats.bootstrap_ci([1.0, 2.0], point_nan_stat, n_resamples=2)
    with pytest.raises(ValueError, match="n_permutations"):
        stats.permutation_test([1.0], [2.0], n_permutations=0)
    with pytest.raises(ValueError, match="constant"):
        stats.correlation_matrix([[1.0, 2.0], [1.0, 3.0]])

    stream = stats.StreamingStats()
    assert stream.push([]) is stream
    with pytest.raises(ValueError, match="finite"):
        stream.push([np.nan])
    empty = stats.StreamingStats()
    merged = empty.merge(stats.StreamingStats())
    assert merged.count_value == 0
    other = stats.StreamingStats().push([1.0, 2.0])
    merged_other = empty.merge(other)
    assert merged_other.count_value == 2
    with pytest.raises(ValueError, match="no observations"):
        stats.StreamingStats().mean
    with pytest.raises(ValueError, match="at least two"):
        stats.StreamingStats().push([1.0]).variance


# ---------------------------------------------------------------------------
# Epidemiology convergence failure guard
# ---------------------------------------------------------------------------


def test_final_size_iteration_wraps_solver_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("solver failed")

    monkeypatch.setattr(scipy.optimize, "brentq", fail)
    with pytest.raises(RuntimeError, match="did not converge"):
        epi.final_size_iteration(2.0)
    with pytest.raises(ValueError, match="tol"):
        epi.final_size_iteration(2.0, tol=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        epi.final_size_iteration(2.0, max_iter=0)
