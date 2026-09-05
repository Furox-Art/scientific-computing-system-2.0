"""Correctness regression tests for hardening wave 2."""

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
    manifest = gf.save_manifest(
        result, datasets, tmp_path / "manifest.json", x_column="x", y_column="y"
    )
    stable = gf.rerun_manifest(manifest)
    assert stable.stability_warning is False
    pd.DataFrame({"x": x, "y": 8.0 * x - 7.0}).to_csv(p2, index=False)
    changed = gf.rerun_manifest(manifest)
    assert changed.stability_warning is True
    assert any("#2" in detail for detail in changed.stability_details)


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
    assert mc.mc_integrate(lambda x: x, 1.0, 0.0, n=200_000, seed=4) == pytest.approx(
        -0.5, abs=0.005
    )
    with pytest.raises(ValueError, match=r"\[0, y_max\]"):
        mc.hit_or_miss(lambda _x: 2.0, 0.0, 1.0, 1.0, n=100, seed=1)
    with pytest.raises(ValueError, match="proposal_scale"):
        mc.metropolis_hastings(lambda v: -0.5 * v[0] ** 2, [0.0], proposal_scale=0.0)


def test_parallel_mc_uses_tiny_exact_budget_without_zero_worker_chunks() -> None:
    assert mc.parallel_mc_integrate(
        _constant_two, 0.0, 1.0, n_total=3, workers=8, seed=3
    ) == pytest.approx(2.0)


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
