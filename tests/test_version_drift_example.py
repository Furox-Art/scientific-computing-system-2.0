"""Tests for the version-drift comparison tool.

The comparison tool is the piece that turns two recorded runs into a
reproducibility verdict, so the checks here target its decision logic rather
than its printing: symmetric deviation accounting, the materiality threshold,
the environment-drift report, and the non-zero exit used as a CI gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compare_drift_results import compare  # noqa: E402

COMPARISON_TOOL = REPO_ROOT / "scripts" / "compare_drift_results.py"
COMMITTED_DIR = REPO_ROOT / "examples" / "data" / "version-drift"

BASELINE_ENV = {
    "python": "3.12.3",
    "numpy": "1.26.4",
    "scipy": "1.11.4",
    "pandas": "2.2.3",
    "matplotlib": "3.10.8",
}
UPGRADED_ENV = {
    "python": "3.12.3",
    "numpy": "2.3.5",
    "scipy": "1.18.1",
    "pandas": "2.2.3",
    "matplotlib": "3.10.8",
}


def _result(
    params: list[float],
    ci: list[list[float]],
    *,
    rmse: float = 3.2285e-4,
    cv_rmse: float = 3.3299e-4,
    r_squared: float = 0.999920691932623,
    cross_check_error: float = 1.7425e-09,
    trust: str = "reliable",
) -> dict[str, object]:
    return {
        "model": "exponential",
        "trust": trust,
        "params": params,
        "parameter_std": [1.0, 0.048, 1.0],
        "confidence_95": ci,
        "rmse": rmse,
        "cv_rmse": cv_rmse,
        "r_squared": r_squared,
        "cross_check_error": cross_check_error,
    }


def _record(result: dict[str, object], env: dict[str, str], **extra: object) -> dict[str, object]:
    return {
        "case": "exponential_weak_offset",
        "seed": 20261006,
        "environment": env,
        "result": result,
        **extra,
    }


def _identical_pair() -> tuple[dict[str, object], dict[str, object]]:
    result = _result(
        [1.644, 0.08097, 2.3566], [[-0.3616, 3.6498], [-0.0143, 0.1762], [0.3500, 4.3633]]
    )
    return _record(result, BASELINE_ENV), _record(dict(result), BASELINE_ENV)


def test_identical_runs_have_zero_deviation() -> None:
    """Replaying a manifest in the same environment must measure zero drift."""
    baseline, rerun = _identical_pair()
    report = compare(baseline, rerun)

    metrics = report["metrics"]
    assert metrics["max_abs_ci_bound_shift"] == pytest.approx(0.0, abs=1e-30)
    assert metrics["relative_ci_l2"] == pytest.approx(0.0, abs=1e-30)
    assert metrics["parameter_l2_shift"] == pytest.approx(0.0, abs=1e-30)
    assert metrics["rmse_relative_change"] == pytest.approx(0.0, abs=1e-30)
    assert not report["verdict"]["material_drift"]
    assert not report["rerun_stability_warning"]


def test_identical_runs_report_no_environment_drift() -> None:
    baseline, rerun = _identical_pair()
    assert compare(baseline, rerun)["environments"]["baseline"] == BASELINE_ENV
    assert compare(baseline, rerun)["environments"]["rerun"] == BASELINE_ENV


def test_version_change_is_surfaced_in_the_environments() -> None:
    """The comparison must expose which side ran under which stack."""
    baseline, _ = _identical_pair()
    _, rerun_old = _identical_pair()
    upgraded = dict(rerun_old)
    upgraded["environment"] = UPGRADED_ENV

    environments = compare(baseline, upgraded)["environments"]
    assert environments["baseline"]["numpy"] == "1.26.4"
    assert environments["rerun"]["numpy"] == "2.3.5"
    assert environments["baseline"]["scipy"] != environments["rerun"]["scipy"]


def test_absolute_and_relative_deviation_are_both_recorded() -> None:
    """A known perturbation must give the expected absolute and relative change."""
    result = _result([1.0, 0.0, 0.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    baseline = _record(result, BASELINE_ENV)

    perturbed = dict(result)
    perturbed["params"] = [1.001, 0.0, 0.0]
    report = compare(baseline, _record(perturbed, BASELINE_ENV))

    param_rows = {row["index"]: row for row in report["per_parameter"]}
    assert param_rows[0]["absolute_deviation"] == pytest.approx(0.001)
    assert param_rows[0]["relative_deviation"] == pytest.approx(0.001)
    assert param_rows[1]["absolute_deviation"] == pytest.approx(0.0)
    assert report["metrics"]["parameter_l2_shift"] == pytest.approx(0.001)


def test_ci_bound_deviation_is_tracked_per_bound() -> None:
    result = _result([1.0, 0.0, 0.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    baseline = _record(result, BASELINE_ENV)

    perturbed = dict(result)
    perturbed["confidence_95"] = [[1.5, 2.0], [3.0, 4.0], [5.0, 6.0]]
    report = compare(baseline, _record(perturbed, BASELINE_ENV))

    first_lower = next(
        row
        for row in report["per_ci_bound"]
        if row["parameter_index"] == 0 and row["bound"] == "lower"
    )
    assert first_lower["absolute_deviation"] == pytest.approx(0.5)
    assert first_lower["relative_deviation"] == pytest.approx(0.5)
    assert report["metrics"]["max_abs_ci_bound_shift"] == pytest.approx(0.5)


def test_small_drift_stays_below_threshold() -> None:
    """Drift far below the materiality threshold must not be escalated."""
    baseline, _ = _identical_pair()
    result = _result(
        [1.644 + 1e-9, 0.08097, 2.3566], [[-0.3616, 3.6498], [-0.0143, 0.1762], [0.3500, 4.3633]]
    )
    report = compare(baseline, _record(result, UPGRADED_ENV))

    assert report["verdict"]["material_ci_drift"] is False
    assert report["verdict"]["material_fit_drift"] is False
    assert report["verdict"]["material_drift"] is False


def test_large_drift_is_flagged_as_material() -> None:
    """A change beyond the threshold must be escalated, not rounded away."""
    baseline, _ = _identical_pair()
    shifted = [
        [-0.3616 * 1.5, 3.6498 * 1.5],
        [-0.0143 * 1.5, 0.1762 * 1.5],
        [0.35 * 1.5, 4.3633 * 1.5],
    ]
    result = _result([1.644 * 1.5, 0.08097, 2.3566], shifted)
    report = compare(baseline, _record(result, UPGRADED_ENV))

    assert report["verdict"]["material_ci_drift"] is True
    assert report["verdict"]["material_drift"] is True


def test_threshold_is_configurable() -> None:
    """The same measurement must flip verdict when the threshold tightens."""
    baseline, _ = _identical_pair()
    result = _result(
        [1.644 * 1.01, 0.08097, 2.3566], [[-0.3616, 3.6498], [-0.0143, 0.1762], [0.3500, 4.3633]]
    )
    rerun = _record(result, UPGRADED_ENV)

    assert (
        compare(baseline, rerun, materiality_threshold=0.05)["verdict"]["material_drift"] is False
    )
    assert compare(baseline, rerun, materiality_threshold=1e-4)["verdict"]["material_drift"] is True


def test_trust_label_change_alone_is_material() -> None:
    """A changed reliability label invalidates the saved analysis on its own."""
    baseline, _ = _identical_pair()
    result = _result(
        [1.644, 0.08097, 2.3566],
        [[-0.3616, 3.6498], [-0.0143, 0.1762], [0.3500, 4.3633]],
        trust="caution",
    )
    report = compare(baseline, _record(result, UPGRADED_ENV))

    assert report["verdict"]["trust_changed"] is True
    assert report["verdict"]["material_drift"] is True
    assert report["scalars"]["trust"]["baseline"] == "reliable"
    assert report["scalars"]["trust"]["rerun"] == "caution"


def test_rerun_stability_details_are_carried_through() -> None:
    baseline, _ = _identical_pair()
    result = _result(
        [1.644, 0.08097, 2.3566], [[-0.3616, 3.6498], [-0.0143, 0.1762], [0.3500, 4.3633]]
    )
    rerun = _record(result, UPGRADED_ENV)
    rerun["stability_warning"] = True
    rerun["stability_details"] = ["runtime version changed: numpy 1.26.4 -> 2.3.5"]

    report = compare(baseline, rerun)
    assert report["rerun_stability_warning"] is True
    assert "numpy 1.26.4 -> 2.3.5" in report["rerun_stability_details"][0]


def test_mismatched_ci_shape_is_rejected() -> None:
    """Comparing unlike shapes silently would hide a real change."""
    baseline, _ = _identical_pair()
    result = _result([1.644, 0.08097, 2.3566], [[-0.3616, 3.6498], [-0.0143, 0.1762]])
    with pytest.raises(ValueError, match="confidence interval shape changed"):
        compare(baseline, _record(result, UPGRADED_ENV))


def test_cli_exits_zero_for_stable_result(tmp_path: Path) -> None:
    """Exit 0 lets the check pass as a CI gate when nothing material changed."""
    baseline, rerun = _identical_pair()
    baseline_path = tmp_path / "baseline.json"
    rerun_path = tmp_path / "rerun.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    rerun_path.write_text(json.dumps(rerun), encoding="utf-8")

    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(COMPARISON_TOOL),
            "--baseline",
            str(baseline_path),
            "--rerun",
            str(rerun_path),
            "--output",
            str(tmp_path / "comparison.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "WARNING" not in completed.stdout
    assert (
        json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))["verdict"][
            "material_drift"
        ]
        is False
    )


def test_cli_exits_nonzero_and_warns_on_material_drift(tmp_path: Path) -> None:
    """Material drift must both warn loudly and fail the gate."""
    baseline, _ = _identical_pair()
    result = _result(
        [1.644 * 2.0, 0.08097, 2.3566],
        [[-0.3616 * 2.0, 3.6498 * 2.0], [-0.0143, 0.1762], [0.3500, 4.3633]],
    )
    rerun = _record(result, UPGRADED_ENV)
    baseline_path = tmp_path / "baseline.json"
    rerun_path = tmp_path / "rerun.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    rerun_path.write_text(json.dumps(rerun), encoding="utf-8")

    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(COMPARISON_TOOL),
            "--baseline",
            str(baseline_path),
            "--rerun",
            str(rerun_path),
            "--output",
            str(tmp_path / "comparison.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "WARNING" in completed.stdout
    assert "materiality threshold" in completed.stdout


@pytest.mark.parametrize(
    "name", ["baseline-result.json", "upgraded-rerun.json", "guided_fit_manifest.json"]
)
def test_committed_records_exist_and_are_valid_json(name: str) -> None:
    path = COMMITTED_DIR / name
    assert path.is_file(), f"missing committed record: {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


def test_committed_records_document_a_real_stack_upgrade() -> None:
    """The committed artifact must record a genuine NumPy/SciPy change, not a stub."""
    baseline = json.loads((COMMITTED_DIR / "baseline-result.json").read_text(encoding="utf-8"))
    rerun = json.loads((COMMITTED_DIR / "upgraded-rerun.json").read_text(encoding="utf-8"))

    baseline_env = baseline["environment"]
    rerun_env = rerun["environment"]
    assert baseline_env["numpy"] != rerun_env["numpy"]
    assert baseline_env["scipy"] != rerun_env["scipy"]
    assert rerun["stability_warning"] is True
    assert any("numpy" in detail for detail in rerun["stability_details"])

    report = compare(baseline, rerun)
    assert report["verdict"]["material_drift"] is False
    assert report["verdict"]["trust_changed"] is False
    # A committed claim of stability must be backed by a real measured margin.
    assert report["metrics"]["relative_ci_l2"] < report["materiality_threshold"]
