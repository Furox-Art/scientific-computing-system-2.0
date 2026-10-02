"""CLI tests for guided fitting."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import cds2.guided_fit as gf
from cds2.cli import _ask_yes_no, _guided_report_choice, main


def _write_csv(path, *, missing=False, outlier=False, nonlinear=False) -> None:  # type: ignore[no-untyped-def]
    x = np.linspace(1.0, 10.0, 60)
    y = np.sin(3.0 * x) if nonlinear else 2.0 * x + 1.0
    if missing:
        y = y.copy()
        y[4] = np.nan
    if outlier:
        y = y.copy()
        y[30] += 40.0
    pd.DataFrame({"x": x, "y": y, "sigma": np.full_like(x, 0.2)}).to_csv(path, index=False)


def test_yes_no_and_report_choice(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    answers = iter(["", "", "yes", "no", "markdown", "bad"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert _ask_yes_no("q") is True
    assert _ask_yes_no("q", default=False) is False
    assert _ask_yes_no("q") is True
    assert _ask_yes_no("q") is False
    assert _guided_report_choice("html") == "html"
    assert _guided_report_choice("ask") == "markdown"
    with pytest.raises(ValueError, match="report format"):
        _guided_report_choice("ask")


def test_noninteractive_guided_fit_and_rerun(tmp_path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Rerun trusts the manifest directory and the invocation directory, so run
    # from the data directory like a real user would.
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "linear.csv"
    _write_csv(csv_path)
    output = tmp_path / "out"
    code = main(
        [
            "guided-fit",
            str(csv_path),
            "--x",
            "x",
            "--y",
            "y",
            "--sigma",
            "sigma",
            "--model",
            "linear",
            "--missing",
            "interpolate",
            "--outliers",
            "keep",
            "--report",
            "markdown",
            "--output-dir",
            str(output),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "verdict   reliable" in out
    assert (output / "linear_fit.png").exists()
    assert (output / "linear_fit.pdf").exists()
    assert (output / "linear_residuals.png").exists()
    assert (output / "linear_residuals.pdf").exists()
    manifest = output / "guided_fit_manifest.json"
    assert manifest.exists()
    assert (output / "guided_fit_report.md").exists()
    assert main(["guided-fit-rerun", str(manifest)]) == 0
    stable_out = capsys.readouterr().out
    assert "stability consistent" in stable_out

    frame = pd.read_csv(csv_path)
    frame["y"] = 5.0 * frame["x"] + 4.0
    frame.to_csv(csv_path, index=False)
    assert main(["guided-fit-rerun", str(manifest)]) == 0
    changed_out = capsys.readouterr().out
    assert "rerun differs materially" in changed_out


def test_interactive_missing_model_outlier_and_report(tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    csv_path = tmp_path / "dirty.csv"
    _write_csv(csv_path, missing=True, outlier=True)
    output = tmp_path / "interactive"
    answers = iter(["", "", "y", "html"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    code = main(
        [
            "guided-fit",
            str(csv_path),
            "--x",
            "x",
            "--y",
            "y",
            "--missing",
            "ask",
            "--outliers",
            "ask",
            "--report",
            "ask",
            "--output-dir",
            str(output),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "missing" in out
    assert "recommend" in out
    assert "outliers" in out
    assert "effect" in out
    assert (output / "guided_fit_report.html").exists()


def test_interactive_reject_recommendation_and_keep_outlier(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    csv_path = tmp_path / "outlier.csv"
    _write_csv(csv_path, outlier=True)
    answers = iter(["n", "linear", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    code = main(
        [
            "guided-fit",
            str(csv_path),
            "--x",
            "x",
            "--y",
            "y",
            "--missing",
            "ask",
            "--outliers",
            "ask",
            "--report",
            "none",
            "--output-dir",
            str(tmp_path / "reject"),
        ]
    )
    assert code == 0


def test_bad_interactive_choices_and_io_errors(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    answers = iter(["n", "not-a-model"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert (
        main(
            [
                "guided-fit",
                str(csv_path),
                "--x",
                "x",
                "--y",
                "y",
                "--missing",
                "interpolate",
                "--outliers",
                "keep",
                "--report",
                "none",
            ]
        )
        == 1
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: "invalid")
    assert (
        main(
            [
                "guided-fit",
                str(csv_path),
                "--x",
                "x",
                "--y",
                "y",
                "--model",
                "linear",
                "--missing",
                "interpolate",
                "--outliers",
                "keep",
                "--report",
                "ask",
            ]
        )
        == 1
    )

    assert main(["guided-fit", str(tmp_path / "missing.csv"), "--x", "x", "--y", "y"]) == 1
    assert main(["guided-fit-rerun", str(tmp_path / "missing.json")]) == 1


def test_guided_fit_rerun_cli_confines_foreign_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    csv_path = elsewhere / "data.csv"
    x = np.linspace(1.0, 8.0, 20)
    pd.DataFrame({"x": x, "y": 2.0 * x + 1.0}).to_csv(csv_path, index=False)
    dataset = gf.load_csv_dataset(csv_path, "x", "y")
    result = gf.run_guided_fit((dataset,), "linear")
    manifest = gf.save_manifest(
        result, (dataset,), run_dir / "manifest.json", x_column="x", y_column="y"
    )
    assert manifest.exists()

    # Foreign manifest: the CSV lives outside both the run directory and the
    # invocation directory, so rerun must refuse it by default ...
    assert main(["guided-fit-rerun", str(manifest)]) == 1
    # ... and honour the explicit opt-out flag.
    assert main(["guided-fit-rerun", str(manifest), "--allow-outside-run-dir"]) == 0
    assert json.loads(manifest.read_text(encoding="utf-8"))["inputs"][0]["source_path"] == str(
        csv_path
    )


def test_guided_fit_rerun_cli_relative_csv_smoke_flow(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Regression test for the CI package smoke flow: CWD-relative CSV plus
    an output subdirectory, then a rerun of the saved manifest."""
    monkeypatch.chdir(tmp_path)
    _write_csv(tmp_path / "smoke.csv")
    assert (
        main(
            [
                "guided-fit",
                "smoke.csv",
                "--x",
                "x",
                "--y",
                "y",
                "--sigma",
                "sigma",
                "--model",
                "linear",
                "--missing",
                "interpolate",
                "--outliers",
                "keep",
                "--report",
                "none",
                "--output-dir",
                "smoke-output",
            ]
        )
        == 0
    )
    manifest = tmp_path / "smoke-output" / "guided_fit_manifest.json"
    assert manifest.exists()
    assert main(["guided-fit-rerun", str(manifest)]) == 0


def test_missing_policy_drop_choice_and_next_recommendation(tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    csv_path = tmp_path / "nonlinear.csv"
    _write_csv(csv_path, missing=True, nonlinear=True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    code = main(
        [
            "guided-fit",
            str(csv_path),
            "--x",
            "x",
            "--y",
            "y",
            "--model",
            "linear",
            "--missing",
            "ask",
            "--outliers",
            "keep",
            "--report",
            "none",
            "--output-dir",
            str(tmp_path / "next"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "next" in out


def test_common_model_warning_is_printed(tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_csv(first)
    _write_csv(second)

    original = gf.recommend_model

    def fake_recommend(*args, **kwargs):  # type: ignore[no-untyped-def]
        rec = original(*args, **kwargs)
        return gf.ModelRecommendation(
            rec.model,
            rec.reason,
            rec.speed,
            rec.accuracy,
            rec.simplicity,
            rec.score,
            True,
            (("first", "linear"), ("second", "quadratic")),
        )

    monkeypatch.setattr(gf, "recommend_model", fake_recommend)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    code = main(
        [
            "guided-fit",
            str(first),
            str(second),
            "--x",
            "x",
            "--y",
            "y",
            "--missing",
            "interpolate",
            "--outliers",
            "keep",
            "--report",
            "none",
            "--output-dir",
            str(tmp_path / "multi"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "one common model" in out
    assert "separate  first=linear; second=quadratic" in out


def test_interactive_no_outlier_defaults_to_keep(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    csv_path = tmp_path / "clean.csv"
    _write_csv(csv_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert (
        main(
            [
                "guided-fit",
                str(csv_path),
                "--x",
                "x",
                "--y",
                "y",
                "--model",
                "linear",
                "--missing",
                "interpolate",
                "--outliers",
                "ask",
                "--report",
                "none",
                "--output-dir",
                str(tmp_path / "clean-out"),
            ]
        )
        == 0
    )
