from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_section(text: str, start: str, end: str, replacement: str, label: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        raise RuntimeError(f"{label}: start marker not found")
    end_index = text.find(end, start_index)
    if end_index < 0:
        raise RuntimeError(f"{label}: end marker not found")
    return text[:start_index] + replacement + text[end_index:]


root = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# guided_fit.py
# ---------------------------------------------------------------------------
path = root / "src/cds2/guided_fit.py"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "from dataclasses import asdict, dataclass\n",
    "from dataclasses import asdict, dataclass, replace\n",
    "dataclasses import",
)
text = replace_once(
    text,
    "    common_model_warning: bool\n\n\n@dataclass(frozen=True)\nclass DatasetResult:",
    "    common_model_warning: bool\n    separate_models: tuple[tuple[str, ModelName], ...] = ()\n\n\n@dataclass(frozen=True)\nclass DatasetResult:",
    "recommendation field",
)
text = replace_once(
    text,
    "    y_mean: float\n    y_std: float\n\n\n@dataclass(frozen=True)\nclass GuidedFitResult:",
    "    y_mean: float\n    y_std: float\n    outlier_rmse_reduction_pct: float = 0.0\n\n\n@dataclass(frozen=True)\nclass GuidedFitResult:",
    "dataset outlier effect field",
)
text = replace_once(
    text,
    "    package_versions: dict[str, str]\n    data_hashes: dict[str, str]\n\n\ndef _linear",
    "    package_versions: dict[str, str]\n    data_hashes: dict[str, str]\n    stability_warning: bool = False\n    stability_details: tuple[str, ...] = ()\n\n\ndef _linear",
    "rerun stability fields",
)
text = replace_once(
    text,
    """    scores: dict[ModelName, float] = {}\n    per_dataset: dict[ModelName, list[float]] = {}\n    for model in candidates:\n""",
    """    scores: dict[ModelName, float] = {}\n    per_dataset: dict[ModelName, list[float]] = {}\n    complexity_penalty: dict[ModelName, float] = {\n        \"linear\": 0.0,\n        \"quadratic\": 0.01,\n        \"exponential\": 0.02,\n        \"power\": 0.02,\n        \"logistic\": 0.03,\n    }\n    for model in candidates:\n""",
    "complexity table",
)
text = replace_once(
    text,
    """        per_dataset[model] = values\n        complexity = {\n            \"linear\": 0.0,\n            \"quadratic\": 0.01,\n            \"exponential\": 0.02,\n            \"power\": 0.02,\n            \"logistic\": 0.03,\n        }[model]\n        scores[model] = float(np.mean(values)) + complexity\n""",
    """        per_dataset[model] = values\n        scores[model] = float(np.mean(values)) + complexity_penalty[model]\n""",
    "complexity scoring",
)
text = replace_once(
    text,
    """    common_warning = False\n    if len(prepared) > 1:\n        for j in range(len(prepared)):\n            best_single = min(per_dataset[m][j] for m in candidates)\n            chosen = per_dataset[model][j]\n            if np.isfinite(best_single) and chosen > 1.5 * max(best_single, 1e-12):\n                common_warning = True\n                break\n""",
    """    common_warning = False\n    separate_models: list[tuple[str, ModelName]] = []\n    if len(prepared) > 1:\n        for j, dataset in enumerate(prepared):\n            best_model = min(\n                candidates,\n                key=lambda candidate: per_dataset[candidate][j] + complexity_penalty[candidate],\n            )\n            separate_models.append((dataset.name, best_model))\n            best_single = per_dataset[best_model][j]\n            chosen = per_dataset[model][j]\n            if np.isfinite(best_single) and chosen > 1.5 * max(best_single, 1e-12):\n                common_warning = True\n""",
    "per-dataset recommendation",
)
text = replace_once(
    text,
    """        scores[model],\n        common_warning,\n    )\n""",
    """        scores[model],\n        common_warning,\n        tuple(separate_models),\n    )\n""",
    "recommendation return",
)
text = replace_once(
    text,
    """        fit = _fit_arrays(model, prepared.x, prepared.y, prepared.sigma)\n        residuals = np.asarray(fit.residuals, dtype=np.float64)\n        outlier_indices = _outliers(residuals)\n        used = prepared\n        if outlier_policy == \"exclude\" and outlier_indices.size:\n            keep = np.ones(prepared.x.size, dtype=bool)\n            keep[outlier_indices] = False\n            sigma = None if prepared.sigma is None else prepared.sigma[keep]\n            used = FitDataset(\n                prepared.name,\n                prepared.x[keep],\n                prepared.y[keep],\n                sigma,\n                prepared.source_path,\n            )\n            fit = _fit_arrays(model, used.x, used.y, used.sigma)\n        cv = _cv_rmse(used, model, seed)\n""",
    """        fit = _fit_arrays(model, prepared.x, prepared.y, prepared.sigma)\n        residuals = np.asarray(fit.residuals, dtype=np.float64)\n        outlier_indices = _outliers(residuals)\n        used = prepared\n        outlier_rmse_reduction_pct = 0.0\n        if outlier_indices.size:\n            keep = np.ones(prepared.x.size, dtype=bool)\n            keep[outlier_indices] = False\n            sigma = None if prepared.sigma is None else prepared.sigma[keep]\n            diagnostic_used = FitDataset(\n                prepared.name,\n                prepared.x[keep],\n                prepared.y[keep],\n                sigma,\n                prepared.source_path,\n            )\n            diagnostic_fit = _fit_arrays(\n                model, diagnostic_used.x, diagnostic_used.y, diagnostic_used.sigma\n            )\n            baseline_rmse = cast(float, fit.rmse)\n            diagnostic_rmse = cast(float, diagnostic_fit.rmse)\n            scale = max(abs(baseline_rmse), np.finfo(float).eps)\n            outlier_rmse_reduction_pct = 100.0 * (baseline_rmse - diagnostic_rmse) / scale\n            if outlier_policy == \"exclude\":\n                used = diagnostic_used\n                fit = diagnostic_fit\n        cv = _cv_rmse(used, model, seed)\n""",
    "outlier influence fit",
)
text = replace_once(
    text,
    """                float(np.mean(used.y)),\n                float(np.std(used.y)),\n            )\n""",
    """                float(np.mean(used.y)),\n                float(np.std(used.y)),\n                float(outlier_rmse_reduction_pct),\n            )\n""",
    "dataset result outlier effect",
)
text = replace_once(
    text,
    """    operations.append(f\"outlier policy: {outlier_policy}\")\n    operations.append(\"3x repeated 5-fold cross-validation completed\")\n""",
    """    operations.append(f\"outlier policy: {outlier_policy}\")\n    operations.append(\"outlier influence quantified by refit comparison\")\n    operations.append(\"3x repeated 5-fold cross-validation completed\")\n""",
    "outlier operation",
)
new_plot = '''def plot_result(\n    result: GuidedFitResult,\n    datasets: tuple[FitDataset, ...],\n    output_dir: str | Path,\n) -> list[Path]:\n    target = Path(output_dir)\n    target.mkdir(parents=True, exist_ok=True)\n    paths: list[Path] = []\n    for ds, item in zip(datasets, result.datasets, strict=True):\n        prepared = _prepare(ds, result.missing_policy)\n        order = np.argsort(prepared.x)\n        y_fit = _MODEL_FUNCS[result.model](prepared.x[order], *item.params)\n\n        fig, ax = plt.subplots()\n        ax.scatter(prepared.x, prepared.y, label="data")\n        if item.outlier_indices.size:\n            ax.scatter(\n                prepared.x[item.outlier_indices],\n                prepared.y[item.outlier_indices],\n                marker="x",\n                label="outlier",\n            )\n        ax.plot(prepared.x[order], y_fit, label=f"{result.model} fit")\n        if prepared.sigma is not None:\n            ax.errorbar(\n                prepared.x,\n                prepared.y,\n                yerr=prepared.sigma,\n                fmt="none",\n                alpha=0.5,\n            )\n        ax.set_title(f"{item.name}: {result.trust}")\n        ax.legend()\n        for suffix in ("png", "pdf"):\n            path = target / f"{item.name}_fit.{suffix}"\n            fig.savefig(path, bbox_inches="tight")\n            paths.append(path)\n        plt.close(fig)\n\n        predictions = np.asarray(\n            _MODEL_FUNCS[result.model](prepared.x, *item.params), dtype=np.float64\n        )\n        residuals = prepared.y - predictions\n        residual_fig, residual_ax = plt.subplots()\n        residual_ax.scatter(prepared.x, residuals, label="residual")\n        if item.outlier_indices.size:\n            residual_ax.scatter(\n                prepared.x[item.outlier_indices],\n                residuals[item.outlier_indices],\n                marker="x",\n                label="outlier",\n            )\n        residual_ax.axhline(0.0, linewidth=1.0)\n        residual_ax.set_xlabel("x")\n        residual_ax.set_ylabel("residual")\n        residual_ax.set_title(f"{item.name}: residuals")\n        residual_ax.legend()\n        for suffix in ("png", "pdf"):\n            path = target / f"{item.name}_residuals.{suffix}"\n            residual_fig.savefig(path, bbox_inches="tight")\n            paths.append(path)\n        plt.close(residual_fig)\n    return paths\n'''
text = replace_section(text, "def plot_result(\n", "\n\ndef manifest_dict(\n", new_plot, "plot_result")
text = replace_once(
    text,
    """            \"package_versions\": result.package_versions,\n            \"data_hashes\": result.data_hashes,\n            \"datasets\": [\n""",
    """            \"package_versions\": result.package_versions,\n            \"data_hashes\": result.data_hashes,\n            \"stability_warning\": result.stability_warning,\n            \"stability_details\": list(result.stability_details),\n            \"datasets\": [\n""",
    "manifest stability",
)
new_rerun = '''def rerun_manifest(path: str | Path) -> GuidedFitResult:\n    payload = json.loads(Path(path).read_text(encoding="utf-8"))\n    cfg = payload["result"]\n    datasets = []\n    for item in payload["inputs"]:\n        if not item["source_path"] or not item["x_column"] or not item["y_column"]:\n            raise ValueError("manifest does not contain reusable CSV source metadata")\n        datasets.append(\n            load_csv_dataset(\n                item["source_path"],\n                item["x_column"],\n                item["y_column"],\n                item["sigma_column"],\n            )\n        )\n    rerun = run_guided_fit(\n        tuple(datasets),\n        cfg["model"],\n        missing_policy=cfg["missing_policy"],\n        outlier_policy=cfg["outlier_policy"],\n        seed=int(cfg["seed"]),\n    )\n\n    details: list[str] = []\n    previous_hashes = cast(dict[str, str], cfg["data_hashes"])\n    for name, digest in rerun.data_hashes.items():\n        if previous_hashes.get(name) != digest:\n            details.append(f"input data changed: {name}")\n\n    previous_results = {\n        cast(str, item["name"]): item\n        for item in cast(list[dict[str, object]], cfg["datasets"])\n    }\n    for item in rerun.datasets:\n        previous = previous_results[item.name]\n        old_rmse = float(cast(float, previous["rmse"]))\n        rmse_change = abs(item.rmse - old_rmse) / max(abs(old_rmse), 1e-12)\n        old_params = np.asarray(cast(list[float], previous["params"]), dtype=np.float64)\n        param_scale = max(float(np.linalg.norm(old_params)), 1e-12)\n        param_change = float(np.linalg.norm(item.params - old_params)) / param_scale\n        if max(rmse_change, param_change) > 0.05:\n            details.append(\n                f"fit changed materially for {item.name}: "\n                f"rmse={rmse_change:.1%}, parameters={param_change:.1%}"\n            )\n\n    if cast(str, cfg["trust"]) != rerun.trust:\n        details.append(\n            f"reliability label changed: {cfg['trust']} -> {rerun.trust}"\n        )\n\n    return replace(\n        rerun,\n        stability_warning=bool(details),\n        stability_details=tuple(details),\n    )\n'''
text = replace_section(text, "def rerun_manifest(\n", "\n\ndef write_report(\n", new_rerun, "rerun_manifest")
text = replace_once(
    text,
    """                f\"- Outliers detected: {item.outlier_indices.tolist()}\",\n                f\"- Parameters: {item.params.tolist()}\",\n""",
    """                f\"- Outliers detected: {item.outlier_indices.tolist()}\",\n                f\"- Estimated RMSE reduction without detected outliers: {item.outlier_rmse_reduction_pct:.2f}%\",\n                f\"- Parameters: {item.params.tolist()}\",\n""",
    "report outlier effect",
)
path.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# cli.py
# ---------------------------------------------------------------------------
path = root / "src/cds2/cli.py"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    """            if recommendation.common_model_warning:\n                print(\"warning   one common model is weaker for at least one dataset\")\n            if _ask_yes_no(\"Use this model?\"):\n""",
    """            if recommendation.common_model_warning:\n                print(\"warning   one common model is weaker for at least one dataset\")\n                separate = \"; \".join(\n                    f\"{name}={candidate}\" for name, candidate in recommendation.separate_models\n                )\n                print(f\"separate  {separate}\")\n            if _ask_yes_no(\"Use this model?\"):\n""",
    "CLI separate model recommendation",
)
text = replace_once(
    text,
    """        outlier_total = sum(item.outlier_indices.size for item in preliminary.datasets)\n        outlier_policy = args.outliers\n""",
    """        outlier_total = sum(item.outlier_indices.size for item in preliminary.datasets)\n        for item in preliminary.datasets:\n            if item.outlier_indices.size:\n                print(\n                    f\"effect    {item.name}: estimated RMSE reduction without suspicious points=\"\n                    f\"{item.outlier_rmse_reduction_pct:.2f}%\"\n                )\n        outlier_policy = args.outliers\n""",
    "CLI outlier effect",
)
new_rerun_cli = '''def cmd_guided_fit_rerun(args: argparse.Namespace) -> int:\n    from .guided_fit import rerun_manifest\n\n    try:\n        result = rerun_manifest(args.manifest)\n    except (OSError, ValueError, RuntimeError, KeyError) as exc:\n        print(f"error: {exc}", file=sys.stderr)\n        return 1\n    print(f"model     {result.model}")\n    print(f"verdict   {result.trust}: {result.comment}")\n    for item in result.datasets:\n        print(f"dataset   {item.name}: rmse={item.rmse:.6g}; cv_rmse={item.cv_rmse:.6g}")\n    if result.stability_warning:\n        print("warning   rerun differs materially from the saved analysis")\n        for detail in result.stability_details:\n            print(f"detail    {detail}")\n    else:\n        print("stability consistent with the saved analysis")\n    return 0\n'''
text = replace_section(text, "def cmd_guided_fit_rerun(\n", "\n\ndef build_parser()", new_rerun_cli, "CLI rerun")
path.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# tests/test_guided_fit.py
# ---------------------------------------------------------------------------
path = root / "tests/test_guided_fit.py"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    """    assert rec.speed == \"fastest\"\n    assert rec.common_model_warning is False\n""",
    """    assert rec.speed == \"fastest\"\n    assert rec.common_model_warning is False\n    assert rec.separate_models == ()\n""",
    "single recommendation assertion",
)
text = replace_once(
    text,
    """    assert excluded.datasets[0].n_points < kept.datasets[0].n_points\n    assert excluded.datasets[0].rmse < kept.datasets[0].rmse\n""",
    """    assert excluded.datasets[0].n_points < kept.datasets[0].n_points\n    assert excluded.datasets[0].rmse < kept.datasets[0].rmse\n    assert kept.datasets[0].outlier_rmse_reduction_pct > 0.0\n    assert excluded.datasets[0].outlier_rmse_reduction_pct > 0.0\n""",
    "outlier effect assertion",
)
text = replace_once(
    text,
    """    paths = gf.plot_result(result, (dataset,), tmp_path)\n    assert {path.suffix for path in paths} == {\".png\", \".pdf\"}\n""",
    """    paths = gf.plot_result(result, (dataset,), tmp_path)\n    assert {path.suffix for path in paths} == {\".png\", \".pdf\"}\n    assert len(paths) == 4\n    assert (tmp_path / \"weighted_residuals.png\").exists()\n    assert (tmp_path / \"weighted_residuals.pdf\").exists()\n""",
    "residual plot assertions",
)
text = replace_once(
    text,
    """    assert payload[\"result\"][\"model\"] == \"linear\"\n    assert gf.rerun_manifest(manifest).model == \"linear\"\n""",
    """    assert payload[\"result\"][\"model\"] == \"linear\"\n    rerun = gf.rerun_manifest(manifest)\n    assert rerun.model == \"linear\"\n    assert rerun.stability_warning is False\n    assert rerun.stability_details == ()\n""",
    "stable rerun assertions",
)
text = replace_once(
    text,
    """    rec = gf.recommend_model((first, second))\n    assert rec.common_model_warning is True\n    monkeypatch.undo()\n""",
    """    rec = gf.recommend_model((first, second))\n    assert rec.common_model_warning is True\n    assert dict(rec.separate_models) == {\"first\": \"linear\", \"second\": \"quadratic\"}\n    monkeypatch.undo()\n""",
    "separate model assertion",
)
text = replace_once(
    text,
    """    paths = gf.plot_result(result, (contaminated,), tmp_path)\n    assert len(paths) == 2\n""",
    """    paths = gf.plot_result(result, (contaminated,), tmp_path)\n    assert len(paths) == 4\n""",
    "outlier plot path count",
)
insert = '''\n\ndef test_rerun_manifest_warns_on_changed_data_and_saved_verdict(tmp_path) -> None:  # type: ignore[no-untyped-def]\n    csv_path = tmp_path / "stable.csv"\n    x = np.linspace(1.0, 8.0, 40)\n    frame = pd.DataFrame({"x": x, "y": 3.0 * x + 2.0})\n    frame.to_csv(csv_path, index=False)\n    dataset = gf.load_csv_dataset(csv_path, "x", "y")\n    result = gf.run_guided_fit((dataset,), "linear")\n    manifest = gf.save_manifest(\n        result,\n        (dataset,),\n        tmp_path / "rerun.json",\n        x_column="x",\n        y_column="y",\n    )\n\n    payload = json.loads(manifest.read_text(encoding="utf-8"))\n    payload["result"]["trust"] = "unreliable"\n    manifest.write_text(json.dumps(payload), encoding="utf-8")\n    changed_verdict = gf.rerun_manifest(manifest)\n    assert changed_verdict.stability_warning is True\n    assert any("reliability label changed" in detail for detail in changed_verdict.stability_details)\n\n    payload["result"]["trust"] = result.trust\n    manifest.write_text(json.dumps(payload), encoding="utf-8")\n    frame["y"] = 7.0 * x + 5.0\n    frame.to_csv(csv_path, index=False)\n    changed_data = gf.rerun_manifest(manifest)\n    assert changed_data.stability_warning is True\n    assert any("input data changed" in detail for detail in changed_data.stability_details)\n    assert any("fit changed materially" in detail for detail in changed_data.stability_details)\n\n\n@pytest.mark.parametrize("dataset_name", ["diabetes", "linnerud"])\ndef test_packaged_real_world_scientific_data(dataset_name: str) -> None:\n    from sklearn.datasets import load_diabetes, load_linnerud\n\n    if dataset_name == "diabetes":\n        bunch = load_diabetes()\n        x = np.asarray(bunch.data[:, 2], dtype=np.float64)\n        y = np.asarray(bunch.target, dtype=np.float64)\n    else:\n        bunch = load_linnerud()\n        x = np.asarray(bunch.data[:, 0], dtype=np.float64)\n        y = np.asarray(bunch.target[:, 2], dtype=np.float64)\n    result = gf.run_guided_fit((gf.FitDataset(dataset_name, x, y),), "linear", seed=11)\n    item = result.datasets[0]\n    assert np.isfinite(item.rmse)\n    assert np.isfinite(item.cv_rmse)\n    assert item.r_squared is not None\n    assert np.isfinite(item.r_squared)\n    assert np.isfinite(item.cross_check_error)\n'''
anchor = "\n\ndef test_manifest_without_source_metadata_cannot_rerun"
if anchor not in text:
    raise RuntimeError("test insert anchor not found")
text = text.replace(anchor, insert + anchor, 1)
path.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# tests/test_guided_cli.py
# ---------------------------------------------------------------------------
path = root / "tests/test_guided_cli.py"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    """    assert (output / \"linear_fit.png\").exists()\n    assert (output / \"linear_fit.pdf\").exists()\n    manifest = output / \"guided_fit_manifest.json\"\n""",
    """    assert (output / \"linear_fit.png\").exists()\n    assert (output / \"linear_fit.pdf\").exists()\n    assert (output / \"linear_residuals.png\").exists()\n    assert (output / \"linear_residuals.pdf\").exists()\n    manifest = output / \"guided_fit_manifest.json\"\n""",
    "CLI residual files",
)
text = replace_once(
    text,
    """    assert (output / \"guided_fit_report.md\").exists()\n    assert main([\"guided-fit-rerun\", str(manifest)]) == 0\n""",
    """    assert (output / \"guided_fit_report.md\").exists()\n    assert main([\"guided-fit-rerun\", str(manifest)]) == 0\n    stable_out = capsys.readouterr().out\n    assert \"stability consistent\" in stable_out\n\n    frame = pd.read_csv(csv_path)\n    frame[\"y\"] = 5.0 * frame[\"x\"] + 4.0\n    frame.to_csv(csv_path, index=False)\n    assert main([\"guided-fit-rerun\", str(manifest)]) == 0\n    changed_out = capsys.readouterr().out\n    assert \"rerun differs materially\" in changed_out\n""",
    "CLI rerun stability",
)
text = replace_once(
    text,
    """    assert \"outliers\" in out\n    assert (output / \"guided_fit_report.html\").exists()\n""",
    """    assert \"outliers\" in out\n    assert \"effect\" in out\n    assert (output / \"guided_fit_report.html\").exists()\n""",
    "CLI outlier effect assertion",
)
text = replace_once(
    text,
    """            rec.score,\n            True,\n        )\n""",
    """            rec.score,\n            True,\n            ((\"first\", \"linear\"), (\"second\", \"quadratic\")),\n        )\n""",
    "CLI fake separate recommendations",
)
text = replace_once(
    text,
    """    assert code == 0\n    assert \"one common model\" in capsys.readouterr().out\n""",
    """    assert code == 0\n    out = capsys.readouterr().out\n    assert \"one common model\" in out\n    assert \"separate  first=linear; second=quadratic\" in out\n""",
    "CLI separate recommendation assertion",
)
path.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------
path = root / "docs/api/guided_fit.md"
text = path.read_text(encoding="utf-8")
old_list = '''1. inspects one or more CSV datasets and reports missing values;\n2. runs small pilot fits and repeated cross-validation before recommending one model;\n3. explains the recommendation with speed, expected fit behaviour and complexity;\n4. leaves final model selection to the user;\n5. includes measurement uncertainty when a `sigma` column is supplied;\n6. detects suspicious residual outliers and asks before excluding them;\n7. fits the selected model, calculates parameter uncertainty and 95% confidence intervals;\n8. checks held-out performance with repeated 5-fold cross-validation;\n9. cross-checks the fitted result with an independent numerical method;\n10. produces Matplotlib PNG/PDF figures, a reproducibility manifest and an optional PDF/HTML/Markdown report;\n11. labels the overall result `reliable`, `caution` or `unreliable`;\n12. recommends a different model when the selected fit is weak.\n'''
new_list = '''1. inspects one or more CSV datasets and reports missing values;\n2. runs small pilot fits and repeated cross-validation before recommending one model;\n3. explains the recommendation with speed, expected fit behaviour and complexity;\n4. leaves final model selection to the user;\n5. includes measurement uncertainty when a `sigma` column is supplied;\n6. detects suspicious residual outliers, quantifies their estimated RMSE effect and asks before excluding them;\n7. fits the selected model, calculates parameter uncertainty and 95% confidence intervals;\n8. checks held-out performance with repeated 5-fold cross-validation;\n9. cross-checks the fitted result with an independent numerical method;\n10. produces Matplotlib fit and residual PNG/PDF figures, a reproducibility manifest and an optional PDF/HTML/Markdown report;\n11. labels the overall result `reliable`, `caution` or `unreliable`;\n12. recommends a different model when the selected fit is weak;\n13. recommends dataset-specific models when a single common model is materially weaker;\n14. compares manifest reruns with the saved analysis and warns when inputs, fit parameters, RMSE or the reliability label change materially.\n'''
text = replace_once(text, old_list, new_list, "guided-fit workflow docs")
text = replace_once(
    text,
    "The manifest stores the selected model, policies, seed, data hashes, input source metadata and NumPy/SciPy/pandas/Matplotlib/Python versions.\n",
    "The manifest stores the selected model, policies, seed, data hashes, input source metadata and NumPy/SciPy/pandas/Matplotlib/Python versions. Reruns compare those saved results with the new calculation and surface material instability instead of silently replacing the prior result.\n",
    "manifest rerun docs",
)
path.write_text(text, encoding="utf-8")

path = root / "README.md"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    """reproducibility metadata, cross-checks fits numerically, reports uncertainty\nand held-out validation metrics, and can generate PNG/PDF plots plus\nPDF/HTML/Markdown reports.\n""",
    """reproducibility metadata, cross-checks fits numerically, reports uncertainty\nand held-out validation metrics, and can generate PNG/PDF fit and residual\nplots plus PDF/HTML/Markdown reports.\n""",
    "README guided summary",
)
text = replace_once(
    text,
    """fit plot as both PNG and PDF; reports are available as PDF, HTML or Markdown.\n""",
    """fit and residual plots as both PNG and PDF; reports are available as PDF, HTML or Markdown.\nReruns warn when saved results change materially, and multi-dataset analysis can\nrecommend separate models when a single common model is a poor compromise.\n""",
    "README CLI guided details",
)
path.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Wheel-level CLI smoke test
# ---------------------------------------------------------------------------
path = root / ".github/workflows/tests.yml"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    """          test -f cli-smoke-output/cli-smoke_fit.png\n          test -f cli-smoke-output/cli-smoke_fit.pdf\n          .wheel-smoke/bin/cds2 guided-fit-rerun cli-smoke-output/guided_fit_manifest.json\n""",
    """          test -f cli-smoke-output/cli-smoke_fit.png\n          test -f cli-smoke-output/cli-smoke_fit.pdf\n          test -f cli-smoke-output/cli-smoke_residuals.png\n          test -f cli-smoke-output/cli-smoke_residuals.pdf\n          .wheel-smoke/bin/cds2 guided-fit-rerun cli-smoke-output/guided_fit_manifest.json\n""",
    "wheel residual smoke",
)
path.write_text(text, encoding="utf-8")

print("guided-fit scientific validation enhancements applied")
