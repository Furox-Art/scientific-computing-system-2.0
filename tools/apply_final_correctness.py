from __future__ import annotations

import ast
import textwrap
from pathlib import Path


def replace_function(path: str, name: str, source: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(
        (item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name),
        None,
    )
    if node is None or node.end_lineno is None:
        raise SystemExit(f"function {name} not found in {path}")
    lines = text.splitlines(keepends=True)
    lines[node.lineno - 1 : node.end_lineno] = [textwrap.dedent(source).strip("\n") + "\n\n"]
    target.write_text("".join(lines), encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"marker not found in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_all(path: str, old: str, new: str, expected: int) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found != expected:
        raise SystemExit(f"expected {expected} markers in {path}, found {found}")
    target.write_text(text.replace(old, new), encoding="utf-8")


replace_function(
    "src/cds2/guided_fit.py",
    "_dataset_keys",
    r'''
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
''',
)

replace_function(
    "src/cds2/guided_fit.py",
    "_dataset_file_stems",
    r'''
def _dataset_file_stems(datasets: tuple[FitDataset, ...]) -> tuple[str, ...]:
    def safe(value: str) -> str:
        cleaned = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in value
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
''',
)

replace_function(
    "src/cds2/guided_fit.py",
    "manifest_dict",
    r'''
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
''',
)

replace_function(
    "src/cds2/guided_fit.py",
    "rerun_manifest",
    r'''
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
    for runtime_key, saved_key, dataset in zip(
        runtime_keys, saved_keys, datasets, strict=True
    ):
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
''',
)

for function_name, scipy_name in (
    ("pearson_correlation", "pearsonr"),
    ("spearman_correlation", "spearmanr"),
    ("kendall_tau", "kendalltau"),
):
    replace_function(
        "src/cds2/stats.py",
        function_name,
        f'''
def {function_name}(x: object, y: object) -> CorrelationResult:
    """Correlation between equal-length finite, non-constant samples."""
    a, b = _paired_arrays(x, y)
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        raise ValueError("correlation is undefined for constant samples")
    res = sps.{scipy_name}(a, b)
    return CorrelationResult(r=float(res.statistic), p_value=float(res.pvalue))
''',
    )

replace_all(
    "src/cds2/epidemiology.py",
    "return math.inf if self.gamma == 0.0 else self.beta / self.gamma",
    "return (0.0 if self.beta == 0.0 else math.inf) if self.gamma == 0.0 else self.beta / self.gamma",
    2,
)
replace_once(
    "src/cds2/epidemiology.py",
    '    if not np.isfinite(beta) or beta <= 0.0:\n        raise ValueError("beta must be positive and finite")\n',
    '    if not np.isfinite(beta) or beta < 0.0:\n        raise ValueError("beta must be non-negative and finite")\n',
)
replace_function(
    "src/cds2/epidemiology.py",
    "herd_immunity_threshold",
    r'''
def herd_immunity_threshold(r0: float) -> float:
    """Critical immune fraction; zero when transmission is already subcritical."""
    if not np.isfinite(r0) or r0 < 0.0:
        raise ValueError("r0 must be non-negative and finite")
    if r0 <= 1.0:
        return 0.0
    return float(1.0 - 1.0 / r0)
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
    integer_args = (("n_samples", n_samples), ("burn_in", burn_in), ("thin", thin))
    for name, value in integer_args:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
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

replace_once(
    "tests/test_epidemiology.py",
    '    @pytest.mark.parametrize("r0", [0.0, -2.5])\n    def test_invalid_raises(self, r0: float) -> None:\n        with pytest.raises(ValueError, match="r0 must be positive"):\n            epi.herd_immunity_threshold(r0)\n',
    '    @pytest.mark.parametrize("r0", [-2.5])\n    def test_invalid_raises(self, r0: float) -> None:\n        with pytest.raises(ValueError, match="r0 must be non-negative"):\n            epi.herd_immunity_threshold(r0)\n\n    def test_zero_is_subcritical(self) -> None:\n        assert epi.herd_immunity_threshold(0.0) == 0.0\n',
)
replace_once(
    "tests/test_epidemiology.py",
    '    def test_beta_not_positive(self) -> None:\n        with pytest.raises(ValueError, match="beta must be positive"):\n            epi.simulate_sir(1000.0, 0.0, 0.1, 10)\n',
    '    def test_beta_negative(self) -> None:\n        with pytest.raises(ValueError, match="beta must be non-negative"):\n            epi.simulate_sir(1000.0, -0.1, 0.1, 10)\n',
)
replace_once(
    "tests/test_epidemiology.py",
    '    def test_beta_not_positive(self) -> None:\n        with pytest.raises(ValueError, match="beta must be positive"):\n            epi.simulate_seir(1000.0, -1.0, 0.2, 0.1, 10)\n',
    '    def test_beta_negative(self) -> None:\n        with pytest.raises(ValueError, match="beta must be non-negative"):\n            epi.simulate_seir(1000.0, -1.0, 0.2, 0.1, 10)\n',
)

test_path = Path("tests/test_correctness_hardening_wave2.py")
test_text = test_path.read_text(encoding="utf-8")
if "test_generated_dataset_keys_cannot_collide_with_real_names" not in test_text:
    test_text += r'''


def test_generated_dataset_keys_cannot_collide_with_real_names() -> None:
    x = np.linspace(1.0, 8.0, 30)
    datasets = (
        gf.FitDataset("a", x, 2.0 * x + 1.0),
        gf.FitDataset("a", x, 3.0 * x + 2.0),
        gf.FitDataset("a#1", x, 4.0 * x + 3.0),
    )
    result = gf.run_guided_fit(datasets, "linear")
    assert len(result.data_hashes) == 3
    assert len(set(result.data_hashes)) == 3
    assert "a#1" in result.data_hashes


def test_plot_file_stems_are_unique_and_cannot_escape_output_dir(tmp_path: Path) -> None:
    x = np.linspace(1.0, 8.0, 30)
    datasets = (
        gf.FitDataset("foo", x, 2.0 * x + 1.0),
        gf.FitDataset("foo", x, 3.0 * x + 2.0),
        gf.FitDataset("foo_1", x, 4.0 * x + 3.0),
        gf.FitDataset("../foo", x, 5.0 * x + 4.0),
    )
    result = gf.run_guided_fit(datasets, "linear")
    paths = gf.plot_result(result, datasets, tmp_path)
    assert len(paths) == 16
    assert len({path.name for path in paths}) == 16
    assert all(path.parent == tmp_path for path in paths)


def test_manifest_rerun_preserves_custom_dataset_name(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    x = np.linspace(1.0, 8.0, 30)
    y = 2.0 * x + 1.0
    pd.DataFrame({"x": x, "y": y}).to_csv(source, index=False)
    dataset = gf.FitDataset("custom scientific name", x, y, source_path=str(source))
    result = gf.run_guided_fit((dataset,), "linear")
    manifest = gf.save_manifest(
        result,
        (dataset,),
        tmp_path / "manifest.json",
        x_column="x",
        y_column="y",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["result"]["data_hashes"] != payload["result"]["rerun_data_hashes"]
    rerun = gf.rerun_manifest(manifest)
    assert rerun.stability_warning is False
    assert rerun.datasets[0].name == "custom scientific name"
    assert set(rerun.data_hashes) == {"custom scientific name"}


@pytest.mark.parametrize(
    "correlation",
    [stats.pearson_correlation, stats.spearman_correlation, stats.kendall_tau],
)
def test_correlations_reject_constant_samples(correlation) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="constant"):
        correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


def test_epidemiology_allows_zero_transmission() -> None:
    result = epi.simulate_sir(100.0, beta=0.0, gamma=0.2, days=3, i0=5.0)
    assert result.r0 == 0.0
    assert epi.herd_immunity_threshold(0.0) == 0.0
    no_change = epi.simulate_sir(100.0, beta=0.0, gamma=0.0, days=2, i0=5.0)
    assert no_change.r0 == 0.0
    assert no_change.infected.tolist() == pytest.approx([5.0, 5.0, 5.0])


def test_metropolis_rejects_noninteger_control_counts() -> None:
    with pytest.raises(ValueError, match="n_samples must be an integer"):
        mc.metropolis_hastings(
            lambda v: -0.5 * v[0] ** 2,
            [0.0],
            n_samples=10.5,  # type: ignore[arg-type]
        )
'''
    test_text = test_text.replace(
        "from pathlib import Path\n",
        "from pathlib import Path\nimport json\n",
        1,
    )
    test_path.write_text(test_text, encoding="utf-8")
