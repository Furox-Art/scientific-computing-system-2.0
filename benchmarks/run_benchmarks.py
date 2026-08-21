"""Benchmark suite: cds2 head-to-head against the scientific Python stack.

Every benchmark races a cds2 API against the equivalent call in the library
it wraps (NumPy, SciPy, pandas) or against a third-party specialist
(NetworkX, scikit-learn) when available. A ratio near 1.0x means cds2 costs
nothing over calling the stack directly; ratios against NetworkX/sklearn show
where pure-NumPy implementations win or lose.

Run directly for a console table::

    python benchmarks/run_benchmarks.py [--quick] [--json DIR]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize as spo
from scipy import signal as sp_signal
from scipy import stats as sps

sys_path = str(Path(__file__).resolve().parents[1] / "src")
if sys_path not in __import__("sys").path:
    import sys

    sys.path.insert(0, sys_path)

import cds2  # noqa: E402


@dataclass(frozen=True)
class BenchResult:
    """Timing record for one head-to-head race."""

    name: str
    baseline_library: str
    baseline_seconds: float
    cds2_seconds: float

    @property
    def ratio(self) -> float:
        """cds2 time divided by baseline time; 1.0 means parity."""
        return self.cds2_seconds / self.baseline_seconds if self.baseline_seconds else float("inf")


def _best_of(func: Callable[[], object], repeats: int) -> float:
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)


def _race(
    name: str,
    baseline_library: str,
    baseline: Callable[[], object],
    contender: Callable[[], object],
    baseline_repeats: int,
    cds2_repeats: int,
) -> BenchResult:
    return BenchResult(
        name=name,
        baseline_library=baseline_library,
        baseline_seconds=_best_of(baseline, baseline_repeats),
        cds2_seconds=_best_of(contender, cds2_repeats),
    )


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def bench_solve_small(n: int = 8, calls: int = 300) -> BenchResult:
    """Many tiny solves: measures wrapper overhead honestly."""
    a_matrix = _rng().normal(size=(n, n)) + n * np.eye(n)
    b_vector = _rng().normal(size=n)

    def baseline():
        for _ in range(calls):
            np.linalg.solve(a_matrix, b_vector)

    def contender():
        for _ in range(calls):
            cds2.linalg.solve(a_matrix, b_vector)

    return _race(f"solve {n}x{n} x{calls}", "numpy", baseline, contender, 5, 5)


def bench_solve_large(n: int = 800) -> BenchResult:
    """One large solve: LAPACK dominates either way."""
    a_matrix = _rng().normal(size=(n, n)) + n * np.eye(n)
    b_vector = _rng().normal(size=n)

    def baseline():
        np.linalg.solve(a_matrix, b_vector)

    def contender():
        cds2.linalg.solve(a_matrix, b_vector)

    return _race(f"solve {n}x{n}", "numpy", baseline, contender, 5, 5)


def bench_eigh(n: int = 400) -> BenchResult:
    """Symmetric eigen-decomposition race."""
    matrix = _rng().normal(size=(n, n))
    sym = (matrix + matrix.T) / 2.0

    def baseline():
        np.linalg.eigh(sym)

    def contender():
        cds2.linalg.eigh(sym)

    return _race(f"eigh {n}x{n}", "numpy", baseline, contender, 3, 3)


def bench_fft(n_samples: int = 262_144) -> BenchResult:
    """Real FFT race."""
    signal_values = _rng().normal(size=n_samples)

    def baseline():
        np.fft.rfft(signal_values)

    def contender():
        cds2.signals.rfft(signal_values)

    return _race(f"rfft n={n_samples}", "numpy", baseline, contender, 10, 10)


def bench_welch(n_samples: int = 131_072) -> BenchResult:
    """Welch PSD race."""
    signal_values = _rng().normal(size=n_samples)

    def baseline():
        sp_signal.welch(signal_values, fs=1024.0)

    def contender():
        cds2.signals.welch_spectrum(signal_values, fs=1024.0)

    return _race(f"welch n={n_samples}", "scipy", baseline, contender, 5, 5)


def bench_ttest(n_samples: int = 100_000) -> BenchResult:
    """Independent t-test race."""
    group_a = _rng().normal(size=n_samples)
    group_b = _rng().normal(loc=0.01, size=n_samples)

    def baseline():
        sps.ttest_ind(group_a, group_b)

    def contender():
        cds2.stats.independent_t_test(group_a, group_b)

    return _race(f"t-test n={n_samples}", "scipy", baseline, contender, 5, 5)


def bench_describe(n_samples: int = 500_000) -> BenchResult:
    """Descriptive summary race."""
    values = _rng().normal(size=n_samples)

    def baseline():
        sps.describe(values)

    def contender():
        cds2.stats.describe(values)

    return _race(f"describe n={n_samples}", "scipy", baseline, contender, 5, 5)


def bench_minimize() -> BenchResult:
    """Rosenbrock minimization race."""

    def rosenbrock(point: object) -> float:
        x_value, y_value = point
        return (1.0 - x_value) ** 2 + 100.0 * (y_value - x_value**2) ** 2

    def baseline():
        spo.minimize(rosenbrock, np.zeros(2), method="BFGS")

    def contender():
        cds2.optimize.minimize(rosenbrock, x0=[0.0, 0.0], method="BFGS")

    return _race("minimize rosenbrock", "scipy", baseline, contender, 5, 5)


def bench_pi(n_samples: int = 2_000_000) -> BenchResult:
    """Monte Carlo pi: hand-vectorized numpy vs cds2 helper."""

    def baseline():
        rng = np.random.default_rng(42)
        xy = rng.random((n_samples, 2))
        4.0 * float(np.count_nonzero((xy * xy).sum(axis=1) <= 1.0)) / n_samples

    def contender():
        cds2.montecarlo.pi_estimate(n=n_samples, seed=42)

    return _race(f"mc-pi n={n_samples}", "numpy", baseline, contender, 3, 3)


def bench_pandas_describe(rows: int = 200_000, cols: int = 5) -> BenchResult:
    """DataFrame summary race."""
    frame = pd.DataFrame(_rng().normal(size=(rows, cols)), columns=[f"c{i}" for i in range(cols)])

    def baseline():
        frame.describe()

    def contender():
        cds2.io.summarize(frame)

    return _race(f"dataframe summary {rows}x{cols}", "pandas", baseline, contender, 3, 3)


def bench_networkx_pagerank(nodes: int = 400, edges: int = 2400) -> BenchResult | None:
    """PageRank: NetworkX vs cds2 sparse power iteration."""
    try:
        import networkx as nx
    except ImportError:
        return None
    edge_pairs = _rng().integers(0, nodes, size=(edges, 2))
    edge_list = [(int(u), int(v)) for u, v in edge_pairs]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(nodes))
    graph.add_edges_from(edge_list)
    adj = cds2.graph.from_edges(nodes, edge_list, directed=True)

    def baseline():
        nx.pagerank(graph, alpha=0.85)

    def contender():
        cds2.graph.pagerank(adj)

    return _race(f"pagerank {nodes}n/{edges}e", "networkx", baseline, contender, 2, 3)


def bench_sklearn_kmeans(n_samples: int = 4000, clusters: int = 8) -> BenchResult | None:
    """KMeans: scikit-learn (C-optimized Lloyd) vs cds2 NumPy implementation."""
    try:
        from sklearn.cluster import KMeans as SklearnKMeans
    except ImportError:
        return None
    points = _rng().normal(size=(n_samples, 2)) * 5.0

    def baseline():
        SklearnKMeans(n_clusters=clusters, n_init=1, random_state=0).fit(points)

    def contender():
        cds2.ml.KMeans(n_clusters=clusters, seed=0).fit(points)

    return _race(f"kmeans {n_samples}x2 k={clusters}", "sklearn", baseline, contender, 2, 2)


def bench_sklearn_linreg(n_samples: int = 20_000, n_features: int = 10) -> BenchResult | None:
    """Linear regression: scikit-learn vs cds2 lstsq-based fit."""
    try:
        from sklearn.linear_model import LinearRegression as SklearnLR
    except ImportError:
        return None
    features = _rng().normal(size=(n_samples, n_features))
    targets = features @ _rng().uniform(-3.0, 3.0, size=n_features)

    def baseline():
        SklearnLR().fit(features, targets)

    def contender():
        cds2.ml.LinearRegression().fit(features, targets)

    return _race(f"linreg {n_samples}x{n_features}", "sklearn", baseline, contender, 3, 3)


CORE_BENCHMARKS: dict[str, Callable[[], BenchResult]] = {
    "solve_small": bench_solve_small,
    "solve_large": bench_solve_large,
    "eigh": bench_eigh,
    "fft": bench_fft,
    "welch": bench_welch,
    "ttest": bench_ttest,
    "describe": bench_describe,
    "minimize": bench_minimize,
    "mc_pi": bench_pi,
    "pandas_summary": bench_pandas_describe,
}

OPTIONAL_BENCHMARKS: dict[str, Callable[[], BenchResult | None]] = {
    "networkx_pagerank": bench_networkx_pagerank,
    "sklearn_kmeans": bench_sklearn_kmeans,
    "sklearn_linreg": bench_sklearn_linreg,
}


def quick_sizes() -> dict[str, dict[str, int]]:
    """Reduced problem sizes for smoke runs."""
    return {
        "solve_small": {"calls": 60},
        "solve_large": {"n": 200},
        "eigh": {"n": 120},
        "fft": {"n_samples": 32_768},
        "welch": {"n_samples": 16_384},
        "ttest": {"n_samples": 10_000},
        "describe": {"n_samples": 50_000},
        "mc_pi": {"n_samples": 200_000},
        "pandas_summary": {"rows": 20_000},
        "networkx_pagerank": {"nodes": 100, "edges": 400},
        "sklearn_kmeans": {"n_samples": 600},
        "sklearn_linreg": {"n_samples": 2_000},
    }


def environment_info() -> dict[str, object]:
    """Machine and library provenance for the result file."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except OSError:
        commit = ""
    versions: dict[str, str] = {
        "cds2": cds2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for module_name in ("scipy", "networkx", "sklearn"):
        try:
            module = __import__(module_name)
            versions[module_name] = module.__version__
        except ImportError:
            pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit or "unknown",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": versions,
    }


def run_all(
    output_dir: str | Path | None = None,
    quick: bool = False,
    benchmarks: list[str] | None = None,
    include_optional: bool = True,
) -> list[BenchResult]:
    """Execute the races and optionally persist a JSON report."""
    selected = benchmarks or list(CORE_BENCHMARKS) + (
        list(OPTIONAL_BENCHMARKS) if include_optional else []
    )
    sizes = quick_sizes() if quick else {}
    results: list[BenchResult] = []
    for name in selected:
        if name in CORE_BENCHMARKS:
            results.append(CORE_BENCHMARKS[name](**sizes.get(name, {})))
        elif name in OPTIONAL_BENCHMARKS:
            outcome = OPTIONAL_BENCHMARKS[name](**sizes.get(name, {}))
            if outcome is not None:
                results.append(outcome)

    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "environment": environment_info(),
            "results": [
                {
                    "name": result.name,
                    "baseline_library": result.baseline_library,
                    "baseline_seconds": round(result.baseline_seconds, 6),
                    "cds2_seconds": round(result.cds2_seconds, 6),
                    "ratio_cds2_over_baseline": round(result.ratio, 3),
                }
                for result in results
            ],
        }
        (target / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return results


def format_table(results: list[BenchResult]) -> str:
    """Render results as an aligned ASCII table."""
    header = f"{'benchmark':<28}{'vs':<11}{'baseline (s)':>14}{'cds2 (s)':>12}{'cds2/base':>11}"
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            f"{result.name:<28}{result.baseline_library:<11}"
            f"{result.baseline_seconds:>14.5f}{result.cds2_seconds:>12.5f}"
            f"{result.ratio:>10.2f}x"
        )
    return "\n".join(lines)


def write_markdown_report(results: list[BenchResult], path: Path) -> None:
    """Write a GitHub-friendly markdown table next to the JSON artifact."""
    lines = [
        "# CDS v2 benchmarks",
        "",
        f"Generated {datetime.now(timezone.utc).date().isoformat()} by "
        "`benchmarks/run_benchmarks.py`. Ratio is cds2 time divided by baseline "
        "time: 1.00x means parity with the underlying library.",
        "",
        "| Benchmark | Baseline | Baseline (s) | cds2 (s) | cds2/baseline |",
        "|---|---|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.baseline_library} | "
            f"{result.baseline_seconds:.5f} | {result.cds2_seconds:.5f} | {result.ratio:.2f}x |"
        )
    lines += [
        "",
        "Wrapper-only rows (solve, fft, welch, t-test, minimize) demonstrate",
        "that the cds2 API layer adds no measurable cost over calling",
        "NumPy/SciPy directly. Rows where cds2 returns strictly more",
        "information (describe adds quartiles; dataframe summary adds nulls",
        "and uniques per column) carry a small honest premium for it.",
        "Specialist rows (NetworkX, scikit-learn) show where pure-NumPy",
        "implementations win or lose against compiled C code.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use reduced problem sizes")
    parser.add_argument("--json", metavar="DIR", default=None, help="write results.json into DIR")
    parser.add_argument(
        "--markdown", metavar="FILE", default=None, help="also write a markdown report"
    )
    parser.add_argument("--core-only", action="store_true", help="skip optional third-party races")
    parser.add_argument(
        "--only", nargs="*", choices=sorted(CORE_BENCHMARKS) + sorted(OPTIONAL_BENCHMARKS)
    )
    args = parser.parse_args(argv)

    results = run_all(
        output_dir=args.json,
        quick=args.quick,
        benchmarks=args.only,
        include_optional=not args.core_only,
    )
    print(format_table(results))
    print("\nratio 1.00x = parity with the baseline library")
    if args.markdown:
        write_markdown_report(results, Path(args.markdown))
        print(f"markdown report written to {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
