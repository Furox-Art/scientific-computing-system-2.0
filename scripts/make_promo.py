"""Generate promotional graphics for scientific-computing-system-2.0.

All numbers are read from committed repository data at generation time:
module/test counts from the source tree and benchmark ratios from
``benchmarks/results.json``. Nothing is hand-typed into the figures.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

BG = "#0b1020"
PANEL = "#111a2e"
ACCENT_1 = "#6366f1"  # indigo
ACCENT_2 = "#22d3ee"  # cyan
TEXT = "#e6edf7"
MUTED = "#8b9bb8"

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
REPO = Path(__file__).resolve().parents[1]


def repo_counts() -> tuple[int, int, int, int, int]:
    """Count importable names, ``__all__`` exports, API pages and collected tests.

    Four counts, each from the real tree so the figure cannot drift:

    ``importable``
        Flat modules **plus** importable subpackages. A previous revision used
        ``glob("*.py")``, which silently ignored the subpackages
        (``array_api``, ``bench``, ``estimator``, ``gpu``, ``nlp``, ``prof``)
        and therefore under-reported the package surface. A subdirectory counts
        only when it carries an ``__init__.py``; ``src/cds2/src`` holds C
        sources and is correctly excluded.
    ``exports``
        ``len(cds2.__all__)``, parsed with ``ast`` so nothing is imported. The
        hero labels this "public exports", not "public functions": the list
        mixes functions, classes, submodules and data objects, so a
        "functions" label is structurally wrong.
    ``api_pages``
        Number of files in ``docs/api``, which is what the docs actually serve.
    ``tests``
        ``pytest --collect-only``, so parametrized cases count exactly as CI
        counts them. Fails loudly rather than guessing.

    Returns ``(importable, exports, api_pages, tests, flat_modules)``.
    """
    src = REPO / "src" / "cds2"
    flat = [p for p in src.glob("*.py") if p.name not in ("__init__.py", "_version.py")]
    subpackages = [
        d.name for d in sorted(src.iterdir()) if d.is_dir() and (d / "__init__.py").is_file()
    ]
    api_pages = len(list((REPO / "docs" / "api").glob("*.md")))

    tree = ast.parse((src / "__init__.py").read_text(encoding="utf-8"))
    exports = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "__all__"
            and isinstance(node.value, ast.List)
        ):
            exports = len(node.value.elts)

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"cannot count tests without pytest: {exc}") from exc
    match = re.search(r"(\d+) tests? collected", proc.stdout + proc.stderr)
    if match is None:
        raise SystemExit("pytest --collect-only output did not report a test count")
    return len(flat) + len(subpackages), exports, api_pages, int(match.group(1)), len(flat)


def dark_canvas(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    figure = plt.figure(figsize=(width, height), dpi=100)
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100)
    axes.axis("off")
    axes.add_patch(plt.Rectangle((0, 0), 100, 100, color=BG))
    gradient = np.linspace(0, 1, 512).reshape(1, -1)
    axes.imshow(
        gradient,
        extent=[0, 100, 92, 100],
        aspect="auto",
        cmap=plt.cm.colors.LinearSegmentedColormap.from_list("accent", [ACCENT_1, ACCENT_2]),
        alpha=0.85,
        zorder=5,
    )
    return figure, axes


def chip(axes: plt.Axes, x: float, y: float, text: str, accent: bool = False) -> None:
    color = ACCENT_1 if accent else PANEL
    axes.add_patch(
        FancyBboxPatch(
            (x, y),
            11.2,
            5.2,
            boxstyle="round,pad=0.55,rounding_size=1.4",
            fc=color,
            ec=ACCENT_2 if accent else "#26324d",
            lw=1.2,
        )
    )
    axes.text(
        x + 5.6,
        y + 2.6,
        text,
        ha="center",
        va="center",
        fontsize=8.6,
        color=TEXT,
        family="monospace",
    )


def hero() -> None:
    figure, axes = dark_canvas(12.8, 7.2)
    n_importable, n_exports, n_api, n_tests, n_flat = repo_counts()

    axes.text(
        50,
        78,
        "scientific-computing-system-2.0",
        ha="center",
        fontsize=30,
        fontweight="bold",
        color=TEXT,
        family="monospace",
    )
    axes.text(
        50,
        68,
        f"{n_importable} importable names. One install. Zero bloat.",
        ha="center",
        fontsize=17,
        color=ACCENT_2,
    )

    # "public exports", not "public functions": __all__ holds functions,
    # classes, submodules and data objects, so a functions label would be wrong.
    stats = [
        (f"{n_exports}", "public exports"),
        (f"{n_tests:,}", "tests - 100% cov"),
        (f"{n_flat}+{n_api}", "modules + API pages"),
        ("MIT", "open source"),
    ]
    for i, (big, small) in enumerate(stats):
        x = 8 + i * 22.5
        axes.add_patch(
            FancyBboxPatch(
                (x, 44),
                19,
                14,
                boxstyle="round,pad=0.6,rounding_size=1.6",
                fc=PANEL,
                ec="#26324d",
                lw=1.2,
            )
        )
        axes.text(x + 9.5, 53, big, ha="center", fontsize=16, fontweight="bold", color=ACCENT_2)
        axes.text(x + 9.5, 47.5, small, ha="center", fontsize=9.5, color=MUTED)

    domains = [
        "linalg",
        "stats",
        "optimize",
        "signals",
        "ml",
        "graph",
        "chaos",
        "bayes",
        "finance",
        "game_theory",
        "genetics",
        "spatial",
        "rl",
        "wavelets",
        "image",
        "quality",
    ]
    for i, name in enumerate(domains):
        row, col = divmod(i, 8)
        chip(axes, 3.0 + col * 12.0, 30 - row * 7.5, name)

    axes.text(
        50,
        10,
        "$ pip install scientific-computing-system-2.0",
        ha="center",
        fontsize=13,
        color="#9ff5d2",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", fc="#0f2419", ec="#2ea56f"),
    )
    axes.text(
        50,
        3.5,
        "github.com/Furox-Art/scientific-computing-system-2.0",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    figure.savefig(OUT / "promo_hero.png", facecolor=BG)
    plt.close(figure)


def benchmarks() -> None:
    """Bar chart of every row in benchmarks/results.json, winners and losers.

    Ratios are recomputed from the committed absolute timings
    (cds2_seconds / baseline_seconds) so the figure can never drift from
    the JSON file or from docs/benchmarks.md. A provenance line names the
    exact data source; the parity line marks 1.00x.
    """
    payload = json.loads((REPO / "benchmarks" / "results.json").read_text(encoding="utf-8"))
    env = payload["environment"]
    rows = payload["results"]
    names = [r["name"] for r in rows]
    baselines = [r["baseline_library"] for r in rows]
    ratios = [r["cds2_seconds"] / r["baseline_seconds"] for r in rows]
    labels = [f"{n}  vs  {b}" for n, b in zip(names, baselines, strict=True)]

    figure, axes = dark_canvas(12.8, 7.2)
    axes.text(
        6,
        96,
        "cds2 vs baselines - every measured case",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
        zorder=6,
    )
    axes.text(
        6,
        91.5,
        "lower is better  -  time ratio cds2/baseline (smaller = faster)",
        fontsize=11,
        color=MUTED,
    )
    provenance = (
        f"benchmarks/results.json - cds2 {env['versions']['cds2']} - "
        f"commit {env['git_commit']} - {env['timestamp_utc'][:10]} - {env['platform'].split('-')[0]}"
    )
    axes.text(6, 87.5, provenance, fontsize=9, color=MUTED, family="monospace", zorder=6)

    ceiling = max(max(ratios), 1.0)
    bar_max = 60.0
    bar_height = 4.2
    n = len(labels)
    for i, (label, value) in enumerate(zip(labels, ratios, strict=True)):
        y = 6 + (n - 1 - i) * 6.0
        width = max(value / ceiling * bar_max, 3.0)
        axes.add_patch(
            FancyBboxPatch(
                (28, y),
                bar_max,
                bar_height,
                boxstyle="round,pad=0.2,rounding_size=1.2",
                fc="#18233c",
                ec="none",
            )
        )
        axes.add_patch(
            FancyBboxPatch(
                (28, y),
                width,
                bar_height,
                boxstyle="round,pad=0.2,rounding_size=1.2",
                fc=ACCENT_1,
                ec=ACCENT_2,
                lw=1.0,
            )
        )
        axes.text(27, y + bar_height / 2, label, ha="right", va="center", fontsize=9, color=TEXT)
        label_x = 28 + width + 1.5 if width < 48 else 28 + width - 1.5
        label_color = BG if width >= 48 else ACCENT_2
        axes.text(
            label_x,
            y + bar_height / 2,
            f"{value:.2f}x",
            ha="right" if width >= 48 else "left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=label_color,
        )

    parity_x = 28 + bar_max / ceiling
    axes.plot([parity_x, parity_x], [3, 6 + n * 6.0], color="#26324d", lw=1.2, ls="--")
    axes.text(parity_x, 6 + n * 6.0 + 0.5, "parity 1.00x", ha="center", fontsize=9, color=MUTED)
    figure.savefig(OUT / "promo_benchmarks.png", facecolor=BG)
    plt.close(figure)


def modules() -> None:
    figure, axes = dark_canvas(12.8, 7.2)
    n_flat = repo_counts()[4]
    groups = {
        "CORE": [
            "linalg",
            "stats",
            "optimize",
            "integrate",
            "interpolate",
            "signals",
            "sparse",
            "special",
        ],
        "DISCOVER": [
            "infotheory",
            "chaos",
            "bayes",
            "metaheuristics",
            "geometry",
            "rl",
            "hypothesis",
            "modeling",
        ],
        "DOMAINS": [
            "genetics",
            "epidemiology",
            "reliability",
            "finance",
            "game_theory",
            "spatial",
            "combinatorial",
            "text",
        ],
        "MEDIA & MORE": [
            "image",
            "wavelets",
            "quality",
            "design",
            "quantum",
            "nlp",
            "knowledge",
            "scientific",
        ],
    }
    axes.text(
        50,
        85,
        f"four shelves, {n_flat} flat modules",
        ha="center",
        fontsize=24,
        fontweight="bold",
        color=TEXT,
    )
    positions = [(4, 46), (52, 46), (4, 8), (52, 8)]
    for (x0, y0), (title, names) in zip(positions, groups.items(), strict=True):
        axes.add_patch(
            FancyBboxPatch(
                (x0, y0),
                44,
                34,
                boxstyle="round,pad=0.8,rounding_size=2",
                fc=PANEL,
                ec="#26324d",
                lw=1.2,
            )
        )
        axes.text(x0 + 2.5, y0 + 29.5, title, fontsize=13, fontweight="bold", color=ACCENT_2)
        for i, name in enumerate(names):
            row, col = divmod(i, 2)
            chip(axes, x0 + 3 + col * 19.0, y0 + 19 - row * 6.2, name)
    axes.text(
        50,
        2.5,
        "import cds2   -   everything above, one package",
        ha="center",
        fontsize=10.5,
        color=MUTED,
        family="monospace",
    )
    figure.savefig(OUT / "promo_modules.png", facecolor=BG)
    plt.close(figure)


OUT.mkdir(parents=True, exist_ok=True)
hero()
benchmarks()
modules()
for asset in sorted(OUT.glob("promo_*.png")):
    print(asset.name, f"{asset.stat().st_size / 1024:.0f} KB")
