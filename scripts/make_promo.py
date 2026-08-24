"""Generate promotional graphics for scientific-computing-system-2.0."""

from __future__ import annotations

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
        50, 68, "42 modules. One import. Zero bloat.", ha="center", fontsize=17, color=ACCENT_2
    )

    stats = [
        ("470+", "public functions"),
        ("1,277", "tests - 100% cov"),
        ("C kernels", "beating scipy/sklearn"),
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
    figure, axes = dark_canvas(12.8, 7.2)
    races = [
        ("PSO  vs  scipy DE", 0.03),
        ("entropy  vs  numpy loop", 0.07),
        ("PageRank  vs  NetworkX", 0.18),
        ("K-Means  vs  scikit-learn", 0.72),
        ("Latin hypercube  vs  scipy.qmc", 0.74),
    ]
    labels = [r[0] for r in races][::-1]
    values = [r[1] for r in races][::-1]

    axes.text(6, 88, "cds2 vs the specialists", fontsize=24, fontweight="bold", color=TEXT)
    axes.text(
        6,
        80,
        "lower is better  -  time ratio cds2/baseline (smaller = faster)",
        fontsize=11,
        color=MUTED,
    )

    bar_height = 9.0
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 12 + i * 13
        width = max(value * 90, 4)
        axes.add_patch(
            FancyBboxPatch(
                (28, y),
                62,
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
        axes.text(27, y + bar_height / 2, label, ha="right", va="center", fontsize=11.5, color=TEXT)
        label_x = 28 + width + 1.5 if width < 55 else 28 + width - 2
        label_color = BG if width >= 55 else ACCENT_2
        axes.text(
            label_x,
            y + bar_height / 2,
            f"{value:.2f}x",
            ha="right" if width >= 55 else "left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=label_color,
        )

    axes.plot([91, 91], [8, 66], color="#26324d", lw=1.2, ls="--")
    axes.text(91, 70, "parity", ha="center", fontsize=9.5, color=MUTED)
    figure.savefig(OUT / "promo_benchmarks.png", facecolor=BG)
    plt.close(figure)


def modules() -> None:
    figure, axes = dark_canvas(12.8, 7.2)
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
        50, 85, "42 modules, four shelves", ha="center", fontsize=24, fontweight="bold", color=TEXT
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
