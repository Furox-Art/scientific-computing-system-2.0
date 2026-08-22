"""Matplotlib plotting helpers for common scientific charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "plot_series",
    "plot_histogram",
    "plot_scatter",
    "plot_heatmap",
    "plot_spectrum",
    "plot_regression",
    "plot_confusion_matrix",
    "save_figure",
]


def plot_series(
    data: object,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save: str | Path | None = None,
) -> Figure:
    """Line plot of one or more series; a 2-D input plots each row."""
    values = np.asarray(data, dtype=float)
    fig, ax = plt.subplots()
    if values.ndim == 1:
        ax.plot(values)
    else:
        for row in values:
            ax.plot(row)
    _decorate(ax, title, xlabel, ylabel)
    if save is not None:
        save_figure(fig, save)
    return fig


def plot_histogram(
    data: object,
    bins: int = 30,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "frequency",
    save: str | Path | None = None,
) -> Figure:
    """Histogram of a sample."""
    values = np.asarray(data, dtype=float).ravel()
    fig, ax = plt.subplots()
    ax.hist(values, bins=bins, edgecolor="black", linewidth=0.4)
    _decorate(ax, title, xlabel, ylabel)
    if save is not None:
        save_figure(fig, save)
    return fig


def plot_scatter(
    x: object,
    y: object,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save: str | Path | None = None,
) -> Figure:
    """Scatter plot of paired samples."""
    fig, ax = plt.subplots()
    ax.scatter(np.asarray(x), np.asarray(y), s=18, alpha=0.8)
    _decorate(ax, title, xlabel, ylabel)
    if save is not None:
        save_figure(fig, save)
    return fig


def plot_heatmap(
    matrix: object,
    cmap: str = "viridis",
    colorbar: bool = True,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save: str | Path | None = None,
) -> Figure:
    """Heatmap of a 2-D matrix."""
    values = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots()
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    if colorbar:
        fig.colorbar(image, ax=ax)
    _decorate(ax, title, xlabel, ylabel)
    if save is not None:
        save_figure(fig, save)
    return fig


def plot_spectrum(
    x: object,
    fs: float,
    title: str = "Power spectrum",
    xlabel: str = "frequency (Hz)",
    ylabel: str = "PSD",
    save: str | Path | None = None,
) -> Figure:
    """Log-scale power spectral density of a sampled signal."""
    from .signals import power_spectrum

    spectrum = power_spectrum(x, fs)
    keep = spectrum.frequencies > 0
    fig, ax = plt.subplots()
    ax.semilogy(spectrum.frequencies[keep], spectrum.power[keep])
    _decorate(ax, title, xlabel, ylabel)
    if save is not None:
        save_figure(fig, save)
    return fig


def plot_regression(
    x: object,
    y: object,
    degree: int = 1,
    points: int = 200,
    title: str = "Regression fit",
    xlabel: str = "",
    ylabel: str = "",
    save: str | Path | None = None,
) -> Figure:
    """Scatter plot with a least-squares polynomial overlay."""
    x_values = np.asarray(x, dtype=float).ravel()
    y_values = np.asarray(y, dtype=float).ravel()
    coefficients = np.polyfit(x_values, y_values, degree)
    grid = np.linspace(float(x_values.min()), float(x_values.max()), points)
    fig, ax = plt.subplots()
    ax.scatter(x_values, y_values, s=18, alpha=0.8, label="data")
    ax.plot(
        grid,
        np.polyval(coefficients, grid),
        color="crimson",
        linewidth=2,
        label=f"degree {degree} fit",
    )
    ax.legend(loc="best")
    _decorate(ax, title, xlabel, ylabel)
    if save is not None:
        save_figure(fig, save)
    return fig


def plot_confusion_matrix(
    matrix: object,
    labels: list[str] | None = None,
    title: str = "Confusion matrix",
    save: str | Path | None = None,
) -> Figure:
    """Annotated confusion-matrix heatmap."""
    values = np.asarray(matrix)
    n_classes = values.shape[0]
    names = labels if labels is not None else [str(i) for i in range(n_classes)]
    fig, ax = plt.subplots()
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax)
    threshold = float(values.max()) / 2.0 if values.size else 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if values[i, j] > threshold else "black"
            ax.text(j, i, str(values[i, j]), ha="center", va="center", color=color)
    ticks = list(range(n_classes))
    ax.set_xticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticks(ticks)
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    if save is not None:
        save_figure(fig, save)
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 150, close: bool = True) -> Path:
    """Save a figure to disk and optionally close it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    if close:
        matplotlib.pyplot.close(fig)
    return target


def _decorate(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
