"""Tests for cds2.viz (Agg backend, no display)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cds2 import viz


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPlots:
    def test_plot_series_returns_figure(self) -> None:
        fig = viz.plot_series([1.0, 3.0, 2.0], title="t", xlabel="x", ylabel="y")
        assert isinstance(fig, plt.Figure)
        assert fig.axes[0].get_title() == "t"

    def test_plot_series_multi_row(self) -> None:
        fig = viz.plot_series(np.array([[0.0, 1.0], [1.0, 0.5]]))
        line_count = sum(len(ax.lines) for ax in fig.axes)
        assert line_count == 2

    def test_plot_histogram(self) -> None:
        fig = viz.plot_histogram(np.random.default_rng(0).normal(size=200), bins=15)
        assert len(fig.axes[0].patches) == 15

    def test_plot_scatter(self) -> None:
        fig = viz.plot_scatter([1, 2, 3], [2, 4, 6])
        offsets = fig.axes[0].collections[0].get_offsets()
        assert len(offsets) == 3

    def test_plot_heatmap_colorbar(self) -> None:
        fig = viz.plot_heatmap(np.arange(12, dtype=float).reshape(3, 4))
        assert len(fig.axes) == 2

    def test_plot_spectrum(self) -> None:
        t_values = np.arange(512) / 128.0
        signal = np.sin(2 * np.pi * 20.0 * t_values)
        fig = viz.plot_spectrum(signal, fs=128.0)
        assert fig.axes[0].get_ylabel() == "PSD"

    def test_plot_regression_overlay(self) -> None:
        x = np.linspace(0, 1, 30)
        y = 3.0 * x + 0.05 * np.random.default_rng(1).normal(size=30)
        fig = viz.plot_regression(x, y, degree=1)
        assert len(fig.axes[0].lines) == 1

    def test_plot_confusion_matrix_annotation(self) -> None:
        fig = viz.plot_confusion_matrix([[5, 1], [2, 8]], labels=["neg", "pos"])
        texts = [t for ax in fig.axes for t in ax.texts]
        assert any(t.get_text() == "8" for t in texts)


class TestSaveFigure:
    def test_save_png(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        target = tmp_path / "out.png"
        fig = viz.plot_series([1.0, 2.0, 3.0])
        saved = viz.save_figure(fig, target)
        assert saved.exists()
        assert saved.stat().st_size > 0

    def test_save_creates_parent_dirs(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        target = tmp_path / "deep" / "nested" / "fig.png"
        fig = viz.plot_scatter([1], [1])
        viz.save_figure(fig, target)
        assert target.exists()

    def test_plot_with_save_argument(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        target = tmp_path / "direct.png"
        viz.plot_series([1.0, 2.0], save=target)
        assert target.exists()
