"""Oracle tests: cds2.stats vs SciPy ground truth."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import cds2  # noqa: E402


class TestStatsOracles:
    """Verify cds2.stats against SciPy."""

    @pytest.mark.parametrize("seed", range(10))
    def test_t_test_vs_scipy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        a = rng.normal(0, 1, size=50)
        b = rng.normal(0.5, 1, size=50)
        res = cds2.stats.independent_t_test(a, b)
        sp_t, sp_p = sps.ttest_ind(a, b)
        assert np.isclose(res.statistic, sp_t, atol=1e-6)
        assert np.isclose(res.p_value, sp_p, atol=1e-6)

    @pytest.mark.parametrize("seed", range(10))
    def test_pearson_vs_scipy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=100)
        y = x + rng.normal(0, 0.1, size=100)
        res = cds2.stats.pearson_correlation(x, y)
        sp_r, sp_p = sps.pearsonr(x, y)
        assert np.isclose(res.r, sp_r, atol=1e-6)
        assert np.isclose(res.p_value, sp_p, atol=1e-6)

    @pytest.mark.parametrize("seed", range(10))
    def test_normality_vs_scipy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        data = rng.normal(0, 1, size=100)
        res = cds2.stats.normality_test(data)
        sp_stat, sp_p = sps.shapiro(data)
        assert np.isclose(res.statistic, sp_stat, atol=1e-6)
        assert np.isclose(res.p_value, sp_p, atol=1e-6)
