"""Oracle tests: cds2.integrate vs SciPy ground truth."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate as spi

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import cds2  # noqa: E402


class TestIntegrateOracles:
    """Verify cds2.integrate against SciPy."""

    @pytest.mark.parametrize(
        "func,a,b",
        [
            (np.sin, 0, np.pi),
            (np.exp, 0, 1),
            (lambda x: x**2, 0, 1),
            (lambda x: np.log(x + 1), 0, 1),
            (lambda x: np.sin(x) ** 2, 0, 2 * np.pi),
        ],
    )
    def test_quad_vs_scipy(self, func: callable, a: float, b: float) -> None:
        res = cds2.integrate.quad(func, a, b)
        sp_val, _ = spi.quad(func, a, b)
        assert np.isclose(res.value, sp_val, rtol=1e-6)

    @pytest.mark.parametrize("seed", range(10))
    def test_trapz_vs_numpy(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = np.sort(rng.uniform(0, 10, size=100))
        y = rng.normal(size=100)
        res = cds2.integrate.trapezoid(y, x)
        np_val = np.trapezoid(y, x)
        assert np.isclose(res, np_val, atol=1e-10)
