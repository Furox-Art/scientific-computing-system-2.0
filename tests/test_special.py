"""Tests for cds2.special."""

import numpy as np
import pytest

from cds2 import special


class TestGammaFamily:
    def test_gamma_integer_factorial(self) -> None:
        assert special.gamma_fn(5.0)[0] == pytest.approx(24.0)
        assert special.gamma_fn([1.0, 2.0, 3.0])[2] == pytest.approx(2.0)

    def test_gammaln_consistent_with_gamma(self) -> None:
        values = np.array([0.5, 1.5, 7.25])
        assert np.allclose(special.gammaln(values), np.log(special.gamma_fn(values)))

    def test_beta_known_value(self) -> None:
        assert special.beta_fn(2.0, 3.0) == pytest.approx(1.0 / 12.0)
        assert special.betaln(2.0, 3.0) == pytest.approx(np.log(1.0 / 12.0))


class TestErrorFunctions:
    def test_erf_bounds_and_origin(self) -> None:
        assert special.erf(0.0)[0] == 0.0
        assert special.erf(5.0)[0] == pytest.approx(1.0)
        assert special.erf(-1.0)[0] < 0

    def test_erfc_complement_accurate_in_tail(self) -> None:
        assert special.erfc(0.0)[0] == pytest.approx(1.0)
        deep_tail = special.erfc(6.0)[0]
        assert 0.0 < deep_tail < 1e-8

    def test_erfinv_roundtrip(self) -> None:
        grid = np.linspace(-0.9, 0.9, 21)
        recovered = special.erfinv(special.erf(grid))
        assert np.allclose(recovered, grid)


class TestBesselAndZeta:
    def test_j0_at_zero(self) -> None:
        assert special.bessel_j0(0.0)[0] == pytest.approx(1.0)
        assert special.bessel_j1(0.0)[0] == pytest.approx(0.0)

    def test_y0_negative_for_small_positive_args(self) -> None:
        assert special.bessel_y0(0.5)[0] < 0

    @pytest.mark.parametrize("argument", [0.5, 1.0, 2.5])
    def test_j0_finite(self, argument: float) -> None:
        assert np.isfinite(special.bessel_j0(argument)[0])

    def test_zeta_two_is_pi_squared_over_six(self) -> None:
        assert special.zeta(2.0) == pytest.approx((np.pi**2) / 6.0, rel=1e-12)
