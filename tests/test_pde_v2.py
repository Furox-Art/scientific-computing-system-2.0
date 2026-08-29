"""Tests for :mod:`cds2.pde` — accelerated numpy FTCS and leapfrog solvers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cds2.pde import (
    Heat2DResult,
    HeatResult,
    Wave2DResult,
    WaveResult,
    heat_equation_1d,
    heat_equation_2d,
    wave_equation_1d,
    wave_equation_2d,
)


def _gaussian_1d(nx: int, length: float, center: float, width: float) -> np.ndarray:
    xs = np.linspace(0.0, length, nx, dtype=np.float64)
    return np.exp(-(((xs - center) / width) ** 2)).astype(np.float64)


def test_heat_1d_dirichlet_pinning_and_auto_dt() -> None:
    nx, length, alpha = 21, 2.0, 0.7
    u0 = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    res = heat_equation_1d(u0, alpha, length, 0.01, nx)
    assert isinstance(res, HeatResult)
    dx = length / (nx - 1)
    assert res.dt == pytest.approx(0.9 * dx * dx / (2.0 * alpha))
    assert res.n_steps == math.ceil(0.01 / res.dt)
    assert res.u_final.shape == (nx,)
    assert res.u_final[0] == pytest.approx(u0[0])
    assert res.u_final[-1] == pytest.approx(u0[-1])


def test_heat_1d_neumann_mirrors_and_flat_stationary() -> None:
    nx = 41
    flat = np.full(nx, 3.25, dtype=np.float64)
    res_flat = heat_equation_1d(flat, 1.3, 1.0, 0.5, nx, boundary="neumann")
    np.testing.assert_allclose(res_flat.u_final, flat)
    # also Dirichlet flat must stay stationary
    res_flat_d = heat_equation_1d(flat, 1.3, 1.0, 0.5, nx, boundary="dirichlet")
    np.testing.assert_allclose(res_flat_d.u_final, flat)
    g = _gaussian_1d(nx, 1.0, 0.5, 0.15)
    res = heat_equation_1d(g, 1.0, 1.0, 0.05, nx, boundary="neumann")
    assert res.u_final[0] == pytest.approx(res.u_final[1])
    assert res.u_final[-1] == pytest.approx(res.u_final[-2])


def test_heat_1d_sine_analytic_decay() -> None:
    alpha, nx, length, t_final = 1.0, 101, 1.0, 0.1
    xs = np.linspace(0.0, length, nx, dtype=np.float64)
    u0 = np.sin(math.pi * xs)
    res = heat_equation_1d(u0, alpha, length, t_final, nx)
    t_sim = res.n_steps * res.dt
    expected = math.exp(-alpha * math.pi**2 * t_sim)
    assert res.u_final[nx // 2] == pytest.approx(expected, rel=0.05)
    # gaussian peak must decay
    g0 = _gaussian_1d(201, 1.0, 0.5, 0.08)
    res_g = heat_equation_1d(g0, 0.5, 1.0, 0.05, 201)
    assert res_g.u_final[100] < 0.75 * g0[100]


def test_heat_1d_validation_rejects_bad_inputs() -> None:
    # u0 size mismatch
    with pytest.raises(ValueError, match="exactly 5 points"):
        heat_equation_1d(np.zeros(4), 1.0, 1.0, 0.1, 5)
    with pytest.raises(ValueError, match="at least 3"):
        heat_equation_1d(np.zeros(2), 1.0, 1.0, 0.1, 2)
    with pytest.raises(ValueError, match="alpha must be positive"):
        heat_equation_1d(np.zeros(5), 0.0, 1.0, 0.1, 5)
    with pytest.raises(ValueError, match="length must be positive"):
        heat_equation_1d(np.zeros(5), 1.0, -1.0, 0.1, 5)
    with pytest.raises(ValueError, match="t_final must be positive"):
        heat_equation_1d(np.zeros(5), 1.0, 1.0, 0.0, 5)
    with pytest.raises(ValueError, match="boundary must be"):
        heat_equation_1d(np.zeros(5), 1.0, 1.0, 0.1, 5, boundary="periodic")
    with pytest.raises(ValueError, match="dt must be positive"):
        heat_equation_1d(np.zeros(11), 1.0, 1.0, 0.1, 11, dt=0.0)
    # unstable dt: dx=0.1, alpha=1, dt=0.06 -> r=6
    with pytest.raises(ValueError, match="violates heat stability"):
        heat_equation_1d(np.zeros(11), 1.0, 1.0, 0.05, 11, dt=0.06)


def test_wave_1d_cfl_guarded_and_auto_dt() -> None:
    nx, length, c = 21, 2.0, 1.4
    res = wave_equation_1d(np.zeros(nx), np.zeros(nx), c, length, 0.1, nx)
    assert isinstance(res, WaveResult)
    dx = length / (nx - 1)
    assert res.dt == pytest.approx(0.9 * dx / c)
    assert res.n_steps == math.ceil(0.1 / res.dt)
    # user dt within CFL is respected
    res2 = wave_equation_1d(np.zeros(11), np.zeros(11), 1.0, 1.0, 0.03, 11, dt=0.01)
    assert res2.dt == pytest.approx(0.01)
    assert res2.n_steps == 3
    # CFL violation
    with pytest.raises(ValueError, match="violates wave CFL"):
        wave_equation_1d(np.zeros(11), np.zeros(11), 1.0, 1.0, 0.05, 11, dt=0.15)
    with pytest.raises(ValueError, match="v0 must have exactly"):
        wave_equation_1d(np.zeros(7), np.zeros(6), 1.0, 1.0, 0.1, 7)
    with pytest.raises(ValueError, match="c must be positive"):
        wave_equation_1d(np.zeros(5), np.zeros(5), 0.0, 1.0, 0.1, 5)
    # pinned ends
    u0 = np.linspace(0.0, 1.0, 41, dtype=np.float64)
    res_pin = wave_equation_1d(u0, np.zeros(41), 1.0, 1.0, 0.25, 41)
    assert res_pin.u_final[0] == pytest.approx(0.0)
    assert res_pin.u_final[-1] == pytest.approx(1.0)


def test_wave_1d_standing_wave_physics() -> None:
    nx, c, length = 201, 1.0, 1.0
    xs = np.linspace(0.0, length, nx, dtype=np.float64)
    u0 = np.sin(math.pi * xs)
    v0 = np.zeros(nx, dtype=np.float64)
    # reconstruct after one period (2*L/c = 2.0)
    res = wave_equation_1d(u0, v0, c, length, 2.0, nx)
    phase = math.pi * c * (res.n_steps * res.dt) / length
    expected = np.cos(phase) * u0
    np.testing.assert_allclose(res.u_final, expected, atol=0.02)
    # invert after half period
    res_half = wave_equation_1d(u0, v0, c, length, 1.0, nx)
    phase_half = math.pi * c * (res_half.n_steps * res_half.dt) / length
    expected_half = np.cos(phase_half) * u0
    np.testing.assert_allclose(res_half.u_final, expected_half, atol=0.02)
    assert res_half.u_final[nx // 2] < 0.0
    # velocity kick starts linear motion (nx=101 for this sub-check)
    nx2 = 101
    xs2 = np.linspace(0.0, length, nx2, dtype=np.float64)
    v_sine2 = np.sin(math.pi * xs2)
    res_v = wave_equation_1d(np.zeros(nx2), v_sine2, c, length, 0.01, nx2)
    t_sim = res_v.n_steps * res_v.dt
    omega = math.pi * c / length
    scale = math.sin(omega * t_sim) / omega
    expected_v = v_sine2 * scale
    np.testing.assert_allclose(res_v.u_final, expected_v, atol=5e-4)


def test_heat_2d_vectorised_shapes_stability_and_boundaries() -> None:
    nx, ny, lx, ly = 21, 16, 2.0, 1.5
    # flat field stays flat for both boundaries
    flat = np.full((ny, nx), 2.0, dtype=np.float64)
    res_flat_d = heat_equation_2d(flat, 0.5, lx, ly, 0.1, nx, ny, boundary="dirichlet")
    assert isinstance(res_flat_d, Heat2DResult)
    np.testing.assert_allclose(res_flat_d.u_final, flat)
    res_flat_n = heat_equation_2d(flat, 0.5, lx, ly, 0.1, nx, ny, boundary="neumann")
    np.testing.assert_allclose(res_flat_n.u_final, flat)
    # gaussian blob in centre must decay and stay bounded
    xs = np.linspace(0.0, lx, nx, dtype=np.float64)
    ys = np.linspace(0.0, ly, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    blob = np.exp(-(((xx - lx / 2) / 0.4) ** 2 + ((yy - ly / 2) / 0.3) ** 2))
    res_blob = heat_equation_2d(blob, 0.3, lx, ly, 0.05, nx, ny)
    assert res_blob.u_final.shape == (ny, nx)
    assert res_blob.u_final[ny // 2, nx // 2] < blob[ny // 2, nx // 2]
    assert np.all(np.isfinite(res_blob.u_final))
    # neumann mirrors borders
    res_n = heat_equation_2d(blob, 0.3, lx, ly, 0.05, nx, ny, boundary="neumann")
    np.testing.assert_allclose(res_n.u_final[0, :], res_n.u_final[1, :])
    np.testing.assert_allclose(res_n.u_final[-1, :], res_n.u_final[-2, :])
    np.testing.assert_allclose(res_n.u_final[:, 0], res_n.u_final[:, 1])
    np.testing.assert_allclose(res_n.u_final[:, -1], res_n.u_final[:, -2])
    # stability guard
    with pytest.raises(ValueError, match="violates heat stability"):
        heat_equation_2d(np.full((11, 11), 1.0), 1.0, 1.0, 1.0, 0.02, 11, 11, dt=0.1)
    with pytest.raises(ValueError, match="shape"):
        heat_equation_2d(np.zeros((5, 5)), 1.0, 1.0, 1.0, 0.01, 11, 11)
    with pytest.raises(ValueError, match="boundary must be"):
        heat_equation_2d(flat, 1.0, 1.0, 1.0, 0.01, nx, ny, boundary="periodic")


def test_wave_2d_vectorised_shapes_cfl_and_purity() -> None:
    nx, ny, lx, ly, c = 21, 21, 1.0, 1.0, 1.0
    u0 = np.zeros((ny, nx), dtype=np.float64)
    v0 = np.zeros((ny, nx), dtype=np.float64)
    # add a sine bump in interior
    xs = np.linspace(0.0, lx, nx, dtype=np.float64)
    ys = np.linspace(0.0, ly, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    u0_sine = np.sin(math.pi * xx) * np.sin(math.pi * yy)
    res = wave_equation_2d(u0_sine, v0, c, lx, ly, 0.05, nx, ny)
    assert isinstance(res, Wave2DResult)
    assert res.u_final.shape == (ny, nx)
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    assert res.dx == pytest.approx(dx)
    assert res.dy == pytest.approx(dy)
    auto_dt = 0.9 / (c * math.sqrt(1.0 / dx**2 + 1.0 / dy**2))
    assert res.dt == pytest.approx(auto_dt)
    # Dirichlet borders must stay at initial (zero) level
    assert np.allclose(res.u_final[0, :], 0.0)
    assert np.allclose(res.u_final[-1, :], 0.0)
    assert np.allclose(res.u_final[:, 0], 0.0)
    assert np.allclose(res.u_final[:, -1], 0.0)
    # CFL guard
    with pytest.raises(ValueError, match="violates wave CFL"):
        wave_equation_2d(u0, v0, c, lx, ly, 0.05, nx, ny, dt=1.0)
    with pytest.raises(ValueError, match="shape"):
        wave_equation_2d(np.zeros((5, 5)), v0, c, lx, ly, 0.05, 11, 11)
    # purity: inputs not mutated
    u0_copy = u0_sine.copy()
    v0_copy = v0.copy()
    wave_equation_2d(u0_sine, v0, c, lx, ly, 0.02, nx, ny)
    np.testing.assert_array_equal(u0_sine, u0_copy)
    np.testing.assert_array_equal(v0, v0_copy)
    # also check 1-D purity
    u1 = np.linspace(0.0, 1.0, 11)
    v1 = np.zeros(11)
    u1_snap = u1.copy()
    v1_snap = v1.copy()
    heat_equation_1d(u1, 1.0, 1.0, 0.01, 11)
    wave_equation_1d(u1, v1, 1.0, 1.0, 0.01, 11)
    np.testing.assert_array_equal(u1, u1_snap)
    np.testing.assert_array_equal(v1, v1_snap)
