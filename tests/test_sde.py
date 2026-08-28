"""Tests for :mod:`cds2.sde`.

The SDE solvers are checked against results that hold independently of the
implementation: closed-form moments of geometric Brownian motion, the
Ornstein-Uhlenbeck stationary variance, the measured strong convergence order,
and the fact that Milstein must reduce to Euler-Maruyama when the diffusion is
constant (its correction term carries ``db/dy = 0``).
"""

from __future__ import annotations

import numpy as np
import pytest

from cds2.sde import SdeEnsemble, ensemble_stats, sde_euler_maruyama, sde_milstein

MU = 0.05
SIGMA = 0.20
Y0 = 100.0


def gbm_drift(y: np.ndarray, t: float) -> np.ndarray:
    return MU * y


def gbm_diffusion(y: np.ndarray, t: float) -> np.ndarray:
    return SIGMA * y


def gbm_jacobian(y: np.ndarray, t: float) -> np.ndarray:
    return SIGMA * np.ones_like(y)


# ---------------------------------------------------------------------------
# Closed-form moments
# ---------------------------------------------------------------------------


def test_gbm_mean_matches_closed_form() -> None:
    """E[y_T] = y_0 exp(mu T) for geometric Brownian motion."""
    ens = sde_euler_maruyama(
        gbm_drift, gbm_diffusion, [Y0], (0.0, 1.0), 1e-3, n_paths=20_000, seed=7
    )
    expected = Y0 * np.exp(MU * 1.0)
    # Monte Carlo standard error of the mean is sigma_T / sqrt(n) ~ 0.15 here;
    # 1.0 is a ~7-sigma band, loose enough not to flake but tight enough that a
    # missing drift or mis-scaled dW term fails it.
    assert abs(float(ens.terminal.mean()) - expected) < 1.0


def test_gbm_variance_matches_closed_form() -> None:
    """Var[y_T] = y_0^2 exp(2 mu T) (exp(sigma^2 T) - 1)."""
    ens = sde_euler_maruyama(
        gbm_drift, gbm_diffusion, [Y0], (0.0, 1.0), 1e-3, n_paths=40_000, seed=11
    )
    expected = Y0**2 * np.exp(2 * MU) * (np.exp(SIGMA**2) - 1.0)
    assert float(ens.terminal.var()) == pytest.approx(expected, rel=0.06)


def test_ou_process_reaches_stationary_variance() -> None:
    """dy = -theta y dt + sigma dW has stationary variance sigma^2/(2 theta)."""
    theta, sigma = 2.0, 0.5
    ens = sde_euler_maruyama(
        lambda y, t: -theta * y,
        lambda y, t: sigma * np.ones_like(y),
        [0.0],
        (0.0, 8.0),
        1e-3,
        n_paths=20_000,
        seed=3,
    )
    assert float(ens.terminal.var()) == pytest.approx(sigma**2 / (2 * theta), rel=0.05)
    assert float(ens.terminal.mean()) == pytest.approx(0.0, abs=0.02)


def test_zero_diffusion_recovers_deterministic_solution() -> None:
    """With no noise the solver must reproduce exponential growth."""
    ens = sde_euler_maruyama(
        lambda y, t: MU * y,
        lambda y, t: np.zeros_like(y),
        [Y0],
        (0.0, 1.0),
        1e-4,
        n_paths=3,
        seed=0,
    )
    assert float(ens.terminal.mean()) == pytest.approx(Y0 * np.exp(MU), rel=1e-3)
    # every path identical: no noise means no spread
    assert float(ens.terminal.std()) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Convergence order — the reason Milstein exists
# ---------------------------------------------------------------------------


def test_euler_and_milstein_separate_by_the_expected_order() -> None:
    """The gap between the two schemes must shrink like ``sqrt(dt)``.

    At a fixed seed and a fixed ``dt`` both solvers consume the identical
    Wiener increments, so their difference is a genuine pathwise quantity — no
    reference solution is needed and no path-matching problem arises. That
    difference is dominated by the Euler-Maruyama error term, which is order
    ``1/2``, so halving ``dt`` should divide it by about ``sqrt(2) = 1.41``.

    A finer-grid reference would *not* be a valid comparison here: changing
    ``dt`` changes the shape of the increment array, so the same seed yields a
    different Brownian path rather than a refinement of the same one.
    """
    gaps = []
    for dt in (2e-2, 1e-2, 5e-3):
        shared = {
            "y0": [Y0],
            "t_span": (0.0, 1.0),
            "dt": dt,
            "n_paths": 2_000,
            "seed": 5,
        }
        euler = sde_euler_maruyama(gbm_drift, gbm_diffusion, **shared)
        milstein = sde_milstein(gbm_drift, gbm_diffusion, jacobian=gbm_jacobian, **shared)
        gaps.append(float(np.abs(euler.terminal - milstein.terminal).mean()))

    assert gaps[0] > gaps[1] > gaps[2], f"gap must shrink monotonically, got {gaps}"
    for coarse, fine in zip(gaps, gaps[1:]):
        # sqrt(2) ~ 1.41 for order 1/2. The band rejects order 0 (ratio ~ 1)
        # and order 1 (ratio ~ 2) while tolerating Monte Carlo noise.
        assert 1.2 < coarse / fine < 1.7


def test_milstein_equals_euler_for_additive_noise() -> None:
    """Constant diffusion has db/dy = 0, so the correction term vanishes."""
    kwargs = {"y0": [1.0], "t_span": (0.0, 1.0), "dt": 1e-2, "n_paths": 64, "seed": 9}
    euler = sde_euler_maruyama(lambda y, t: -y, lambda y, t: 0.3 * np.ones_like(y), **kwargs)
    milstein = sde_milstein(lambda y, t: -y, lambda y, t: 0.3 * np.ones_like(y), **kwargs)
    assert np.allclose(euler.paths, milstein.paths, atol=1e-12)


def test_analytic_and_numeric_jacobian_agree() -> None:
    """The finite-difference fallback must match the supplied derivative."""
    kwargs = {"y0": [Y0], "t_span": (0.0, 0.5), "dt": 1e-2, "n_paths": 128, "seed": 4}
    supplied = sde_milstein(gbm_drift, gbm_diffusion, jacobian=gbm_jacobian, **kwargs)
    numeric = sde_milstein(gbm_drift, gbm_diffusion, **kwargs)
    assert np.allclose(supplied.paths, numeric.paths, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Shape, grid and reproducibility contracts
# ---------------------------------------------------------------------------


def test_shapes_and_grid_land_on_endpoint() -> None:
    ens = sde_euler_maruyama(
        lambda y, t: np.zeros_like(y),
        lambda y, t: np.ones_like(y),
        [0.0, 1.0, 2.0],
        (0.0, 1.0),
        0.1,
        n_paths=7,
        seed=1,
    )
    assert ens.paths.shape == (7, 11, 3)
    assert ens.t.shape == (11,)
    assert ens.terminal.shape == (7, 3)
    assert float(ens.t[-1]) == pytest.approx(1.0)
    assert ens.dt == pytest.approx(0.1)


def test_step_is_never_larger_than_requested() -> None:
    """A span that does not divide evenly must shrink dt, not overshoot t1."""
    ens = sde_euler_maruyama(
        lambda y, t: np.zeros_like(y),
        lambda y, t: np.zeros_like(y),
        [0.0],
        (0.0, 1.0),
        0.3,
        n_paths=2,
        seed=0,
    )
    assert ens.dt <= 0.3
    assert ens.dt == pytest.approx(0.25)
    assert float(ens.t[-1]) == pytest.approx(1.0)


def test_initial_condition_is_shared_by_every_path() -> None:
    ens = sde_euler_maruyama(
        gbm_drift, gbm_diffusion, [Y0, 2 * Y0], (0.0, 0.1), 1e-2, n_paths=5, seed=2
    )
    assert np.allclose(ens.paths[:, 0, :], [Y0, 2 * Y0])


def test_same_seed_reproduces_and_different_seed_diverges() -> None:
    kwargs = {"y0": [Y0], "t_span": (0.0, 1.0), "dt": 1e-2, "n_paths": 32}
    a = sde_euler_maruyama(gbm_drift, gbm_diffusion, seed=42, **kwargs)
    b = sde_euler_maruyama(gbm_drift, gbm_diffusion, seed=42, **kwargs)
    c = sde_euler_maruyama(gbm_drift, gbm_diffusion, seed=43, **kwargs)
    assert np.array_equal(a.paths, b.paths)
    assert not np.array_equal(a.paths, c.paths)


def test_unseeded_runs_differ() -> None:
    kwargs = {"y0": [Y0], "t_span": (0.0, 1.0), "dt": 1e-2, "n_paths": 16, "seed": None}
    a = sde_euler_maruyama(gbm_drift, gbm_diffusion, **kwargs)
    b = sde_euler_maruyama(gbm_drift, gbm_diffusion, **kwargs)
    assert a.seed is None
    assert not np.array_equal(a.paths, b.paths)


def test_metadata_records_the_solver() -> None:
    euler = sde_euler_maruyama(gbm_drift, gbm_diffusion, [Y0], (0.0, 0.1), 1e-2, n_paths=4, seed=1)
    milstein = sde_milstein(gbm_drift, gbm_diffusion, [Y0], (0.0, 0.1), 1e-2, n_paths=4, seed=1)
    assert (euler.method, euler.n_paths, euler.seed) == ("euler", 4, 1)
    assert milstein.method == "milstein"


# ---------------------------------------------------------------------------
# ensemble_stats
# ---------------------------------------------------------------------------


def test_stats_squeeze_scalar_and_keep_vector_shape() -> None:
    scalar = sde_euler_maruyama(
        gbm_drift, gbm_diffusion, [Y0], (0.0, 0.5), 1e-2, n_paths=256, seed=1
    )
    vector = sde_euler_maruyama(
        gbm_drift, gbm_diffusion, [Y0, Y0], (0.0, 0.5), 1e-2, n_paths=256, seed=1
    )
    s_scalar = ensemble_stats(scalar)
    s_vector = ensemble_stats(vector)
    assert s_scalar["mean"].shape == scalar.t.shape
    assert s_vector["mean"].shape == (vector.t.size, 2)
    assert set(s_scalar["quantiles"]) == {0.05, 0.5, 0.95}


def test_stats_values_match_numpy_directly() -> None:
    ens = sde_euler_maruyama(gbm_drift, gbm_diffusion, [Y0], (0.0, 0.5), 1e-2, n_paths=512, seed=6)
    stats = ensemble_stats(ens, quantiles=[0.25, 0.75])
    assert np.allclose(stats["mean"], ens.paths.mean(axis=0)[:, 0])
    assert np.allclose(stats["std"], ens.paths.std(axis=0)[:, 0])
    assert np.allclose(stats["quantiles"][0.25], np.quantile(ens.paths, 0.25, axis=0)[:, 0])
    assert np.array_equal(stats["t"], ens.t)


def test_stats_start_from_a_degenerate_distribution() -> None:
    """At t=0 every path equals y0, so std is 0 and quantiles collapse."""
    ens = sde_euler_maruyama(gbm_drift, gbm_diffusion, [Y0], (0.0, 0.5), 1e-2, n_paths=128, seed=8)
    stats = ensemble_stats(ens, quantiles=[0.1, 0.9])
    assert float(stats["std"][0]) == pytest.approx(0.0)
    assert float(stats["quantiles"][0.1][0]) == pytest.approx(Y0)
    assert float(stats["quantiles"][0.9][0]) == pytest.approx(Y0)
    # spread grows with time
    assert float(stats["std"][-1]) > 0.0


# ---------------------------------------------------------------------------
# Rejected input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"y0": [[1.0, 2.0]]}, "1-D"),
        ({"y0": []}, "1-D"),
        ({"y0": [np.nan]}, "finite"),
        ({"t_span": (1.0, 1.0)}, "t0 < t1"),
        ({"t_span": (1.0, 0.0)}, "t0 < t1"),
        ({"t_span": (0.0, np.inf)}, "finite"),
        ({"t_span": (0.0,)}, "two finite values"),
        ({"dt": 0.0}, "positive and finite"),
        ({"dt": -0.1}, "positive and finite"),
        ({"dt": np.nan}, "positive and finite"),
        ({"n_paths": 0}, "integer >= 1"),
        ({"n_paths": 2.5}, "integer >= 1"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict[str, object], match: str) -> None:
    call = {"y0": [1.0], "t_span": (0.0, 1.0), "dt": 0.1, "n_paths": 4, "seed": 0}
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        sde_euler_maruyama(gbm_drift, gbm_diffusion, **call)  # type: ignore[arg-type]


def test_non_callable_drift_or_diffusion_is_rejected() -> None:
    with pytest.raises(TypeError, match="callable"):
        sde_euler_maruyama(1.0, gbm_diffusion, [1.0], (0.0, 1.0), 0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        sde_euler_maruyama(gbm_drift, "nope", [1.0], (0.0, 1.0), 0.1)  # type: ignore[arg-type]


def test_non_callable_jacobian_is_rejected() -> None:
    with pytest.raises(TypeError, match="jacobian must be callable"):
        sde_milstein(
            gbm_drift,
            gbm_diffusion,
            [1.0],
            (0.0, 1.0),
            0.1,
            jacobian=3.0,  # type: ignore[arg-type]
        )


def test_wrong_return_shape_is_reported_with_both_shapes() -> None:
    """A callable that does not broadcast must fail loudly, not silently."""
    with pytest.raises(ValueError, match=r"drift\(y, t\) returned shape"):
        sde_euler_maruyama(
            lambda y, t: np.zeros(3),
            gbm_diffusion,
            [1.0],
            (0.0, 1.0),
            0.1,
            n_paths=4,
        )
    with pytest.raises(ValueError, match=r"diffusion\(y, t\) returned shape"):
        sde_euler_maruyama(
            gbm_drift,
            lambda y, t: np.zeros((2, 2)),
            [1.0],
            (0.0, 1.0),
            0.1,
            n_paths=4,
        )


def test_wrong_jacobian_shape_is_reported() -> None:
    with pytest.raises(ValueError, match=r"jacobian\(y, t\) returned shape"):
        sde_milstein(
            gbm_drift,
            gbm_diffusion,
            [1.0],
            (0.0, 1.0),
            0.1,
            n_paths=4,
            jacobian=lambda y, t: np.zeros(9),
        )


def test_divergence_raises_with_the_offending_step() -> None:
    """Explosive drift must be reported, not returned as inf."""
    with pytest.raises(FloatingPointError, match="non-finite state"):
        sde_euler_maruyama(
            lambda y, t: 1e300 * y,
            lambda y, t: np.zeros_like(y),
            [1.0],
            (0.0, 1.0),
            0.1,
            n_paths=2,
            seed=0,
        )


def test_stats_rejects_a_non_ensemble() -> None:
    with pytest.raises(TypeError, match="SdeEnsemble"):
        ensemble_stats([1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_stats_rejects_out_of_range_quantiles(bad: float) -> None:
    ens = sde_euler_maruyama(gbm_drift, gbm_diffusion, [Y0], (0.0, 0.1), 1e-2, n_paths=8, seed=0)
    with pytest.raises(ValueError, match="open interval"):
        ensemble_stats(ens, quantiles=[bad])


def test_ensemble_is_frozen() -> None:
    ens = sde_euler_maruyama(gbm_drift, gbm_diffusion, [Y0], (0.0, 0.1), 1e-2, n_paths=4, seed=0)
    assert isinstance(ens, SdeEnsemble)
    with pytest.raises(AttributeError):
        ens.method = "other"  # type: ignore[misc]
