"""Stochastic differential equation ensemble solvers.

SciPy integrates ODEs (``solve_ivp``) and quadratures (``quad``) but has no
stochastic integrator: a ``dW`` term is not a parameter flip on a
deterministic solver, because Ito calculus changes what convergence even
means (strong vs weak order, and the Ito-Stratonovich choice). Users
otherwise reach for unmaintained packages or pull in a full autodiff stack.

This module solves diagonal-noise systems

    dy = drift(y, t) dt + diffusion(y, t) dW

with Euler-Maruyama (strong order 1/2) and Milstein (strong order 1), both
as *ensembles*: every path is advanced together as one array operation, so
the cost of 10,000 trajectories is one vectorised step per time point rather
than 10,000 sequential integrations.

Drift and diffusion are called with a stacked state of shape
``(n_paths, dim)`` and must return the same shape. Anything written with
NumPy operators satisfies this automatically::

    def drift(y, t):      return 0.05 * y      # works for (dim,) and (n, dim)
    def diffusion(y, t):  return 0.20 * y

That single convention is what keeps this module honest — it neither probes
callables nor silently falls back to per-path loops, so a shape mistake
surfaces as an error instead of a slow path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SdeEnsemble",
    "sde_euler_maruyama",
    "sde_milstein",
    "ensemble_stats",
]

FloatArray = NDArray[np.float64]
#: Called as ``f(y, t)`` with ``y`` of shape ``(n_paths, dim)``; returns the same shape.
EnsembleFn = Callable[[FloatArray, float], FloatArray]


@dataclass(frozen=True)
class SdeEnsemble:
    """Ensemble of SDE trajectories.

    Attributes:
        t: Time grid, shape ``(n_steps + 1,)``.
        paths: Trajectories, shape ``(n_paths, n_steps + 1, dim)``.
        method: Solver name, ``"euler"`` or ``"milstein"``.
        dt: Achieved step size. May be slightly below the requested ``dt``
            because the grid is made to land exactly on ``t_span[1]``.
        n_paths: Number of trajectories.
        seed: Seed used, or ``None`` when the generator was unseeded.
    """

    t: FloatArray
    paths: FloatArray
    method: str
    dt: float
    n_paths: int
    seed: int | None

    @property
    def terminal(self) -> FloatArray:
        """Final state of every path, shape ``(n_paths, dim)``."""
        return np.asarray(self.paths[:, -1, :], dtype=np.float64)


def _time_grid(
    t_span: tuple[float, float] | Sequence[float], dt: float
) -> tuple[FloatArray, float]:
    """Build a uniform grid that ends exactly on ``t_span[1]``.

    The returned step is therefore ``span / ceil(span / dt) <= dt`` — never
    larger than requested, so an error bound stated in terms of ``dt`` still
    holds.
    """
    span_arr = np.asarray(t_span, dtype=np.float64)
    if span_arr.shape != (2,) or not np.all(np.isfinite(span_arr)):
        msg = "t_span must be two finite values (t0, t1)"
        raise ValueError(msg)
    t0, t1 = float(span_arr[0]), float(span_arr[1])
    if t1 <= t0:
        msg = f"t_span must satisfy t0 < t1 (got {t0} >= {t1})"
        raise ValueError(msg)
    if not np.isfinite(dt) or dt <= 0.0:
        msg = f"dt must be positive and finite (got {dt})"
        raise ValueError(msg)
    n_steps = max(1, int(np.ceil((t1 - t0) / float(dt))))
    grid = np.linspace(t0, t1, n_steps + 1, dtype=np.float64)
    return grid, float(grid[1] - grid[0])


def _solve(
    method: str,
    drift: EnsembleFn,
    diffusion: EnsembleFn,
    y0: object,
    t_span: tuple[float, float] | Sequence[float],
    dt: float,
    n_paths: int,
    seed: int | None,
    jacobian: EnsembleFn | None,
) -> SdeEnsemble:
    if not callable(drift) or not callable(diffusion):
        msg = "drift and diffusion must be callable"
        raise TypeError(msg)
    if jacobian is not None and not callable(jacobian):
        msg = "jacobian must be callable or None"
        raise TypeError(msg)

    state0 = np.asarray(y0, dtype=np.float64)
    if state0.ndim != 1 or state0.size == 0 or not np.all(np.isfinite(state0)):
        msg = "y0 must be a non-empty finite 1-D array"
        raise ValueError(msg)
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        msg = f"n_paths must be an integer >= 1 (got {n_paths!r})"
        raise ValueError(msg)

    grid, step = _time_grid(t_span, dt)
    n_steps = int(grid.size - 1)
    dim = int(state0.size)

    rng = np.random.default_rng(seed)
    # One Wiener increment per (path, step, dimension). Drawn up front so the
    # stream depends only on the seed and the grid, not on solver branching —
    # that is what makes Euler and Milstein comparable on identical noise.
    increments = rng.standard_normal((int(n_paths), n_steps, dim)) * np.sqrt(step)

    paths = np.empty((int(n_paths), n_steps + 1, dim), dtype=np.float64)
    paths[:, 0, :] = state0

    for k in range(n_steps):
        t_k = float(grid[k])
        y_k = paths[:, k, :]
        dw = increments[:, k, :]

        a = _call(drift, y_k, t_k, "drift")
        b = _call(diffusion, y_k, t_k, "diffusion")
        y_next = y_k + a * step + b * dw

        if method == "milstein":
            # Milstein adds 0.5 * b * db/dy * (dW^2 - dt). The correction has
            # zero mean (E[dW^2] = dt), so it does not move the weak solution;
            # it removes the leading strong-order error term, lifting strong
            # order from 1/2 to 1 for diagonal noise.
            prime = (
                _call(jacobian, y_k, t_k, "jacobian")
                if jacobian is not None
                else _central_difference(diffusion, y_k, t_k)
            )
            y_next += 0.5 * b * prime * (dw * dw - step)

        if not np.all(np.isfinite(y_next)):
            msg = (
                f"non-finite state at t={float(grid[k + 1])!r} "
                f"(step {k + 1} of {n_steps}); drift or diffusion diverged"
            )
            raise FloatingPointError(msg)
        paths[:, k + 1, :] = y_next

    return SdeEnsemble(
        t=grid,
        paths=paths,
        method=method,
        dt=step,
        n_paths=int(n_paths),
        seed=seed,
    )


def _call(func: EnsembleFn, y: FloatArray, t: float, name: str) -> FloatArray:
    """Evaluate an ensemble callable and insist on the ensemble shape."""
    out = np.asarray(func(y, t), dtype=np.float64)
    if out.shape != y.shape:
        msg = (
            f"{name}(y, t) returned shape {out.shape}, expected {y.shape}; "
            "write it with NumPy operators so it broadcasts over the ensemble"
        )
        raise ValueError(msg)
    return out


def _central_difference(diffusion: EnsembleFn, y: FloatArray, t: float) -> FloatArray:
    """Diagonal of ``d diffusion / d y`` by central differences.

    Only the diagonal is needed: with diagonal noise, component ``i`` of the
    Milstein correction involves ``d b_i / d y_i`` alone. Each perturbation is
    applied to one component across the whole ensemble at once, so this costs
    ``2 * dim`` vectorised evaluations rather than ``2 * dim * n_paths``.

    The step is scaled to the magnitude of the state — a fixed absolute step
    loses all precision once ``|y|`` is large.
    """
    dim = y.shape[1]
    prime = np.empty_like(y)
    for j in range(dim):
        h = 1e-7 * np.maximum(1.0, np.abs(y[:, j]))
        offset = np.zeros_like(y)
        offset[:, j] = h
        forward = _call(diffusion, y + offset, t, "diffusion")
        backward = _call(diffusion, y - offset, t, "diffusion")
        prime[:, j] = (forward[:, j] - backward[:, j]) / (2.0 * h)
    return prime


def sde_euler_maruyama(
    drift: EnsembleFn,
    diffusion: EnsembleFn,
    y0: object,
    t_span: tuple[float, float] | Sequence[float],
    dt: float,
    n_paths: int = 1024,
    seed: int | None = None,
) -> SdeEnsemble:
    """Euler-Maruyama ensemble for ``dy = drift dt + diffusion dW``.

    Strong order 1/2, weak order 1. Each dimension is driven by its own
    independent Wiener process (diagonal noise).

    Args:
        drift: ``drift(y, t)`` with ``y`` of shape ``(n_paths, dim)``,
            returning the same shape.
        diffusion: ``diffusion(y, t)``, same convention.
        y0: Initial state, shape ``(dim,)``, shared by every path.
        t_span: ``(t0, t1)`` with ``t0 < t1``.
        dt: Target step size. The achieved step is
            ``(t1 - t0) / ceil((t1 - t0) / dt)``, never larger.
        n_paths: Number of trajectories to advance together.
        seed: Seed for the Wiener increments; ``None`` for a fresh stream.

    Returns:
        The :class:`SdeEnsemble`.

    Raises:
        TypeError: If ``drift`` or ``diffusion`` is not callable.
        ValueError: If ``y0``, ``t_span``, ``dt`` or ``n_paths`` is invalid,
            or a callable returns the wrong shape.
        FloatingPointError: If the state becomes non-finite.

    Example:
        Geometric Brownian motion, where ``E[y_T] = y_0 exp(mu T)`` is known
        in closed form and so serves as a check on the solver::

            >>> import numpy as np
            >>> ens = sde_euler_maruyama(
            ...     lambda y, t: 0.05 * y,
            ...     lambda y, t: 0.20 * y,
            ...     y0=[100.0], t_span=(0.0, 1.0), dt=1e-3,
            ...     n_paths=20000, seed=7,
            ... )
            >>> bool(abs(ens.terminal.mean() - 100.0 * np.exp(0.05)) < 1.0)
            True
    """
    return _solve("euler", drift, diffusion, y0, t_span, dt, n_paths, seed, None)


def sde_milstein(
    drift: EnsembleFn,
    diffusion: EnsembleFn,
    y0: object,
    t_span: tuple[float, float] | Sequence[float],
    dt: float,
    n_paths: int = 1024,
    seed: int | None = None,
    jacobian: EnsembleFn | None = None,
) -> SdeEnsemble:
    """Milstein ensemble: Euler-Maruyama plus the ``0.5 b b' (dW^2 - dt)`` term.

    Strong order 1 for diagonal noise, against 1/2 for Euler-Maruyama. The
    gain is in *pathwise* accuracy; both schemes have weak order 1, so an
    expectation like a European option price converges at the same rate while
    a path-dependent quantity (barrier crossing, maximum drawdown) does not.

    Args:
        drift: ``drift(y, t)``, ensemble convention as in
            :func:`sde_euler_maruyama`.
        diffusion: ``diffusion(y, t)``, same convention.
        y0: Initial state, shape ``(dim,)``.
        t_span: ``(t0, t1)`` with ``t0 < t1``.
        dt: Target step size.
        n_paths: Number of trajectories.
        seed: Seed for the Wiener increments.
        jacobian: Diagonal of ``d diffusion / d y``, same call convention.
            When ``None`` it is approximated by central differences, which
            costs ``2 * dim`` extra evaluations per step and limits accuracy
            to about ``1e-7`` relative — pass it explicitly when known.

    Returns:
        The :class:`SdeEnsemble`.

    Raises:
        TypeError: If ``drift``, ``diffusion`` or ``jacobian`` is not callable.
        ValueError: On invalid arguments or a wrong return shape.
        FloatingPointError: If the state becomes non-finite.

    Example:
        For ``b(y) = sigma * y`` the derivative is just ``sigma``, so the
        analytic Jacobian is available and the finite-difference step can be
        skipped::

            >>> ens = sde_milstein(
            ...     lambda y, t: 0.05 * y,
            ...     lambda y, t: 0.20 * y,
            ...     y0=[100.0], t_span=(0.0, 1.0), dt=1e-3,
            ...     n_paths=4096, seed=7,
            ...     jacobian=lambda y, t: 0.20 * np.ones_like(y),
            ... )
            >>> ens.method
            'milstein'
    """
    return _solve("milstein", drift, diffusion, y0, t_span, dt, n_paths, seed, jacobian)


def ensemble_stats(
    ensemble: SdeEnsemble,
    quantiles: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Summarise an ensemble across paths at every time point.

    Args:
        ensemble: Result of :func:`sde_euler_maruyama` or :func:`sde_milstein`.
        quantiles: Levels in the open interval ``(0, 1)``. Defaults to
            ``(0.05, 0.5, 0.95)``.

    Returns:
        Mapping with ``"t"``, ``"mean"``, ``"std"`` and ``"quantiles"``. Each
        statistic has shape ``(n_steps + 1, dim)``, or ``(n_steps + 1,)`` when
        ``dim == 1`` so that a scalar SDE plots without squeezing.

    Raises:
        TypeError: If ``ensemble`` is not an :class:`SdeEnsemble`.
        ValueError: If any quantile lies outside ``(0, 1)``.

    Example:
        >>> ens = sde_euler_maruyama(
        ...     lambda y, t: -y, lambda y, t: 0.1 * np.ones_like(y),
        ...     y0=[1.0], t_span=(0.0, 1.0), dt=1e-2, n_paths=512, seed=1,
        ... )
        >>> stats = ensemble_stats(ens, quantiles=[0.5])
        >>> stats["mean"].shape == ens.t.shape
        True
    """
    if not isinstance(ensemble, SdeEnsemble):
        msg = f"ensemble must be an SdeEnsemble (got {type(ensemble).__name__})"
        raise TypeError(msg)

    levels = (0.05, 0.5, 0.95) if quantiles is None else tuple(float(q) for q in quantiles)
    for q in levels:
        if not 0.0 < q < 1.0:
            msg = f"quantiles must lie in the open interval (0, 1) (got {q})"
            raise ValueError(msg)

    paths = ensemble.paths
    scalar = paths.shape[2] == 1

    def _shape(arr: FloatArray) -> FloatArray:
        return np.asarray(arr[:, 0] if scalar else arr, dtype=np.float64)

    return {
        "t": ensemble.t,
        "mean": _shape(paths.mean(axis=0)),
        "std": _shape(paths.std(axis=0, ddof=0)),
        "quantiles": {q: _shape(np.quantile(paths, q, axis=0)) for q in levels},
    }
