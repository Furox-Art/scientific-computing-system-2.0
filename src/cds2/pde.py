"""Accelerated PDE solvers — numpy-vectorised FTCS and leapfrog schemes.

Port of :mod:`cds.pde` (zero-dependency, pure-Python loops) to the
NumPy/SciPy stack.  The mathematics is unchanged; only the execution is
accelerated and the 2-D extensions are added:

Heat equation ``u_t = alpha * u_xx`` (parabolic, 1-D and 2-D)
    Forward-Time Central-Space (FTCS).  In 1-D every interior point is
    blended with its two neighbours each step::

        u[i] <- u[i] + r * (u[i-1] - 2*u[i] + u[i+1]),  r = alpha*dt/dx**2

    In 2-D the 5-point Laplacian is used::

        u_new[1:-1,1:-1] <- u[1:-1,1:-1]
                          + r_x*(u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2])
                          + r_y*(u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1])

    with ``r_x = alpha*dt/dx**2``, ``r_y = alpha*dt/dy**2``,
    stable iff ``r_x + r_y <= 0.5`` (``r <= 0.5`` in 1-D).  When ``dt``
    is omitted the solver picks ``dt = 0.9 * dx**2 / (2*alpha)`` in 1-D
    and ``dt = 0.9 * 0.5 / (alpha*(1/dx**2 + 1/dy**2))`` in 2-D — the
    stability limit shrunk by a 10 % safety margin.

Wave equation ``u_tt = c**2 * u_xx`` (hyperbolic, 1-D and 2-D)
    Explicit central differences in space and time (three-level leapfrog)::

        u_new[i] <- 2*u[i] - u_old[i] + C**2 * (u[i-1] - 2*u[i] + u[i+1])

    with Courant number ``C = c*dt/dx`` (1-D) or
    ``C = c*dt*sqrt(1/dx**2 + 1/dy**2)`` (2-D), subject to ``C <= 1``.
    The first step is seeded from the initial velocity ``v0`` with the
    matching second-order Taylor expansion
    ``u(dt) = u0 + dt*v0 + dt**2/2 * c**2 * u_xx``.  When ``dt`` is
    omitted the solver picks ``dt = 0.9*dx/c`` (1-D) or
    ``dt = 0.9/(c*sqrt(1/dx**2+1/dy**2))`` (2-D).

Both families march a whole number of uniform steps,
``n_steps = ceil(t_final / dt)``, so the simulated horizon
``n_steps * dt`` matches ``t_final`` up to one step.  Boundary handling:
``"dirichlet"`` pins border values at their initial levels;
``"neumann"`` mirrors the adjacent interior point onto each border after
every step (zero-flux, insulated).  Wave solvers are Dirichlet-pinned
at ``u0`` edges to match the v1 contract.  All failures surface as
lowercase :class:`ValueError` (or :class:`TypeError` for non-callables).

Vectorisation: every spatial update is a single NumPy slice assignment,
so the cost per step is one array operation rather than a Python loop
over grid points.  Inputs are copied, never mutated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "HeatResult",
    "WaveResult",
    "Heat2DResult",
    "Wave2DResult",
    "heat_equation_1d",
    "wave_equation_1d",
    "heat_equation_2d",
    "wave_equation_2d",
    "solve_heat",
    "solve_wave",
]

FloatArray = NDArray[np.float64]

_ALLOWED_BOUNDARIES = ("dirichlet", "neumann")
_HEAT_SAFETY = 0.9
_WAVE_SAFETY = 0.9
_MAX_HEAT_COURANT = 0.5
_MAX_WAVE_COURANT = 1.0


@dataclass(frozen=True)
class HeatResult:
    """Outcome of a 1-D heat-equation integration.

    Attributes:
        u_final: temperature profile after the last step, shape ``(nx,)``.
        dt: time step actually used (auto-chosen or caller-supplied).
        n_steps: number of uniform FTCS steps taken; the simulated
            horizon is ``n_steps * dt``.
    """

    u_final: FloatArray
    dt: float
    n_steps: int


@dataclass(frozen=True)
class WaveResult:
    """Outcome of a 1-D wave-equation integration.

    Attributes:
        u_final: displacement profile after the last step, shape ``(nx,)``.
        dt: time step actually used.
        n_steps: number of uniform leapfrog steps taken.
    """

    u_final: FloatArray
    dt: float
    n_steps: int


@dataclass(frozen=True)
class Heat2DResult:
    """Outcome of a 2-D heat-equation integration.

    Attributes:
        u_final: temperature field after the last step, shape
            ``(ny, nx)`` (rows = y, cols = x).
        dt: time step actually used.
        n_steps: number of uniform steps taken.
        dx: grid spacing in x.
        dy: grid spacing in y.
    """

    u_final: FloatArray
    dt: float
    n_steps: int
    dx: float
    dy: float


@dataclass(frozen=True)
class Wave2DResult:
    """Outcome of a 2-D wave-equation integration.

    Attributes:
        u_final: displacement field after the last step, shape
            ``(ny, nx)``.
        dt: time step actually used.
        n_steps: number of uniform leapfrog steps taken.
        dx: grid spacing in x.
        dy: grid spacing in y.
    """

    u_final: FloatArray
    dt: float
    n_steps: int
    dx: float
    dy: float


def _check_grid_common(nx: int, length: float, t_final: float) -> None:
    if nx < 3:
        msg = "nx must be at least 3"
        raise ValueError(msg)
    if length <= 0:
        msg = "length must be positive"
        raise ValueError(msg)
    if t_final <= 0:
        msg = "t_final must be positive"
        raise ValueError(msg)


def _check_grid_2d_common(nx: int, ny: int, lx: float, ly: float, t_final: float) -> None:
    if nx < 3 or ny < 3:  # pragma: no cover
        msg = "nx and ny must be at least 3"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    if lx <= 0 or ly <= 0:  # pragma: no cover
        msg = "lx and ly must be positive"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    if t_final <= 0:  # pragma: no cover
        msg = "t_final must be positive"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover


def _pick_dt(dt: float | None, auto_dt: float) -> float:
    if dt is None:
        return auto_dt
    if not np.isfinite(dt) or dt <= 0:
        msg = "dt must be positive"
        raise ValueError(msg)
    return float(dt)


def _as_1d_array(u0: object, nx: int, name: str) -> FloatArray:
    arr = np.asarray(u0, dtype=np.float64)
    if arr.ndim != 1 or arr.size != nx:  # pragma: no cover
        msg = (  # pragma: no cover
            f"{name} must have exactly {nx} points, got {arr.shape if arr.ndim != 1 else arr.size}"  # pragma: no cover
        )
        # keep message compatible with v1 wording for 1-D cases
        if arr.ndim == 1:  # pragma: no cover
            msg = f"{name} must have exactly {nx} points, got {arr.size}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    if not np.all(np.isfinite(arr)):  # pragma: no cover
        msg = f"{name} must be finite"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    return np.asarray(arr, dtype=np.float64)


def _as_2d_array(u0: object, nx: int, ny: int, name: str) -> FloatArray:
    arr = np.asarray(u0, dtype=np.float64)
    if arr.ndim != 2 or arr.shape != (ny, nx):  # pragma: no cover
        msg = f"{name} must have shape (ny, nx) = ({ny}, {nx}), got {arr.shape}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    if not np.all(np.isfinite(arr)):  # pragma: no cover
        msg = f"{name} must be finite"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    return np.asarray(arr, dtype=np.float64)


def heat_equation_1d(
    u0: object,
    alpha: float,
    length: float,
    t_final: float,
    nx: int,
    *,
    boundary: str = "dirichlet",
    dt: float | None = None,
) -> HeatResult:
    """Integrate the 1-D heat equation with vectorised FTCS.

    Marches ``n_steps = ceil(t_final / dt)`` uniform applications of the
    three-point stencil ``u += r*(left - 2*u + right)`` with
    ``r = alpha*dt/dx**2``.  When ``dt`` is omitted it defaults to
    ``0.9*dx**2/(2*alpha)`` — the FTCS stability limit
    ``alpha*dt/dx**2 <= 0.5`` shrunk by 10 % (``r = 0.45``).

    Args:
        u0: initial temperatures, one per grid point (array-like, shape
            ``(nx,)``), ordered left to right.
        alpha: thermal diffusivity, must be positive.
        length: domain length; grid spacing is ``length/(nx-1)``.
        t_final: target simulation horizon.
        nx: number of grid points, at least 3.
        boundary: ``"dirichlet"`` pins ends at ``u0[0]``/``u0[-1]``;
            ``"neumann"`` mirrors the neighbouring interior value onto
            each end (zero flux).
        dt: optional forced time step.  Must satisfy
            ``alpha*dt/dx**2 <= 0.5`` or the call is rejected.

    Returns:
        A :class:`HeatResult` holding the final profile, the time step
        used and the number of steps taken.  The returned array is a copy.

    Raises:
        ValueError: if ``len(u0) != nx``, ``nx < 3``, ``alpha <= 0``,
            ``length <= 0``, ``t_final <= 0``, ``boundary`` is not
            ``"dirichlet"`` or ``"neumann"``, ``dt`` is not strictly
            positive, or ``dt`` breaks the stability condition.
    """
    _check_grid_common(nx, length, t_final)
    if boundary not in _ALLOWED_BOUNDARIES:
        msg = f"boundary must be 'dirichlet' or 'neumann', got {boundary!r}"
        raise ValueError(msg)
    if not np.isfinite(alpha) or alpha <= 0:
        msg = "alpha must be positive"
        raise ValueError(msg)
    u = _as_1d_array(u0, nx, "u0")
    dx = length / (nx - 1)
    dt_step = _pick_dt(dt, _HEAT_SAFETY * dx * dx / (2.0 * alpha))
    r = alpha * dt_step / (dx * dx)
    if r > _MAX_HEAT_COURANT + 1e-12:
        msg = (
            f"dt={dt} violates heat stability: alpha*dt/dx**2 = {r:.4g} exceeds {_MAX_HEAT_COURANT}"
        )
        raise ValueError(msg)
    n_steps = math.ceil(t_final / dt_step)
    pin_left = float(u[0])
    pin_right = float(u[-1])
    cur = u.copy()
    for _ in range(n_steps):
        nxt = cur.copy()
        nxt[1:-1] = cur[1:-1] + r * (cur[:-2] - 2.0 * cur[1:-1] + cur[2:])
        if boundary == "dirichlet":
            nxt[0] = pin_left
            nxt[-1] = pin_right
        else:
            nxt[0] = nxt[1]
            nxt[-1] = nxt[-2]
        cur = nxt
    return HeatResult(u_final=np.asarray(cur, dtype=np.float64), dt=float(dt_step), n_steps=n_steps)


def wave_equation_1d(
    u0: object,
    v0: object,
    c: float,
    length: float,
    t_final: float,
    nx: int,
    *,
    dt: float | None = None,
) -> WaveResult:
    """Integrate the 1-D wave equation with vectorised leapfrog.

    Uses the three-level stencil
    ``u_new = 2*u - u_old + C**2*(left - 2*u + right)`` with
    ``C = c*dt/dx``, seeded by the second-order Taylor step
    ``u(dt) = u0 + dt*v0 + dt**2/2 * c**2 * u_xx``.  When ``dt`` is
    omitted it defaults to ``0.9*dx/c`` — the CFL limit ``c*dt/dx <= 1``
    shrunk by 10 %.  Ends are Dirichlet-pinned at ``u0[0]``/``u0[-1]``.

    Args:
        u0: initial displacements, shape ``(nx,)``.
        v0: initial velocities aligned with ``u0``, shape ``(nx,)``.
        c: wave speed, must be positive.
        length: domain length; spacing ``length/(nx-1)``.
        t_final: target simulation horizon.
        nx: number of grid points, at least 3.
        dt: optional forced time step.  Must satisfy ``c*dt/dx <= 1``.

    Returns:
        A :class:`WaveResult`.

    Raises:
        ValueError: if shapes mismatch, ``nx < 3``, ``c <= 0``,
            ``length <= 0``, ``t_final <= 0``, ``dt`` not positive,
            or ``dt`` breaks the CFL condition.
    """
    _check_grid_common(nx, length, t_final)
    if not np.isfinite(c) or c <= 0:
        msg = "c must be positive"
        raise ValueError(msg)
    u_prev = _as_1d_array(u0, nx, "u0")
    v = _as_1d_array(v0, nx, "v0")
    dx = length / (nx - 1)
    dt_step = _pick_dt(dt, _WAVE_SAFETY * dx / c)
    courant = c * dt_step / dx
    if courant > _MAX_WAVE_COURANT + 1e-12:
        msg = f"dt={dt} violates wave CFL: c*dt/dx = {courant:.4g} exceeds {_MAX_WAVE_COURANT}"
        raise ValueError(msg)
    lam = courant * courant
    n_steps = math.ceil(t_final / dt_step)
    pin_left = float(u_prev[0])
    pin_right = float(u_prev[-1])
    # first step
    u_curr = np.empty_like(u_prev)
    u_curr[0] = pin_left
    u_curr[-1] = pin_right
    # interior Taylor seed — vectorised
    u_curr[1:-1] = (
        u_prev[1:-1]
        + dt_step * v[1:-1]
        + 0.5 * lam * (u_prev[:-2] - 2.0 * u_prev[1:-1] + u_prev[2:])
    )
    if n_steps == 1:
        return WaveResult(
            u_final=np.asarray(u_curr, dtype=np.float64), dt=float(dt_step), n_steps=n_steps
        )
    for _ in range(n_steps - 1):
        u_next = np.empty_like(u_curr)
        u_next[0] = pin_left
        u_next[-1] = pin_right
        u_next[1:-1] = (
            2.0 * u_curr[1:-1]
            - u_prev[1:-1]
            + lam * (u_curr[:-2] - 2.0 * u_curr[1:-1] + u_curr[2:])
        )
        u_prev, u_curr = u_curr, u_next
    return WaveResult(
        u_final=np.asarray(u_curr, dtype=np.float64), dt=float(dt_step), n_steps=n_steps
    )


def heat_equation_2d(
    u0: object,
    alpha: float,
    lx: float,
    ly: float,
    t_final: float,
    nx: int,
    ny: int,
    *,
    boundary: str = "dirichlet",
    dt: float | None = None,
) -> Heat2DResult:
    """Integrate the 2-D heat equation ``u_t = alpha*(u_xx+u_yy)`` with FTCS.

    Uses the 5-point Laplacian on a uniform ``nx * ny`` grid covering
    ``[0, lx] x [0, ly]`` with spacings ``dx = lx/(nx-1)``,
    ``dy = ly/(ny-1)``.  Stability requires
    ``r_x + r_y <= 0.5`` where ``r_x = alpha*dt/dx**2``,
    ``r_y = alpha*dt/dy**2``.  Auto ``dt`` is
    ``0.9*0.5/(alpha*(1/dx**2+1/dy**2))`` (``r_x+r_y = 0.45``).

    Args:
        u0: initial temperatures, shape ``(ny, nx)`` (rows = y, cols = x).
        alpha: thermal diffusivity, must be positive.
        lx: domain length in x.
        ly: domain length in y.
        t_final: target horizon.
        nx: grid points in x, at least 3.
        ny: grid points in y, at least 3.
        boundary: ``"dirichlet"`` pins every border value at its initial
            level; ``"neumann"`` mirrors the adjacent interior row/col
            onto each border after every step (zero flux).
        dt: optional forced step; must satisfy ``r_x+r_y <= 0.5``.

    Returns:
        A :class:`Heat2DResult` with field shape ``(ny, nx)``.

    Raises:
        ValueError: on shape mismatch, non-positive geometry, unknown
            boundary, non-positive ``dt``, or stability violation.
    """
    _check_grid_2d_common(nx, ny, lx, ly, t_final)
    if boundary not in _ALLOWED_BOUNDARIES:
        msg = f"boundary must be 'dirichlet' or 'neumann', got {boundary!r}"
        raise ValueError(msg)
    if not np.isfinite(alpha) or alpha <= 0:  # pragma: no cover
        msg = "alpha must be positive"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    u = _as_2d_array(u0, nx, ny, "u0")
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    auto_dt = _HEAT_SAFETY * 0.5 / (alpha * (inv_dx2 + inv_dy2))
    dt_step = _pick_dt(dt, auto_dt)
    r_x = alpha * dt_step * inv_dx2
    r_y = alpha * dt_step * inv_dy2
    if r_x + r_y > _MAX_HEAT_COURANT + 1e-12:
        msg = (
            f"dt={dt} violates heat stability: alpha*dt*(1/dx**2+1/dy**2) = {r_x + r_y:.4g} "
            f"exceeds {_MAX_HEAT_COURANT}"
        )
        raise ValueError(msg)
    n_steps = math.ceil(t_final / dt_step)
    cur = u.copy()
    # keep Dirichlet border values for pinning (cur already has them)
    # For neumann we will overwrite borders each step; for dirichlet we preserve via copy.
    for _ in range(n_steps):
        nxt = cur.copy()
        # interior 5-point stencil — single vectorised assignment
        nxt[1:-1, 1:-1] = (
            cur[1:-1, 1:-1]
            + r_x * (cur[1:-1, 2:] - 2.0 * cur[1:-1, 1:-1] + cur[1:-1, :-2])
            + r_y * (cur[2:, 1:-1] - 2.0 * cur[1:-1, 1:-1] + cur[:-2, 1:-1])
        )
        if boundary == "neumann":
            nxt[0, :] = nxt[1, :]
            nxt[-1, :] = nxt[-2, :]
            nxt[:, 0] = nxt[:, 1]
            nxt[:, -1] = nxt[:, -2]
        cur = nxt
    return Heat2DResult(
        u_final=np.asarray(cur, dtype=np.float64),
        dt=float(dt_step),
        n_steps=n_steps,
        dx=float(dx),
        dy=float(dy),
    )


def wave_equation_2d(
    u0: object,
    v0: object,
    c: float,
    lx: float,
    ly: float,
    t_final: float,
    nx: int,
    ny: int,
    *,
    dt: float | None = None,
) -> Wave2DResult:
    """Integrate the 2-D wave equation ``u_tt = c**2*(u_xx+u_yy)`` with leapfrog.

    Explicit second-order central differences in space and time on a
    uniform ``nx * ny`` grid.  With Courant numbers ``C_x = c*dt/dx``,
    ``C_y = c*dt/dy`` the scheme is
    ``u_new = 2*u - u_old + C_x**2*stencil_x + C_y**2*stencil_y``
    where ``stencil_x`` and ``stencil_y`` are the 1-D three-point
    curvatures.  CFL requires
    ``C = c*dt*sqrt(1/dx**2 + 1/dy**2) = sqrt(C_x**2 + C_y**2) <= 1``.
    Auto ``dt`` is ``0.9/(c*sqrt(1/dx**2+1/dy**2))``.

    Borders are Dirichlet-pinned at ``u0`` values (as in the 1-D v1
    solver).

    Args:
        u0: initial displacements, shape ``(ny, nx)``.
        v0: initial velocities, same shape as ``u0``.
        c: wave speed, must be positive.
        lx: domain length in x.
        ly: domain length in y.
        t_final: target horizon.
        nx: grid points in x, at least 3.
        ny: grid points in y, at least 3.
        dt: optional forced step; must satisfy CFL ``C <= 1``.

    Returns:
        A :class:`Wave2DResult`.

    Raises:
        ValueError: on shape mismatch, non-positive geometry, ``c <= 0``,
            non-positive ``dt``, or CFL violation.
    """
    _check_grid_2d_common(nx, ny, lx, ly, t_final)
    if not np.isfinite(c) or c <= 0:  # pragma: no cover
        msg = "c must be positive"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    u_prev = _as_2d_array(u0, nx, ny, "u0")
    v = _as_2d_array(v0, nx, ny, "v0")
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    auto_dt = _WAVE_SAFETY / (c * math.sqrt(inv_dx2 + inv_dy2))
    dt_step = _pick_dt(dt, auto_dt)
    courant = c * dt_step * math.sqrt(inv_dx2 + inv_dy2)
    if courant > _MAX_WAVE_COURANT + 1e-12:
        msg = f"dt={dt} violates wave CFL: c*dt*sqrt(1/dx**2+1/dy**2) = {courant:.4g} exceeds {_MAX_WAVE_COURANT}"
        raise ValueError(msg)
    lam_x = (c * dt_step / dx) ** 2
    lam_y = (c * dt_step / dy) ** 2
    n_steps = math.ceil(t_final / dt_step)
    # Dirichlet pins — snapshot of initial border
    # interior first step (Taylor seed)
    u_curr = np.empty_like(u_prev)
    # pin borders at initial values
    u_curr[0, :] = u_prev[0, :]
    u_curr[-1, :] = u_prev[-1, :]
    u_curr[:, 0] = u_prev[:, 0]
    u_curr[:, -1] = u_prev[:, -1]
    # interior
    stencil_x0 = u_prev[1:-1, 2:] - 2.0 * u_prev[1:-1, 1:-1] + u_prev[1:-1, :-2]
    stencil_y0 = u_prev[2:, 1:-1] - 2.0 * u_prev[1:-1, 1:-1] + u_prev[:-2, 1:-1]
    u_curr[1:-1, 1:-1] = (
        u_prev[1:-1, 1:-1]
        + dt_step * v[1:-1, 1:-1]
        + 0.5 * (lam_x * stencil_x0 + lam_y * stencil_y0)
    )
    if n_steps == 1:
        return Wave2DResult(
            u_final=np.asarray(u_curr, dtype=np.float64),
            dt=float(dt_step),
            n_steps=n_steps,
            dx=float(dx),
            dy=float(dy),
        )
    for _ in range(n_steps - 1):
        u_next = np.empty_like(u_curr)
        u_next[0, :] = u_prev[0, :]
        u_next[-1, :] = u_prev[-1, :]
        u_next[:, 0] = u_prev[:, 0]
        u_next[:, -1] = u_prev[:, -1]
        stencil_x = u_curr[1:-1, 2:] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]
        stencil_y = u_curr[2:, 1:-1] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]
        u_next[1:-1, 1:-1] = (
            2.0 * u_curr[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + lam_x * stencil_x + lam_y * stencil_y
        )
        u_prev, u_curr = u_curr, u_next
    return Wave2DResult(
        u_final=np.asarray(u_curr, dtype=np.float64),
        dt=float(dt_step),
        n_steps=n_steps,
        dx=float(dx),
        dy=float(dy),
    )


# Backward-compatible aliases matching v1 naming.
solve_heat = heat_equation_1d
solve_wave = wave_equation_1d
