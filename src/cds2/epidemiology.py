"""Compartmental epidemic models: SIR/SEIR trajectories and threshold quantities."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SIRResult",
    "SEIRResult",
    "simulate_sir",
    "simulate_seir",
    "herd_immunity_threshold",
    "effective_reproduction",
    "final_size_iteration",
]

FloatArray = NDArray[np.float64]

Derivative = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class SIRResult:
    """Daily SIR trajectory produced by :func:`simulate_sir`."""

    times: FloatArray
    susceptible: FloatArray
    infected: FloatArray
    recovered: FloatArray
    population: float
    beta: float
    gamma: float

    @property
    def r0(self) -> float:
        """Basic reproduction number ``beta / gamma``."""
        return math.inf if self.gamma == 0.0 else self.beta / self.gamma

    @property
    def peak_day(self) -> float:
        """Day on which the infected curve attains its maximum."""
        return float(self.times[int(np.argmax(self.infected))])

    @property
    def attack_rate(self) -> float:
        """Fraction of the population ultimately recovered."""
        return float(self.recovered[-1] / self.population)


@dataclass(frozen=True)
class SEIRResult:
    """Daily SEIR trajectory produced by :func:`simulate_seir`."""

    times: FloatArray
    susceptible: FloatArray
    exposed: FloatArray
    infected: FloatArray
    recovered: FloatArray
    population: float
    beta: float
    sigma: float
    gamma: float

    @property
    def r0(self) -> float:
        """Basic reproduction number ``beta / gamma``."""
        return math.inf if self.gamma == 0.0 else self.beta / self.gamma

    @property
    def peak_day(self) -> float:
        """Day on which the infected curve attains its maximum."""
        return float(self.times[int(np.argmax(self.infected))])

    @property
    def attack_rate(self) -> float:
        """Fraction of the population ultimately recovered."""
        return float(self.recovered[-1] / self.population)


def _validate_common(
    population: float, days: int, steps_per_day: int, beta: float, gamma: float
) -> None:
    """Shared parameter validation for the compartmental simulators."""
    if not np.isfinite(population) or population <= 0.0:
        raise ValueError("population must be positive and finite")
    if not isinstance(days, (int, np.integer)) or isinstance(days, bool) or days < 1:
        raise ValueError("days must be at least 1 and an integer")
    if (
        not isinstance(steps_per_day, (int, np.integer))
        or isinstance(steps_per_day, bool)
        or steps_per_day < 1
    ):
        raise ValueError("steps_per_day must be at least 1 and an integer")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be non-negative and finite")


def _rk4_integrate(state: FloatArray, deriv: Derivative, dt: float, steps: int) -> FloatArray:
    """Advance ``state`` by ``steps`` classical RK4 steps of size ``dt``."""
    for _ in range(steps):
        k1 = deriv(state)
        k2 = deriv(state + (dt / 2.0) * k1)
        k3 = deriv(state + (dt / 2.0) * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def simulate_sir(
    population: float,
    beta: float,
    gamma: float,
    days: int,
    i0: float = 1.0,
    steps_per_day: int = 10,
) -> SIRResult:
    """Integrate the SIR model with RK4 and sample the state once per day.

    The system is ``dS/dt = -beta*S*I/N``, ``dI/dt = beta*S*I/N - gamma*I``
    and ``dR/dt = gamma*I``; ``times`` runs from ``0`` to ``days`` inclusive.
    """
    _validate_common(population, days, steps_per_day, beta, gamma)
    if not np.isfinite(i0) or not 0.0 <= i0 <= population:
        msg = "initial infections must lie within the population"
        raise ValueError(msg)

    def deriv(y: FloatArray) -> FloatArray:
        transmission = beta * y[0] * y[1] / population
        return np.array(
            [-transmission, transmission - gamma * y[1], gamma * y[1]],
            dtype=np.float64,
        )

    times = np.linspace(0.0, float(days), days + 1)
    susceptible = np.empty(days + 1, dtype=np.float64)
    infected = np.empty(days + 1, dtype=np.float64)
    recovered = np.empty(days + 1, dtype=np.float64)
    state: FloatArray = np.array([population - i0, i0, 0.0], dtype=np.float64)
    susceptible[0] = state[0]
    infected[0] = state[1]
    recovered[0] = state[2]

    for day in range(1, days + 1):
        state = _rk4_integrate(state, deriv, 1.0 / steps_per_day, steps_per_day)
        susceptible[day] = state[0]
        infected[day] = state[1]
        recovered[day] = state[2]

    return SIRResult(
        times=times,
        susceptible=susceptible,
        infected=infected,
        recovered=recovered,
        population=population,
        beta=beta,
        gamma=gamma,
    )


def simulate_seir(
    population: float,
    beta: float,
    sigma: float,
    gamma: float,
    days: int,
    i0: float = 1.0,
    e0: float = 0.0,
    steps_per_day: int = 10,
) -> SEIRResult:
    """Integrate the SEIR model with RK4 and sample the state once per day.

    The system is ``dS/dt = -beta*S*I/N``, ``dE/dt = beta*S*I/N - sigma*E``,
    ``dI/dt = sigma*E - gamma*I`` and ``dR/dt = gamma*I``; ``times`` runs from
    ``0`` to ``days`` inclusive.
    """
    _validate_common(population, days, steps_per_day, beta, gamma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        msg = "sigma must be positive"
        raise ValueError(msg)
    if not np.isfinite(i0) or not 0.0 <= i0 <= population:
        msg = "initial infections must lie within the population"
        raise ValueError(msg)
    if not np.isfinite(e0) or e0 < 0.0:
        msg = "initial exposures must be non-negative"
        raise ValueError(msg)
    if e0 + i0 > population:
        msg = "initial exposed and infected must fit within the population"
        raise ValueError(msg)

    def deriv(y: FloatArray) -> FloatArray:
        transmission = beta * y[0] * y[2] / population
        onset = sigma * y[1]
        return np.array(
            [
                -transmission,
                transmission - onset,
                onset - gamma * y[2],
                gamma * y[2],
            ],
            dtype=np.float64,
        )

    times = np.linspace(0.0, float(days), days + 1)
    susceptible = np.empty(days + 1, dtype=np.float64)
    exposed = np.empty(days + 1, dtype=np.float64)
    infected = np.empty(days + 1, dtype=np.float64)
    recovered = np.empty(days + 1, dtype=np.float64)
    state: FloatArray = np.array([population - i0 - e0, e0, i0, 0.0], dtype=np.float64)
    susceptible[0] = state[0]
    exposed[0] = state[1]
    infected[0] = state[2]
    recovered[0] = state[3]

    for day in range(1, days + 1):
        state = _rk4_integrate(state, deriv, 1.0 / steps_per_day, steps_per_day)
        susceptible[day] = state[0]
        exposed[day] = state[1]
        infected[day] = state[2]
        recovered[day] = state[3]

    return SEIRResult(
        times=times,
        susceptible=susceptible,
        exposed=exposed,
        infected=infected,
        recovered=recovered,
        population=population,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
    )


def herd_immunity_threshold(r0: float) -> float:
    """Critical immune fraction; zero when transmission is already subcritical."""
    if not np.isfinite(r0) or r0 <= 0.0:
        raise ValueError("r0 must be positive and finite")
    if r0 <= 1.0:
        return 0.0
    return float(1.0 - 1.0 / r0)


def effective_reproduction(r0_value: float, susceptible_fraction: float) -> float:
    """Effective reproduction number ``r0 * susceptible_fraction``."""
    if not np.isfinite(r0_value) or r0_value < 0.0:
        raise ValueError("r0_value must be non-negative and finite")
    if not np.isfinite(susceptible_fraction) or not 0.0 <= susceptible_fraction <= 1.0:
        raise ValueError("susceptible_fraction must be in [0, 1]")
    return float(r0_value * susceptible_fraction)


def final_size_iteration(r0: float, tol: float = 1e-10, max_iter: int = 200) -> float:
    """Final epidemic size solving ``z = 1 - exp(-r0*z)`` for the nonzero root."""
    if not np.isfinite(r0) or r0 < 0.0:
        raise ValueError("r0 must be non-negative and finite")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be positive and finite")
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if r0 <= 1.0:
        return 0.0

    def objective(z: float) -> float:
        return z - (1.0 - math.exp(-r0 * z))

    epsilon = min(1e-8, 0.1 * (r0 - 1.0) / r0)
    try:
        return float(
            __import__("scipy").optimize.brentq(
                objective,
                epsilon,
                1.0 - np.finfo(float).eps,
                xtol=tol,
                rtol=max(4.0 * np.finfo(float).eps, tol),
                maxiter=max_iter,
            )
        )
    except RuntimeError as exc:
        raise RuntimeError("final-size iteration did not converge") from exc
