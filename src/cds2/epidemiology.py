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
        return self.beta / self.gamma

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
        return self.beta / self.gamma

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
    if population <= 0.0:
        msg = "population must be positive"
        raise ValueError(msg)
    if days < 1:
        msg = "days must be at least 1"
        raise ValueError(msg)
    if steps_per_day < 1:
        msg = "steps_per_day must be at least 1"
        raise ValueError(msg)
    if beta <= 0.0:
        msg = "beta must be positive"
        raise ValueError(msg)
    if gamma < 0.0:
        msg = "gamma must be non-negative"
        raise ValueError(msg)


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
    if not 0.0 <= i0 <= population:
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
    if sigma <= 0.0:
        msg = "sigma must be positive"
        raise ValueError(msg)
    if not 0.0 <= i0 <= population:
        msg = "initial infections must lie within the population"
        raise ValueError(msg)
    if e0 < 0.0:
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
    """Critical immune fraction ``1 - 1/r0`` needed to stop expansion."""
    if r0 <= 0.0:
        msg = "r0 must be positive"
        raise ValueError(msg)
    return 1.0 - 1.0 / r0


def effective_reproduction(r0_value: float, susceptible_fraction: float) -> float:
    """Effective reproduction number ``r0 * susceptible_fraction``."""
    return r0_value * susceptible_fraction


def final_size_iteration(r0: float, tol: float = 1e-10, max_iter: int = 200) -> float:
    """Final outbreak size via the fixed point ``z = 1 - exp(-r0 * z)``.

    Starts from ``z = 0.5`` and iterates until successive updates differ by
    less than ``tol``; raises :class:`RuntimeError` after ``max_iter`` sweeps.
    """
    z = 0.5
    for _ in range(max_iter):
        z_next = 1.0 - math.exp(-r0 * z)
        if abs(z_next - z) < tol:
            return z_next
        z = z_next
    msg = "final-size iteration did not converge"
    raise RuntimeError(msg)
