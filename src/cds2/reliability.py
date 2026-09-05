"""Survival and reliability analysis: Kaplan-Meier curves, Weibull fits, MTBF/availability."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as sp_optimize
from scipy import stats as sp_stats

__all__ = [
    "KMResult",
    "WeibullFit",
    "availability",
    "bathtub_curve",
    "kaplan_meier",
    "mtbf",
    "weibull_fit",
    "weibull_survival",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class KMResult:
    """Product-limit survival estimate evaluated at each unique failure time."""

    times: FloatArray
    survival: FloatArray

    @property
    def median(self) -> float | None:
        """First time at which survival drops to 0.5 or below; None if never reached."""
        hits = np.flatnonzero(self.survival <= 0.5)
        if hits.size == 0:
            return None
        return float(self.times[int(hits[0])])


@dataclass(frozen=True)
class WeibullFit:
    """Fitted two-parameter Weibull distribution (location fixed at zero)."""

    shape: float
    scale: float


def kaplan_meier(
    durations: Sequence[float] | FloatArray,
    events: Sequence[float] | FloatArray,
) -> KMResult:
    """Kaplan-Meier product-limit estimate; events must be exactly 0 or 1."""
    durations_array = np.asarray(durations, dtype=float)
    events_array = np.asarray(events, dtype=float)
    if (
        durations_array.ndim != 1
        or events_array.ndim != 1
        or durations_array.size == 0
        or durations_array.size != events_array.size
    ):
        raise ValueError("durations and events must be equal-length non-empty series")
    if not bool(np.all(np.isfinite(durations_array))) or not bool(
        np.all(np.isfinite(events_array))
    ):
        raise ValueError("durations and events must be finite")
    if np.any(durations_array < 0.0):
        raise ValueError("durations must be non-negative")
    if not bool(np.all((events_array == 0.0) | (events_array == 1.0))):
        raise ValueError("events must contain only 0 or 1")
    if not np.any(events_array == 1.0):
        raise ValueError("at least one event is required")
    failure_mask = events_array == 1.0
    times = np.unique(durations_array[failure_mask])
    survival = 1.0
    estimates: list[float] = []
    for time in times.tolist():
        at_risk = int(np.count_nonzero(durations_array >= time))
        deaths = int(np.count_nonzero((durations_array == time) & failure_mask))
        survival *= 1.0 - deaths / at_risk
        estimates.append(survival)
    return KMResult(
        times=np.asarray(times, dtype=float), survival=np.asarray(estimates, dtype=float)
    )


def weibull_fit(
    durations: Sequence[float] | FloatArray,
    failures_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
) -> WeibullFit:
    """Maximum-likelihood two-parameter Weibull fit with optional right censoring."""
    data = np.asarray(durations, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("at least two positive durations are required")
    if not bool(np.all(np.isfinite(data))) or np.any(data < 0.0):
        raise ValueError("durations must be finite and non-negative")

    if failures_mask is None:
        failures_only = data[data > 0.0]
        if failures_only.size < 2:
            raise ValueError("at least two positive durations are required")
        shape, _loc, scale = sp_stats.weibull_min.fit(failures_only, floc=0.0)
        return WeibullFit(shape=float(shape), scale=float(scale))

    mask = np.asarray(failures_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape != data.shape:
        raise ValueError("failures_mask must match durations shape")
    failure_times = data[mask]
    if failure_times.size < 2 or np.any(failure_times <= 0.0):
        raise ValueError("at least two positive failure durations are required")

    initial_shape, _loc, initial_scale = sp_stats.weibull_min.fit(failure_times, floc=0.0)

    def negative_log_likelihood(log_params: NDArray[np.float64]) -> float:
        shape = float(np.exp(log_params[0]))
        scale = float(np.exp(log_params[1]))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cumulative_hazard = np.power(data / scale, shape)
            failure_log_density = (
                np.log(shape) + (shape - 1.0) * np.log(failure_times) - shape * np.log(scale)
            )
            value = float(np.sum(cumulative_hazard) - np.sum(failure_log_density))
        return value if np.isfinite(value) else float("inf")

    result = sp_optimize.minimize(
        negative_log_likelihood,
        np.log(np.array([initial_shape, initial_scale], dtype=np.float64)),
        method="L-BFGS-B",
    )
    if not result.success or not bool(np.all(np.isfinite(result.x))):
        raise RuntimeError(f"censored Weibull fit failed: {result.message}")
    shape, scale = np.exp(result.x)
    return WeibullFit(shape=float(shape), scale=float(scale))


def mtbf(total_operating_time: float, failures: int) -> float:
    """Mean time between failures from cumulative operating time and failure count."""
    if not np.isfinite(total_operating_time) or total_operating_time <= 0.0:
        raise ValueError("total_operating_time must be positive and finite")
    if not isinstance(failures, (int, np.integer)) or isinstance(failures, bool) or failures < 1:
        raise ValueError("at least one failure is required and failures must be an integer")
    return float(total_operating_time / failures)


def availability(mtbf_value: float, mttr: float) -> float:
    """Steady-state availability ``MTBF / (MTBF + MTTR)``."""
    if not np.isfinite(mtbf_value) or mtbf_value <= 0.0:
        raise ValueError("mtbf must be positive and finite")
    if not np.isfinite(mttr) or mttr < 0.0:
        raise ValueError("mttr must be non-negative and finite")
    return float(mtbf_value / (mtbf_value + mttr))


def weibull_survival(
    time_values: Sequence[float] | FloatArray,
    shape: float,
    scale: float,
) -> FloatArray:
    """Weibull reliability function exp(-(t/scale)**shape) evaluated at each time."""
    if not np.isfinite(shape) or shape <= 0.0:
        raise ValueError("shape must be positive and finite")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    times = np.asarray(time_values, dtype=float)
    if not bool(np.all(np.isfinite(times))) or np.any(times < 0.0):
        raise ValueError("time values must be finite and non-negative")
    ratio = np.asarray(times / scale, dtype=float)
    return np.asarray(np.exp(-np.power(ratio, shape)), dtype=float)


def bathtub_curve(
    time_values: Sequence[float] | FloatArray,
    early_rate: float,
    intrinsic_rate: float,
    wearout_rate: float,
    knee_early: float,
    knee_wearout: float,
) -> FloatArray:
    """Bathtub hazard combining early-life decay, intrinsic floor and wear-out growth."""
    parameters = np.asarray(
        [early_rate, intrinsic_rate, wearout_rate, knee_early, knee_wearout], dtype=float
    )
    if not bool(np.all(np.isfinite(parameters))):
        raise ValueError("rates and knees must be finite")
    if min(early_rate, intrinsic_rate, wearout_rate) < 0.0:
        raise ValueError("rates must be non-negative")
    if knee_early <= 0.0 or knee_wearout <= 0.0:
        raise ValueError("knees must be positive")
    times = np.asarray(time_values, dtype=float)
    if not bool(np.all(np.isfinite(times))) or np.any(times < 0.0):
        raise ValueError("time values must be finite and non-negative")
    hazard = (
        intrinsic_rate
        + early_rate * np.exp(-times / knee_early)
        + wearout_rate * np.exp((times - knee_wearout) / knee_wearout)
    )
    return np.asarray(hazard, dtype=float)
