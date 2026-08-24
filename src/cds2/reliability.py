"""Survival and reliability analysis: Kaplan-Meier curves, Weibull fits, MTBF/availability."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
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
    """Kaplan-Meier product-limit estimate; events use 1 for failure, 0 for censored."""
    durations_array = np.asarray(durations, dtype=float)
    events_array = np.asarray(events, dtype=float)
    if (
        durations_array.ndim != 1
        or events_array.ndim != 1
        or durations_array.size == 0
        or durations_array.size != events_array.size
    ):
        msg = "durations and events must be equal-length non-empty series"
        raise ValueError(msg)
    if np.any(durations_array < 0.0):
        msg = "durations must be non-negative"
        raise ValueError(msg)
    if not np.any(events_array != 0.0):
        msg = "at least one event is required"
        raise ValueError(msg)
    failure_mask = events_array != 0.0
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
    """Maximum-likelihood Weibull fit over failure times only (censored rows masked out)."""
    data = np.asarray(durations, dtype=float)
    if failures_mask is not None:
        data = data[np.asarray(failures_mask, dtype=bool)]
    failures_only = data[data > 0.0]
    if failures_only.size < 2:
        msg = "at least two positive durations are required"
        raise ValueError(msg)
    shape, _loc, scale = sp_stats.weibull_min.fit(failures_only, floc=0.0)
    return WeibullFit(shape=float(shape), scale=float(scale))


def mtbf(total_operating_time: float, failures: int) -> float:
    """Mean time between failures from cumulative operating time and failure count."""
    if total_operating_time <= 0.0:
        msg = "total_operating_time must be positive"
        raise ValueError(msg)
    if failures < 1:
        msg = "at least one failure is required"
        raise ValueError(msg)
    return total_operating_time / failures


def availability(mtbf_value: float, mttr: float) -> float:
    """Steady-state availability mtbf / (mtbf + mttr); requires mttr < mtbf."""
    if mtbf_value <= 0.0 or mttr <= 0.0:
        msg = "mtbf and mttr must be positive"
        raise ValueError(msg)
    if mttr >= mtbf_value:
        msg = "mttr must be smaller than mtbf"
        raise ValueError(msg)
    return mtbf_value / (mtbf_value + mttr)


def weibull_survival(
    time_values: Sequence[float] | FloatArray,
    shape: float,
    scale: float,
) -> FloatArray:
    """Weibull reliability function exp(-(t/scale)**shape) evaluated at each time."""
    if shape <= 0.0:
        msg = "shape must be positive"
        raise ValueError(msg)
    if scale <= 0.0:
        msg = "scale must be positive"
        raise ValueError(msg)
    times = np.asarray(time_values, dtype=float)
    if np.any(times < 0.0):
        msg = "time values must be non-negative"
        raise ValueError(msg)
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
    rates = (early_rate, intrinsic_rate, wearout_rate)
    if min(rates) < 0.0:
        msg = "rates must be non-negative"
        raise ValueError(msg)
    if knee_early <= 0.0 or knee_wearout <= 0.0:
        msg = "knees must be positive"
        raise ValueError(msg)
    times = np.asarray(time_values, dtype=float)
    hazard = (
        intrinsic_rate
        + early_rate * np.exp(-times / knee_early)
        + wearout_rate * np.exp((times - knee_wearout) / knee_wearout)
    )
    return np.asarray(hazard, dtype=float)
