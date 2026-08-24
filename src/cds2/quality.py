"""Statistical process control: control charts and capability indices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ControlChartResult",
    "CapabilityResult",
    "xbar_chart",
    "ewma_chart",
    "cusum_chart",
    "p_chart",
    "process_capability",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# Shewhart constants indexed by subgroup size n.
D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}


@dataclass(frozen=True)
class ControlChartResult:
    """Control-chart statistics with out-of-control point indices."""

    statistic: FloatArray
    center_line: float
    upper_limit: float
    lower_limit: float
    violations: IntArray


def _as_series(values: Sequence[float] | FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        msg = "values must be a 1-D series of at least two points"
        raise ValueError(msg)
    return array


def _subgroups(
    samples: Sequence[Sequence[float]] | FloatArray, subgroup_size: int | None
) -> FloatArray:
    array = np.asarray(samples, dtype=float)
    if array.ndim == 2:
        if subgroup_size is not None:
            msg = "subgroup_size cannot be combined with 2-D input"
            raise ValueError(msg)
        if array.shape[0] < 2 or array.shape[1] not in A2:
            msg = "need at least two subgroups with sizes between 2 and 10"
            raise ValueError(msg)
        return array
    if array.ndim == 1:
        size = subgroup_size or 5
        if size not in A2:
            msg = "subgroup_size must lie between 2 and 10"
            raise ValueError(msg)
        usable = (array.size // size) * size
        if usable < size * 2:
            msg = "need at least two full subgroups"
            raise ValueError(msg)
        return array[:usable].reshape(-1, size)
    msg = "samples must be 1-D or 2-D"
    raise ValueError(msg)


def xbar_chart(
    samples: Sequence[Sequence[float]] | FloatArray, subgroup_size: int | None = None
) -> ControlChartResult:
    """Shewhart X-bar chart; limits come from the A2 * R-bar constants.

    1-D input is split into subgroups of ``subgroup_size`` consecutive
    observations; 2-D input uses rows as subgroups directly.
    """
    subgroups = _subgroups(samples, subgroup_size)
    width = subgroups.shape[1]
    means = subgroups.mean(axis=1)
    ranges = np.ptp(subgroups, axis=1)
    grand_mean = float(means.mean())
    spread = float(A2[width] * ranges.mean())

    violations = np.flatnonzero(
        (means > grand_mean + spread) | (means < grand_mean - spread)
    ).astype(np.int64)
    return ControlChartResult(
        statistic=np.asarray(means, dtype=float),
        center_line=grand_mean,
        upper_limit=grand_mean + spread,
        lower_limit=grand_mean - spread,
        violations=violations,
    )


def ewma_chart(
    values: Sequence[float] | FloatArray,
    lambda_smooth: float = 0.2,
    sigma: float | None = None,
    width: float = 3.0,
) -> ControlChartResult:
    """EWMA control chart with the standard time-varying limit formula."""
    series = _as_series(values)
    if not 0.0 < lambda_smooth <= 1.0:
        msg = "lambda_smooth must lie in (0, 1]"
        raise ValueError(msg)
    if sigma is None:
        differences = np.diff(series)
        sigma = float(
            differences.std(ddof=1) / np.sqrt(2.0 * lambda_smooth / (2.0 - lambda_smooth))
        )
    if sigma <= 0:
        msg = "sigma must be positive"
        raise ValueError(msg)

    mu_0 = float(series[0])
    z = np.empty(series.size)
    z[0] = mu_0
    for t in range(1, series.size):
        z[t] = lambda_smooth * series[t] + (1.0 - lambda_smooth) * z[t - 1]

    steps = np.arange(1, series.size + 1, dtype=float)
    decay = 1.0 - (1.0 - lambda_smooth) ** (2.0 * steps)
    limit_factor = width * sigma * np.sqrt((lambda_smooth / (2.0 - lambda_smooth)) * decay)
    violations = np.flatnonzero(np.abs(z - mu_0) > limit_factor).astype(np.int64)
    return ControlChartResult(
        statistic=z,
        center_line=mu_0,
        upper_limit=float(mu_0 + limit_factor.max()),
        lower_limit=float(mu_0 - limit_factor.max()),
        violations=violations,
    )


def cusum_chart(
    values: Sequence[float] | FloatArray,
    target: float | None = None,
    sigma: float | None = None,
    allowance: float = 0.5,
    decision_interval: float = 4.0,
) -> ControlChartResult:
    """Two-sided CUSUM chart returning the net upper-minus-lower cumulative sum."""
    series = _as_series(values)
    mu_0 = float(np.mean(series)) if target is None else target
    if sigma is None:
        sigma = float(series.std(ddof=1))
    if sigma <= 0:
        msg = "sigma must be positive"
        raise ValueError(msg)

    slack = allowance * sigma
    threshold = decision_interval * sigma
    c_plus = 0.0
    c_minus = 0.0
    net = np.empty(series.size)
    breaches: list[int] = []
    for i, value in enumerate(series):
        deviation = value - mu_0
        c_plus = max(c_plus + deviation - slack, 0.0)
        c_minus = max(c_minus - deviation - slack, 0.0)
        net[i] = c_plus - c_minus
        if c_plus >= threshold or c_minus >= threshold:
            breaches.append(i)
    return ControlChartResult(
        statistic=net,
        center_line=mu_0,
        upper_limit=threshold,
        lower_limit=-threshold,
        violations=np.asarray(breaches, dtype=np.int64),
    )


def p_chart(
    defectives: Sequence[int] | FloatArray, lot_sizes: Sequence[int] | FloatArray
) -> ControlChartResult:
    """Attribute control chart for proportions across lots of varying size."""
    counts = np.asarray(defectives, dtype=float)
    sizes = np.asarray(lot_sizes, dtype=float)
    if counts.shape != sizes.shape or counts.ndim != 1 or counts.size < 2:
        msg = "defectives and lot_sizes must be equal-length 1-D series"
        raise ValueError(msg)
    if np.any(counts > sizes) or np.any(sizes <= 0):
        msg = "each lot size must be positive and at least the defectives count"
        raise ValueError(msg)
    proportion = counts / sizes
    p_bar = float(counts.sum() / sizes.sum())
    sigma_i = np.sqrt(p_bar * (1.0 - p_bar) / sizes)
    upper = p_bar + 3.0 * sigma_i
    lower = np.maximum(p_bar - 3.0 * sigma_i, 0.0)
    violations = np.flatnonzero((proportion > upper) | (proportion < lower)).astype(np.int64)
    return ControlChartResult(
        statistic=np.asarray(proportion, dtype=float),
        center_line=p_bar,
        upper_limit=float(upper.max()),
        lower_limit=float(lower.min()),
        violations=violations,
    )


@dataclass(frozen=True)
class CapabilityResult:
    """Process-capability indices against specification limits."""

    cp: float
    cpk: float
    ppm_defective: float


def process_capability(
    values: Sequence[float] | FloatArray,
    lsl: float | None = None,
    usl: float | None = None,
) -> CapabilityResult:
    """Cp/Cpk plus expected defective parts-per-million under normality.

    With only one specification limit given, Cp is infinite by convention
    and Cpk measures the existing side only.
    """
    from scipy import stats as sp_stats

    series = _as_series(values)
    if lsl is None and usl is None:
        msg = "at least one specification limit is required"
        raise ValueError(msg)
    sigma = float(series.std(ddof=1))
    mean = float(series.mean())
    if sigma <= 0:
        msg = "zero variation: process capability is undefined"
        raise ValueError(msg)

    if lsl is not None and usl is not None:
        cp = (usl - lsl) / (6.0 * sigma)
        cpk = min(usl - mean, mean - lsl) / (3.0 * sigma)
    elif usl is not None:
        cp = float("inf")
        cpk = (usl - mean) / (3.0 * sigma)
    else:
        cp = float("inf")
        assert lsl is not None
        cpk = (mean - lsl) / (3.0 * sigma)

    ppm_upper = float(sp_stats.norm.sf((usl - mean) / sigma)) if usl is not None else 0.0
    ppm_lower = float(sp_stats.norm.cdf((lsl - mean) / sigma)) if lsl is not None else 0.0
    return CapabilityResult(
        cp=float(cp), cpk=float(cpk), ppm_defective=(ppm_upper + ppm_lower) * 1e6
    )
