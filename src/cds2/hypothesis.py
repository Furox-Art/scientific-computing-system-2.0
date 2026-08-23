"""Heuristic hypothesis generation from observed data patterns."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stats import pearson_correlation

__all__ = [
    "Domain",
    "Hypothesis",
    "HypothesisEngine",
]


class Domain:
    """Registry of scientific domains a hypothesis may belong to."""

    PHYSICS = "physics"
    STATISTICS = "statistics"
    BIOLOGY = "biology"
    ECONOMICS = "economics"
    ENGINEERING = "engineering"
    GENERAL = "general"

    ALL: frozenset[str] = frozenset({PHYSICS, STATISTICS, BIOLOGY, ECONOMICS, ENGINEERING, GENERAL})


@dataclass(frozen=True)
class Hypothesis:
    """A testable claim with a confidence heuristic attached."""

    statement: str
    domain: str = Domain.GENERAL
    confidence: float = 0.5
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            msg = "confidence must lie in [0, 1]"
            raise ValueError(msg)
        if self.domain not in Domain.ALL:
            msg = f"unknown domain: {self.domain!r}"
            raise ValueError(msg)


@dataclass
class HypothesisEngine:
    """Emits candidate hypotheses from numeric observation series."""

    minimum_confidence: float = 0.4
    generated: list[Hypothesis] = field(default_factory=list)

    def from_series(
        self,
        name: str,
        values: object,
        domain: str = Domain.GENERAL,
    ) -> list[Hypothesis]:
        """Scan one series for trend, periodicity and outlier patterns."""
        data_values = np.asarray(values, dtype=float).ravel()
        if data_values.size < 4:
            msg = "at least four observations are required"
            raise ValueError(msg)
        found: list[Hypothesis] = []
        found.extend(self._trend_hypotheses(name, data_values, domain))
        found.extend(self._periodicity_hypothesis(name, data_values, domain))
        found.extend(self._outlier_hypothesis(name, data_values, domain))
        kept = [item for item in found if item.confidence >= self.minimum_confidence]
        self.generated.extend(kept)
        return kept

    def from_pair(
        self,
        first_name: str,
        first_values: object,
        second_name: str,
        second_values: object,
        domain: str = Domain.GENERAL,
    ) -> list[Hypothesis]:
        """Emit a correlation hypothesis between two aligned series."""
        first_array = np.asarray(first_values, dtype=float).ravel()
        second_array = np.asarray(second_values, dtype=float).ravel()
        if first_array.size != second_array.size or first_array.size < 4:
            msg = "series must be aligned and contain at least four points"
            raise ValueError(msg)
        correlation = pearson_correlation(first_array, second_array)
        strength = abs(correlation.r)
        if strength < 0.5:
            return []
        direction = "drives" if correlation.r > 0 else "suppresses"
        confidence = min(0.95, strength)
        hypothesis = Hypothesis(
            statement=f"{first_name} {direction} {second_name} (pearson r={correlation.r:+.3f})",
            domain=domain,
            confidence=confidence,
            rationale="linear correlation strength above 0.5",
        )
        if confidence >= self.minimum_confidence:
            self.generated.append(hypothesis)
        return [hypothesis]

    def _trend_hypotheses(self, name: str, values: np.ndarray, domain: str) -> list[Hypothesis]:
        indices = np.arange(values.size, dtype=float)
        slope, intercept = np.polyfit(indices, values, 1)
        fitted = slope * indices + intercept
        residual_std = float(np.std(values - fitted, ddof=1)) or 1e-12
        signal = abs(slope) * values.size / (residual_std * 4.0)
        confidence = float(np.clip(signal / (1.0 + signal), 0.0, 0.95))
        if confidence < self.minimum_confidence:
            return []
        direction = "rising" if slope > 0 else "falling"
        return [
            Hypothesis(
                statement=f"{name} shows a {direction} linear trend (slope={slope:.4g} per step)",
                domain=domain,
                confidence=confidence,
                rationale="linear fit slope dominates residual noise",
            )
        ]

    def _periodicity_hypothesis(
        self, name: str, values: np.ndarray, domain: str
    ) -> list[Hypothesis]:
        centered = values - values.mean()
        if float(np.std(centered)) == 0.0:
            return []
        correlations = np.correlate(centered, centered, mode="full")[values.size - 1 :]
        correlations /= correlations[0] if correlations[0] != 0 else 1.0
        interior = correlations[1:]
        best_lag = int(np.argmax(np.abs(interior))) + 1
        peak = abs(float(correlations[best_lag]))
        confidence = float(np.clip(peak * 0.9, 0.0, 0.9))
        if confidence < self.minimum_confidence or best_lag == values.size - 1:
            return []
        return [
            Hypothesis(
                statement=f"{name} repeats with a period of about "
                f"{best_lag} observations (autocorrelation {correlations[best_lag]:+.3f})",
                domain=domain,
                confidence=confidence,
                rationale="autocorrelation peak at non-zero lag",
            )
        ]

    def _outlier_hypothesis(self, name: str, values: np.ndarray, domain: str) -> list[Hypothesis]:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) or 1e-12
        robust_z = np.abs(values - median) / (1.4826 * mad)
        outlier_count = int(np.count_nonzero(robust_z > 3.5))
        if outlier_count == 0:
            return []
        confidence = float(np.clip(0.4 + 0.1 * outlier_count, 0.0, 0.85))
        return [
            Hypothesis(
                statement=f"{name} contains {outlier_count} anomalous "
                "observation(s) worth investigating",
                domain=domain,
                confidence=confidence,
                rationale="MAD-robust z-score above 3.5",
            )
        ]
