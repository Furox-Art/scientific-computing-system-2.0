"""Heuristic hypothesis generation from observed data patterns."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .stats import (
    benjamini_hochberg_correction,
    bonferroni_correction,
    holm_correction,
    pearson_correlation,
)

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
        """Emit a correlation hypothesis between two aligned series.

        This is a single uncorrected scan: when many pairs are tried, the
        chance that at least one looks correlated grows with the number of
        attempts (p-hacking). Use :meth:`from_pairs` to scan a batch with a
        multiple-comparison correction instead.
        """
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

    def from_pairs(
        self,
        pairs: Sequence[tuple[str, object, str, object]],
        domain: str = Domain.GENERAL,
        alpha: float = 0.05,
        method: str = "bh",
    ) -> list[Hypothesis]:
        """Scan many series pairs at once with multiple-comparison correction.

        The anti-p-hacking counterpart of :meth:`from_pair`: every testable
        pair contributes a Pearson p-value, the whole family of p-values is
        corrected (Benjamini-Hochberg FDR by default, or Holm/Bonferroni
        family-wise), and only pairs that survive both the ``|r| >= 0.5``
        strength gate and ``adjusted p < alpha`` are emitted. Scanning the
        same pairs one-by-one with :meth:`from_pair` would ignore how many
        attempts were made; this method charges for all of them.

        Args:
            pairs: ``(first_name, first_values, second_name, second_values)``
                tuples. Pairs that cannot be tested (misaligned, too short or
                constant series) are skipped, not silently counted as passes.
            domain: domain label for emitted hypotheses.
            alpha: significance level for the corrected p-values, in (0, 1).
            method: ``"bh"`` (FDR), ``"holm"`` or ``"bonferroni"`` (family-wise).

        Raises:
            ValueError: unknown method or alpha outside (0, 1).
        """
        corrections = {
            "bh": benjamini_hochberg_correction,
            "holm": holm_correction,
            "bonferroni": bonferroni_correction,
        }
        if method not in corrections:
            msg = f"method must be one of {sorted(corrections)}: {method!r}"
            raise ValueError(msg)
        if not 0.0 < alpha < 1.0:
            msg = "alpha must be in (0, 1)"
            raise ValueError(msg)
        scanned: list[tuple[str, str, float, float]] = []
        for first_name, first_values, second_name, second_values in pairs:
            try:
                correlation = pearson_correlation(first_values, second_values)
            except ValueError:
                continue
            scanned.append((first_name, second_name, correlation.r, correlation.p_value))
        if not scanned:
            return []
        adjusted = corrections[method]([item[3] for item in scanned])
        kept: list[Hypothesis] = []
        for (first_name, second_name, r_value, _p), adjusted_p in zip(
            scanned, adjusted, strict=True
        ):
            strength = abs(r_value)
            if strength < 0.5 or adjusted_p >= alpha:
                continue
            confidence = min(0.95, strength)
            if confidence < self.minimum_confidence:
                continue
            direction = "drives" if r_value > 0 else "suppresses"
            hypothesis = Hypothesis(
                statement=(f"{first_name} {direction} {second_name} (pearson r={r_value:+.3f})"),
                domain=domain,
                confidence=confidence,
                rationale=(
                    f"linear correlation strength above 0.5; {method}-adjusted "
                    f"p={adjusted_p:.3g} < {alpha:g} across {len(scanned)} scanned pairs"
                ),
            )
            kept.append(hypothesis)
        self.generated.extend(kept)
        return kept

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
