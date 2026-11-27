"""Reference-value tests for multiple-comparison p-value corrections.

Hand-computed families pin the exact step-down/step-up arithmetic, seeded
random families are cross-checked against SciPy's independent
``false_discovery_control`` and a naive Holm implementation, and the
known ordering BH <= Holm <= Bonferroni is asserted elementwise.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from cds2.stats import (
    benjamini_hochberg_correction,
    bonferroni_correction,
    holm_correction,
)

CORRECTIONS = [
    bonferroni_correction,
    holm_correction,
    benjamini_hochberg_correction,
]

EXAMPLE = [0.01, 0.04, 0.03, 0.002]


def _reference_holm(p: np.ndarray) -> np.ndarray:
    """Naive step-down Holm with an explicit loop."""
    m = p.size
    order = sorted(range(m), key=lambda i: p[i])
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


class TestReferenceValues:
    def test_bonferroni_hand_computed(self) -> None:
        np.testing.assert_allclose(
            bonferroni_correction(EXAMPLE), [0.04, 0.16, 0.12, 0.008], rtol=1e-12
        )

    def test_holm_hand_computed(self) -> None:
        # sorted [0.002, 0.01, 0.03, 0.04] * [4, 3, 2, 1]
        #   = [0.008, 0.03, 0.06, 0.04] -> cummax [0.008, 0.03, 0.06, 0.06]
        np.testing.assert_allclose(holm_correction(EXAMPLE), [0.03, 0.06, 0.06, 0.008], rtol=1e-12)

    def test_bh_hand_computed(self) -> None:
        # sorted * m/rank = [0.008, 0.02, 0.04, 0.04]; reverse cummin keeps it
        np.testing.assert_allclose(
            benjamini_hochberg_correction(EXAMPLE), [0.02, 0.04, 0.04, 0.008], rtol=1e-12
        )

    def test_bh_matches_scipy_false_discovery_control(self) -> None:
        rng = np.random.default_rng(101)
        p = rng.uniform(0.0, 1.0, size=50)
        np.testing.assert_allclose(
            benjamini_hochberg_correction(p),
            sps.false_discovery_control(p, method="bh"),
            rtol=1e-12,
        )

    def test_holm_matches_naive_reference(self) -> None:
        rng = np.random.default_rng(103)
        p = rng.uniform(0.0, 1.0, size=40)
        np.testing.assert_allclose(holm_correction(p), _reference_holm(p), rtol=1e-12)

    def test_rejection_counts_at_alpha_005(self) -> None:
        alpha = 0.05
        # bonf [0.04, 0.16, 0.12, 0.008] -> 2; holm [0.03, 0.06, 0.06, 0.008] -> 2
        # bh [0.02, 0.04, 0.04, 0.008] -> 4 (FDR is less conservative)
        assert int(np.sum(bonferroni_correction(EXAMPLE) < alpha)) == 2
        assert int(np.sum(holm_correction(EXAMPLE) < alpha)) == 2
        assert int(np.sum(benjamini_hochberg_correction(EXAMPLE) < alpha)) == 4


class TestProperties:
    @pytest.mark.parametrize("correction", CORRECTIONS)
    def test_single_p_value_is_unchanged(self, correction) -> None:  # type: ignore[no-untyped-def]
        assert correction([0.03])[0] == pytest.approx(0.03)

    @pytest.mark.parametrize("correction", CORRECTIONS)
    def test_bounds_and_extremes(self, correction) -> None:  # type: ignore[no-untyped-def]
        zeros = correction([0.0, 0.0, 0.0])
        ones = correction([1.0, 1.0, 1.0])
        np.testing.assert_allclose(zeros, 0.0)
        np.testing.assert_allclose(ones, 1.0)
        rng = np.random.default_rng(107)
        p = rng.uniform(0.0, 1.0, size=25)
        adjusted = correction(p)
        assert bool(np.all((adjusted >= 0.0) & (adjusted <= 1.0)))
        assert bool(np.all(adjusted >= p - 1e-12))

    @pytest.mark.parametrize("correction", [holm_correction, benjamini_hochberg_correction])
    def test_adjusted_monotone_in_raw_order(self, correction) -> None:  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(109)
        p = rng.uniform(0.0, 1.0, size=30)
        adjusted = correction(p)
        order = np.argsort(p, kind="stable")
        assert bool(np.all(np.diff(adjusted[order]) >= -1e-12))

    def test_power_ordering_bh_le_holm_le_bonferroni(self) -> None:
        rng = np.random.default_rng(113)
        p = rng.uniform(0.0, 1.0, size=60)
        bh = benjamini_hochberg_correction(p)
        holm = holm_correction(p)
        bonf = bonferroni_correction(p)
        assert bool(np.all(bh <= holm + 1e-12))
        assert bool(np.all(holm <= bonf + 1e-12))

    def test_ties_are_handled_consistently(self) -> None:
        p = [0.02, 0.02, 0.5, 0.02]
        np.testing.assert_allclose(holm_correction(p), _reference_holm(np.asarray(p)), rtol=1e-12)
        np.testing.assert_allclose(
            benjamini_hochberg_correction(p),
            sps.false_discovery_control(p, method="bh"),
            rtol=1e-12,
        )


class TestValidation:
    @pytest.mark.parametrize("correction", CORRECTIONS)
    @pytest.mark.parametrize(
        "bad",
        [[], [[0.1, 0.2], [0.3, 0.4]], [float("nan")], [-0.1], [1.2], [0.5, float("inf")]],
    )
    def test_invalid_inputs_rejected(self, correction, bad) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError):
            correction(bad)
