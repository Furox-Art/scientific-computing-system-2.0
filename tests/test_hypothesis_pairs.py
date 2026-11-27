"""Tests for HypothesisEngine.from_pairs: batch scanning with FDR control.

The point of these tests is the p-hacking mitigation: a batch of mostly
noise pairs must not emit hypotheses just because one correlation looks
strong, and the survivors must match an independent correction computed
with SciPy on the same Pearson p-values.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from cds2.hypothesis import Domain, HypothesisEngine


def _noise_pairs(seed: int, count: int, n: int = 12) -> list[tuple[str, object, str, object]]:
    rng = np.random.default_rng(seed)
    pairs: list[tuple[str, object, str, object]] = []
    for i in range(count):
        pairs.append((f"a{i}", rng.normal(size=n), f"b{i}", rng.normal(size=n)))
    return pairs


class TestFromPairs:
    def test_true_effect_survives_a_noise_batch(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=40)
        y = 2.0 * x + rng.normal(scale=0.3, size=40)
        pairs = [("signal", x, "response", y), *_noise_pairs(11, 24)]
        engine = HypothesisEngine()
        kept = engine.from_pairs(pairs, domain=Domain.PHYSICS)
        assert len(kept) == 1
        assert "signal drives response" in kept[0].statement
        assert kept[0].domain == Domain.PHYSICS
        assert "bh-adjusted" in kept[0].rationale
        assert "25 scanned pairs" in kept[0].rationale
        assert engine.generated == kept

    def test_survivors_match_independent_scipy_correction(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.normal(size=30)
        pairs = [
            ("strong", x, "copy", x + rng.normal(scale=0.2, size=30)),
            *_noise_pairs(29, 30),
        ]
        engine = HypothesisEngine()
        kept = engine.from_pairs(pairs, alpha=0.05, method="bh")
        # Independent expectation: Pearson p-values corrected by SciPy.
        results = [sps.pearsonr(a, b) for _, a, _, b in pairs]
        adjusted = sps.false_discovery_control([r.pvalue for r in results], method="bh")
        expected = {
            pairs[i][0]
            for i, r in enumerate(results)
            if abs(r.statistic) >= 0.5 and adjusted[i] < 0.05
        }
        assert {h.statement.split(" ")[0] for h in kept} == expected

    def test_noise_batch_emits_at_most_few(self) -> None:
        engine = HypothesisEngine()
        kept = engine.from_pairs(_noise_pairs(123, 25), alpha=0.05)
        assert len(kept) <= 1
        assert engine.generated == kept

    def test_family_wise_methods_are_stricter_or_equal(self) -> None:
        rng = np.random.default_rng(31)
        x = rng.normal(size=20)
        pairs = [
            ("medium", x, "noisy", x + rng.normal(scale=1.1, size=20)),
            *_noise_pairs(37, 15),
        ]
        bh = {h.statement for h in HypothesisEngine().from_pairs(pairs, method="bh")}
        holm = {h.statement for h in HypothesisEngine().from_pairs(pairs, method="holm")}
        bonf = {h.statement for h in HypothesisEngine().from_pairs(pairs, method="bonferroni")}
        assert holm <= bh
        assert bonf <= holm

    def test_strong_but_underpowered_pair_is_not_emitted(self) -> None:
        # r = 0.8 passes the strength gate, but with n=5 the two-sided
        # p-value is ~0.10; the uncorrected from_pair would still emit it.
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 3.0, 2.0, 5.0, 4.0]
        engine = HypothesisEngine()
        assert engine.from_pairs([("x", x, "y", y)]) == []
        assert engine.generated == []
        assert len(engine.from_pair("x", x, "y", y)) == 1

    def test_negative_correlation_uses_suppresses(self) -> None:
        rng = np.random.default_rng(41)
        x = rng.normal(size=40)
        kept = HypothesisEngine().from_pairs([("x", x, "y", -x + rng.normal(scale=0.2, size=40))])
        assert len(kept) == 1
        assert "x suppresses y" in kept[0].statement

    def test_weak_correlation_below_strength_gate_skipped(self) -> None:
        rng = np.random.default_rng(43)
        x = rng.normal(size=200)
        y = 0.2 * x + rng.normal(size=200)  # r ~ 0.2, significant but weak
        assert HypothesisEngine().from_pairs([("x", x, "y", y)]) == []

    def test_untestable_pairs_are_skipped(self) -> None:
        engine = HypothesisEngine()
        pairs = [
            ("const", [1.0] * 6, "other", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("short", [1.0, 2.0], "other", [1.0, 2.0]),
            ("misaligned", [1.0, 2.0, 3.0, 4.0], "other", [1.0, 2.0, 3.0]),
        ]
        assert engine.from_pairs(pairs) == []
        assert engine.generated == []

    def test_empty_batch(self) -> None:
        assert HypothesisEngine().from_pairs([]) == []

    def test_minimum_confidence_still_applies(self) -> None:
        rng = np.random.default_rng(47)
        x = rng.normal(size=40)
        y = x + rng.normal(scale=0.25, size=40)  # r ~ 0.97 -> confidence capped 0.95
        engine = HypothesisEngine(minimum_confidence=0.99)
        assert engine.from_pairs([("x", x, "y", y)]) == []
        assert engine.generated == []

    def test_invalid_method_rejected(self) -> None:
        with pytest.raises(ValueError, match="method"):
            HypothesisEngine().from_pairs([], method="sidak")

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_alpha_rejected(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            HypothesisEngine().from_pairs([], alpha=alpha)
