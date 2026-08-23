"""Tests for the hypothesis generation engine."""

import numpy as np
import pytest

from cds2.hypothesis import Domain, Hypothesis, HypothesisEngine


class TestHypothesisValidation:
    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Hypothesis(statement="x", confidence=1.5)

    def test_unknown_domain(self) -> None:
        with pytest.raises(ValueError, match="domain"):
            Hypothesis(statement="x", domain="astrology")


class TestTrendDetection:
    def test_rising_trend_detected(self) -> None:
        engine = HypothesisEngine()
        values = np.linspace(0.0, 50.0, 60) + np.random.default_rng(0).normal(scale=0.5, size=60)
        found = engine.from_series("temperature", values, domain=Domain.ECONOMICS)
        assert any("rising" in item.statement for item in found)
        assert all(item.domain == Domain.ECONOMICS for item in found)

    def test_falling_trend_detected(self) -> None:
        engine = HypothesisEngine()
        values = np.linspace(50.0, 0.0, 60)
        found = engine.from_series("battery", values)
        assert any("falling" in item.statement for item in found)

    def test_flat_series_no_trend(self) -> None:
        engine = HypothesisEngine()
        values = np.random.default_rng(1).normal(scale=0.001, size=60)
        assert not any("trend" in item.statement for item in engine.from_series("flat", values))

    def test_generated_accumulates(self) -> None:
        engine = HypothesisEngine()
        first = engine.from_series("a", np.linspace(0, 10, 40))
        second = engine.from_series("b", np.linspace(10, 0, 40))
        assert len(engine.generated) == len(first) + len(second)


class TestPeriodicityAndOutliers:
    def test_periodic_series_detected(self) -> None:
        engine = HypothesisEngine()
        steps = np.arange(120)
        values = 5.0 * np.sin(2 * np.pi * steps / 12.0)
        found = engine.from_series("seasonal", values)
        assert any("period" in item.statement for item in found)

    def test_outliers_flagged(self) -> None:
        engine = HypothesisEngine()
        values = np.concatenate([np.zeros(50), [100.0]])
        found = engine.from_series("spiky", values)
        assert any("anomal" in item.statement for item in found)

    def test_quiet_series_no_outliers(self) -> None:
        engine = HypothesisEngine()
        values = np.zeros(40)
        assert not any("anomal" in item.statement for item in engine.from_series("calm", values))


class TestPairCorrelation:
    def test_strong_positive_pair(self) -> None:
        engine = HypothesisEngine()
        first = np.linspace(0, 10, 50)
        second = first * 2 + np.random.default_rng(2).normal(scale=0.1, size=50)
        found = engine.from_pair("sunlight", first, "growth", second, domain=Domain.BIOLOGY)
        assert found
        assert "drives" in found[0].statement
        assert found[0].domain == Domain.BIOLOGY

    def test_strong_negative_pair_suppresses(self) -> None:
        engine = HypothesisEngine()
        first = np.linspace(0, 10, 50)
        second = -first + np.random.default_rng(3).normal(scale=0.1, size=50)
        found = engine.from_pair("price", first, "demand", second, domain=Domain.ECONOMICS)
        assert "suppresses" in found[0].statement

    def test_uncorrelated_pair_no_hypothesis(self) -> None:
        engine = HypothesisEngine()
        first = np.random.default_rng(4).normal(size=50)
        second = np.random.default_rng(5).normal(size=50)
        assert engine.from_pair("a", first, "b", second) == []

    def test_misaligned_series_raise(self) -> None:
        engine = HypothesisEngine()
        with pytest.raises(ValueError, match="aligned"):
            engine.from_pair("a", np.zeros(10), "b", np.zeros(5))


class TestValidationAndConfidence:
    def test_short_series_raise(self) -> None:
        engine = HypothesisEngine()
        with pytest.raises(ValueError, match="four"):
            engine.from_series("tiny", [1.0, 2.0])

    def test_minimum_confidence_filter(self) -> None:
        strict = HypothesisEngine(minimum_confidence=0.99)
        values = np.linspace(0, 10, 40) + np.random.default_rng(6).normal(scale=3, size=40)
        assert strict.from_series("noisy", values) == []

    def test_confidence_within_bounds(self) -> None:
        engine = HypothesisEngine()
        values = np.linspace(0, 100, 80)
        for hypothesis in engine.from_series("clean", values):
            assert 0.0 <= hypothesis.confidence <= 1.0
