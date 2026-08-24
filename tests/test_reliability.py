"""Tests for cds2.reliability."""

import numpy as np
import pytest
from scipy import stats as sp_stats

from cds2 import reliability as rel


class TestKaplanMeier:
    def test_hand_computed_example(self) -> None:
        result = rel.kaplan_meier([3.0, 5.0, 7.0, 8.0, 10.0], [1, 0, 1, 1, 1])
        assert result.times.tolist() == [3.0, 7.0, 8.0, 10.0]
        s_after_7 = 0.8 * 2.0 / 3.0
        expected = [0.8, s_after_7, s_after_7 * 0.5, 0.0]
        assert result.survival.tolist() == pytest.approx(expected)
        assert result.median == 8.0

    def test_median_none_when_long_survivors(self) -> None:
        result = rel.kaplan_meier([1.0, 50.0, 80.0, 120.0], [1, 0, 0, 0])
        assert result.survival[0] == pytest.approx(0.75)
        assert result.median is None

    def test_all_censored_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one event"):
            rel.kaplan_meier([1.0, 2.0, 3.0], [0, 0, 0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="equal-length non-empty"):
            rel.kaplan_meier([], [])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal-length non-empty"):
            rel.kaplan_meier([1.0, 2.0], [1])

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            rel.kaplan_meier([-1.0, 2.0], [1, 1])


class TestWeibullFit:
    def test_recovers_known_parameters(self) -> None:
        data = sp_stats.weibull_min.rvs(c=1.5, scale=1000.0, size=400, random_state=1)
        fit = rel.weibull_fit(data)
        assert fit.shape == pytest.approx(1.5, rel=0.20)
        assert fit.scale == pytest.approx(1000.0, rel=0.20)

    def test_mask_excludes_censored_entries(self) -> None:
        data = sp_stats.weibull_min.rvs(c=1.5, scale=1000.0, size=400, random_state=1)
        padded = np.concatenate([data, [10.0, 9999.0]])
        mask = np.concatenate([np.ones(400, dtype=bool), [False, False]])
        masked_fit = rel.weibull_fit(padded, failures_mask=mask)
        reference = rel.weibull_fit(data)
        assert masked_fit.shape == pytest.approx(reference.shape, rel=1e-12)
        assert masked_fit.scale == pytest.approx(reference.scale, rel=1e-12)
        assert masked_fit.shape == pytest.approx(1.5, rel=0.20)

    @pytest.mark.parametrize(
        "durations", [[1.0], [], [0.0, 5.0]], ids=["single", "empty", "nonpositive"]
    )
    def test_too_few_positive_durations_raises(self, durations: list[float]) -> None:
        with pytest.raises(ValueError, match="at least two positive durations"):
            rel.weibull_fit(durations)


class TestMtbfAvailability:
    def test_mtbf_known_value(self) -> None:
        assert rel.mtbf(1000.0, 4) == pytest.approx(250.0)

    @pytest.mark.parametrize(("time", "failures"), [(0.0, 1), (-50.0, 2)])
    def test_nonpositive_time_raises(self, time: float, failures: int) -> None:
        with pytest.raises(ValueError, match="total_operating_time must be positive"):
            rel.mtbf(time, failures)

    def test_zero_failures_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one failure"):
            rel.mtbf(500.0, 0)

    def test_availability_known_value(self) -> None:
        assert rel.availability(1000.0, 100.0) == pytest.approx(1000.0 / 1100.0)

    @pytest.mark.parametrize("mttr", [100.0, 150.0])
    def test_availability_mttr_not_below_mtbf_raises(self, mttr: float) -> None:
        with pytest.raises(ValueError, match="mttr must be smaller than mtbf"):
            rel.availability(100.0, mttr)

    @pytest.mark.parametrize(
        ("mtbf_value", "mttr"),
        [(0.0, 1.0), (-1.0, 1.0), (100.0, 0.0), (100.0, -1.0)],
    )
    def test_availability_nonpositive_inputs_raise(self, mtbf_value: float, mttr: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            rel.availability(mtbf_value, mttr)


class TestWeibullSurvival:
    def test_s_zero_is_one_and_decreasing(self) -> None:
        survival = rel.weibull_survival([0.0, 250.0, 1000.0, 3000.0], shape=1.5, scale=1000.0)
        assert survival[0] == pytest.approx(1.0)
        assert np.all(np.diff(survival) < 0.0)

    def test_matches_scipy_sf(self) -> None:
        times = np.array([100.0, 700.0, 1500.0])
        survival = rel.weibull_survival(times, shape=2.0, scale=900.0)
        expected = sp_stats.weibull_min.sf(times, c=2.0, scale=900.0)
        assert survival.tolist() == pytest.approx(expected.tolist())

    @pytest.mark.parametrize(("shape", "scale"), [(0.0, 1.0), (-2.0, 1.0)])
    def test_nonpositive_shape_raises(self, shape: float, scale: float) -> None:
        with pytest.raises(ValueError, match="shape must be positive"):
            rel.weibull_survival([1.0], shape=shape, scale=scale)

    @pytest.mark.parametrize(("shape", "scale"), [(1.0, 0.0), (1.0, -1.0)])
    def test_nonpositive_scale_raises(self, shape: float, scale: float) -> None:
        with pytest.raises(ValueError, match="scale must be positive"):
            rel.weibull_survival([1.0], shape=shape, scale=scale)

    def test_negative_times_raise(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            rel.weibull_survival([0.0, -5.0], shape=1.0, scale=2.0)


class TestBathtubCurve:
    def test_three_regimes(self) -> None:
        early, intrinsic, wearout = 0.05, 0.01, 0.002
        hazard = rel.bathtub_curve(
            [1.0, 1000.0, 20000.0],
            early_rate=early,
            intrinsic_rate=intrinsic,
            wearout_rate=wearout,
            knee_early=50.0,
            knee_wearout=5000.0,
        )
        assert hazard[0] > hazard[1]
        assert hazard[1] == pytest.approx(intrinsic, rel=0.15)
        assert hazard[2] > hazard[1]

    def test_zero_rates_reduce_to_intrinsic(self) -> None:
        hazard = rel.bathtub_curve(
            [10.0, 100.0],
            early_rate=0.0,
            intrinsic_rate=0.02,
            wearout_rate=0.0,
            knee_early=5.0,
            knee_wearout=50.0,
        )
        assert hazard.tolist() == pytest.approx([0.02, 0.02])

    @pytest.mark.parametrize(
        ("field", "value"),
        [("early_rate", -0.01), ("intrinsic_rate", -0.01), ("wearout_rate", -0.01)],
    )
    def test_negative_rate_raises(self, field: str, value: float) -> None:
        kwargs: dict[str, float] = {
            "early_rate": 0.05,
            "intrinsic_rate": 0.01,
            "wearout_rate": 0.002,
            "knee_early": 50.0,
            "knee_wearout": 5000.0,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match="rates must be non-negative"):
            rel.bathtub_curve([1.0, 100.0], **kwargs)

    @pytest.mark.parametrize(("knee_early", "knee_wearout"), [(0.0, 5000.0), (50.0, 0.0)])
    def test_nonpositive_knee_raises(self, knee_early: float, knee_wearout: float) -> None:
        with pytest.raises(ValueError, match="knees must be positive"):
            rel.bathtub_curve(
                [1.0],
                early_rate=0.05,
                intrinsic_rate=0.01,
                wearout_rate=0.002,
                knee_early=knee_early,
                knee_wearout=knee_wearout,
            )
