"""Tests for cds2.quality statistical process control."""

import numpy as np
import pytest
from scipy import stats as sp_stats

from cds2 import quality


class TestXbarChart:
    def test_in_control_series_has_no_violations(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=10.0, scale=1.0, size=(20, 4))
        chart = quality.xbar_chart(samples)
        assert chart.violations.size == 0
        assert chart.lower_limit < chart.center_line < chart.upper_limit

    def test_out_of_control_point_is_flagged(self) -> None:
        rng = np.random.default_rng(1)
        subgroups = rng.normal(loc=10.0, scale=0.5, size=(15, 5))
        subgroups[7] += 8.0
        chart = quality.xbar_chart(subgroups)
        assert 7 in chart.violations.tolist()

    def test_1d_input_splitting(self) -> None:
        values = np.tile([10.0, 10.2, 9.8, 10.1, 9.9], 6)
        chart = quality.xbar_chart(values, subgroup_size=5)
        assert chart.statistic.size == 6

    def test_invalid_subgroup_size(self) -> None:
        with pytest.raises(ValueError, match="between 2 and 10"):
            quality.xbar_chart(np.arange(40.0), subgroup_size=11)

    def test_too_few_subgroups(self) -> None:
        with pytest.raises(ValueError, match="two full subgroups"):
            quality.xbar_chart(np.arange(7.0), subgroup_size=5)

    def test_subgroup_size_with_2d_rejected(self) -> None:
        block = np.ones((3, 4))
        with pytest.raises(ValueError, match="cannot be combined"):
            quality.xbar_chart(block, subgroup_size=3)


class TestEwmaChart:
    def test_level_shift_is_detected(self) -> None:
        rng = np.random.default_rng(2)
        series = np.concatenate([rng.normal(5.0, 0.2, 60), rng.normal(6.0, 0.2, 40)])
        chart = quality.ewma_chart(series)
        assert chart.violations.size > 0
        assert int(chart.violations.max()) > 60

    def test_stable_process_clean(self) -> None:
        rng = np.random.default_rng(3)
        series = rng.normal(0.0, 0.05, 200)
        chart = quality.ewma_chart(series, lambda_smooth=0.1)
        assert chart.violations.size == 0

    def test_invalid_lambda(self) -> None:
        with pytest.raises(ValueError, match="lambda_smooth"):
            quality.ewma_chart([1.0, 2.0], lambda_smooth=0.0)

    def test_zero_sigma_rejected(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            quality.ewma_chart([1.0, 1.0, 1.0, 1.0], sigma=0.0)


class TestCusumChart:
    def test_sustained_shift_detected(self) -> None:
        rng = np.random.default_rng(4)
        series = np.concatenate([rng.normal(0.0, 0.5, 80), rng.normal(2.0, 0.5, 40)])
        chart = quality.cusum_chart(series)
        assert chart.violations.size > 0

    def test_target_and_sigma_respected(self) -> None:
        series = [0.0] * 30 + [3.0] * 10
        chart = quality.cusum_chart(series, target=0.0, sigma=1.0)
        assert chart.center_line == 0.0
        assert chart.statistic.size == len(series)

    def test_zero_sigma_rejected(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            quality.cusum_chart([2.0] * 10)


class TestPChart:
    def test_proportions_and_limits(self) -> None:
        defectives = [12, 15, 11, 14, 13]
        lots = [200, 210, 190, 205, 195]
        chart = quality.p_chart(defectives, lots)
        assert chart.center_line == pytest.approx(65 / 1000)
        assert chart.violations.size == 0
        assert np.all(chart.statistic <= chart.upper_limit)

    def test_extreme_lot_flagged(self) -> None:
        defectives = [10, 11, 12, 60]
        lots = [300, 300, 300, 300]
        chart = quality.p_chart(defectives, lots)
        assert 3 in chart.violations.tolist()

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            quality.p_chart([1, 2], [100])

    def test_defectives_exceeding_lot_size(self) -> None:
        with pytest.raises(ValueError, match="at least the defectives"):
            quality.p_chart([150, 10], [100, 100])


class TestProcessCapability:
    def test_centered_normal_process(self) -> None:
        rng = np.random.default_rng(5)
        values = rng.normal(loc=50.0, scale=2.0, size=500)
        result = quality.process_capability(values, lsl=40.0, usl=60.0)
        assert result.cp == pytest.approx((60 - 40) / (6 * 2.0), rel=0.05)
        assert abs(result.cpk - result.cp) < 0.15
        assert 0 <= result.ppm_defective < 2000

    def test_off_center_cpk_below_cp(self) -> None:
        rng = np.random.default_rng(6)
        values = rng.normal(loc=55.0, scale=2.0, size=500)
        result = quality.process_capability(values, lsl=40.0, usl=60.0)
        assert result.cpk < result.cp

    def test_one_sided_upper(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.normal(loc=5.0, scale=1.0, size=400)
        result = quality.process_capability(values, usl=11.0)
        assert result.cp == float("inf")
        assert result.cpk > 0
        assert result.ppm_defective > 0

    def test_one_sided_lower(self) -> None:
        rng = np.random.default_rng(9)
        values = rng.normal(loc=5.0, scale=1.0, size=400)
        result = quality.process_capability(values, lsl=-1.0)
        assert result.cp == float("inf")
        assert result.cpk > 0
        assert result.ppm_defective > 0

    def test_no_limits_raises(self) -> None:
        with pytest.raises(ValueError, match="specification limit"):
            quality.process_capability([1.0, 2.0])

    def test_zero_variation_raises(self) -> None:
        with pytest.raises(ValueError, match="zero variation"):
            quality.process_capability([3.0] * 10, lsl=0.0, usl=9.0)

    def test_ppm_matches_normal_tail(self) -> None:
        values = np.linspace(-3.9, 3.9, 781)
        result = quality.process_capability(values, usl=1.96)
        expected_upper = float(sp_stats.norm.sf(1.96 / values.std(ddof=1))) * 1e6
        assert result.ppm_defective == pytest.approx(expected_upper, rel=0.05)


class TestQualityCoverageEdges:
    def test_series_validation(self) -> None:
        with pytest.raises(ValueError, match="1-D series"):
            quality.ewma_chart([1.0])
        with pytest.raises(ValueError, match="1-D series"):
            quality.cusum_chart([[1.0, 2.0]])

    def test_subgroup_width_rejected(self) -> None:
        block = np.ones((4, 11))
        with pytest.raises(ValueError, match="sizes between 2 and 10"):
            quality.xbar_chart(block)

    def test_three_dimensional_input_rejected(self) -> None:
        cube = np.ones((2, 3, 2))
        with pytest.raises(ValueError, match="1-D or 2-D"):
            quality.xbar_chart(cube)
