"""Tests for cds2.design of experiments."""

import numpy as np
import pytest

from cds2 import design


class TestFullFactorial:
    def test_run_count_and_levels(self) -> None:
        result = design.full_factorial([3, 4, 2])
        assert result.matrix.shape == (24, 3)
        assert sorted(set(result.matrix[:, 1].tolist())) == [0, 1, 2, 3]

    def test_factor_names(self) -> None:
        result = design.full_factorial([2, 2], factor_names=["temp", "press"])
        assert result.factor_names == ("temp", "press")

    def test_name_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="match"):
            design.full_factorial([2, 2], factor_names=["only"])

    def test_invalid_levels(self) -> None:
        with pytest.raises(ValueError, match="two levels"):
            design.full_factorial([1, 3])

    def test_huge_design_rejected(self) -> None:
        with pytest.raises(ValueError, match="refusing"):
            design.full_factorial([100, 100, 100])

    def test_randomized_reproducible(self) -> None:
        first = design.full_factorial([3, 3], randomize=True, seed=9)
        second = design.full_factorial([3, 3], randomize=True, seed=9)
        np.testing.assert_array_equal(first.matrix, second.matrix)
        assert first.randomized


class TestFractionalFactorial:
    def test_full_two_level(self) -> None:
        result = design.fractional_factorial_2k(3)
        assert result.matrix.shape == (8, 3)
        assert set(result.matrix[:, 0].tolist()) == {-1.0, 1.0}

    def test_half_fraction_generator(self) -> None:
        result = design.fractional_factorial_2k(4, generator="D=ABC")
        assert result.matrix.shape == (8, 4)
        expected = result.matrix[:, 0] * result.matrix[:, 1] * result.matrix[:, 2]
        np.testing.assert_allclose(result.matrix[:, 3], expected)

    def test_too_few_factors(self) -> None:
        with pytest.raises(ValueError, match="between 2 and 26"):
            design.fractional_factorial_2k(1)

    def test_bad_generator_word(self) -> None:
        with pytest.raises(ValueError, match="generator must look"):
            design.fractional_factorial_2k(3, generator="C")

    def test_unknown_generator_factor(self) -> None:
        with pytest.raises(ValueError, match="unknown factors"):
            design.fractional_factorial_2k(3, generator="D=ABZ")


class TestLatinHypercube:
    def test_shape_and_stratification(self) -> None:
        samples = design.latin_hypercube(50, 3, seed=1)
        assert samples.shape == (50, 3)
        for column in range(3):
            counts, _ = np.histogram(samples[:, column], bins=10, range=(0.0, 1.0))
            assert counts.max() - counts.min() <= 1

    def test_centered_midpoints(self) -> None:
        samples = design.latin_hypercube(20, 2, seed=2, centered=True)
        bins = (samples * 20).astype(int)
        np.testing.assert_allclose(
            samples * 20 - bins,
            0.5,
            atol=1e-12,
        )

    def test_bounds_respected_and_seed_stable(self) -> None:
        first = design.latin_hypercube(30, 2, seed=5)
        second = design.latin_hypercube(30, 2, seed=5)
        np.testing.assert_allclose(first, second)
        assert first.min() >= 0.0
        assert first.max() < 1.0

    def test_invalid_sizes(self) -> None:
        with pytest.raises(ValueError, match="at least two samples"):
            design.latin_hypercube(1, 2)


class TestCentralComposite:
    def test_run_count(self) -> None:
        result = design.central_composite(3)
        cube, star, centre = 8, 6, 6
        assert result.matrix.shape == (cube + star + centre, 3)

    def test_rotatable_alpha(self) -> None:
        result = design.central_composite(2)
        alpha = float(np.abs(result.matrix).max())
        assert alpha == pytest.approx((2**2) ** 0.25)

    def test_face_centered(self) -> None:
        result = design.central_composite(3, face_centered=True)
        assert float(np.abs(result.matrix).max()) == 1.0

    def test_explicit_alpha(self) -> None:
        result = design.central_composite(2, alpha=2.0)
        assert float(np.abs(result.matrix).max()) == 2.0

    def test_factor_range_enforced(self) -> None:
        with pytest.raises(ValueError, match="between 2 and 8"):
            design.central_composite(9)


class TestPluckFactors:
    def test_coded_to_physical(self) -> None:
        design_result = design.full_factorial([2, 2], factor_names=("t", "p"))
        physical = design.pluck_factors(design_result, low=[300.0, 1.0], high=[400.0, 2.0])
        assert set(physical[:, 0].tolist()) == {300.0, 400.0}
        assert set(physical[:, 1].tolist()) == {1.0, 2.0}

    def test_plain_matrix_accepted(self) -> None:
        physical = design.pluck_factors([[-1.0], [1.0]], low=[10.0], high=[30.0])
        np.testing.assert_allclose(sorted(physical[:, 0]), [10.0, 30.0])

    def test_width_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="design width"):
            design.pluck_factors([[1.0, -1.0]], low=[0.0], high=[1.0])


class TestDesignCoverageEdges:
    def test_generator_without_equals_sign(self) -> None:
        with pytest.raises(ValueError, match='must look like "D=ABC"'):
            design.fractional_factorial_2k(4, generator="DABC")

    def test_generator_multi_letter_new_factor(self) -> None:
        with pytest.raises(ValueError, match='must look like "D=ABC"'):
            design.fractional_factorial_2k(4, generator="DD=AB")

    def test_generator_non_alpha_interaction(self) -> None:
        with pytest.raises(ValueError, match="unknown factors"):
            design.fractional_factorial_2k(4, generator="D=AB2")
