"""Tests for cds2.infotheory."""

import math

import numpy as np
import pytest

from cds2 import infotheory


class TestEntropy:
    def test_uniform_maximal(self) -> None:
        assert infotheory.entropy([0.25] * 4) == pytest.approx(2.0)

    def test_deterministic_zero(self) -> None:
        assert infotheory.entropy([1.0, 0.0, 0.0]) == 0.0

    def test_nats(self) -> None:
        assert infotheory.entropy([0.5, 0.5], base=np.e) == pytest.approx(np.log(2))

    def test_rejects_unnormalized(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            infotheory.entropy([0.5, 0.6])

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            infotheory.entropy([-0.5, 1.5])

    def test_rejects_bad_base(self) -> None:
        with pytest.raises(ValueError, match="base"):
            infotheory.entropy([0.5, 0.5], base=1.0)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            infotheory.entropy([])


class TestJointAndConditional:
    def test_joint_of_independent_bits(self) -> None:
        joint = [[0.25, 0.25], [0.25, 0.25]]
        assert infotheory.joint_entropy(joint) == pytest.approx(2.0)

    def test_conditional_equals_joint_for_independent(self) -> None:
        joint = [[0.25, 0.25], [0.25, 0.25]]
        assert infotheory.conditional_entropy(joint) == pytest.approx(1.0)

    def test_conditional_zero_when_determined(self) -> None:
        joint = [[0.5, 0.0], [0.5, 0.0]]
        assert infotheory.conditional_entropy(joint) == pytest.approx(0.0)

    def test_marginal_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="marginal shape"):
            infotheory.conditional_entropy([[0.5, 0.5]], [0.5, 0.5])

    def test_zero_row_skipped(self) -> None:
        joint = [[0.0, 0.0], [1.0, 0.0]]
        assert infotheory.conditional_entropy(joint, marginal=[0.0, 1.0]) == pytest.approx(0.0)

    def test_marginal_must_match_row_probabilities(self) -> None:
        joint = [[0.2, 0.3], [0.1, 0.4]]
        with pytest.raises(ValueError, match="marginal probabilities"):
            infotheory.conditional_entropy(joint, marginal=[0.1, 0.9])


class TestDivergences:
    def test_kl_self_is_zero(self) -> None:
        p = [0.3, 0.7]
        assert infotheory.kl_divergence(p, p) == pytest.approx(0.0)

    def test_kl_asymmetry_and_support(self) -> None:
        value = infotheory.kl_divergence([0.9, 0.1], [0.5, 0.5])
        assert value > 0
        assert infotheory.kl_divergence([0.5, 0.5], [0.9, 0.1]) > value

    def test_kl_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            infotheory.kl_divergence([1.0], [0.5, 0.5])

    def test_zero_q_on_positive_p_gives_infinity(self) -> None:
        p, q = [0.4, 0.6, 0.0], [0.0, 0.5, 0.5]
        assert infotheory.kl_divergence(p, q) == math.inf
        assert infotheory.cross_entropy(p, q) == math.inf
        assert infotheory.kl_divergence([1.0, 0.0], [1.0, 0.0]) == 0.0
        assert infotheory.cross_entropy([1.0, 0.0], [1.0, 0.0]) == 0.0

    def test_js_symmetric_and_bounded(self) -> None:
        a = infotheory.js_divergence([0.9, 0.1], [0.1, 0.9])
        b = infotheory.js_divergence([0.1, 0.9], [0.9, 0.1])
        assert a == pytest.approx(b)
        assert 0.0 <= a <= 1.0
        assert infotheory.js_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_js_stays_finite_with_a_subnormal_probability(self) -> None:
        tiny = math.ulp(0.0)
        assert math.isfinite(infotheory.js_divergence([1.0, tiny], [1.0, 0.0]))

    def test_cross_entropy_identity(self) -> None:
        p = [0.4, 0.6]
        expected = infotheory.entropy(p) + infotheory.kl_divergence(p, [0.5, 0.5])
        assert infotheory.cross_entropy(p, [0.5, 0.5]) == pytest.approx(expected)


class TestMutualInformation:
    def test_independent_zero(self) -> None:
        joint = [[0.25, 0.25], [0.25, 0.25]]
        assert infotheory.mutual_information(joint) == pytest.approx(0.0, abs=1e-12)

    def test_identical_variables_one_bit(self) -> None:
        joint = [[0.5, 0.0], [0.0, 0.5]]
        assert infotheory.mutual_information(joint) == pytest.approx(1.0)

    def test_normalized_bounded(self) -> None:
        joint = [[0.4, 0.1], [0.1, 0.4]]
        value = infotheory.normalized_mutual_information(joint)
        assert 0.0 < value <= 1.0

    def test_rejects_unnormalized_joint(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            infotheory.mutual_information([[0.7, 0.7], [0.7, 0.7]])

    def test_agreement_with_rare_outcome(self) -> None:
        tiny = 1e-200
        joint = [[1.0, 0.0], [0.0, tiny]]
        expected = -tiny * math.log2(tiny)
        assert infotheory.mutual_information(joint) == pytest.approx(expected, abs=0.0)
        assert infotheory.normalized_mutual_information(joint) == pytest.approx(1.0)


class TestPermutationEntropy:
    def test_regular_signal_low_entropy(self) -> None:
        regular = np.sin(np.linspace(0.0, 20.0 * np.pi, 2000))
        chaotic = np.random.default_rng(0).uniform(size=2000)
        assert infotheory.permutation_entropy(regular, order=3) < infotheory.permutation_entropy(
            chaotic, order=3
        )

    def test_short_series_raises(self) -> None:
        with pytest.raises(ValueError, match="samples"):
            infotheory.permutation_entropy([1.0, 2.0], order=3)

    def test_invalid_order_raises(self) -> None:
        with pytest.raises(ValueError, match="order"):
            infotheory.permutation_entropy([1.0, 2.0, 3.0], order=1)

    def test_invalid_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="delay"):
            infotheory.permutation_entropy([1.0, 2.0, 3.0], delay=0)

    def test_delay_uses_all_starting_indices(self) -> None:
        spectrum = [0.0, 1.0, 3.0, 0.0, 4.0]
        expected = infotheory.entropy([2 / 3, 1 / 3])
        assert infotheory.permutation_entropy(spectrum, order=2, delay=2) == pytest.approx(expected)

    def test_long_patterns_do_not_overflow_integer_keys(self) -> None:
        signal = np.arange(17, dtype=float)
        assert infotheory.permutation_entropy(signal, order=17) == 0.0

    @pytest.mark.parametrize("signal", [[0.0, math.nan, 1.0], [[0.0, 1.0, 2.0]]])
    def test_rejects_nonfinite_or_multidimensional_signal(self, signal: object) -> None:
        with pytest.raises(ValueError, match="finite 1-D"):
            infotheory.permutation_entropy(np.asarray(signal, dtype=float))


class TestValidationBranches:
    @pytest.mark.parametrize("base", [0.0, 1.0, math.inf, math.nan])
    def test_rejects_invalid_log_bases(self, base: float) -> None:
        with pytest.raises(ValueError, match="base"):
            infotheory.joint_entropy([[1.0]], base=base)
        with pytest.raises(ValueError, match="base"):
            infotheory.cross_entropy([1.0], [1.0], base=base)
        with pytest.raises(ValueError, match="base"):
            infotheory.normalized_mutual_information([[1.0]], base=base)

    def test_rejects_negative_joint_probabilities(self) -> None:
        joint = [[0.7, -0.2], [0.2, 0.3]]
        with pytest.raises(ValueError, match="non-negative"):
            infotheory.conditional_entropy(joint)
        with pytest.raises(ValueError, match="non-negative"):
            infotheory.mutual_information(joint)
        with pytest.raises(ValueError, match="non-negative"):
            infotheory.normalized_mutual_information(joint)

    def test_probabilities_reject_2d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            infotheory.entropy([[0.5, 0.5]])

    def test_joint_entropy_rejects_flat_and_empty(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            infotheory.joint_entropy([0.5, 0.5])
        with pytest.raises(ValueError, match="2-D"):
            infotheory.joint_entropy(np.empty((0, 2)))

    def test_conditional_entropy_rejects_flat(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            infotheory.conditional_entropy([0.25] * 4)

    def test_mutual_information_rejects_flat(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            infotheory.mutual_information([0.25] * 4)

    def test_js_divergence_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            infotheory.js_divergence([1.0], [0.5, 0.5])

    def test_cross_entropy_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            infotheory.cross_entropy([1.0], [0.5, 0.5])

    def test_normalized_nmi_zero_denominator(self) -> None:
        joint = [[0.0, 0.0], [0.0, 1.0]]
        assert infotheory.normalized_mutual_information(joint) == 0.0
