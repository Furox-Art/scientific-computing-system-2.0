"""Mutual information rejects invalid probability tables and ordinal codes do not overflow."""

from __future__ import annotations

import math

import pytest

from cds2.infotheory import (
    mutual_information,
    normalized_mutual_information,
    permutation_entropy,
)


@pytest.mark.parametrize(
    "invalid",
    [
        [[-0.5, 1.0], [0.0, 0.5]],
        [[-0.2, 0.7], [0.2, 0.3]],
        [[math.inf, -math.inf], [0.5, 0.5]],
        [[math.nan, 0.5], [0.0, 0.5]],
    ],
)
@pytest.mark.parametrize("metric", [mutual_information, normalized_mutual_information])
def test_invalid_joint_tables_cannot_produce_mutual_information(metric, invalid) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        metric(invalid)


@pytest.mark.parametrize("base", [-1.0, 0.0, 1.0, math.inf, math.nan])
def test_mutual_information_requires_finite_valid_logarithm_base(base: float) -> None:
    with pytest.raises(ValueError, match="base"):
        mutual_information([[0.5, 0.0], [0.0, 0.5]], base=base)


def test_valid_perfectly_associated_bits_still_have_unit_information() -> None:
    joint = [[0.5, 0.0], [0.0, 0.5]]
    assert mutual_information(joint) == pytest.approx(1.0)
    assert normalized_mutual_information(joint) == pytest.approx(1.0)


def test_degenerate_distribution_preserves_defined_zero_nmi() -> None:
    assert normalized_mutual_information([[1.0, 0.0], [0.0, 0.0]]) == 0.0


@pytest.mark.parametrize("order", [17, 18, 21, 24])
def test_high_order_permutation_entropy_uses_exact_ordinal_tuples(order: int) -> None:
    # Two overlapping windows have distinct ordinal patterns.
    signal = list(range(order)) + [-1]
    assert permutation_entropy(signal, order=order) == pytest.approx(1.0)


@pytest.mark.parametrize("order", [17, 20, 24])
def test_high_order_monotonic_signal_has_zero_permutation_entropy(order: int) -> None:
    assert permutation_entropy(list(range(order + 2)), order=order) == 0.0
