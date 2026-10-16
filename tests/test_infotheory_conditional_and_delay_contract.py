"""Independent numerical reference checks for entropy and delayed ordinal patterns."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cds2.infotheory import conditional_entropy, permutation_entropy


@pytest.mark.parametrize(
    "joint",
    [
        [[-0.2, 0.7], [0.5, 0.0]],
        [[-0.5, 1.0], [0.0, 0.5]],
        [[0.5, 0.3], [0.1, 0.3]],
        [[math.nan, 0.5], [0.0, 0.5]],
        [[math.inf, 0.0], [0.0, 0.0]],
    ],
)
def test_conditional_entropy_rejects_invalid_joint_probabilities(joint: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        conditional_entropy(joint)


@pytest.mark.parametrize("marginal", [[0.25, 0.75], [0.0, 1.0], [0.75, 0.25]])
def test_conditional_entropy_refuses_mismatched_marginals(marginal: list[float]) -> None:
    with pytest.raises(ValueError, match="marginal probabilities must match"):
        conditional_entropy([[0.5, 0.0], [0.0, 0.5]], marginal)


def test_conditional_entropy_preserves_valid_marginals_and_zero_rows() -> None:
    joint = [[0.25, 0.25], [0.25, 0.25]]
    assert conditional_entropy(joint, [0.5, 0.5]) == pytest.approx(1.0)
    assert conditional_entropy([[0.0, 0.0], [1.0, 0.0]], [0.0, 1.0]) == 0.0


@pytest.mark.parametrize("base", [math.inf, math.nan, 0.0, 1.0])
def test_conditional_entropy_rejects_invalid_log_base(base: float) -> None:
    with pytest.raises(ValueError, match="base"):
        conditional_entropy([[0.5, 0.5]], base=base)


def test_delay_two_uses_every_overlapping_ordinal_pattern() -> None:
    signal = [0.0, 4.0, 1.0, 3.0, 2.0, 5.0]
    # There are four valid length-3 windows: 3 rising and 1 falling.
    expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert permutation_entropy(signal, order=2, delay=2) == pytest.approx(expected)


def test_delay_one_keeps_previous_reference_result() -> None:
    assert permutation_entropy([0.0, 1.0, 2.0, 3.0], order=2, delay=1) == 0.0


@pytest.mark.parametrize("order,delay", [(2.5, 1), (True, 1), (2, 1.5), (2, False)])
def test_permutation_entropy_requires_integral_order_and_delay(
    order: object, delay: object
) -> None:
    with pytest.raises(ValueError, match="integer"):
        permutation_entropy([0.0, 1.0, 2.0, 3.0], order=order, delay=delay)  # type: ignore[arg-type]


@pytest.mark.parametrize("signal", [[[1.0, 2.0], [3.0, 4.0]], [0.0, math.nan, 1.0]])
def test_permutation_entropy_refuses_nonfinite_and_nonvector_input(signal: object) -> None:
    with pytest.raises(ValueError, match="finite 1-D"):
        permutation_entropy(signal)  # type: ignore[arg-type]


def test_delay_two_reference_matches_explicit_numpy_windows() -> None:
    signal = np.array([0.0, 4.0, 1.0, 3.0, 2.0, 5.0])
    embeddings = np.lib.stride_tricks.sliding_window_view(signal, 3)[:, ::2]
    patterns = np.argsort(embeddings, axis=1, kind="stable")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    p = counts / counts.sum()
    expected = -float(np.sum(p * np.log2(p)))
    assert permutation_entropy(signal, order=2, delay=2) == pytest.approx(expected)
