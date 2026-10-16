"""KL and cross-entropy become infinite when Q misses positive P support."""

from __future__ import annotations

import math

import pytest

from cds2.infotheory import cross_entropy, js_divergence, kl_divergence


@pytest.mark.parametrize("metric", [kl_divergence, cross_entropy])
@pytest.mark.parametrize(
    ("p", "q"),
    [
        ([1.0, 0.0], [0.0, 1.0]),
        ([0.2, 0.8], [1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    ],
)
def test_positive_p_mass_outside_q_support_has_infinite_loss(metric, p, q) -> None:
    assert math.isinf(metric(p, q))
    assert metric(p, q) > 0


@pytest.mark.parametrize("metric", [kl_divergence, cross_entropy])
def test_zero_p_mass_does_not_require_matching_q_support(metric) -> None:
    # The convention 0 * log(0 / 0) = 0 is mathematically well defined.
    assert metric([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_js_divergence_remains_bounded_for_disjoint_support() -> None:
    assert js_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_regular_kl_divergence_is_unaffected() -> None:
    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
