"""Reference-value tests for the split-R-hat and ESS MCMC diagnostics.

The estimators follow Gelman et al., Bayesian Data Analysis (3rd ed.),
chapter 11: split-R-hat from the between/within half-chain variances, and
ESS from Geyer's initial monotone positive sequence of paired
autocorrelations. Small cases are checked against a naive loop
implementation written independently of the FFT-based production code,
and large seeded cases against known theoretical limits (iid chains,
AR(1) autocorrelation).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cds2.montecarlo import effective_sample_size, split_rhat


def _reference_split_rhat(chains: np.ndarray) -> float:
    """Naive BDA3 split-R-hat with explicit loops (no vectorised shortcuts)."""
    m, n = chains.shape
    if n % 2:
        chains = chains[:, :-1]
        n -= 1
    half = n // 2
    halves = [chains[j, :half] for j in range(m)] + [chains[j, half:] for j in range(m)]
    means = [float(np.mean(h)) for h in halves]
    variances = [float(np.var(h, ddof=1)) for h in halves]
    within = sum(variances) / len(variances)
    grand = sum(means) / len(means)
    between = half * sum((mu - grand) ** 2 for mu in means) / (len(means) - 1)
    var_plus = (half - 1) / half * within + between / half
    return math.sqrt(var_plus / within)


def _reference_ess(chains: np.ndarray) -> float:
    """Naive BDA3 ESS: direct autocovariance sums, same truncation rule."""
    m, n = chains.shape
    means = [float(np.mean(chains[j])) for j in range(m)]
    within = sum(float(np.var(chains[j], ddof=1)) for j in range(m)) / m
    grand = sum(means) / m
    between = n * sum((mu - grand) ** 2 for mu in means) / (m - 1) if m > 1 else 0.0
    var_plus = (n - 1) / n * within + between / n
    rho = [1.0]
    for lag in range(1, n):
        acov = 0.0
        for j in range(m):
            centered = chains[j] - means[j]
            acov += float(np.sum(centered[: n - lag] * centered[lag:])) / n
        acov /= m
        rho.append(1.0 - (within - acov) / var_plus)
    total = 0.0
    monotone = math.inf
    for lag in range(n // 2):
        pair = rho[2 * lag] + rho[2 * lag + 1]
        if pair <= 0.0:
            break
        monotone = min(monotone, pair)
        total += monotone
    tau = max(1.0, 2.0 * total - 1.0)
    return m * n / tau


class TestSplitRhat:
    def test_hand_computed_two_chain_value(self) -> None:
        # halves [1,2],[3,4],[2,3],[4,5]: W=0.5, B=2*var([1.5,3.5,2.5,4.5])=10/3
        # var_plus = 0.25 + 5/3 = 23/12  ->  R-hat = sqrt((23/12)/0.5) = sqrt(23/6)
        chains = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
        assert split_rhat(chains) == pytest.approx(math.sqrt(23.0 / 6.0), rel=1e-12)

    def test_matches_naive_reference_on_seeded_chains(self) -> None:
        rng = np.random.default_rng(11)
        chains = rng.normal(size=(4, 37))  # odd length exercises the drop rule
        assert split_rhat(chains) == pytest.approx(_reference_split_rhat(chains), rel=1e-12)

    def test_iid_chains_are_near_one(self) -> None:
        rng = np.random.default_rng(42)
        chains = rng.normal(size=(4, 4000))
        value = split_rhat(chains)
        assert 0.98 < value < 1.02

    def test_shifted_chains_flag_non_convergence(self) -> None:
        rng = np.random.default_rng(7)
        chains = rng.normal(size=(4, 2000)) + np.array([0.0, 10.0, 20.0, 30.0])[:, None]
        assert split_rhat(chains) > 1.5

    def test_drifting_chain_detected_by_splitting(self) -> None:
        # One stationary chain vs one with a strong within-chain drift: the
        # split halves disagree, so split-R-hat must rise above the unsplit one.
        rng = np.random.default_rng(3)
        stationary = rng.normal(size=(2, 2000))
        drifting = stationary.copy()
        drifting[1] += np.linspace(0.0, 8.0, 2000)
        assert split_rhat(drifting) > split_rhat(stationary) + 0.2

    def test_odd_length_drops_last_draw(self) -> None:
        rng = np.random.default_rng(5)
        chains = rng.normal(size=(3, 101))
        assert split_rhat(chains) == pytest.approx(split_rhat(chains[:, :100]), rel=1e-12)

    def test_list_of_chains_accepted(self) -> None:
        chains = [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]
        assert split_rhat(chains) == pytest.approx(math.sqrt(23.0 / 6.0), rel=1e-12)

    def test_single_chain_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two chains"):
            split_rhat(np.zeros((1, 10)))

    def test_constant_chains_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero within-chain variance"):
            split_rhat(np.ones((2, 10)))

    def test_ragged_chains_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            split_rhat([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])

    def test_three_dimensional_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            split_rhat(np.zeros((2, 2, 8)))

    def test_short_chains_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least four draws"):
            split_rhat(np.zeros((2, 3)))

    def test_empty_chain_set_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least four draws"):
            split_rhat(np.empty((0, 8)))

    def test_non_finite_draws_rejected(self) -> None:
        chains = np.ones((2, 8))
        chains[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            split_rhat(chains)


class TestEffectiveSampleSize:
    def test_matches_naive_reference_on_seeded_chains(self) -> None:
        rng = np.random.default_rng(13)
        chains = rng.normal(size=(3, 40))
        assert effective_sample_size(chains) == pytest.approx(_reference_ess(chains), rel=1e-10)

    def test_single_chain_matches_naive_reference(self) -> None:
        rng = np.random.default_rng(17)
        chain = rng.normal(size=64)
        assert effective_sample_size(chain) == pytest.approx(
            _reference_ess(chain[None, :]), rel=1e-10
        )

    def test_iid_chains_keep_most_of_the_budget(self) -> None:
        rng = np.random.default_rng(42)
        chains = rng.normal(size=(4, 4000))
        total = chains.size
        value = effective_sample_size(chains)
        assert 0.8 * total <= value <= total

    def test_ar1_matches_theoretical_autocorrelation_time(self) -> None:
        # Stationary AR(1) with rho=0.5: tau = (1+rho)/(1-rho) = 3, so the
        # ESS of m*n draws is about m*n/3.
        rho = 0.5
        rng = np.random.default_rng(23)
        m, n = 4, 8000
        chains = np.empty((m, n))
        innovation_scale = math.sqrt(1.0 - rho * rho)
        for j in range(m):
            value = float(rng.normal())
            for t in range(n):
                value = rho * value + innovation_scale * float(rng.normal())
                chains[j, t] = value
        expected = m * n * (1.0 - rho) / (1.0 + rho)
        assert effective_sample_size(chains) == pytest.approx(expected, rel=0.15)
        assert split_rhat(chains) < 1.05

    def test_anticorrelated_chain_is_capped_at_the_draw_count(self) -> None:
        # Perfect alternation drives the first rho pair non-positive; the
        # documented cap keeps ESS at the total draw count instead of negative.
        chain = np.array([1.0, -1.0] * 8)
        assert effective_sample_size(chain) == pytest.approx(float(chain.size))

    def test_shifted_chains_do_not_inflate_ess(self) -> None:
        rng = np.random.default_rng(19)
        chains = rng.normal(size=(4, 1000)) + np.array([0.0, 10.0, 20.0, 30.0])[:, None]
        assert effective_sample_size(chains) <= chains.size

    def test_constant_chains_rejected(self) -> None:
        with pytest.raises(ValueError, match="constant chains"):
            effective_sample_size(np.ones((2, 10)))

    def test_ragged_chains_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            effective_sample_size([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0]])

    def test_short_chains_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least four draws"):
            effective_sample_size([1.0, 2.0, 3.0])

    def test_non_finite_draws_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            effective_sample_size([1.0, 2.0, np.inf, 4.0])

    def test_seeded_reproducible(self) -> None:
        rng = np.random.default_rng(29)
        chains = rng.normal(size=(3, 200))
        assert effective_sample_size(chains) == effective_sample_size(chains.copy())
        assert split_rhat(chains) == split_rhat(chains.copy())
