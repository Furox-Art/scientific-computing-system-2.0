"""Tests for cds2.game_theory."""

from types import SimpleNamespace

import numpy as np
import pytest

from cds2 import game_theory as gt

DOMINANCE_ROW = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 2.0, 1.0]]
DOMINANCE_COL = [[1.0, 0.0, 4.0], [0.0, 1.0, 4.0], [0.0, 0.0, 4.0]]


def _fake_linprog_result(success: bool, x: object, marginals: object) -> SimpleNamespace:
    return SimpleNamespace(
        success=success,
        x=x,
        fun=2.0,
        ineqlin=SimpleNamespace(marginals=marginals),
    )


class TestStrictlyDominatedActions:
    matrix = [[2.0, 1.0], [1.0, 0.0]]

    def test_dominated_row_found(self) -> None:
        np.testing.assert_array_equal(gt.strictly_dominated_actions(self.matrix), [1])

    def test_dominated_column_found(self) -> None:
        result = gt.strictly_dominated_actions(self.matrix, player="col")
        np.testing.assert_array_equal(result, [1])

    def test_no_dominance_yields_empty(self) -> None:
        result = gt.strictly_dominated_actions([[1.0, 0.0], [0.0, 1.0]])
        assert result.size == 0

    def test_partial_advantage_is_not_strict(self) -> None:
        result = gt.strictly_dominated_actions([[2.0, 0.0], [1.0, 1.0]])
        assert result.size == 0

    def test_invalid_player_raises(self) -> None:
        with pytest.raises(ValueError, match="player must be row or col"):
            gt.strictly_dominated_actions(self.matrix, player="diagonal")

    def test_invalid_matrices_raise(self) -> None:
        with pytest.raises(ValueError, match="non-empty and 2-D"):
            gt.strictly_dominated_actions([1.0, 2.0])
        with pytest.raises(ValueError, match="non-empty and 2-D"):
            gt.strictly_dominated_actions([[]])


class TestIteratedElimination:
    def test_survivors_after_two_rounds(self) -> None:
        result = gt.iterated_elimination(DOMINANCE_ROW, DOMINANCE_COL)
        np.testing.assert_array_equal(result.row_actions, [2])
        np.testing.assert_array_equal(result.col_actions, [2])
        assert result.rounds == 2

    def test_max_rounds_caps_elimination(self) -> None:
        result = gt.iterated_elimination(DOMINANCE_ROW, DOMINANCE_COL, max_rounds=1)
        np.testing.assert_array_equal(result.row_actions, [0, 1, 2])
        np.testing.assert_array_equal(result.col_actions, [2])
        assert result.rounds == 1

    def test_nothing_to_eliminate(self) -> None:
        identity = [[1.0, 0.0], [0.0, 1.0]]
        result = gt.iterated_elimination(identity, identity)
        np.testing.assert_array_equal(result.row_actions, [0, 1])
        np.testing.assert_array_equal(result.col_actions, [0, 1])
        assert result.rounds == 0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="share a shape"):
            gt.iterated_elimination([[1.0, 0.0], [0.0, 1.0]], [[1.0]])

    def test_max_rounds_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_rounds"):
            gt.iterated_elimination(DOMINANCE_ROW, DOMINANCE_COL, max_rounds=0)


class TestPureNashEquilibria:
    def test_prisoners_dilemma_unique_equilibrium(self) -> None:
        equilibria = gt.pure_nash_equilibria(
            [[3.0, 0.0], [5.0, 1.0]],
            [[3.0, 5.0], [0.0, 1.0]],
        )
        assert equilibria == [(1, 1)]

    def test_coordination_game_two_equilibria(self) -> None:
        game = [[2.0, 0.0], [0.0, 1.0]]
        assert gt.pure_nash_equilibria(game, game) == [(0, 0), (1, 1)]

    def test_ties_count_as_best_responses(self) -> None:
        zeros = [[0.0, 0.0], [0.0, 0.0]]
        result = gt.pure_nash_equilibria(zeros, zeros)
        assert sorted(result) == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="share a shape"):
            gt.pure_nash_equilibria([[1.0]], [[1.0, 0.0], [0.0, 1.0]])


class TestZeroSumMixed:
    def test_matching_pennies_value_zero_uniform(self) -> None:
        equilibrium = gt.zero_sum_mixed([[1.0, -1.0], [-1.0, 1.0]])
        assert equilibrium.value == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(equilibrium.row_strategy, [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(equilibrium.col_strategy, [0.5, 0.5], atol=1e-6)

    def test_biased_single_cell(self) -> None:
        equilibrium = gt.zero_sum_mixed([[2.0]])
        assert equilibrium.value == pytest.approx(2.0)
        np.testing.assert_allclose(equilibrium.row_strategy, [1.0], atol=1e-9)
        np.testing.assert_allclose(equilibrium.col_strategy, [1.0], atol=1e-9)

    def test_rectangular_game_and_probabilities_sum_to_one(self) -> None:
        equilibrium = gt.zero_sum_mixed([[3.0, 1.0, 2.0], [2.0, 1.0, 4.0]])
        assert equilibrium.value == pytest.approx(1.0)
        assert equilibrium.row_strategy.sum() == pytest.approx(1.0)
        assert equilibrium.col_strategy.sum() == pytest.approx(1.0)
        assert np.all(equilibrium.row_strategy >= 0.0)

    def test_failed_lp_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            gt,
            "linprog",
            lambda *args, **kwargs: _fake_linprog_result(False, None, [1.0]),
        )
        with pytest.raises(ValueError, match="no finite value"):
            gt.zero_sum_mixed([[1.0]])

    def test_degenerate_dual_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            gt,
            "linprog",
            lambda *args, **kwargs: _fake_linprog_result(True, [1.0, -2.0], [0.0]),
        )
        with pytest.raises(ValueError, match="no finite value"):
            gt.zero_sum_mixed([[2.0]])


class TestExpectedPayoff:
    def test_weighted_average(self) -> None:
        value = gt.expected_payoff([[2.0, 0.0], [0.0, 4.0]], [0.25, 0.75], [0.5, 0.5])
        assert value == pytest.approx(1.75)

    def test_matching_pennies_averages_to_zero(self) -> None:
        game = [[1.0, -1.0], [-1.0, 1.0]]
        value = gt.expected_payoff(game, [0.5, 0.5], [0.5, 0.5])
        assert value == pytest.approx(0.0)

    def test_non_probability_vector_rejected(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            gt.expected_payoff([[1.0]], [0.5], [1.0])
        with pytest.raises(ValueError, match="probability"):
            gt.expected_payoff([[1.0]], [1.0], [0.3])

    def test_dimension_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="match"):
            gt.expected_payoff([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0], [1.0, 0.0, 0.0])


class TestPlayIteratedPD:
    def test_tit_for_tat_vs_always_defect_hand_calculation(self) -> None:
        result = gt.play_iterated_pd("tit_for_tat", "always_defect", rounds=3)
        assert result.final_a == pytest.approx(5.0)
        assert result.final_b == pytest.approx(10.0)
        np.testing.assert_array_equal(result.moves_a, [1, 0, 0])
        np.testing.assert_array_equal(result.moves_b, [0, 0, 0])

    def test_swapped_roles_mirror_scores(self) -> None:
        result = gt.play_iterated_pd("always_defect", "tit_for_tat", rounds=3)
        assert result.final_a == pytest.approx(10.0)
        assert result.final_b == pytest.approx(5.0)
        np.testing.assert_array_equal(result.moves_a, [0, 0, 0])
        np.testing.assert_array_equal(result.moves_b, [1, 0, 0])

    def test_grim_trigger_punishes_after_first_defection(self) -> None:
        result = gt.play_iterated_pd("grim_trigger", "always_defect", rounds=4)
        np.testing.assert_array_equal(result.moves_a, [1, 0, 0, 0])
        assert result.final_a == pytest.approx(7.5)
        assert result.final_b == pytest.approx(12.5)

    def test_mutual_cooperation_pays_reward(self) -> None:
        result = gt.play_iterated_pd("always_cooperate", "always_cooperate")
        assert result.final_a == pytest.approx(300.0)
        assert result.final_b == pytest.approx(300.0)
        assert np.all(result.moves_a == 1)

    def test_mutual_defection_pays_punishment(self) -> None:
        result = gt.play_iterated_pd("always_defect", "always_defect")
        assert result.final_a == pytest.approx(250.0)
        assert result.final_b == pytest.approx(250.0)
        assert np.all(result.moves_a == 0)

    def test_custom_payoffs_applied_per_round(self) -> None:
        result = gt.play_iterated_pd(
            "always_cooperate",
            "always_defect",
            rounds=2,
            temptation=4.0,
            sucker=-1.0,
            punishment=0.0,
            reward=1.0,
        )
        assert result.final_a == pytest.approx(-2.0)
        assert result.final_b == pytest.approx(8.0)

    def test_random_reproducible_with_seed(self) -> None:
        first = gt.play_iterated_pd("random", "random", rounds=60, seed=123)
        second = gt.play_iterated_pd("random", "random", rounds=60, seed=123)
        other = gt.play_iterated_pd("random", "random", rounds=60, seed=124)
        np.testing.assert_array_equal(first.moves_a, second.moves_a)
        assert first.final_a == second.final_a
        assert not np.array_equal(first.moves_a, other.moves_a)
        assert set(np.unique(first.moves_a)).issubset({0, 1})

    def test_unknown_strategy_message_lists_options(self) -> None:
        with pytest.raises(ValueError, match="unknown strategy 'prober'") as excinfo:
            gt.play_iterated_pd("prober", "tit_for_tat")
        message = str(excinfo.value)
        for option in (
            "always_cooperate",
            "always_defect",
            "grim_trigger",
            "random",
            "tit_for_tat",
        ):
            assert option in message
