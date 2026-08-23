"""Tests for cds2.rl."""

import numpy as np
import pytest

from cds2 import rl


class TestBandit:
    def test_pull_returns_reward(self) -> None:
        bandit = rl.Bandit([1.0, 0.0], seed=1)
        assert bandit.pull(0) == 1
        assert bandit.pull(1) == 0

    def test_optimal_reward(self) -> None:
        assert rl.Bandit([0.3, 0.9]).optimal_reward == pytest.approx(0.9)

    def test_invalid_arms(self) -> None:
        with pytest.raises(ValueError, match="probabilities"):
            rl.Bandit([1.5])

    def test_arm_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            rl.Bandit([0.5, 0.5]).pull(7)


class TestEpsilonGreedy:
    def test_converges_to_best_arm(self) -> None:
        bandit = rl.Bandit([0.2, 0.8, 0.4], seed=5)
        result = rl.epsilon_greedy(bandit, episodes=3000, epsilon=0.1)
        assert int(np.argmax(result.estimates)) == 1
        assert result.counts[1] == result.counts.max()

    def test_total_reward_consistent_with_regret(self) -> None:
        bandit = rl.Bandit([0.5, 0.5], seed=2)
        result = rl.epsilon_greedy(bandit, episodes=500)
        expected_regret = bandit.optimal_reward * 500 - result.total_reward
        assert result.regret[-1] == pytest.approx(expected_regret)

    def test_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            rl.epsilon_greedy(rl.Bandit([0.1, 0.2]), epsilon=2.0)


class TestUCB1:
    def test_explores_each_arm_once(self) -> None:
        bandit = rl.Bandit([0.1, 0.9], seed=3)
        result = rl.ucb1(bandit, episodes=100)
        assert np.all(result.counts[:2] >= 1)

    def test_prefers_best_arm(self) -> None:
        bandit = rl.Bandit([0.1, 0.9], seed=4)
        result = rl.ucb1(bandit, episodes=2000)
        assert result.counts[1] > 10 * result.counts[0]

    def test_invalid_episodes(self) -> None:
        with pytest.raises(ValueError, match="episodes"):
            rl.ucb1(rl.Bandit([0.1, 0.2]), episodes=0)


class TestGridWorld:
    def test_goal_reached_right_down(self) -> None:
        env = rl.GridWorld(rows=3, columns=3)
        state = env.reset()
        for action in (1, 1, 3, 3):
            state, reward, done = env.step(state, action)
        assert done and reward == 1.0 and state == 8

    def test_walls_clamp(self) -> None:
        env = rl.GridWorld(rows=2, columns=2)
        next_state, reward, done = env.step(env.reset(), action=0)
        assert next_state == 0 and reward == 0.0 and not done

    def test_invalid_action(self) -> None:
        with pytest.raises(ValueError, match="unknown action"):
            rl.GridWorld().step(0, 99)

    def test_min_size_enforced(self) -> None:
        with pytest.raises(ValueError, match="2x2"):
            rl.GridWorld(rows=1)


class TestQLearning:
    def test_learns_optimal_policy(self) -> None:
        env = rl.GridWorld(rows=4, columns=4)
        q_values, returns = rl.q_learn(env, episodes=600, alpha=0.6, gamma=0.95, seed=12)
        assert returns[-50:].mean() > 0.9
        best_actions = np.argmax(q_values, axis=1)
        assert set(np.unique(best_actions)).issubset({0, 1, 2, 3})

    def test_deterministic_with_seed(self) -> None:
        env_a, env_b = rl.GridWorld(3, 3), rl.GridWorld(3, 3)
        values_a, _ = rl.q_learn(env_a, episodes=80, seed=7)
        values_b, _ = rl.q_learn(env_b, episodes=80, seed=7)
        np.testing.assert_allclose(values_a, values_b)

    def test_rejects_non_environment(self) -> None:
        with pytest.raises(TypeError, match="reset"):
            rl.q_learn(object())

    def test_rejects_bad_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            rl.q_learn(rl.GridWorld(), alpha=0.0)

    def test_missing_n_states_raises(self) -> None:
        class Minimal:
            actions = [0, 1]

            def reset(self) -> int:
                return 0

            def step(self, state: int, action: int) -> tuple[int, float, bool]:
                return state, 0.0, True

        with pytest.raises(TypeError, match="n_states"):
            rl.q_learn(Minimal())


class TestQTable:
    def test_greedy_tie_breaking_is_valid(self) -> None:
        rng = np.random.default_rng(0)
        table = rl.QTable(n_states=2, n_actions=3)
        choices = {table.act_greedy(0, rng) for _ in range(30)}
        assert choices.issubset({0, 1, 2})

    def test_init_value(self) -> None:
        table = rl.QTable(n_states=2, n_actions=2, init_value=0.5)
        assert table.values[1, 1] == 0.5

    def test_invalid_sizes(self) -> None:
        with pytest.raises(ValueError, match="state"):
            rl.QTable(n_states=0, n_actions=1)


class TestCoverageEdges:
    def test_q_learn_rejects_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            rl.q_learn(rl.GridWorld(), epsilon=1.5)
