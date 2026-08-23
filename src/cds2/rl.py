"""Reinforcement learning: multi-armed bandits, Q-learning and a tiny grid world."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BanditResult",
    "Bandit",
    "epsilon_greedy",
    "ucb1",
    "QTable",
    "q_learn",
    "GridWorld",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BanditResult:
    """Statistics from a bandit run."""

    estimates: FloatArray
    counts: IntArray
    total_reward: float
    regret: FloatArray
    choices: IntArray


class Bandit:
    """Bernoulli multi-armed bandit with hidden success probabilities."""

    def __init__(self, probabilities: Sequence[float], seed: int | None = None) -> None:
        arms = np.asarray(probabilities, dtype=float)
        if arms.ndim != 1 or arms.size < 2 or np.any(arms < 0) or np.any(arms > 1):
            msg = "probabilities must be a 1-D sequence of values in [0, 1]"
            raise ValueError(msg)
        self.probabilities = arms
        self._rng = np.random.default_rng(seed)

    def pull(self, arm: int) -> int:
        """Draw one Bernoulli reward from ``arm``."""
        if not 0 <= arm < self.probabilities.size:
            msg = f"arm index out of range: {arm}"
            raise ValueError(msg)
        return int(self._rng.random() < self.probabilities[arm])

    @property
    def optimal_reward(self) -> float:
        return float(self.probabilities.max())


def epsilon_greedy(
    bandit: Bandit,
    episodes: int = 2_000,
    epsilon: float = 0.1,
    decay: float = 1.0,
) -> BanditResult:
    """Epsilon-greedy value tracking with an exponentially decaying rate."""
    if episodes < 1 or not 0.0 <= epsilon <= 1.0 or not 0.0 < decay <= 1.0:
        msg = "episodes >= 1 and epsilon/decay in [0, 1] required"
        raise ValueError(msg)
    n_arms = bandit.probabilities.size
    counts = np.zeros(n_arms, dtype=np.int64)
    estimates = np.zeros(n_arms)
    total = 0.0
    regret = np.empty(episodes)
    choices = np.empty(episodes, dtype=np.int64)
    current_epsilon = epsilon

    for step in range(episodes):
        if counts.min() == 0:
            arm = int(np.argmin(counts))
        elif bandit._rng.random() < current_epsilon:
            arm = int(bandit._rng.integers(n_arms))
        else:
            arm = int(np.argmax(estimates))
        reward = float(bandit.pull(arm))
        counts[arm] += 1
        estimates[arm] += (reward - estimates[arm]) / counts[arm]
        total += reward
        regret[step] = bandit.optimal_reward * (step + 1) - total
        choices[step] = arm
        current_epsilon *= decay
    return BanditResult(
        estimates=estimates, counts=counts, total_reward=total, regret=regret, choices=choices
    )


def ucb1(bandit: Bandit, episodes: int = 2_000) -> BanditResult:
    """Upper-confidence-bound (UCB1) bandit policy."""
    if episodes < 1:
        msg = "episodes must be at least 1"
        raise ValueError(msg)
    n_arms = bandit.probabilities.size
    counts = np.zeros(n_arms, dtype=np.int64)
    estimates = np.zeros(n_arms)
    total = 0.0
    regret = np.empty(episodes)
    choices = np.empty(episodes, dtype=np.int64)

    for step in range(episodes):
        if counts.min() == 0:
            arm = int(np.argmin(counts))
        else:
            bonus = np.sqrt(2.0 * np.log(step + 1) / counts)
            arm = int(np.argmax(estimates + bonus))
        reward = float(bandit.pull(arm))
        counts[arm] += 1
        estimates[arm] += (reward - estimates[arm]) / counts[arm]
        total += reward
        regret[step] = bandit.optimal_reward * (step + 1) - total
        choices[step] = arm
    return BanditResult(
        estimates=estimates, counts=counts, total_reward=total, regret=regret, choices=choices
    )


class GridWorld:
    """Deterministic 4-connected grid world with one goal state.

    States are flattened row-major cells; the agent starts in the top-left
    corner and receives reward only on reaching the goal.
    """

    actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, rows: int = 4, columns: int = 4) -> None:
        if rows < 2 or columns < 2:
            msg = "grid needs at least 2x2 cells"
            raise ValueError(msg)
        self.rows = rows
        self.columns = columns
        self.goal = rows * columns - 1
        self.start = 0
        self.reset()

    def reset(self) -> int:
        """Move the agent back to the start cell."""
        self.state = self.start
        return self.state

    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        """Apply ``action``; returns ``(next_state, reward, done)``."""
        if action not in self.actions:
            msg = f"unknown action: {action}"
            raise ValueError(msg)
        row, column = divmod(state, self.columns)
        delta_row, delta_column = self.actions[action]
        row = min(max(row + delta_row, 0), self.rows - 1)
        column = min(max(column + delta_column, 0), self.columns - 1)
        next_state = row * self.columns + column
        done = next_state == self.goal
        return next_state, (1.0 if done else 0.0), done

    @property
    def n_states(self) -> int:
        return self.rows * self.columns


class QTable:
    """Tabular action-value store with optimistic initialization."""

    def __init__(self, n_states: int, n_actions: int, init_value: float = 0.0) -> None:
        if n_states < 1 or n_actions < 1:
            msg = "need at least one state and action"
            raise ValueError(msg)
        self.values = np.full((n_states, n_actions), float(init_value))

    def act_greedy(self, state: int, rng: np.random.Generator) -> int:
        """Random tie-broken greedy action for ``state``."""
        row = self.values[state]
        winners = np.flatnonzero(row == row.max())
        return int(rng.choice(winners))


def q_learn(
    environment: object,
    episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    max_steps: int | None = None,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Tabular Q-learning on any ``reset``/``step`` environment.

    Returns ``(q_values, episode_returns)``; ties are broken randomly and
    exploration uses uniform epsilon-greedy sampling.
    """
    if not hasattr(environment, "reset") or not hasattr(environment, "step"):
        msg = "environment must provide reset() and step()"
        raise TypeError(msg)
    if episodes < 1 or not 0.0 < alpha <= 1.0 or not 0.0 <= gamma <= 1.0:
        msg = "episodes >= 1 and valid alpha/gamma required"
        raise ValueError(msg)
    if not 0.0 <= epsilon <= 1.0:
        msg = "epsilon must lie in [0, 1]"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    n_states = getattr(environment, "n_states", None)
    if n_states is None:
        msg = "environment must expose n_states"
        raise TypeError(msg)
    q_table = QTable(n_states=n_states, n_actions=len(getattr(environment, "actions", [0, 1])))
    limit = max_steps or 4 * int(np.sqrt(n_states)) + 16
    returns = np.empty(episodes)

    for episode in range(episodes):
        state = environment.reset()
        total = 0.0
        for _ in range(limit):
            explore = rng.random() < epsilon
            action = (
                int(rng.integers(q_table.values.shape[1]))
                if explore
                else q_table.act_greedy(state, rng)
            )
            next_state, reward, done = environment.step(state, action)
            target = reward + gamma * q_table.values[next_state].max() * (1.0 - float(done))
            q_table.values[state, action] += alpha * (target - q_table.values[state, action])
            state = next_state
            total += reward
            if done:
                break
        returns[episode] = total
    return q_table.values, returns
