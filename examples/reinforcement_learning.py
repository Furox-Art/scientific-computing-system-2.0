"""Case study 12: Reinforcement learning on bandits and a grid world.

UCB1 and epsilon-greedy are compared on the same three-armed Bernoulli
bandit, then tabular Q-learning solves a 6x6 grid world; the learned
greedy path is replayed deterministically.
"""

from __future__ import annotations

import numpy as np

import cds2


def replay_greedy(env: cds2.rl.GridWorld, q_values: np.ndarray, max_steps: int = 40) -> list[int]:
    state = env.reset()
    path = [state]
    for _ in range(max_steps):
        action = int(np.argmax(q_values[state]))
        state, _reward, done = env.step(state, action)
        path.append(state)
        if done:
            break
    return path


def main() -> None:
    arms = (0.25, 0.55, 0.80)
    print("== Bandit policies ==")
    for label, policy in (
        ("UCB1", cds2.rl.ucb1),
        ("epsilon-greedy", lambda b: cds2.rl.epsilon_greedy(b, episodes=3000, epsilon=0.10)),
    ):
        result = policy(cds2.rl.Bandit(arms, seed=31))
        best = int(np.argmax(result.estimates))
        regret = float(result.regret[-1])
        print(f"{label:15s} best arm={best}  pulls={result.counts.tolist()}  regret={regret:.1f}")

    env = cds2.rl.GridWorld(rows=6, columns=6)
    q_values, returns = cds2.rl.q_learn(
        env, episodes=1200, alpha=0.5, gamma=0.95, epsilon=0.15, seed=17
    )
    print("\n== Q-learning grid world ==")
    print(f"mean return, first 100 eps : {returns[:100].mean():.3f}")
    print(f"mean return, last 100 eps  : {returns[-100:].mean():.3f}")
    path = replay_greedy(env, q_values)
    print(f"greedy path length         : {len(path) - 1} steps (optimal is 10)")
    print(f"path                       : {path}")


if __name__ == "__main__":
    main()
