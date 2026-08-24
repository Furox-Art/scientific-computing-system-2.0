"""Case study 20: Game theory tournament.

Prisoner's dilemma strategies face each other in a round-robin; pure Nash
equilibria of the one-shot game are enumerated and a zero-sum payoff matrix
is solved for its mixed minimax strategy.
"""

from __future__ import annotations

import numpy as np

import cds2

STRATEGIES = ["tit_for_tat", "always_cooperate", "always_defect", "grim_trigger", "random"]


def main() -> None:
    print("== One-shot prisoner's dilemma ==")
    row_payoffs = [[3.0, 0.0], [5.0, 1.0]]
    col_payoffs = [[3.0, 5.0], [0.0, 1.0]]
    equilibria = cds2.game_theory.pure_nash_equilibria(row_payoffs, col_payoffs)
    print(f"pure Nash equilibria (row, col): {equilibria}")

    dominated_rows = cds2.game_theory.strictly_dominated_actions(row_payoffs)
    print(f"strictly dominated rows        : {dominated_rows.tolist()}")

    print("\n== Round-robin iterated PD (200 rounds) ==")
    scores: dict[str, float] = {name: 0.0 for name in STRATEGIES}
    for i, strategy_a in enumerate(STRATEGIES):
        for strategy_b in STRATEGIES[i + 1 :]:
            match = cds2.game_theory.play_iterated_pd(
                strategy_a, strategy_b, rounds=200, seed=i * 7 + len(strategy_b)
            )
            scores[strategy_a] += match.final_a
            scores[strategy_b] += match.final_b

    ranking = sorted(scores.items(), key=lambda kv: -kv[1])
    for rank, (name, score) in enumerate(ranking, start=1):
        bar = "#" * int(score / 40)
        print(f"{rank}. {name:<16s} {score:>7.0f} |{bar}")

    sample = cds2.game_theory.play_iterated_pd("tit_for_tat", "always_defect", rounds=10, seed=1)
    print("\n== tit-for-tat vs always-defect (10 rounds) ==")
    print(f"tft moves : {sample.moves_a.tolist()}")
    print(f"def moves : {sample.moves_b.tolist()}")
    print(f"scores    : tft {sample.final_a:.0f} vs defect {sample.final_b:.0f}")

    zero_sum = [[2.0, -1.0, 0.5], [-1.0, 1.5, -0.5], [0.5, -0.5, 1.0]]
    mixed = cds2.game_theory.zero_sum_mixed(zero_sum)
    print("\n== Zero-sum minimax ==")
    print(f"row strategy : {np.round(mixed.row_strategy, 3).tolist()}")
    print(f"game value   : {mixed.value:.3f}")


if __name__ == "__main__":
    main()
