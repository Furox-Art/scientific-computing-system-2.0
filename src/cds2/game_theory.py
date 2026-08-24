"""Two-player game theory: dominance, Nash equilibria, zero-sum LP solving, iterated PD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

__all__ = [
    "EliminationResult",
    "MixedEquilibrium",
    "PDResult",
    "expected_payoff",
    "iterated_elimination",
    "play_iterated_pd",
    "pure_nash_equilibria",
    "strictly_dominated_actions",
    "zero_sum_mixed",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_SUPPORTED_STRATEGIES = (
    "always_cooperate",
    "always_defect",
    "grim_trigger",
    "random",
    "tit_for_tat",
)


def _as_matrix(matrix: object) -> FloatArray:
    """Convert input to a validated non-empty 2-D float payoff matrix."""
    converted = np.asarray(matrix, dtype=float)
    if converted.ndim != 2 or converted.size == 0:
        msg = "payoff matrix must be non-empty and 2-D"
        raise ValueError(msg)
    return converted


@dataclass(frozen=True)
class EliminationResult:
    """Surviving original action indices after iterated dominance elimination."""

    row_actions: IntArray
    col_actions: IntArray
    rounds: int


@dataclass(frozen=True)
class MixedEquilibrium:
    """Mixed-strategy equilibrium of a zero-sum game."""

    row_strategy: FloatArray
    col_strategy: FloatArray
    value: float


@dataclass(frozen=True)
class PDResult:
    """Scores and move histories of an iterated prisoner's dilemma match."""

    final_a: float
    final_b: float
    moves_a: IntArray
    moves_b: IntArray


def strictly_dominated_actions(payoffs: object, player: str = "row") -> IntArray:
    """Indices of actions beaten everywhere by some rival action of the same player."""
    matrix = _as_matrix(payoffs)
    if player == "row":
        profiles = matrix
    elif player == "col":
        profiles = matrix.T
    else:
        msg = "player must be row or col"
        raise ValueError(msg)
    count = profiles.shape[0]
    dominated: list[int] = []
    for action in range(count):
        for rival in range(count):
            if rival != action and bool(np.all(profiles[rival] > profiles[action])):
                dominated.append(action)
                break
    return np.asarray(dominated, dtype=np.int64)


def iterated_elimination(
    payoff_row: object, payoff_col: object, max_rounds: int = 50
) -> EliminationResult:
    """Repeatedly delete strictly dominated actions until stable or ``max_rounds``."""
    row_payoffs = _as_matrix(payoff_row)
    col_payoffs = _as_matrix(payoff_col)
    if row_payoffs.shape != col_payoffs.shape:
        msg = "row and column payoffs must share a shape"
        raise ValueError(msg)
    if max_rounds < 1:
        msg = "max_rounds >= 1 is required"
        raise ValueError(msg)
    rows = list(range(row_payoffs.shape[0]))
    cols = list(range(row_payoffs.shape[1]))
    rounds = 0
    while rounds < max_rounds and rows and cols:
        dead_rows = strictly_dominated_actions(row_payoffs[np.ix_(rows, cols)], player="row")
        dead_cols = strictly_dominated_actions(col_payoffs[np.ix_(rows, cols)], player="col")
        removed_rows = {rows[int(index)] for index in dead_rows}
        removed_cols = {cols[int(index)] for index in dead_cols}
        if not removed_rows and not removed_cols:
            break
        rows = [action for action in rows if action not in removed_rows]
        cols = [action for action in cols if action not in removed_cols]
        rounds += 1
    return EliminationResult(
        row_actions=np.asarray(rows, dtype=np.int64),
        col_actions=np.asarray(cols, dtype=np.int64),
        rounds=rounds,
    )


def pure_nash_equilibria(payoff_row: object, payoff_col: object) -> list[tuple[int, int]]:
    """Cells where each action is a weak best response to the other player's action."""
    row_payoffs = _as_matrix(payoff_row)
    col_payoffs = _as_matrix(payoff_col)
    if row_payoffs.shape != col_payoffs.shape:
        msg = "row and column payoffs must share a shape"
        raise ValueError(msg)
    equilibria: list[tuple[int, int]] = []
    for col in range(row_payoffs.shape[1]):
        best_row_value = row_payoffs[:, col].max()
        for row in range(row_payoffs.shape[0]):
            best_col_value = col_payoffs[row, :].max()
            if row_payoffs[row, col] == best_row_value and col_payoffs[row, col] == best_col_value:
                equilibria.append((row, col))
    return equilibria


def zero_sum_mixed(payoff_row: object) -> MixedEquilibrium:
    """Solve a zero-sum game for the maximin mixed strategy via linear programming."""
    matrix = _as_matrix(payoff_row)
    n_rows, n_cols = matrix.shape
    costs = np.zeros(n_rows + 1)
    costs[-1] = 1.0
    a_ub = np.hstack([-matrix.T, -np.ones((n_cols, 1))])
    b_ub = np.zeros(n_cols)
    a_eq = np.zeros((1, n_rows + 1))
    a_eq[0, :n_rows] = 1.0
    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * n_rows + [(None, None)]
    solution = linprog(
        costs,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.ones(1),
        bounds=bounds,
        method="highs",
    )
    if not solution.success or solution.x is None:
        msg = "game has no finite value"
        raise ValueError(msg)
    row_strategy = np.asarray(solution.x, dtype=float)[:n_rows]
    duals = -np.asarray(solution.ineqlin.marginals, dtype=float)
    total = float(duals.sum())
    if total <= 0.0:
        msg = "game has no finite value"
        raise ValueError(msg)
    return MixedEquilibrium(
        row_strategy=row_strategy,
        col_strategy=duals / total,
        value=float(-solution.fun),
    )


def expected_payoff(payoff_row: object, row_strategy: object, col_strategy: object) -> float:
    """Expected payoff ``r^T A c`` of mixed strategies over the row player's matrix."""
    matrix = _as_matrix(payoff_row)
    row_vector = np.asarray(row_strategy, dtype=float)
    col_vector = np.asarray(col_strategy, dtype=float)
    if abs(float(row_vector.sum()) - 1.0) > 1e-9 or abs(float(col_vector.sum()) - 1.0) > 1e-9:
        msg = "strategies must be probability vectors"
        raise ValueError(msg)
    if row_vector.size != matrix.shape[0] or col_vector.size != matrix.shape[1]:
        msg = "strategy lengths must match the payoff matrix"
        raise ValueError(msg)
    return float(row_vector @ matrix @ col_vector)


def _pd_move(strategy: str, opp_history: list[int], rng: np.random.Generator) -> int:
    if strategy == "always_cooperate":
        return 1
    if strategy == "always_defect":
        return 0
    if strategy == "tit_for_tat":
        return 1 if not opp_history else opp_history[-1]
    if strategy == "grim_trigger":
        return 0 if 0 in opp_history else 1
    return int(rng.integers(0, 2))


def play_iterated_pd(
    strategy_a: str,
    strategy_b: str,
    rounds: int = 100,
    temptation: float = 5.0,
    sucker: float = 0.0,
    punishment: float = 2.5,
    reward: float = 3.0,
    seed: int | None = None,
) -> PDResult:
    """Play the iterated prisoner's dilemma between two built-in strategies."""
    for strategy in (strategy_a, strategy_b):
        if strategy not in _SUPPORTED_STRATEGIES:
            msg = (
                f"unknown strategy {strategy!r}; "
                f"supported strategies: {', '.join(_SUPPORTED_STRATEGIES)}"
            )
            raise ValueError(msg)
    rng = np.random.default_rng(seed)
    moves_a: list[int] = []
    moves_b: list[int] = []
    score_a = 0.0
    score_b = 0.0
    for _ in range(rounds):
        move_a = _pd_move(strategy_a, moves_b, rng)
        move_b = _pd_move(strategy_b, moves_a, rng)
        if move_a == 1 and move_b == 1:
            payoff_a, payoff_b = reward, reward
        elif move_a == 0 and move_b == 0:
            payoff_a, payoff_b = punishment, punishment
        elif move_a == 1:
            payoff_a, payoff_b = sucker, temptation
        else:
            payoff_a, payoff_b = temptation, sucker
        score_a += payoff_a
        score_b += payoff_b
        moves_a.append(move_a)
        moves_b.append(move_b)
    return PDResult(
        final_a=score_a,
        final_b=score_b,
        moves_a=np.asarray(moves_a, dtype=np.int64),
        moves_b=np.asarray(moves_b, dtype=np.int64),
    )
