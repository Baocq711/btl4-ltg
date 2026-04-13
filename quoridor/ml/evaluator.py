"""Evaluate neural agent against baseline AIs."""

from __future__ import annotations

import numpy as np

from quoridor.ai import choose_mcts_action, choose_minimax_action, choose_random_action
from quoridor.core import Action, GameState, create_initial_state, simulate_action
from quoridor.ml.alpha_mcts import AlphaMCTS
from quoridor.ml.config import AlphaZeroConfig
from quoridor.ml.encoding import index_to_action


def _choose_baseline_action(state: GameState, kind: str) -> Action:
    """Select action using a baseline AI."""
    pid = state.current_turn
    if kind == "random":
        return choose_random_action(state, pid)
    elif kind == "minimax":
        return choose_minimax_action(state, pid, depth=2)
    elif kind == "mcts":
        return choose_mcts_action(state, pid, iterations=150, time_budget=0.5)
    raise ValueError(f"Unknown baseline kind: {kind}")


def _play_eval_game(
    mcts: AlphaMCTS,
    opponent_kind: str,
    neural_is_p1: bool,
    rng: np.random.Generator,
    max_moves: int = 200,
) -> int | None:
    """Play one evaluation game. Returns winner player_id or None for draw."""
    neural_pid = 0 if neural_is_p1 else 1
    state = create_initial_state(num_players=2, player_kinds=["human", "human"], mode="LOCAL")

    for _ in range(max_moves):
        if state.winner_id is not None:
            break

        if state.current_turn == neural_pid:
            action_probs, _ = mcts.search(state, neural_pid, seed=int(rng.integers(0, 2**31)))
            action_idx = mcts.select_action(action_probs, temperature=0.1, rng=rng)
            flip = state.get_player(state.current_turn).goal == "TOP"
            action = index_to_action(action_idx, flip=flip)
        else:
            action = _choose_baseline_action(state, opponent_kind)

        state = simulate_action(state, action)

    return state.winner_id


def evaluate_against_baseline(
    predict_fn,
    variables,
    config: AlphaZeroConfig,
    opponent_kind: str = "random",
    num_games: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate neural agent vs a baseline, alternating sides.

    Returns:
        Dict with keys: win_rate, draw_rate, avg_game_length, wins, losses, draws.
    """
    n = num_games if num_games is not None else config.eval_games
    mcts = AlphaMCTS(predict_fn, variables, config)
    rng = np.random.default_rng(seed)

    wins = 0
    losses = 0
    draws = 0

    for i in range(n):
        neural_is_p1 = (i % 2 == 0)
        neural_pid = 0 if neural_is_p1 else 1
        winner = _play_eval_game(mcts, opponent_kind, neural_is_p1, rng)

        if winner is None:
            draws += 1
        elif winner == neural_pid:
            wins += 1
        else:
            losses += 1

    total = wins + losses + draws
    return {
        "win_rate": wins / max(total, 1),
        "draw_rate": draws / max(total, 1),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total": total,
    }


def evaluate_checkpoints(
    predict_fn,
    new_variables,
    old_variables,
    config: AlphaZeroConfig,
    num_games: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Head-to-head evaluation between two checkpoints.

    Returns:
        Dict with new_win_rate and whether new model is better.
    """
    n = num_games if num_games is not None else config.eval_games
    new_mcts = AlphaMCTS(predict_fn, new_variables, config)
    old_mcts = AlphaMCTS(predict_fn, old_variables, config)
    rng = np.random.default_rng(seed)

    new_wins = 0
    old_wins = 0
    draws = 0

    for i in range(n):
        # Alternate who goes first
        new_is_p1 = (i % 2 == 0)
        new_pid = 0 if new_is_p1 else 1
        old_pid = 1 - new_pid

        state = create_initial_state(num_players=2, player_kinds=["human", "human"], mode="LOCAL")

        for _ in range(200):
            if state.winner_id is not None:
                break

            if state.current_turn == new_pid:
                probs, _ = new_mcts.search(state, new_pid, seed=int(rng.integers(0, 2**31)))
                idx = new_mcts.select_action(probs, temperature=0.1, rng=rng)
            else:
                probs, _ = old_mcts.search(state, old_pid, seed=int(rng.integers(0, 2**31)))
                idx = old_mcts.select_action(probs, temperature=0.1, rng=rng)

            flip = state.get_player(state.current_turn).goal == "TOP"
            action = index_to_action(idx, flip=flip)
            state = simulate_action(state, action)

        if state.winner_id is None:
            draws += 1
        elif state.winner_id == new_pid:
            new_wins += 1
        else:
            old_wins += 1

    total = new_wins + old_wins + draws
    new_win_rate = new_wins / max(total, 1)
    return {
        "new_win_rate": new_win_rate,
        "new_wins": new_wins,
        "old_wins": old_wins,
        "draws": draws,
        "is_better": new_win_rate >= config.win_rate_threshold,
    }
