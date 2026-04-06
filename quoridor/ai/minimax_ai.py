"""Minimax AI for two-player Quoridor."""

from __future__ import annotations

import time
from math import inf

from quoridor.core import Action, GameState, MovePawnAction, simulate_action

from .action_pruning import pruned_actions
from .heuristic import evaluate_state


def _state_key(state: GameState) -> tuple[object, ...]:
    players = tuple(
        (player.id, player.pos.x, player.pos.y, player.walls_left, player.goal, player.active)
        for player in state.players
    )
    return (players, state.walls_h, state.walls_v, state.current_turn, state.winner_id)


def _ordered_actions(state: GameState) -> list[Action]:
    actions = pruned_actions(state, state.current_turn, wall_limit=10)
    actions.sort(key=lambda action: (0 if isinstance(action, MovePawnAction) else 1))
    return actions


def _search(
    state: GameState,
    root_player_id: int,
    depth: int,
    alpha: float,
    beta: float,
    cache: dict[tuple[tuple[object, ...], int], float],
) -> float:
    if depth == 0 or state.winner_id is not None:
        return evaluate_state(state, root_player_id)

    cache_key = (_state_key(state), depth)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    actions = _ordered_actions(state)
    if not actions:
        score = evaluate_state(state, root_player_id)
        cache[cache_key] = score
        return score

    maximizing = state.current_turn == root_player_id
    if maximizing:
        value = -inf
        for action in actions:
            value = max(value, _search(simulate_action(state, action), root_player_id, depth - 1, alpha, beta, cache))
            alpha = max(alpha, value)
            if beta <= alpha:
                break
    else:
        value = inf
        for action in actions:
            value = min(value, _search(simulate_action(state, action), root_player_id, depth - 1, alpha, beta, cache))
            beta = min(beta, value)
            if beta <= alpha:
                break

    cache[cache_key] = value
    return value


def choose_minimax_action(
    state: GameState,
    player_id: int | None = None,
    depth: int = 3,
    time_budget: float = 2.0,
) -> Action:
    actor_id = state.current_turn if player_id is None else player_id
    if len(state.players) != 2:
        raise ValueError("Minimax AI is only supported for 2-player mode.")
    if actor_id != state.current_turn:
        raise ValueError("Minimax must act for the current turn player.")

    actions = _ordered_actions(state)
    if not actions:
        raise ValueError("No legal action available for minimax AI.")

    best_action: Action = actions[0]
    deadline = time.perf_counter() + time_budget

    # Iterative deepening: search deeper layers as long as time allows.
    for d in range(1, depth + 1):
        if time.perf_counter() >= deadline:
            break
        cache: dict[tuple[tuple[object, ...], int], float] = {}
        current_best_score = -inf
        current_best: Action | None = None
        timed_out = False
        for action in actions:
            if time.perf_counter() >= deadline:
                timed_out = True
                break
            score = _search(simulate_action(state, action), actor_id, d - 1, -inf, inf, cache)
            if score > current_best_score:
                current_best_score = score
                current_best = action
        if not timed_out and current_best is not None:
            best_action = current_best

    return best_action
