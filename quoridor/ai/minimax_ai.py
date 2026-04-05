"""Minimax AI for two-player Quoridor."""

from __future__ import annotations

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


def choose_minimax_action(state: GameState, player_id: int | None = None, depth: int = 2) -> Action:
    actor_id = state.current_turn if player_id is None else player_id
    if len(state.players) != 2:
        raise ValueError("Minimax AI is only supported for 2-player mode.")
    if actor_id != state.current_turn:
        raise ValueError("Minimax must act for the current turn player.")

    best_score = -inf
    best_action: Action | None = None
    cache: dict[tuple[tuple[object, ...], int], float] = {}
    for action in _ordered_actions(state):
        score = _search(simulate_action(state, action), actor_id, depth - 1, -inf, inf, cache)
        if score > best_score:
            best_score = score
            best_action = action

    if best_action is None:
        raise ValueError("No legal action available for minimax AI.")
    return best_action
