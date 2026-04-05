"""Action pruning for expensive tree search algorithms."""

from __future__ import annotations

from quoridor.core import (
    Action,
    GameState,
    MovePawnAction,
    PlaceWallAction,
    generate_pawn_moves,
    is_wall_legal,
)
from quoridor.core.pathfinding import shortest_path


def _edge_blockers(a, b):
    if a.x != b.x:
        x = min(a.x, b.x)
        y = a.y
        return ((x, y, "V"), (x, y - 1, "V"))
    x = a.x
    y = min(a.y, b.y)
    return ((x, y, "H"), (x - 1, y, "H"))


def _add_candidate(state: GameState, store: dict[tuple[int, int, str], tuple[int, int, int]], candidate, priority: int, depth: int) -> None:
    x, y, orientation = candidate
    if not (0 <= x < state.board_size - 1 and 0 <= y < state.board_size - 1):
        return
    center_bias = abs(x - (state.board_size // 2)) + abs(y - (state.board_size // 2))
    ranking = (priority, depth, center_bias)
    key = (x, y, orientation)
    previous = store.get(key)
    if previous is None or ranking < previous:
        store[key] = ranking


def _candidate_wall_actions(state: GameState, player_id: int, wall_limit: int) -> list[PlaceWallAction]:
    player = state.get_player(player_id)
    if player.walls_left <= 0 or wall_limit <= 0:
        return []

    candidates: dict[tuple[int, int, str], tuple[int, int, int]] = {}
    for path_owner in state.players:
        if not path_owner.active:
            continue
        path = shortest_path(state, path_owner.id) or []
        priority = 0 if path_owner.id != player_id else 1
        for depth, (left, right) in enumerate(zip(path, path[1:])):
            for candidate in _edge_blockers(left, right):
                _add_candidate(state, candidates, candidate, priority, depth)

    for other in state.players:
        if not other.active:
            continue
        priority = 2 if other.id != player_id else 3
        for dx in (-1, 0):
            for dy in (-1, 0):
                for orientation in ("H", "V"):
                    _add_candidate(state, candidates, (other.pos.x + dx, other.pos.y + dy, orientation), priority, 0)

    valid_actions: list[PlaceWallAction] = []
    for (x, y, orientation), _ in sorted(candidates.items(), key=lambda item: item[1]):
        action = PlaceWallAction(x=x, y=y, orientation=orientation)
        if is_wall_legal(state, player_id, action):
            valid_actions.append(action)
            if len(valid_actions) >= wall_limit:
                break
    return valid_actions


def pruned_actions(state: GameState, player_id: int, wall_limit: int = 24) -> list[Action]:
    pawn_actions = generate_pawn_moves(state, player_id)
    wall_actions = _candidate_wall_actions(state, player_id, wall_limit)
    return [*pawn_actions, *wall_actions]
