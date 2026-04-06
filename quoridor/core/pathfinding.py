"""Pathfinding helpers used for wall validation and AI heuristics."""

from __future__ import annotations

from collections import deque

from .board import edge_blocked, is_goal_reached, is_in_bounds
from .constants import DIRECTIONS
from .models import Position
from .state import GameState


def neighbors_without_players(state: GameState, pos: Position) -> list[Position]:
    result: list[Position] = []
    for dx, dy in DIRECTIONS.values():
        candidate = pos.moved(dx, dy)
        if not is_in_bounds(candidate, state.board_size):
            continue
        if edge_blocked(state, pos, candidate):
            continue
        result.append(candidate)
    return result


def shortest_path(state: GameState, player_id: int) -> list[Position] | None:
    player = state.get_player(player_id)
    start = player.pos
    queue = deque([start])
    parents: dict[Position, Position | None] = {start: None}

    while queue:
        current = queue.popleft()
        if is_goal_reached(current, player.goal, state.board_size):
            path: list[Position] = []
            cursor: Position | None = current
            while cursor is not None:
                path.append(cursor)
                cursor = parents[cursor]
            path.reverse()
            return path

        for nxt in neighbors_without_players(state, current):
            if nxt in parents:
                continue
            parents[nxt] = current
            queue.append(nxt)

    return None


def shortest_path_len(state: GameState, player_id: int) -> int | None:
    """BFS that returns only the distance, without constructing the path."""
    player = state.get_player(player_id)
    start = player.pos
    queue = deque([(start, 0)])
    visited: set[Position] = {start}

    while queue:
        current, dist = queue.popleft()
        if is_goal_reached(current, player.goal, state.board_size):
            return dist
        for nxt in neighbors_without_players(state, current):
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, dist + 1))

    return None
