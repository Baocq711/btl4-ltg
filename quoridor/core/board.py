"""Board geometry and wall interaction helpers."""

from __future__ import annotations

from .models import Position
from .state import GameState


def is_in_bounds(pos: Position, board_size: int) -> bool:
    return 0 <= pos.x < board_size and 0 <= pos.y < board_size


def is_goal_reached(pos: Position, goal: str, board_size: int) -> bool:
    if goal == "TOP":
        return pos.y == 0
    if goal == "BOTTOM":
        return pos.y == board_size - 1
    if goal == "LEFT":
        return pos.x == 0
    if goal == "RIGHT":
        return pos.x == board_size - 1
    raise ValueError(f"Unknown goal: {goal}")


def occupied_positions(state: GameState) -> dict[tuple[int, int], int]:
    return {(player.pos.x, player.pos.y): player.id for player in state.players if player.active}


def edge_blocked(state: GameState, a: Position, b: Position) -> bool:
    dx = b.x - a.x
    dy = b.y - a.y
    if abs(dx) + abs(dy) != 1:
        raise ValueError("Only adjacent cells can be checked for wall blocking.")

    if dx == 1:
        return (a.x, a.y) in state.walls_v or (a.x, a.y - 1) in state.walls_v
    if dx == -1:
        return (b.x, b.y) in state.walls_v or (b.x, b.y - 1) in state.walls_v
    if dy == 1:
        return (a.x, a.y) in state.walls_h or (a.x - 1, a.y) in state.walls_h
    return (b.x, b.y) in state.walls_h or (b.x - 1, b.y) in state.walls_h


def wall_in_bounds(state: GameState, x: int, y: int) -> bool:
    return 0 <= x < state.board_size - 1 and 0 <= y < state.board_size - 1
