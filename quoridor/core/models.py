"""Core data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Goal = Literal["TOP", "BOTTOM", "LEFT", "RIGHT"]
PlayerKind = Literal["human", "random", "mcts", "minimax", "remote"]


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def moved(self, dx: int, dy: int) -> "Position":
        return Position(self.x + dx, self.y + dy)


@dataclass(frozen=True)
class Player:
    id: int
    name: str
    pos: Position
    walls_left: int
    goal: Goal
    kind: PlayerKind = "human"
    active: bool = True


@dataclass(frozen=True)
class Wall:
    x: int
    y: int
    orientation: Literal["H", "V"]
