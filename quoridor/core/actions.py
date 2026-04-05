"""Action definitions used across local play, AI, and networking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class MovePawnAction:
    to_x: int
    to_y: int
    kind: str = "move_pawn"


@dataclass(frozen=True)
class PlaceWallAction:
    x: int
    y: int
    orientation: str
    kind: str = "place_wall"


Action = Union[MovePawnAction, PlaceWallAction]
