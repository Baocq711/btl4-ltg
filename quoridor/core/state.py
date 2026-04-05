"""Game state model and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .models import Player

GameMode = Literal["LOCAL", "LOCAL_AI", "ONLINE"]


@dataclass(frozen=True)
class GameState:
    board_size: int
    players: tuple[Player, ...]
    walls_h: frozenset[tuple[int, int]]
    walls_v: frozenset[tuple[int, int]]
    current_turn: int
    move_count: int = 0
    winner_id: int | None = None
    mode: GameMode = "LOCAL"
    move_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def get_player(self, player_id: int) -> Player:
        return self.players[player_id]

    def replace_player(self, player_id: int, player: Player) -> "GameState":
        players = list(self.players)
        players[player_id] = player
        return replace(self, players=tuple(players))
