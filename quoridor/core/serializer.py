"""JSON persistence for save/load and Continue."""

from __future__ import annotations

import json
from pathlib import Path

from .actions import MovePawnAction, PlaceWallAction
from .models import Player, Position
from .state import GameState


def _player_to_dict(player: Player) -> dict[str, object]:
    return {
        "id": player.id,
        "name": player.name,
        "pos": {"x": player.pos.x, "y": player.pos.y},
        "walls_left": player.walls_left,
        "goal": player.goal,
        "kind": player.kind,
        "active": player.active,
    }


def state_to_dict(state: GameState) -> dict[str, object]:
    return {
        "board_size": state.board_size,
        "players": [_player_to_dict(player) for player in state.players],
        "walls_h": [list(wall) for wall in sorted(state.walls_h)],
        "walls_v": [list(wall) for wall in sorted(state.walls_v)],
        "current_turn": state.current_turn,
        "move_count": state.move_count,
        "winner_id": state.winner_id,
        "mode": state.mode,
        "move_history": list(state.move_history),
    }


def state_from_dict(payload: dict[str, object]) -> GameState:
    players = []
    for raw_player in payload["players"]:
        assert isinstance(raw_player, dict)
        raw_pos = raw_player["pos"]
        assert isinstance(raw_pos, dict)
        players.append(
            Player(
                id=int(raw_player["id"]),
                name=str(raw_player["name"]),
                pos=Position(x=int(raw_pos["x"]), y=int(raw_pos["y"])),
                walls_left=int(raw_player["walls_left"]),
                goal=str(raw_player["goal"]),
                kind=str(raw_player.get("kind", "human")),
                active=bool(raw_player.get("active", True)),
            )
        )

    return GameState(
        board_size=int(payload["board_size"]),
        players=tuple(players),
        walls_h=frozenset(tuple(item) for item in payload.get("walls_h", [])),
        walls_v=frozenset(tuple(item) for item in payload.get("walls_v", [])),
        current_turn=int(payload["current_turn"]),
        move_count=int(payload.get("move_count", 0)),
        winner_id=payload.get("winner_id"),
        mode=str(payload.get("mode", "LOCAL")),
        move_history=tuple(payload.get("move_history", [])),
    )


def action_from_record(record: dict[str, object]) -> MovePawnAction | PlaceWallAction:
    kind = record["kind"]
    if kind == "move_pawn":
        return MovePawnAction(to_x=int(record["to_x"]), to_y=int(record["to_y"]))
    return PlaceWallAction(
        x=int(record["x"]),
        y=int(record["y"]),
        orientation=str(record["orientation"]),
    )


def save_game(state: GameState, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(state_to_dict(state), indent=2), encoding="utf-8")


def load_game(path: str | Path) -> GameState:
    source = Path(path)
    return state_from_dict(json.loads(source.read_text(encoding="utf-8")))
