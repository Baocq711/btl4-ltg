"""JSON protocol helpers for the websocket server."""

from __future__ import annotations

from quoridor.core import MovePawnAction, PlaceWallAction
from quoridor.core.serializer import state_to_dict
from quoridor.core.state import GameState


def action_to_dict(action: MovePawnAction | PlaceWallAction) -> dict[str, object]:
    if isinstance(action, MovePawnAction):
        return {"kind": action.kind, "to_x": action.to_x, "to_y": action.to_y}
    return {
        "kind": action.kind,
        "x": action.x,
        "y": action.y,
        "orientation": action.orientation,
    }


def action_from_dict(payload: dict[str, object]) -> MovePawnAction | PlaceWallAction:
    kind = payload.get("kind")
    if kind == "move_pawn":
        return MovePawnAction(to_x=int(payload["to_x"]), to_y=int(payload["to_y"]))
    if kind == "place_wall":
        return PlaceWallAction(
            x=int(payload["x"]),
            y=int(payload["y"]),
            orientation=str(payload["orientation"]),
        )
    raise ValueError(f"Unknown action kind: {kind}")


def state_update_message(state: GameState) -> dict[str, object]:
    return {"type": "state_update", "state": state_to_dict(state)}
