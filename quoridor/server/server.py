"""Async websocket server for online Quoridor matches."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any

from .protocol import action_from_dict, state_update_message
from .room_manager import RoomManager


class QuoridorServer:
    def __init__(self) -> None:
        self.room_manager = RoomManager()
        self.connections: dict[Any, tuple[str, int]] = {}
        self.room_connections: dict[str, set[Any]] = defaultdict(set)

    async def send_json(self, websocket: Any, payload: dict[str, object]) -> None:
        await websocket.send(json.dumps(payload))

    async def broadcast(self, room_id: str, payload: dict[str, object]) -> None:
        sockets = list(self.room_connections.get(room_id, set()))
        if not sockets:
            return
        message = json.dumps(payload)
        await asyncio.gather(*(socket.send(message) for socket in sockets), return_exceptions=True)

    async def handle_message(self, websocket: Any, payload: dict[str, object]) -> None:
        message_type = payload.get("type")
        if message_type == "join_room":
            num_players = int(payload.get("num_players", 4))
            if num_players not in (2, 4):
                num_players = 4
            room, participant = self.room_manager.join_room(
                str(payload.get("room_id", "")),
                str(payload.get("player_name", "Player")),
                max_players=num_players,
            )
            self.connections[websocket] = (room.room_id, participant.player_id)
            self.room_connections[room.room_id].add(websocket)
            await self.send_json(
                websocket,
                {
                    "type": "room_joined",
                    "room_id": room.room_id,
                    "player_id": participant.player_id,
                    "player_name": participant.name,
                    "seat_count": len(room.players),
                    "max_players": room.max_players,
                },
            )
            await self.broadcast(
                room.room_id,
                {"type": "player_joined", "seat_count": len(room.players)},
            )
            return

        if websocket not in self.connections:
            await self.send_json(websocket, {"type": "error", "message": "Join a room first."})
            return

        room_id, player_id = self.connections[websocket]
        if message_type == "ready":
            started = self.room_manager.mark_ready(room_id, player_id)
            if started:
                room = self.room_manager.rooms[room_id]
                await self.broadcast(room_id, {"type": "game_started"})
                await self.broadcast(room_id, state_update_message(room.state))
            return

        if message_type == "action":
            try:
                action = action_from_dict(payload["action"])
                state = self.room_manager.apply_action(room_id, player_id, action)
            except Exception as exc:  # noqa: BLE001
                await self.send_json(websocket, {"type": "error", "message": str(exc)})
                return

            await self.broadcast(room_id, state_update_message(state))
            if state.winner_id is not None:
                await self.broadcast(room_id, {"type": "game_over", "winner_id": state.winner_id})
            return

        await self.send_json(websocket, {"type": "error", "message": f"Unknown message type: {message_type}"})

    async def handler(self, websocket: Any) -> None:
        try:
            async for raw_message in websocket:
                try:
                    payload = json.loads(raw_message)
                    await self.handle_message(websocket, payload)
                except json.JSONDecodeError:
                    await self.send_json(websocket, {"type": "error", "message": "Invalid JSON payload."})
        except Exception:  # noqa: BLE001 – handle abrupt disconnects gracefully
            pass
        finally:
            connection = self.connections.pop(websocket, None)
            if connection is None:
                return
            room_id, player_id = connection
            self.room_manager.disconnect(room_id, player_id)
            self.room_connections[room_id].discard(websocket)


async def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("The 'websockets' package is required for online play.") from exc

    server = QuoridorServer()
    async with websockets.serve(server.handler, host, port):
        print(f"Quoridor server listening on ws://{host}:{port}")
        await asyncio.Future()
