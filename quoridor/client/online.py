"""Background websocket client used by the pygame application."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Any

from quoridor.core.actions import MovePawnAction, PlaceWallAction
from quoridor.server.protocol import action_to_dict


class OnlineSession:
    def __init__(self, server_url: str, player_name: str, room_id: str, num_players: int = 4) -> None:
        self.server_url = server_url
        self.player_name = player_name
        self.room_id = room_id
        self.num_players = num_players
        self._incoming: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._outgoing: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.run(self._runner())

    async def _runner(self) -> None:
        try:
            import websockets
        except ImportError:
            self._incoming.put({"type": "error", "message": "Missing dependency: websockets"})
            return

        try:
            async with websockets.connect(self.server_url) as websocket:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "join_room",
                            "room_id": self.room_id,
                            "player_name": self.player_name,
                            "num_players": self.num_players,
                        }
                    )
                )
                while not self._stop.is_set():
                    await self._flush_outgoing(websocket)
                    try:
                        raw = await asyncio.wait_for(websocket.recv(), timeout=0.05)
                    except TimeoutError:
                        continue
                    self._incoming.put(json.loads(raw))
        except Exception as exc:  # noqa: BLE001
            self._incoming.put({"type": "error", "message": str(exc)})

    async def _flush_outgoing(self, websocket: Any) -> None:
        while True:
            try:
                message = self._outgoing.get_nowait()
            except queue.Empty:
                return
            await websocket.send(json.dumps(message))

    def send_ready(self) -> None:
        self._outgoing.put({"type": "ready"})

    def send_action(self, action: MovePawnAction | PlaceWallAction) -> None:
        self._outgoing.put({"type": "action", "action": action_to_dict(action)})

    def poll_messages(self) -> list[dict[str, Any]]:
        messages = []
        while True:
            try:
                messages.append(self._incoming.get_nowait())
            except queue.Empty:
                return messages

    def close(self) -> None:
        self._stop.set()
