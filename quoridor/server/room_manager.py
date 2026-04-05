"""Room lifecycle and authoritative game-state management."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field

from quoridor.core import Action, GameState, apply_action, create_initial_state


@dataclass
class RoomPlayer:
    player_id: int
    name: str
    ready: bool = False
    connected: bool = True


@dataclass
class Room:
    room_id: str
    max_players: int = 4
    players: dict[int, RoomPlayer] = field(default_factory=dict)
    state: GameState | None = None
    started: bool = False

    def join(self, player_name: str) -> RoomPlayer:
        if self.started:
            raise ValueError("Cannot join a room that has already started.")
        if len(self.players) >= self.max_players:
            raise ValueError("Room is full.")

        player_id = len(self.players)
        participant = RoomPlayer(player_id=player_id, name=player_name)
        self.players[player_id] = participant
        return participant

    def mark_ready(self, player_id: int) -> bool:
        if player_id not in self.players:
            raise ValueError("Player is not in this room.")
        self.players[player_id].ready = True
        if len(self.players) == self.max_players and all(player.ready for player in self.players.values()):
            self.started = True
            kinds = ["remote"] * self.max_players
            self.state = create_initial_state(num_players=self.max_players, player_kinds=kinds, mode="ONLINE")
            return True
        return False

    def apply_player_action(self, player_id: int, action: Action) -> GameState:
        if not self.started or self.state is None:
            raise ValueError("Match has not started yet.")
        if player_id != self.state.current_turn:
            raise ValueError("It is not this player's turn.")
        self.state = apply_action(self.state, action)
        return self.state

    def disconnect(self, player_id: int) -> None:
        if player_id in self.players:
            self.players[player_id].connected = False


class RoomManager:
    def __init__(self) -> None:
        self.rooms: dict[str, Room] = {}

    def _generate_room_id(self) -> str:
        while True:
            room_id = "".join(random.choice(string.ascii_uppercase) for _ in range(4))
            if room_id not in self.rooms:
                return room_id

    def get_or_create(self, room_id: str | None = None, max_players: int = 4) -> Room:
        clean_room_id = (room_id or "").strip().upper()
        if not clean_room_id:
            clean_room_id = self._generate_room_id()
        room = self.rooms.get(clean_room_id)
        if room is None:
            room = Room(room_id=clean_room_id, max_players=max_players)
            self.rooms[clean_room_id] = room
        return room

    def join_room(self, room_id: str | None, player_name: str, max_players: int = 4) -> tuple[Room, RoomPlayer]:
        room = self.get_or_create(room_id, max_players=max_players)
        participant = room.join(player_name)
        return room, participant

    def mark_ready(self, room_id: str, player_id: int) -> bool:
        room = self.rooms[room_id]
        return room.mark_ready(player_id)

    def apply_action(self, room_id: str, player_id: int, action: Action) -> GameState:
        room = self.rooms[room_id]
        return room.apply_player_action(player_id, action)

    def disconnect(self, room_id: str, player_id: int) -> None:
        room = self.rooms.get(room_id)
        if room is not None:
            room.disconnect(player_id)
