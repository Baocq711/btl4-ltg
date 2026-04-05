import unittest

from quoridor.core import MovePawnAction
from quoridor.server.room_manager import RoomManager


class RoomManagerTests(unittest.TestCase):
    def test_room_starts_after_four_ready_players(self) -> None:
        manager = RoomManager()
        room_id = "ABCD"
        player_ids = []
        for index in range(4):
            room, participant = manager.join_room(room_id, f"Player {index + 1}")
            player_ids.append(participant.player_id)
        self.assertEqual(room.room_id, room_id)

        for player_id in player_ids[:-1]:
            self.assertFalse(manager.mark_ready(room_id, player_id))
        self.assertTrue(manager.mark_ready(room_id, player_ids[-1]))
        self.assertIsNotNone(manager.rooms[room_id].state)

    def test_room_starts_after_two_ready_players(self) -> None:
        manager = RoomManager()
        room_id = "TW0P"
        player_ids = []
        for index in range(2):
            room, participant = manager.join_room(room_id, f"Player {index + 1}", max_players=2)
            player_ids.append(participant.player_id)
        self.assertEqual(room.max_players, 2)
        self.assertEqual(len(room.players), 2)

        self.assertFalse(manager.mark_ready(room_id, player_ids[0]))
        self.assertTrue(manager.mark_ready(room_id, player_ids[1]))
        self.assertIsNotNone(manager.rooms[room_id].state)
        self.assertEqual(len(manager.rooms[room_id].state.players), 2)

    def test_two_player_room_rejects_third_player(self) -> None:
        manager = RoomManager()
        room_id = "FULL"
        manager.join_room(room_id, "Player 1", max_players=2)
        manager.join_room(room_id, "Player 2", max_players=2)
        with self.assertRaises(ValueError):
            manager.join_room(room_id, "Player 3")

    def test_apply_action_changes_authoritative_state(self) -> None:
        manager = RoomManager()
        room_id = "EFGH"
        for index in range(4):
            room, participant = manager.join_room(room_id, f"Player {index + 1}")
            manager.mark_ready(room.room_id, participant.player_id)

        room = manager.rooms[room_id]
        original_turn = room.state.current_turn
        state = manager.apply_action(room.room_id, original_turn, MovePawnAction(4, 1))
        self.assertEqual(state.move_count, 1)
        self.assertNotEqual(state.current_turn, original_turn)


if __name__ == "__main__":
    unittest.main()
