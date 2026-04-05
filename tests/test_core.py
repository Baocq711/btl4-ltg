import unittest
from dataclasses import replace
from pathlib import Path

from quoridor.core import MovePawnAction, PlaceWallAction, apply_action, create_initial_state, is_action_legal, load_game, save_game
from quoridor.core.models import Position
from quoridor.core.rules import check_winner, generate_pawn_moves


class CoreRulesTests(unittest.TestCase):
    def test_initial_pawn_moves_for_player_one(self) -> None:
        state = create_initial_state(2)
        moves = {(move.to_x, move.to_y) for move in generate_pawn_moves(state, 0)}
        self.assertEqual(moves, {(3, 0), (5, 0), (4, 1)})

    def test_jump_over_adjacent_player(self) -> None:
        state = create_initial_state(2)
        players = list(state.players)
        players[0] = replace(players[0], pos=Position(4, 4))
        players[1] = replace(players[1], pos=Position(4, 5))
        state = replace(state, players=tuple(players), current_turn=0)
        moves = {(move.to_x, move.to_y) for move in generate_pawn_moves(state, 0)}
        self.assertIn((4, 6), moves)

    def test_diagonal_move_when_jump_is_blocked(self) -> None:
        state = create_initial_state(2)
        players = list(state.players)
        players[0] = replace(players[0], pos=Position(4, 4))
        players[1] = replace(players[1], pos=Position(4, 5))
        state = replace(state, players=tuple(players), current_turn=0, walls_h=frozenset({(4, 5)}))
        moves = {(move.to_x, move.to_y) for move in generate_pawn_moves(state, 0)}
        self.assertIn((3, 5), moves)
        self.assertIn((5, 5), moves)
        self.assertNotIn((4, 6), moves)

    def test_wall_cannot_remove_last_path(self) -> None:
        state = create_initial_state(2)
        players = list(state.players)
        players[0] = replace(players[0], pos=Position(1, 1))
        players[1] = replace(players[1], pos=Position(8, 8))
        state = replace(
            state,
            players=tuple(players),
            current_turn=0,
            walls_h=frozenset({(0, 0)}),
            walls_v=frozenset({(0, 1), (1, 0)}),
        )
        self.assertFalse(is_action_legal(state, PlaceWallAction(1, 1, "H")))

    def test_apply_action_updates_state_and_save_round_trip(self) -> None:
        state = create_initial_state(2)
        next_state = apply_action(state, MovePawnAction(4, 1))
        self.assertEqual(next_state.current_turn, 1)
        self.assertEqual(next_state.move_count, 1)
        self.assertIsNone(check_winner(next_state))

        save_path = Path("d:/Hoc/BTL/LTG/4/codex/saves/test_save.json")
        save_game(next_state, save_path)
        loaded = load_game(save_path)
        self.assertEqual(loaded.current_turn, next_state.current_turn)
        self.assertEqual(loaded.players[0].pos, Position(4, 1))
        if save_path.exists():
            save_path.unlink()


if __name__ == "__main__":
    unittest.main()
