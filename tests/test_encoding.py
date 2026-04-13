"""Tests for state/action encoding."""

from __future__ import annotations

import numpy as np
import pytest

from quoridor.core import (
    MovePawnAction,
    PlaceWallAction,
    create_initial_state,
    generate_legal_actions,
    simulate_action,
)
from quoridor.ml.encoding import (
    ACTION_SIZE,
    NUM_CHANNELS,
    action_to_index,
    encode_state,
    index_to_action,
    legal_actions_mask,
)
from quoridor.core.constants import BOARD_SIZE


class TestActionEncoding:
    def test_round_trip_move(self):
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                action = MovePawnAction(to_x=x, to_y=y)
                idx = action_to_index(action, flip=False)
                assert 0 <= idx < 81
                recovered = index_to_action(idx, flip=False)
                assert recovered == action

    def test_round_trip_wall_h(self):
        for x in range(BOARD_SIZE - 1):
            for y in range(BOARD_SIZE - 1):
                action = PlaceWallAction(x=x, y=y, orientation="H")
                idx = action_to_index(action, flip=False)
                assert 81 <= idx < 145
                recovered = index_to_action(idx, flip=False)
                assert recovered == action

    def test_round_trip_wall_v(self):
        for x in range(BOARD_SIZE - 1):
            for y in range(BOARD_SIZE - 1):
                action = PlaceWallAction(x=x, y=y, orientation="V")
                idx = action_to_index(action, flip=False)
                assert 145 <= idx < 209
                recovered = index_to_action(idx, flip=False)
                assert recovered == action

    def test_flip_round_trip_move(self):
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                action = MovePawnAction(to_x=x, to_y=y)
                idx = action_to_index(action, flip=True)
                recovered = index_to_action(idx, flip=True)
                assert recovered == action

    def test_flip_round_trip_wall(self):
        for x in range(BOARD_SIZE - 1):
            for y in range(BOARD_SIZE - 1):
                for orient in ("H", "V"):
                    action = PlaceWallAction(x=x, y=y, orientation=orient)
                    idx = action_to_index(action, flip=True)
                    recovered = index_to_action(idx, flip=True)
                    assert recovered == action

    def test_action_size(self):
        assert ACTION_SIZE == 209

    def test_index_out_of_range(self):
        with pytest.raises(ValueError):
            index_to_action(-1)
        with pytest.raises(ValueError):
            index_to_action(ACTION_SIZE)


class TestStateEncoding:
    def test_shape(self):
        state = create_initial_state(num_players=2)
        encoded = encode_state(state, 0)
        assert encoded.shape == (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        assert encoded.dtype == np.float32

    def test_initial_pawns(self):
        state = create_initial_state(num_players=2)
        # P0 at (4, 0), goal BOTTOM -> no flip
        encoded = encode_state(state, 0)
        assert encoded[0, 0, 4] == 1.0  # my pawn at (4,0) -> plane[0][y=0][x=4]
        assert encoded[1, 8, 4] == 1.0  # opp pawn at (4,8) -> plane[1][y=8][x=4]
        assert encoded[0].sum() == 1.0
        assert encoded[1].sum() == 1.0

    def test_initial_pawns_p2_perspective(self):
        state = create_initial_state(num_players=2)
        # P1 at (4, 8), goal TOP -> flip: y -> 8-y
        encoded = encode_state(state, 1)
        # P1's pawn at (4,8) flipped -> (4, 0)
        assert encoded[0, 0, 4] == 1.0
        # P0's pawn at (4,0) flipped -> (4, 8)
        assert encoded[1, 8, 4] == 1.0

    def test_walls_remaining(self):
        state = create_initial_state(num_players=2)
        encoded = encode_state(state, 0)
        # 10 walls / 10 = 1.0
        assert np.allclose(encoded[6], 1.0)
        assert np.allclose(encoded[7], 1.0)

    def test_goal_masks(self):
        state = create_initial_state(num_players=2)
        encoded = encode_state(state, 0)
        # Channel 4: my goal = BOTTOM = row 8
        assert np.all(encoded[4, 8, :] == 1.0)
        assert encoded[4, :8, :].sum() == 0.0
        # Channel 5: opp goal = TOP = row 0
        assert np.all(encoded[5, 0, :] == 1.0)
        assert encoded[5, 1:, :].sum() == 0.0


class TestLegalActionsMask:
    def test_mask_matches_legal_actions(self):
        state = create_initial_state(num_players=2)
        mask = legal_actions_mask(state, 0)
        assert mask.shape == (ACTION_SIZE,)
        legal = generate_legal_actions(state, 0)
        assert mask.sum() == len(legal)

    def test_mask_after_move(self):
        state = create_initial_state(num_players=2)
        # Make a move
        legal = generate_legal_actions(state, 0)
        move = [a for a in legal if isinstance(a, MovePawnAction)][0]
        new_state = simulate_action(state, move)
        # P1's turn now
        mask = legal_actions_mask(new_state, 1)
        legal_p1 = generate_legal_actions(new_state, 1)
        assert mask.sum() == len(legal_p1)
