"""State and action encoding for the neural network.

State is encoded as an (8, 9, 9) tensor from the current player's perspective.
Actions are mapped to a flat index space of size 209.
"""

from __future__ import annotations

import numpy as np

from quoridor.core import (
    GameState,
    MovePawnAction,
    PlaceWallAction,
    generate_legal_actions,
)
from quoridor.core.constants import BOARD_SIZE

# Action space layout:
#   0..80   -> MovePawnAction(x, y)  index = y * 9 + x
#   81..144 -> PlaceWallAction(x, y, "H")  index = 81 + y * 8 + x
#   145..208-> PlaceWallAction(x, y, "V")  index = 145 + y * 8 + x
ACTION_SIZE = 81 + 64 + 64  # 209
NUM_CHANNELS = 8
_WALL_GRID = BOARD_SIZE - 1  # 8


def _needs_flip(state: GameState, player_id: int) -> bool:
    """Return True if we need to flip the board vertically for canonical form.

    Player 0 (goal BOTTOM, starts y=0) is the canonical perspective.
    Player 1 (goal TOP, starts y=8) needs a vertical flip.
    """
    return state.get_player(player_id).goal == "TOP"


def _flip_y_pos(y: int) -> int:
    return BOARD_SIZE - 1 - y


def _flip_y_wall(y: int) -> int:
    return _WALL_GRID - 1 - y


def encode_state(state: GameState, player_id: int) -> np.ndarray:
    """Encode game state into (8, 9, 9) float32 array from player_id's perspective."""
    planes = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    flip = _needs_flip(state, player_id)

    me = state.get_player(player_id)
    opp_id = 1 - player_id  # 2-player only
    opp = state.get_player(opp_id)

    # Channel 0: current player pawn
    my, mx = (me.pos.y, me.pos.x)
    if flip:
        my = _flip_y_pos(my)
    planes[0, my, mx] = 1.0

    # Channel 1: opponent pawn
    oy, ox = (opp.pos.y, opp.pos.x)
    if flip:
        oy = _flip_y_pos(oy)
    planes[1, oy, ox] = 1.0

    # Channel 2: horizontal walls
    for wx, wy in state.walls_h:
        if flip:
            wy = _flip_y_wall(wy)
        planes[2, wy, wx] = 1.0

    # Channel 3: vertical walls
    for wx, wy in state.walls_v:
        if flip:
            wy = _flip_y_wall(wy)
        planes[3, wy, wx] = 1.0

    # Channel 4: current player goal edge (always BOTTOM = row 8 in canonical)
    planes[4, BOARD_SIZE - 1, :] = 1.0

    # Channel 5: opponent goal edge (always TOP = row 0 in canonical)
    planes[5, 0, :] = 1.0

    # Channel 6: current player walls remaining (normalized)
    max_walls = 10  # 2-player
    planes[6, :, :] = me.walls_left / max_walls

    # Channel 7: opponent walls remaining (normalized)
    planes[7, :, :] = opp.walls_left / max_walls

    return planes


def action_to_index(action: MovePawnAction | PlaceWallAction, flip: bool = False) -> int:
    """Convert an action to a flat index in [0, 209)."""
    if isinstance(action, MovePawnAction):
        x, y = action.to_x, action.to_y
        if flip:
            y = _flip_y_pos(y)
        return y * BOARD_SIZE + x
    elif isinstance(action, PlaceWallAction):
        x, y = action.x, action.y
        if flip:
            y = _flip_y_wall(y)
        if action.orientation == "H":
            return 81 + y * _WALL_GRID + x
        else:
            return 145 + y * _WALL_GRID + x
    raise ValueError(f"Unknown action type: {type(action)}")


def index_to_action(index: int, flip: bool = False) -> MovePawnAction | PlaceWallAction:
    """Convert a flat index in [0, 209) to an action."""
    if index < 0 or index >= ACTION_SIZE:
        raise ValueError(f"Index {index} out of range [0, {ACTION_SIZE})")

    if index < 81:
        y, x = divmod(index, BOARD_SIZE)
        if flip:
            y = _flip_y_pos(y)
        return MovePawnAction(to_x=x, to_y=y)
    elif index < 145:
        idx = index - 81
        y, x = divmod(idx, _WALL_GRID)
        if flip:
            y = _flip_y_wall(y)
        return PlaceWallAction(x=x, y=y, orientation="H")
    else:
        idx = index - 145
        y, x = divmod(idx, _WALL_GRID)
        if flip:
            y = _flip_y_wall(y)
        return PlaceWallAction(x=x, y=y, orientation="V")


def legal_actions_mask(state: GameState, player_id: int) -> np.ndarray:
    """Return a binary mask of shape (209,) for legal actions from player_id's canonical perspective."""
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    flip = _needs_flip(state, player_id)
    for action in generate_legal_actions(state, player_id):
        idx = action_to_index(action, flip=flip)
        mask[idx] = 1.0
    return mask
