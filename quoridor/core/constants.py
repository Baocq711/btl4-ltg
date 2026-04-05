"""Shared constants for the Quoridor engine."""

BOARD_SIZE = 9
TWO_PLAYER_WALLS = 10
FOUR_PLAYER_WALLS = 5
DEFAULT_SAVE_PATH = "saves/latest_local_game.json"

ORIENTATIONS = {"H", "V"}
GOALS = {"TOP", "BOTTOM", "LEFT", "RIGHT"}
PLAYER_KINDS = {"human", "random", "mcts", "minimax", "remote"}

DIRECTIONS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
