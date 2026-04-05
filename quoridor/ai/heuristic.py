"""Heuristic evaluation used by adversarial AIs."""

from __future__ import annotations

from quoridor.core import GameState, generate_pawn_moves
from quoridor.core.pathfinding import shortest_path_len


def evaluate_state(state: GameState, player_id: int) -> float:
    if state.winner_id is not None:
        if state.winner_id == player_id:
            return 1_000_000.0
        return -1_000_000.0

    my_player = state.get_player(player_id)
    my_distance = shortest_path_len(state, player_id)
    if my_distance is None:
        return -1_000_000.0

    opponent_distances = []
    for player in state.players:
        if player.id == player_id or not player.active:
            continue
        distance = shortest_path_len(state, player.id)
        if distance is None:
            opponent_distances.append(100.0)
        else:
            opponent_distances.append(float(distance))

    opp_avg = sum(opponent_distances) / max(1, len(opponent_distances))
    mobility = len(generate_pawn_moves(state, player_id))
    center = (state.board_size - 1) / 2
    center_score = state.board_size - (abs(my_player.pos.x - center) + abs(my_player.pos.y - center))
    danger = max(0.0, float(my_distance) - min(opponent_distances, default=float(my_distance)))

    return (
        10.0 * (opp_avg - float(my_distance))
        + 2.0 * my_player.walls_left
        + 0.75 * mobility
        + 0.35 * center_score
        - 1.5 * danger
    )
