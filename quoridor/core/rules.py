"""Game rules, validation, and reducer logic."""

from __future__ import annotations

from dataclasses import replace

from .actions import Action, MovePawnAction, PlaceWallAction
from .board import edge_blocked, is_goal_reached, is_in_bounds, occupied_positions, wall_in_bounds
from .constants import BOARD_SIZE, DIRECTIONS, FOUR_PLAYER_WALLS, ORIENTATIONS, TWO_PLAYER_WALLS
from .models import Player, Position
from .pathfinding import shortest_path_len
from .state import GameMode, GameState


def create_initial_state(
    num_players: int = 4,
    player_kinds: list[str] | None = None,
    mode: GameMode = "LOCAL",
) -> GameState:
    if num_players not in {2, 4}:
        raise ValueError("Quoridor supports either 2 or 4 players in this implementation.")

    player_kinds = player_kinds or ["human"] * num_players
    if len(player_kinds) != num_players:
        raise ValueError("player_kinds must match num_players.")

    walls = TWO_PLAYER_WALLS if num_players == 2 else FOUR_PLAYER_WALLS
    if num_players == 2:
        seeds = [
            ("Player 1", Position(BOARD_SIZE // 2, 0), "BOTTOM"),
            ("Player 2", Position(BOARD_SIZE // 2, BOARD_SIZE - 1), "TOP"),
        ]
    else:
        seeds = [
            ("Player 1", Position(BOARD_SIZE // 2, 0), "BOTTOM"),
            ("Player 2", Position(BOARD_SIZE // 2, BOARD_SIZE - 1), "TOP"),
            ("Player 3", Position(0, BOARD_SIZE // 2), "RIGHT"),
            ("Player 4", Position(BOARD_SIZE - 1, BOARD_SIZE // 2), "LEFT"),
        ]

    players = tuple(
        Player(id=index, name=name, pos=pos, walls_left=walls, goal=goal, kind=player_kinds[index])
        for index, (name, pos, goal) in enumerate(seeds)
    )
    return GameState(
        board_size=BOARD_SIZE,
        players=players,
        walls_h=frozenset(),
        walls_v=frozenset(),
        current_turn=0,
        mode=mode,
    )


def current_player(state: GameState) -> Player:
    return state.get_player(state.current_turn)


def next_player_index(state: GameState, from_index: int | None = None) -> int:
    start = state.current_turn if from_index is None else from_index
    count = len(state.players)
    for offset in range(1, count + 1):
        candidate = (start + offset) % count
        if state.players[candidate].active:
            return candidate
    return start


def check_winner(state: GameState) -> int | None:
    for player in state.players:
        if player.active and is_goal_reached(player.pos, player.goal, state.board_size):
            return player.id
    return None


def _orthogonal_directions(dx: int, dy: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if dx != 0:
        return ((0, -1), (0, 1))
    return ((-1, 0), (1, 0))


def generate_pawn_moves(state: GameState, player_id: int) -> list[MovePawnAction]:
    player = state.get_player(player_id)
    occupied = occupied_positions(state)
    moves: set[tuple[int, int]] = set()

    for dx, dy in DIRECTIONS.values():
        adjacent = player.pos.moved(dx, dy)
        if not is_in_bounds(adjacent, state.board_size):
            continue
        if edge_blocked(state, player.pos, adjacent):
            continue

        if (adjacent.x, adjacent.y) not in occupied:
            moves.add((adjacent.x, adjacent.y))
            continue

        jump = adjacent.moved(dx, dy)
        can_jump_forward = (
            is_in_bounds(jump, state.board_size)
            and not edge_blocked(state, adjacent, jump)
            and (jump.x, jump.y) not in occupied
        )
        if can_jump_forward:
            moves.add((jump.x, jump.y))
            continue

        for ox, oy in _orthogonal_directions(dx, dy):
            diagonal = adjacent.moved(ox, oy)
            if not is_in_bounds(diagonal, state.board_size):
                continue
            if edge_blocked(state, adjacent, diagonal):
                continue
            if (diagonal.x, diagonal.y) in occupied:
                continue
            moves.add((diagonal.x, diagonal.y))

    return [MovePawnAction(to_x=x, to_y=y) for x, y in sorted(moves)]


def _wall_conflicts(state: GameState, x: int, y: int, orientation: str) -> bool:
    if orientation == "H":
        return (
            (x, y) in state.walls_h
            or (x - 1, y) in state.walls_h
            or (x + 1, y) in state.walls_h
            or (x, y) in state.walls_v
        )
    return (
        (x, y) in state.walls_v
        or (x, y - 1) in state.walls_v
        or (x, y + 1) in state.walls_v
        or (x, y) in state.walls_h
    )


def _place_wall_unchecked(state: GameState, action: PlaceWallAction) -> GameState:
    if action.orientation == "H":
        return replace(state, walls_h=frozenset(set(state.walls_h) | {(action.x, action.y)}))
    return replace(state, walls_v=frozenset(set(state.walls_v) | {(action.x, action.y)}))


def _is_wall_legal(state: GameState, player_id: int, action: PlaceWallAction) -> bool:
    player = state.get_player(player_id)
    if player.walls_left <= 0:
        return False
    if action.orientation not in ORIENTATIONS:
        return False
    if not wall_in_bounds(state, action.x, action.y):
        return False
    if _wall_conflicts(state, action.x, action.y, action.orientation):
        return False

    cloned = _place_wall_unchecked(state, action)
    for other in cloned.players:
        if shortest_path_len(cloned, other.id) is None:
            return False
    return True


def is_wall_legal(state: GameState, player_id: int, action: PlaceWallAction) -> bool:
    return _is_wall_legal(state, player_id, action)


def generate_wall_actions(state: GameState, player_id: int) -> list[PlaceWallAction]:
    player = state.get_player(player_id)
    if player.walls_left <= 0:
        return []

    actions: list[PlaceWallAction] = []
    for x in range(state.board_size - 1):
        for y in range(state.board_size - 1):
            for orientation in ("H", "V"):
                action = PlaceWallAction(x=x, y=y, orientation=orientation)
                if _is_wall_legal(state, player_id, action):
                    actions.append(action)
    return actions


def generate_legal_actions(state: GameState, player_id: int | None = None) -> list[Action]:
    actor_id = state.current_turn if player_id is None else player_id
    return [*generate_pawn_moves(state, actor_id), *generate_wall_actions(state, actor_id)]


def is_action_legal(state: GameState, action: Action, player_id: int | None = None) -> bool:
    actor_id = state.current_turn if player_id is None else player_id
    if state.winner_id is not None:
        return False
    if actor_id != state.current_turn:
        return False

    if isinstance(action, MovePawnAction):
        return any(candidate == action for candidate in generate_pawn_moves(state, actor_id))
    if isinstance(action, PlaceWallAction):
        return _is_wall_legal(state, actor_id, action)
    return False


def _action_to_record(player_id: int, action: Action) -> dict[str, object]:
    if isinstance(action, MovePawnAction):
        return {
            "player_id": player_id,
            "kind": action.kind,
            "to_x": action.to_x,
            "to_y": action.to_y,
        }
    return {
        "player_id": player_id,
        "kind": action.kind,
        "x": action.x,
        "y": action.y,
        "orientation": action.orientation,
    }


def _apply_action_internal(
    state: GameState,
    action: Action,
    *,
    validate: bool,
    record_history: bool,
) -> GameState:
    actor_id = state.current_turn
    if validate and not is_action_legal(state, action, actor_id):
        raise ValueError(f"Illegal action for player {actor_id}: {action}")

    player = state.get_player(actor_id)
    updated = state
    if isinstance(action, MovePawnAction):
        moved_player = replace(player, pos=Position(action.to_x, action.to_y))
        updated = updated.replace_player(actor_id, moved_player)
    else:
        moved_player = replace(player, walls_left=player.walls_left - 1)
        updated = updated.replace_player(actor_id, moved_player)
        updated = _place_wall_unchecked(updated, action)

    winner_id = check_winner(updated)
    move_history = updated.move_history
    if record_history:
        history = list(updated.move_history)
        history.append(_action_to_record(actor_id, action))
        move_history = tuple(history)

    return replace(
        updated,
        current_turn=actor_id if winner_id is not None else next_player_index(updated, actor_id),
        move_count=state.move_count + 1,
        winner_id=winner_id,
        move_history=move_history,
    )


def simulate_action(state: GameState, action: Action) -> GameState:
    return _apply_action_internal(state, action, validate=False, record_history=False)


def apply_action(state: GameState, action: Action) -> GameState:
    return _apply_action_internal(state, action, validate=True, record_history=True)
