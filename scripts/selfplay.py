from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quoridor.ai import choose_mcts_action, choose_minimax_action, choose_random_action
from quoridor.core import apply_action, create_initial_state


def choose_action(state):
    player = state.get_player(state.current_turn)
    if player.kind == "random":
        return choose_random_action(state, state.current_turn)
    if player.kind == "minimax":
        return choose_minimax_action(state, state.current_turn, depth=2)
    return choose_mcts_action(state, state.current_turn, iterations=150, time_budget=0.35)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a console self-play match.")
    parser.add_argument("--turn-limit", type=int, default=120)
    args = parser.parse_args()

    state = create_initial_state(4, ["mcts", "random", "random", "mcts"], "LOCAL_AI")
    print("Starting self-play match")
    while state.winner_id is None and state.move_count < args.turn_limit:
        actor_id = state.current_turn
        action = choose_action(state)
        state = apply_action(state, action)
        print(f"Turn {state.move_count:03d}: player {actor_id + 1} -> {action}")

    if state.winner_id is None:
        print("Reached turn limit without a winner.")
    else:
        print(f"Winner: Player {state.winner_id + 1}")


if __name__ == "__main__":
    main()
