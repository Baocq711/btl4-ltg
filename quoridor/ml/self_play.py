"""Self-play game generation for AlphaZero training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from quoridor.core import GameState, create_initial_state, simulate_action
from quoridor.ml.alpha_mcts import AlphaMCTS
from quoridor.ml.config import AlphaZeroConfig
from quoridor.ml.encoding import (
    ACTION_SIZE,
    encode_state,
    index_to_action,
    legal_actions_mask,
)


@dataclass
class TrainingExample:
    """A single training sample from self-play."""
    state: np.ndarray       # (8, 9, 9) encoded state
    policy: np.ndarray      # (209,) MCTS visit-count policy
    value: float            # +1 winner, -1 loser (from current player's perspective)


def _play_one_game(
    mcts: AlphaMCTS,
    config: AlphaZeroConfig,
    rng: np.random.Generator,
) -> list[TrainingExample]:
    """Play a single self-play game and collect training examples."""
    state = create_initial_state(num_players=2, player_kinds=["human", "human"], mode="LOCAL")
    examples: list[tuple[np.ndarray, np.ndarray, int]] = []  # (encoded, policy, current_player)

    move_count = 0
    while state.winner_id is None and move_count < 200:
        player_id = state.current_turn
        temperature = 1.0 if move_count < config.temperature_threshold else 0.1

        # Run MCTS search
        action_probs, _ = mcts.search(state, player_id, seed=int(rng.integers(0, 2**31)))

        # Store training example
        encoded = encode_state(state, player_id)
        examples.append((encoded, action_probs, player_id))

        # Select and apply action
        action_idx = mcts.select_action(action_probs, temperature=temperature, rng=rng)
        flip = state.get_player(state.current_turn).goal == "TOP"
        action = index_to_action(action_idx, flip=flip)
        state = simulate_action(state, action)
        move_count += 1

    # Assign values based on outcome
    winner = state.winner_id
    training_examples = []
    for encoded, policy, player_id in examples:
        if winner is None:
            value = 0.0  # draw / timeout
        elif winner == player_id:
            value = 1.0
        else:
            value = -1.0
        training_examples.append(TrainingExample(state=encoded, policy=policy, value=value))

    return training_examples


def generate_self_play_data(
    predict_fn,
    variables,
    config: AlphaZeroConfig,
    seed: int = 0,
) -> list[TrainingExample]:
    """Generate training data from ``num_games_per_iter`` self-play games.

    Games are played sequentially (batched NN calls happen within MCTS).
    For true parallelism across games, run multiple processes.

    Args:
        predict_fn: JIT-compiled inference function (variables, x) -> (logits, values).
        variables: Network parameters.
        config: Training configuration.
        seed: Base random seed.

    Returns:
        All training examples from the self-play games.
    """
    import time as _time

    mcts = AlphaMCTS(predict_fn, variables, config)
    rng = np.random.default_rng(seed)
    all_examples: list[TrainingExample] = []

    for game_idx in range(config.num_games_per_iter):
        game_start = _time.perf_counter()
        game_rng = np.random.default_rng(rng.integers(0, 2**63))
        examples = _play_one_game(mcts, config, game_rng)
        all_examples.extend(examples)
        elapsed = _time.perf_counter() - game_start
        print(f"    Game {game_idx + 1}/{config.num_games_per_iter}: "
              f"{len(examples)} moves in {elapsed:.1f}s", flush=True)

    return all_examples


def generate_self_play_batch(
    predict_fn,
    variables,
    config: AlphaZeroConfig,
    num_games: int | None = None,
    seed: int = 0,
) -> list[TrainingExample]:
    """Convenience wrapper with configurable game count."""
    effective_games = num_games if num_games is not None else config.num_games_per_iter
    orig = config.num_games_per_iter
    config.num_games_per_iter = effective_games
    try:
        return generate_self_play_data(predict_fn, variables, config, seed)
    finally:
        config.num_games_per_iter = orig
