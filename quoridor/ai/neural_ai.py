"""Neural network AI wrapper for game integration.

Provides the same interface as other AI functions: (state, player_id) -> Action.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from quoridor.core import Action, GameState
from quoridor.ml.config import AlphaZeroConfig
from quoridor.ml.encoding import index_to_action

logger = logging.getLogger(__name__)

# Module-level singleton for lazy-loaded model
_cached_mcts: object | None = None
_cached_checkpoint_path: str | None = None


def _get_mcts(checkpoint_dir: str | None = None):
    """Lazy-load model and return an AlphaMCTS instance."""
    global _cached_mcts, _cached_checkpoint_path

    # Use default path if not specified
    if checkpoint_dir is None:
        checkpoint_dir = "checkpoints"

    # Return cached if same checkpoint
    if _cached_mcts is not None and _cached_checkpoint_path == checkpoint_dir:
        return _cached_mcts

    # Lazy imports to avoid loading JAX when not needed
    import jax
    from quoridor.ml.alpha_mcts import AlphaMCTS
    from quoridor.ml.checkpoint import load_latest
    from quoridor.ml.network import create_network, init_params, predict

    config = AlphaZeroConfig()

    # Try loading config from checkpoint dir
    config_path = Path(checkpoint_dir) / "config.json"
    if config_path.exists():
        config = AlphaZeroConfig.load(config_path)

    net = create_network(config)
    rng_key = jax.random.PRNGKey(0)
    variables = init_params(net, rng_key)

    # Try loading trained weights
    result = load_latest(checkpoint_dir, variables)
    if result is not None:
        loaded_vars, _, step = result
        variables = loaded_vars
        logger.info("Loaded neural model from step %d", step)
    else:
        logger.warning("No checkpoint found in %s, using random weights", checkpoint_dir)

    def _predict_fn(variables, x):
        return predict(net, variables, x)

    mcts = AlphaMCTS(_predict_fn, variables, config)
    _cached_mcts = mcts
    _cached_checkpoint_path = checkpoint_dir
    return mcts


def choose_neural_action(
    state: GameState,
    player_id: int | None = None,
    checkpoint_dir: str | None = None,
) -> Action:
    """Choose an action using the neural MCTS agent.

    Args:
        state: Current game state.
        player_id: Player to act for (defaults to current_turn).
        checkpoint_dir: Path to checkpoint directory.

    Returns:
        The selected action.
    """
    actor_id = state.current_turn if player_id is None else player_id
    if actor_id != state.current_turn:
        raise ValueError("Neural AI must act for the current turn player.")

    mcts = _get_mcts(checkpoint_dir)
    rng = np.random.default_rng()

    # Use low temperature for play (near-greedy)
    action_probs, _ = mcts.search(state, actor_id, seed=int(rng.integers(0, 2**31)))
    action_idx = mcts.select_action(action_probs, temperature=0.1, rng=rng)

    flip = state.get_player(state.current_turn).goal == "TOP"
    return index_to_action(action_idx, flip=flip)


def clear_cache() -> None:
    """Clear cached model (e.g. to reload after training)."""
    global _cached_mcts, _cached_checkpoint_path
    _cached_mcts = None
    _cached_checkpoint_path = None
