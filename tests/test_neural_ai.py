"""Tests for neural AI integration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from quoridor.core import create_initial_state, generate_legal_actions, apply_action
from quoridor.ml.config import AlphaZeroConfig
from quoridor.ml.network import create_network, init_params
from quoridor.ml.checkpoint import save_checkpoint


class TestNeuralAI:
    @pytest.fixture()
    def checkpoint_dir(self, tmp_path):
        config = AlphaZeroConfig(num_res_blocks=1, num_filters=16, num_simulations=10)
        net = create_network(config)
        rng = jax.random.PRNGKey(0)
        variables = init_params(net, rng)

        save_checkpoint(variables, None, step=1, path=str(tmp_path))
        # Save config for neural_ai to load
        config.save(str(tmp_path / "config.json"))
        return str(tmp_path)

    def test_choose_neural_action_is_legal(self, checkpoint_dir):
        from quoridor.ai.neural_ai import choose_neural_action, clear_cache

        clear_cache()  # Ensure fresh load
        state = create_initial_state(num_players=2)
        action = choose_neural_action(state, player_id=0, checkpoint_dir=checkpoint_dir)

        legal = generate_legal_actions(state)
        assert action in legal

    def test_choose_neural_action_player2(self, checkpoint_dir):
        from quoridor.ai.neural_ai import choose_neural_action, clear_cache

        clear_cache()
        state = create_initial_state(num_players=2)
        # Make a move so it's player 2's turn
        legal = generate_legal_actions(state)
        state = apply_action(state, list(legal)[0])

        action = choose_neural_action(state, player_id=1, checkpoint_dir=checkpoint_dir)
        legal2 = generate_legal_actions(state)
        assert action in legal2
