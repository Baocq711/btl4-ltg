"""Tests for AlphaZero MCTS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quoridor.core import create_initial_state, generate_legal_actions
from quoridor.ml.config import AlphaZeroConfig
from quoridor.ml.network import create_network, init_params, predict
from quoridor.ml.alpha_mcts import AlphaMCTS


class TestAlphaMCTS:
    @pytest.fixture()
    def mcts(self):
        config = AlphaZeroConfig(
            num_res_blocks=1,
            num_filters=16,
            num_simulations=20,
            c_puct=1.5,
        )
        net = create_network(config)
        rng = jax.random.PRNGKey(0)
        variables = init_params(net, rng)

        def pred_fn(variables, x):
            return predict(net, variables, x)

        return AlphaMCTS(pred_fn, variables, config)

    def test_search_returns_valid_probs(self, mcts):
        state = create_initial_state(num_players=2)
        action_probs, root_value = mcts.search(state, player_id=0)

        # action_probs is ndarray of shape (209,)
        assert isinstance(action_probs, np.ndarray)
        assert action_probs.shape == (209,)

        total = action_probs.sum()
        np.testing.assert_almost_equal(total, 1.0, decimal=5)

        # All probabilities should be non-negative
        assert (action_probs >= 0.0).all()

    def test_search_only_legal_actions(self, mcts):
        state = create_initial_state(num_players=2)
        action_probs, _ = mcts.search(state, player_id=0)

        legal = generate_legal_actions(state)
        from quoridor.ml.encoding import index_to_action, _needs_flip
        flip = _needs_flip(state, state.current_turn)
        for idx in range(209):
            if action_probs[idx] > 0:
                action = index_to_action(idx, flip=flip)
                assert action in legal, f"MCTS returned illegal action at idx {idx}: {action}"

    def test_select_action_returns_legal(self, mcts):
        state = create_initial_state(num_players=2)
        action_probs, _ = mcts.search(state, player_id=0)

        action_idx = mcts.select_action(action_probs, temperature=1.0)
        assert 0 <= action_idx < 209
        assert action_probs[action_idx] > 0

    def test_select_action_greedy(self, mcts):
        state = create_initial_state(num_players=2)
        action_probs, _ = mcts.search(state, player_id=0)

        # With temperature close to 0, should pick the most visited action
        action_idx = mcts.select_action(action_probs, temperature=0.01)
        best_idx = int(np.argmax(action_probs))
        assert action_idx == best_idx

    def test_root_value_bounded(self, mcts):
        state = create_initial_state(num_players=2)
        _, root_value = mcts.search(state, player_id=0)
        assert 0.0 <= root_value <= 1.0  # q_value mapped to [0,1]
