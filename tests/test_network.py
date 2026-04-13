"""Tests for neural network and checkpoint save/load."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quoridor.ml.config import AlphaZeroConfig
from quoridor.ml.network import QuoridorNet, create_network, init_params, predict


class TestNetwork:
    def test_forward_pass_shape(self):
        config = AlphaZeroConfig(num_res_blocks=2, num_filters=32)
        net = create_network(config)
        rng = jax.random.PRNGKey(0)
        variables = init_params(net, rng)

        batch = jnp.zeros((4, 8, 9, 9))
        policy_logits, values = predict(net, variables, batch)

        assert policy_logits.shape == (4, 209)
        assert values.shape == (4,)

    def test_single_sample(self):
        config = AlphaZeroConfig(num_res_blocks=1, num_filters=16)
        net = create_network(config)
        rng = jax.random.PRNGKey(42)
        variables = init_params(net, rng)

        x = jnp.ones((1, 8, 9, 9))
        policy_logits, values = predict(net, variables, x)

        assert policy_logits.shape == (1, 209)
        assert values.shape == (1,)
        # Value should be in [-1, 1] due to tanh
        assert -1.0 <= float(values[0]) <= 1.0

    def test_policy_logits_not_all_same(self):
        config = AlphaZeroConfig(num_res_blocks=2, num_filters=32)
        net = create_network(config)
        rng = jax.random.PRNGKey(7)
        variables = init_params(net, rng)

        x = jax.random.normal(rng, (1, 8, 9, 9))
        policy_logits, _ = predict(net, variables, x)

        # With random input and weights, logits should not be uniform
        logits = np.array(policy_logits[0])
        assert logits.std() > 0.01


class TestCheckpoint:
    def test_save_load_round_trip(self):
        from quoridor.ml.checkpoint import load_checkpoint, save_checkpoint

        config = AlphaZeroConfig(num_res_blocks=1, num_filters=16)
        net = create_network(config)
        rng = jax.random.PRNGKey(0)
        variables = init_params(net, rng)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = save_checkpoint(variables, None, step=100, path=tmpdir)
            loaded_vars, _, step = load_checkpoint(ckpt_path, variables)
            assert step == 100

            # Check params match
            orig_leaves = jax.tree.leaves(variables)
            loaded_leaves = jax.tree.leaves(loaded_vars)
            assert len(orig_leaves) == len(loaded_leaves)
            for a, b in zip(orig_leaves, loaded_leaves):
                np.testing.assert_array_almost_equal(np.array(a), np.array(b))

    def test_list_checkpoints(self):
        from quoridor.ml.checkpoint import list_checkpoints, save_checkpoint

        config = AlphaZeroConfig(num_res_blocks=1, num_filters=16)
        net = create_network(config)
        rng = jax.random.PRNGKey(0)
        variables = init_params(net, rng)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(variables, None, step=10, path=tmpdir)
            save_checkpoint(variables, None, step=20, path=tmpdir)
            save_checkpoint(variables, None, step=5, path=tmpdir)

            ckpts = list_checkpoints(tmpdir)
            assert len(ckpts) == 3
            assert ckpts[0].name == "step_000005"
            assert ckpts[-1].name == "step_000020"
