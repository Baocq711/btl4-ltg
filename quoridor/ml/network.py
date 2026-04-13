"""ResNet-based policy-value network for Quoridor using Flax."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn


class ResBlock(nn.Module):
    """Residual block: Conv → BN → ReLU → Conv → BN → skip → ReLU."""

    filters: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        y = nn.Conv(self.filters, (3, 3), padding="SAME", use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(self.filters, (3, 3), padding="SAME", use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        return nn.relu(y + residual)


class QuoridorNet(nn.Module):
    """AlphaZero-style dual-head network.

    Attributes:
        num_res_blocks: Number of residual blocks in the tower.
        num_filters: Number of convolutional filters.
        action_size: Size of the action space (209 for Quoridor).
        value_hidden: Hidden units in the value head MLP.
    """

    num_res_blocks: int = 4
    num_filters: int = 64
    action_size: int = 209
    value_hidden: int = 64

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (batch, channels, height, width) — channels-first
        # Transpose to channels-last for Flax Conv defaults
        x = jnp.transpose(x, (0, 2, 3, 1))  # (batch, H, W, C)

        # Initial conv block
        x = nn.Conv(self.num_filters, (3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Residual tower
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.num_filters)(x, train=train)

        # --- Policy head ---
        p = nn.Conv(2, (1, 1), use_bias=False)(x)
        p = nn.BatchNorm(use_running_average=not train)(p)
        p = nn.relu(p)
        p = p.reshape((p.shape[0], -1))  # flatten
        policy_logits = nn.Dense(self.action_size)(p)

        # --- Value head ---
        v = nn.Conv(1, (1, 1), use_bias=False)(x)
        v = nn.BatchNorm(use_running_average=not train)(v)
        v = nn.relu(v)
        v = v.reshape((v.shape[0], -1))  # flatten
        v = nn.Dense(self.value_hidden)(v)
        v = nn.relu(v)
        value = nn.Dense(1)(v)
        value = nn.tanh(value)

        return policy_logits, value.squeeze(-1)


def create_network(config) -> QuoridorNet:
    """Create a QuoridorNet from an AlphaZeroConfig."""
    return QuoridorNet(
        num_res_blocks=config.num_res_blocks,
        num_filters=config.num_filters,
        action_size=config.action_size,
        value_hidden=config.value_hidden,
    )


def init_params(net: QuoridorNet, rng_key: jax.Array):
    """Initialize network parameters with a dummy input."""
    dummy = jnp.zeros((1, 8, 9, 9))
    variables = net.init(rng_key, dummy, train=False)
    return variables


@partial(jax.jit, static_argnums=(0,))
def predict(net: QuoridorNet, variables, x):
    """Run inference (no batch norm updates)."""
    return net.apply(variables, x, train=False)
