"""Training loop with replay buffer and TensorBoard logging."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from quoridor.ml.config import AlphaZeroConfig
from quoridor.ml.network import QuoridorNet, create_network, init_params
from quoridor.ml.self_play import TrainingExample


class ReplayBuffer:
    """Circular buffer of training examples."""

    def __init__(self, max_size: int):
        self._buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add(self, examples: list[TrainingExample]) -> None:
        self._buffer.extend(examples)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        indices = rng.integers(0, len(self._buffer), size=batch_size)
        states = np.stack([self._buffer[i].state for i in indices])
        policies = np.stack([self._buffer[i].policy for i in indices])
        values = np.array([self._buffer[i].value for i in indices], dtype=np.float32)
        return {"states": states, "policies": policies, "values": values}

    def __len__(self) -> int:
        return len(self._buffer)


def create_optimizer(config: AlphaZeroConfig):
    """Create an Optax optimizer chain."""
    schedule = optax.cosine_decay_schedule(
        init_value=config.lr,
        decay_steps=config.epochs_per_iter * 1000,  # approximate
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )


def _loss_fn(params, batch_net_state, net, batch):
    """Compute AlphaZero loss: policy cross-entropy + value MSE."""
    variables = {"params": params, **batch_net_state}
    (policy_logits, values), updates = net.apply(
        variables, batch["states"], train=True, mutable=["batch_stats"]
    )

    # Policy loss: cross-entropy with MCTS target
    log_probs = jax.nn.log_softmax(policy_logits)
    policy_loss = -jnp.mean(jnp.sum(batch["policies"] * log_probs, axis=-1))

    # Value loss: MSE
    value_loss = jnp.mean((values - batch["values"]) ** 2)

    total_loss = policy_loss + value_loss
    return total_loss, (policy_loss, value_loss, updates)


class Trainer:
    """Manages the training loop and optimizer state."""

    def __init__(self, config: AlphaZeroConfig, rng_seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(rng_seed)
        self.jax_rng = jax.random.PRNGKey(rng_seed)

        # Network
        self.net = create_network(config)
        variables = init_params(self.net, self.jax_rng)
        self.params = variables["params"]
        self.batch_stats = variables.get("batch_stats", {})

        # Optimizer
        self.optimizer = create_optimizer(config)
        self.opt_state = self.optimizer.init(self.params)

        # Replay buffer
        self.buffer = ReplayBuffer(config.replay_buffer_size)

        self.global_step = 0

    def get_variables(self):
        """Get variables dict for inference."""
        return {"params": self.params, "batch_stats": self.batch_stats}

    @staticmethod
    def _train_step(params, batch_stats, opt_state, optimizer, net, batch):
        """Single gradient update step."""
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (total_loss, (policy_loss, value_loss, updates)), grads = grad_fn(
            params, {"batch_stats": batch_stats}, net, batch
        )
        param_updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, param_updates)
        new_batch_stats = updates["batch_stats"]
        return new_params, new_batch_stats, new_opt_state, {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
        }

    def train_epoch(self, writer=None) -> dict[str, float]:
        """Run one epoch of training over the replay buffer.

        Args:
            writer: Optional TensorBoard SummaryWriter.

        Returns:
            Average losses for the epoch.
        """
        if len(self.buffer) < self.config.batch_size:
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        num_batches = max(1, len(self.buffer) // self.config.batch_size)
        epoch_losses = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        for _ in range(num_batches):
            batch = self.buffer.sample(self.config.batch_size, self.rng)
            # Convert to jax arrays
            jax_batch = {k: jnp.array(v) for k, v in batch.items()}

            self.params, self.batch_stats, self.opt_state, losses = self._train_step(
                self.params, self.batch_stats, self.opt_state,
                self.optimizer, self.net, jax_batch,
            )

            for k, v in losses.items():
                epoch_losses[k] += float(v)

            self.global_step += 1

            if writer is not None:
                writer.add_scalar("loss/total", float(losses["total_loss"]), self.global_step)
                writer.add_scalar("loss/policy", float(losses["policy_loss"]), self.global_step)
                writer.add_scalar("loss/value", float(losses["value_loss"]), self.global_step)

        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        return epoch_losses
