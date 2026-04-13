"""AlphaZero-style MCTS guided by a policy-value neural network."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from quoridor.core import (
    Action,
    GameState,
    MovePawnAction,
    PlaceWallAction,
    generate_legal_actions,
    simulate_action,
)
from quoridor.ml.encoding import (
    ACTION_SIZE,
    action_to_index,
    encode_state,
    index_to_action,
    legal_actions_mask,
)
from quoridor.ml.config import AlphaZeroConfig


@dataclass
class AlphaMCTSNode:
    state: GameState
    player_id: int  # root player (the one we're optimizing for)
    parent: "AlphaMCTSNode | None" = None
    parent_action_idx: int | None = None
    children: dict[int, "AlphaMCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _puct_score(parent: AlphaMCTSNode, child: AlphaMCTSNode, c_puct: float) -> float:
    """Compute PUCT score for child selection."""
    q = child.q_value
    u = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return q + u


def _select_child(node: AlphaMCTSNode, c_puct: float) -> AlphaMCTSNode:
    """Select child with highest PUCT score."""
    best_score = -float("inf")
    best_child = None
    for child in node.children.values():
        score = _puct_score(node, child, c_puct)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def _expand_node(
    node: AlphaMCTSNode,
    policy: np.ndarray,
    legal_mask: np.ndarray,
) -> None:
    """Expand a leaf node using network policy output."""
    # Mask illegal actions and renormalize
    masked_policy = policy * legal_mask
    policy_sum = masked_policy.sum()
    if policy_sum > 0:
        masked_policy /= policy_sum
    else:
        # Uniform over legal actions if no probability mass
        masked_policy = legal_mask / max(legal_mask.sum(), 1)

    flip = node.state.get_player(node.state.current_turn).goal == "TOP"

    for idx in range(ACTION_SIZE):
        if legal_mask[idx] > 0:
            action = index_to_action(idx, flip=flip)
            next_state = simulate_action(node.state, action)
            child = AlphaMCTSNode(
                state=next_state,
                player_id=node.player_id,
                parent=node,
                parent_action_idx=idx,
                prior=float(masked_policy[idx]),
            )
            node.children[idx] = child
    node.is_expanded = True


def _backpropagate(node: AlphaMCTSNode, value: float, root_player: int) -> None:
    """Backpropagate value up the tree, negating for opponent turns."""
    current = node
    while current is not None:
        current.visit_count += 1
        # Value is from root_player's perspective
        if current.state.current_turn == root_player:
            current.value_sum += value
        else:
            current.value_sum += (1.0 - value)  # opponent: inverted
        current = current.parent


def _add_dirichlet_noise(
    node: AlphaMCTSNode, alpha: float, epsilon: float, rng: np.random.Generator
) -> None:
    """Add Dirichlet noise to root node priors for exploration."""
    if not node.children:
        return
    actions = list(node.children.keys())
    noise = rng.dirichlet([alpha] * len(actions))
    for i, action_idx in enumerate(actions):
        child = node.children[action_idx]
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


class AlphaMCTS:
    """Neural MCTS that uses a policy-value network instead of rollouts."""

    def __init__(self, predict_fn, variables, config: AlphaZeroConfig):
        """
        Args:
            predict_fn: A callable (variables, x) -> (policy_logits, values)
                        where x is (batch, 8, 9, 9) and outputs are batched.
            variables: Network parameters/state for predict_fn.
            config: AlphaZero configuration.
        """
        self.predict_fn = predict_fn
        self.variables = variables
        self.config = config

    def _evaluate(self, state: GameState, player_id: int):
        """Run NN on a single state. Returns (policy_probs, value)."""
        encoded = encode_state(state, state.current_turn)
        x = jnp.array(encoded[np.newaxis])  # (1, 8, 9, 9)
        logits, value = self.predict_fn(self.variables, x)
        policy = np.array(jax.nn.softmax(logits[0]))
        val = float(value[0])
        # Convert value to root player's perspective
        if state.current_turn != player_id:
            val = -val
        # Map from [-1, 1] to [0, 1]
        val = (val + 1.0) / 2.0
        return policy, val

    def search(
        self,
        state: GameState,
        player_id: int,
        seed: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run MCTS search from the given state.

        Returns:
            action_probs: Array of shape (ACTION_SIZE,) with visit-count based probabilities.
            root_value: Estimated value of the root state.
        """
        rng = np.random.default_rng(seed)

        # Create and expand root
        root = AlphaMCTSNode(state=state, player_id=player_id)
        mask = legal_actions_mask(state, state.current_turn)
        policy, value = self._evaluate(state, player_id)
        _expand_node(root, policy, np.array(mask))
        _backpropagate(root, value, player_id)
        _add_dirichlet_noise(root, self.config.dirichlet_alpha, self.config.dirichlet_epsilon, rng)

        # Run simulations
        for _ in range(self.config.num_simulations - 1):
            node = root

            # SELECT: traverse tree to leaf
            while node.is_expanded and node.children and node.state.winner_id is None:
                node = _select_child(node, self.config.c_puct)

            # Terminal check
            if node.state.winner_id is not None:
                val = 1.0 if node.state.winner_id == player_id else 0.0
                _backpropagate(node, val, player_id)
                continue

            # EXPAND + EVALUATE
            mask = legal_actions_mask(node.state, node.state.current_turn)
            policy, value = self._evaluate(node.state, player_id)
            _expand_node(node, policy, np.array(mask))

            # BACKPROPAGATE
            _backpropagate(node, value, player_id)

        # Extract action probabilities from visit counts
        action_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action_idx, child in root.children.items():
            action_probs[action_idx] = child.visit_count
        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        root_value = root.q_value
        return action_probs, root_value

    def select_action(
        self,
        action_probs: np.ndarray,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Select an action index from MCTS probabilities.

        Args:
            action_probs: Visit-count-based action distribution.
            temperature: 1.0 = proportional sampling, near 0 = greedy.
            rng: Numpy random generator.
        """
        if rng is None:
            rng = np.random.default_rng()

        if temperature < 1e-6:
            # Greedy: pick most-visited action
            return int(np.argmax(action_probs))

        # Apply temperature
        probs = action_probs ** (1.0 / temperature)
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs /= prob_sum
        else:
            # Shouldn't happen, but fallback to uniform over nonzero
            nonzero = action_probs > 0
            probs = nonzero.astype(np.float32)
            probs /= probs.sum()

        return int(rng.choice(len(probs), p=probs))


# Need to import jax for softmax
import jax
