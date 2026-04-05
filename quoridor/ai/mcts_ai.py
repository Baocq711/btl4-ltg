"""Monte Carlo Tree Search AI for multiplayer Quoridor."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from quoridor.core import Action, GameState, MovePawnAction, simulate_action
from quoridor.core.pathfinding import shortest_path_len

from .action_pruning import pruned_actions
from .random_ai import choose_random_action


@dataclass
class MCTSNode:
    state: GameState
    root_player_id: int
    parent: "MCTSNode | None" = None
    action: Action | None = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    reward: float = 0.0
    untried_actions: list[Action] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.untried_actions:
            self.untried_actions = pruned_actions(self.state, self.state.current_turn, wall_limit=10)
            self.untried_actions.sort(key=lambda action: (0 if isinstance(action, MovePawnAction) else 1))

    def best_child(self, exploration: float) -> "MCTSNode":
        def score(child: MCTSNode) -> float:
            if child.visits == 0:
                return float("inf")
            exploit = child.reward / child.visits
            explore = exploration * math.sqrt(math.log(max(1, self.visits)) / child.visits)
            return exploit + explore

        return max(self.children, key=score)


def _rollout_policy(state: GameState, rng: random.Random) -> Action:
    actions = pruned_actions(state, state.current_turn, wall_limit=6)
    if not actions:
        return choose_random_action(state, state.current_turn, rng)

    pawn_actions = [action for action in actions if isinstance(action, MovePawnAction)]
    if pawn_actions and rng.random() < 0.85:
        player = state.get_player(state.current_turn)
        before = shortest_path_len(state, player.id) or 99
        scored_moves = []
        for action in pawn_actions:
            nxt = simulate_action(state, action)
            after = shortest_path_len(nxt, player.id) or 99
            scored_moves.append((after - before, rng.random(), action))
        scored_moves.sort(key=lambda item: (item[0], item[1]))
        return scored_moves[0][2]

    return rng.choice(actions)


def _simulate(state: GameState, root_player_id: int, rollout_depth: int, rng: random.Random) -> float:
    current = state
    for _ in range(rollout_depth):
        if current.winner_id is not None:
            break
        action = _rollout_policy(current, rng)
        current = simulate_action(current, action)

    if current.winner_id is None:
        my_distance = shortest_path_len(current, root_player_id) or 99
        other_best = min(
            shortest_path_len(current, player.id) or 99
            for player in current.players
            if player.id != root_player_id and player.active
        )
        return 0.6 if my_distance <= other_best else 0.2
    return 1.0 if current.winner_id == root_player_id else 0.0


def choose_mcts_action(
    state: GameState,
    player_id: int | None = None,
    iterations: int = 250,
    time_budget: float = 1.0,
    rollout_depth: int = 20,
    seed: int | None = None,
) -> Action:
    actor_id = state.current_turn if player_id is None else player_id
    if actor_id != state.current_turn:
        raise ValueError("MCTS must act for the current turn player.")

    rng = random.Random(seed)
    root = MCTSNode(state=state, root_player_id=actor_id)
    deadline = time.perf_counter() + time_budget
    completed = 0

    while completed < iterations and time.perf_counter() < deadline:
        node = root

        while not node.untried_actions and node.children and node.state.winner_id is None:
            node = node.best_child(exploration=1.2)

        if node.untried_actions and node.state.winner_id is None:
            action = node.untried_actions.pop(0)
            next_state = simulate_action(node.state, action)
            child = MCTSNode(state=next_state, root_player_id=actor_id, parent=node, action=action)
            node.children.append(child)
            node = child

        reward = _simulate(node.state, actor_id, rollout_depth, rng)
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent
        completed += 1

    if not root.children:
        return choose_random_action(state, actor_id, rng)
    best = max(root.children, key=lambda child: (child.visits, child.reward / max(1, child.visits)))
    return best.action
