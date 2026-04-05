"""Simple random AI."""

from __future__ import annotations

import random

from quoridor.core import Action, GameState, generate_legal_actions


def choose_random_action(state: GameState, player_id: int | None = None, rng: random.Random | None = None) -> Action:
    actor_id = state.current_turn if player_id is None else player_id
    legal_actions = generate_legal_actions(state, actor_id)
    if not legal_actions:
        raise ValueError("No legal actions available.")
    chooser = rng or random
    return chooser.choice(legal_actions)
