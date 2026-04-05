import unittest

from quoridor.ai import choose_mcts_action, choose_minimax_action, choose_random_action
from quoridor.core import create_initial_state, generate_legal_actions


class AITests(unittest.TestCase):
    def test_random_ai_returns_legal_action(self) -> None:
        state = create_initial_state(2, ["random", "human"], "LOCAL_AI")
        action = choose_random_action(state, 0)
        self.assertIn(action, generate_legal_actions(state, 0))

    def test_minimax_returns_legal_action(self) -> None:
        state = create_initial_state(2, ["minimax", "human"], "LOCAL_AI")
        action = choose_minimax_action(state, 0, depth=1)
        self.assertIn(action, generate_legal_actions(state, 0))

    def test_mcts_returns_legal_action(self) -> None:
        state = create_initial_state(4, ["mcts", "human", "human", "human"], "LOCAL_AI")
        action = choose_mcts_action(state, 0, iterations=12, time_budget=0.05, rollout_depth=8, seed=7)
        self.assertIn(action, generate_legal_actions(state, 0))


if __name__ == "__main__":
    unittest.main()
