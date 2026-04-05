import time
import unittest

from quoridor.client.ai_worker import AIWorker
from quoridor.core import create_initial_state, generate_legal_actions


class AIWorkerTests(unittest.TestCase):
    def test_worker_returns_legal_action(self) -> None:
        worker = AIWorker()
        try:
            state = create_initial_state(2, ["random", "human"], "LOCAL_AI")
            token = (1, state.move_count, state.current_turn)
            worker.submit(token, state)

            deadline = time.perf_counter() + 2.0
            result = None
            while time.perf_counter() < deadline:
                results = worker.poll_results()
                if results:
                    result = results[0]
                    break
                time.sleep(0.01)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.token, token)
            self.assertIsNone(result.error)
            self.assertIn(result.action, generate_legal_actions(state, 0))
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
