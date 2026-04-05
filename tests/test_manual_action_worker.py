import time
import unittest

from quoridor.client.manual_action_worker import ManualActionWorker
from quoridor.core import MovePawnAction, PlaceWallAction, create_initial_state


class ManualActionWorkerTests(unittest.TestCase):
    def test_worker_returns_only_horizontal_wall_previews(self) -> None:
        worker = ManualActionWorker()
        try:
            state = create_initial_state(2, ["human", "human"], "LOCAL")
            token = (2, state.move_count, state.current_turn, "H")
            worker.submit(token, state, "H")

            deadline = time.perf_counter() + 3.0
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
            self.assertTrue(result.actions)
            self.assertTrue(all(isinstance(action, PlaceWallAction) for action in result.actions))
            self.assertTrue(all(action.orientation == "H" for action in result.actions))
        finally:
            worker.close()

    def test_worker_returns_move_previews(self) -> None:
        worker = ManualActionWorker()
        try:
            state = create_initial_state(2, ["human", "human"], "LOCAL")
            token = (3, state.move_count, state.current_turn, "MOVE")
            worker.submit(token, state, "MOVE")

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
            self.assertTrue(result.actions)
            self.assertTrue(all(isinstance(action, MovePawnAction) for action in result.actions))
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
