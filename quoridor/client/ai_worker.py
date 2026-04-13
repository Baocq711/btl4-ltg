"""Background AI worker so pygame rendering stays responsive during bot turns."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from quoridor.ai import choose_mcts_action, choose_minimax_action, choose_neural_action, choose_random_action
from quoridor.core import Action, GameState


@dataclass(frozen=True)
class AIResult:
    token: tuple[int, int, int]
    action: Action | None = None
    error: str | None = None


class AIWorker:
    def __init__(self) -> None:
        self._tasks: "queue.Queue[tuple[int, int, int] | None | tuple[tuple[int, int, int], GameState]]" = queue.Queue()
        self._results: "queue.Queue[AIResult]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, token: tuple[int, int, int], state: GameState) -> None:
        self._tasks.put((token, state))

    def poll_results(self) -> list[AIResult]:
        results: list[AIResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def close(self) -> None:
        self._stop.set()
        self._tasks.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._tasks.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:
                continue

            token, state = task
            try:
                action = choose_ai_action(state)
                self._results.put(AIResult(token=token, action=action))
            except Exception as exc:  # noqa: BLE001
                self._results.put(AIResult(token=token, error=str(exc)))


def choose_ai_action(state: GameState) -> Action:
    player = state.get_player(state.current_turn)
    if player.kind == "random":
        return choose_random_action(state, state.current_turn)
    if player.kind == "minimax":
        return choose_minimax_action(state, state.current_turn, depth=2)
    if player.kind == "mcts":
        return choose_mcts_action(state, state.current_turn, iterations=300, time_budget=1.0, rollout_depth=20)
    if player.kind == "neural":
        return choose_neural_action(state, state.current_turn)
    raise ValueError(f"Player kind '{player.kind}' is not AI-controlled.")
