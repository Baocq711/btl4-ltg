"""Background worker for manual move and wall previews."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from quoridor.core import GameState, MovePawnAction, PlaceWallAction, generate_pawn_moves, generate_wall_actions


@dataclass(frozen=True)
class ManualActionPreview:
    token: tuple[int, int, int, str]
    actions: tuple[MovePawnAction | PlaceWallAction, ...] = ()
    error: str | None = None


class ManualActionWorker:
    def __init__(self) -> None:
        self._tasks: "queue.Queue[None | tuple[tuple[int, int, int, str], GameState, str]]" = queue.Queue()
        self._results: "queue.Queue[ManualActionPreview]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, token: tuple[int, int, int, str], state: GameState, selected_mode: str) -> None:
        self._tasks.put((token, state, selected_mode))

    def poll_results(self) -> list[ManualActionPreview]:
        results: list[ManualActionPreview] = []
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

            token, state, selected_mode = task
            try:
                if selected_mode == "MOVE":
                    actions = tuple(generate_pawn_moves(state, state.current_turn))
                elif selected_mode in {"H", "V"}:
                    actions = tuple(
                        action
                        for action in generate_wall_actions(state, state.current_turn)
                        if action.orientation == selected_mode
                    )
                else:
                    raise ValueError(f"Unsupported manual action mode: {selected_mode}")
                self._results.put(ManualActionPreview(token=token, actions=actions))
            except Exception as exc:  # noqa: BLE001
                self._results.put(ManualActionPreview(token=token, error=str(exc)))
