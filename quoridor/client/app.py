"""Pygame client for local and online Quoridor."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from quoridor.client.ai_worker import AIWorker
from quoridor.client.config import (
    BACKGROUND,
    BOARD_BG,
    BOARD_ORIGIN,
    BOARD_PIXELS,
    BUTTON,
    BUTTON_HOVER,
    BUTTON_TEXT,
    CELL_SIZE,
    FPS,
    GRID,
    HIGHLIGHT,
    HUD_X,
    PANEL,
    PANEL_BORDER,
    PLAYER_COLORS,
    TEXT,
    WALL_THICKNESS,
    WARNING,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from quoridor.client.manual_action_worker import ManualActionWorker
from quoridor.client.online import OnlineSession
from quoridor.client.widgets import Button, TextInput
from quoridor.core import (
    DEFAULT_SAVE_PATH,
    GameState,
    MovePawnAction,
    PlaceWallAction,
    apply_action,
    create_initial_state,
    generate_pawn_moves,
    load_game,
    save_game,
)
from quoridor.core.serializer import state_from_dict


_DEFAULT_SERVER_URL = "ws://127.0.0.1:8765"
_DEFAULT_PLAYER_NAME = "Player"


def _load_external_config() -> dict[str, str]:
    """Load config.json from next to the executable or the project root."""
    candidates = [
        Path.cwd() / "config.json",
        Path(__file__).resolve().parents[2] / "config.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
    return {}


@dataclass(frozen=True)
class LocalPreset:
    label: str
    num_players: int
    player_kinds: list[str]
    mode: str


PRESETS = {
    "local_2": LocalPreset("Local 2 Players", 2, ["human", "human"], "LOCAL"),
    "local_4": LocalPreset("Local 4 Players", 4, ["human", "human", "human", "human"], "LOCAL"),
    "vs_random": LocalPreset("Human vs Random", 2, ["human", "random"], "LOCAL_AI"),
    "vs_minimax": LocalPreset("Human vs Minimax", 2, ["human", "minimax"], "LOCAL_AI"),
    "vs_mcts": LocalPreset("Human vs 3x MCTS", 4, ["human", "mcts", "mcts", "mcts"], "LOCAL_AI"),
}


class QuoridorApp:
    def __init__(self, server_url: str | None = None, player_name: str | None = None) -> None:
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("The 'pygame' package is required to run the client.") from exc

        self.pg = pygame
        self.pg.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clipboard_enabled = False
        try:
            self.pg.scrap.init()
            self.clipboard_enabled = True
        except self.pg.error:
            self.clipboard_enabled = False
        pygame.display.set_caption("Quoridor 4P")
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.SysFont("georgia", 38)
        self.body_font = pygame.font.SysFont("georgia", 24)
        self.small_font = pygame.font.SysFont("georgia", 18)
        self.scene = "menu"
        self.state: GameState | None = None
        self.selected_mode = "MOVE"
        self.running = True
        self.status = ""
        self.paused = False
        self.local_player_id: int | None = None
        self.online_session: OnlineSession | None = None
        self.online_room_id = ""
        self.online_started = False
        self.online_seat_count = 0
        self.online_num_players = 4
        self.online_max_players = 4
        self.ai_worker = AIWorker()
        self.preview_worker = ManualActionWorker()
        self._session_id = 0
        self._ai_pending_token: tuple[int, int, int] | None = None
        self._manual_action_pending_token: tuple[int, int, int, str] | None = None
        self._manual_action_cache_key: tuple[int, int, int, str] | None = None
        self._manual_action_cache: list[MovePawnAction | PlaceWallAction] = []
        _cfg = _load_external_config()
        _server = server_url or _cfg.get("server_url") or _DEFAULT_SERVER_URL
        _name = player_name or _cfg.get("player_name") or _DEFAULT_PLAYER_NAME
        self._server_url = _server
        self.text_inputs: dict[str, TextInput] = {
            "name": TextInput(pygame.Rect(520, 250, 300, 44), _name, "Player Name"),
            "room": TextInput(pygame.Rect(520, 320, 300, 44), "", "Room Code (blank to create)"),
        }

    def _set_active_text_input(self, target_name: str | None) -> None:
        for name, text_input in self.text_inputs.items():
            text_input.set_active(name == target_name, select_all=name == target_name)

    def _active_text_input(self) -> TextInput | None:
        for text_input in self.text_inputs.values():
            if text_input.active:
                return text_input
        return None

    def _focus_next_text_input(self) -> None:
        names = list(self.text_inputs.keys())
        for index, name in enumerate(names):
            if self.text_inputs[name].active:
                next_name = names[(index + 1) % len(names)]
                self._set_active_text_input(next_name)
                return
        if names:
            self._set_active_text_input(names[0])

    def _clipboard_put(self, text: str) -> bool:
        if not self.clipboard_enabled:
            return False
        try:
            self.pg.scrap.put(self.pg.SCRAP_TEXT, text.encode("utf-8") + b"\x00")
            return True
        except self.pg.error:
            return False

    def _clipboard_get(self) -> str:
        if not self.clipboard_enabled:
            return ""
        try:
            data = self.pg.scrap.get(self.pg.SCRAP_TEXT)
        except self.pg.error:
            return ""
        if not data:
            return ""
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore").replace("\x00", "").replace("\r", "").replace("\n", "")
        return str(data).replace("\x00", "").replace("\r", "").replace("\n", "")

    def _handle_online_input_shortcuts(self, text_input: TextInput, event: object) -> bool:
        if event.key == self.pg.K_TAB:
            self._focus_next_text_input()
            return True

        ctrl_pressed = bool(event.mod & self.pg.KMOD_CTRL)
        if not ctrl_pressed:
            return False

        if event.key == self.pg.K_a:
            text_input.select_all()
            self.status = f"Selected {text_input.name}."
            return True
        if event.key == self.pg.K_c:
            self.status = f"Copied {text_input.name}." if self._clipboard_put(text_input.value) else "Clipboard unavailable."
            return True
        if event.key == self.pg.K_x:
            if self._clipboard_put(text_input.value):
                text_input.clear()
                self.status = f"Cut {text_input.name}."
            else:
                self.status = "Clipboard unavailable."
            return True
        if event.key == self.pg.K_v:
            pasted = self._clipboard_get()
            if pasted:
                text_input.paste_text(pasted)
                self.status = f"Pasted into {text_input.name}."
            else:
                self.status = "Clipboard empty or unavailable."
            return True
        return False

    def run(self) -> None:
        while self.running:
            self._process_events()
            self._update()
            self._draw()
            self.clock.tick(FPS)

        self._shutdown()

    def _shutdown(self) -> None:
        if self.online_session is not None:
            self.online_session.close()
        self.ai_worker.close()
        self.preview_worker.close()
        self.pg.quit()

    def _begin_new_session(self) -> None:
        self._session_id += 1
        self._ai_pending_token = None
        self._invalidate_manual_action_cache()

    def _replace_state(self, state: GameState | None) -> None:
        self.state = state
        self._invalidate_manual_action_cache()

    def _invalidate_manual_action_cache(self) -> None:
        self._manual_action_pending_token = None
        self._manual_action_cache_key = None
        self._manual_action_cache = []

    def _state_token(self, state: GameState) -> tuple[int, int, int]:
        return (self._session_id, state.move_count, state.current_turn)

    def _manual_action_token(self, state: GameState) -> tuple[int, int, int, str]:
        return (self._session_id, state.move_count, state.current_turn, self.selected_mode)

    def _describe_action(self, action: MovePawnAction | PlaceWallAction) -> str:
        if isinstance(action, MovePawnAction):
            return f"moved to ({action.to_x}, {action.to_y})"
        return f"placed {action.orientation} wall at ({action.x}, {action.y})"

    def _start_local_game(self, preset: LocalPreset) -> None:
        self._begin_new_session()
        self._replace_state(create_initial_state(preset.num_players, preset.player_kinds, preset.mode))
        self.scene = "game"
        self.status = preset.label
        self.paused = False
        self.local_player_id = None
        self.online_started = False
        self.selected_mode = "MOVE"

    def _continue_latest(self) -> None:
        save_path = Path(DEFAULT_SAVE_PATH)
        if not save_path.exists():
            self.status = "No save file found for Continue."
            return
        self._begin_new_session()
        self._replace_state(load_game(save_path))
        self.scene = "game"
        self.selected_mode = "MOVE"
        self.paused = False
        self.local_player_id = None
        self.online_started = False
        self.status = f"Loaded save from {save_path}."

    def _open_online_menu(self) -> None:
        self.scene = "online_menu"
        self.status = "Connect to a room, then press Ready when everyone has joined."

    def _connect_online(self) -> None:
        if self.online_session is not None:
            self.online_session.close()
        self._begin_new_session()
        self._replace_state(None)
        self.online_started = False
        self.online_room_id = ""
        self.online_seat_count = 0
        self.online_max_players = self.online_num_players
        self.local_player_id = None
        self.online_session = OnlineSession(
            server_url=self._server_url,
            player_name=self.text_inputs["name"].value.strip() or "Player",
            room_id=self.text_inputs["room"].value.strip().upper(),
            num_players=self.online_num_players,
        )
        self.online_session.connect()
        self.scene = "online_lobby"
        self.status = "Connecting to server..."

    def _return_to_menu(self) -> None:
        if self.online_session is not None:
            self.online_session.close()
        self.online_session = None
        self._begin_new_session()
        self._replace_state(None)
        self.local_player_id = None
        self.scene = "menu"
        self.selected_mode = "MOVE"
        self.paused = False
        self.online_started = False

    def _process_events(self) -> None:
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                self.running = False
                return

            if self.scene == "menu":
                self._handle_menu_event(event)
            elif self.scene == "online_menu":
                self._handle_online_menu_event(event)
            elif self.scene == "online_lobby":
                self._handle_online_lobby_event(event)
            elif self.scene == "game":
                self._handle_game_event(event)

    def _handle_menu_event(self, event: object) -> None:
        if event.type != self.pg.MOUSEBUTTONDOWN or event.button != 1:
            return
        for button in self._menu_buttons():
            if not button.hit(event.pos):
                continue
            if button.action in PRESETS:
                self._start_local_game(PRESETS[button.action])
            elif button.action == "continue":
                self._continue_latest()
            elif button.action == "online":
                self._open_online_menu()
            elif button.action == "quit":
                self.running = False
            return

    def _handle_online_menu_event(self, event: object) -> None:
        if event.type == self.pg.MOUSEBUTTONDOWN and event.button == 1:
            clicked_name = None
            for name, text_input in self.text_inputs.items():
                if text_input.rect.collidepoint(event.pos):
                    clicked_name = name
                    break
            self._set_active_text_input(clicked_name)
            for button in self._online_menu_buttons():
                if not button.hit(event.pos):
                    continue
                if button.action == "join_online":
                    self._connect_online()
                elif button.action == "back_menu":
                    self._return_to_menu()
                elif button.action == "toggle_players":
                    self.online_num_players = 2 if self.online_num_players == 4 else 4
                return
        elif event.type == self.pg.KEYDOWN:
            active_input = self._active_text_input()
            if active_input is None:
                return
            if self._handle_online_input_shortcuts(active_input, event):
                return
            active_input.handle_key(event)

    def _handle_online_lobby_event(self, event: object) -> None:
        if event.type != self.pg.MOUSEBUTTONDOWN or event.button != 1:
            return
        for button in self._online_lobby_buttons():
            if not button.hit(event.pos):
                continue
            if button.action == "ready" and self.online_session is not None:
                self.online_session.send_ready()
                self.status = f"Ready sent. Waiting for all {self.online_max_players} players..."
            elif button.action == "back_menu":
                self._return_to_menu()
            return

    def _handle_game_event(self, event: object) -> None:
        if event.type == self.pg.KEYDOWN and event.key == self.pg.K_ESCAPE:
            self.paused = not self.paused
            return

        if event.type != self.pg.MOUSEBUTTONDOWN or event.button != 1:
            return

        if self.paused:
            for button in self._pause_buttons():
                if not button.hit(event.pos):
                    continue
                if button.action == "resume":
                    self.paused = False
                elif button.action == "save":
                    self._save_current_game()
                elif button.action == "menu":
                    self._return_to_menu()
                return
            return

        for button in self._game_buttons():
            if not button.hit(event.pos):
                continue
            if button.action == "mode_move":
                self.selected_mode = "MOVE"
                self._invalidate_manual_action_cache()
            elif button.action == "mode_h":
                self.selected_mode = "H"
                self._invalidate_manual_action_cache()
            elif button.action == "mode_v":
                self.selected_mode = "V"
                self._invalidate_manual_action_cache()
            elif button.action == "save":
                self._save_current_game()
            elif button.action == "menu":
                self.paused = True
            return

        if not self._can_accept_manual_input() or self.state is None:
            return
        action = self._action_from_click(event.pos)
        if action is not None:
            self._submit_action(action)

    def _update(self) -> None:
        self._poll_online_messages()
        self._poll_ai_results()
        self._poll_manual_action_previews()
        if self.scene != "game" or self.paused or self.state is None or self.state.winner_id is not None:
            return
        if self.online_session is None:
            self._queue_ai_turn_if_needed()
        self._queue_manual_action_preview_if_needed()

    def _poll_online_messages(self) -> None:
        if self.online_session is None:
            return
        for message in self.online_session.poll_messages():
            message_type = message.get("type")
            if message_type == "room_joined":
                self.local_player_id = int(message["player_id"])
                self.online_room_id = str(message["room_id"])
                self.online_seat_count = int(message.get("seat_count", 1))
                self.online_max_players = int(message.get("max_players", 4))
                self.scene = "online_lobby"
                self.status = f"Joined room {self.online_room_id} as Player {self.local_player_id + 1}."
            elif message_type == "player_joined":
                self.online_seat_count = int(message.get("seat_count", self.online_seat_count))
            elif message_type == "game_started":
                self.online_started = True
                self.status = "All players are ready. Match starting..."
            elif message_type == "state_update":
                self._replace_state(state_from_dict(message["state"]))
                self.scene = "game"
                self.status = "Online match synced."
            elif message_type == "game_over":
                self.status = f"Player {int(message['winner_id']) + 1} wins."
            elif message_type == "error":
                self.status = str(message.get("message", "Unknown online error."))

    def _queue_ai_turn_if_needed(self) -> None:
        assert self.state is not None
        player = self.state.get_player(self.state.current_turn)
        if player.kind == "human":
            return
        token = self._state_token(self.state)
        if self._ai_pending_token == token or self._ai_pending_token is not None:
            return
        self._ai_pending_token = token
        self.ai_worker.submit(token, self.state)
        self.status = f"{player.name} is thinking..."

    def _poll_ai_results(self) -> None:
        for result in self.ai_worker.poll_results():
            if result.token != self._ai_pending_token:
                continue
            self._ai_pending_token = None
            if result.error is not None:
                self.status = f"AI error: {result.error}"
                continue
            if self.state is None or result.action is None:
                continue
            if result.token != self._state_token(self.state):
                continue

            state_before = self.state
            actor = state_before.get_player(state_before.current_turn)
            self._replace_state(apply_action(state_before, result.action))
            self.status = f"{actor.name} {self._describe_action(result.action)}."

    def _queue_manual_action_preview_if_needed(self) -> None:
        if self.state is None or not self._can_accept_manual_input():
            return
        if self.selected_mode == "MOVE":
            return
        token = self._manual_action_token(self.state)
        if token == self._manual_action_cache_key or token == self._manual_action_pending_token:
            return
        self._manual_action_pending_token = token
        self.preview_worker.submit(token, self.state, self.selected_mode)
        self.status = f"Calculating {self.selected_mode} wall previews..."

    def _poll_manual_action_previews(self) -> None:
        for preview in self.preview_worker.poll_results():
            if preview.token != self._manual_action_pending_token:
                continue
            self._manual_action_pending_token = None
            if preview.error is not None:
                self.status = f"Preview error: {preview.error}"
                continue
            if self.state is None or preview.token != self._manual_action_token(self.state):
                continue
            self._manual_action_cache_key = preview.token
            self._manual_action_cache = list(preview.actions)

    def _submit_action(self, action: MovePawnAction | PlaceWallAction) -> None:
        if self.state is None:
            return
        if self.online_session is not None:
            self.online_session.send_action(action)
            return
        actor = self.state.get_player(self.state.current_turn)
        self._replace_state(apply_action(self.state, action))
        self.status = f"{actor.name} {self._describe_action(action)}."

    def _save_current_game(self) -> None:
        if self.state is None:
            return
        if self.online_session is not None:
            self.status = "Save/Continue is only enabled for local mode."
            return
        save_game(self.state, DEFAULT_SAVE_PATH)
        self.status = f"Saved current game to {DEFAULT_SAVE_PATH}."

    def _can_accept_manual_input(self) -> bool:
        if self.state is None or self.state.winner_id is not None:
            return False
        if self.online_session is not None:
            return self.local_player_id == self.state.current_turn
        return self.state.get_player(self.state.current_turn).kind == "human"

    def _current_manual_actions(self) -> list[MovePawnAction | PlaceWallAction]:
        if self.state is None or not self._can_accept_manual_input():
            return []
        token = self._manual_action_token(self.state)
        if token == self._manual_action_cache_key:
            return self._manual_action_cache

        if self.selected_mode == "MOVE":
            actions: list[MovePawnAction | PlaceWallAction] = list(generate_pawn_moves(self.state, self.state.current_turn))
            self._manual_action_cache_key = token
            self._manual_action_cache = actions
            return actions

        return []

    def _cell_rect(self, x: int, y: int) -> object:
        return self.pg.Rect(
            BOARD_ORIGIN[0] + x * CELL_SIZE,
            BOARD_ORIGIN[1] + y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
        )

    def _wall_rect(self, x: int, y: int, orientation: str) -> object:
        if orientation == "H":
            return self.pg.Rect(
                BOARD_ORIGIN[0] + x * CELL_SIZE + 8,
                BOARD_ORIGIN[1] + (y + 1) * CELL_SIZE - WALL_THICKNESS // 2,
                CELL_SIZE * 2 - 16,
                WALL_THICKNESS,
            )
        return self.pg.Rect(
            BOARD_ORIGIN[0] + (x + 1) * CELL_SIZE - WALL_THICKNESS // 2,
            BOARD_ORIGIN[1] + y * CELL_SIZE + 8,
            WALL_THICKNESS,
            CELL_SIZE * 2 - 16,
        )

    def _action_from_click(self, mouse_pos: tuple[int, int]) -> MovePawnAction | PlaceWallAction | None:
        for action in self._current_manual_actions():
            if isinstance(action, MovePawnAction):
                if self._cell_rect(action.to_x, action.to_y).collidepoint(mouse_pos):
                    return action
            elif self._wall_rect(action.x, action.y, action.orientation).collidepoint(mouse_pos):
                return action
        return None

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND)
        if self.scene == "menu":
            self._draw_menu()
        elif self.scene == "online_menu":
            self._draw_online_menu()
        elif self.scene == "online_lobby":
            self._draw_online_lobby()
        elif self.scene == "game":
            self._draw_game()
        self._draw_footer_status()
        self.pg.display.flip()

    def _draw_text(self, text: str, pos: tuple[int, int], font: object, color: tuple[int, int, int] = TEXT) -> None:
        surface = font.render(text, True, color)
        self.screen.blit(surface, pos)

    def _draw_button(self, button: Button) -> None:
        mouse_pos = self.pg.mouse.get_pos()
        color = BUTTON_HOVER if button.hit(mouse_pos) else BUTTON
        self.pg.draw.rect(self.screen, color, button.rect, border_radius=8)
        self.pg.draw.rect(self.screen, PANEL_BORDER, button.rect, width=2, border_radius=8)
        label = self.body_font.render(button.label, True, BUTTON_TEXT)
        label_rect = label.get_rect(center=button.rect.center)
        self.screen.blit(label, label_rect)

    def _draw_input(self, text_input: TextInput) -> None:
        color = HIGHLIGHT if text_input.active else PANEL_BORDER
        fill = (231, 241, 236) if text_input.selected else PANEL
        self.pg.draw.rect(self.screen, fill, text_input.rect, border_radius=6)
        self.pg.draw.rect(self.screen, color, text_input.rect, width=2, border_radius=6)
        self._draw_text(text_input.name, (text_input.rect.x, text_input.rect.y - 24), self.small_font)
        self._draw_text(text_input.value or "", (text_input.rect.x + 12, text_input.rect.y + 10), self.body_font)

    def _draw_menu(self) -> None:
        self._draw_text("Quoridor 4P", (450, 100), self.title_font)
        self._draw_text("Local first, AI complete, online MVP ready.", (350, 150), self.body_font)
        for button in self._menu_buttons():
            self._draw_button(button)

    def _draw_online_menu(self) -> None:
        self._draw_text("Online Setup", (460, 120), self.title_font)
        self._draw_text("Join an existing room code or leave it blank to create one.", (320, 170), self.body_font)
        self._draw_text("Tip: click a field, then use Ctrl+A / Ctrl+C / Ctrl+V / Ctrl+X.", (315, 205), self.small_font)
        for text_input in self.text_inputs.values():
            self._draw_input(text_input)
        for button in self._online_menu_buttons():
            self._draw_button(button)

    def _draw_online_lobby(self) -> None:
        self._draw_text("Online Lobby", (470, 120), self.title_font)
        self._draw_text(f"Room: {self.online_room_id or '...'}", (470, 220), self.body_font)
        self._draw_text(f"Seats joined: {self.online_seat_count}/{self.online_max_players}", (470, 260), self.body_font)
        self._draw_text(
            f"Your seat: Player {(self.local_player_id + 1) if self.local_player_id is not None else '?'}",
            (470, 300),
            self.body_font,
        )
        self._draw_text(
            f"When all {self.online_max_players} players have joined, each player presses Ready.",
            (330, 350),
            self.body_font,
        )
        for button in self._online_lobby_buttons():
            self._draw_button(button)

    def _draw_game(self) -> None:
        assert self.state is not None
        self._draw_board()
        self._draw_hud()
        if self.paused:
            self._draw_pause_overlay()

    def _draw_board(self) -> None:
        assert self.state is not None
        board_rect = self.pg.Rect(BOARD_ORIGIN[0], BOARD_ORIGIN[1], BOARD_PIXELS, BOARD_PIXELS)
        self.pg.draw.rect(self.screen, BOARD_BG, board_rect)
        self.pg.draw.rect(self.screen, GRID, board_rect, width=3)

        for y in range(self.state.board_size):
            for x in range(self.state.board_size):
                rect = self._cell_rect(x, y)
                self.pg.draw.rect(self.screen, BOARD_BG, rect)
                self.pg.draw.rect(self.screen, GRID, rect, width=1)

        for action in self._current_manual_actions():
            if isinstance(action, MovePawnAction):
                rect = self._cell_rect(action.to_x, action.to_y)
                self.pg.draw.circle(self.screen, HIGHLIGHT, rect.center, CELL_SIZE // 6)
            else:
                rect = self._wall_rect(action.x, action.y, action.orientation)
                self.pg.draw.rect(self.screen, HIGHLIGHT, rect, width=2, border_radius=4)

        for x, y in self.state.walls_h:
            self.pg.draw.rect(self.screen, GRID, self._wall_rect(x, y, "H"), border_radius=4)
        for x, y in self.state.walls_v:
            self.pg.draw.rect(self.screen, GRID, self._wall_rect(x, y, "V"), border_radius=4)

        for player in self.state.players:
            rect = self._cell_rect(player.pos.x, player.pos.y)
            color = PLAYER_COLORS[player.id % len(PLAYER_COLORS)]
            self.pg.draw.circle(self.screen, color, rect.center, CELL_SIZE // 3)
            self.pg.draw.circle(self.screen, (250, 248, 242), rect.center, CELL_SIZE // 3, width=2)

    def _draw_hud(self) -> None:
        assert self.state is not None
        panel_rect = self.pg.Rect(HUD_X, 48, 320, 640)
        self.pg.draw.rect(self.screen, PANEL, panel_rect, border_radius=12)
        self.pg.draw.rect(self.screen, PANEL_BORDER, panel_rect, width=2, border_radius=12)
        self._draw_text("Match HUD", (HUD_X + 24, 72), self.title_font)

        if self.state.winner_id is None:
            current = self.state.get_player(self.state.current_turn)
            self._draw_text(f"Turn: {current.name}", (HUD_X + 24, 132), self.body_font)
        else:
            self._draw_text(f"Winner: Player {self.state.winner_id + 1}", (HUD_X + 24, 132), self.body_font, WARNING)
        self._draw_text(f"Move count: {self.state.move_count}", (HUD_X + 24, 166), self.body_font)
        self._draw_text(f"Mode: {self.selected_mode}", (HUD_X + 24, 200), self.body_font)

        if self._ai_pending_token is not None and self.online_session is None and self.state.winner_id is None:
            self._draw_text("AI thinking...", (HUD_X + 24, 228), self.body_font, HIGHLIGHT)
        elif self._manual_action_pending_token is not None and self.selected_mode in {"H", "V"}:
            self._draw_text("Loading wall previews...", (HUD_X + 24, 228), self.body_font, HIGHLIGHT)

        y = 270
        for player in self.state.players:
            color = PLAYER_COLORS[player.id % len(PLAYER_COLORS)]
            self.pg.draw.circle(self.screen, color, (HUD_X + 36, y + 14), 10)
            control = "You" if self.online_session is not None and player.id == self.local_player_id else player.kind
            label = f"P{player.id + 1}: ({player.pos.x}, {player.pos.y})  walls={player.walls_left}  {control}"
            self._draw_text(label, (HUD_X + 58, y), self.small_font)
            y += 36

        for button in self._game_buttons():
            self._draw_button(button)

    def _draw_pause_overlay(self) -> None:
        overlay = self.pg.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), self.pg.SRCALPHA)
        overlay.fill((0, 0, 0, 110))
        self.screen.blit(overlay, (0, 0))
        panel = self.pg.Rect(420, 220, 360, 240)
        self.pg.draw.rect(self.screen, PANEL, panel, border_radius=12)
        self.pg.draw.rect(self.screen, PANEL_BORDER, panel, width=2, border_radius=12)
        self._draw_text("Pause", (560, 250), self.title_font)
        for button in self._pause_buttons():
            self._draw_button(button)

    def _draw_footer_status(self) -> None:
        color = WARNING if "error" in self.status.lower() else TEXT
        self._draw_text(self.status, (48, 710), self.small_font, color)

    def _menu_buttons(self) -> list[Button]:
        labels = [
            ("Local 2P", "local_2"),
            ("Local 4P", "local_4"),
            ("Vs Random", "vs_random"),
            ("Vs Minimax", "vs_minimax"),
            ("Vs MCTS", "vs_mcts"),
            ("Continue", "continue"),
            ("Online", "online"),
            ("Quit", "quit"),
        ]
        buttons = []
        x, y = 470, 220
        for label, action in labels:
            buttons.append(Button(self.pg.Rect(x, y, 260, 46), label, action))
            y += 58
        return buttons

    def _online_menu_buttons(self) -> list[Button]:
        label = f"{self.online_num_players} Players"
        return [
            Button(self.pg.Rect(520, 400, 300, 44), label, "toggle_players"),
            Button(self.pg.Rect(520, 470, 140, 44), "Join/Create", "join_online"),
            Button(self.pg.Rect(680, 470, 140, 44), "Back", "back_menu"),
        ]

    def _online_lobby_buttons(self) -> list[Button]:
        return [
            Button(self.pg.Rect(520, 420, 140, 44), "Ready", "ready"),
            Button(self.pg.Rect(680, 420, 140, 44), "Back", "back_menu"),
        ]

    def _game_buttons(self) -> list[Button]:
        return [
            Button(self.pg.Rect(HUD_X + 24, 520, 128, 42), "Move", "mode_move"),
            Button(self.pg.Rect(HUD_X + 168, 520, 128, 42), "Wall H", "mode_h"),
            Button(self.pg.Rect(HUD_X + 24, 572, 128, 42), "Wall V", "mode_v"),
            Button(self.pg.Rect(HUD_X + 168, 572, 128, 42), "Save", "save"),
            Button(self.pg.Rect(HUD_X + 24, 624, 272, 42), "Pause / Menu", "menu"),
        ]

    def _pause_buttons(self) -> list[Button]:
        return [
            Button(self.pg.Rect(490, 300, 220, 40), "Resume", "resume"),
            Button(self.pg.Rect(490, 350, 220, 40), "Save", "save"),
            Button(self.pg.Rect(490, 400, 220, 40), "Exit to Menu", "menu"),
        ]



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Quoridor pygame client.")
    parser.add_argument("--server", default=None, help="WebSocket server URL (e.g. ws://192.168.1.5:8765)")
    parser.add_argument("--name", default=None, help="Default player name")
    args = parser.parse_args()
    QuoridorApp(server_url=args.server, player_name=args.name).run()


if __name__ == "__main__":
    main()
