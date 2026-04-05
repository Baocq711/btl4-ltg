"""Reusable pygame widgets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Button:
    rect: object
    label: str
    action: str

    def hit(self, mouse_pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(mouse_pos)


@dataclass
class TextInput:
    rect: object
    value: str
    name: str
    active: bool = False
    selected: bool = False

    def set_active(self, active: bool, select_all: bool = False) -> None:
        self.active = active
        self.selected = active and select_all

    def select_all(self) -> None:
        if self.active:
            self.selected = True

    def clear(self) -> None:
        self.value = ""
        self.selected = False

    def _replace_or_append(self, text: str) -> None:
        if self.selected:
            self.value = text
        else:
            self.value += text
        self.selected = False

    def paste_text(self, text: str) -> None:
        clean = text.replace("\r", "").replace("\n", "").replace("\x00", "")
        if clean:
            self._replace_or_append(clean)

    def handle_key(self, event: object) -> None:
        if not self.active:
            return
        if event.key == 8:
            if self.selected:
                self.clear()
            else:
                self.value = self.value[:-1]
        elif event.key == 13:
            self.active = False
            self.selected = False
        elif event.unicode and event.unicode.isprintable():
            self._replace_or_append(event.unicode)