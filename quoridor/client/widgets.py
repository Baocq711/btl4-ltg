"""Reusable pygame widgets."""

from __future__ import annotations

from dataclasses import dataclass, field


def truncate_text(font: object, text: str, max_width: int, ellipsis: str = "...") -> str:
    """Return *text* truncated so that it fits within *max_width* pixels."""
    if font.size(text)[0] <= max_width:
        return text
    ell_w = font.size(ellipsis)[0]
    while text and font.size(text)[0] + ell_w > max_width:
        text = text[:-1]
    return text + ellipsis


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
    max_length: int = 30

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
        if self.max_length and len(self.value) > self.max_length:
            self.value = self.value[: self.max_length]
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