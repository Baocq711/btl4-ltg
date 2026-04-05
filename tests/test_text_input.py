import unittest
from types import SimpleNamespace

from quoridor.client.widgets import TextInput


class TextInputTests(unittest.TestCase):
    def test_paste_text_strips_newlines_and_nulls(self) -> None:
        text_input = TextInput(rect=None, value="", name="Room")
        text_input.set_active(True)
        text_input.paste_text("AB\r\nCD\x00")
        self.assertEqual(text_input.value, "ABCD")

    def test_select_all_then_type_replaces_value(self) -> None:
        text_input = TextInput(rect=None, value="old", name="Name")
        text_input.set_active(True, select_all=True)
        event = SimpleNamespace(key=65, unicode="N")
        text_input.handle_key(event)
        self.assertEqual(text_input.value, "N")
        self.assertFalse(text_input.selected)

    def test_backspace_with_selection_clears_value(self) -> None:
        text_input = TextInput(rect=None, value="server", name="Server")
        text_input.set_active(True, select_all=True)
        event = SimpleNamespace(key=8, unicode="")
        text_input.handle_key(event)
        self.assertEqual(text_input.value, "")


if __name__ == "__main__":
    unittest.main()