"""Abstract base class for application scenes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import moderngl


class Scene(ABC):
    """A self-contained screen (menu, gameplay, etc.)."""

    def __init__(self, app):
        self.app = app

    @abstractmethod
    def on_enter(self) -> None:
        """Called when this scene becomes active."""

    @abstractmethod
    def on_leave(self) -> None:
        """Called when this scene is about to be replaced."""

    @abstractmethod
    def on_render(self, time: float, frametime: float) -> None:
        ...

    def on_resize(self, width: int, height: int) -> None:
        pass

    def on_key_event(self, key, action, modifiers) -> None:
        pass

    def on_mouse_press(self, x: int, y: int, button: int) -> None:
        pass

    def on_mouse_scroll(self, x_offset: float, y_offset: float) -> None:
        pass

    def on_mouse_release(self, x: int, y: int, button: int) -> None:
        pass

    def on_mouse_move(self, x: int, y: int, dx: int, dy: int) -> None:
        pass

    def on_unicode_char_entered(self, char: str) -> None:
        pass
