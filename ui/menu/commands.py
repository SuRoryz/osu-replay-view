"""Render command buffer for menu/UI drawing."""

from __future__ import annotations

from dataclasses import dataclass

import moderngl

from profiling import profiler
from ui.menu.layout import Rect


@dataclass(slots=True)
class _Command:
    kind: str
    args: tuple


class RenderCommandBuffer:
    """Collects UI commands and flushes them in-order."""

    def __init__(self) -> None:
        self._commands: list[_Command] = []

    def clear(self) -> None:
        self._commands.clear()

    def panel(
        self,
        rect: Rect,
        *,
        radius: float = 8.0,
        color: tuple = (0.1, 0.1, 0.15, 0.85),
        border_color: tuple = (0.3, 0.3, 0.4, 1.0),
        border_width: float = 1.0,
    ) -> None:
        self._commands.append(
            _Command("panel", (rect, radius, color, border_color, border_width))
        )

    def gradient_bar(
        self,
        rect: Rect,
        *,
        spawn_x: float,
        fade_width: float,
        color: tuple,
    ) -> None:
        self._commands.append(
            _Command("gradient", (rect, spawn_x, fade_width, color))
        )

    def text(
        self,
        value: str,
        x: float,
        y: float,
        size: int,
        *,
        color: tuple = (1.0, 1.0, 1.0),
        alpha: float = 1.0,
    ) -> None:
        self._commands.append(
            _Command("text", (value, x, y, size, color, alpha))
        )

    def clip_push(self, rect: Rect) -> None:
        self._commands.append(_Command("clip_push", (rect,)))

    def clip_pop(self) -> None:
        self._commands.append(_Command("clip_pop", ()))

    def flush(self, *, ctx, text, panels, window_height: int) -> None:
        profiler.count("menu.command_buffer.flushes")
        initial_scissor = ctx.scissor
        clip_stack: list[tuple[int, int, int, int] | None] = []
        panel_batching = False

        def ensure_panel_batch() -> None:
            nonlocal panel_batching
            if not panel_batching:
                panels.begin_batch()
                panel_batching = True

        def flush_panel_batch() -> None:
            nonlocal panel_batching
            if panel_batching:
                panels.end_batch()
                panel_batching = False

        text.begin()
        try:
            for command in self._commands:
                kind = command.kind
                if kind == "panel":
                    ensure_panel_batch()
                    rect, radius, color, border_color, border_width = command.args
                    panels.draw(
                        rect.x,
                        rect.y,
                        rect.w,
                        rect.h,
                        radius=radius,
                        color=color,
                        border_color=border_color,
                        border_width=border_width,
                    )
                elif kind == "gradient":
                    ensure_panel_batch()
                    rect, spawn_x, fade_width, color = command.args
                    panels.draw_gradient_bar(
                        rect.x,
                        rect.y,
                        rect.w,
                        rect.h,
                        spawn_x=spawn_x,
                        fade_width=fade_width,
                        color=color,
                    )
                elif kind == "text":
                    flush_panel_batch()
                    value, x, y, size, color, alpha = command.args
                    text.draw(value, x, y, size, color=color, alpha=alpha)
                elif kind == "clip_push":
                    flush_panel_batch()
                    text.end()
                    (rect,) = command.args
                    clip_stack.append(ctx.scissor)
                    ctx.scissor = (
                        max(0, int(rect.x)),
                        max(0, int(window_height - (rect.y + rect.h))),
                        max(1, int(rect.w)),
                        max(1, int(rect.h)),
                    )
                    text.begin()
                elif kind == "clip_pop":
                    flush_panel_batch()
                    text.end()
                    ctx.scissor = clip_stack.pop() if clip_stack else None
                    text.begin()
        finally:
            flush_panel_batch()
            text.end()
            ctx.scissor = initial_scissor
            self._commands.clear()

