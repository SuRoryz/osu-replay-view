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
        command_count = len(self._commands)
        if command_count:
            panel_commands = 0
            gradient_commands = 0
            text_commands = 0
            clip_push_commands = 0
            clip_pop_commands = 0
            for command in self._commands:
                if command.kind == "panel":
                    panel_commands += 1
                elif command.kind == "gradient":
                    gradient_commands += 1
                elif command.kind == "text":
                    text_commands += 1
                elif command.kind == "clip_push":
                    clip_push_commands += 1
                elif command.kind == "clip_pop":
                    clip_pop_commands += 1
            profiler.count("menu.command_buffer.commands", command_count)
            profiler.count("menu.command_buffer.commands.panel", panel_commands)
            profiler.count("menu.command_buffer.commands.gradient", gradient_commands)
            profiler.count("menu.command_buffer.commands.text", text_commands)
            profiler.count("menu.command_buffer.commands.clip_push", clip_push_commands)
            profiler.count("menu.command_buffer.commands.clip_pop", clip_pop_commands)
        initial_scissor = ctx.scissor
        clip_stack: list[tuple[int, int, int, int] | None] = []
        panel_batching = False
        panel_batches = 0

        def ensure_panel_batch() -> None:
            nonlocal panel_batching, panel_batches
            if not panel_batching:
                panels.begin_batch()
                panel_batching = True
                panel_batches += 1

        def flush_panel_batch() -> None:
            nonlocal panel_batching
            if panel_batching:
                panels.end_batch()
                panel_batching = False

        with profiler.timer("menu.command_buffer.flush"):
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
        if panel_batches:
            profiler.count("menu.command_buffer.panel_batches", panel_batches)

