from __future__ import annotations

from dataclasses import dataclass

from ui.design import draw_linear_progress, draw_overlay_band
from ui.menu.commands import RenderCommandBuffer
from ui.menu.layout import Rect


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class _TransientAlert:
    text: str
    ttl: float
    fade_in: float
    fade_out: float
    elapsed: float = 0.0


@dataclass(slots=True)
class _ProgressAlert:
    text: str
    progress: float = 0.0
    visible: bool = False


class AlertOverlay:
    def __init__(self, app) -> None:
        self.app = app
        self._commands = RenderCommandBuffer()
        self._transient: _TransientAlert | None = None
        self._progress = _ProgressAlert(text="")

    def show_message(
        self,
        text: str,
        *,
        ttl: float = 1.9,
        fade_in: float = 0.10,
        fade_out: float = 0.34,
    ) -> None:
        self._transient = _TransientAlert(text=text, ttl=ttl, fade_in=fade_in, fade_out=fade_out)

    def show_progress(self, text: str, progress: float) -> None:
        self._progress.text = text
        self._progress.progress = _clamp01(progress)
        self._progress.visible = True

    def hide_progress(self) -> None:
        self._progress.visible = False
        self._progress.progress = 0.0
        self._progress.text = ""

    def update(self, dt: float) -> None:
        if self._transient is None:
            return
        self._transient.elapsed += max(0.0, dt)
        total = self._transient.ttl + self._transient.fade_out
        if self._transient.elapsed >= total:
            self._transient = None

    def _alpha_for_transient(self, alert: _TransientAlert) -> float:
        if alert.elapsed < alert.fade_in:
            return _clamp01(alert.elapsed / max(alert.fade_in, 1e-6))
        if alert.elapsed <= alert.ttl:
            return 1.0
        return _clamp01(1.0 - ((alert.elapsed - alert.ttl) / max(alert.fade_out, 1e-6)))

    def draw(self) -> None:
        context = self.app.menu_context()
        layout = context
        density = layout.density
        theme = layout.theme
        colors = theme.colors
        text = self.app.text
        alert_text = ""
        progress = None
        alpha = 0.0
        if self._progress.visible:
            alert_text = self._progress.text
            progress = self._progress.progress
            alpha = 1.0
        elif self._transient is not None:
            alert_text = self._transient.text
            alpha = self._alpha_for_transient(self._transient)
        if not alert_text or alpha <= 0.001:
            return
        commands = self._commands
        commands.clear()
        band_h = 68.0 * density
        margin_x = 26.0 * density
        rect = Rect(
            margin_x,
            layout.viewport.h * 0.5 - band_h * 0.5,
            max(240.0 * density, layout.viewport.w - margin_x * 2.0),
            band_h,
        )
        draw_overlay_band(commands, theme, rect, alpha=0.96 * alpha, radius=10.0 * density)
        text_size = int(28 * density)
        text_w, _ = text.measure(alert_text, text_size)
        commands.text(
            alert_text,
            rect.center_x - text_w * 0.5,
            rect.y + rect.h * 0.5 - text_size * 0.5 - 4.0 * density,
            text_size,
            color=colors.text_primary,
            alpha=0.98 * alpha,
        )
        if progress is not None:
            progress_rect = Rect(rect.x, rect.bottom - 4.0 * density, rect.w, 4.0 * density)
            draw_linear_progress(
                commands,
                theme,
                progress_rect,
                value=progress,
                fill_color=(1.0, 1.0, 1.0),
                alpha=alpha,
                track_alpha=0.0,
                track_border_width=0.0,
            )
        commands.flush(
            ctx=self.app.ctx,
            text=self.app.text,
            panels=self.app.panels,
            window_height=self.app.wnd.buffer_size[1],
        )
