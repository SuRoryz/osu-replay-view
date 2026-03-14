"""Animated settings side-sheet with anchored desktop dropdowns."""

from __future__ import annotations

from dataclasses import dataclass

from ui.design import (
    InteractionState,
    draw_button,
    draw_dropdown_menu,
    draw_select_field,
    draw_slider,
    draw_supporting_text,
    draw_surface,
    draw_text_field,
)
from ui.menu.animation import (
    AnimatedFloat,
    ease_in_out_cubic,
    ease_out_back,
    ease_out_cubic,
    stagger,
)
from ui.menu.commands import RenderCommandBuffer
from ui.menu.layout import LayoutContext, Rect, build_layout_context, clamp


_AUDIO_KEYS = (("music", "Music"), ("sfx", "Hitsounds"))
_SCREEN_MODES = ("windowed", "borderless", "fullscreen")
_FPS_LIMIT_OPTIONS = (
    (60, "60 FPS"),
    (120, "120 FPS"),
    (144, "144 FPS"),
    (180, "180 FPS"),
    (0, "Unlimited"),
)
_SECTION_COUNT = 4
_GRAPHICS_TOGGLES = (
    ("gameplay_background_bloom", "Background bloom"),
    ("gameplay_background_image", "Beatmap background image"),
    ("gameplay_cursor_trail", "Cursor trail"),
)
_GRAPHICS_SLIDERS = (
    ("gameplay_background_dim", "Background dim"),
    ("gameplay_cursor_trail_max_len", "Max cursor trail len"),
    ("gameplay_circle_bloom", "Circle bloom"),
)
_GAMEPLAY_HUD_TOGGLES = (
    ("draw_gameplay_timeline", "", "Timeline"),
    ("draw_gameplay_acc_pp", "ACC and PP", "AP"),
    ("draw_gameplay_leaderboard", "", "Leaderboard"),
    ("draw_gameplay_combo", "", "Combo"),
    ("draw_gameplay_hp", "", "HP"),
    ("draw_gameplay_keys", "", "Keys"),
)


@dataclass(slots=True)
class _SelectLayout:
    key: str
    label: str
    rect: Rect
    options: list[tuple[object, str]]
    selected_index: int
    value_text: str


@dataclass(slots=True)
class _ToggleTileLayout:
    key: str
    label: str
    short_label: str
    rect: Rect
    radius: float


@dataclass(slots=True)
class _SettingsLayout:
    context: LayoutContext
    button_rect: Rect | None
    backdrop_rect: Rect
    drawer_rect: Rect
    header_rect: Rect
    title_rect: Rect
    subtitle_rect: Rect
    close_rect: Rect
    scroll_rect: Rect
    footer_rect: Rect
    back_rect: Rect
    build_rect: Rect
    content_height: float
    audio_section_rect: Rect
    display_section_rect: Rect
    gameplay_section_rect: Rect
    gameplay_preview_rect: Rect
    profile_section_rect: Rect
    audio_controls: dict[str, dict]
    selects: dict[str, _SelectLayout]
    graphics_toggles: dict[str, dict]
    graphics_sliders: dict[str, dict]
    gameplay_toggles: dict[str, _ToggleTileLayout]
    display_helper_rect: Rect
    nickname_label_rect: Rect
    nickname_rect: Rect


class SettingsOverlay:
    def __init__(self, app) -> None:
        self.app = app
        self._open = False
        self._open_anim = AnimatedFloat(0.0, 0.0, 8.0)
        self._scroll = AnimatedFloat(0.0, 0.0, 14.0)
        self._content_anim = AnimatedFloat(0.0, 0.0, 9.0)
        self._dropdown_anim = AnimatedFloat(0.0, 0.0, 16.0)
        self._nickname_focus = False
        self._slider_dragging: str | None = None
        self._mouse_x = 0
        self._mouse_y = 0
        self._layout: _SettingsLayout | None = None
        self._layout_key: tuple[int, int, int, int] | None = None
        self._button_hover = False
        self._hovered_id: str | None = None
        self._hover_mix: dict[str, AnimatedFloat] = {}
        self._menu_key: str | None = None
        self._menu_open = False
        self._menu_highlight_index = -1

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def is_visible(self) -> bool:
        return self._open or self._open_anim.value > 0.001

    def invalidate(self) -> None:
        self._layout = None
        self._layout_key = None

    def toggle(self) -> None:
        if self._open:
            self.close()
        else:
            self.open()

    def open(self) -> None:
        self._open = True
        self._open_anim.set_target(1.0)
        self._content_anim.set_target(1.0)
        self._button_hover = False

    def close(self) -> None:
        self._open = False
        self._open_anim.set_target(0.0)
        self._content_anim.set_target(0.0)
        self._close_select()
        self._nickname_focus = False
        self._slider_dragging = None

    def update(self, dt: float) -> None:
        self._open_anim.update(dt)
        self._content_anim.update(dt)
        self._scroll.update(dt)
        self._dropdown_anim.update(dt)
        if not self._menu_open and self._dropdown_anim.value <= 0.001:
            self._menu_key = None
            self._menu_highlight_index = -1
        self._sync_hover()
        self._update_hover_mix(dt)

    def draw(self, dt: float) -> None:
        self.update(dt)
        layout = self._build_layout()
        ctx = self.app.ctx
        base_commands = RenderCommandBuffer()
        overlay_commands = RenderCommandBuffer()
        theme = layout.context.theme
        colors = theme.colors
        density = layout.context.density
        drawer_progress = ease_out_cubic(self._open_anim.value)
        content_progress = ease_in_out_cubic(self._content_anim.value)
        drawer_alpha = 0.98 * drawer_progress

        if self._open_anim.value <= 0.001:
            base_commands.flush(
                ctx=ctx,
                text=self.app.text,
                panels=self.app.panels,
                window_height=self.app.wnd.buffer_size[1],
            )
            return

        draw_surface(
            base_commands,
            theme,
            layout.backdrop_rect,
            role="scrim",
            radius=0.0,
            alpha=0.54 * drawer_progress,
            border_width=0.0,
        )
        draw_surface(
            base_commands,
            theme,
            layout.drawer_rect,
            role="drawer",
            radius=0.0,
            alpha=drawer_alpha,
            border_width=0.0,
        )
        base_commands.panel(
            Rect(layout.drawer_rect.x, layout.drawer_rect.y, layout.drawer_rect.w, 92.0 * density),
            radius=0.0,
            color=(colors.surface_container_low[0], colors.surface_container_low[1], colors.surface_container_low[2], 0.34 * drawer_progress),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        base_commands.panel(
            layout.footer_rect,
            radius=0.0,
            color=(colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.20 * drawer_progress),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        title_pill = Rect(layout.title_rect.x, layout.title_rect.y - 14.0 * density, 92.0 * density, 18.0 * density)
        base_commands.panel(
            title_pill,
            radius=title_pill.h * 0.5,
            color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], 0.52 * drawer_progress),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )

        base_commands.text(
            "Settings",
            layout.title_rect.x,
            layout.title_rect.y,
            layout.context.theme.typography.headline,
            color=colors.text_primary,
            alpha=drawer_progress,
        )
        draw_supporting_text(
            base_commands,
            theme,
            "Audio, display, gameplay, and profile preferences",
            layout.subtitle_rect.x,
            layout.subtitle_rect.y + 2.0,
            layout.context.tokens.typography.body_s,
            alpha=0.92 * drawer_progress,
            tone="secondary",
        )
        draw_button(
            base_commands,
            self.app.text,
            theme,
            layout.close_rect,
            label="X",
            size=layout.context.tokens.typography.body_m,
            variant="quiet",
            state=InteractionState.HOVER if self._hovered_id == "button:close" else InteractionState.REST,
            radius=layout.context.tokens.radius_m,
            alpha=0.92 * drawer_progress,
        )

        base_commands.clip_push(layout.scroll_rect)
        self._draw_audio_section(base_commands, layout, density, content_progress)
        self._draw_display_section(base_commands, layout, density, content_progress)
        self._draw_gameplay_section(base_commands, layout, density, content_progress)
        self._draw_profile_section(base_commands, layout, density, content_progress)
        base_commands.clip_pop()

        draw_button(
            base_commands,
            self.app.text,
            theme,
            layout.back_rect,
            label="Back",
            size=layout.context.tokens.typography.body_m,
            variant="quiet",
            state=InteractionState.HOVER if self._hovered_id == "button:back" else InteractionState.REST,
            radius=layout.context.tokens.radius_m,
            alpha=0.94 * drawer_progress,
        )
        draw_supporting_text(
            base_commands,
            theme,
            f"Build {self.app.build_version}",
            layout.build_rect.x,
            layout.build_rect.y,
            layout.context.tokens.typography.body_s,
            alpha=0.92 * drawer_progress,
            tone="secondary",
        )

        menu_geo = self._menu_geometry(layout)
        if menu_geo is not None and self._menu_key is not None and self._dropdown_anim.value > 0.001:
            menu_progress = ease_out_back(self._dropdown_anim.value)
            menu_alpha = clamp(self._dropdown_anim.value, 0.0, 1.0)
            trigger_rect = menu_geo["trigger_rect"]
            option_rects = [
                rect.translate(dy=(1.0 - menu_progress) * 8.0 * density)
                for rect in menu_geo["option_rects"]
            ]
            menu_rect = menu_geo["menu_rect"].translate(dy=(1.0 - menu_progress) * 8.0 * density)
            draw_select_field(
                overlay_commands,
                self.app.text,
                theme,
                trigger_rect,
                value=layout.selects[self._menu_key].value_text,
                size=layout.context.tokens.typography.body_m,
                density=density,
                hovered=True,
                focused=True,
                open_progress=self._dropdown_anim.value,
                alpha=1.0,
            )
            draw_dropdown_menu(
                overlay_commands,
                self.app.text,
                theme,
                menu_rect,
                option_rects=option_rects,
                labels=menu_geo["labels"],
                selected_index=menu_geo["selected_index"],
                hovered_index=self._menu_highlight_index,
                size=layout.context.tokens.typography.body_m,
                alpha=menu_alpha,
                progress=self._dropdown_anim.value,
            )

        base_commands.flush(
            ctx=ctx,
            text=self.app.text,
            panels=self.app.panels,
            window_height=self.app.wnd.buffer_size[1],
        )
        if self._menu_key is not None and self._dropdown_anim.value > 0.001:
            overlay_commands.flush(
                ctx=ctx,
                text=self.app.text,
                panels=self.app.panels,
                window_height=self.app.wnd.buffer_size[1],
            )

    def handle_key_event(self, key, action) -> bool:
        keys = self.app.wnd.keys
        if action != keys.ACTION_PRESS:
            return False
        if key == keys.F10:
            self.toggle()
            return True
        if not self.is_visible:
            return False
        if self._menu_key is not None and self._menu_open:
            if key == keys.ESCAPE:
                self._close_select()
                return True
            if key == keys.UP:
                self._move_menu_highlight(-1)
                return True
            if key == keys.DOWN:
                self._move_menu_highlight(1)
                return True
            if key in {keys.ENTER, keys.SPACE}:
                if self._menu_highlight_index >= 0:
                    self._choose_dropdown_index(self._menu_key, self._menu_highlight_index)
                return True
            return True
        if key == keys.ESCAPE:
            self.close()
            return True
        if self._nickname_focus and key == keys.BACKSPACE:
            self.app._set_nickname(self.app.settings.nickname[:-1])
            return True
        if self._nickname_focus and key == keys.ENTER:
            self._nickname_focus = False
            return True
        return self._open

    def handle_text(self, char: str) -> bool:
        if not (self._open and self._nickname_focus):
            return False
        if not char or ord(char[0]) < 32:
            return True
        if len(self.app.settings.nickname) >= 16:
            return True
        self.app._set_nickname(self.app.settings.nickname + char[0])
        return True

    def handle_mouse_press(self, x: int, y: int, button: int) -> bool:
        self._mouse_x = x
        self._mouse_y = y
        layout = self._build_layout()
        if button != 1:
            return self.is_visible
        if not self.is_visible and layout.button_rect is not None and layout.button_rect.contains(x, y):
            self.toggle()
            return True
        if not self.is_visible:
            return False

        menu_geo = self._menu_geometry(layout)
        if self._menu_key is not None and menu_geo is not None:
            if menu_geo["menu_rect"].contains(x, y):
                for idx, rect in enumerate(menu_geo["option_rects"]):
                    if rect.contains(x, y):
                        self._choose_dropdown_index(self._menu_key, idx)
                        return True
            elif menu_geo["trigger_rect"].contains(x, y):
                self._close_select()
                return True

        if not layout.drawer_rect.contains(x, y):
            self.close()
            return True
        if layout.close_rect.contains(x, y):
            self.close()
            return True
        if layout.back_rect.contains(x, y):
            self.close()
            return True

        nickname_rect = self._animated_rect(layout, layout.nickname_rect, section_index=3)
        self._nickname_focus = nickname_rect.contains(x, y)
        if self._nickname_focus:
            self._close_select()

        for key_name, select in layout.selects.items():
            if self._animated_rect(layout, select.rect, section_index=1).contains(x, y):
                self._toggle_select(key_name, layout)
                self._nickname_focus = False
                return True

        for key_name, toggle in layout.graphics_toggles.items():
            row_rect = self._animated_rect(layout, toggle["row_rect"], section_index=1)
            button_rect = self._animated_rect(layout, toggle["button_rect"], section_index=1)
            if row_rect.contains(x, y) or button_rect.contains(x, y):
                self._toggle_graphics_setting(key_name)
                return True

        for key_name, control in layout.graphics_sliders.items():
            row_rect = self._animated_rect(layout, control["row_rect"], section_index=1)
            track_rect = self._animated_rect(layout, control["track_rect"], section_index=1)
            if row_rect.contains(x, y) or track_rect.contains(x, y):
                self._slider_dragging = key_name
                self._apply_slider_value(key_name, x, track_rect, persist=False)
                return True

        for key_name, toggle in layout.gameplay_toggles.items():
            if self._animated_rect(layout, toggle.rect, section_index=2).contains(x, y):
                current = bool(getattr(self.app.settings, key_name, True))
                self.app.set_gameplay_overlay_visible(key_name, not current)
                return True

        for key_name, control in layout.audio_controls.items():
            row_rect = self._animated_rect(layout, control["row_rect"], section_index=0)
            track_rect = self._animated_rect(layout, control["track_rect"], section_index=0)
            mute_rect = self._animated_rect(layout, control["mute_rect"], section_index=0)
            if mute_rect.contains(x, y):
                if key_name == "music":
                    self.app.set_music_muted(not self.app.settings.music_muted)
                else:
                    self.app.set_sfx_muted(not self.app.settings.sfx_muted)
                return True
            if row_rect.contains(x, y) or track_rect.contains(x, y):
                self._slider_dragging = key_name
                self._apply_slider_value(key_name, x, track_rect, persist=False)
                return True

        self._close_select()
        return True

    def handle_mouse_move(self, x: int, y: int) -> bool:
        self._mouse_x = x
        self._mouse_y = y
        if not self.is_visible:
            button_rect = self._build_layout().button_rect
            self._button_hover = button_rect.contains(x, y) if button_rect is not None else False
            return self._button_hover
        self._sync_hover()
        if self._slider_dragging is None:
            return self._open
        layout = self._build_layout()
        control_info = self._slider_control(layout, self._slider_dragging)
        if control_info is None:
            self._slider_dragging = None
            return self._open
        control, section_index = control_info
        track_rect = self._animated_rect(layout, control["track_rect"], section_index=section_index)
        self._apply_slider_value(self._slider_dragging, x, track_rect, persist=False)
        return True

    def handle_mouse_release(self, button: int) -> bool:
        if button == 1 and self._slider_dragging is not None:
            self.app._save_settings()
            self._slider_dragging = None
            return True
        return self._open

    def handle_scroll(self, y_offset: float) -> bool:
        if not self.is_visible:
            return False
        layout = self._build_layout()
        menu_geo = self._menu_geometry(layout)
        if menu_geo is not None and menu_geo["menu_rect"].contains(self._mouse_x, self._mouse_y):
            return True
        if not layout.drawer_rect.contains(self._mouse_x, self._mouse_y):
            return True
        max_scroll = max(0.0, layout.content_height - layout.scroll_rect.h)
        self._scroll.set_target(
            clamp(
                self._scroll.target - y_offset * 52.0 * layout.context.density,
                0.0,
                max_scroll,
            )
        )
        return True

    def _draw_section_shell(
        self,
        commands: RenderCommandBuffer,
        layout: _SettingsLayout,
        rect: Rect,
        *,
        title: str,
        subtitle: str,
        alpha: float,
    ) -> Rect:
        theme = layout.context.theme
        density = layout.context.density
        colors = theme.colors
        commands.panel(
            rect,
            radius=theme.shape.corner_l,
            color=(colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.42 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        band_h = 34.0 * density
        commands.panel(
            Rect(rect.x, rect.y, rect.w, band_h),
            radius=theme.shape.corner_l,
            color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], 0.22 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        pill_w = max(84.0 * density, self.app.text.measure(title, layout.context.tokens.typography.body_l)[0] + 18.0 * density)
        pill_rect = Rect(rect.x + 16.0 * density, rect.y + 8.0 * density, pill_w, 18.0 * density)
        commands.panel(
            pill_rect,
            radius=pill_rect.h * 0.5,
            color=(colors.surface_container_low[0], colors.surface_container_low[1], colors.surface_container_low[2], 0.56 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        commands.text(
            title,
            pill_rect.x + 9.0 * density,
            pill_rect.y + 4.0 * density,
            layout.context.tokens.typography.body_l,
            color=colors.text_primary,
            alpha=0.94 * alpha,
        )
        draw_supporting_text(
            commands,
            theme,
            subtitle,
            pill_rect.x + 9.0 * density,
            rect.y + band_h + 2.0 * density,
            layout.context.tokens.typography.body_s,
            alpha=0.86 * alpha,
            tone="secondary",
        )
        return Rect(
            rect.x + 16.0 * density,
            rect.y + band_h + 28.0 * density,
            rect.w - 32.0 * density,
            rect.h - band_h - 44.0 * density,
        )

    def _draw_audio_section(
        self,
        commands: RenderCommandBuffer,
        layout: _SettingsLayout,
        density: float,
        content_progress: float,
    ) -> None:
        theme = layout.context.theme
        colors = theme.colors
        alpha, offset_y = self._section_visual(layout, 0, content_progress)
        section_rect = layout.audio_section_rect.translate(dy=offset_y)
        self._draw_section_shell(
            commands,
            layout,
            section_rect,
            title="Audio",
            subtitle="Custom hitsound support soon...",
            alpha=alpha,
        )
        for key_name, control in layout.audio_controls.items():
            row_rect = control["row_rect"].translate(dy=offset_y)
            track_rect = control["track_rect"].translate(dy=offset_y)
            mute_rect = control["mute_rect"].translate(dy=offset_y)
            hover_mix = self._mix(f"slider:{key_name}")
            commands.panel(
                row_rect,
                radius=theme.shape.corner_m,
                color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], (0.54 + hover_mix * 0.12) * alpha),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            draw_supporting_text(
                commands,
                theme,
                control["label"],
                row_rect.x + 12.0 * density,
                row_rect.y + 8.0 * density,
                layout.context.tokens.typography.body_s,
                alpha=alpha,
                tone="secondary",
            )
            value = self.app.settings.music_volume if key_name == "music" else self.app.settings.sfx_volume
            value_text = f"{int(round(value * 100))}%"
            value_w, _ = self.app.text.measure(value_text, layout.context.tokens.typography.body_s)
            commands.text(
                value_text,
                mute_rect.x - value_w - 22.0 * density,
                row_rect.y + 8.0 * density,
                layout.context.tokens.typography.body_s,
                color=colors.text_primary,
                alpha=0.88 * alpha,
            )
            draw_slider(
                commands,
                theme,
                track_rect,
                value=value,
                density=density,
                thumb_scale=1.0 + hover_mix * 0.18 + (0.10 if self._slider_dragging == key_name else 0.0),
                alpha=alpha,
            )
            muted = self.app.settings.music_muted if key_name == "music" else self.app.settings.sfx_muted
            draw_button(
                commands,
                self.app.text,
                theme,
                mute_rect,
                label="Muted" if muted else "Mute",
                size=layout.context.tokens.typography.body_s,
                variant="danger" if muted else "quiet",
                state=InteractionState.HOVER if self._hovered_id == f"mute:{key_name}" else (
                    InteractionState.SELECTED if muted else InteractionState.REST
                ),
                radius=layout.context.tokens.radius_s,
                alpha=alpha,
            )

    def _draw_display_section(
        self,
        commands: RenderCommandBuffer,
        layout: _SettingsLayout,
        density: float,
        content_progress: float,
    ) -> None:
        theme = layout.context.theme
        colors = theme.colors
        alpha, offset_y = self._section_visual(layout, 1, content_progress)
        section_rect = layout.display_section_rect.translate(dy=offset_y)
        self._draw_section_shell(
            commands,
            layout,
            section_rect,
            title="Graphics",
            subtitle="Display changes apply after restart. Gameplay visual settings apply immediately.",
            alpha=alpha,
        )
        for key_name in ("screen_mode", "resolution", "fps_limit"):
            select = layout.selects[key_name]
            field_rect = select.rect.translate(dy=offset_y)
            label_y = field_rect.y - layout.context.tokens.typography.body_s - 10.0 * density
            draw_supporting_text(
                commands,
                theme,
                select.label,
                field_rect.x,
                label_y,
                layout.context.tokens.typography.body_s,
                alpha=alpha,
                tone="secondary",
            )
            draw_select_field(
                commands,
                self.app.text,
                theme,
                field_rect,
                value=select.value_text,
                size=layout.context.tokens.typography.body_m,
                density=density,
                hovered=self._hovered_id == f"select:{key_name}",
                focused=self._menu_key == key_name and self._menu_open,
                open_progress=self._dropdown_anim.value if self._menu_key == key_name else 0.0,
                alpha=alpha,
                border_width=0.0,
            )

        helper_rect = layout.display_helper_rect.translate(dy=offset_y)
        draw_supporting_text(
            commands,
            theme,
            "Gameplay visuals",
            helper_rect.x,
            helper_rect.y,
            layout.context.tokens.typography.body_s,
            alpha=alpha * 0.92,
            tone="secondary",
        )

        for key_name, toggle in layout.graphics_toggles.items():
            row_rect = toggle["row_rect"].translate(dy=offset_y)
            button_rect = toggle["button_rect"].translate(dy=offset_y)
            hover_mix = self._mix(f"graphics_toggle:{key_name}")
            enabled = self._graphics_toggle_value(key_name)
            commands.panel(
                row_rect,
                radius=theme.shape.corner_m,
                color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], (0.52 + hover_mix * 0.10) * alpha),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            draw_supporting_text(
                commands,
                theme,
                toggle["label"],
                row_rect.x + 12.0 * density,
                row_rect.y + 8.0 * density,
                layout.context.tokens.typography.body_s,
                alpha=alpha,
                tone="secondary",
            )
            draw_supporting_text(
                commands,
                theme,
                toggle["hint"],
                row_rect.x + 12.0 * density,
                row_rect.y + 24.0 * density,
                layout.context.tokens.typography.body_s,
                alpha=alpha * 0.72,
                tone="muted",
            )
            draw_button(
                commands,
                self.app.text,
                theme,
                button_rect,
                label="ON" if enabled else "OFF",
                size=layout.context.tokens.typography.body_s,
                variant="quiet",
                state=InteractionState.SELECTED if enabled else (
                    InteractionState.HOVER if self._hovered_id == f"graphics_toggle:{key_name}" else InteractionState.REST
                ),
                radius=layout.context.tokens.radius_s,
                alpha=alpha,
            )

        for key_name, control in layout.graphics_sliders.items():
            row_rect = control["row_rect"].translate(dy=offset_y)
            track_rect = control["track_rect"].translate(dy=offset_y)
            hover_mix = self._mix(f"graphics_slider:{key_name}")
            commands.panel(
                row_rect,
                radius=theme.shape.corner_m,
                color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], (0.52 + hover_mix * 0.10) * alpha),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            draw_supporting_text(
                commands,
                theme,
                control["label"],
                row_rect.x + 12.0 * density,
                row_rect.y + 8.0 * density,
                layout.context.tokens.typography.body_s,
                alpha=alpha,
                tone="secondary",
            )
            value_text = self._graphics_slider_display(key_name)
            value_w, _ = self.app.text.measure(value_text, layout.context.tokens.typography.body_s)
            commands.text(
                value_text,
                row_rect.right - value_w - 12.0 * density,
                row_rect.y + 8.0 * density,
                layout.context.tokens.typography.body_s,
                color=colors.text_primary,
                alpha=0.88 * alpha,
            )
            draw_supporting_text(
                commands,
                theme,
                control["hint"],
                row_rect.x + 12.0 * density,
                row_rect.y + 24.0 * density,
                layout.context.tokens.typography.body_s,
                alpha=alpha * 0.72,
                tone="muted",
            )
            min_value, max_value = self._graphics_slider_bounds(key_name)
            value = self._graphics_slider_value(key_name)
            progress = 0.0 if max_value <= min_value else clamp((value - min_value) / (max_value - min_value), 0.0, 1.0)
            draw_slider(
                commands,
                theme,
                track_rect,
                value=progress,
                density=density,
                thumb_scale=1.0 + hover_mix * 0.18 + (0.10 if self._slider_dragging == key_name else 0.0),
                alpha=alpha,
            )

    def _draw_profile_section(
        self,
        commands: RenderCommandBuffer,
        layout: _SettingsLayout,
        density: float,
        content_progress: float,
    ) -> None:
        theme = layout.context.theme
        alpha, offset_y = self._section_visual(layout, 3, content_progress)
        section_rect = layout.profile_section_rect.translate(dy=offset_y)
        self._draw_section_shell(
            commands,
            layout,
            section_rect,
            title="Profile",
            subtitle="Personalize what appears in the UI.",
            alpha=alpha,
        )
        label_rect = layout.nickname_label_rect.translate(dy=offset_y)
        nickname_rect = layout.nickname_rect.translate(dy=offset_y)
        draw_supporting_text(
            commands,
            theme,
            "Nickname",
            label_rect.x,
            label_rect.y - 4.0,
            layout.context.tokens.typography.body_s,
            alpha=alpha,
            tone="secondary",
        )
        nickname = self.app.settings.nickname or "Enter nickname"
        draw_text_field(
            commands,
            self.app.text,
            theme,
            nickname_rect,
            value=nickname[:16],
            placeholder=not bool(self.app.settings.nickname),
            focused=self._nickname_focus or self._mix("field:nickname") > 0.45,
            size=layout.context.tokens.typography.body_m,
            density=density,
            alpha=alpha,
            border_width=0.0,
        )
        draw_supporting_text(
            commands,
            theme,
            "Shown in replay labels and future player-facing views.",
            nickname_rect.x,
            nickname_rect.bottom + 8.0 * density,
            layout.context.tokens.typography.body_s,
            alpha=alpha * 0.9,
            tone="muted",
        )

    def _draw_gameplay_section(
        self,
        commands: RenderCommandBuffer,
        layout: _SettingsLayout,
        density: float,
        content_progress: float,
    ) -> None:
        theme = layout.context.theme
        colors = theme.colors
        alpha, offset_y = self._section_visual(layout, 2, content_progress)
        section_rect = layout.gameplay_section_rect.translate(dy=offset_y)
        self._draw_section_shell(
            commands,
            layout,
            section_rect,
            title="Gameplay HUD",
            subtitle="Choose which gameplay overlays stay visible.",
            alpha=alpha,
        )
        preview_rect = layout.gameplay_preview_rect.translate(dy=offset_y)
        commands.panel(
            preview_rect,
            radius=16.0 * density,
            color=(colors.surface_container_low[0], colors.surface_container_low[1], colors.surface_container_low[2], 0.44 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        commands.panel(
            Rect(
                preview_rect.x + 10.0 * density,
                preview_rect.y + 10.0 * density,
                preview_rect.w - 20.0 * density,
                preview_rect.h - 20.0 * density,
            ),
            radius=12.0 * density,
            color=(colors.surface[0], colors.surface[1], colors.surface[2], 0.08 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        for key_name, toggle in layout.gameplay_toggles.items():
            tile_rect = toggle.rect.translate(dy=offset_y)
            enabled = bool(getattr(self.app.settings, key_name, True))
            hovered = self._hovered_id == f"toggle:{key_name}"
            if enabled:
                fill = (
                    colors.primary_container[0],
                    colors.primary_container[1],
                    colors.primary_container[2],
                    0.92 * alpha,
                )
                fg = colors.text_primary
                sub_fg = colors.text_primary
            elif hovered:
                fill = (
                    colors.surface_variant[0],
                    colors.surface_variant[1],
                    colors.surface_variant[2],
                    0.86 * alpha,
                )
                fg = colors.text_primary
                sub_fg = colors.text_secondary
            else:
                fill = (
                    colors.surface_variant_soft[0],
                    colors.surface_variant_soft[1],
                    colors.surface_variant_soft[2],
                    0.62 * alpha,
                )
                fg = colors.text_secondary
                sub_fg = colors.text_muted
            commands.panel(
                tile_rect,
                radius=toggle.radius,
                color=fill,
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            accent_rect = Rect(
                tile_rect.x + 8.0 * density,
                tile_rect.y + 7.0 * density,
                max(3.0 * density, min(4.0 * density, tile_rect.w * 0.08)),
                max(12.0 * density, tile_rect.h - 14.0 * density),
            )
            commands.panel(
                accent_rect,
                radius=accent_rect.w * 0.5,
                color=(colors.focus_ring[0], colors.focus_ring[1], colors.focus_ring[2], (0.90 if enabled else 0.42) * alpha),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            short_size = 12 if tile_rect.h < 42.0 * density else 13
            short_w, _ = self.app.text.measure(toggle.short_label, short_size)
            commands.text(
                toggle.short_label,
                tile_rect.x + 20.0 * density,
                tile_rect.y + 4.0 * density,
                short_size,
                color=fg,
                alpha=0.98 * alpha,
            )
            label_text = toggle.label
            if hasattr(self.app.text, "truncate"):
                label_text = self.app.text.truncate(toggle.label, 10, tile_rect.w - 28.0 * density)
            commands.text(
                label_text,
                tile_rect.x + 20.0 * density,
                tile_rect.y + 24.0 - 2.0 * density,
                10,
                color=sub_fg,
                alpha=0.82 * alpha,
            )

    def _section_visual(
        self,
        layout: _SettingsLayout,
        index: int,
        content_progress: float,
    ) -> tuple[float, float]:
        progress = stagger(content_progress, index, 0.14)
        alpha = progress * ease_out_cubic(self._open_anim.value)
        offset = (1.0 - progress) * 16.0 * layout.context.density - self._scroll.value
        return (alpha, offset)

    def _animated_rect(self, layout: _SettingsLayout, rect: Rect, *, section_index: int) -> Rect:
        _, offset_y = self._section_visual(layout, section_index, ease_in_out_cubic(self._content_anim.value))
        return rect.translate(dy=offset_y)

    def _mode_options(self) -> list[tuple[str, str]]:
        return [(value, value.replace("_", " ").title()) for value in _SCREEN_MODES]

    def _resolution_options(self) -> list[tuple[tuple[int, int], str]]:
        return [
            ((width, height), f"{width}x{height}")
            for width, height in self.app.common_resolutions
        ]

    def _fps_limit_options(self) -> list[tuple[int, str]]:
        return list(_FPS_LIMIT_OPTIONS)

    def _select_options(self, key: str) -> list[tuple[object, str]]:
        if key == "screen_mode":
            return self._mode_options()
        if key == "resolution":
            return self._resolution_options()
        return self._fps_limit_options()

    def _selected_option_index(self, key: str, options: list[tuple[object, str]]) -> int:
        current_value: object
        if key == "screen_mode":
            current_value = self.app.settings.screen_mode
        elif key == "resolution":
            current_value = (
                self.app.settings.resolution_width,
                self.app.settings.resolution_height,
            )
        else:
            current_value = int(getattr(self.app.settings, "fps_limit", 0))
        for idx, (value, _label) in enumerate(options):
            if value == current_value:
                return idx
        return 0

    def _current_select_text(self, key: str, options: list[tuple[object, str]]) -> str:
        idx = self._selected_option_index(key, options)
        return options[idx][1] if options else ""

    def _toggle_select(self, key: str, layout: _SettingsLayout) -> None:
        if self._menu_key == key and self._menu_open:
            self._close_select()
            return
        self._menu_key = key
        self._menu_open = True
        self._dropdown_anim.set_target(1.0)
        self._nickname_focus = False
        self._menu_highlight_index = layout.selects[key].selected_index

    def _close_select(self) -> None:
        self._menu_open = False
        self._dropdown_anim.set_target(0.0)

    def _move_menu_highlight(self, delta: int) -> None:
        if self._menu_key is None:
            return
        options = self._select_options(self._menu_key)
        if not options:
            self._menu_highlight_index = -1
            return
        if self._menu_highlight_index < 0:
            self._menu_highlight_index = self._selected_option_index(self._menu_key, options)
        self._menu_highlight_index = (self._menu_highlight_index + delta) % len(options)

    def _choose_dropdown_index(self, key: str, index: int) -> None:
        options = self._select_options(key)
        if not (0 <= index < len(options)):
            return
        value, _label = options[index]
        if key == "screen_mode":
            self.app._set_screen_mode(str(value))
        elif key == "resolution":
            width, height = value
            self.app._set_resolution(int(width), int(height))
        else:
            self.app._set_fps_limit(int(value))
        self.invalidate()
        self._close_select()

    def _graphics_toggle_value(self, key: str) -> bool:
        return bool(getattr(self.app.settings, key, False))

    def _graphics_slider_value(self, key: str) -> float:
        if key == "gameplay_cursor_trail_max_len":
            return float(getattr(self.app.settings, key, 256))
        return float(getattr(self.app.settings, key, 0.0))

    def _graphics_slider_bounds(self, key: str) -> tuple[float, float]:
        if key == "gameplay_cursor_trail_max_len":
            return (8.0, 256.0)
        return (0.0, 1.0)

    def _graphics_slider_display(self, key: str) -> str:
        value = self._graphics_slider_value(key)
        if key == "gameplay_cursor_trail_max_len":
            return f"{int(round(value))}"
        return f"{int(round(value * 100.0))}%"

    def _slider_control(self, layout: _SettingsLayout, key: str) -> tuple[dict, int] | None:
        if key in layout.audio_controls:
            return layout.audio_controls[key], 0
        if key in layout.graphics_sliders:
            return layout.graphics_sliders[key], 1
        return None

    def _toggle_graphics_setting(self, key: str) -> None:
        self.app.set_graphics_setting(key, not self._graphics_toggle_value(key))
        self.invalidate()

    def _menu_geometry(self, layout: _SettingsLayout):
        if self._menu_key is None:
            return None
        select = layout.selects.get(self._menu_key)
        if select is None:
            return None
        density = layout.context.density
        viewport = layout.context.viewport
        trigger_rect = self._animated_rect(layout, select.rect, section_index=1)
        row_h = 36.0 * density
        pad = 6.0 * density
        labels = [label for _value, label in select.options]
        text_size = layout.context.tokens.typography.body_m
        text_width = max(
            (self.app.text.measure(label, text_size)[0] for label in labels),
            default=0.0,
        )
        menu_w = max(trigger_rect.w, text_width + 36.0 * density)
        menu_h = pad * 2.0 + row_h * len(labels)
        x = clamp(
            trigger_rect.x,
            viewport.x + 12.0 * density,
            viewport.right - menu_w - 12.0 * density,
        )
        below_y = trigger_rect.bottom + 8.0 * density
        above_y = trigger_rect.y - menu_h - 8.0 * density
        if below_y + menu_h <= viewport.bottom - 12.0 * density or above_y < viewport.y + 12.0 * density:
            y = below_y
        else:
            y = above_y
        menu_rect = Rect(x, y, menu_w, menu_h)
        option_rects = [
            Rect(x + pad, y + pad + row_h * idx, menu_w - pad * 2.0, row_h)
            for idx in range(len(labels))
        ]
        return {
            "trigger_rect": trigger_rect,
            "menu_rect": menu_rect,
            "option_rects": option_rects,
            "labels": labels,
            "selected_index": select.selected_index,
        }

    def _apply_slider_value(
        self,
        key: str,
        x: float,
        track_rect: Rect,
        *,
        persist: bool,
    ) -> None:
        value = clamp((x - track_rect.x) / max(1.0, track_rect.w), 0.0, 1.0)
        if key == "music":
            self.app.set_music_volume(value, persist=persist)
            self.app.set_music_muted(value <= 0.0001, persist=persist)
        elif key == "sfx":
            self.app.set_sfx_volume(value, persist=persist)
            self.app.set_sfx_muted(value <= 0.0001, persist=persist)
        else:
            min_value, max_value = self._graphics_slider_bounds(key)
            actual = min_value + (max_value - min_value) * value
            if key == "gameplay_cursor_trail_max_len":
                actual = int(round(actual))
            self.app.set_graphics_setting(key, actual, persist=persist)
            self.invalidate()

    def _update_hover_mix(self, dt: float) -> None:
        tracked = [
            "button:close",
            "button:back",
            "field:nickname",
            "select:screen_mode",
            "select:resolution",
            "select:fps_limit",
        ]
        tracked.extend(f"graphics_toggle:{key}" for key, _label in _GRAPHICS_TOGGLES)
        tracked.extend(f"graphics_slider:{key}" for key, _label in _GRAPHICS_SLIDERS)
        tracked.extend(f"toggle:{key}" for key, _label, _short in _GAMEPLAY_HUD_TOGGLES)
        tracked.extend(f"slider:{key}" for key, _ in _AUDIO_KEYS)
        tracked.extend(f"mute:{key}" for key, _ in _AUDIO_KEYS)
        for key in tracked:
            anim = self._hover_mix.setdefault(key, AnimatedFloat(0.0, 0.0, 14.0))
            active = self._hovered_id == key
            if key == f"select:{self._menu_key}" and self._menu_open:
                active = True
            if key == "field:nickname" and self._nickname_focus:
                active = True
            if self._slider_dragging is not None and key == f"slider:{self._slider_dragging}":
                active = True
            anim.set_target(1.0 if active else 0.0)
            anim.update(dt)

    def _mix(self, key: str) -> float:
        anim = self._hover_mix.get(key)
        return anim.value if anim is not None else 0.0

    def _sync_hover(self) -> None:
        layout = self._build_layout()
        self._button_hover = (
            layout.button_rect.contains(self._mouse_x, self._mouse_y)
            if layout.button_rect is not None
            else False
        )
        self._hovered_id = None
        if not self._open:
            return
        if layout.close_rect.contains(self._mouse_x, self._mouse_y):
            self._hovered_id = "button:close"
            return
        if layout.back_rect.contains(self._mouse_x, self._mouse_y):
            self._hovered_id = "button:back"
            return
        if self._animated_rect(layout, layout.nickname_rect, section_index=3).contains(
            self._mouse_x,
            self._mouse_y,
        ):
            self._hovered_id = "field:nickname"
            return

        menu_geo = self._menu_geometry(layout)
        if self._menu_key is not None and menu_geo is not None:
            if menu_geo["trigger_rect"].contains(self._mouse_x, self._mouse_y):
                self._hovered_id = f"select:{self._menu_key}"
                return
            if menu_geo["menu_rect"].contains(self._mouse_x, self._mouse_y):
                for idx, rect in enumerate(menu_geo["option_rects"]):
                    if rect.contains(self._mouse_x, self._mouse_y):
                        self._menu_highlight_index = idx
                        self._hovered_id = f"option:{self._menu_key}"
                        return

        for key_name, select in layout.selects.items():
            if self._animated_rect(layout, select.rect, section_index=1).contains(
                self._mouse_x,
                self._mouse_y,
            ):
                self._hovered_id = f"select:{key_name}"
                return

        for key_name, toggle in layout.graphics_toggles.items():
            if self._animated_rect(layout, toggle["row_rect"], section_index=1).contains(
                self._mouse_x,
                self._mouse_y,
            ):
                self._hovered_id = f"graphics_toggle:{key_name}"
                return

        for key_name, control in layout.graphics_sliders.items():
            if self._animated_rect(layout, control["row_rect"], section_index=1).contains(
                self._mouse_x,
                self._mouse_y,
            ):
                self._hovered_id = f"graphics_slider:{key_name}"
                return

        for key_name, toggle in layout.gameplay_toggles.items():
            if self._animated_rect(layout, toggle.rect, section_index=2).contains(
                self._mouse_x,
                self._mouse_y,
            ):
                self._hovered_id = f"toggle:{key_name}"
                return

        for key_name, control in layout.audio_controls.items():
            if self._animated_rect(layout, control["mute_rect"], section_index=0).contains(
                self._mouse_x,
                self._mouse_y,
            ):
                self._hovered_id = f"mute:{key_name}"
                return
            if self._animated_rect(layout, control["row_rect"], section_index=0).contains(
                self._mouse_x,
                self._mouse_y,
            ):
                self._hovered_id = f"slider:{key_name}"
                return

    def _build_layout(self) -> _SettingsLayout:
        width, height = self.app.wnd.buffer_size
        progress_key = int(round(self._open_anim.value * 1000.0))
        scroll_key = int(round(self._scroll.value))
        key = (width, height, progress_key, scroll_key)
        if self._layout is not None and self._layout_key == key:
            return self._layout

        context = build_layout_context(width, height)
        density = context.density
        theme = context.theme
        tokens = context.tokens
        spacing = theme.spacing

        button_rect = None
        if self.app._settings_button_visible and self.app._settings_button_rect is not None:
            bx, by, bw, bh = self.app._settings_button_rect
            button_rect = Rect(bx, by, bw, bh)

        drawer_progress = ease_out_cubic(self._open_anim.value)
        drawer_w = clamp(tokens.drawer_width, tokens.drawer_min_width, context.content_rect.w * 0.52)
        hidden_x = -drawer_w - tokens.gap_xl
        visible_x = context.viewport.x
        drawer_x = hidden_x + (visible_x - hidden_x) * drawer_progress
        drawer_rect = Rect(drawer_x, context.viewport.y, drawer_w, context.viewport.h)
        backdrop_rect = context.viewport

        side_pad = 24.0 * density
        header_h = 96.0 * density
        footer_h = 74.0 * density
        header_rect = Rect(drawer_rect.x, drawer_rect.y, drawer_rect.w, header_h)
        title_rect = Rect(drawer_rect.x + side_pad + 12.0, drawer_rect.y + 22.0 * density, drawer_rect.w, theme.typography.headline)
        subtitle_rect = Rect(drawer_rect.x + side_pad, title_rect.y + theme.typography.headline + 6.0 * density, drawer_rect.w, tokens.typography.body_s)
        close_rect = Rect(drawer_rect.right - side_pad - 42.0 * density, drawer_rect.y + 20.0 * density, 42.0 * density, 36.0 * density)
        scroll_rect = Rect(
            drawer_rect.x + side_pad,
            header_rect.bottom,
            drawer_rect.w - side_pad * 2.0,
            drawer_rect.h - header_h - footer_h,
        )
        footer_rect = Rect(drawer_rect.x, drawer_rect.bottom - footer_h, drawer_rect.w, footer_h)
        back_rect = Rect(drawer_rect.x + side_pad, footer_rect.y + 18.0 * density, 118.0 * density, 38.0 * density)
        build_rect = Rect(back_rect.right + 16.0 * density, footer_rect.y + 28.0 * density, drawer_rect.w, tokens.typography.body_s)

        card_w = scroll_rect.w
        content_x = scroll_rect.x
        y = scroll_rect.y + 10.0 * density
        section_gap = spacing.lg
        card_pad = 26.0 * density
        title_block_h = tokens.typography.body_l + tokens.typography.body_s + 18.0 * density
        row_h = 58.0 * density
        field_h = 50.0 * density

        audio_section_h = card_pad * 2.0 + title_block_h + row_h * len(_AUDIO_KEYS) + spacing.sm
        audio_section_rect = Rect(content_x, y, card_w, audio_section_h)
        audio_controls: dict[str, dict] = {}
        audio_y = audio_section_rect.y + card_pad + title_block_h
        for idx, (key_name, label) in enumerate(_AUDIO_KEYS):
            row_rect = Rect(content_x + card_pad, audio_y + idx * row_h, card_w - card_pad * 2.0, 48.0 * density)
            mute_rect = Rect(row_rect.right - 72.0 * density, row_rect.y + 16.0 * density, 60.0 * density, 28.0 * density)
            track_rect = Rect(row_rect.x + 12.0 * density, row_rect.y + 28.0 * density, row_rect.w - 108.0 * density, 8.0 * density)
            audio_controls[key_name] = {
                "label": label,
                "row_rect": row_rect,
                "track_rect": track_rect,
                "mute_rect": mute_rect,
            }
        y = audio_section_rect.bottom + section_gap

        graphics_toggle_row_h = 46.0 * density
        graphics_slider_row_h = 64.0 * density
        graphics_gap = 8.0 * density
        select_label_gap = tokens.typography.body_s + 6.0 * density
        select_stack_h = (
            select_label_gap * 3.0
            + field_h * 3.0
            + spacing.md * 2.0
        )
        graphics_toggle_stack_h = (
            len(_GRAPHICS_TOGGLES) * graphics_toggle_row_h
            + max(0, len(_GRAPHICS_TOGGLES) - 1) * graphics_gap
        )
        graphics_slider_stack_h = (
            len(_GRAPHICS_SLIDERS) * graphics_slider_row_h
            + max(0, len(_GRAPHICS_SLIDERS) - 1) * graphics_gap
        )
        display_section_h = (
            card_pad * 2.0
            + title_block_h
            + select_stack_h
            + 12.0 * density
            + tokens.typography.body_s * 2.0
            + 18.0 * density
            + graphics_toggle_stack_h
            + 14.0 * density
            + graphics_slider_stack_h
        )
        display_section_rect = Rect(content_x, y, card_w, display_section_h)
        display_inner_y = display_section_rect.y + card_pad + title_block_h + tokens.typography.body_s + 6.0 * density
        mode_options = self._mode_options()
        resolution_options = self._resolution_options()
        fps_limit_options = self._fps_limit_options()
        mode_field_rect = Rect(content_x + card_pad, display_inner_y, card_w - card_pad * 2.0, field_h)
        resolution_field_rect = Rect(content_x + card_pad, mode_field_rect.bottom + spacing.md + tokens.typography.body_s + 6.0 * density, card_w - card_pad * 2.0, field_h)
        fps_limit_field_rect = Rect(content_x + card_pad, resolution_field_rect.bottom + spacing.md + tokens.typography.body_s + 6.0 * density, card_w - card_pad * 2.0, field_h)
        display_helper_rect = Rect(content_x + card_pad, fps_limit_field_rect.bottom + 12.0 * density, card_w - card_pad * 2.0, tokens.typography.body_s * 2.0)
        selects = {
            "screen_mode": _SelectLayout(
                key="screen_mode",
                label="Screen Mode",
                rect=mode_field_rect,
                options=mode_options,
                selected_index=self._selected_option_index("screen_mode", mode_options),
                value_text=self._current_select_text("screen_mode", mode_options),
            ),
            "resolution": _SelectLayout(
                key="resolution",
                label="Resolution",
                rect=resolution_field_rect,
                options=resolution_options,
                selected_index=self._selected_option_index("resolution", resolution_options),
                value_text=self._current_select_text("resolution", resolution_options),
            ),
            "fps_limit": _SelectLayout(
                key="fps_limit",
                label="FPS Limit",
                rect=fps_limit_field_rect,
                options=fps_limit_options,
                selected_index=self._selected_option_index("fps_limit", fps_limit_options),
                value_text=self._current_select_text("fps_limit", fps_limit_options),
            ),
        }
        graphics_toggles: dict[str, dict] = {}
        graphics_toggle_y = display_helper_rect.bottom + 18.0 * density
        for idx, (key_name, label) in enumerate(_GRAPHICS_TOGGLES):
            row_rect = Rect(
                content_x + card_pad,
                graphics_toggle_y + idx * (graphics_toggle_row_h + graphics_gap),
                card_w - card_pad * 2.0,
                graphics_toggle_row_h,
            )
            button_rect = Rect(
                row_rect.right - 68.0 * density,
                row_rect.y + 9.0 * density,
                56.0 * density,
                28.0 * density,
            )
            graphics_toggles[key_name] = {
                "label": label,
                "hint": (
                    "Corner glow over the solid gameplay background."
                    if key_name == "gameplay_background_bloom"
                    else (
                        "Uses the beatmap background instead of the solid fill."
                        if key_name == "gameplay_background_image"
                        else "Show or hide the cursor trail behind cursors."
                    )
                ),
                "row_rect": row_rect,
                "button_rect": button_rect,
            }

        graphics_sliders: dict[str, dict] = {}
        graphics_slider_y = graphics_toggle_y + len(_GRAPHICS_TOGGLES) * (graphics_toggle_row_h + graphics_gap) - graphics_gap + 14.0 * density
        for idx, (key_name, label) in enumerate(_GRAPHICS_SLIDERS):
            row_rect = Rect(
                content_x + card_pad,
                graphics_slider_y + idx * (graphics_slider_row_h + graphics_gap),
                card_w - card_pad * 2.0,
                graphics_slider_row_h,
            )
            track_rect = Rect(
                row_rect.x + 12.0 * density,
                row_rect.y + 46.0 * density,
                row_rect.w - 24.0 * density,
                8.0 * density,
            )
            graphics_sliders[key_name] = {
                "label": label,
                "hint": (
                    "Applies only when beatmap background image is enabled."
                    if key_name == "gameplay_background_dim"
                    else (
                        "Limits how many points are used by the trail."
                        if key_name == "gameplay_cursor_trail_max_len"
                        else "Controls additive bloom strength during gameplay."
                    )
                ),
                "row_rect": row_rect,
                "track_rect": track_rect,
            }
        y = display_section_rect.bottom + section_gap

        gameplay_preview_w = card_w - card_pad * 2.0
        gameplay_preview_h = clamp(gameplay_preview_w * 0.54, 168.0 * density, 238.0 * density)
        gameplay_section_h = card_pad * 2.0 + title_block_h + gameplay_preview_h + 18.0 * density
        gameplay_section_rect = Rect(content_x, y, card_w, gameplay_section_h)
        gameplay_preview_rect = Rect(
            content_x + card_pad,
            gameplay_section_rect.y + card_pad + title_block_h,
            gameplay_preview_w,
            gameplay_preview_h,
        )
        gameplay_toggles: dict[str, _ToggleTileLayout] = {}
        px = gameplay_preview_rect.x
        py = gameplay_preview_rect.y
        pw = gameplay_preview_rect.w
        ph = gameplay_preview_rect.h
        inset_x = pw * 0.04
        inset_y = ph * 0.06
        hp_w = pw * 0.48
        hp_h = ph * 0.13
        acc_w = pw * 0.29
        acc_h = ph * 0.18
        leaderboard_w = pw * 0.27
        leaderboard_h = ph * 0.38
        keys_w = pw * 0.12
        keys_h = ph * 0.24
        combo_w = pw * 0.18
        combo_h = ph * 0.11
        timeline_w = pw * 0.45
        timeline_h = ph * 0.10
        toggle_frames = {
            "draw_gameplay_hp": Rect(px + inset_x, py + inset_y, hp_w, hp_h),
            "draw_gameplay_acc_pp": Rect(px + pw - inset_x - acc_w, py + inset_y, acc_w, acc_h),
            "draw_gameplay_leaderboard": Rect(px + inset_x, py + (ph - leaderboard_h) * 0.5, leaderboard_w, leaderboard_h),
            "draw_gameplay_keys": Rect(px + pw - inset_x - keys_w, py + (ph - keys_h) * 0.5, keys_w, keys_h),
            "draw_gameplay_combo": Rect(px + inset_x, py + ph - inset_y - combo_h, combo_w, combo_h),
            "draw_gameplay_timeline": Rect(px + pw - inset_x - timeline_w, py + ph - inset_y - timeline_h, timeline_w, timeline_h),
        }
        toggle_radii = {
            "draw_gameplay_hp": 12.0 * density,
            "draw_gameplay_acc_pp": 14.0 * density,
            "draw_gameplay_leaderboard": 12.0 * density,
            "draw_gameplay_keys": 11.0 * density,
            "draw_gameplay_combo": 12.0 * density,
            "draw_gameplay_timeline": 10.0 * density,
        }
        for key_name, label, short_label in _GAMEPLAY_HUD_TOGGLES:
            tile_rect = toggle_frames[key_name]
            gameplay_toggles[key_name] = _ToggleTileLayout(
                key=key_name,
                label=label,
                short_label=short_label,
                rect=tile_rect,
                radius=toggle_radii[key_name],
            )
        y = gameplay_section_rect.bottom + section_gap

        profile_section_h = card_pad * 2.0 + title_block_h + field_h + 44.0 * density
        profile_section_rect = Rect(content_x, y, card_w, profile_section_h)
        nickname_label_rect = Rect(content_x + card_pad, profile_section_rect.y + card_pad + title_block_h, card_w, tokens.typography.body_s)
        nickname_rect = Rect(content_x + card_pad, nickname_label_rect.bottom + 6.0 * density, card_w - card_pad * 2.0, field_h)
        y = profile_section_rect.bottom + max(spacing.xxl, 44.0 * density)

        content_height = y - scroll_rect.y
        max_scroll = max(0.0, content_height - scroll_rect.h)
        self._scroll.value = clamp(self._scroll.value, 0.0, max_scroll)
        self._scroll.target = clamp(self._scroll.target, 0.0, max_scroll)

        layout = _SettingsLayout(
            context=context,
            button_rect=button_rect,
            backdrop_rect=backdrop_rect,
            drawer_rect=drawer_rect,
            header_rect=header_rect,
            title_rect=title_rect,
            subtitle_rect=subtitle_rect,
            close_rect=close_rect,
            scroll_rect=scroll_rect,
            footer_rect=footer_rect,
            back_rect=back_rect,
            build_rect=build_rect,
            content_height=content_height,
            audio_section_rect=audio_section_rect,
            display_section_rect=display_section_rect,
            gameplay_section_rect=gameplay_section_rect,
            gameplay_preview_rect=gameplay_preview_rect,
            profile_section_rect=profile_section_rect,
            audio_controls=audio_controls,
            selects=selects,
            graphics_toggles=graphics_toggles,
            graphics_sliders=graphics_sliders,
            gameplay_toggles=gameplay_toggles,
            display_helper_rect=display_helper_rect,
            nickname_label_rect=nickname_label_rect,
            nickname_rect=nickname_rect,
        )
        self._layout = layout
        self._layout_key = key
        return layout

