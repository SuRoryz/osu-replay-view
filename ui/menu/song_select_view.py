"""Responsive song select menu view built on the shared layout system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from profiling import profiler
from runtime_paths import MAPS_DIR, replay_dir_for_set
from ui.design import (
    InteractionState,
    draw_button,
    draw_chip,
    draw_divider,
    draw_dropdown_menu,
    draw_header_pair,
    draw_linear_progress,
    draw_surface,
    draw_tab,
)
from ui.menu.animation import ease_out_back, ease_out_cubic
from ui.menu.commands import RenderCommandBuffer
from ui.menu.layout import Insets, LayoutContext, LayoutMode, Rect, split_columns


@dataclass(slots=True)
class SongSelectBaseLayout:
    context: LayoutContext
    main_rect: Rect
    info_panel_rect: Rect
    song_list_rect: Rect
    bottom_bar_rect: Rect


def _truncate_text(text, value: str, size: int, max_width: float) -> str:
    if max_width <= 0:
        return ""
    if text.measure(value, size)[0] <= max_width:
        return value
    return text.truncate(value, size, max_width)


def _hover_anim(scene, key: str, hovered: bool, *, speed: float = 0.22) -> float:
    if not hasattr(scene, "_ui_hover_anim"):
        scene._ui_hover_anim = {}
    current = float(scene._ui_hover_anim.get(key, 0.0))
    target = 1.0 if hovered else 0.0
    current += (target - current) * speed
    if current < 0.001 and not hovered:
        current = 0.0
    scene._ui_hover_anim[key] = current
    return current


class SongSelectMenuView:
    def __init__(self) -> None:
        self._commands = RenderCommandBuffer()

    def _draw_eye_metric(self, theme, rect: Rect, *, value: str, size: int, alpha: float) -> None:
        colors = theme.colors
        eye_rect = Rect(rect.x - 4.0, rect.y + 3.0, 12.0, 8.0)
        self._commands.panel(
            eye_rect,
            radius=eye_rect.h * 0.5,
            color=(colors.surface_variant[0], colors.surface_variant[1], colors.surface_variant[2], 0.18 * alpha),
            border_color=(colors.text_muted[0], colors.text_muted[1], colors.text_muted[2], 0.52 * alpha),
            border_width=1.0,
        )
        pupil_size = 3.0
        self._commands.panel(
            Rect(eye_rect.center_x - pupil_size * 0.5, eye_rect.center_y - pupil_size * 0.5, pupil_size, pupil_size),
            radius=pupil_size * 0.5,
            color=(colors.text_muted[0], colors.text_muted[1], colors.text_muted[2], 0.76 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        self._commands.text(
            value,
            eye_rect.right + 5.0,
            rect.y - 2.0,
            size,
            color=colors.text_muted,
            alpha=0.82 * alpha,
        )

    def _draw_kebab_button(self, theme, rect: Rect, *, hovered: bool, alpha: float) -> None:
        draw_surface(
            self._commands,
            theme,
            rect,
            role="section",
            radius=min(rect.w, rect.h) * 0.45,
            alpha=alpha * (0.48 if hovered else 0.26),
            border_width=0.0,
        )
        dot_x = rect.center_x - 1.5
        dot_r = 1.6
        color = theme.colors.text_primary if hovered else theme.colors.text_secondary
        for idx in range(3):
            cy = rect.y + rect.h * 0.28 + idx * rect.h * 0.22 - 2.0
            self._commands.panel(
                Rect(dot_x, cy, dot_r * 2.0, dot_r * 2.0),
                radius=dot_r,
                color=(color[0], color[1], color[2], 0.92 * alpha),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )

    def _draw_settings_button(self, text, theme, rect: Rect, *, hovered: bool, alpha: float) -> None:
        draw_button(
            self._commands,
            text,
            theme,
            rect,
            label="",
            size=1,
            variant="secondary",
            state=InteractionState.HOVER if hovered else InteractionState.REST,
            radius=min(theme.shape.corner_m, rect.h * 0.3),
            alpha=alpha,
        )
        self._draw_settings_icon(theme, rect, hovered=hovered, alpha=alpha)

    def _draw_chat_button(self, text, theme, rect: Rect, *, hovered: bool, alpha: float) -> None:
        colors = theme.colors
        draw_button(
            self._commands,
            text,
            theme,
            rect,
            label="",
            size=1,
            variant="secondary",
            state=InteractionState.HOVER if hovered else InteractionState.REST,
            radius=min(theme.shape.corner_m, rect.h * 0.3),
            alpha=alpha,
        )
        icon_color = colors.text_primary if hovered else colors.text_secondary
        bubble_w = rect.w * 0.315
        bubble_h = rect.h * 0.21
        bubble_rect = Rect(
            rect.center_x - bubble_w * 0.5,
            rect.center_y - bubble_h * 0.5 - rect.h * 0.05,
            bubble_w,
            bubble_h,
        )
        self._commands.panel(
            bubble_rect,
            radius=bubble_h * 0.42,
            color=(icon_color[0], icon_color[1], icon_color[2], 0.94 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        tail_rect = Rect(
            bubble_rect.x + bubble_w * 0.20,
            bubble_rect.bottom - rect.h * 0.02,
            rect.w * 0.075,
            rect.h * 0.075,
        )
        self._commands.panel(
            tail_rect,
            radius=tail_rect.w * 0.25,
            color=(icon_color[0], icon_color[1], icon_color[2], 0.94 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        dot_r = max(1.0, rect.w * 0.0225)
        dot_y = bubble_rect.center_y - dot_r
        for idx in range(3):
            dot_x = bubble_rect.x + bubble_w * (0.30 + idx * 0.20) - dot_r
            self._commands.panel(
                Rect(dot_x, dot_y, dot_r * 2.0, dot_r * 2.0),
                radius=dot_r,
                color=(theme.colors.surface_variant_soft[0], theme.colors.surface_variant_soft[1], theme.colors.surface_variant_soft[2], 0.98 * alpha),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )

    def _draw_settings_icon(self, theme, rect: Rect, *, hovered: bool, alpha: float) -> None:
        colors = theme.colors
        icon_color = colors.text_primary if hovered else colors.text_secondary
        cx = rect.center_x
        cy = rect.center_y
        line_w = rect.w * 0.33
        line_h = max(1.5, rect.h * 0.05625)
        knob_r = rect.h * 0.0975
        top_y = cy - rect.h * 0.18
        bottom_y = cy + rect.h * 0.18

        self._commands.panel(
            Rect(cx - line_w * 0.5, top_y - line_h * 0.5, line_w, line_h),
            radius=line_h * 0.5,
            color=(icon_color[0], icon_color[1], icon_color[2], 0.94 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        self._commands.panel(
            Rect(cx - line_w * 0.5, bottom_y - line_h * 0.5, line_w, line_h),
            radius=line_h * 0.5,
            color=(icon_color[0], icon_color[1], icon_color[2], 0.94 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        for knob_x, knob_y in (
            (cx - line_w * 0.28, top_y),
            (cx + line_w * 0.28, bottom_y),
        ):
            outer = knob_r * 2.0
            inner = knob_r * 1.08
            self._commands.panel(
                Rect(knob_x - outer * 0.5, knob_y - outer * 0.5, outer, outer),
                radius=outer * 0.5,
                color=(icon_color[0], icon_color[1], icon_color[2], 0.94 * alpha),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            self._commands.panel(
                Rect(knob_x - inner * 0.5, knob_y - inner * 0.5, inner, inner),
                radius=inner * 0.5,
                color=(theme.colors.surface_variant_soft[0], theme.colors.surface_variant_soft[1], theme.colors.surface_variant_soft[2], 0.98 * alpha),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )

    def base_layout(self, context: LayoutContext) -> SongSelectBaseLayout:
        tokens = context.tokens
        if context.mode in {LayoutMode.WIDE, LayoutMode.ULTRAWIDE}:
            outer_base = Rect(
                context.viewport.x,
                context.safe_area.y,
                context.viewport.w,
                context.safe_area.h,
            )
        else:
            outer_base = context.content_rect
        outer = outer_base.inset(Insets.symmetric(tokens.gutter * 0.4, tokens.gutter * 0.4))
        bottom_bar_rect = Rect(
            context.viewport.x,
            context.viewport.bottom - tokens.bottom_bar_height,
            context.viewport.w,
            tokens.bottom_bar_height,
        )
        main_rect = Rect(
            outer.x,
            outer.y,
            outer.w,
            max(0.0, bottom_bar_rect.y - outer.y - tokens.gap_l),
        )
        if context.mode == LayoutMode.NARROW:
            info_h = min(main_rect.h * 0.44, 460.0 * context.density)
            info_panel_rect = Rect(main_rect.x, main_rect.y, main_rect.w, info_h)
            song_list_rect = Rect(main_rect.x, info_panel_rect.bottom + tokens.gap_l, main_rect.w, max(0.0, main_rect.bottom - info_panel_rect.bottom - tokens.gap_l))
        else:
            left_ratio = {
                LayoutMode.STANDARD: 0.41,
                LayoutMode.WIDE: 0.39,
                LayoutMode.ULTRAWIDE: 0.36,
            }.get(context.mode, 0.40)
            info_panel_rect, song_list_rect = split_columns(main_rect, ratio=left_ratio, gap=tokens.gap_l)
        return SongSelectBaseLayout(
            context=context,
            main_rect=main_rect,
            info_panel_rect=info_panel_rect,
            song_list_rect=song_list_rect,
            bottom_bar_rect=bottom_bar_rect,
        )

    def song_list_metrics(self, scene, *, x_offset: float = 0.0) -> dict:
        layout = self.base_layout(scene.app.menu_context())
        density = layout.context.density
        header_h = 42.0 * density
        visible_rect = Rect(
            layout.song_list_rect.x + x_offset,
            layout.song_list_rect.y + header_h,
            layout.song_list_rect.w,
            max(0.0, layout.song_list_rect.h - header_h),
        )
        item_h = (94.0 + 10.0) * density
        card_w = max(300.0 * density, visible_rect.w - 18.0 * density)
        return {
            "layout": layout,
            "visible_rect": visible_rect,
            "header_y": layout.song_list_rect.y + 10.0 * density,
            "item_h": item_h,
            "card_w": card_w,
            "list_x": visible_rect.x,
        }

    def song_card_rect(self, scene, metrics: dict, flat_idx: int, *, set_idx: int, diff_idx: int) -> Rect:
        density = metrics["layout"].context.density
        visible_rect: Rect = metrics["visible_rect"]
        list_h = visible_rect.h
        card_h = 94.0 * density
        cy = flat_idx * metrics["item_h"] - scene._scroll_current + visible_rect.y
        center_y = visible_rect.y + list_h * 0.5 if list_h > 0 else visible_rect.y
        depth = min(1.0, abs((cy + card_h * 0.5) - center_y) / max(1.0, list_h * 0.5))
        base_x = metrics["list_x"] + 8.0 * density * (1.0 - depth)
        if 0 <= set_idx < len(scene._sets):
            bset = scene._sets[set_idx]
            if set_idx == scene._selected_idx and len(bset.maps) > 1:
                base_x += 14.0 * density
        if scene._hover_idx == flat_idx:
            base_x += 10.0 * density
        return Rect(base_x, cy, metrics["card_w"], card_h)

    def draw(self, scene, *, info_x_off: float = 0.0, cards_x_off: float = 0.0, bar_y_off: float = 0.0) -> SongSelectBaseLayout:
        context = scene.app.menu_context()
        layout = self.base_layout(context)
        text = scene.app.text
        panels = scene.app.panels
        self._commands.clear()
        scene._song_card_rects = []
        scene._replay_rects = []
        scene._mod_rects = []
        scene._mods_trigger_rect = None
        scene._mods_surface_rect = None
        scene._songs_open_btn_rect = None
        scene._multi_toggle_rect = None
        scene._danser_toggle_rect = None
        scene._hover_idx = -1
        scene._song_list_interact_rect = layout.song_list_rect

        self._draw_info_panel(scene, layout, x_offset=info_x_off)
        self._draw_song_list(scene, layout, x_offset=cards_x_off)
        self._draw_bottom_bar(scene, layout, y_offset=bar_y_off)
        self._commands.flush(
            ctx=scene.app.ctx,
            text=text,
            panels=panels,
            window_height=scene.app.wnd.buffer_size[1],
        )
        return layout

    def _draw_song_list(self, scene, layout: SongSelectBaseLayout, *, x_offset: float = 0.0) -> None:
        text = scene.app.text
        density = layout.context.density
        theme = layout.context.theme
        colors = theme.colors
        metrics = self.song_list_metrics(scene, x_offset=x_offset)
        visible_rect: Rect = metrics["visible_rect"]
        if scene._debug_layout:
            scene._debug_rect(visible_rect.x, visible_rect.y, visible_rect.w, visible_rect.h, (0.25, 0.85, 1.0, 0.85))
        header_y = metrics["header_y"]
        if scene._sets:
            draw_header_pair(
                self._commands,
                text,
                theme,
                label="Songs",
                meta=f"{len(scene._sets)} mapsets",
                x=visible_rect.x + 4.0 * density,
                y=header_y,
                label_size=layout.context.tokens.typography.body_l,
                meta_size=layout.context.tokens.typography.body_s,
                label_color=colors.text_secondary,
                meta_color=colors.text_muted,
            )
        else:
            self._commands.text("Songs", visible_rect.x + 4.0 * density, header_y, layout.context.tokens.typography.body_l, color=colors.text_secondary)

        songs_btn_w = max(126.0 * density, text.measure("Open folder", layout.context.tokens.typography.caption)[0] + 28.0 * density)
        songs_btn_h = 26.0 * density
        songs_btn_rect = Rect(
            visible_rect.right - songs_btn_w,
            header_y - 2.0 * density,
            songs_btn_w,
            songs_btn_h,
        )
        songs_btn_hover = songs_btn_rect.contains(scene._mouse_x, scene._mouse_y)
        draw_button(
            self._commands,
            text,
            theme,
            songs_btn_rect,
            label="Open folder",
            size=layout.context.tokens.typography.caption,
            variant="secondary",
            state=InteractionState.HOVER if songs_btn_hover else InteractionState.REST,
            radius=layout.context.tokens.radius_s,
            alpha=0.9,
        )
        scene._songs_open_btn_rect = songs_btn_rect.tuple()

        flattened = scene._build_flattened_list()
        for flat_i, (set_idx, diff_idx) in enumerate(flattened):
            card_rect = self.song_card_rect(scene, metrics, flat_i, set_idx=set_idx, diff_idx=diff_idx)
            if card_rect.bottom < visible_rect.y or card_rect.y > visible_rect.bottom:
                continue

            bset = scene._sets[set_idx]
            info = bset.maps[diff_idx] if bset.maps else None
            is_expanded_row = (set_idx == scene._selected_idx and len(bset.maps) > 1)
            is_selected = (set_idx == scene._selected_idx and diff_idx == scene._selected_diff_idx)
            is_hovered = (
                not scene._dragging_songs
                and card_rect.contains(scene._mouse_x, scene._mouse_y)
            )
            if is_hovered:
                scene._hover_idx = flat_i
            hover_anim = _hover_anim(scene, f"song.card.{flat_i}", is_hovered, speed=0.20)
            card_rect = card_rect.translate(dx=6.0 * density * hover_anim, dy=-2.0 * density * hover_anim)

            base_alpha = 0.78 if is_selected else (0.60 + 0.08 * hover_anim)
            base_fill = colors.surface_selected if is_selected else (
                colors.surface_container_high if is_hovered else colors.surface_container_low
            )
            self._commands.panel(
                card_rect,
                radius=11.0 * density,
                color=(base_fill[0], base_fill[1], base_fill[2], base_alpha),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            self._commands.panel(
                Rect(card_rect.x + 10.0 * density, card_rect.y + 10.0 * density, 3.0 * density, card_rect.h - 20.0 * density),
                radius=1.5 * density,
                color=(
                    colors.focus_ring[0],
                    colors.focus_ring[1],
                    colors.focus_ring[2],
                    (0.90 if is_selected else 0.20 + 0.42 * hover_anim),
                ),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            if scene._debug_layout:
                scene._debug_rect(card_rect.x, card_rect.y, card_rect.w, card_rect.h, (0.88, 0.38, 0.95, 0.7))

            show_badge = len(bset.maps) > 1 and (not is_expanded_row or diff_idx == 0)
            badge_text = str(len(bset.maps)) if show_badge else ""
            badge_w = 0.0
            title_size = max(layout.context.tokens.typography.body_l + 1, layout.context.tokens.typography.title_s - 2)
            artist_size = layout.context.tokens.typography.body_m
            meta_size = layout.context.tokens.typography.body_s
            if show_badge:
                badge_w = max(36.0 * density, text.measure(badge_text, layout.context.tokens.typography.caption)[0] + 24.0 * density)
            text_max_w = max(80.0, card_rect.w - 28.0 * density - (badge_w + 24.0 * density if show_badge else 18.0 * density))
            title = _truncate_text(text, bset.display_title, title_size, text_max_w)
            artist = _truncate_text(text, bset.display_artist, artist_size, text_max_w)
            if is_expanded_row and info:
                meta_text = f"[{info.version}]  mapped by {info.creator}"
            elif bset.maps:
                shown = " | ".join(m.version for m in bset.maps[:3])
                extra = f" +{len(bset.maps) - 3}" if len(bset.maps) > 3 else ""
                meta_text = shown + extra
            else:
                meta_text = "No difficulties found"
            meta_text = _truncate_text(text, meta_text, meta_size, text_max_w)
            content_x = card_rect.x + 22.0 * density
            self._commands.text(title, content_x, card_rect.y + 14.0 * density, title_size, color=colors.text_primary, alpha=0.96)
            self._commands.text(artist, content_x, card_rect.y + 39.0 * density, artist_size, color=colors.text_secondary, alpha=0.88)
            self._commands.text(
                meta_text,
                content_x,
                card_rect.y + 60.0 * density,
                meta_size,
                color=colors.text_secondary if is_selected else colors.text_muted,
                alpha=0.90 if is_selected else 0.78,
            )

            if show_badge:
                badge_rect = Rect(card_rect.right - badge_w - 12.0 * density, card_rect.y + 11.0 * density, badge_w, 24.0 * density)
                badge_anim = _hover_anim(scene, f"song.badge.{flat_i}", is_hovered or is_selected, speed=0.18)
                badge_draw_rect = badge_rect.translate(dy=-1.0 * badge_anim)
                self._commands.panel(
                    badge_draw_rect,
                    radius=badge_draw_rect.h * 0.5,
                    color=(
                        colors.surface_variant[0],
                        colors.surface_variant[1],
                        colors.surface_variant[2],
                        0.70 + 0.08 * badge_anim,
                    ),
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0,
                )
                bw, _ = text.measure(badge_text, layout.context.tokens.typography.caption)
                self._commands.text(
                    badge_text,
                    badge_draw_rect.x + (badge_draw_rect.w - bw) * 0.5,
                    badge_draw_rect.y + (badge_draw_rect.h - layout.context.tokens.typography.caption) * 0.5 - 1.0,
                    layout.context.tokens.typography.caption,
                    color=colors.text_primary if is_selected else colors.text_secondary,
                )

            scene._song_card_rects.append((card_rect.x, card_rect.y, card_rect.w, card_rect.h, flat_i))

    def _draw_divider(self, theme, rect: Rect) -> None:
        draw_divider(self._commands, theme, rect)

    def _draw_toggle_chip(self, text, theme, rect: Rect, label: str, enabled: bool, size: int, *, hovered: bool = False) -> tuple[float, float, float, float]:
        draw_chip(
            self._commands,
            text,
            theme,
            rect,
            label=label,
            size=size,
            selected=enabled,
            alpha=0.96 if hovered or enabled else 0.82,
        )
        return rect.tuple()

    def _draw_info_panel(self, scene, layout: SongSelectBaseLayout, *, x_offset: float = 0.0) -> None:
        text = scene.app.text
        density = layout.context.density
        theme = layout.context.theme
        colors = theme.colors
        panel_rect = layout.info_panel_rect.translate(dx=x_offset)
        if scene._debug_layout:
            scene._debug_rect(panel_rect.x, panel_rect.y, panel_rect.w, panel_rect.h, (0.25, 1.0, 0.35, 0.85))
        draw_surface(
            self._commands,
            theme,
            panel_rect,
            role="panel",
            radius=layout.context.tokens.radius_l,
            alpha=0.74,
            border_width=0.0,
        )
        inner = panel_rect.inset(Insets.all(20.0 * density))
        info = scene._selected_info()
        bset = scene._sets[scene._selected_idx] if scene._sets else None
        title_size = layout.context.tokens.typography.title_m
        artist_size = layout.context.tokens.typography.body_m
        caption_size = layout.context.tokens.typography.caption
        if info is None:
            if not scene.app.scanner.scan_complete:
                self._commands.text("Loading maps...", inner.x, inner.y + 18.0 * density, title_size, color=colors.text_primary)
            else:
                self._commands.text("No maps found", inner.x, inner.y + 18.0 * density, title_size, color=colors.text_primary)
                self._commands.text(f"Place mapset folders in {MAPS_DIR.name}/", inner.x, inner.y + 52.0 * density, artist_size, color=colors.text_secondary)
            return

        y = inner.y
        title_raw = info.title_unicode or info.title
        artist_raw = info.artist_unicode or info.artist
        hero_h = min(168.0 * density, panel_rect.h * 0.34)
        hero_rect = Rect(inner.x, inner.y, inner.w, hero_h)
        self._commands.panel(
            hero_rect,
            radius=12.0 * density,
            color=(colors.surface_container_low[0], colors.surface_container_low[1], colors.surface_container_low[2], 0.40 + (0.05 if scene._preview_playing else 0.0)),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        top_band_h = 34.0 * density
        self._commands.panel(
            Rect(hero_rect.x, hero_rect.y, hero_rect.w, top_band_h),
            radius=12.0 * density,
            color=(colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.32),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )

        title_plate_rect = Rect(
            hero_rect.x + 14.0 * density,
            hero_rect.y + 0.0 * density,
            hero_rect.w - 28.0 * density,
            max(54.0 * density, hero_rect.h - 84.0 * density),
        )
        self._commands.panel(
            title_plate_rect,
            radius=10.0 * density,
            color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], 0.26),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )

        hero_inner_x = title_plate_rect.x + 14.0 * density
        hero_inner_w = title_plate_rect.w - 28.0 * density
        diff_chip_h = 24.0 * density
        diff_text = _truncate_text(text, f"[{info.version}]", caption_size, hero_inner_w * 0.46)
        diff_chip_w = max(88.0 * density, text.measure(diff_text, caption_size)[0] + 18.0 * density)
        diff_chip_rect = Rect(hero_rect.right - diff_chip_w - 14.0 * density, hero_rect.y + 6.0 * density, diff_chip_w, diff_chip_h)
        diff_hover = diff_chip_rect.contains(scene._mouse_x, scene._mouse_y)
        diff_anim = _hover_anim(scene, "info.diff_chip", diff_hover, speed=0.18)
        diff_draw_rect = diff_chip_rect.translate(dy=-1.5 * diff_anim)
        self._commands.panel(
            diff_draw_rect,
            radius=diff_chip_h * 0.5,
            color=(colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.66 + 0.10 * diff_anim),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        self._commands.text(
            diff_text,
            diff_draw_rect.x + 9.0 * density,
            diff_draw_rect.y + 5.0 * density,
            caption_size,
            color=colors.text_primary,
        )

        title_max_w = max(120.0, title_plate_rect.w - 28.0 * density)
        title = _truncate_text(text, title_raw, title_size, title_max_w)
        self._commands.text(title, hero_inner_x, title_plate_rect.y + 10.0 * density, title_size, color=colors.text_primary)

        artist = _truncate_text(text, artist_raw, artist_size, hero_inner_w)
        self._commands.text(artist, hero_inner_x, title_plate_rect.y + 42.0 * density, artist_size, color=colors.text_secondary)
        mapper = _truncate_text(text, f"mapped by {info.creator}", caption_size, hero_inner_w)
        self._commands.text(mapper, hero_inner_x, title_plate_rect.y + 63.0 * density, caption_size, color=colors.text_muted, alpha=0.76)

        chip_y = hero_rect.y + hero_rect.h - diff_chip_h - 14.0 * density
        chip_gap = 8.0 * density
        stat_specs = [
            ("CS", f"{info.cs:.1f}"),
            ("AR", f"{info.ar:.1f}"),
            ("OD", f"{info.od:.1f}"),
            ("HP", f"{info.hp:.1f}"),
        ]
        chip_x = hero_inner_x
        for label, value in stat_specs:
            label_w, _ = text.measure(label, caption_size)
            value_w, _ = text.measure(value, layout.context.tokens.typography.body_s)
            chip_w = max(54.0 * density, max(label_w, value_w) + 18.0 * density)
            chip_rect = Rect(chip_x - 5.0, chip_y, chip_w, diff_chip_h)
            chip_hover = chip_rect.contains(scene._mouse_x, scene._mouse_y)
            chip_anim = _hover_anim(scene, f"info.stat.{label}", chip_hover, speed=0.18)
            chip_draw_rect = chip_rect.translate(dy=-1.0 * chip_anim)
            self._commands.panel(
                chip_draw_rect,
                radius=chip_draw_rect.h * 0.5,
                color=(colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.52 + 0.08 * chip_anim),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            self._commands.text(label, chip_draw_rect.x + 8.0 * density, chip_draw_rect.y + 5.0 * density, caption_size, color=colors.text_muted, alpha=0.80 + 0.08 * chip_anim)
            self._commands.text(value, chip_draw_rect.x + chip_draw_rect.w - value_w - 4.0 * density, chip_draw_rect.y + 4.0 * density, layout.context.tokens.typography.body_s, color=colors.text_primary, alpha=0.92 + 0.04 * chip_anim)
            chip_x += chip_w + chip_gap

        y = hero_rect.bottom + 18.0 * density

        all_replays = scene._replays_for_set(bset) if bset else []
        replays = scene._visible_replays_for_set(bset) if bset else []
        beatmap_id = scene._selected_online_beatmap_id()
        online_state = scene.app.social_client.online_replays(beatmap_id) if beatmap_id is not None else None
        online_items = [] if online_state is None else online_state.items
        if scene._replay_source_tab == "online":
            if online_state is None:
                replay_meta = "No beatmap selected"
            elif online_state.loading and not online_items:
                replay_meta = "Loading..."
            elif online_state.error and not online_items:
                replay_meta = "Unavailable"
            elif not online_items and online_state.loaded_at > 0.0:
                replay_meta = "No replays found"
            else:
                replay_meta = f"{len(online_items)} online"
        else:
            replay_meta = f"{len(replays)} shown" if len(replays) == len(all_replays) else f"{len(replays)} shown / {len(all_replays)} saved"
        draw_header_pair(
            self._commands,
            text,
            theme,
            label="Replays",
            meta=replay_meta,
            x=inner.x,
            y=y,
            label_size=layout.context.tokens.typography.body_l,
            meta_size=layout.context.tokens.typography.body_s,
            label_color=colors.text_secondary,
            meta_color=colors.text_muted,
        )
        toggle_h = 24.0 * density
        toggle_y = y - 1.0 * density
        tabs_strip_rect = Rect(inner.right - 162.0 * density, toggle_y - 5.0 * density, 162.0 * density, 34.0 * density)
        draw_surface(
            self._commands,
            theme,
            tabs_strip_rect,
            role="toolbar",
            radius=10.0 * density,
            alpha=0.50,
            border_width=0.0,
        )
        tab_y = tabs_strip_rect.y + 5.0 * density
        tab_w = (tabs_strip_rect.w - 22.0 * density) * 0.5
        local_tab_rect = Rect(tabs_strip_rect.x + 8.0 * density, tab_y, tab_w, toggle_h)
        online_tab_rect = Rect(local_tab_rect.right + 6.0 * density, tab_y, tab_w, toggle_h)
        draw_tab(
            self._commands,
            text,
            theme,
            local_tab_rect,
            label="Local",
            size=layout.context.tokens.typography.caption,
            selected=scene._replay_source_tab == "local",
            hovered=local_tab_rect.contains(scene._mouse_x, scene._mouse_y),
            alpha=0.92,
        )
        draw_tab(
            self._commands,
            text,
            theme,
            online_tab_rect,
            label="Online",
            size=layout.context.tokens.typography.caption,
            selected=scene._replay_source_tab == "online",
            hovered=online_tab_rect.contains(scene._mouse_x, scene._mouse_y),
            alpha=0.92,
        )
        scene._replay_local_tab_rect = local_tab_rect.tuple()
        scene._replay_online_tab_rect = online_tab_rect.tuple()
        scene._multi_toggle_rect = None
        scene._danser_toggle_rect = None
        if scene._replay_source_tab == "local":
            toggle_w = 82.0 * density
            toggle_x = local_tab_rect.x - toggle_w - 10.0 * density
            multi_rect = Rect(toggle_x, toggle_y, toggle_w, toggle_h)
            scene._multi_toggle_rect = self._draw_toggle_chip(
                text,
                theme,
                multi_rect,
                "Multi" if scene._multi_replay_enabled else "Single",
                scene._multi_replay_enabled,
                layout.context.tokens.typography.caption,
                hovered=multi_rect.contains(scene._mouse_x, scene._mouse_y),
            )
            if scene._multi_replay_enabled:
                toggle_x -= toggle_w + 8.0 * density
                danser_rect = Rect(toggle_x, toggle_y, toggle_w, toggle_h)
                scene._danser_toggle_rect = self._draw_toggle_chip(
                    text,
                    theme,
                    danser_rect,
                    "Danser",
                    scene._danser_replay_enabled,
                    layout.context.tokens.typography.caption,
                    hovered=danser_rect.contains(scene._mouse_x, scene._mouse_y),
                )
        y += 34.0 * density

        footer_h = 30.0 * density
        section_y = y
        button_y = panel_rect.bottom - 40.0 * density
        section_h = max(60.0 * density, button_y - section_y - 6.0 * density)
        section_rect = Rect(inner.x, section_y, inner.w, section_h)
        scene._replay_section_rect = section_rect.tuple()
        scene._clamp_replay_scroll()
        if scene._debug_layout:
            scene._debug_rect(section_rect.x, section_rect.y, section_rect.w, section_rect.h, (1.0, 0.65, 0.25, 0.85))

        draw_surface(
            self._commands,
            theme,
            section_rect,
            role="section",
            radius=10.0 * density,
            alpha=0.46,
            border_width=0.0,
        )
        self._commands.clip_push(section_rect)
        scene._replay_action_rects = []
        visible_top = section_rect.y
        visible_bottom = section_rect.bottom
        if scene._replay_source_tab == "online" and online_state is not None:
            row_h = 40.0 * density
            row_gap = 6.0 * density
            ry = section_rect.y + 8.0 * density - scene._replay_scroll_current
            if online_state.loading and not online_items:
                self._commands.text("Loading online replays...", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
            elif online_state.error and not online_items:
                self._commands.text("Replay server unavailable", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
                self._commands.text("Use refresh to try again.", section_rect.x + 12.0 * density, section_rect.y + 34.0 * density, layout.context.tokens.typography.caption, color=colors.text_muted, alpha=0.65)
            elif not online_items:
                self._commands.text("No replays found", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
                self._commands.text("Upload one from the local tab or refresh later.", section_rect.x + 12.0 * density, section_rect.y + 34.0 * density, layout.context.tokens.typography.caption, color=colors.text_muted, alpha=0.65)
            for item in online_items:
                scene._sync_online_replay_local_state(item)
                row_rect = Rect(section_rect.x + 4.0 * density, ry, section_rect.w - 8.0 * density, row_h)
                if row_rect.bottom < visible_top:
                    ry += row_h + row_gap
                    continue
                if row_rect.y > visible_bottom:
                    break
                row_hover = row_rect.contains(scene._mouse_x, scene._mouse_y)
                row_anim = _hover_anim(scene, f"info.online_replay.{item.replay_id}", row_hover, speed=0.22)
                row_draw_rect = row_rect.translate(dx=4.0 * density * row_anim)
                selected = scene._selected_online_replay_id == item.replay_id
                is_downloaded = item.is_downloaded
                alpha_scale = 0.98 if is_downloaded else 0.64
                self._commands.panel(
                    row_draw_rect,
                    radius=8.0 * density,
                    color=(
                        colors.surface_selected[0] if selected else (colors.surface_container[0] if is_downloaded else colors.surface_variant[0]),
                        colors.surface_selected[1] if selected else (colors.surface_container[1] if is_downloaded else colors.surface_variant[1]),
                        colors.surface_selected[2] if selected else (colors.surface_container[2] if is_downloaded else colors.surface_variant[2]),
                        (0.74 if selected else (0.48 if is_downloaded else 0.28)) * alpha_scale + 0.10 * row_anim,
                    ),
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0,
                )
                if is_downloaded:
                    self._commands.panel(
                        Rect(row_draw_rect.x + 8.0 * density, row_draw_rect.y + 7.0 * density, 3.0 * density, row_draw_rect.h - 14.0 * density),
                        radius=1.5 * density,
                        color=(colors.success[0], colors.success[1], colors.success[2], 0.88 if not selected else 0.64),
                        border_color=(0.0, 0.0, 0.0, 0.0),
                        border_width=0.0,
                    )
                if item.is_downloading:
                    progress_rect = Rect(row_draw_rect.x + 8.0 * density, row_draw_rect.bottom - 9.0 * density, row_draw_rect.w - 16.0 * density, 5.0 * density)
                    draw_linear_progress(
                        self._commands,
                        theme,
                        progress_rect,
                        value=item.download_progress,
                        fill_color=colors.text_secondary,
                        alpha=0.88,
                        track_alpha=0.22,
                    )
                label_text = f"{item.player_name or Path(item.original_filename).stem}  {scene.mod_string(item.mods) or '+NM'}"
                label = _truncate_text(text, label_text, layout.context.tokens.typography.body_s, row_draw_rect.w - 92.0 * density)
                self._commands.text(
                    label,
                    row_draw_rect.x + (18.0 * density if not is_downloaded else 20.0 * density),
                    row_draw_rect.y + 10.0 * density,
                    layout.context.tokens.typography.body_s,
                    color=colors.text_primary if is_downloaded else colors.text_secondary,
                    alpha=0.92 + 0.05 * row_anim,
                )
                views_text = str(item.views)
                vw, _ = text.measure(views_text, layout.context.tokens.typography.caption)
                metric_w = 18.0 * density + vw
                self._draw_eye_metric(
                    theme,
                    Rect(row_draw_rect.right - 28.0 * density - metric_w, row_draw_rect.y + 10.0 * density, metric_w, 10.0 * density),
                    value=views_text,
                    size=layout.context.tokens.typography.caption,
                    alpha=0.84 if not is_downloaded else 0.72,
                )
                dots_rect = Rect(row_draw_rect.right - 28.0 * density, row_draw_rect.y + 8.0 * density, 20.0 * density, 20.0 * density)
                self._draw_kebab_button(theme, dots_rect, hovered=dots_rect.contains(scene._mouse_x, scene._mouse_y), alpha=0.92)
                scene._replay_rects.append((row_draw_rect.x, row_draw_rect.y, row_draw_rect.w, row_draw_rect.h, f"online:{item.replay_id}"))
                scene._replay_action_rects.append((dots_rect.x, dots_rect.y, dots_rect.w, dots_rect.h, f"online:{item.replay_id}"))
                ry += row_h + row_gap
            scene._replay_total_h = 8.0 * density + len(online_items) * row_h + max(0, len(online_items) - 1) * row_gap + 8.0 * density
        elif replays:
            replay_dir = replay_dir_for_set(bset.directory)
            selected_replay_entities = scene.selected_replay_entity_keys()
            row_h = 36.0 * density
            row_gap = 4.0 * density
            ry = section_rect.y + 8.0 * density - scene._replay_scroll_current
            for rp in replays:
                full_path = str(replay_dir / rp)
                entity_info = scene._replay_entity_info(full_path)
                entity_key = entity_info[0] if entity_info is not None else full_path
                row_rect = Rect(section_rect.x + 4.0 * density, ry, section_rect.w - 8.0 * density, row_h)
                if row_rect.bottom < visible_top:
                    ry += row_h + row_gap
                    continue
                if row_rect.y > visible_bottom:
                    break
                is_sel = entity_key in selected_replay_entities
                row_hover = row_rect.contains(scene._mouse_x, scene._mouse_y)
                row_anim = _hover_anim(scene, f"info.replay.{full_path}", row_hover, speed=0.22)
                row_draw_rect = row_rect.translate(dx=4.0 * density * row_anim)
                self._commands.panel(
                    row_draw_rect,
                    radius=8.0 * density,
                    color=(
                        colors.surface_variant[0],
                        colors.surface_variant[1],
                        colors.surface_variant[2],
                        (0.74 if is_sel else 0.42) + 0.10 * row_anim,
                    ),
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0,
                )
                if is_sel:
                    self._commands.panel(
                        Rect(row_draw_rect.x + 8.0 * density, row_draw_rect.y + 7.0 * density, 3.0 * density, row_draw_rect.h - 14.0 * density),
                        radius=1.5 * density,
                        color=(colors.focus_ring[0], colors.focus_ring[1], colors.focus_ring[2], 0.90),
                        border_color=(0.0, 0.0, 0.0, 0.0),
                        border_width=0.0,
                    )
                summary = scene._replay_summary_cache.get(full_path)
                label_text = rp
                if summary is not None and summary.player_name:
                    label_text = f"{summary.player_name}  {scene.mod_string(summary.mods) or '+NM'}"
                elif full_path in scene._replay_summary_loading:
                    label_text = f"{Path(rp).stem}  Loading..."
                label = _truncate_text(text, label_text, layout.context.tokens.typography.body_s, row_draw_rect.w - 76.0 * density)
                self._commands.text(label, row_draw_rect.x + 18.0 * density, row_draw_rect.y + 8.0 * density, layout.context.tokens.typography.body_s, color=colors.text_primary if is_sel else colors.text_secondary, alpha=0.92 + 0.06 * row_anim)
                dots_rect = Rect(row_draw_rect.right - 28.0 * density, row_draw_rect.y + 7.0 * density, 20.0 * density, 20.0 * density)
                self._draw_kebab_button(theme, dots_rect, hovered=dots_rect.contains(scene._mouse_x, scene._mouse_y), alpha=0.88)
                scene._replay_rects.append((row_draw_rect.x, row_draw_rect.y, row_draw_rect.w, row_draw_rect.h, full_path))
                scene._replay_action_rects.append((dots_rect.x, dots_rect.y, dots_rect.w, dots_rect.h, full_path))
                ry += row_h + row_gap
            scene._replay_total_h = 8.0 * density + len(replays) * row_h + max(0, len(replays) - 1) * row_gap + 8.0 * density
        else:
            self._commands.text("No local replays yet", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
            self._commands.text("Drop .osr files into the mapset's replay folder.", section_rect.x + 12.0 * density, section_rect.y + 34.0 * density, layout.context.tokens.typography.caption, color=colors.text_muted, alpha=0.65)
            scene._replay_total_h = 0.0
        self._commands.clip_pop()

        if scene._replay_context_menu_rect is not None and scene._replay_context_menu_options:
            menu_x, menu_y, menu_w, menu_h = scene._replay_context_menu_rect
            menu_progress = ease_out_back(scene._replay_context_menu_anim.value)
            menu_rect = Rect(menu_x, menu_y + (1.0 - menu_progress) * 8.0 * density, menu_w, menu_h)
            option_rects = [
                Rect(ox, oy + (1.0 - menu_progress) * 8.0 * density, ow, oh)
                for ox, oy, ow, oh, _ in scene._replay_context_menu_options
            ]
            labels = [
                {
                    "download": "Download",
                    "delete": "Delete from disk",
                    "upload": "Upload",
                }.get(action, action.capitalize())
                for *_rest, action in scene._replay_context_menu_options
            ]
            hovered_index = next((idx for idx, rect in enumerate(option_rects) if rect.contains(scene._mouse_x, scene._mouse_y)), -1)
            draw_dropdown_menu(
                self._commands,
                text,
                theme,
                menu_rect,
                option_rects=option_rects,
                labels=labels,
                selected_index=-1,
                hovered_index=hovered_index,
                size=layout.context.tokens.typography.caption,
                alpha=0.98 * menu_progress,
                progress=scene._replay_context_menu_anim.value,
                edge_padding=4.0 * density,
            )

        open_btn_rect = Rect(inner.x, button_y - 2.0 * density, inner.w, footer_h + 6.0 * density)
        open_hover = open_btn_rect.contains(scene._mouse_x, scene._mouse_y)
        draw_button(
            self._commands,
            text,
            theme,
            open_btn_rect,
            label="Open replays folder" if scene._replay_source_tab == "local" else "Refresh online replays",
            size=layout.context.tokens.typography.caption,
            variant="secondary",
            state=InteractionState.HOVER if open_hover else InteractionState.REST,
            alpha=0.85,
            radius=layout.context.tokens.radius_s,
        )
        scene._open_btn_rect = open_btn_rect.tuple()

    def _draw_bottom_bar(self, scene, layout: SongSelectBaseLayout, *, y_offset: float = 0.0) -> None:
        text = scene.app.text
        density = layout.context.density
        theme = layout.context.theme
        colors = theme.colors
        outer_bar_rect = layout.bottom_bar_rect.translate(dy=y_offset)
        bar_rect = Rect(
            layout.main_rect.x,
            outer_bar_rect.y + 5.0 * density,
            layout.main_rect.w,
            max(42.0 * density, outer_bar_rect.h - 10.0 * density),
        )
        if scene._debug_layout:
            scene._debug_rect(bar_rect.x, bar_rect.y, bar_rect.w, bar_rect.h, (1.0, 0.25, 0.25, 0.85))
        draw_surface(
            self._commands,
            theme,
            bar_rect,
            role="toolbar",
            radius=layout.context.tokens.radius_l,
            alpha=0.84,
            border_width=0.0,
        )
        inner_pad = 14.0 * density
        play_button_w = 116.0 * density
        mods_button_w = 92.0 * density
        play_h = max(30.0 * density, bar_rect.h - 12.0 * density)
        icon_button_size = play_h * 0.75
        chat_button_w = icon_button_size
        settings_button_w = icon_button_size
        play_y = bar_rect.y + (bar_rect.h - play_h) * 0.5
        icon_button_y = bar_rect.y + (bar_rect.h - icon_button_size) * 0.5
        play_x = bar_rect.right - play_button_w - inner_pad
        settings_x = play_x - settings_button_w - 10.0 * density
        chat_x = settings_x - chat_button_w - 8.0 * density
        mods_rect = Rect(bar_rect.x + inner_pad, play_y, mods_button_w, play_h)
        scene._mods_trigger_rect = mods_rect.tuple()

        mods_hover = mods_rect.contains(scene._mouse_x, scene._mouse_y)
        mods_state = InteractionState.HOVER if mods_hover else InteractionState.REST
        draw_button(
            self._commands,
            text,
            theme,
            mods_rect,
            label="MODS",
            size=layout.context.tokens.typography.body_m,
            variant="secondary",
            state=mods_state,
            radius=layout.context.tokens.radius_m,
            alpha=0.90 if scene._mods_palette_anim > 0.05 else 0.82,
        )

        active_count = sum(1 for _, flag in scene.mod_flag_map.items() if flag and scene._active_mods & flag)
        summary_x = mods_rect.right + 16.0 * density
        summary_w = max(120.0, chat_x - summary_x - 14.0 * density)
        selected_mods = scene.mod_string(scene._active_mods) or "No mod"
        if scene._multi_replay_enabled:
            selected_mods = f"{scene._selected_replay_count()} replays  {selected_mods}"
        elif active_count > 0:
            selected_mods = f"{active_count} active  {selected_mods}"
        summary = _truncate_text(text, selected_mods, layout.context.tokens.typography.body_m, summary_w)
        summary_h = layout.context.tokens.typography.body_m
        self._commands.text(
            summary,
            summary_x,
            play_y + (play_h - summary_h) * 0.5 - 4.0,
            layout.context.tokens.typography.body_m,
            color=colors.text_secondary if active_count == 0 else colors.text_primary,
            alpha=0.92 if active_count > 0 else 0.78,
        )

        play_hover = Rect(play_x, play_y, play_button_w, play_h).contains(scene._mouse_x, scene._mouse_y)
        play_rect = Rect(play_x, play_y, play_button_w, play_h)
        chat_rect = Rect(chat_x, icon_button_y, chat_button_w, icon_button_size)
        settings_rect = Rect(settings_x, icon_button_y, settings_button_w, icon_button_size)
        chat_hover = chat_rect.contains(scene._mouse_x, scene._mouse_y)
        settings_hover = settings_rect.contains(scene._mouse_x, scene._mouse_y)
        self._draw_chat_button(text, theme, chat_rect, hovered=chat_hover, alpha=0.90)
        self._draw_settings_button(text, theme, settings_rect, hovered=settings_hover, alpha=0.90)
        draw_button(
            self._commands,
            text,
            theme,
            play_rect,
            label="PLAY",
            size=layout.context.tokens.typography.title_s,
            variant="primary",
            state=InteractionState.HOVER if play_hover else InteractionState.REST,
            radius=layout.context.tokens.radius_m,
            alpha=0.92,
        )
        scene._play_btn_rect = play_rect.tuple()
        scene._chat_btn_rect = chat_rect.tuple()
        scene._settings_btn_rect = settings_rect.tuple()
        scene.app.set_settings_button(settings_rect.tuple(), visible=True)
        self._draw_mods_palette(scene, theme, bar_rect, mods_rect, density)

    def _draw_mods_palette(self, scene, theme, bar_rect: Rect, anchor_rect: Rect, density: float) -> None:
        text = scene.app.text
        colors = theme.colors
        progress = ease_out_cubic(scene._mods_palette_anim)
        if progress <= 0.01:
            return

        columns = 5
        tile_size = 42.0 * density
        tile_gap = 6.0 * density
        rows = max(1, (len(scene._mod_labels) + columns - 1) // columns)
        palette_w = columns * tile_size + (columns - 1) * tile_gap + 24.0 * density
        palette_h = rows * tile_size + (rows - 1) * tile_gap + 34.0 * density
        palette_x = max(bar_rect.x, min(anchor_rect.x, bar_rect.right - palette_w))
        palette_y = bar_rect.y - palette_h - 10.0 * density - (1.0 - progress) * 12.0 * density
        palette_rect = Rect(palette_x, palette_y, palette_w, palette_h)
        scene._mods_surface_rect = palette_rect.tuple()

        draw_surface(
            self._commands,
            theme,
            palette_rect,
            role="toolbar",
            radius=14.0 * density,
            alpha=0.92 * progress,
            border_width=0.0,
        )
        self._commands.text("Mods", palette_rect.x + 12.0 * density, palette_rect.y + 10.0 * density, 11, color=colors.text_muted, alpha=0.74 * progress)

        start_x = palette_rect.x + 12.0 * density
        start_y = palette_rect.y + 24.0 * density
        for idx, label in enumerate(scene._mod_labels):
            flag = scene.mod_flag_map.get(label, 0)
            short = scene.mod_short.get(flag, label[:2].upper())
            col = idx % columns
            row = idx // columns
            stagger = min(1.0, max(0.0, progress * 1.18 - idx * 0.035))
            if stagger <= 0.01:
                continue

            base_x = start_x + col * (tile_size + tile_gap)
            base_y = start_y + row * (tile_size + tile_gap)
            rise = (1.0 - stagger) * 10.0 * density
            tile_rect = Rect(base_x, base_y + rise, tile_size, tile_size)
            is_active = bool(scene._active_mods & flag)
            disabled = flag == scene.fl_flag
            hovered = tile_rect.contains(scene._mouse_x, scene._mouse_y)
            if is_active:
                fill = (colors.primary_container[0], colors.primary_container[1], colors.primary_container[2], 0.92 * stagger)
                fg = colors.text_primary
            elif hovered:
                fill = (colors.surface_variant[0], colors.surface_variant[1], colors.surface_variant[2], 0.86 * stagger)
                fg = colors.text_primary
            else:
                fill = (colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.62 * stagger)
                fg = colors.text_secondary if not disabled else colors.text_muted

            self._commands.panel(
                tile_rect,
                radius=10.0 * density,
                color=fill,
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            short_w, _ = text.measure(short, 13)
            self._commands.text(
                short,
                tile_rect.x + (tile_rect.w - short_w) * 0.5,
                tile_rect.y + 8.0 * density,
                13,
                color=fg,
                alpha=0.98 * stagger,
            )
            label_text = _truncate_text(text, label, 10, tile_rect.w - 8.0 * density)
            label_w, _ = text.measure(label_text, 10)
            self._commands.text(
                label_text,
                tile_rect.x + (tile_rect.w - label_w) * 0.5,
                tile_rect.y + 23.0 * density,
                10,
                color=colors.text_muted if not is_active else colors.text_primary,
                alpha=0.70 * stagger,
            )
            if stagger > 0.72:
                scene._mod_rects.append((tile_rect.x, tile_rect.y, tile_rect.w, tile_rect.h, flag))

