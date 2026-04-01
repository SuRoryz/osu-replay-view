"""Responsive song select menu view built on the shared layout system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

import moderngl
import numpy as np
from PIL import Image

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


def _fit_text(text, value: str, size: int, max_width: float) -> tuple[str, float]:
    if max_width <= 0:
        return ("", 0.0)
    if hasattr(text, "truncate_with_width"):
        return text.truncate_with_width(value, size, max_width)
    measured_width, _ = text.measure(value, size)
    if measured_width <= max_width:
        return (value, measured_width)
    fitted = text.truncate(value, size, max_width)
    fitted_width, _ = text.measure(fitted, size)
    return (fitted, fitted_width)


def _truncate_text(text, value: str, size: int, max_width: float) -> str:
    return _fit_text(text, value, size, max_width)[0]


def _prefer_renderable_text(primary: str, secondary: str, *, empty_fallback: str) -> str:
    primary_value = str(primary or "").strip()
    secondary_value = str(secondary or "").strip()
    if secondary_value:
        return secondary_value
    if primary_value:
        return primary_value
    return empty_fallback


def _hover_anim(scene, key: object, hovered: bool, *, speed: float = 0.22) -> float:
    if not hasattr(scene, "_ui_hover_anim"):
        scene._ui_hover_anim = {}
    current = float(scene._ui_hover_anim.get(key, 0.0))
    target = 1.0 if hovered else 0.0
    current += (target - current) * speed
    if current < 0.001 and not hovered:
        current = 0.0
    scene._ui_hover_anim[key] = current
    return current


def _edge_fade_alpha(rect: Rect, bounds: Rect, fade_h: float) -> float:
    if fade_h <= 1e-6:
        return 1.0
    center_y = rect.center_y
    top_factor = max(0.0, min(1.0, (center_y - bounds.y) / fade_h))
    bottom_factor = max(0.0, min(1.0, (bounds.bottom - center_y) / fade_h))
    factor = min(top_factor, bottom_factor)
    return factor * factor * (3.0 - 2.0 * factor)


_CARD_COVER_VERT = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
uniform mat4 projection;
uniform vec2 u_rect_pos;
uniform vec2 u_rect_size;
out vec2 v_uv;
out vec2 v_local_pos;
void main() {
    vec2 pos = u_rect_pos + in_pos * u_rect_size;
    gl_Position = projection * vec4(pos, 0.0, 1.0);
    v_uv = in_uv;
    v_local_pos = in_pos * u_rect_size;
}
"""

_CARD_COVER_FRAG = """
#version 330 core
in vec2 v_uv;
in vec2 v_local_pos;
uniform sampler2D tex;
uniform vec2 u_rect_size;
uniform float u_radius;
uniform float u_alpha;
uniform float u_dim;
uniform vec3 u_overlay_color;
uniform float u_overlay_alpha;
out vec4 frag_color;

float rounded_rect_mask(vec2 p, vec2 size, float radius) {
    vec2 center = size * 0.5;
    vec2 q = abs(p - center) - (size * 0.5 - vec2(radius));
    float dist = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
    return 1.0 - smoothstep(0.0, 1.5, dist);
}

void main() {
    vec2 tex_size = vec2(textureSize(tex, 0));
    float rect_aspect = u_rect_size.x / max(u_rect_size.y, 1.0);
    float tex_aspect = tex_size.x / max(tex_size.y, 1.0);
    vec2 sample_uv = v_uv;
    if (rect_aspect > tex_aspect) {
        float scale = tex_aspect / rect_aspect;
        sample_uv.y = (v_uv.y - 0.5) * scale + 0.5;
    } else {
        float scale = rect_aspect / tex_aspect;
        sample_uv.x = (v_uv.x - 0.5) * scale + 0.5;
    }
    sample_uv = clamp(sample_uv, vec2(0.001), vec2(0.999));
    vec3 image = texture(tex, sample_uv).rgb * u_dim;
    float bottom_mix = smoothstep(0.46, 1.0, v_uv.y) * u_overlay_alpha;
    vec3 color = mix(image, u_overlay_color, bottom_mix);
    float mask = rounded_rect_mask(v_local_pos, u_rect_size, u_radius);
    frag_color = vec4(color, u_alpha * mask);
}
"""


@dataclass(slots=True)
class _LocalCoverEntry:
    texture: moderngl.Texture | None = None
    size: tuple[int, int] = (1, 1)
    loading: bool = False
    failed: bool = False
    fade: float = 0.0


@dataclass(slots=True)
class _SongCardTextLayout:
    title: str
    title_w: float
    artist: str
    artist_w: float
    diff_label: str
    diff_w: float


@dataclass(slots=True)
class _ReplayRowTextLayout:
    primary: str
    secondary: str = ""


class _SongCardCoverRenderer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._prog = ctx.program(vertex_shader=_CARD_COVER_VERT, fragment_shader=_CARD_COVER_FRAG)
        self._prog["tex"].value = 0
        verts = np.array([
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
        ], dtype="f4")
        idx = np.array([0, 1, 2, 0, 2, 3], dtype="i4")
        self._vbo = ctx.buffer(verts.tobytes())
        self._ibo = ctx.buffer(idx.tobytes())
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, "2f 2f", "in_pos", "in_uv")],
            index_buffer=self._ibo,
        )
        self._projection = np.eye(4, dtype="f4")
        self._projection_bytes = self._projection.tobytes()
        self._entries: dict[str, _LocalCoverEntry] = {}
        self._pending_results: list[tuple[str, tuple[int, int], bytes]] = []
        self._lock = threading.Lock()

    def update(self, dt: float) -> None:
        fade_step = 0.0 if dt <= 0.0 else min(1.0, dt / 0.24)
        if fade_step <= 0.0:
            return
        for entry in self._entries.values():
            if entry.texture is not None and entry.fade < 1.0:
                entry.fade = min(1.0, entry.fade + fade_step)

    def set_projection(self, width: int, height: int) -> None:
        p = np.eye(4, dtype="f4")
        p[0, 0] = 2.0 / max(1.0, float(width))
        p[1, 1] = -2.0 / max(1.0, float(height))
        p[3, 0] = -1.0
        p[3, 1] = 1.0
        self._projection = p
        self._projection_bytes = p.tobytes()
        self._prog["projection"].write(self._projection_bytes)

    def request(self, image_path: str | None) -> None:
        value = str(image_path or "").strip()
        if not value:
            return
        entry = self._entries.get(value)
        if entry is not None and (entry.texture is not None or entry.loading or entry.failed):
            return
        if not Path(value).is_file():
            return
        if entry is None:
            entry = _LocalCoverEntry()
            self._entries[value] = entry
        entry.loading = True
        threading.Thread(target=self._load_worker, args=(value,), daemon=True, name="song-card-cover-load").start()

    def _load_worker(self, image_path: str) -> None:
        try:
            image = Image.open(image_path).convert("RGB")
            max_side = max(image.size)
            if max_side > 1024:
                scale = 1024.0 / float(max_side)
                image = image.resize(
                    (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale))),
                    Image.LANCZOS,
                )
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            payload = image.tobytes()
            with self._lock:
                self._pending_results.append((image_path, image.size, payload))
        except Exception:
            entry = self._entries.get(image_path)
            if entry is not None:
                entry.loading = False
                entry.failed = True

    def _apply_pending(self) -> None:
        with self._lock:
            results = list(self._pending_results)
            self._pending_results.clear()
        for image_path, size, payload in results:
            entry = self._entries.setdefault(image_path, _LocalCoverEntry())
            if entry.texture is not None:
                entry.texture.release()
            texture = self.ctx.texture(size, 3, payload)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False
            entry.texture = texture
            entry.size = size
            entry.loading = False
            entry.failed = False
            entry.fade = 0.0

    def process_pending(self) -> None:
        self._apply_pending()

    def draw(
        self,
        image_path: str | None,
        rect: Rect,
        *,
        radius: float,
        alpha: float,
        overlay_color: tuple[float, float, float],
        overlay_alpha: float,
        dim: float,
    ) -> bool:
        value = str(image_path or "").strip()
        if not value:
            return False
        self.request(value)
        entry = self._entries.get(value)
        if entry is None or entry.texture is None:
            return False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        entry.texture.use(location=0)
        self._prog["u_rect_pos"].value = (rect.x, rect.y)
        self._prog["u_rect_size"].value = (rect.w, rect.h)
        self._prog["u_radius"].value = max(0.0, radius)
        self._prog["u_alpha"].value = max(0.0, alpha) * max(0.0, min(1.0, entry.fade))
        self._prog["u_dim"].value = max(0.0, dim)
        self._prog["u_overlay_color"].value = overlay_color
        self._prog["u_overlay_alpha"].value = max(0.0, overlay_alpha)
        self._vao.render(moderngl.TRIANGLES)
        return True


class SongSelectMenuView:
    def __init__(self) -> None:
        self._commands = RenderCommandBuffer()
        self._cover_renderer: _SongCardCoverRenderer | None = None
        self._sorted_diff_cache: dict[tuple[str, tuple[str, ...]], tuple] = {}
        self._card_text_cache: dict[tuple[str, str, str, int, int, int, int, int, int], _SongCardTextLayout] = {}
        self._replay_text_cache: dict[tuple[str, str, int, int, int, int], _ReplayRowTextLayout] = {}
        self._visible_row_overscan = 2

    @staticmethod
    def _prune_cache(cache: dict, limit: int) -> None:
        if len(cache) > limit:
            cache.pop(next(iter(cache)))

    def _sorted_diff_infos(self, bset) -> tuple:
        key = (bset.directory, tuple(info.path for info in bset.maps))
        cached = self._sorted_diff_cache.get(key)
        if cached is not None:
            return cached
        cached = tuple(sorted(bset.maps, key=self._approx_local_difficulty))
        self._sorted_diff_cache[key] = cached
        self._prune_cache(self._sorted_diff_cache, 2048)
        return cached

    def _card_text_layout(
        self,
        text,
        *,
        raw_title: str,
        raw_artist: str,
        raw_diff: str,
        title_size: int,
        artist_size: int,
        meta_size: int,
        text_max_w: float,
        density: float,
    ) -> _SongCardTextLayout:
        title_max_w = max(0.0, text_max_w - 20.0 * density)
        artist_max_w = max(0.0, text_max_w - 18.0 * density)
        diff_max_w = max(0.0, text_max_w - 18.0 * density)
        cache_key = (
            raw_title,
            raw_artist,
            raw_diff,
            title_size,
            artist_size,
            meta_size,
            int(title_max_w),
            int(artist_max_w),
            int(diff_max_w),
        )
        cached = self._card_text_cache.get(cache_key)
        if cached is not None:
            return cached
        title, title_w = _fit_text(text, raw_title, title_size, title_max_w)
        artist, artist_w = _fit_text(text, raw_artist, artist_size, artist_max_w)
        diff_label, diff_w = _fit_text(text, raw_diff, meta_size, diff_max_w)
        cached = _SongCardTextLayout(
            title=title,
            title_w=title_w,
            artist=artist,
            artist_w=artist_w,
            diff_label=diff_label,
            diff_w=diff_w,
        )
        self._card_text_cache[cache_key] = cached
        self._prune_cache(self._card_text_cache, 8192)
        return cached

    def _replay_text_layout(
        self,
        text,
        *,
        primary: str,
        secondary: str = "",
        primary_size: int,
        secondary_size: int,
        primary_max_w: float,
        secondary_max_w: float,
    ) -> _ReplayRowTextLayout:
        cache_key = (
            primary,
            secondary,
            primary_size,
            secondary_size,
            int(primary_max_w),
            int(secondary_max_w),
        )
        cached = self._replay_text_cache.get(cache_key)
        if cached is not None:
            return cached
        fitted_primary = _truncate_text(text, primary, primary_size, primary_max_w)
        fitted_secondary = _truncate_text(text, secondary, secondary_size, secondary_max_w) if secondary else ""
        cached = _ReplayRowTextLayout(primary=fitted_primary, secondary=fitted_secondary)
        self._replay_text_cache[cache_key] = cached
        self._prune_cache(self._replay_text_cache, 8192)
        return cached

    def _visible_song_row_range(self, scene, metrics: dict, row_count: int) -> tuple[int, int]:
        if row_count <= 0:
            return (0, 0)
        density = metrics["layout"].context.density
        visible_rect: Rect = metrics["visible_rect"]
        item_h = metrics["item_h"]
        overscan_px = 108.0 * density + 100.0 * density
        start_y = max(0.0, scene._scroll_current - overscan_px)
        end_y = max(start_y, scene._scroll_current + visible_rect.h + overscan_px)
        start_idx = max(0, int(start_y // item_h) - self._visible_row_overscan)
        end_idx = min(row_count, int(end_y // item_h) + self._visible_row_overscan + 2)
        return (start_idx, end_idx)

    def _visible_replay_row_range(
        self,
        scene,
        *,
        total_rows: int,
        row_h: float,
        row_gap: float,
        top_pad: float,
        section_h: float,
    ) -> tuple[int, int]:
        if total_rows <= 0:
            return (0, 0)
        row_pitch = row_h + row_gap
        overscan_px = row_pitch * max(2, self._visible_row_overscan)
        start_y = max(0.0, scene._replay_scroll_current - top_pad - overscan_px)
        end_y = max(start_y, scene._replay_scroll_current - top_pad + section_h + overscan_px)
        start_idx = max(0, int(start_y // row_pitch))
        end_idx = min(total_rows, int(end_y // row_pitch) + 2)
        return (start_idx, end_idx)

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

    def _draw_map_button(self, text, theme, rect: Rect, *, hovered: bool, alpha: float) -> None:
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
        label = "o!"
        label_size = max(11, int(rect.h * 0.32))
        label_w, _ = text.measure(label, label_size)
        self._commands.text(
            label,
            rect.x + (rect.w - label_w) * 0.5,
            rect.y + (rect.h - label_size) * 0.5 - 3.0,
            label_size,
            color=theme.colors.text_primary if hovered else theme.colors.text_secondary,
            alpha=0.94 * alpha,
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

    @staticmethod
    def _approx_local_difficulty(info) -> float:
        return max(0.1, min(10.5, (float(info.ar) * 0.45) + (float(info.od) * 0.35) + (float(info.hp) * 0.15) + (float(info.cs) * 0.10) - 1.8))

    @staticmethod
    def _difficulty_color(stars: float) -> tuple[float, float, float]:
        anchors = (
            (0.10, (0.31, 0.75, 1.00)),
            (2.00, (0.40, 1.00, 0.57)),
            (2.70, (0.97, 0.91, 0.36)),
            (4.00, (1.00, 0.49, 0.41)),
            (5.30, (0.996, 0.235, 0.443)),
            (6.50, (0.40, 0.38, 0.87)),
            (8.00, (0.17, 0.14, 0.32)),
            (9.00, (0.06, 0.05, 0.08)),
            (10.50, (0.02, 0.02, 0.03)),
        )
        value = max(0.0, float(stars))
        if value <= anchors[0][0]:
            return anchors[0][1]
        for idx in range(1, len(anchors)):
            left_star, left_color = anchors[idx - 1]
            right_star, right_color = anchors[idx]
            if value <= right_star:
                mix = (value - left_star) / max(1e-6, right_star - left_star)
                return tuple(
                    left_color[channel] + (right_color[channel] - left_color[channel]) * mix
                    for channel in range(3)
                )
        return anchors[-1][1]

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
        item_h = (108.0 + 10.0) * density
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
        card_h = 108.0 * density
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
        if self._cover_renderer is None:
            self._cover_renderer = _SongCardCoverRenderer(scene.app.ctx)
        self._cover_renderer.set_projection(*scene.app.wnd.buffer_size)
        self._cover_renderer.update(1.0 / 60.0)
        with profiler.timer("song_select.view.cover_prepare"):
            self._cover_renderer.process_pending()
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

        with profiler.timer("song_select.view.info_panel"):
            self._draw_info_panel(scene, layout, x_offset=info_x_off)
        with profiler.timer("song_select.view.song_list"):
            self._draw_song_list(scene, layout, x_offset=cards_x_off)
        with profiler.timer("song_select.view.bottom_bar"):
            self._draw_bottom_bar(scene, layout, y_offset=bar_y_off)
        with profiler.timer("song_select.view.command_flush"):
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
        with profiler.timer("song_select.song_list.header"):
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

        with profiler.timer("song_select.song_list.flattened_list"):
            flattened = scene._build_flattened_list()
        if flattened:
            profiler.count("song_select.song_list.rows.total", len(flattened))
        row_start, row_end = self._visible_song_row_range(scene, metrics, len(flattened))
        if row_end > row_start:
            profiler.count("song_select.song_list.rows.iterated", row_end - row_start)
        previous_scissor = scene.app.ctx.scissor
        scene.app.ctx.scissor = (
            max(0, int(visible_rect.x)),
            max(0, int(scene.app.wnd.buffer_size[1] - (visible_rect.y + visible_rect.h))),
            max(1, int(visible_rect.w)),
            max(1, int(visible_rect.h)),
        )
        visible_cards = 0
        hovered_cards = 0
        cover_requests = 0
        cover_hits = 0
        diff_markers = 0
        title_size = max(layout.context.tokens.typography.body_l + 1, layout.context.tokens.typography.title_s - 2)
        artist_size = layout.context.tokens.typography.body_m
        meta_size = layout.context.tokens.typography.body_s
        diff_size = 18.0 * density
        diff_gap = 7.0 * density
        min_text_w = 240.0 * density
        with profiler.timer("song_select.song_list.cards"):
            for flat_i in range(row_start, row_end):
                set_idx, diff_idx = flattened[flat_i]
                card_rect = self.song_card_rect(scene, metrics, flat_i, set_idx=set_idx, diff_idx=diff_idx)
                if card_rect.bottom < visible_rect.y or card_rect.y > visible_rect.bottom:
                    continue

                bset = scene._sets[set_idx]
                info = bset.maps[diff_idx] if bset.maps else None
                fade_alpha = _edge_fade_alpha(card_rect, visible_rect, 100.0 * density)
                if fade_alpha <= 0.01:
                    continue
                visible_cards += 1
                is_selected = (set_idx == scene._selected_idx and diff_idx == scene._selected_diff_idx)
                background_path = bset.background_path
                is_hovered = (
                    fade_alpha > 0.08
                    and
                    not scene._dragging_songs
                    and card_rect.contains(scene._mouse_x, scene._mouse_y)
                )
                if is_hovered:
                    scene._hover_idx = flat_i
                    hovered_cards += 1
                hover_anim = _hover_anim(scene, ("song.card", flat_i), is_hovered, speed=0.20)
                card_rect = card_rect.translate(dx=6.0 * density * hover_anim, dy=-2.0 * density * hover_anim)
                if self._cover_renderer is not None and background_path:
                    cover_requests += 1
                    if self._cover_renderer.draw(
                        background_path,
                        card_rect,
                        radius=14.0 * density,
                        alpha=(0.95 if is_selected else (0.90 if is_hovered else 0.82)) * fade_alpha,
                        overlay_color=(
                            colors.surface_container[0],
                            colors.surface_container[1],
                            colors.surface_container[2],
                        ),
                        overlay_alpha=0.86,
                        dim=0.96 if is_selected else (0.92 if is_hovered else 0.84),
                    ):
                        cover_hits += 1
                self._commands.panel(
                    card_rect,
                    color=(
                        colors.surface_container[0],
                        colors.surface_container[1],
                        colors.surface_container[2],
                        (0.14 if background_path else (0.72 if is_selected else (0.62 if is_hovered else 0.56))) * fade_alpha,
                    ),
                    border_color=(
                        colors.focus_ring[0],
                        colors.focus_ring[1],
                        colors.focus_ring[2],
                        (0.28 if is_selected else (0.12 if is_hovered else 0.06)) * fade_alpha,
                    ),
                    border_width=1.0,
                )
                if scene._debug_layout:
                    scene._debug_rect(card_rect.x, card_rect.y, card_rect.w, card_rect.h, (0.88, 0.38, 0.95, 0.7))

                is_expanded_row = (set_idx == scene._selected_idx and len(bset.maps) > 1)
                diff_infos = (info,) if is_expanded_row and info is not None else self._sorted_diff_infos(bset)
                max_diff_space = max(0.0, card_rect.w - 36.0 * density - min_text_w)
                visible_diff_count = min(
                    len(diff_infos),
                    max(0, int((max_diff_space + diff_gap) // (diff_size + diff_gap))),
                )
                diff_markers += visible_diff_count
                diff_row_w = max(0.0, visible_diff_count * diff_size + max(0, visible_diff_count - 1) * diff_gap)
                diff_y = card_rect.y + 14.0 * density
                text_max_w = max(140.0 * density, card_rect.w - 36.0 * density - diff_row_w - 14.0 * density)
                raw_title = _prefer_renderable_text(
                    info.title_unicode if info is not None else "",
                    info.title if info is not None else bset.display_title,
                    empty_fallback="No Title",
                )
                raw_artist = _prefer_renderable_text(
                    info.artist_unicode if info is not None else "",
                    info.artist if info is not None else bset.display_artist,
                    empty_fallback="No Author",
                )
                if info is not None:
                    raw_diff = (info.version or "").strip() or "No difficulties found"
                else:
                    raw_diff = "No difficulties found"
                text_layout = self._card_text_layout(
                    text,
                    raw_title=raw_title,
                    raw_artist=raw_artist,
                    raw_diff=raw_diff,
                    title_size=title_size,
                    artist_size=artist_size,
                    meta_size=meta_size,
                    text_max_w=text_max_w,
                    density=density,
                )
                content_x = card_rect.x + 18.0 * density
                title_capsule = Rect(content_x - 2.0 * density, card_rect.y + 11.0 * density, min(text_max_w, text_layout.title_w + 20.0 * density), 20.0 * density)
                artist_capsule = Rect(content_x - 2.0 * density, card_rect.y + 34.0 * density, min(text_max_w, text_layout.artist_w + 18.0 * density), 18.0 * density)
                diff_capsule = Rect(content_x - 2.0 * density, card_rect.y + 57.0 * density, min(text_max_w, text_layout.diff_w + 18.0 * density), 18.0 * density)
                self._commands.panel(
                    title_capsule,
                    radius=10.0 * density,
                    color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], 0.44 * fade_alpha),
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0,
                )
                self._commands.panel(
                    artist_capsule,
                    radius=9.0 * density,
                    color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], 0.38 * fade_alpha),
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0,
                )
                self._commands.panel(
                    diff_capsule,
                    radius=9.0 * density,
                    color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], 0.38 * fade_alpha),
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0,
                )
                self._commands.text(text_layout.title, content_x + 8.0 * density, card_rect.y + 7.0 * density, title_size, color=colors.text_primary, alpha=0.96 * fade_alpha)
                self._commands.text(text_layout.artist, content_x + 7.0 * density, card_rect.y + 31.0 * density, artist_size, color=colors.text_secondary, alpha=0.88 * fade_alpha)
                self._commands.text(
                    text_layout.diff_label,
                    content_x + 7.0 * density,
                    card_rect.y + 56.0 * density,
                    meta_size,
                    color=colors.text_secondary if is_selected else colors.text_muted,
                    alpha=(0.90 if is_selected else 0.78) * fade_alpha,
                )
                diff_x = card_rect.right - 16.0 * density - diff_row_w
                for local_info in diff_infos[:visible_diff_count]:
                    approx_stars = self._approx_local_difficulty(local_info)
                    color = self._difficulty_color(approx_stars)
                    is_current_diff = local_info is info
                    circle_rect = Rect(diff_x, diff_y, diff_size, diff_size)
                    self._commands.panel(
                        circle_rect,
                        radius=diff_size * 0.5,
                        color=(
                            color[0],
                            color[1],
                            color[2],
                            (0.98 if is_current_diff else (0.94 if is_selected else 0.88)) * fade_alpha,
                        ),
                        border_color=(1.0, 1.0, 1.0, (0.10 if is_current_diff else 0.04) * fade_alpha),
                        border_width=0.0,
                    )
                    self._commands.panel(
                        Rect(circle_rect.x + 3.0 * density, circle_rect.y + 3.0 * density, circle_rect.w - 6.0 * density, circle_rect.h - 6.0 * density),
                        radius=max(0.0, circle_rect.w * 0.5 - 3.0 * density),
                        color=(color[0] * 0.88, color[1] * 0.88, color[2] * 0.88, (0.28 if is_current_diff else 0.18) * fade_alpha),
                        border_color=(0.0, 0.0, 0.0, 0.0),
                        border_width=0.0,
                    )
                    diff_x += diff_size + diff_gap

                scene._song_card_rects.append((card_rect.x, card_rect.y, card_rect.w, card_rect.h, flat_i))
        scene.app.ctx.scissor = previous_scissor
        if visible_cards:
            profiler.count("song_select.song_list.rows.visible", visible_cards)
        if hovered_cards:
            profiler.count("song_select.song_list.rows.hovered", hovered_cards)
        if cover_requests:
            profiler.count("song_select.song_list.covers.requested", cover_requests)
        if cover_hits:
            profiler.count("song_select.song_list.covers.drawn", cover_hits)
        if diff_markers:
            profiler.count("song_select.song_list.diff_markers", diff_markers)

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

        with profiler.timer("song_select.info_panel.header"):
            y = inner.y
            title_raw = _prefer_renderable_text(info.title_unicode, info.title, empty_fallback="No Title")
            artist_raw = _prefer_renderable_text(info.artist_unicode, info.artist, empty_fallback="No Author")
            hero_h = min(128.0 * density, panel_rect.h * 0.34)
            hero_rect = Rect(inner.x, inner.y, inner.w, hero_h)
            self._commands.panel(
                hero_rect,
                radius=12.0 * density,
                color=(colors.surface_container_low[0], colors.surface_container_low[1], colors.surface_container_low[2], 0.40 + (0.05 if scene._preview_playing else 0.0)),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )

            title_plate_rect = Rect(
                hero_rect.x + 0.0 * density,
                hero_rect.y - 10.0 * density,
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
                diff_draw_rect.y + 2.0 * density,
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

        all_replays = []
        replays = []
        beatmap_id = None
        online_state = None
        online_items = []
        official_beatmap_id = None
        official_score_state = None
        official_items = []
        with profiler.timer("song_select.info_panel.state_prep"):
            if scene._replay_source_tab == "online" and scene._online_replay_scope_tab == "official":
                auth = scene.app.official_osu_client.auth
                official_beatmap_id = scene._selected_official_beatmap_id()
                official_score_state = scene._official_score_state() if official_beatmap_id is not None else None
                official_items = [] if official_score_state is None else official_score_state.items
                if official_items:
                    profiler.count("song_select.info_panel.replays.official.total", len(official_items))
                if not auth.enabled:
                    replay_meta = "Server not configured"
                elif not auth.linked:
                    replay_meta = "Login required"
                elif official_score_state is None:
                    replay_meta = "Matching beatmap..."
                elif official_score_state.loading and not official_items:
                    replay_meta = "Loading..."
                elif official_score_state.error and not official_items:
                    replay_meta = "Unavailable"
                elif not official_items and official_score_state.loaded_at > 0.0:
                    replay_meta = "No official replays"
                else:
                    replay_meta = f"{len(official_items)} official"
            elif scene._replay_source_tab == "online":
                beatmap_id = scene._selected_online_beatmap_id()
                online_state = scene.app.social_client.online_replays(beatmap_id) if beatmap_id is not None else None
                online_items = [] if online_state is None else online_state.items
                if online_items:
                    profiler.count("song_select.info_panel.replays.online.total", len(online_items))
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
                all_replays = scene._replays_for_set(bset) if bset else []
                replays = scene._visible_replays_for_set(bset) if bset else []
                if replays:
                    profiler.count("song_select.info_panel.replays.local.total", len(replays))
                replay_meta = f"{len(replays)} shown" if len(replays) == len(all_replays) else f"{len(replays)} shown / {len(all_replays)} saved"
        draw_header_pair(
            self._commands,
            text,
            theme,
            label="Replays",
            meta=replay_meta,
            x=inner.x + 12.0,
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
        scene._replay_online_community_tab_rect = None
        scene._replay_online_official_tab_rect = None
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
        if scene._replay_source_tab == "online":
            scope_strip_rect = Rect(inner.right - 198.0 * density, y - 2.0 * density, 198.0 * density, 32.0 * density)
            draw_surface(
                self._commands,
                theme,
                scope_strip_rect,
                role="toolbar",
                radius=10.0 * density,
                alpha=0.44,
                border_width=0.0,
            )
            scope_tab_w = (scope_strip_rect.w - 22.0 * density) * 0.5
            scope_y = scope_strip_rect.y + 4.0 * density
            community_rect = Rect(scope_strip_rect.x + 8.0 * density, scope_y, scope_tab_w, 24.0 * density)
            official_rect = Rect(community_rect.right + 6.0 * density, scope_y, scope_tab_w, 24.0 * density)
            draw_tab(
                self._commands,
                text,
                theme,
                community_rect,
                label="Community",
                size=layout.context.tokens.typography.caption,
                selected=scene._online_replay_scope_tab == "community",
                hovered=community_rect.contains(scene._mouse_x, scene._mouse_y),
                alpha=0.92,
            )
            draw_tab(
                self._commands,
                text,
                theme,
                official_rect,
                label="Official",
                size=layout.context.tokens.typography.caption,
                selected=scene._online_replay_scope_tab == "official",
                hovered=official_rect.contains(scene._mouse_x, scene._mouse_y),
                alpha=0.92,
            )
            scene._replay_online_community_tab_rect = community_rect.tuple()
            scene._replay_online_official_tab_rect = official_rect.tuple()
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
        visible_replay_rows = 0
        with profiler.timer("song_select.info_panel.replay_rows"):
            if scene._replay_source_tab == "online" and scene._online_replay_scope_tab == "official":
                auth = scene.app.official_osu_client.auth
                row_h = 40.0 * density
                row_gap = 6.0 * density
                row_top_pad = 8.0 * density
                row_pitch = row_h + row_gap
                row_start, row_end = self._visible_replay_row_range(
                    scene,
                    total_rows=len(official_items),
                    row_h=row_h,
                    row_gap=row_gap,
                    top_pad=row_top_pad,
                    section_h=section_rect.h,
                )
                if row_end > row_start:
                    profiler.count("song_select.info_panel.replay_rows.iterated", row_end - row_start)
                if not auth.enabled:
                    self._commands.text("Official osu support is disabled on the server", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
                elif not auth.linked:
                    self._commands.text("Login in Settings to view official replays", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
                elif official_beatmap_id is None:
                    self._commands.text("Matching local difficulty to official beatmap...", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
                elif official_score_state is None or (official_score_state.loading and not official_items):
                    self._commands.text("Loading official replays...", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
                elif official_score_state.error and not official_items:
                    self._commands.text("Official replay list unavailable", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
                    self._commands.text(official_score_state.error, section_rect.x + 12.0 * density, section_rect.y + 34.0 * density, layout.context.tokens.typography.caption, color=colors.text_muted, alpha=0.65)
                elif not official_items:
                    self._commands.text("No official replay available for this beatmap", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
                for item_idx in range(row_start, row_end):
                    item = official_items[item_idx]
                    ry = section_rect.y + row_top_pad - scene._replay_scroll_current + item_idx * row_pitch
                    row_rect = Rect(section_rect.x + 4.0 * density, ry, section_rect.w - 8.0 * density, row_h)
                    visible_replay_rows += 1
                    row_hover = row_rect.contains(scene._mouse_x, scene._mouse_y)
                    row_anim = _hover_anim(scene, f"info.official_score.{item.score_id}", row_hover, speed=0.22)
                    row_draw_rect = row_rect.translate(dx=4.0 * density * row_anim)
                    selected = scene._selected_official_score_id == item.score_id
                    alpha_scale = 0.96 if item.has_replay else 0.60
                    self._commands.panel(
                        row_draw_rect,
                        radius=8.0 * density,
                        color=(
                            colors.surface_selected[0] if selected else colors.surface_container[0],
                            colors.surface_selected[1] if selected else colors.surface_container[1],
                            colors.surface_selected[2] if selected else colors.surface_container[2],
                            (0.72 if selected else 0.42) * alpha_scale + 0.10 * row_anim,
                        ),
                        border_color=(0.0, 0.0, 0.0, 0.0),
                        border_width=0.0,
                    )
                    if item.is_downloaded:
                        self._commands.panel(
                            Rect(row_draw_rect.x + 8.0 * density, row_draw_rect.y + 7.0 * density, 3.0 * density, row_draw_rect.h - 14.0 * density),
                            radius=1.5 * density,
                            color=(colors.success[0], colors.success[1], colors.success[2], 0.88),
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
                    label_text = f"{item.username or 'Unknown'}  +{item.mods_text}"
                    if not item.has_replay:
                        label_text += "  (no replay)"
                    text_layout = self._replay_text_layout(
                        text,
                        primary=label_text,
                        primary_size=layout.context.tokens.typography.body_s,
                        secondary_size=layout.context.tokens.typography.caption,
                        primary_max_w=row_draw_rect.w - 118.0 * density,
                        secondary_max_w=row_draw_rect.w - 136.0 * density,
                    )
                    self._commands.text(
                        text_layout.primary,
                        row_draw_rect.x + 18.0 * density,
                        row_draw_rect.y + 3.0 * density,
                        layout.context.tokens.typography.body_s,
                        color=colors.text_primary if item.has_replay else colors.text_muted,
                        alpha=0.92 + 0.04 * row_anim,
                    )
                    stat_text = f"{item.accuracy * 100.0:.2f}%  {item.max_combo}x"
                    self._commands.text(
                        self._replay_text_layout(
                            text,
                            primary=stat_text,
                            primary_size=layout.context.tokens.typography.caption,
                            secondary_size=layout.context.tokens.typography.caption,
                            primary_max_w=row_draw_rect.w - 136.0 * density,
                            secondary_max_w=0.0,
                        ).primary,
                        row_draw_rect.x + 18.0 * density,
                        row_draw_rect.y + 19.0 * density,
                        layout.context.tokens.typography.caption,
                        color=colors.text_secondary,
                        alpha=0.80,
                    )
                    action_text = "Open" if item.is_downloaded else ("Download" if item.has_replay else "No replay")
                    action_w = max(74.0 * density, text.measure(action_text, layout.context.tokens.typography.caption)[0] + 20.0 * density)
                    button_rect = Rect(row_draw_rect.right - action_w - 12.0 * density, row_draw_rect.y + 8.0 * density, action_w, 24.0 * density)
                    draw_button(
                        self._commands,
                        text,
                        theme,
                        button_rect,
                        label=action_text,
                        size=layout.context.tokens.typography.caption,
                        variant="quiet",
                        state=InteractionState.HOVER if button_rect.contains(scene._mouse_x, scene._mouse_y) else InteractionState.REST,
                        alpha=0.90 if item.has_replay or item.is_downloaded else 0.55,
                        radius=layout.context.tokens.radius_s,
                    )
                    scene._replay_rects.append((row_draw_rect.x, row_draw_rect.y, row_draw_rect.w, row_draw_rect.h, f"official:{item.score_id}"))
                    scene._replay_action_rects.append((button_rect.x, button_rect.y, button_rect.w, button_rect.h, f"official:{item.score_id}"))
                scene._replay_total_h = 8.0 * density + len(official_items) * row_h + max(0, len(official_items) - 1) * row_gap + 8.0 * density
            elif scene._replay_source_tab == "online" and scene._online_replay_scope_tab == "community" and online_state is not None:
                row_h = 40.0 * density
                row_gap = 6.0 * density
                row_top_pad = 8.0 * density
                row_pitch = row_h + row_gap
                row_start, row_end = self._visible_replay_row_range(
                    scene,
                    total_rows=len(online_items),
                    row_h=row_h,
                    row_gap=row_gap,
                    top_pad=row_top_pad,
                    section_h=section_rect.h,
                )
                if row_end > row_start:
                    profiler.count("song_select.info_panel.replay_rows.iterated", row_end - row_start)
                if online_state.loading and not online_items:
                    self._commands.text("Loading online replays...", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
                elif online_state.error and not online_items:
                    self._commands.text("Replay server unavailable", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
                    self._commands.text("Use refresh to try again.", section_rect.x + 12.0 * density, section_rect.y + 34.0 * density, layout.context.tokens.typography.caption, color=colors.text_muted, alpha=0.65)
                elif not online_items:
                    self._commands.text("No replays found", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
                    self._commands.text("Upload one from the local tab or refresh later.", section_rect.x + 12.0 * density, section_rect.y + 34.0 * density, layout.context.tokens.typography.caption, color=colors.text_muted, alpha=0.65)
                for item_idx in range(row_start, row_end):
                    item = online_items[item_idx]
                    scene._sync_online_replay_local_state(item)
                    ry = section_rect.y + row_top_pad - scene._replay_scroll_current + item_idx * row_pitch
                    row_rect = Rect(section_rect.x + 4.0 * density, ry, section_rect.w - 8.0 * density, row_h)
                    visible_replay_rows += 1
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
                    label = self._replay_text_layout(
                        text,
                        primary=label_text,
                        primary_size=layout.context.tokens.typography.body_s,
                        secondary_size=layout.context.tokens.typography.caption,
                        primary_max_w=row_draw_rect.w - 92.0 * density,
                        secondary_max_w=0.0,
                    ).primary
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
                scene._replay_total_h = 8.0 * density + len(online_items) * row_h + max(0, len(online_items) - 1) * row_gap + 8.0 * density
            elif replays:
                replay_dir = replay_dir_for_set(bset.directory)
                selected_replay_entities = scene.selected_replay_entity_keys()
                row_h = 36.0 * density
                row_gap = 4.0 * density
                row_top_pad = 8.0 * density
                row_pitch = row_h + row_gap
                row_start, row_end = self._visible_replay_row_range(
                    scene,
                    total_rows=len(replays),
                    row_h=row_h,
                    row_gap=row_gap,
                    top_pad=row_top_pad,
                    section_h=section_rect.h,
                )
                if row_end > row_start:
                    profiler.count("song_select.info_panel.replay_rows.iterated", row_end - row_start)
                for item_idx in range(row_start, row_end):
                    rp = replays[item_idx]
                    full_path = str(replay_dir / rp)
                    ry = section_rect.y + row_top_pad - scene._replay_scroll_current + item_idx * row_pitch
                    row_rect = Rect(section_rect.x + 4.0 * density, ry, section_rect.w - 8.0 * density, row_h)
                    visible_replay_rows += 1
                    entity_info = scene._replay_entity_info(full_path)
                    entity_key = entity_info[0] if entity_info is not None else full_path
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
                    label = self._replay_text_layout(
                        text,
                        primary=label_text,
                        primary_size=layout.context.tokens.typography.body_s,
                        secondary_size=layout.context.tokens.typography.caption,
                        primary_max_w=row_draw_rect.w - 76.0 * density,
                        secondary_max_w=0.0,
                    ).primary
                    self._commands.text(label, row_draw_rect.x + 18.0 * density, row_draw_rect.y + 8.0 * density, layout.context.tokens.typography.body_s, color=colors.text_primary if is_sel else colors.text_secondary, alpha=0.92 + 0.06 * row_anim)
                    dots_rect = Rect(row_draw_rect.right - 28.0 * density, row_draw_rect.y + 7.0 * density, 20.0 * density, 20.0 * density)
                    self._draw_kebab_button(theme, dots_rect, hovered=dots_rect.contains(scene._mouse_x, scene._mouse_y), alpha=0.88)
                    scene._replay_rects.append((row_draw_rect.x, row_draw_rect.y, row_draw_rect.w, row_draw_rect.h, full_path))
                    scene._replay_action_rects.append((dots_rect.x, dots_rect.y, dots_rect.w, dots_rect.h, full_path))
                scene._replay_total_h = 8.0 * density + len(replays) * row_h + max(0, len(replays) - 1) * row_gap + 8.0 * density
            else:
                self._commands.text("No local replays yet", section_rect.x + 12.0 * density, section_rect.y + 14.0 * density, layout.context.tokens.typography.body_m, color=colors.text_muted)
                self._commands.text("Drop .osr files into the mapset's replay folder.", section_rect.x + 12.0 * density, section_rect.y + 34.0 * density, layout.context.tokens.typography.caption, color=colors.text_muted, alpha=0.65)
                scene._replay_total_h = 0.0
        self._commands.clip_pop()
        if visible_replay_rows:
            profiler.count("song_select.info_panel.replay_rows.visible", visible_replay_rows)

        with profiler.timer("song_select.info_panel.context_menu"):
            if scene._replay_context_menu_rect is not None and scene._replay_context_menu_options:
                profiler.count("song_select.info_panel.context_menu.options", len(scene._replay_context_menu_options))
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

        with profiler.timer("song_select.info_panel.footer"):
            open_btn_rect = Rect(inner.x, button_y - 2.0 * density, inner.w, footer_h + 6.0 * density)
            open_hover = open_btn_rect.contains(scene._mouse_x, scene._mouse_y)
            draw_button(
                self._commands,
                text,
                theme,
                open_btn_rect,
                label=(
                    "Open replays folder"
                    if scene._replay_source_tab == "local"
                    else ("Refresh official replays" if scene._online_replay_scope_tab == "official" else "Refresh online replays")
                ),
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
        map_button_w = icon_button_size
        chat_button_w = icon_button_size
        settings_button_w = icon_button_size
        play_y = bar_rect.y + (bar_rect.h - play_h) * 0.5
        icon_button_y = bar_rect.y + (bar_rect.h - icon_button_size) * 0.5
        play_x = bar_rect.right - play_button_w - inner_pad
        settings_x = play_x - settings_button_w - 10.0 * density
        chat_x = settings_x - chat_button_w - 8.0 * density
        map_x = chat_x - map_button_w - 8.0 * density
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
        summary_w = max(120.0, map_x - summary_x - 14.0 * density)
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
        map_rect = Rect(map_x, icon_button_y, map_button_w, icon_button_size)
        chat_rect = Rect(chat_x, icon_button_y, chat_button_w, icon_button_size)
        settings_rect = Rect(settings_x, icon_button_y, settings_button_w, icon_button_size)
        map_hover = map_rect.contains(scene._mouse_x, scene._mouse_y)
        chat_hover = chat_rect.contains(scene._mouse_x, scene._mouse_y)
        settings_hover = settings_rect.contains(scene._mouse_x, scene._mouse_y)
        self._draw_map_button(text, theme, map_rect, hovered=map_hover, alpha=0.90)
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
        scene._map_browser_btn_rect = map_rect.tuple()
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

