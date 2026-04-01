from __future__ import annotations

import threading
from dataclasses import dataclass
from io import BytesIO

import moderngl
import numpy as np
import requests
from PIL import Image

from runtime_paths import MAPS_DIR
from ui.design import (
    InteractionState,
    draw_button,
    draw_linear_progress,
    draw_supporting_text,
    draw_surface,
    draw_tab,
    draw_tooltip,
    draw_text_field,
)
from ui.menu.animation import AnimatedFloat, ease_out_cubic
from ui.menu.commands import RenderCommandBuffer
from ui.menu.layout import Rect, build_layout_context


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


@dataclass(slots=True)
class _BrowserLayout:
    viewport: Rect
    backdrop_rect: Rect
    drawer_rect: Rect
    close_rect: Rect
    title_rect: Rect
    subtitle_rect: Rect
    search_rect: Rect
    search_button_rect: Rect
    status_tabs: list[tuple[str, Rect]]
    results_rect: Rect
    footer_rect: Rect
    footer_button_rect: Rect


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
class _CoverEntry:
    texture: moderngl.Texture | None = None
    size: tuple[int, int] = (1, 1)
    loading: bool = False
    failed: bool = False
    fade: float = 0.0


class _CoverArtRenderer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._prog = ctx.program(vertex_shader=_CARD_COVER_VERT, fragment_shader=_CARD_COVER_FRAG)
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
        self._entries: dict[str, _CoverEntry] = {}
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

    def request(self, url: str) -> None:
        value = str(url or "").strip()
        if not value:
            return
        entry = self._entries.get(value)
        if entry is None:
            entry = _CoverEntry()
            self._entries[value] = entry
        if entry.texture is not None or entry.loading or entry.failed:
            return
        entry.loading = True
        threading.Thread(target=self._load_worker, args=(value,), daemon=True, name="osu-cover-load").start()

    def _load_worker(self, url: str) -> None:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
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
                self._pending_results.append((url, image.size, payload))
        except Exception:
            entry = self._entries.get(url)
            if entry is not None:
                entry.loading = False
                entry.failed = True

    def _apply_pending(self) -> None:
        with self._lock:
            results = list(self._pending_results)
            self._pending_results.clear()
        for url, size, payload in results:
            entry = self._entries.setdefault(url, _CoverEntry())
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

    def draw(
        self,
        url: str,
        rect: Rect,
        *,
        radius: float,
        alpha: float,
        overlay_color: tuple[float, float, float],
        overlay_alpha: float,
        dim: float,
    ) -> bool:
        self._apply_pending()
        self.request(url)
        entry = self._entries.get(str(url or "").strip())
        if entry is None or entry.texture is None:
            return False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        entry.texture.use(location=0)
        self._prog["tex"].value = 0
        self._prog["projection"].write(self._projection_bytes)
        self._prog["u_rect_pos"].value = (rect.x, rect.y)
        self._prog["u_rect_size"].value = (rect.w, rect.h)
        self._prog["u_radius"].value = radius
        self._prog["u_alpha"].value = alpha * max(0.0, min(1.0, entry.fade))
        self._prog["u_dim"].value = dim
        self._prog["u_overlay_color"].value = overlay_color
        self._prog["u_overlay_alpha"].value = overlay_alpha
        self._vao.render(moderngl.TRIANGLES)
        return True


class OsuMapBrowserOverlay:
    _STATUS_TABS = (
        ("", "All"),
        ("ranked", "Ranked"),
        ("loved", "Loved"),
        ("pending", "Pending"),
    )

    def __init__(self, app) -> None:
        self.app = app
        self._commands = RenderCommandBuffer()
        self._open = False
        self._open_anim = AnimatedFloat(0.0, 0.0, 8.0)
        self._scroll = AnimatedFloat(0.0, 0.0, 14.0)
        self._mouse_x = 0
        self._mouse_y = 0
        self._search_query = ""
        self._search_focus = False
        self._status_filter = ""
        self._result_rects: list[tuple[Rect, int]] = []
        self._download_rects: list[tuple[Rect, int]] = []
        self._layout: _BrowserLayout | None = None
        self._cover_renderer = _CoverArtRenderer(app.ctx)

    @property
    def is_visible(self) -> bool:
        return self._open or self._open_anim.value > 0.001

    def open(self) -> None:
        self._open = True
        self._open_anim.set_target(1.0)
        self.app.official_osu_client.refresh_auth_status(force=True)

    def close(self) -> None:
        self._open = False
        self._open_anim.set_target(0.0)
        self._search_focus = False

    def toggle(self) -> None:
        if self._open:
            self.close()
        else:
            self.open()

    def update(self, dt: float) -> None:
        self._open_anim.update(dt)
        self._scroll.update(dt)
        if self._scroll.value < 0.01 and self._scroll.target < 0.01:
            self._scroll.snap(0.0)

    def draw(self, dt: float) -> None:
        self.update(dt)
        if not self.is_visible:
            return
        layout = self._build_layout()
        self._cover_renderer.set_projection(*self.app.wnd.buffer_size)
        self._cover_renderer.update(dt)
        density = build_layout_context(*self.app.wnd.buffer_size).density
        theme = self.app.menu_context().theme
        text = self.app.text
        colors = theme.colors
        commands = self._commands
        commands.clear()
        progress = ease_out_cubic(self._open_anim.value)
        draw_surface(commands, theme, layout.backdrop_rect, role="scrim", radius=0.0, alpha=0.52 * progress, border_width=0.0)
        draw_surface(commands, theme, layout.drawer_rect, role="drawer", radius=0.0, alpha=0.98 * progress, border_width=0.0)
        draw_button(
            commands,
            text,
            theme,
            layout.close_rect,
            label="X",
            size=self.app.menu_context().tokens.typography.body_m,
            variant="quiet",
            state=InteractionState.HOVER if layout.close_rect.contains(self._mouse_x, self._mouse_y) else InteractionState.REST,
            alpha=progress,
            radius=self.app.menu_context().tokens.radius_s,
        )
        commands.text("Beatmaps search", layout.title_rect.x, layout.title_rect.y, self.app.menu_context().theme.typography.headline, color=colors.text_primary, alpha=0.98 * progress)
        draw_supporting_text(
            commands,
            theme,
            "Search & download maps. Based on official API + mirrors",
            layout.subtitle_rect.x,
            layout.subtitle_rect.y,
            self.app.menu_context().tokens.typography.body_s,
            alpha=0.90 * progress,
            tone="secondary",
        )
        draw_text_field(
            commands,
            text,
            theme,
            layout.search_rect,
            value=self._search_query or "Search title, artist, mapper, tags...",
            placeholder=not bool(self._search_query),
            focused=self._search_focus,
            size=self.app.menu_context().tokens.typography.body_m,
            density=density,
            alpha=0.96 * progress,
            border_width=0.0,
        )
        draw_button(
            commands,
            text,
            theme,
            layout.search_button_rect,
            label="Search",
            size=self.app.menu_context().tokens.typography.body_s,
            variant="primary",
            state=InteractionState.HOVER if layout.search_button_rect.contains(self._mouse_x, self._mouse_y) else InteractionState.REST,
            alpha=0.96 * progress,
            radius=self.app.menu_context().tokens.radius_s,
        )
        for value, rect in layout.status_tabs:
            draw_tab(
                commands,
                text,
                theme,
                rect,
                label=next(label for current, label in self._STATUS_TABS if current == value),
                size=self.app.menu_context().tokens.typography.caption,
                selected=self._status_filter == value,
                hovered=rect.contains(self._mouse_x, self._mouse_y),
                alpha=0.92 * progress,
            )
        commands.flush(
            ctx=self.app.ctx,
            text=self.app.text,
            panels=self.app.panels,
            window_height=self.app.wnd.buffer_size[1],
        )
        commands.clear()

        auth = self.app.official_osu_client.auth
        search_state = self.app.official_osu_client.search
        commands.clip_push(layout.results_rect)
        self._result_rects = []
        self._download_rects = []
        hover_tooltip: tuple[float, float, str, float] | None = None
        if not auth.enabled:
            commands.text("Official osu OAuth is not configured on the server.", layout.results_rect.x + 12.0 * density, layout.results_rect.y + 12.0 * density, self.app.menu_context().tokens.typography.body_m, color=colors.text_secondary)
        elif not auth.linked:
            commands.text("Login from Settings to use official map search.", layout.results_rect.x + 12.0 * density, layout.results_rect.y + 12.0 * density, self.app.menu_context().tokens.typography.body_m, color=colors.text_secondary)
        elif search_state.loading and not search_state.items:
            commands.text("Searching beatmaps...", layout.results_rect.x + 12.0 * density, layout.results_rect.y + 12.0 * density, self.app.menu_context().tokens.typography.body_m, color=colors.text_secondary)
        elif search_state.error and not search_state.items:
            commands.text(search_state.error, layout.results_rect.x + 12.0 * density, layout.results_rect.y + 12.0 * density, self.app.menu_context().tokens.typography.body_m, color=colors.text_secondary)
        elif not search_state.items:
            commands.text("Type a query and press Enter or Search.", layout.results_rect.x + 12.0 * density, layout.results_rect.y + 12.0 * density, self.app.menu_context().tokens.typography.body_m, color=colors.text_secondary)
        row_h = 108.0 * density
        row_gap = 12.0 * density
        row_y = layout.results_rect.y + 8.0 * density - self._scroll.value
        previous_scissor = self.app.ctx.scissor
        self.app.ctx.scissor = (
            max(0, int(layout.results_rect.x)),
            max(0, int(self.app.wnd.buffer_size[1] - (layout.results_rect.y + layout.results_rect.h))),
            max(1, int(layout.results_rect.w)),
            max(1, int(layout.results_rect.h)),
        )
        for item in search_state.items:
            row_rect = Rect(layout.results_rect.x + 4.0 * density, row_y, layout.results_rect.w - 8.0 * density, row_h)
            row_y += row_h + row_gap
            if row_rect.bottom < layout.results_rect.y:
                continue
            if row_rect.y > layout.results_rect.bottom:
                break
            row_hover = row_rect.contains(self._mouse_x, self._mouse_y)
            button_rect = Rect(row_rect.right - 114.0 * density, row_rect.y + row_rect.h - 44.0 * density, 98.0 * density, 32.0 * density)
            installed_dir = self.app.installed_beatmap_dir_for_checksums(
                [str(beatmap.get("checksum") or "") for beatmap in item.beatmaps]
            )
            is_installed = bool(installed_dir or item.downloaded_dir)
            status_rect = Rect(row_rect.right - 34.0 * density, row_rect.y + 10.0 * density, 14.0 * density, 14.0 * density)
            if item.cover_url:
                self._cover_renderer.draw(
                    item.cover_url,
                    row_rect,
                    radius=14.0 * density,
                    alpha=(0.92 if row_hover else 0.84) * progress,
                    overlay_color=(
                        colors.surface_container[0],
                        colors.surface_container[1],
                        colors.surface_container[2],
                    ),
                    overlay_alpha=0.84,
                    dim=0.90 if row_hover else 0.82,
                )
            commands.panel(
                row_rect,
                radius=14.0 * density,
                color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], 0.18 if row_hover else 0.12),
                border_color=(1.0, 1.0, 1.0, 0.08 if row_hover else 0.04),
                border_width=1.0,
            )
            title = f"{item.artist} - {item.title}"
            subtitle = f"mapped by {item.creator}"
            available_text_w = max(120.0 * density, button_rect.x - row_rect.x - 66.0 * density)
            title_raw = title[:512]
            subtitle_raw = subtitle[:512]
            title_size = self.app.menu_context().tokens.typography.body_m
            subtitle_size = self.app.menu_context().tokens.typography.body_s
            title_h_pad = 20.0 * density
            subtitle_h_pad = 18.0 * density
            title_inner_max = max(40.0, available_text_w - title_h_pad)
            subtitle_max_outer = max(80.0, available_text_w - 8.0 * density)
            subtitle_inner_max = max(40.0, subtitle_max_outer - subtitle_h_pad)
            title_text = text.truncate(title_raw, title_size, title_inner_max)
            subtitle_text = text.truncate(subtitle_raw, subtitle_size, subtitle_inner_max)
            title_w, _ = text.measure(title_text, title_size)
            subtitle_w, _ = text.measure(subtitle_text, subtitle_size)
            title_capsule_rect = Rect(
                row_rect.x + 10.0 * density,
                row_rect.y + 10.0 * density,
                min(title_w + title_h_pad, available_text_w),
                20.0 * density,
            )
            subtitle_capsule_rect = Rect(
                row_rect.x + 10.0 * density,
                row_rect.y + 31.0 * density,
                min(subtitle_w + subtitle_h_pad, subtitle_max_outer),
                18.0 * density,
            )
            commands.panel(
                title_capsule_rect,
                radius=10.0 * density,
                color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], 0.44),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            commands.panel(
                subtitle_capsule_rect,
                radius=9.0 * density,
                color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], 0.38),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            commands.text(
                title_text,
                title_capsule_rect.x + 10.0 * density,
                row_rect.y + 8.0 * density,
                self.app.menu_context().tokens.typography.body_m,
                color=colors.text_primary,
                alpha=0.98,
            )
            draw_supporting_text(
                commands,
                theme,
                subtitle_text,
                subtitle_capsule_rect.x + 9.0 * density,
                row_rect.y + 29.0 * density,
                self.app.menu_context().tokens.typography.body_s,
                alpha=0.90,
                tone="secondary",
            )
            status_color = self._status_color(item.status)
            status_hover = status_rect.contains(self._mouse_x, self._mouse_y)
            commands.panel(
                status_rect,
                radius=status_rect.w * 0.5,
                color=(status_color[0], status_color[1], status_color[2], 0.10),
                border_color=(status_color[0], status_color[1], status_color[2], 0.98 if status_hover else 0.86),
                border_width=max(1.0, 2.0 * density),
            )
            diff_size = 18.0 * density
            diff_gap = 7.0 * density
            diff_x = row_rect.x + 14.0 * density
            diff_y = row_rect.bottom - diff_size - 14.0 * density
            sorted_beatmaps = sorted(
                item.beatmaps,
                key=lambda beatmap: float(beatmap.get("difficulty_rating") or 0.0),
            )
            diff_tooltip: tuple[float, float, str, float] | None = None
            for beatmap in sorted_beatmaps:
                if diff_x + diff_size > button_rect.x - 12.0 * density:
                    break
                stars = float(beatmap.get("difficulty_rating") or 0.0)
                version = str(beatmap.get("version") or "?")
                color = self._difficulty_color(stars)
                circle_rect = Rect(diff_x, diff_y, diff_size, diff_size)
                hovered_diff = circle_rect.contains(self._mouse_x, self._mouse_y)
                commands.panel(
                    circle_rect,
                    radius=diff_size * 0.5,
                    color=(color[0], color[1], color[2], 0.98 if hovered_diff else 0.88),
                    border_color=(1.0, 1.0, 1.0, 0.14 if hovered_diff else 0.06),
                    border_width=0.0,
                )
                commands.panel(
                    Rect(circle_rect.x + 3.0 * density, circle_rect.y + 3.0 * density, circle_rect.w - 6.0 * density, circle_rect.h - 6.0 * density),
                    radius=max(0.0, circle_rect.w * 0.5 - 3.0 * density),
                    color=(color[0] * 0.88, color[1] * 0.88, color[2] * 0.88, 0.28 if hovered_diff else 0.18),
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0,
                )
                if hovered_diff:
                    label = f"{version}  ★ {stars:.2f}"
                    diff_tooltip = (circle_rect.x + circle_rect.w * 0.5, circle_rect.y, label, circle_rect.h)
                diff_x += diff_size + diff_gap
            row_tooltip: tuple[float, float, str, float] | None = None
            if title_text != title_raw and title_capsule_rect.contains(self._mouse_x, self._mouse_y):
                row_tooltip = (
                    title_capsule_rect.x + title_capsule_rect.w * 0.5,
                    title_capsule_rect.y,
                    title_raw,
                    title_capsule_rect.h,
                )
            if status_hover:
                row_tooltip = (
                    status_rect.x + status_rect.w * 0.5,
                    status_rect.y,
                    self._status_label(item.status),
                    status_rect.h,
                )
            if diff_tooltip is not None:
                row_tooltip = diff_tooltip
            if row_tooltip is not None:
                hover_tooltip = row_tooltip
            draw_button(
                commands,
                text,
                theme,
                button_rect,
                label="Installing..." if item.is_downloading else ("Installed" if is_installed else "Download"),
                size=self.app.menu_context().tokens.typography.caption,
                variant="primary" if not is_installed else "quiet",
                state=InteractionState.HOVER if button_rect.contains(self._mouse_x, self._mouse_y) else InteractionState.REST,
                alpha=0.96,
                radius=self.app.menu_context().tokens.radius_s,
            )
            if item.is_downloading:
                draw_linear_progress(
                    commands,
                    theme,
                    Rect(row_rect.x + 14.0 * density, row_rect.bottom - 8.0 * density, row_rect.w - 28.0 * density, 4.0 * density),
                    value=item.download_progress,
                    alpha=0.96,
                    track_alpha=0.22,
                )
            self._result_rects.append((row_rect, item.beatmapset_id))
            self._download_rects.append((button_rect, item.beatmapset_id))
        self.app.ctx.scissor = previous_scissor
        commands.clip_pop()
        if hover_tooltip is not None:
            tip_anchor_x, tip_anchor_y, hover_text, ref_h = hover_tooltip
            cap = self.app.menu_context().tokens.typography.caption
            draw_tooltip(
                commands,
                text,
                theme,
                layout.results_rect,
                anchor_x=tip_anchor_x,
                anchor_y=tip_anchor_y,
                value=hover_text,
                size=cap,
                ref_h=ref_h,
                gap=10.0 * density,
                pad_x=18.0 * density,
                pad_y=4.0 * density,
                line_gap=2.0 * density,
                max_inner_width=max(40.0, layout.drawer_rect.w - 28.0 * density - 18.0 * density),
            )
        draw_surface(commands, theme, layout.footer_rect, role="toolbar", radius=0.0, alpha=0.84 * progress, border_width=0.0)
        draw_button(
            commands,
            text,
            theme,
            layout.footer_button_rect,
            label=f"Open maps folder ({MAPS_DIR.name})",
            size=self.app.menu_context().tokens.typography.caption,
            variant="secondary",
            state=InteractionState.HOVER if layout.footer_button_rect.contains(self._mouse_x, self._mouse_y) else InteractionState.REST,
            alpha=0.92,
            radius=self.app.menu_context().tokens.radius_s,
        )
        commands.flush(
            ctx=self.app.ctx,
            text=self.app.text,
            panels=self.app.panels,
            window_height=self.app.wnd.buffer_size[1],
        )

    def handle_key_event(self, key, action) -> bool:
        keys = self.app.wnd.keys
        if action != keys.ACTION_PRESS:
            return False
        if not self.is_visible:
            return False
        if key == keys.ESCAPE:
            self.close()
            return True
        if self._search_focus and key == keys.BACKSPACE:
            self._search_query = self._search_query[:-1]
            return True
        if self._search_focus and key == keys.ENTER:
            self._run_search()
            return True
        return self._open

    def handle_text(self, char: str) -> bool:
        if not (self._open and self._search_focus):
            return False
        if not char or ord(char[0]) < 32:
            return True
        self._search_query += char[0]
        return True

    def handle_mouse_press(self, x: int, y: int, button: int) -> bool:
        self._mouse_x = x
        self._mouse_y = y
        if button != 1:
            return self.is_visible
        if not self.is_visible:
            return False
        layout = self._build_layout()
        if not layout.drawer_rect.contains(x, y):
            self.close()
            return True
        if layout.close_rect.contains(x, y):
            self.close()
            return True
        if layout.search_rect.contains(x, y):
            self._search_focus = True
            return True
        self._search_focus = False
        if layout.search_button_rect.contains(x, y):
            self._run_search()
            return True
        for value, rect in layout.status_tabs:
            if rect.contains(x, y):
                self._status_filter = value
                self._run_search()
                return True
        for rect, beatmapset_id in self._download_rects:
            if rect.contains(x, y):
                item = next((entry for entry in self.app.official_osu_client.search.items if entry.beatmapset_id == beatmapset_id), None)
                installed_dir = None
                if item is not None:
                    installed_dir = self.app.installed_beatmap_dir_for_checksums(
                        [str(beatmap.get("checksum") or "") for beatmap in item.beatmaps]
                    )
                if installed_dir or (item is not None and item.downloaded_dir):
                    self.app.alert_overlay.show_message("Beatmap already installed.")
                    return True
                self.app.official_osu_client.download_mapset(beatmapset_id, str(MAPS_DIR))
                return True
        if layout.footer_button_rect.contains(x, y):
            self.app.open_maps_folder()
            return True
        if layout.results_rect.contains(x, y):
            self._scroll.snap(self._scroll.target)
            return True
        return True

    def handle_mouse_move(self, x: int, y: int) -> bool:
        self._mouse_x = x
        self._mouse_y = y
        return self.is_visible

    def handle_mouse_release(self, button: int) -> bool:
        return self._open

    def handle_scroll(self, y_offset: float) -> bool:
        if not self.is_visible:
            return False
        layout = self._build_layout()
        if not layout.results_rect.contains(self._mouse_x, self._mouse_y):
            return True
        density = self.app.menu_context().density
        row_h = 108.0 * density
        row_gap = 12.0 * density
        total_h = max(0.0, len(self.app.official_osu_client.search.items) * row_h + max(0, len(self.app.official_osu_client.search.items) - 1) * row_gap)
        max_scroll = max(0.0, total_h - layout.results_rect.h + 16.0 * self.app.menu_context().density)
        self._scroll.set_target(_clamp(self._scroll.target - y_offset * 42.0 * self.app.menu_context().density, 0.0, max_scroll))
        return True

    def wants_hand_cursor(self) -> bool:
        return self.is_visible

    def wants_text_cursor(self) -> bool:
        return self.is_visible and self._search_focus

    def _run_search(self) -> None:
        self._scroll.snap(0.0)
        self.app.official_osu_client.search_beatmapsets(
            self._search_query,
            status=self._status_filter,
            mode="osu",
            cursor_string="",
        )

    def _build_layout(self) -> _BrowserLayout:
        width, height = self.app.wnd.buffer_size
        context = build_layout_context(width, height)
        density = context.density
        drawer_w = min(560.0 * density, context.viewport.w * 0.42)
        visible_x = context.viewport.right - drawer_w
        hidden_x = context.viewport.right + 18.0 * density
        drawer_x = hidden_x + (visible_x - hidden_x) * ease_out_cubic(self._open_anim.value)
        drawer_rect = Rect(drawer_x, context.viewport.y, drawer_w, context.viewport.h)
        side_pad = 22.0 * density
        title_rect = Rect(drawer_rect.x + side_pad, drawer_rect.y + 24.0 * density, drawer_rect.w, context.theme.typography.headline)
        subtitle_rect = Rect(drawer_rect.x + side_pad, title_rect.y + context.theme.typography.headline + 6.0 * density, drawer_rect.w, context.tokens.typography.body_s)
        close_rect = Rect(drawer_rect.right - side_pad - 42.0 * density, drawer_rect.y + 18.0 * density, 42.0 * density, 36.0 * density)
        search_rect = Rect(drawer_rect.x + side_pad, drawer_rect.y + 92.0 * density, drawer_rect.w - side_pad * 2.0 - 98.0 * density, 42.0 * density)
        search_button_rect = Rect(search_rect.right + 10.0 * density, search_rect.y, 88.0 * density, search_rect.h)
        tab_y = search_rect.bottom + 14.0 * density
        tab_w = (drawer_rect.w - side_pad * 2.0 - 18.0 * density) / len(self._STATUS_TABS)
        status_tabs = []
        for idx, (value, _label) in enumerate(self._STATUS_TABS):
            status_tabs.append((value, Rect(drawer_rect.x + side_pad + idx * (tab_w + 6.0 * density), tab_y, tab_w, 28.0 * density)))
        footer_h = 68.0 * density
        footer_rect = Rect(drawer_rect.x, drawer_rect.bottom - footer_h, drawer_rect.w, footer_h)
        footer_button_rect = Rect(drawer_rect.x + side_pad, footer_rect.y + 16.0 * density, drawer_rect.w - side_pad * 2.0, 32.0 * density)
        results_rect = Rect(drawer_rect.x + side_pad, tab_y + 40.0 * density, drawer_rect.w - side_pad * 2.0, footer_rect.y - (tab_y + 52.0 * density))
        self._layout = _BrowserLayout(
            viewport=context.viewport,
            backdrop_rect=context.viewport,
            drawer_rect=drawer_rect,
            close_rect=close_rect,
            title_rect=title_rect,
            subtitle_rect=subtitle_rect,
            search_rect=search_rect,
            search_button_rect=search_button_rect,
            status_tabs=status_tabs,
            results_rect=results_rect,
            footer_rect=footer_rect,
            footer_button_rect=footer_button_rect,
        )
        return self._layout

    def _difficulty_color(self, stars: float) -> tuple[float, float, float]:
        anchors = (
            (0.10, (0.31, 0.75, 1.00)),  # easy
            (2.00, (0.40, 1.00, 0.57)),  # normal
            (2.70, (0.97, 0.91, 0.36)),  # hard
            (4.00, (1.00, 0.49, 0.41)),  # insane
            (5.30, (0.996, 0.235, 0.443)),  # expert
            (6.50, (0.40, 0.38, 0.87)),  # expert+
            (8.00, (0.17, 0.14, 0.32)),  # very dark purple
            (9.00, (0.06, 0.05, 0.08)),  # near black
            (10.50, (0.02, 0.02, 0.03)),  # black top-end
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

    def _status_color(self, status: str) -> tuple[float, float, float]:
        normalized = str(status or "").strip().lower()
        return {
            "ranked": (0.29, 0.72, 0.98),
            "approved": (0.29, 0.72, 0.98),
            "qualified": (0.42, 0.84, 0.49),
            "loved": (0.93, 0.38, 0.70),
            "pending": (0.96, 0.74, 0.35),
            "wip": (0.96, 0.53, 0.35),
            "graveyard": (0.52, 0.56, 0.63),
        }.get(normalized, (0.52, 0.56, 0.63))

    def _status_label(self, status: str) -> str:
        normalized = str(status or "").strip().lower()
        return {
            "ranked": "Ranked",
            "approved": "Approved",
            "qualified": "Qualified",
            "loved": "Loved",
            "pending": "Pending",
            "wip": "WIP",
            "graveyard": "Graveyard",
        }.get(normalized, normalized.capitalize() if normalized else "Unknown")
