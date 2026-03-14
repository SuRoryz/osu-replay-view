"""GPU text renderer using a Pillow-generated glyph atlas."""

from __future__ import annotations

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from profiling import profiler

_TEXT_VERT = """
#version 330 core

in vec2 in_pos;
in vec2 in_uv;
in vec4 in_color;

uniform mat4 projection;

out vec2 v_uv;
out vec4 v_color;

void main() {
    gl_Position = projection * vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
    v_color = in_color;
}
"""

_TEXT_FRAG = """
#version 330 core

in vec2 v_uv;
in vec4 v_color;

uniform sampler2D tex;

out vec4 frag_color;

void main() {
    float a = texture(tex, v_uv).r;
    if (a < 0.01) discard;
    frag_color = vec4(v_color.rgb, v_color.a * a);
}
"""

_FONT_NAMES = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
_ATLAS_SIZES = [16, 20, 24, 32, 48]
_GLYPH_RANGE = range(32, 127)
_ATLAS_PADDING = 2
MAX_QUADS = 4096
MAX_LAYOUT_CACHE = 4096


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    for name in _FONT_NAMES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class _GlyphInfo:
    __slots__ = ("u0", "v0", "u1", "v1", "width", "height", "advance", "y_offset")

    def __init__(self, u0, v0, u1, v1, w, h, adv, y_off):
        self.u0, self.v0, self.u1, self.v1 = u0, v0, u1, v1
        self.width, self.height = w, h
        self.advance = adv
        self.y_offset = y_off


class _SizedAtlas:
    """Atlas for a single font size."""

    def __init__(self, ctx: moderngl.Context, font_size: int):
        font = _find_font(font_size)
        self.glyphs: dict[int, _GlyphInfo] = {}
        self.line_height = font_size

        chars = [chr(c) for c in _GLYPH_RANGE]
        bboxes = []
        for ch in chars:
            bb = font.getbbox(ch)
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]
            bboxes.append((ch, w, h, bb))

        self.max_ascent = max(-b[3][1] for b in bboxes if b[1] > 0) if bboxes else font_size

        cols = 16
        rows = (len(chars) + cols - 1) // cols
        cell_w = max(b[1] for b in bboxes) + _ATLAS_PADDING * 2
        cell_h = max(b[2] for b in bboxes) + _ATLAS_PADDING * 2
        tex_w = cols * cell_w
        tex_h = rows * cell_h

        img = Image.new("L", (tex_w, tex_h), 0)
        draw = ImageDraw.Draw(img)

        for i, (ch, w, h, bb) in enumerate(bboxes):
            col = i % cols
            row = i // cols
            x0 = col * cell_w + _ATLAS_PADDING
            y0 = row * cell_h + _ATLAS_PADDING
            draw.text((x0 - bb[0], y0 - bb[1]), ch, fill=255, font=font)

            u0 = x0 / tex_w
            v0 = y0 / tex_h
            u1 = (x0 + w) / tex_w
            v1 = (y0 + h) / tex_h
            advance = w + 1
            y_off = bb[1]
            self.glyphs[ord(ch)] = _GlyphInfo(u0, v0, u1, v1, w, h, advance, y_off)

        data = img.tobytes()
        self.texture = ctx.texture((tex_w, tex_h), 1, data)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.swizzle = "RRRR"


class TextRenderer:
    """Batched GPU text renderer with multi-size glyph atlases."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._prog = ctx.program(vertex_shader=_TEXT_VERT, fragment_shader=_TEXT_FRAG)
        self._projection_uniform = self._prog["projection"]
        self._tex_uniform = self._prog["tex"]
        self._tex_uniform.value = 0
        self._atlases: dict[int, _SizedAtlas] = {}
        for sz in _ATLAS_SIZES:
            self._atlases[sz] = _SizedAtlas(ctx, sz)

        # 8 floats per vertex, 4 vertices per quad
        self._buf = ctx.buffer(reserve=MAX_QUADS * 4 * 8 * 4)
        self._ibo = ctx.buffer(self._build_indices().tobytes())
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._buf, "2f 2f 4f", "in_pos", "in_uv", "in_color")],
            index_buffer=self._ibo,
        )
        self._projection = np.eye(4, dtype="f4")
        self._projection_bytes = self._projection.tobytes()
        self._measure_cache: dict[tuple[str, int], tuple[float, float]] = {}
        self._truncate_cache: dict[tuple[str, int, int], str] = {}
        self._layout_cache: dict[tuple[str, int], tuple[int, np.ndarray]] = {}
        self._styled_layout_cache: dict[tuple[str, int, float, float, float, float], tuple[int, np.ndarray]] = {}
        self._batch_bufs: dict[int, np.ndarray] = {}
        self._draws_by_size: dict[int, list[tuple[np.ndarray, float, float]]] = {}

    def set_projection(self, w: int, h: int) -> None:
        p = np.eye(4, dtype="f4")
        p[0, 0] = 2.0 / w
        p[1, 1] = -2.0 / h
        p[3, 0] = -1.0
        p[3, 1] = 1.0
        self._projection = p
        self._projection_bytes = p.tobytes()
        self._projection_uniform.write(self._projection_bytes)

    def _closest_size(self, size: int) -> int:
        return min(_ATLAS_SIZES, key=lambda s: abs(s - size))

    def measure(self, text: str, size: int) -> tuple[float, float]:
        profiler.count("text.measure.calls")
        key = (text, size)
        cached = self._measure_cache.get(key)
        if cached is not None:
            return cached
        with profiler.timer("text.measure"):
            atlas = self._atlases[self._closest_size(size)]
            scale = size / atlas.line_height
            w = 0.0
            for ch in text:
                g = atlas.glyphs.get(ord(ch))
                if g:
                    w += g.advance * scale
            result = (w, size)
        self._measure_cache[key] = result
        return result

    def truncate(self, value: str, size: int, max_width: float) -> str:
        if max_width <= 0:
            return ""
        width_px = int(max_width)
        cache_key = (value, size, width_px)
        cached = self._truncate_cache.get(cache_key)
        if cached is not None:
            return cached
        if self.measure(value, size)[0] <= max_width:
            self._truncate_cache[cache_key] = value
            return value

        ellipsis = "..."
        if self.measure(ellipsis, size)[0] > max_width:
            self._truncate_cache[cache_key] = ellipsis
            return ellipsis

        lo = 0
        hi = len(value)
        best = ellipsis
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid].rstrip() + ellipsis
            if self.measure(candidate, size)[0] <= max_width:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        self._truncate_cache[cache_key] = best
        return best

    def _layout_for_text(self, text: str, size: int) -> tuple[int, np.ndarray]:
        key = (text, size)
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached

        atlas_key = self._closest_size(size)
        atlas = self._atlases[atlas_key]
        scale = size / atlas.line_height
        max_verts = len(text) * 4
        layout = np.empty((max_verts, 4), dtype="f4")
        vert_count = 0
        cx = 0.0
        for ch in text:
            gi = atlas.glyphs.get(ord(ch))
            if gi is None:
                cx += size * 0.5
                continue
            w = gi.width * scale
            h = gi.height * scale
            gy = gi.y_offset * scale
            j = vert_count
            layout[j + 0] = [cx, gy, gi.u0, gi.v0]
            layout[j + 1] = [cx + w, gy, gi.u1, gi.v0]
            layout[j + 2] = [cx + w, gy + h, gi.u1, gi.v1]
            layout[j + 3] = [cx, gy + h, gi.u0, gi.v1]
            vert_count += 4
            cx += gi.advance * scale

        result = (atlas_key, layout[:vert_count].copy())
        self._layout_cache[key] = result
        if len(self._layout_cache) > MAX_LAYOUT_CACHE:
            self._layout_cache.pop(next(iter(self._layout_cache)))
        return result

    def _ensure_batch_buf(self, atlas_size: int) -> np.ndarray:
        buf = self._batch_bufs.get(atlas_size)
        if buf is None:
            buf = np.empty((MAX_QUADS * 4, 8), dtype="f4")
            self._batch_bufs[atlas_size] = buf
        return buf

    def _styled_layout_for_text(
        self,
        text: str,
        size: int,
        color: tuple[float, float, float],
        alpha: float,
    ) -> tuple[int, np.ndarray]:
        r, g, b = color
        key = (text, size, r, g, b, alpha)
        cached = self._styled_layout_cache.get(key)
        if cached is not None:
            return cached

        atlas_key, layout = self._layout_for_text(text, size)
        styled = np.empty((layout.shape[0], 8), dtype="f4")
        if layout.size > 0:
            styled[:, 0:2] = layout[:, 0:2]
            styled[:, 2:4] = layout[:, 2:4]
            styled[:, 4] = r
            styled[:, 5] = g
            styled[:, 6] = b
            styled[:, 7] = alpha

        result = (atlas_key, styled)
        self._styled_layout_cache[key] = result
        if len(self._styled_layout_cache) > MAX_LAYOUT_CACHE:
            self._styled_layout_cache.pop(next(iter(self._styled_layout_cache)))
        return result

    def _flush_batch(self, atlas_size: int, batch_buf: np.ndarray, quad_count: int) -> None:
        if quad_count <= 0:
            return
        vert_count = quad_count * 4
        self._buf.write(memoryview(batch_buf[:vert_count]).cast("B"))
        self._atlases[atlas_size].texture.use(location=0)
        self._vao.render(moderngl.TRIANGLES, vertices=quad_count * 6)
        profiler.count("text.quads", quad_count)

    def draw(self, text: str, x: float, y: float, size: int,
             color: tuple = (1.0, 1.0, 1.0), alpha: float = 1.0) -> None:
        """Draw a text string. Must be called between begin()/end()."""
        if not text:
            return
        key, layout = self._styled_layout_for_text(text, size, color, alpha)
        if layout.size == 0:
            return
        self._draws_by_size.setdefault(key, []).append((layout, x, y))

    def begin(self) -> None:
        self._draws_by_size = {}

    def end(self) -> None:
        if not self._draws_by_size:
            return

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._projection_uniform.write(self._projection_bytes)
        profiler.count("text.end.calls")

        with profiler.timer("text.end"):
            for atlas_size, draws in self._draws_by_size.items():
                batch_buf = self._ensure_batch_buf(atlas_size)
                total_quads = sum(layout.shape[0] >> 2 for layout, _, _ in draws)
                if total_quads <= MAX_QUADS:
                    vert_cursor = 0
                    for layout, x, y in draws:
                        vert_count = layout.shape[0]
                        dst = batch_buf[vert_cursor:vert_cursor + vert_count]
                        dst[:, 0] = layout[:, 0] + x
                        dst[:, 1] = layout[:, 1] + y
                        dst[:, 2:] = layout[:, 2:]
                        vert_cursor += vert_count
                    self._flush_batch(atlas_size, batch_buf, total_quads)
                    continue

                quad_cursor = 0
                for layout, x, y in draws:
                    total_quads = layout.shape[0] >> 2
                    quad_start = 0
                    while quad_start < total_quads:
                        if quad_cursor >= MAX_QUADS:
                            self._flush_batch(atlas_size, batch_buf, quad_cursor)
                            quad_cursor = 0
                        take = min(MAX_QUADS - quad_cursor, total_quads - quad_start)
                        src_start = quad_start * 4
                        src_end = src_start + take * 4
                        dst_start = quad_cursor * 4
                        dst_end = dst_start + take * 4
                        src = layout[src_start:src_end]
                        dst = batch_buf[dst_start:dst_end]
                        dst[:, 0] = src[:, 0] + x
                        dst[:, 1] = src[:, 1] + y
                        dst[:, 2:] = src[:, 2:]
                        quad_cursor += take
                        quad_start += take
                self._flush_batch(atlas_size, batch_buf, quad_cursor)

        self._draws_by_size.clear()

    @staticmethod
    def _build_indices() -> np.ndarray:
        idx = np.empty(MAX_QUADS * 6, dtype="i4")
        for i in range(MAX_QUADS):
            v = i * 4
            j = i * 6
            idx[j:j + 6] = [v, v + 1, v + 2, v, v + 2, v + 3]
        return idx
