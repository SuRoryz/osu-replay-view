"""Background image loader and fullscreen renderer."""

from __future__ import annotations

import threading
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageFilter

from profiling import profiler
from ui.design.theme import DEFAULT_BACKGROUND_GRADIENT

_BG_VERT = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
uniform mat4 projection;
out vec2 v_uv;
void main() {
    gl_Position = projection * vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_BG_FRAG = """
#version 330 core
in vec2 v_uv;
uniform sampler2D tex;
uniform float dim;
uniform float fade;
uniform vec2 uv_scale;
uniform vec2 uv_offset;
uniform float uv_zoom;
uniform vec2 uv_drift;
out vec4 frag_color;
void main() {
    vec2 centered_uv = vec2(0.5) + (v_uv - vec2(0.5)) * uv_zoom + uv_drift;
    vec2 sample_uv = uv_offset + centered_uv * uv_scale;
    vec2 texel = 0.5 / vec2(textureSize(tex, 0));
    vec2 uv_min = uv_offset + texel;
    vec2 uv_max = uv_offset + uv_scale - texel;
    sample_uv = clamp(sample_uv, uv_min, uv_max);
    vec3 col = texture(tex, sample_uv).rgb * dim;
    frag_color = vec4(col, fade);
}
"""

_GRAD_FRAG = """
#version 330 core
in vec2 v_uv;
uniform vec3 color_top;
uniform vec3 color_bot;
out vec4 frag_color;
void main() {
    vec3 col = mix(color_top, color_bot, v_uv.y);
    frag_color = vec4(col, 1.0);
}
"""

BLUR_RADIUS = 18
DIM_FACTOR = 0.35
MAX_TEX_SIZE = 2560
FADE_DURATION = 0.35


class BackgroundRenderer:
    """Loads, blurs, and renders beatmap backgrounds or a gradient fallback."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._prog = ctx.program(vertex_shader=_BG_VERT, fragment_shader=_BG_FRAG)
        self._grad_prog = ctx.program(vertex_shader=_BG_VERT, fragment_shader=_GRAD_FRAG)

        verts = np.array([
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
        ], dtype="f4")
        idx = np.array([0, 1, 2, 0, 2, 3], dtype="i4")

        buf = ctx.buffer(verts.tobytes())
        ibo = ctx.buffer(idx.tobytes())
        self._vao = ctx.vertex_array(
            self._prog, [(buf, "2f 2f", "in_pos", "in_uv")], index_buffer=ibo
        )
        self._grad_vao = ctx.vertex_array(
            self._grad_prog, [(buf, "2f 2f", "in_pos", "in_uv")], index_buffer=ibo
        )

        self._tex: moderngl.Texture | None = None
        self._prev_tex: moderngl.Texture | None = None
        self._cached_path: str | None = None
        self._cached_signature: tuple[str | None, int] | None = None
        self._projection = np.eye(4, dtype="f4")
        self._projection_bytes = self._projection.tobytes()
        self._viewport_size = (1, 1)
        self._tex_size = (1, 1)
        self._prev_tex_size = (1, 1)
        self._pending_path: str | None = None
        self._pending_signature: tuple[str | None, int] | None = None
        self._pending_result: tuple | None = None  # (path, blur_radius, size, data) when ready
        self._load_lock = threading.Lock()
        self._fade_timer: float = 1.0  # 1.0 = fully visible, 0 = just applied
        self._drift_timer: float = 0.0
        self._blur_radius = BLUR_RADIUS
        self._dim_factor = DIM_FACTOR
        self._motion_enabled = True

    def set_projection(self, w: int, h: int) -> None:
        p = np.eye(4, dtype="f4")
        p[0, 0] = 2.0 / 1.0
        p[1, 1] = -2.0 / 1.0
        p[3, 0] = -1.0
        p[3, 1] = 1.0
        self._projection = p
        self._projection_bytes = p.tobytes()
        self._viewport_size = (max(1, int(w)), max(1, int(h)))

    def configure(self, *, blur_radius: int | None = None, dim_factor: float | None = None, motion_enabled: bool | None = None) -> None:
        if blur_radius is not None:
            self._blur_radius = max(0, int(blur_radius))
        if dim_factor is not None:
            self._dim_factor = max(0.0, min(1.0, float(dim_factor)))
        if motion_enabled is not None:
            self._motion_enabled = bool(motion_enabled)

    def _do_load_worker(self, image_path: str, blur_radius: int) -> None:
        """Background thread: load, resize, blur. Store result for main thread."""
        with profiler.timer("background.load_worker"):
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception:
                return
            w, h = img.size
            scale = min(MAX_TEX_SIZE / w, MAX_TEX_SIZE / h, 1.0)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            if blur_radius > 0:
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            data = img.tobytes()
        with self._load_lock:
            if self._pending_signature == (image_path, blur_radius):
                self._pending_result = (image_path, blur_radius, img.size, data)

    def load(self, image_path: str | None) -> None:
        signature = (image_path, self._blur_radius)
        # Already showing this, or already loading it
        if signature == self._cached_signature and self._tex is not None:
            return
        with self._load_lock:
            if signature == self._pending_signature:
                return  # already loading this path
            self._pending_path = image_path
            self._pending_signature = signature
            self._pending_result = None
        if image_path is None or not Path(image_path).is_file():
            self._cached_path = image_path
            self._cached_signature = signature
            self._pending_path = None
            self._pending_signature = None
            if self._tex is not None:
                self._tex.release()
                self._tex = None
            if self._prev_tex is not None:
                self._prev_tex.release()
                self._prev_tex = None
            return
        # Kick off async load; keep showing current texture until ready
        t = threading.Thread(target=self._do_load_worker, args=(image_path, self._blur_radius), daemon=True)
        t.start()

    def _apply_pending(self) -> None:
        """Apply completed background load on main thread (call from draw)."""
        with self._load_lock:
            result = self._pending_result
            self._pending_result = None
        if result is None:
            return
        with profiler.timer("background.apply_pending"):
            path, blur_radius, (w, h), data = result
            self._cached_path = path
            self._cached_signature = (path, blur_radius)
            if self._prev_tex is not None:
                self._prev_tex.release()
                self._prev_tex = None
            if self._tex is not None:
                self._prev_tex = self._tex
                self._prev_tex_size = self._tex_size
            self._tex = self.ctx.texture((w, h), 3, data)
            self._tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._tex.repeat_x = False
            self._tex.repeat_y = False
            self._tex_size = (w, h)
            self._fade_timer = 0.0  # start fade-in

    def _uv_params(self, tex_size: tuple[int, int], *, zoom: float, drift_x: float, drift_y: float) -> tuple[tuple[float, float], tuple[float, float], float, tuple[float, float]]:
        viewport_aspect = self._viewport_size[0] / max(1.0, float(self._viewport_size[1]))
        tex_aspect = tex_size[0] / max(1.0, float(tex_size[1]))
        uv_scale_x = 1.0
        uv_scale_y = 1.0
        uv_offset_x = 0.0
        uv_offset_y = 0.0
        if viewport_aspect > tex_aspect:
            uv_scale_y = tex_aspect / viewport_aspect
            uv_offset_y = (1.0 - uv_scale_y) * 0.5
        else:
            uv_scale_x = viewport_aspect / tex_aspect
            uv_offset_x = (1.0 - uv_scale_x) * 0.5
        zoom = min(1.0, max(0.9, zoom))
        drift_limit_x = max(0.0, uv_scale_x * (1.0 - zoom) * 0.5)
        drift_limit_y = max(0.0, uv_scale_y * (1.0 - zoom) * 0.5)
        return (
            (uv_scale_x, uv_scale_y),
            (uv_offset_x, uv_offset_y),
            zoom,
            (
                max(-drift_limit_x, min(drift_limit_x, drift_x)),
                max(-drift_limit_y, min(drift_limit_y, drift_y)),
            ),
        )

    def _draw_texture(
        self,
        texture: moderngl.Texture,
        tex_size: tuple[int, int],
        *,
        fade: float,
        zoom: float,
        drift_x: float,
        drift_y: float,
    ) -> None:
        uv_scale, uv_offset, zoom_value, drift = self._uv_params(
            tex_size,
            zoom=zoom,
            drift_x=drift_x,
            drift_y=drift_y,
        )
        texture.use(location=0)
        self._prog["tex"].value = 0
        self._prog["dim"].value = self._dim_factor
        self._prog["fade"].value = fade
        self._prog["uv_scale"].value = uv_scale
        self._prog["uv_offset"].value = uv_offset
        self._prog["uv_zoom"].value = zoom_value
        self._prog["uv_drift"].value = drift
        self._prog["projection"].write(self._projection_bytes)
        self._vao.render(moderngl.TRIANGLES)

    def draw(self, dt: float = 0.0) -> None:
        with profiler.timer("background.draw"):
            self._apply_pending()
            self._drift_timer += max(0.0, dt)
            if self._tex is not None and self._fade_timer < 1.0:
                self._fade_timer = min(1.0, self._fade_timer + dt / max(1e-6, FADE_DURATION))
            if self._prev_tex is not None and self._fade_timer >= 1.0:
                self._prev_tex.release()
                self._prev_tex = None
            self.ctx.disable(moderngl.DEPTH_TEST)
            if self._tex is not None:
                if self._prev_tex is None and self._fade_timer < 1.0:
                    self._grad_prog["color_top"].value = DEFAULT_BACKGROUND_GRADIENT.top
                    self._grad_prog["color_bot"].value = DEFAULT_BACKGROUND_GRADIENT.bottom
                    self._grad_prog["projection"].write(self._projection_bytes)
                    self._grad_vao.render(moderngl.TRIANGLES)
                self.ctx.enable(moderngl.BLEND)
                self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                t = self._fade_timer
                fade = 1.0 - (1.0 - t) ** 3 if t < 1.0 else 1.0
                if self._motion_enabled:
                    drift_x = 0.006 * np.sin(self._drift_timer * 0.11)
                    drift_y = 0.004 * np.cos(self._drift_timer * 0.09)
                    prev_zoom = 0.988
                    zoom = 0.976
                else:
                    drift_x = 0.0
                    drift_y = 0.0
                    prev_zoom = 1.0
                    zoom = 1.0
                if self._prev_tex is not None:
                    self._prev_tex.repeat_x = False
                    self._prev_tex.repeat_y = False
                    self._draw_texture(
                        self._prev_tex,
                        self._prev_tex_size,
                        fade=1.0,
                        zoom=prev_zoom,
                        drift_x=-drift_x * 0.5,
                        drift_y=-drift_y * 0.5,
                    )
                self._draw_texture(
                    self._tex,
                    self._tex_size,
                    fade=fade,
                    zoom=zoom,
                    drift_x=drift_x,
                    drift_y=drift_y,
                )
            else:
                self._grad_prog["color_top"].value = DEFAULT_BACKGROUND_GRADIENT.top
                self._grad_prog["color_bot"].value = DEFAULT_BACKGROUND_GRADIENT.bottom
                self._grad_prog["projection"].write(self._projection_bytes)
                self._grad_vao.render(moderngl.TRIANGLES)

    def release(self) -> None:
        if self._tex is not None:
            self._tex.release()
            self._tex = None
        if self._prev_tex is not None:
            self._prev_tex.release()
            self._prev_tex = None
        self._fade_timer = 1.0
        with self._load_lock:
            self._cached_path = None
            self._cached_signature = None
            self._pending_path = None
            self._pending_signature = None
            self._pending_result = None
