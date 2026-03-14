"""UI elements: Panel, Button, ScrollList -- all rendered with ModernGL."""

from __future__ import annotations

import moderngl
import numpy as np

from profiling import profiler

_PANEL_VERT = """
#version 330 core
in vec2 in_pos;
uniform mat4 projection;
uniform vec2 u_rect_pos;
uniform vec2 u_rect_size;
out vec2 v_local;
void main() {
    vec2 world = u_rect_pos + in_pos * u_rect_size;
    gl_Position = projection * vec4(world, 0.0, 1.0);
    v_local = in_pos * u_rect_size;
}
"""

_PANEL_FRAG = """
#version 330 core
in vec2 v_local;
uniform vec2  u_rect_size;
uniform float u_radius;
uniform vec4  u_color;
uniform vec4  u_border_color;
uniform float u_border_width;
out vec4 frag_color;

float rounded_rect_sdf(vec2 p, vec2 half_size, float r) {
    vec2 d = abs(p - half_size) - (half_size - vec2(r));
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - r;
}

void main() {
    float d = rounded_rect_sdf(v_local, u_rect_size * 0.5, u_radius);
    float fill_alpha = 1.0 - smoothstep(-0.5, 0.5, d);
    if (fill_alpha < 0.001) discard;

    float border_mask = 0.0;
    if (u_border_width > 0.001) {
        float border_inner = d + u_border_width;
        border_mask = smoothstep(-0.5, 0.5, border_inner);
    }

    vec4 col = mix(u_color, u_border_color, border_mask);
    col.a *= fill_alpha;
    frag_color = col;
}
"""

# Gradient bar: fixed mask from left (alpha 0) to spawn point (alpha 1) - "eats" bars
_GRADIENT_BAR_VERT = """
#version 330 core
in vec2 in_pos;
uniform mat4 projection;
uniform vec2 u_rect_pos;
uniform vec2 u_rect_size;
out vec2 v_world;
void main() {
    vec2 world = u_rect_pos + in_pos * u_rect_size;
    gl_Position = projection * vec4(world, 0.0, 1.0);
    v_world = world;
}
"""

_GRADIENT_BAR_FRAG = """
#version 330 core
in vec2 v_world;
uniform vec4 u_color;
uniform float u_spawn_x;
uniform float u_fade_width;
out vec4 frag_color;

void main() {
    float dist_from_spawn = u_spawn_x - v_world.x;
    float mask = clamp((u_fade_width - dist_from_spawn) / u_fade_width, 0.0, 1.0);
    frag_color = vec4(u_color.rgb, u_color.a * mask);
}
"""

_PANEL_BATCH_VERT = """
#version 330 core
in vec2 in_pos;
in vec2 in_rect_pos;
in vec2 in_rect_size;
in float in_radius;
in vec4 in_color;
in vec4 in_border_color;
in float in_border_width;
uniform mat4 projection;
out vec2 v_local;
flat out vec2 v_rect_size;
flat out float v_radius;
flat out vec4 v_color;
flat out vec4 v_border_color;
flat out float v_border_width;
void main() {
    vec2 world = in_rect_pos + in_pos * in_rect_size;
    gl_Position = projection * vec4(world, 0.0, 1.0);
    v_local = in_pos * in_rect_size;
    v_rect_size = in_rect_size;
    v_radius = in_radius;
    v_color = in_color;
    v_border_color = in_border_color;
    v_border_width = in_border_width;
}
"""

_PANEL_BATCH_FRAG = """
#version 330 core
in vec2 v_local;
flat in vec2 v_rect_size;
flat in float v_radius;
flat in vec4 v_color;
flat in vec4 v_border_color;
flat in float v_border_width;
out vec4 frag_color;

float rounded_rect_sdf(vec2 p, vec2 half_size, float r) {
    vec2 d = abs(p - half_size) - (half_size - vec2(r));
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - r;
}

void main() {
    float d = rounded_rect_sdf(v_local, v_rect_size * 0.5, v_radius);
    float fill_alpha = 1.0 - smoothstep(-0.5, 0.5, d);
    if (fill_alpha < 0.001) discard;

    float border_mask = 0.0;
    if (v_border_width > 0.001) {
        float border_inner = d + v_border_width;
        border_mask = smoothstep(-0.5, 0.5, border_inner);
    }

    vec4 col = mix(v_color, v_border_color, border_mask);
    col.a *= fill_alpha;
    frag_color = col;
}
"""

_GRADIENT_BAR_BATCH_VERT = """
#version 330 core
in vec2 in_pos;
in vec2 in_rect_pos;
in vec2 in_rect_size;
in vec4 in_color;
in float in_spawn_x;
in float in_fade_width;
uniform mat4 projection;
out vec2 v_world;
flat out vec4 v_color;
flat out float v_spawn_x;
flat out float v_fade_width;
void main() {
    vec2 world = in_rect_pos + in_pos * in_rect_size;
    gl_Position = projection * vec4(world, 0.0, 1.0);
    v_world = world;
    v_color = in_color;
    v_spawn_x = in_spawn_x;
    v_fade_width = in_fade_width;
}
"""

_GRADIENT_BAR_BATCH_FRAG = """
#version 330 core
in vec2 v_world;
flat in vec4 v_color;
flat in float v_spawn_x;
flat in float v_fade_width;
out vec4 frag_color;

void main() {
    float dist_from_spawn = v_spawn_x - v_world.x;
    float mask = clamp((v_fade_width - dist_from_spawn) / v_fade_width, 0.0, 1.0);
    frag_color = vec4(v_color.rgb, v_color.a * mask);
}
"""

MAX_BATCHED_PANELS = 4096
MAX_BATCHED_GRADIENTS = 2048


class PanelRenderer:
    """Immediate-mode rounded-rect panel drawing."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._prog = ctx.program(vertex_shader=_PANEL_VERT, fragment_shader=_PANEL_FRAG)
        self._grad_prog = ctx.program(
            vertex_shader=_GRADIENT_BAR_VERT, fragment_shader=_GRADIENT_BAR_FRAG
        )
        self._batch_prog = ctx.program(
            vertex_shader=_PANEL_BATCH_VERT, fragment_shader=_PANEL_BATCH_FRAG
        )
        self._grad_batch_prog = ctx.program(
            vertex_shader=_GRADIENT_BAR_BATCH_VERT, fragment_shader=_GRADIENT_BAR_BATCH_FRAG
        )
        verts = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype="f4")
        idx = np.array([0, 1, 2, 0, 2, 3], dtype="i4")
        self._vbo = ctx.buffer(verts.tobytes())
        self._ibo = ctx.buffer(idx.tobytes())
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, "2f", "in_pos")],
            index_buffer=self._ibo,
        )
        self._grad_vao = ctx.vertex_array(
            self._grad_prog,
            [(self._vbo, "2f", "in_pos")],
            index_buffer=self._ibo,
        )
        self._batch_instance_buf = ctx.buffer(reserve=MAX_BATCHED_PANELS * 14 * 4)
        self._batch_vao = ctx.vertex_array(
            self._batch_prog,
            [
                (self._vbo, "2f", "in_pos"),
                (
                    self._batch_instance_buf,
                    "2f 2f 1f 4f 4f 1f/i",
                    "in_rect_pos",
                    "in_rect_size",
                    "in_radius",
                    "in_color",
                    "in_border_color",
                    "in_border_width",
                ),
            ],
            index_buffer=self._ibo,
        )
        self._grad_batch_instance_buf = ctx.buffer(reserve=MAX_BATCHED_GRADIENTS * 10 * 4)
        self._grad_batch_vao = ctx.vertex_array(
            self._grad_batch_prog,
            [
                (self._vbo, "2f", "in_pos"),
                (
                    self._grad_batch_instance_buf,
                    "2f 2f 4f 1f 1f/i",
                    "in_rect_pos",
                    "in_rect_size",
                    "in_color",
                    "in_spawn_x",
                    "in_fade_width",
                ),
            ],
            index_buffer=self._ibo,
        )
        self._projection = np.eye(4, dtype="f4")
        self._projection_bytes = self._projection.tobytes()
        self._batching = False
        self._panel_batch_items: list[tuple[float, ...]] = []
        self._gradient_batch_items: list[tuple[float, ...]] = []

    def set_projection(self, w: int, h: int) -> None:
        p = np.eye(4, dtype="f4")
        p[0, 0] = 2.0 / w
        p[1, 1] = -2.0 / h
        p[3, 0] = -1.0
        p[3, 1] = 1.0
        self._projection = p
        self._projection_bytes = p.tobytes()
        self._prog["projection"].write(self._projection_bytes)
        self._grad_prog["projection"].write(self._projection_bytes)
        self._batch_prog["projection"].write(self._projection_bytes)
        self._grad_batch_prog["projection"].write(self._projection_bytes)

    def begin_batch(self) -> None:
        self._batching = True
        self._panel_batch_items.clear()
        self._gradient_batch_items.clear()

    def _flush_panel_batch(self) -> None:
        if not self._panel_batch_items:
            return
        data = np.asarray(self._panel_batch_items, dtype="f4")
        self._batch_instance_buf.write(memoryview(data).cast("B"))
        self._batch_vao.render(moderngl.TRIANGLES, instances=len(data))
        profiler.count("panels.draw.calls", len(data))
        self._panel_batch_items.clear()

    def _flush_gradient_batch(self) -> None:
        if not self._gradient_batch_items:
            return
        data = np.asarray(self._gradient_batch_items, dtype="f4")
        self._grad_batch_instance_buf.write(memoryview(data).cast("B"))
        self._grad_batch_vao.render(moderngl.TRIANGLES, instances=len(data))
        profiler.count("panels.draw_gradient.calls", len(data))
        self._gradient_batch_items.clear()

    def end_batch(self) -> None:
        if not self._batching:
            return
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._batch_prog["projection"].write(self._projection_bytes)
        self._grad_batch_prog["projection"].write(self._projection_bytes)
        self._flush_panel_batch()
        self._flush_gradient_batch()
        self._batching = False

    def draw(self, x: float, y: float, w: float, h: float,
             radius: float = 8.0,
             color: tuple = (0.1, 0.1, 0.15, 0.85),
             border_color: tuple = (0.3, 0.3, 0.4, 1.0),
             border_width: float = 1.0) -> None:
        if self._batching:
            self._panel_batch_items.append((
                x, y, w, h, radius,
                color[0], color[1], color[2], color[3],
                border_color[0], border_color[1], border_color[2], border_color[3],
                border_width,
            ))
            if len(self._panel_batch_items) >= MAX_BATCHED_PANELS:
                self._flush_panel_batch()
            return
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        profiler.count("panels.draw.calls")
        self._prog["u_rect_pos"].value = (x, y)
        self._prog["u_rect_size"].value = (w, h)
        self._prog["u_radius"].value = radius
        self._prog["u_color"].value = color
        self._prog["u_border_color"].value = border_color
        self._prog["u_border_width"].value = border_width
        self._vao.render(moderngl.TRIANGLES)

    def draw_gradient_bar(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        spawn_x: float,
        fade_width: float,
        color: tuple = (0.7, 0.5, 1.0, 0.9),
    ) -> None:
        """Draw a bar with alpha gradient mask: left (alpha 0) to spawn (alpha 1)."""
        if self._batching:
            self._gradient_batch_items.append((
                x, y, w, h,
                color[0], color[1], color[2], color[3],
                spawn_x, fade_width,
            ))
            if len(self._gradient_batch_items) >= MAX_BATCHED_GRADIENTS:
                self._flush_gradient_batch()
            return
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        profiler.count("panels.draw_gradient.calls")
        self._grad_prog["u_rect_pos"].value = (x, y)
        self._grad_prog["u_rect_size"].value = (w, h)
        self._grad_prog["u_color"].value = color
        self._grad_prog["u_spawn_x"].value = spawn_x
        self._grad_prog["u_fade_width"].value = fade_width
        self._grad_vao.render(moderngl.TRIANGLES)


class Button:
    """Clickable UI button with hover feedback."""

    def __init__(self, x: float, y: float, w: float, h: float,
                 label: str, font_size: int = 20):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label
        self.font_size = font_size
        self.hovered = False
        self.color = (0.12, 0.1, 0.18, 0.9)
        self.hover_color = (0.22, 0.15, 0.35, 0.95)
        self.border_color = (0.5, 0.3, 0.7, 1.0)
        self.text_color = (1.0, 1.0, 1.0)

    def hit_test(self, mx: int, my: int) -> bool:
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def draw(self, panels: PanelRenderer, text) -> None:
        col = self.hover_color if self.hovered else self.color
        panels.draw(self.x, self.y, self.w, self.h,
                    radius=6.0, color=col,
                    border_color=self.border_color, border_width=1.5)
        tw, th = text.measure(self.label, self.font_size)
        tx = self.x + (self.w - tw) / 2
        ty = self.y + (self.h - self.font_size) / 2
        text.draw(self.label, tx, ty, self.font_size, color=self.text_color)
