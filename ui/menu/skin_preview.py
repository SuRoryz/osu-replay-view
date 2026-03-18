from __future__ import annotations

import moderngl
import numpy as np

from skins import SKIN_REGISTRY
from ui.menu.layout import Rect

_QUAD_VERTS = np.array([
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
], dtype="f4")

_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype="i4")
_PATH_TEX_WIDTH = 64


def _screen_projection(width: int, height: int) -> np.ndarray:
    projection = np.eye(4, dtype="f4")
    projection[0, 0] = 2.0 / max(1.0, float(width))
    projection[1, 1] = -2.0 / max(1.0, float(height))
    projection[2, 2] = 1.0
    projection[3, 0] = -1.0
    projection[3, 1] = 1.0
    return projection


class SkinPreviewRenderer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._skin = SKIN_REGISTRY[0]
        self._quad_vbo = ctx.buffer(_QUAD_VERTS.tobytes())
        self._quad_ibo = ctx.buffer(_QUAD_INDICES.tobytes())
        self._circle_instance_vbo = ctx.buffer(reserve=2 * 10 * 4)
        self._slider_instance_vbo = ctx.buffer(reserve=13 * 4)
        self._ball_instance_vbo = ctx.buffer(reserve=7 * 4)
        self._path_tex = None
        self._path_tex_width = _PATH_TEX_WIDTH
        self._compile_programs()

    def _compile_programs(self) -> None:
        circle_vert, circle_frag = self._skin.circle_shader_source()
        self.circle_prog = self.ctx.program(vertex_shader=circle_vert, fragment_shader=circle_frag)
        self.circle_vao = self.ctx.vertex_array(
            self.circle_prog,
            [
                (self._quad_vbo, "2f", "in_vert"),
                (
                    self._circle_instance_vbo,
                    "2f 1f 3f 1f 1f 1f 1f/i",
                    "in_pos",
                    "in_radius",
                    "in_color",
                    "in_start_time",
                    "in_end_time",
                    "in_z",
                    "in_is_slider_head",
                ),
            ],
            index_buffer=self._quad_ibo,
        )

        approach_src = self._skin.approach_shader_source()
        self.approach_prog = None
        self.approach_vao = None
        if approach_src is not None:
            approach_vert, approach_frag = approach_src
            self.approach_prog = self.ctx.program(vertex_shader=approach_vert, fragment_shader=approach_frag)
            self.approach_vao = self.ctx.vertex_array(
                self.approach_prog,
                [
                    (self._quad_vbo, "2f", "in_vert"),
                    (
                        self._circle_instance_vbo,
                        "2f 1f 3f 1f 1f 1f 1f/i",
                        "in_pos",
                        "in_radius",
                        "in_color",
                        "in_start_time",
                        "in_end_time",
                        "in_z",
                        "in_is_slider_head",
                    ),
                ],
                index_buffer=self._quad_ibo,
            )

        slider_vert, slider_frag = self._skin.slider_shader_source()
        self.slider_prog = self.ctx.program(vertex_shader=slider_vert, fragment_shader=slider_frag)
        self.slider_vao = self.ctx.vertex_array(
            self.slider_prog,
            [
                (self._quad_vbo, "2f", "in_vert"),
                (
                    self._slider_instance_vbo,
                    "2f 2f 1f 1f 1f 3f 1f 1f 1f/i",
                    "in_bbox_min",
                    "in_bbox_max",
                    "in_path_start",
                    "in_path_count",
                    "in_radius",
                    "in_color",
                    "in_start_time",
                    "in_end_time",
                    "in_z",
                ),
            ],
            index_buffer=self._quad_ibo,
        )

        ball_src = self._skin.slider_ball_shader_source()
        self.ball_prog = None
        self.ball_vao = None
        if ball_src is not None:
            ball_vert, ball_frag = ball_src
            self.ball_prog = self.ctx.program(vertex_shader=ball_vert, fragment_shader=ball_frag)
            self.ball_vao = self.ctx.vertex_array(
                self.ball_prog,
                [
                    (self._quad_vbo, "2f", "in_vert"),
                    (
                        self._ball_instance_vbo,
                        "2f 1f 3f 1f/i",
                        "in_pos",
                        "in_radius",
                        "in_color",
                        "in_z",
                    ),
                ],
                index_buffer=self._quad_ibo,
            )

    def _sync_programs(self, width: int, height: int) -> None:
        proj_bytes = _screen_projection(width, height).tobytes()
        current_time = 1400.0
        preempt = 1000.0
        fade_in = 400.0

        for prog in (self.circle_prog, self.slider_prog):
            prog["projection"].write(proj_bytes)
            prog["current_time"].value = current_time
            prog["preempt"].value = preempt
            prog["fade_in"].value = fade_in
            if "u_hidden" in prog:
                prog["u_hidden"].value = 0
        if self.approach_prog is not None:
            self.approach_prog["projection"].write(proj_bytes)
            self.approach_prog["current_time"].value = current_time
            self.approach_prog["preempt"].value = preempt
            self.approach_prog["fade_in"].value = fade_in
            self.approach_prog["approach_scale"].value = 3.2
            if "u_hidden" in self.approach_prog:
                self.approach_prog["u_hidden"].value = 0
        if self.ball_prog is not None:
            self.ball_prog["projection"].write(proj_bytes)
        self._skin.sync_object_uniforms(
            circle_prog=self.circle_prog,
            approach_prog=self.approach_prog,
            slider_prog=self.slider_prog,
            ball_prog=self.ball_prog,
        )

    def _build_path(self, rect: Rect) -> np.ndarray:
        x0 = rect.x + rect.w * 0.18
        x1 = rect.x + rect.w * 0.48
        x2 = rect.x + rect.w * 0.80
        y0 = rect.y + rect.h * 0.72
        y1 = rect.y + rect.h * 0.48
        y2 = rect.y + rect.h * 0.68
        points = []
        for t in np.linspace(0.0, 1.0, 160):
            omt = 1.0 - t
            px = omt * omt * x0 + 2.0 * omt * t * x1 + t * t * x2
            py = omt * omt * y0 + 2.0 * omt * t * y1 + t * t * y2
            points.append((px, py))
        return np.asarray(points, dtype="f4")

    def _ensure_path_texture(self, path_points: np.ndarray) -> None:
        tex_h = max(1, (len(path_points) + self._path_tex_width - 1) // self._path_tex_width)
        padded = np.zeros((tex_h * self._path_tex_width, 2), dtype="f4")
        padded[: len(path_points)] = path_points
        if self._path_tex is not None:
            self._path_tex.release()
        self._path_tex = self.ctx.texture(
            (self._path_tex_width, tex_h),
            2,
            padded.tobytes(),
            dtype="f4",
        )
        self._path_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._path_tex.repeat_x = False
        self._path_tex.repeat_y = False

    def draw(
        self,
        rect: Rect,
        *,
        window_width: int,
        window_height: int,
        clip_rect: Rect | None = None,
    ) -> None:
        if rect.w < 4.0 or rect.h < 4.0:
            return

        self._sync_programs(window_width, window_height)
        path_points = self._build_path(rect)
        self._ensure_path_texture(path_points)

        circle_radius = min(rect.w, rect.h) * 0.14
        slider_radius = circle_radius * 0.92
        circle_center = (rect.x + rect.w * 0.72, rect.y + rect.h * 0.36)
        slider_head_pos = path_points[0]
        slider_ball_pos = path_points[int(len(path_points) * 0.62)]

        circle_data = np.array(
            [
                [
                    circle_center[0],
                    circle_center[1],
                    circle_radius,
                    *self._skin.circle_fill_color(),
                    1800.0,
                    2400.0,
                    -0.20,
                    0.0,
                ],
                [
                    float(slider_head_pos[0]),
                    float(slider_head_pos[1]),
                    slider_radius,
                    *self._skin.circle_fill_color(slider_head=True),
                    2000.0,
                    2550.0,
                    -0.10,
                    1.0,
                ],
            ],
            dtype="f4",
        )
        self._circle_instance_vbo.write(circle_data.tobytes())

        bbox_min = path_points.min(axis=0) - slider_radius - 8.0
        bbox_max = path_points.max(axis=0) + slider_radius + 8.0
        slider_data = np.array(
            [[
                bbox_min[0],
                bbox_min[1],
                bbox_max[0],
                bbox_max[1],
                0.0,
                float(len(path_points)),
                slider_radius,
                *self._skin.slider_fill_color(),
                2000.0,
                2550.0,
                -0.30,
            ]],
            dtype="f4",
        )
        self._slider_instance_vbo.write(slider_data.tobytes())

        if self.ball_prog is not None:
            ball_data = np.array(
                [[
                    float(slider_ball_pos[0]),
                    float(slider_ball_pos[1]),
                    slider_radius,
                    *self._skin.slider_ball_fill_color(),
                    -0.18,
                ]],
                dtype="f4",
            )
            self._ball_instance_vbo.write(ball_data.tobytes())

        visible_rect = rect
        if clip_rect is not None:
            left = max(rect.x, clip_rect.x)
            top = max(rect.y, clip_rect.y)
            right = min(rect.right, clip_rect.right)
            bottom = min(rect.bottom, clip_rect.bottom)
            if right - left < 2.0 or bottom - top < 2.0:
                return
            visible_rect = Rect(left, top, right - left, bottom - top)

        initial_scissor = self.ctx.scissor
        self.ctx.scissor = (
            max(0, int(visible_rect.x)),
            max(0, int(window_height - (visible_rect.y + visible_rect.h))),
            max(1, int(visible_rect.w)),
            max(1, int(visible_rect.h)),
        )
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self._path_tex.use(location=0)
        self.slider_prog["u_path_tex"].value = 0
        self.slider_prog["u_path_tex_width"].value = self._path_tex_width
        self.slider_vao.render(moderngl.TRIANGLES, instances=1)
        self.circle_vao.render(moderngl.TRIANGLES, instances=2)
        if self.ball_vao is not None:
            self.ball_vao.render(moderngl.TRIANGLES, instances=1)
        if self.approach_vao is not None:
            self.approach_vao.render(moderngl.TRIANGLES, instances=1)
        self.ctx.scissor = initial_scissor
