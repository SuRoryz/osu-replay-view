from __future__ import annotations

from collections import deque

import moderngl
import moderngl_window as mglw
import numpy as np
from pyrr import matrix44

from audio import AudioEngine
from audio.hitsounds import HitsoundManager
from cursor.path import CursorPath
from osu_map.beatmap import HitsoundEvent, RenderData
from skins import SKIN_REGISTRY, Skin

OSU_PLAYFIELD_WIDTH = 512
OSU_PLAYFIELD_HEIGHT = 384
PLAYFIELD_CX = OSU_PLAYFIELD_WIDTH / 2
PLAYFIELD_CY = OSU_PLAYFIELD_HEIGHT / 2

PADDING_H = 0.15
PADDING_V = 0.05

PATH_TEX_WIDTH = 1024
APPROACH_SCALE = 4.0
MAX_SLIDER_BALLS = 32
CURSOR_RADIUS = 8.0
TRAIL_MAX = 64
TRAIL_LIFETIME_MS = 180.0

_QUAD_VERTS = np.array([
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0],
], dtype="f4")

_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype="i4")


class OsuRenderer(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1920, 1080)
    title = "osu! replay"
    vsync = False
    samples = 4
    clear_color = (0.05, 0.05, 0.1, 1.0)

    _render_data: RenderData | None = None
    _audio_engine: AudioEngine | None = None
    _hitsound_mgr: HitsoundManager | None = None
    _cursor_path: CursorPath | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._audio = self._audio_engine
        self._hitsounds = self._hitsound_mgr
        self._cpath = self._cursor_path

        self._skins = list(SKIN_REGISTRY)
        self._skin_idx = 0
        self._skin: Skin = self._skins[0]
        self._skin.setup(self.ctx)

        self._compile_programs()

        self.projection = self._build_projection()
        self._sync_uniforms()

        self.n_circle_instances = 0
        self.circle_vao: moderngl.VertexArray | None = None
        self.approach_vao: moderngl.VertexArray | None = None
        self.slider_vao: moderngl.VertexArray | None = None
        self.n_slider_instances = 0
        self.path_tex: moderngl.Texture | None = None
        self.path_tex_width = PATH_TEX_WIDTH

        self.ball_vao: moderngl.VertexArray | None = None
        self._ball_instance_buf: moderngl.Buffer | None = None

        self.cursor_vao: moderngl.VertexArray | None = None
        self._cursor_instance_buf: moderngl.Buffer | None = None
        self._trail: deque[tuple[float, float, float]] = deque(maxlen=TRAIL_MAX)

        self._stored_data: RenderData | None = None
        self._hs_events: list[HitsoundEvent] = []
        self._hs_cursor: int = 0
        self._prev_time_ms: float = -1e9

        if self._render_data is not None:
            self._upload(self._render_data)

        if self._audio is not None:
            self._audio.start()

    # ----------------------------------------------------------- skin
    def _compile_programs(self):
        vert, frag = self._skin.circle_shader_source()
        self.circle_prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)

        vert, frag = self._skin.slider_shader_source()
        self.slider_prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)

        approach_src = self._skin.approach_shader_source()
        if approach_src is not None:
            vert, frag = approach_src
            self.approach_prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        else:
            self.approach_prog = None

        ball_src = self._skin.slider_ball_shader_source()
        if ball_src is not None:
            vert, frag = ball_src
            self.ball_prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        else:
            self.ball_prog = None

        cursor_src = self._skin.cursor_shader_source()
        if cursor_src is not None:
            vert, frag = cursor_src
            self.cursor_prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        else:
            self.cursor_prog = None

    def _sync_uniforms(self):
        proj_bytes = self.projection.astype("f4").tobytes()
        for prog in (self.circle_prog, self.slider_prog):
            prog["projection"].write(proj_bytes)
            prog["preempt"].value = 0.0
            prog["fade_in"].value = 0.0
            prog["current_time"].value = 0.0
        if self.approach_prog is not None:
            self.approach_prog["projection"].write(proj_bytes)
            self.approach_prog["preempt"].value = 0.0
            self.approach_prog["fade_in"].value = 0.0
            self.approach_prog["current_time"].value = 0.0
            self.approach_prog["approach_scale"].value = APPROACH_SCALE
        if self.ball_prog is not None:
            self.ball_prog["projection"].write(proj_bytes)
        if self.cursor_prog is not None:
            self.cursor_prog["projection"].write(proj_bytes)

    def set_skin(self, skin: Skin):
        self._skin.cleanup()
        self._skin = skin
        self._skin.setup(self.ctx)
        self._compile_programs()
        self.projection = self._build_projection()
        self._sync_uniforms()
        if self._stored_data is not None:
            self._upload(self._stored_data)

    def _cycle_skin(self):
        if len(self._skins) < 2:
            return
        self._skin_idx = (self._skin_idx + 1) % len(self._skins)
        self.set_skin(self._skins[self._skin_idx])

    # ----------------------------------------------------------- projection
    def _build_projection(self) -> np.ndarray:
        w, h = self.wnd.buffer_size
        aspect = w / h

        usable_w = 1.0 - 2 * PADDING_H
        usable_h = 1.0 - 2 * PADDING_V

        half_w_need = (OSU_PLAYFIELD_WIDTH / 2) / usable_w
        half_h_need = (OSU_PLAYFIELD_HEIGHT / 2) / usable_h

        if half_w_need / aspect >= half_h_need:
            half_w = half_w_need
            half_h = half_w / aspect
        else:
            half_h = half_h_need
            half_w = half_h * aspect

        return matrix44.create_orthogonal_projection_matrix(
            left=PLAYFIELD_CX - half_w,
            right=PLAYFIELD_CX + half_w,
            bottom=PLAYFIELD_CY - half_h,
            top=PLAYFIELD_CY + half_h,
            near=-1.0,
            far=2.0,
            dtype=np.float32,
        )

    # ----------------------------------------------------------- upload
    def _upload(self, data: RenderData):
        self._stored_data = data
        self._hs_events = data.hitsound_events
        self._hs_cursor = 0

        proj_bytes = self.projection.astype("f4").tobytes()
        for prog in (self.circle_prog, self.slider_prog):
            prog["preempt"].value = data.preempt
            prog["fade_in"].value = data.fade_in
            prog["projection"].write(proj_bytes)
        if self.approach_prog is not None:
            self.approach_prog["preempt"].value = data.preempt
            self.approach_prog["fade_in"].value = data.fade_in
            self.approach_prog["projection"].write(proj_bytes)
        if self.ball_prog is not None:
            self.ball_prog["projection"].write(proj_bytes)

        self._upload_circles(data)
        self._upload_sliders(data)
        self._setup_ball_vao(data)
        self._setup_cursor_vao()

    def _upload_circles(self, data: RenderData):
        n = len(data.circle_positions)
        self.n_circle_instances = n
        if n == 0:
            return

        color = self._skin.combo_colors()[0]

        buf = np.empty((n, 9), dtype="f4")
        buf[:, 0:2] = data.circle_positions
        buf[:, 2] = data.circle_radius
        buf[:, 3:6] = color
        buf[:, 6] = data.circle_start_times
        buf[:, 7] = data.circle_end_times
        buf[:, 8] = data.circle_z

        quad_vbo = self.ctx.buffer(_QUAD_VERTS.tobytes())
        index_buf = self.ctx.buffer(_QUAD_INDICES.tobytes())
        instance_vbo = self.ctx.buffer(buf.tobytes())

        self.circle_vao = self.ctx.vertex_array(
            self.circle_prog,
            [
                (quad_vbo, "2f", "in_vert"),
                (instance_vbo, "2f 1f 3f 1f 1f 1f/i",
                 "in_pos", "in_radius", "in_color", "in_start_time", "in_end_time", "in_z"),
            ],
            index_buffer=index_buf,
        )

        if self.approach_prog is not None:
            self.approach_vao = self.ctx.vertex_array(
                self.approach_prog,
                [
                    (quad_vbo, "2f", "in_vert"),
                    (instance_vbo, "2f 1f 3f 1f 1f 1f/i",
                     "in_pos", "in_radius", "in_color", "in_start_time", "in_end_time", "in_z"),
                ],
                index_buffer=index_buf,
            )

    def _upload_sliders(self, data: RenderData):
        sd = data.slider
        if sd is None or sd.n_sliders == 0:
            self.slider_vao = None
            self.n_slider_instances = 0
            return

        n_points = len(sd.path_points)
        tex_w = PATH_TEX_WIDTH
        tex_h = max(1, (n_points + tex_w - 1) // tex_w)
        padded = np.zeros((tex_h * tex_w, 2), dtype="f4")
        padded[:n_points] = sd.path_points

        if self.path_tex is not None:
            self.path_tex.release()
        self.path_tex = self.ctx.texture(
            (tex_w, tex_h), 2, padded.tobytes(), dtype="f4"
        )
        self.path_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.path_tex_width = tex_w

        n = sd.n_sliders
        color = self._skin.combo_colors()[0]

        buf = np.empty((n, 13), dtype="f4")
        buf[:, 0:2] = sd.bbox_min
        buf[:, 2:4] = sd.bbox_max
        buf[:, 4] = sd.path_starts.astype("f4")
        buf[:, 5] = sd.path_counts.astype("f4")
        buf[:, 6] = data.circle_radius
        buf[:, 7:10] = color
        buf[:, 10] = sd.start_times
        buf[:, 11] = sd.end_times
        buf[:, 12] = sd.z_values

        self.n_slider_instances = n

        quad_vbo = self.ctx.buffer(_QUAD_VERTS.tobytes())
        index_buf = self.ctx.buffer(_QUAD_INDICES.tobytes())
        instance_vbo = self.ctx.buffer(buf.tobytes())

        self.slider_vao = self.ctx.vertex_array(
            self.slider_prog,
            [
                (quad_vbo, "2f", "in_vert"),
                (instance_vbo, "2f 2f 1f 1f 1f 3f 1f 1f 1f/i",
                 "in_bbox_min", "in_bbox_max", "in_path_start", "in_path_count",
                 "in_radius", "in_color", "in_start_time", "in_end_time", "in_z"),
            ],
            index_buffer=index_buf,
        )

    def _setup_ball_vao(self, data: RenderData):
        """Pre-allocate a dynamic instance buffer and VAO for slider balls."""
        if self.ball_prog is None:
            self.ball_vao = None
            return

        # 6 floats per ball: pos(2) radius(1) color(3) z(1) = 7 floats
        self._ball_instance_buf = self.ctx.buffer(
            reserve=MAX_SLIDER_BALLS * 7 * 4
        )
        quad_vbo = self.ctx.buffer(_QUAD_VERTS.tobytes())
        index_buf = self.ctx.buffer(_QUAD_INDICES.tobytes())

        self.ball_vao = self.ctx.vertex_array(
            self.ball_prog,
            [
                (quad_vbo, "2f", "in_vert"),
                (self._ball_instance_buf, "2f 1f 3f 1f/i",
                 "in_pos", "in_radius", "in_color", "in_z"),
            ],
            index_buffer=index_buf,
        )

    def _setup_cursor_vao(self):
        if self.cursor_prog is None:
            self.cursor_vao = None
            return
        # 8 floats per instance: pos(2) radius(1) color_rgba(4) z(1)
        self._cursor_instance_buf = self.ctx.buffer(
            reserve=(TRAIL_MAX + 1) * 8 * 4
        )
        quad_vbo = self.ctx.buffer(_QUAD_VERTS.tobytes())
        index_buf = self.ctx.buffer(_QUAD_INDICES.tobytes())
        self.cursor_vao = self.ctx.vertex_array(
            self.cursor_prog,
            [
                (quad_vbo, "2f", "in_vert"),
                (self._cursor_instance_buf, "2f 1f 4f 1f/i",
                 "in_pos", "in_radius", "in_color", "in_z"),
            ],
            index_buffer=index_buf,
        )

    def _build_cursor_instances(self, current_time_ms: float) -> tuple[np.ndarray, int]:
        """Build instance data for cursor head + trail."""
        if self._cpath is None:
            return np.empty((0, 8), dtype="f4"), 0

        pos = self._cpath.position_at(current_time_ms)
        self._trail.append((current_time_ms, float(pos[0]), float(pos[1])))

        # Gather trail + head
        n = len(self._trail)
        buf = np.empty((n, 8), dtype="f4")

        # Trail (oldest first → draw first, head on top)
        for i, (t, x, y) in enumerate(self._trail):
            age = current_time_ms - t
            fade = max(0.0, 1.0 - age / TRAIL_LIFETIME_MS)
            shrink = 0.3 + 0.7 * fade
            buf[i, 0] = x
            buf[i, 1] = y
            buf[i, 2] = CURSOR_RADIUS * shrink
            buf[i, 3] = 1.0   # r
            buf[i, 4] = 1.0   # g
            buf[i, 5] = 0.85  # b
            buf[i, 6] = fade * 0.6  # alpha
            buf[i, 7] = -0.9  # z (very front)

        # Overwrite last entry as the head (full brightness)
        buf[n - 1, 2] = CURSOR_RADIUS
        buf[n - 1, 3] = 1.0
        buf[n - 1, 4] = 1.0
        buf[n - 1, 5] = 1.0
        buf[n - 1, 6] = 1.0
        buf[n - 1, 7] = -0.95

        return buf, n

    # ----------------------------------------------------------- slider ball
    def _compute_slider_balls(self, current_time_ms: float) -> np.ndarray:
        """Return (N, 7) f32 array of active slider ball instances."""
        data = self._stored_data
        if data is None or data.slider is None or data.slider.n_sliders == 0:
            return np.empty((0, 7), dtype="f4")

        sd = data.slider
        balls = []
        for i in range(sd.n_sliders):
            start = sd.start_times[i]
            end = sd.end_times[i]
            if current_time_ms < start or current_time_ms > end:
                continue

            duration = end - start
            if duration <= 0:
                continue

            repeats = sd.repeat_counts[i]
            progress = (current_time_ms - start) / duration
            total_progress = progress * repeats

            leg = int(total_progress)
            leg_t = total_progress - leg
            if leg >= repeats:
                leg = repeats - 1
                leg_t = 1.0

            if leg % 2 == 1:
                leg_t = 1.0 - leg_t

            path_start = sd.path_starts[i]
            path_count = sd.path_counts[i]

            float_idx = leg_t * (path_count - 1)
            idx = min(int(float_idx), path_count - 2)
            frac = float_idx - idx

            p0 = sd.path_points[path_start + idx]
            p1 = sd.path_points[path_start + idx + 1]
            pos = p0 * (1.0 - frac) + p1 * frac

            z = sd.z_values[i] - 0.003
            balls.append([pos[0], pos[1], data.circle_radius, 1.0, 1.0, 1.0, z])

            if len(balls) >= MAX_SLIDER_BALLS:
                break

        if not balls:
            return np.empty((0, 7), dtype="f4")
        return np.array(balls, dtype="f4")

    # ----------------------------------------------------------- hitsounds
    def _trigger_hitsounds(self, current_time_ms: float) -> None:
        if not self._hs_events or self._hitsounds is None:
            return

        # First call -- position cursor without firing anything
        if self._prev_time_ms < -1e8:
            self._prev_time_ms = current_time_ms
            while (self._hs_cursor < len(self._hs_events)
                   and self._hs_events[self._hs_cursor].time_ms <= current_time_ms):
                self._hs_cursor += 1
            return

        # Backward seek -- rewind cursor
        if current_time_ms < self._prev_time_ms - 100:
            self._hs_cursor = 0
            while (self._hs_cursor < len(self._hs_events)
                   and self._hs_events[self._hs_cursor].time_ms <= current_time_ms):
                self._hs_cursor += 1
            self._prev_time_ms = current_time_ms
            return

        while self._hs_cursor < len(self._hs_events):
            ev = self._hs_events[self._hs_cursor]
            if ev.time_ms > current_time_ms:
                break
            self._hitsounds.play(
                ev.normal_set, ev.addition_set, ev.sound_enum, ev.volume,
            )
            self._hs_cursor += 1

    # ----------------------------------------------------------- render
    def on_render(self, time: float, frametime: float):
        self.ctx.enable(moderngl.BLEND | moderngl.DEPTH_TEST)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.depth_func = '<'

        if self._audio is not None:
            current_time_ms = self._audio.position_ms
        else:
            current_time_ms = time * 1000.0

        self._skin.on_pre_render(self.ctx, current_time_ms, frametime)

        self._trigger_hitsounds(current_time_ms)
        self._prev_time_ms = current_time_ms

        # 1) Sliders (body sits behind same-time circle heads)
        if self.slider_vao is not None and self.n_slider_instances > 0:
            self.path_tex.use(location=0)
            self.slider_prog["u_path_tex"].value = 0
            self.slider_prog["u_path_tex_width"].value = self.path_tex_width
            self.slider_prog["current_time"].value = current_time_ms
            self.slider_vao.render(moderngl.TRIANGLES, instances=self.n_slider_instances)

        # 2) Circle bodies
        if self.circle_vao is not None and self.n_circle_instances > 0:
            self.circle_prog["current_time"].value = current_time_ms
            self.circle_vao.render(moderngl.TRIANGLES, instances=self.n_circle_instances)

        # 3) Slider balls + approach circles -- depth test off so they're on top
        self.ctx.disable(moderngl.DEPTH_TEST)

        if self.ball_vao is not None:
            ball_data = self._compute_slider_balls(current_time_ms)
            n_balls = len(ball_data)
            if n_balls > 0:
                self._ball_instance_buf.write(ball_data.tobytes())
                self.ball_vao.render(moderngl.TRIANGLES, instances=n_balls)

        if self.approach_vao is not None and self.n_circle_instances > 0:
            self.approach_prog["current_time"].value = current_time_ms
            self.approach_vao.render(moderngl.TRIANGLES, instances=self.n_circle_instances)

        # 5) Cursor + trail (always topmost)
        if self.cursor_vao is not None:
            cursor_buf, n_cursor = self._build_cursor_instances(current_time_ms)
            if n_cursor > 0:
                self._cursor_instance_buf.write(cursor_buf.tobytes())
                self.cursor_vao.render(moderngl.TRIANGLES, instances=n_cursor)

        self.ctx.enable(moderngl.DEPTH_TEST)

        self._skin.on_post_render(self.ctx, current_time_ms, frametime)

    # ----------------------------------------------------------- events
    def on_resize(self, width: int, height: int):
        self.projection = self._build_projection()
        proj_bytes = self.projection.astype("f4").tobytes()
        self.circle_prog["projection"].write(proj_bytes)
        self.slider_prog["projection"].write(proj_bytes)
        if self.approach_prog is not None:
            self.approach_prog["projection"].write(proj_bytes)
        if self.ball_prog is not None:
            self.ball_prog["projection"].write(proj_bytes)
        if self.cursor_prog is not None:
            self.cursor_prog["projection"].write(proj_bytes)

    def on_close(self):
        if self._audio is not None:
            self._audio.cleanup()

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS:
            if key == keys.ESCAPE:
                self.wnd.close()
            elif key == keys.TAB:
                self._cycle_skin()
            elif key == keys.SPACE:
                if self._audio is not None:
                    self._audio.toggle_pause()
