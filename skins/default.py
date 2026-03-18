"""Default skin -- current visuals with SDF slider rendering and bloom."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import moderngl

from profiling import profiler
from skins.base import Skin


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _coerce_rgb(value, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return tuple(_clamp01(channel) for channel in value)
        except (TypeError, ValueError):
            pass
    return default


@dataclass(slots=True)
class DefaultSkinVisualSettings:
    circle_fill_color: tuple[float, float, float]
    circle_fill_opacity: float
    circle_border_color: tuple[float, float, float]
    circle_border_width: float
    circle_bloom: float
    circle_bloom_color: tuple[float, float, float]
    slider_use_circle_head: bool
    slider_head_fill_color: tuple[float, float, float]
    slider_head_fill_opacity: float
    slider_head_border_color: tuple[float, float, float]
    slider_head_border_width: float
    slider_head_bloom: float
    slider_head_bloom_color: tuple[float, float, float]
    slider_path_fill_color: tuple[float, float, float]
    slider_path_fill_opacity: float
    slider_path_border_color: tuple[float, float, float]
    slider_path_border_width: float
    slider_ball_fill_color: tuple[float, float, float]
    slider_ball_fill_opacity: float
    slider_ball_border_color: tuple[float, float, float]
    slider_ball_border_width: float
    slider_ball_bloom: float
    slider_ball_bloom_color: tuple[float, float, float]
    cursor_color: tuple[float, float, float]
    cursor_size: float

    def normalized(self) -> "DefaultSkinVisualSettings":
        return DefaultSkinVisualSettings(
            circle_fill_color=_coerce_rgb(self.circle_fill_color, DEFAULT_CIRCLE_FILL_COLOR),
            circle_fill_opacity=_clamp01(self.circle_fill_opacity),
            circle_border_color=_coerce_rgb(self.circle_border_color, (1.0, 1.0, 1.0)),
            circle_border_width=_clamp01(self.circle_border_width),
            circle_bloom=_clamp01(self.circle_bloom),
            circle_bloom_color=_coerce_rgb(self.circle_bloom_color, DEFAULT_CIRCLE_FILL_COLOR),
            slider_use_circle_head=bool(self.slider_use_circle_head),
            slider_head_fill_color=_coerce_rgb(self.slider_head_fill_color, DEFAULT_CIRCLE_FILL_COLOR),
            slider_head_fill_opacity=_clamp01(self.slider_head_fill_opacity),
            slider_head_border_color=_coerce_rgb(self.slider_head_border_color, (1.0, 1.0, 1.0)),
            slider_head_border_width=_clamp01(self.slider_head_border_width),
            slider_head_bloom=_clamp01(self.slider_head_bloom),
            slider_head_bloom_color=_coerce_rgb(self.slider_head_bloom_color, DEFAULT_CIRCLE_FILL_COLOR),
            slider_path_fill_color=_coerce_rgb(self.slider_path_fill_color, DEFAULT_SLIDER_FILL_COLOR),
            slider_path_fill_opacity=_clamp01(self.slider_path_fill_opacity),
            slider_path_border_color=_coerce_rgb(self.slider_path_border_color, DEFAULT_SLIDER_BORDER_COLOR),
            slider_path_border_width=_clamp01(self.slider_path_border_width),
            slider_ball_fill_color=_coerce_rgb(self.slider_ball_fill_color, DEFAULT_CIRCLE_FILL_COLOR),
            slider_ball_fill_opacity=_clamp01(self.slider_ball_fill_opacity),
            slider_ball_border_color=_coerce_rgb(self.slider_ball_border_color, (1.0, 1.0, 1.0)),
            slider_ball_border_width=_clamp01(self.slider_ball_border_width),
            slider_ball_bloom=_clamp01(self.slider_ball_bloom),
            slider_ball_bloom_color=_coerce_rgb(self.slider_ball_bloom_color, DEFAULT_CIRCLE_FILL_COLOR),
            cursor_color=_coerce_rgb(self.cursor_color, DEFAULT_CURSOR_COLOR),
            cursor_size=_clamp01(self.cursor_size),
        )


DEFAULT_CIRCLE_FILL_COLOR = (0.55, 0.75, 1.0)
DEFAULT_CIRCLE_BORDER_COLOR = (1.0, 1.0, 1.0)
DEFAULT_SLIDER_FILL_COLOR = (0.12, 0.1525, 0.22)
DEFAULT_SLIDER_BORDER_COLOR = (0.6175, 0.6875, 0.775)
DEFAULT_CURSOR_COLOR = (1.0, 1.0, 1.0)
DEFAULT_CIRCLE_BORDER_WIDTH = 0.08
DEFAULT_SLIDER_PATH_BORDER_WIDTH = 0.46


def make_default_skin_visual_settings() -> DefaultSkinVisualSettings:
    return DefaultSkinVisualSettings(
        circle_fill_color=DEFAULT_CIRCLE_FILL_COLOR,
        circle_fill_opacity=1.0,
        circle_border_color=DEFAULT_CIRCLE_BORDER_COLOR,
        circle_border_width=DEFAULT_CIRCLE_BORDER_WIDTH,
        circle_bloom=1.0,
        circle_bloom_color=DEFAULT_CIRCLE_FILL_COLOR,
        slider_use_circle_head=True,
        slider_head_fill_color=DEFAULT_CIRCLE_FILL_COLOR,
        slider_head_fill_opacity=1.0,
        slider_head_border_color=DEFAULT_CIRCLE_BORDER_COLOR,
        slider_head_border_width=DEFAULT_CIRCLE_BORDER_WIDTH,
        slider_head_bloom=1.0,
        slider_head_bloom_color=DEFAULT_CIRCLE_FILL_COLOR,
        slider_path_fill_color=DEFAULT_SLIDER_FILL_COLOR,
        slider_path_fill_opacity=0.65,
        slider_path_border_color=DEFAULT_SLIDER_BORDER_COLOR,
        slider_path_border_width=DEFAULT_SLIDER_PATH_BORDER_WIDTH,
        slider_ball_fill_color=DEFAULT_CIRCLE_FILL_COLOR,
        slider_ball_fill_opacity=1.0,
        slider_ball_border_color=DEFAULT_CIRCLE_BORDER_COLOR,
        slider_ball_border_width=DEFAULT_CIRCLE_BORDER_WIDTH,
        slider_ball_bloom=1.0,
        slider_ball_bloom_color=DEFAULT_CIRCLE_FILL_COLOR,
        cursor_color=DEFAULT_CURSOR_COLOR,
        cursor_size=0.5,
    )

# ---------------------------------------------------------------------------
# Circle shaders (body only -- approach ring is a separate pass)
# ---------------------------------------------------------------------------

_CIRCLE_VERT = """
#version 330 core

in vec2  in_vert;
in vec2  in_pos;
in float in_radius;
in vec3  in_color;
in float in_is_slider_head;
in float in_start_time;
in float in_end_time;
in float in_z;

uniform mat4  projection;
uniform float current_time;
uniform float preempt;
uniform float fade_in;
uniform int   u_hidden;

out vec2  v_uv;
out vec3  v_color;
out float v_alpha;
flat out float v_is_slider_head;

void main() {
    float appear_time = in_start_time - preempt;
    float t = current_time - appear_time;
    float alpha_in  = clamp(t / fade_in, 0.0, 1.0);
    float alpha_out = 1.0 - clamp((current_time - in_end_time) / 150.0, 0.0, 1.0);

    if (u_hidden == 1) {
        float hd_start = appear_time + fade_in;
        float hd_dur   = preempt * 0.3;
        float hd_fade  = 1.0 - clamp((current_time - hd_start) / hd_dur, 0.0, 1.0);
        alpha_out = min(alpha_out, hd_fade);
    }

    v_alpha = alpha_in * alpha_out;

    if (v_alpha < 0.001) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        return;
    }

    vec2 world_pos = in_pos + in_vert * in_radius;
    gl_Position = projection * vec4(world_pos, in_z, 1.0);

    v_uv    = in_vert;
    v_color = in_color;
    v_is_slider_head = in_is_slider_head;
}
"""

_CIRCLE_FRAG = """
#version 330 core

in vec2  v_uv;
in vec3  v_color;
in float v_alpha;
flat in float v_is_slider_head;

uniform vec3  u_circle_border_color;
uniform float u_circle_fill_opacity;
uniform float u_circle_border_width;
uniform float u_circle_bloom;
uniform vec3  u_circle_bloom_color;
uniform vec3  u_slider_head_border_color;
uniform float u_slider_head_fill_opacity;
uniform float u_slider_head_border_width;
uniform float u_slider_head_bloom;
uniform vec3  u_slider_head_bloom_color;

out vec4 frag_color;

void main() {
    float dist = length(v_uv);
    float outer = 1.0 - smoothstep(0.96, 1.0, dist);
    float alpha = outer * v_alpha;
    if (alpha < 0.01) discard;

    // Subtle gradient: small enough that overlap artifacts are negligible
    float grad = smoothstep(0.0, 0.95, dist);
    float bloom = mix(u_circle_bloom, u_slider_head_bloom, v_is_slider_head);
    vec3 border_color = mix(u_circle_border_color, u_slider_head_border_color, v_is_slider_head);
    float fill_opacity = mix(u_circle_fill_opacity, u_slider_head_fill_opacity, v_is_slider_head);
    float border_width = mix(u_circle_border_width, u_slider_head_border_width, v_is_slider_head);
    vec3 bloom_color = mix(u_circle_bloom_color, u_slider_head_bloom_color, v_is_slider_head);
    vec3 bloom_top = max(v_color * 1.08, bloom_color);
    vec3 body = mix(v_color, mix(bloom_top, v_color * 0.88, grad), bloom);

    float border_outer = 0.96;
    float border_enabled = smoothstep(0.001, 0.03, border_width);
    float border_inner = max(0.05, border_outer - (0.02 + border_width * 0.16));
    float ring = smoothstep(border_inner - 0.01, border_inner + 0.01, dist)
        * (1.0 - smoothstep(border_outer - 0.01, border_outer + 0.01, dist));
    ring *= border_enabled;
    vec3 col = mix(body, border_color, ring);
    float out_alpha = mix(alpha * fill_opacity, alpha, ring);
    frag_color = vec4(col * out_alpha, out_alpha);
}
"""

# ---------------------------------------------------------------------------
# Approach circle shaders (separate pass, rendered on top of everything)
# ---------------------------------------------------------------------------

_APPROACH_VERT = """
#version 330 core

in vec2  in_vert;
in vec2  in_pos;
in float in_radius;
in vec3  in_color;
in float in_is_slider_head;
in float in_start_time;
in float in_end_time;
in float in_z;

uniform mat4  projection;
uniform float current_time;
uniform float preempt;
uniform float fade_in;
uniform float approach_scale;
uniform int   u_hidden;

out vec2  v_local;
out float v_alpha;
out float v_approach_r;
out vec3  v_color;
flat out float v_is_slider_head;

void main() {
    if (u_hidden == 1) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        return;
    }

    float appear_time = in_start_time - preempt;
    float t = current_time - appear_time;
    float progress = clamp(t / preempt, 0.0, 1.0);
    float alpha_in = clamp(t / fade_in, 0.0, 1.0);

    float past_end = step(in_end_time + 50.0, current_time);
    if (alpha_in < 0.001 || progress > 0.999 || past_end > 0.5) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        return;
    }

    // Fade out as the ring nears the circle edge
    float end_fade = 1.0 - smoothstep(0.85, 0.99, progress);

    float current_r = in_radius * mix(approach_scale, 1.0, progress);
    float quad_r = current_r + 3.0;

    vec2 world_pos = in_pos + in_vert * quad_r;
    gl_Position = projection * vec4(world_pos, in_z, 1.0);

    v_local      = in_vert * quad_r;
    v_alpha      = alpha_in * end_fade;
    v_approach_r = current_r;
    v_color      = in_color;
    v_is_slider_head = in_is_slider_head;
}
"""

_APPROACH_FRAG = """
#version 330 core

in vec2  v_local;
in float v_alpha;
in float v_approach_r;
in vec3  v_color;
flat in float v_is_slider_head;

uniform vec3  u_circle_border_color;
uniform float u_circle_border_width;
uniform float u_circle_bloom;
uniform vec3  u_circle_bloom_color;
uniform vec3  u_slider_head_border_color;
uniform float u_slider_head_border_width;
uniform float u_slider_head_bloom;
uniform vec3  u_slider_head_bloom_color;

out vec4 frag_color;

void main() {
    float dist = length(v_local);
    float d = abs(dist - v_approach_r);

    // Clean ring only, no glow (avoids brightness pop on disappear)
    float ring = 1.0 - smoothstep(0.6, 1.8, d);

    float alpha = ring * v_alpha;
    if (alpha < 0.005) discard;

    float bloom = mix(u_circle_bloom, u_slider_head_bloom, v_is_slider_head);
    vec3 bloom_color = mix(u_circle_bloom_color, u_slider_head_bloom_color, v_is_slider_head);
    float edge_tint = clamp(0.22 + 0.32 * bloom, 0.0, 1.0);
    vec3 col = mix(v_color, bloom_color, edge_tint);
    frag_color = vec4(col * alpha, alpha);
}
"""

# ---------------------------------------------------------------------------
# Spinner shaders
# ---------------------------------------------------------------------------

_SPINNER_VERT = """
#version 330 core

in vec2  in_vert;
in vec2  in_pos;
in float in_radius;
in vec3  in_color;
in float in_start_time;
in float in_end_time;
in float in_z;

uniform mat4  projection;
uniform float current_time;
uniform float preempt;
uniform float fade_in;
uniform int   u_hidden;

out vec2  v_local;
out vec3  v_color;
out float v_alpha;
out float v_progress;
out float v_urgency;
out float v_radius;

void main() {
    float appear_time = in_start_time - preempt;
    float t = current_time - appear_time;
    float alpha_in = clamp(t / fade_in, 0.0, 1.0);
    float alpha_out = 1.0 - clamp((current_time - in_end_time) / 180.0, 0.0, 1.0);

    if (u_hidden == 1) {
        float hd_start = appear_time + fade_in;
        float hd_dur = preempt * 0.3;
        float hd_fade = 1.0 - clamp((current_time - hd_start) / hd_dur, 0.0, 1.0);
        alpha_out = min(alpha_out, hd_fade);
    }

    v_alpha = alpha_in * alpha_out;
    if (v_alpha < 0.001) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        return;
    }

    vec2 world_pos = in_pos + in_vert * in_radius;
    gl_Position = projection * vec4(world_pos, in_z, 1.0);

    float duration = max(in_end_time - in_start_time, 1.0);
    v_progress = clamp((current_time - in_start_time) / duration, 0.0, 1.0);
    v_urgency = smoothstep(0.72, 1.0, v_progress);
    v_local = in_vert * in_radius;
    v_color = in_color;
    v_radius = in_radius;
}
"""

_SPINNER_FRAG = """
#version 330 core

in vec2  v_local;
in vec3  v_color;
in float v_alpha;
in float v_progress;
in float v_urgency;
in float v_radius;

out vec4 frag_color;

void main() {
    float dist_px = length(v_local);
    float angle = atan(v_local.y, v_local.x);

    // Match gameplay cursor radius (4.5px) -> spinner center is 2x cursor size.
    float center_r = 9.0;
    float center = 1.0 - smoothstep(center_r - 1.0, center_r + 1.0, dist_px);

    // Keep a sweep tint, but make the whole ring shrink inward over time.
    float sweep_angle = (1.0 - v_progress) * 6.28318530718 - 3.14159265359;
    float side_delta = abs(mod(angle - sweep_angle + 3.14159265359, 6.28318530718) - 3.14159265359);
    float side_taper = 1.0 - smoothstep(0.0, 1.8, side_delta);
    float ring_half_width = 1.35;

    // Reuse the approach-circle style profile, but animate the radius inward.
    float ring_r_outer = max(v_radius - 2.0, 1.0);
    float ring_r_inner = center_r + 10.0;
    float ring_r = mix(ring_r_outer, ring_r_inner, v_progress);
    float ring_d = abs(dist_px - ring_r);
    float ring = 1.0 - smoothstep(ring_half_width * 0.35, ring_half_width, ring_d);

    float alpha = max(center, ring) * v_alpha;
    if (alpha < 0.004) discard;

    vec3 ring_col = mix(v_color, vec3(1.0), 0.35);
    vec3 sweep_col = mix(vec3(1.0), vec3(1.0, 0.36, 0.32), v_urgency * 0.65);
    ring_col = mix(ring_col, sweep_col, side_taper * (0.45 + 0.25 * v_urgency));
    vec3 center_col = v_color;
    vec3 col = center_col * center + ring_col * ring;

    frag_color = vec4(col * alpha, alpha);
}
"""

# ---------------------------------------------------------------------------
# Slider shaders (SDF in fragment, with vertex culling + fast early exit)
# ---------------------------------------------------------------------------

_SLIDER_VERT = """
#version 330 core

in vec2 in_vert;

in vec2  in_bbox_min;
in vec2  in_bbox_max;
in float in_path_start;
in float in_path_count;
in float in_radius;
in vec3  in_color;
in float in_start_time;
in float in_end_time;
in float in_z;

uniform mat4  projection;
uniform float current_time;
uniform float preempt;
uniform float fade_in;
uniform int   u_hidden;

out vec2  v_world;
out vec3  v_color;
out float v_alpha;
flat out int   v_path_start;
flat out int   v_path_count;
flat out float v_radius;

void main() {
    float appear_time = in_start_time - preempt;
    float t = current_time - appear_time;
    float alpha_in  = clamp(t / fade_in, 0.0, 1.0);
    float alpha_out = 1.0 - clamp((current_time - in_end_time) / 50.0, 0.0, 1.0);

    if (u_hidden == 1) {
        float hd_start = appear_time + fade_in;
        float hd_dur   = preempt * 0.3;
        float hd_fade  = 1.0 - clamp((current_time - hd_start) / hd_dur, 0.0, 1.0);
        alpha_out = min(alpha_out, hd_fade);
    }

    v_alpha = alpha_in * alpha_out;

    if (v_alpha < 0.001) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        return;
    }

    vec2 world = mix(in_bbox_min, in_bbox_max, in_vert * 0.5 + 0.5);
    gl_Position = projection * vec4(world, in_z, 1.0);

    v_world      = world;
    v_color      = in_color;
    v_path_start = int(in_path_start);
    v_path_count = int(in_path_count);
    v_radius     = in_radius;
}
"""

_SLIDER_FRAG = """
#version 330 core

in vec2  v_world;
in vec3  v_color;
in float v_alpha;
flat in int   v_path_start;
flat in int   v_path_count;
flat in float v_radius;

uniform sampler2D u_path_tex;
uniform int       u_path_tex_width;
uniform vec3      u_slider_fill_color;
uniform float     u_slider_fill_opacity;
uniform vec3      u_slider_border_color;
uniform float     u_slider_border_width;

out vec4 frag_color;

vec2 fetch_point(int idx) {
    return texelFetch(u_path_tex,
                      ivec2(idx % u_path_tex_width, idx / u_path_tex_width),
                      0).rg;
}

float dist2_to_segment(vec2 p, vec2 a, vec2 b) {
    vec2  ab  = b - a;
    float ab2 = dot(ab, ab);
    vec2 ap = p - a;
    if (ab2 < 1e-10) return dot(ap, ap);
    float t = clamp(dot(p - a, ab) / ab2, 0.0, 1.0);
    vec2 delta = p - (a + ab * t);
    return dot(delta, delta);
}

void main() {
    if (v_alpha < 0.001) discard;

    float sdf2 = 1e18;
    vec2 a = fetch_point(v_path_start);
    for (int i = 1; i < v_path_count; i++) {
        vec2 b = fetch_point(v_path_start + i);
        sdf2 = min(sdf2, dist2_to_segment(v_world, a, b));
        a = b;
    }
    float sdf = sqrt(max(sdf2, 0.0));

    float mask = 1.0 - smoothstep(v_radius - 1.0, v_radius + 0.5, sdf);
    if (mask < 0.001) discard;

    float border_enabled = smoothstep(0.001, 0.03, u_slider_border_width);
    float border_span = mix(1.2, 6.4, u_slider_border_width);
    float border_in  = smoothstep(v_radius - (border_span + 1.5), v_radius - border_span, sdf);
    float border_out = 1.0 - smoothstep(v_radius - 0.8, v_radius + 0.5, sdf);
    float border = border_in * border_out * border_enabled;

    float rim = smoothstep(v_radius * 0.5, v_radius - 5.0, sdf);
    vec3  fill_col = mix(v_color, u_slider_fill_color, 0.82);
    vec3  border_tint = mix(v_color, u_slider_border_color, 0.82);
    vec3  body_col = mix(fill_col, fill_col * 0.78, rim * 0.55);
    vec3  border_col = border_tint;
    vec3  col = mix(body_col, border_col, border);

    float body_a = u_slider_fill_opacity;
    float alpha = mix(body_a, 1.0, border) * mask * v_alpha;

    frag_color = vec4(col * alpha, alpha);
}
"""


# ---------------------------------------------------------------------------
# Slider ball shaders (always-visible circle following the slider path)
# ---------------------------------------------------------------------------

_BALL_VERT = """
#version 330 core

in vec2  in_vert;
in vec2  in_pos;
in float in_radius;
in vec3  in_color;
in float in_z;

uniform mat4 projection;

out vec2  v_uv;
out vec3  v_color;

void main() {
    vec2 world_pos = in_pos + in_vert * in_radius;
    gl_Position = projection * vec4(world_pos, in_z, 1.0);
    v_uv    = in_vert;
    v_color = in_color;
}
"""

_BALL_FRAG = """
#version 330 core

in vec2  v_uv;
in vec3  v_color;

uniform vec3  u_slider_ball_border_color;
uniform float u_slider_ball_fill_opacity;
uniform float u_slider_ball_border_width;
uniform float u_slider_ball_bloom;
uniform vec3  u_slider_ball_bloom_color;

out vec4 frag_color;

void main() {
    float dist = length(v_uv);
    float edge = 1.0 - smoothstep(0.92, 1.0, dist);
    if (edge < 0.001) discard;

    float grad = smoothstep(0.0, 0.88, dist);
    vec3 bloom_top = max(v_color * 1.3, u_slider_ball_bloom_color);
    vec3 body = mix(v_color, mix(bloom_top, v_color * 0.75, grad), u_slider_ball_bloom);

    float border_outer = 0.93;
    float border_enabled = smoothstep(0.001, 0.03, u_slider_ball_border_width);
    float border_inner = max(0.05, border_outer - (0.02 + u_slider_ball_border_width * 0.16));
    float ring = smoothstep(border_inner - 0.01, border_inner + 0.01, dist)
        * (1.0 - smoothstep(border_outer - 0.01, border_outer + 0.01, dist));
    ring *= border_enabled;
    vec3 col = mix(body, u_slider_ball_border_color, ring);
    float out_alpha = mix(edge * u_slider_ball_fill_opacity, edge, ring);
    frag_color = vec4(col * out_alpha, out_alpha);
}
"""


# ---------------------------------------------------------------------------
# Cursor + trail shaders (per-instance alpha for fading trail)
# ---------------------------------------------------------------------------

_CURSOR_VERT = """
#version 330 core

in vec2  in_vert;
in vec2  in_pos;
in float in_radius;
in vec4  in_color;   // rgba -- trail uses fading alpha
in float in_z;

uniform mat4 projection;

out vec2  v_uv;
out vec4  v_color;

void main() {
    vec2 world_pos = in_pos + in_vert * in_radius;
    gl_Position = projection * vec4(world_pos, in_z, 1.0);
    v_uv    = in_vert;
    v_color = in_color;
}
"""

_CURSOR_FRAG = """
#version 330 core

in vec2  v_uv;
in vec4  v_color;

out vec4 frag_color;

void main() {
    float dist = length(v_uv);
    float edge = 1.0 - smoothstep(0.85, 1.0, dist);
    if (edge < 0.001) discard;
    float a = edge * v_color.a;
    frag_color = vec4(v_color.rgb * a, a);
}
"""

# ---------------------------------------------------------------------------
# Cursor trail ribbon shaders
# ---------------------------------------------------------------------------

_TRAIL_VERT = """
#version 330 core
uniform mat4 projection;
uniform sampler2D u_trail_points;
uniform int u_point_count;
uniform int u_curve_samples;
uniform float u_radius;
uniform float u_lifetime_ms;
uniform float u_tail_taper_fraction;
uniform float u_alpha_scale;
uniform float u_current_time_ms;

out vec2 v_world;
flat out vec2 v_seg_a;
flat out vec2 v_seg_b;
flat out float v_alpha_a;
flat out float v_alpha_b;
flat out float v_radius_a;
flat out float v_radius_b;

vec4 fetch_point(int idx) {
    idx = clamp(idx, 0, max(u_point_count - 1, 0));
    return texelFetch(u_trail_points, ivec2(idx, 0), 0);
}

vec2 safe_dir(vec2 a, vec2 b, vec2 fallback_dir) {
    vec2 d = b - a;
    float len = length(d);
    return len > 0.0001 ? (d / len) : fallback_dir;
}

vec2 safe_normalize(vec2 value, vec2 fallback_value) {
    float len = length(value);
    return len > 0.0001 ? (value / len) : fallback_value;
}

float safe_span(float a, float b) {
    return max(abs(b - a), 1e-5);
}

float smoothstep01(float t) {
    t = clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float trail_life(float age_ms) {
    return clamp(1.0 - age_ms / u_lifetime_ms, 0.0, 1.0);
}

float trail_alpha(float age_ms) {
    float life = trail_life(age_ms);
    float taper = smoothstep01(life / u_tail_taper_fraction);
    return u_alpha_scale * taper * pow(life, 1.2) * 0.72;
}

float trail_width(float age_ms, float tail_t) {
    float life = trail_life(age_ms);
    float end_taper = 1.0 - pow(1.0 - clamp(tail_t, 0.0, 1.0), 5.0);
    float life_taper = smoothstep01(life / u_tail_taper_fraction);
    return u_radius * pow(life, 0.65) * end_taper * life_taper;
}

float tj(float ti, vec2 a, vec2 b) {
    float dist = max(length(b - a), 1e-4);
    return ti + sqrt(dist);
}

vec2 centripetal_catmull_rom(vec2 p0, vec2 p1, vec2 p2, vec2 p3, float t) {
    float t0 = 0.0;
    float t1 = tj(t0, p0, p1);
    float t2 = tj(t1, p1, p2);
    float t3 = tj(t2, p2, p3);
    float u = mix(t1, t2, clamp(t, 0.0, 1.0));

    vec2 a1 = ((t1 - u) / safe_span(t1, t0)) * p0 + ((u - t0) / safe_span(t1, t0)) * p1;
    vec2 a2 = ((t2 - u) / safe_span(t2, t1)) * p1 + ((u - t1) / safe_span(t2, t1)) * p2;
    vec2 a3 = ((t3 - u) / safe_span(t3, t2)) * p2 + ((u - t2) / safe_span(t3, t2)) * p3;
    vec2 b1 = ((t2 - u) / safe_span(t2, t0)) * a1 + ((u - t0) / safe_span(t2, t0)) * a2;
    vec2 b2 = ((t3 - u) / safe_span(t3, t1)) * a2 + ((u - t1) / safe_span(t3, t1)) * a3;
    return ((t2 - u) / safe_span(t2, t1)) * b1 + ((u - t1) / safe_span(t2, t1)) * b2;
}

vec2 curve_pos(vec2 p0, vec2 p1, vec2 p2, vec2 p3, float t) {
    if (u_point_count <= 2) {
        return mix(p1, p2, t);
    }
    return centripetal_catmull_rom(p0, p1, p2, p3, t);
}

vec2 curve_sample(float global_t) {
    float clamped_t = clamp(global_t, 0.0, float(max(u_point_count - 1, 1)));
    int seg_idx = min(int(floor(clamped_t)), max(u_point_count - 2, 0));
    float local_t = min(clamped_t - float(seg_idx), 1.0);

    vec2 p0 = fetch_point(seg_idx - 1).xy;
    vec2 p1 = fetch_point(seg_idx + 0).xy;
    vec2 p2 = fetch_point(seg_idx + 1).xy;
    vec2 p3 = fetch_point(seg_idx + 2).xy;
    return curve_pos(p0, p1, p2, p3, local_t);
}

vec2 curve_tangent(float global_t) {
    float span = float(max(u_point_count - 1, 1));
    float eps = max(0.02, min(0.35, span / float(max(u_curve_samples, 2))));
    vec2 a = curve_sample(global_t - eps);
    vec2 b = curve_sample(global_t + eps);
    return b - a;
}

void main() {
    int sample_count = max(u_curve_samples, 2);
    int segment_count = max(sample_count - 1, 1);
    int vertex_in_segment = gl_VertexID % 6;
    int segment_idx = min(gl_VertexID / 6, segment_count - 1);

    float t0 = float(segment_idx) / float(segment_count);
    float t1 = float(segment_idx + 1) / float(segment_count);
    float g0 = t0 * float(max(u_point_count - 1, 1));
    float g1 = t1 * float(max(u_point_count - 1, 1));
    vec2 pos0 = curve_sample(g0);
    vec2 pos1 = curve_sample(g1);

    float time0 = mix(fetch_point(0).z, fetch_point(u_point_count - 1).z, t0);
    float time1 = mix(fetch_point(0).z, fetch_point(u_point_count - 1).z, t1);
    float age0 = max(0.0, u_current_time_ms - time0);
    float age1 = max(0.0, u_current_time_ms - time1);
    float radius0 = trail_width(age0, t0);
    float radius1 = trail_width(age1, t1);
    float alpha0 = trail_alpha(age0);
    float alpha1 = trail_alpha(age1);

    vec2 fallback = safe_dir(pos0, pos1, vec2(1.0, 0.0));
    vec2 dir = safe_normalize(pos1 - pos0, fallback);
    vec2 normal = vec2(-dir.y, dir.x);
    float max_radius = max(radius0, radius1) + 1.5;
    vec2 start_cap = pos0 - dir * max_radius;
    vec2 end_cap = pos1 + dir * max_radius;

    bool use_end = (vertex_in_segment == 2 || vertex_in_segment == 3 || vertex_in_segment == 5);
    bool is_right = (vertex_in_segment == 1 || vertex_in_segment == 4 || vertex_in_segment == 5);
    vec2 edge_pos = use_end ? end_cap : start_cap;
    float side = is_right ? 1.0 : -1.0;
    vec2 world_pos = edge_pos + normal * (max_radius * side);

    gl_Position = projection * vec4(world_pos, -0.9, 1.0);
    v_world = world_pos;
    v_seg_a = pos0;
    v_seg_b = pos1;
    v_alpha_a = alpha0;
    v_alpha_b = alpha1;
    v_radius_a = radius0;
    v_radius_b = radius1;
}
"""

_TRAIL_FRAG = """
#version 330 core
in vec2 v_world;
flat in vec2 v_seg_a;
flat in vec2 v_seg_b;
flat in float v_alpha_a;
flat in float v_alpha_b;
flat in float v_radius_a;
flat in float v_radius_b;

uniform vec3 u_color;

out vec4 frag_color;

float dist2_to_segment(vec2 p, vec2 a, vec2 b, out float t) {
    vec2 ab = b - a;
    float ab2 = dot(ab, ab);
    if (ab2 < 1e-10) {
        t = 0.0;
        vec2 ap = p - a;
        return dot(ap, ap);
    }
    t = clamp(dot(p - a, ab) / ab2, 0.0, 1.0);
    vec2 delta = p - (a + ab * t);
    return dot(delta, delta);
}

void main() {
    float along_t = 0.0;
    float dist = sqrt(max(dist2_to_segment(v_world, v_seg_a, v_seg_b, along_t), 0.0));
    float radius = mix(v_radius_a, v_radius_b, along_t);
    float alpha_base = mix(v_alpha_a, v_alpha_b, along_t);
    float edge = 1.0 - smoothstep(max(0.0, radius - 1.0), radius + 0.5, dist);
    float a = edge * alpha_base;
    if (a < 0.001) discard;
    frag_color = vec4(u_color * a, a);
}
"""


# ---------------------------------------------------------------------------
# Bloom post-processing shaders
# ---------------------------------------------------------------------------

_BLOOM_QUAD_VERT = """
#version 330 core
in vec2 in_pos;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
}
"""

_BLOOM_THRESHOLD_FRAG = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_threshold;
out vec4 frag_color;
void main() {
    vec4 c = texture(u_tex, v_uv);
    float brightness = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
    float contrib = max(0.0, brightness - u_threshold);
    frag_color = c * (contrib / (brightness + 0.001));
}
"""

_BLOOM_BLUR_FRAG = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec2 u_direction;
out vec4 frag_color;
void main() {
    float w[7] = float[](0.1585, 0.1476, 0.1181, 0.0804, 0.0463, 0.0225, 0.0092);
    vec4 result = texture(u_tex, v_uv) * w[0];
    for (int i = 1; i < 7; i++) {
        vec2 off = u_direction * float(i) * 2.0;
        result += texture(u_tex, v_uv + off) * w[i];
        result += texture(u_tex, v_uv - off) * w[i];
    }
    frag_color = result;
}
"""

_BLOOM_COMPOSITE_FRAG = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_intensity;
out vec4 frag_color;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec4 scene = texture(u_scene, v_uv);
    vec4 bloom = texture(u_bloom, v_uv);
    vec3 col = scene.rgb + bloom.rgb * u_bloom_intensity;
    // Dither to 8-bit output to prevent banding
    float noise = (hash(v_uv * 2048.0) - 0.5) / 255.0;
    frag_color = vec4(col + noise, scene.a);
}
"""

_TRAIL_COMPOSITE_FRAG = """
#version 330 core

in vec2 v_uv;

uniform sampler2D u_trail;

out vec4 frag_color;

void main() {
    frag_color = texture(u_trail, v_uv);
}
"""

# ---------------------------------------------------------------------------
# Atmosphere (corner lightning) shader
# ---------------------------------------------------------------------------

_ATMOSPHERE_FRAG = """
#version 330 core
in vec2 v_uv;
uniform float u_time;
out vec4 frag_color;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 bl = v_uv;
    float d_bl = length(bl);
    float pulse_bl = 0.10 + 0.025 * sin(u_time * 0.7);
    float glow_bl = pulse_bl * pow(max(0.0, 1.0 - d_bl * 0.9), 2.5);

    vec2 br = v_uv - vec2(1.0, 0.0);
    float d_br = length(br);
    float pulse_br = 0.10 + 0.025 * sin(u_time * 0.7 + 1.8);
    float glow_br = pulse_br * pow(max(0.0, 1.0 - d_br * 0.9), 2.5);

    vec3 col = vec3(0.15, 0.35, 0.95) * glow_bl + vec3(0.95, 0.15, 0.25) * glow_br;
    float a = max(glow_bl, glow_br);

    // Dither to break banding
    float noise = (hash(v_uv * 1000.0 + u_time) - 0.5) * (1.0 / 128.0);
    col += noise;
    a = max(0.0, a + noise);

    frag_color = vec4(col, a);
}
"""


class DefaultSkin(Skin):
    name = "default"

    def __init__(self):
        self._visual_settings = make_default_skin_visual_settings()
        self._ctx = None
        self._bloom_w = 0
        self._bloom_h = 0
        self._scene_fbo = None
        self._scene_tex = None
        self._scene_depth = None
        self._bloom_fbo_a = None
        self._bloom_tex_a = None
        self._bloom_fbo_b = None
        self._bloom_tex_b = None
        self._trail_fbo = None
        self._trail_tex = None
        self._threshold_prog = None
        self._blur_prog = None
        self._composite_prog = None
        self._trail_composite_prog = None
        self._quad_vao_threshold = None
        self._quad_vao_blur_a = None
        self._quad_vao_blur_b = None
        self._quad_vao_composite = None
        self._quad_vao_trail_composite = None
        self._atmo_prog = None
        self._atmo_vao = None
        self._bloom_enabled = os.environ.get("OSU_BLOOM", "1").strip().lower() not in {"0", "false", "off", "no"}
        self._atmosphere_enabled = os.environ.get("OSU_ATMOSPHERE", "1").strip().lower() not in {"0", "false", "off", "no"}
        self._bloom_intensity = 0.50
        self._bloom_passes = max(0, int(os.environ.get("OSU_BLOOM_PASSES", "2") or 2))
        self._bloom_scale_div = max(2, int(os.environ.get("OSU_BLOOM_SCALE_DIV", "8") or 8))

    def circle_shader_source(self) -> tuple[str, str]:
        return _CIRCLE_VERT, _CIRCLE_FRAG

    def slider_shader_source(self) -> tuple[str, str]:
        return _SLIDER_VERT, _SLIDER_FRAG

    def approach_shader_source(self) -> tuple[str, str] | None:
        return _APPROACH_VERT, _APPROACH_FRAG

    def spinner_shader_source(self) -> tuple[str, str] | None:
        return _SPINNER_VERT, _SPINNER_FRAG

    def slider_ball_shader_source(self) -> tuple[str, str] | None:
        return _BALL_VERT, _BALL_FRAG

    def cursor_shader_source(self) -> tuple[str, str] | None:
        return _CURSOR_VERT, _CURSOR_FRAG

    def trail_shader_source(self) -> tuple[str, ...] | None:
        return _TRAIL_VERT, _TRAIL_FRAG

    def combo_colors(self) -> list[tuple[float, float, float]]:
        return [
            self.circle_fill_color(),
            (1.0, 0.55, 0.55),
            (0.55, 1.0, 0.7),
            (0.95, 0.75, 0.5),
        ]

    def circle_fill_color(self, *, slider_head: bool = False) -> tuple[float, float, float]:
        if slider_head and not self._visual_settings.slider_use_circle_head:
            return self._visual_settings.slider_head_fill_color
        return self._visual_settings.circle_fill_color

    def slider_fill_color(self) -> tuple[float, float, float]:
        return self._visual_settings.slider_path_fill_color

    def slider_ball_fill_color(self) -> tuple[float, float, float]:
        return self._visual_settings.slider_ball_fill_color

    def visual_settings(self) -> DefaultSkinVisualSettings:
        return self._visual_settings

    def set_visual_settings(self, settings) -> None:
        if isinstance(settings, DefaultSkinVisualSettings):
            self._visual_settings = settings.normalized()
        else:
            self._visual_settings = make_default_skin_visual_settings()

    def sync_object_uniforms(
        self,
        *,
        circle_prog=None,
        approach_prog=None,
        slider_prog=None,
        spinner_prog=None,
        ball_prog=None,
    ) -> None:
        vs = self._visual_settings
        head_border_color = (
            vs.circle_border_color
            if vs.slider_use_circle_head
            else vs.slider_head_border_color
        )
        head_fill_opacity = (
            vs.circle_fill_opacity
            if vs.slider_use_circle_head
            else vs.slider_head_fill_opacity
        )
        head_border_width = (
            vs.circle_border_width
            if vs.slider_use_circle_head
            else vs.slider_head_border_width
        )
        head_bloom = vs.circle_bloom if vs.slider_use_circle_head else vs.slider_head_bloom
        head_bloom_color = (
            vs.circle_bloom_color
            if vs.slider_use_circle_head
            else vs.slider_head_bloom_color
        )

        for prog in (circle_prog, approach_prog):
            if prog is None:
                continue
            if "u_circle_border_color" in prog:
                prog["u_circle_border_color"].value = vs.circle_border_color
            if "u_circle_fill_opacity" in prog:
                prog["u_circle_fill_opacity"].value = vs.circle_fill_opacity
            if "u_circle_border_width" in prog:
                prog["u_circle_border_width"].value = vs.circle_border_width
            if "u_circle_bloom" in prog:
                prog["u_circle_bloom"].value = vs.circle_bloom
            if "u_circle_bloom_color" in prog:
                prog["u_circle_bloom_color"].value = vs.circle_bloom_color
            if "u_slider_head_border_color" in prog:
                prog["u_slider_head_border_color"].value = head_border_color
            if "u_slider_head_fill_opacity" in prog:
                prog["u_slider_head_fill_opacity"].value = head_fill_opacity
            if "u_slider_head_border_width" in prog:
                prog["u_slider_head_border_width"].value = head_border_width
            if "u_slider_head_bloom" in prog:
                prog["u_slider_head_bloom"].value = head_bloom
            if "u_slider_head_bloom_color" in prog:
                prog["u_slider_head_bloom_color"].value = head_bloom_color

        if slider_prog is not None:
            if "u_slider_fill_color" in slider_prog:
                slider_prog["u_slider_fill_color"].value = vs.slider_path_fill_color
            if "u_slider_fill_opacity" in slider_prog:
                slider_prog["u_slider_fill_opacity"].value = vs.slider_path_fill_opacity
            if "u_slider_border_color" in slider_prog:
                slider_prog["u_slider_border_color"].value = vs.slider_path_border_color
            if "u_slider_border_width" in slider_prog:
                slider_prog["u_slider_border_width"].value = vs.slider_path_border_width

        if ball_prog is not None:
            if "u_slider_ball_border_color" in ball_prog:
                ball_prog["u_slider_ball_border_color"].value = vs.slider_ball_border_color
            if "u_slider_ball_fill_opacity" in ball_prog:
                ball_prog["u_slider_ball_fill_opacity"].value = vs.slider_ball_fill_opacity
            if "u_slider_ball_border_width" in ball_prog:
                ball_prog["u_slider_ball_border_width"].value = vs.slider_ball_border_width
            if "u_slider_ball_bloom" in ball_prog:
                ball_prog["u_slider_ball_bloom"].value = vs.slider_ball_bloom
            if "u_slider_ball_bloom_color" in ball_prog:
                ball_prog["u_slider_ball_bloom_color"].value = vs.slider_ball_bloom_color

    # ----------------------------------------------------------------- bloom

    def setup(self, ctx) -> None:
        self._ctx = ctx
        self._threshold_prog = ctx.program(
            vertex_shader=_BLOOM_QUAD_VERT,
            fragment_shader=_BLOOM_THRESHOLD_FRAG,
        )
        self._blur_prog = ctx.program(
            vertex_shader=_BLOOM_QUAD_VERT,
            fragment_shader=_BLOOM_BLUR_FRAG,
        )
        self._composite_prog = ctx.program(
            vertex_shader=_BLOOM_QUAD_VERT,
            fragment_shader=_BLOOM_COMPOSITE_FRAG,
        )
        self._trail_composite_prog = ctx.program(
            vertex_shader=_BLOOM_QUAD_VERT,
            fragment_shader=_TRAIL_COMPOSITE_FRAG,
        )
        self._atmo_prog = ctx.program(
            vertex_shader=_BLOOM_QUAD_VERT,
            fragment_shader=_ATMOSPHERE_FRAG,
        )
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._quad_buf = ctx.buffer(quad.tobytes())

    def cleanup(self) -> None:
        for attr in (
            "_scene_fbo", "_scene_tex", "_scene_depth",
            "_bloom_fbo_a", "_bloom_tex_a",
            "_bloom_fbo_b", "_bloom_tex_b",
            "_trail_fbo", "_trail_tex",
            "_quad_vao_threshold", "_quad_vao_blur_a",
            "_quad_vao_blur_b", "_quad_vao_composite", "_quad_vao_trail_composite",
            "_threshold_prog", "_blur_prog", "_composite_prog", "_trail_composite_prog",
            "_atmo_prog", "_atmo_vao",
            "_quad_buf",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
                setattr(self, attr, None)
        self._bloom_w = 0
        self._bloom_h = 0

    def _ensure_fbos(self, w: int, h: int) -> None:
        if w == self._bloom_w and h == self._bloom_h:
            return
        self._bloom_w = w
        self._bloom_h = h
        ctx = self._ctx

        for attr in (
            "_scene_fbo", "_scene_tex", "_scene_depth",
            "_bloom_fbo_a", "_bloom_tex_a",
            "_bloom_fbo_b", "_bloom_tex_b",
            "_trail_fbo", "_trail_tex",
            "_quad_vao_threshold", "_quad_vao_blur_a",
            "_quad_vao_blur_b", "_quad_vao_composite", "_quad_vao_trail_composite",
            "_atmo_vao",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass

        self._scene_tex = ctx.texture((w, h), 4, dtype="f2")
        self._scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._scene_tex.repeat_x = False
        self._scene_tex.repeat_y = False
        self._scene_depth = ctx.depth_renderbuffer((w, h))
        self._scene_fbo = ctx.framebuffer(
            color_attachments=[self._scene_tex],
            depth_attachment=self._scene_depth,
        )

        self._trail_tex = ctx.texture((w, h), 4, dtype="f2")
        self._trail_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._trail_tex.repeat_x = False
        self._trail_tex.repeat_y = False
        self._trail_fbo = ctx.framebuffer(color_attachments=[self._trail_tex])

        bw, bh = max(1, w // self._bloom_scale_div), max(1, h // self._bloom_scale_div)
        self._bloom_tex_a = ctx.texture((bw, bh), 4, dtype="f2")
        self._bloom_tex_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._bloom_tex_a.repeat_x = False
        self._bloom_tex_a.repeat_y = False
        self._bloom_fbo_a = ctx.framebuffer(color_attachments=[self._bloom_tex_a])

        self._bloom_tex_b = ctx.texture((bw, bh), 4, dtype="f2")
        self._bloom_tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._bloom_tex_b.repeat_x = False
        self._bloom_tex_b.repeat_y = False
        self._bloom_fbo_b = ctx.framebuffer(color_attachments=[self._bloom_tex_b])

        self._quad_vao_threshold = ctx.vertex_array(
            self._threshold_prog, [(self._quad_buf, "2f", "in_pos")],
        )
        self._quad_vao_blur_a = ctx.vertex_array(
            self._blur_prog, [(self._quad_buf, "2f", "in_pos")],
        )
        self._quad_vao_blur_b = ctx.vertex_array(
            self._blur_prog, [(self._quad_buf, "2f", "in_pos")],
        )
        self._quad_vao_composite = ctx.vertex_array(
            self._composite_prog, [(self._quad_buf, "2f", "in_pos")],
        )
        self._quad_vao_trail_composite = ctx.vertex_array(
            self._trail_composite_prog, [(self._quad_buf, "2f", "in_pos")],
        )
        self._atmo_vao = ctx.vertex_array(
            self._atmo_prog, [(self._quad_buf, "2f", "in_pos")],
        )

    def on_pre_render(self, ctx, time_ms: float, frametime: float) -> None:
        w, h = ctx.screen.viewport[2], ctx.screen.viewport[3]
        if w < 1 or h < 1:
            return
        self._ensure_fbos(w, h)
        if self._bloom_enabled:
            self._scene_fbo.use()
            self._scene_fbo.clear(0.02, 0.02, 0.06, 1.0)

    def render_atmosphere(self, ctx, time_ms: float) -> None:
        """Render corner lighting into the scene FBO (before bloom)."""
        if not self._atmosphere_enabled or self._atmo_vao is None or self._scene_fbo is None:
            return
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        self._atmo_prog["u_time"].value = time_ms / 1000.0
        self._atmo_vao.render(moderngl.TRIANGLE_STRIP)

    def begin_trail_pass(self, ctx) -> None:
        if self._trail_fbo is None:
            return
        self._trail_fbo.use()
        self._trail_fbo.clear(0.0, 0.0, 0.0, 0.0)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_equation = moderngl.MAX
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)

    def end_trail_pass(self, ctx) -> None:
        if self._trail_fbo is None or self._quad_vao_trail_composite is None:
            return
        if self._bloom_enabled and self._scene_fbo is not None:
            self._scene_fbo.use()
        else:
            ctx.screen.use()
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_equation = moderngl.FUNC_ADD
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        self._trail_tex.use(location=0)
        self._trail_composite_prog["u_trail"].value = 0
        self._quad_vao_trail_composite.render(moderngl.TRIANGLE_STRIP)

    def on_post_render(self, ctx, time_ms: float, frametime: float) -> None:
        if not self._bloom_enabled or self._scene_fbo is None:
            return

        with profiler.timer("skin.default.postprocess"):
            ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)

            self._bloom_fbo_a.use()
            self._scene_tex.use(location=0)
            self._threshold_prog["u_tex"].value = 0
            self._threshold_prog["u_threshold"].value = 0.30
            self._quad_vao_threshold.render(moderngl.TRIANGLE_STRIP)

            bw = self._bloom_tex_a.width
            bh = self._bloom_tex_a.height
            for _ in range(self._bloom_passes):
                self._bloom_fbo_b.use()
                self._bloom_tex_a.use(location=0)
                self._blur_prog["u_tex"].value = 0
                self._blur_prog["u_direction"].value = (1.0 / bw, 0.0)
                self._quad_vao_blur_a.render(moderngl.TRIANGLE_STRIP)

                self._bloom_fbo_a.use()
                self._bloom_tex_b.use(location=0)
                self._blur_prog["u_tex"].value = 0
                self._blur_prog["u_direction"].value = (0.0, 1.0 / bh)
                self._quad_vao_blur_b.render(moderngl.TRIANGLE_STRIP)

            ctx.screen.use()
            self._scene_tex.use(location=0)
            self._bloom_tex_a.use(location=1)
            self._composite_prog["u_scene"].value = 0
            self._composite_prog["u_bloom"].value = 1
            self._composite_prog["u_bloom_intensity"].value = self._bloom_intensity
            self._quad_vao_composite.render(moderngl.TRIANGLE_STRIP)

    def set_atmosphere_enabled(self, enabled: bool) -> None:
        self._atmosphere_enabled = bool(enabled)

    def set_bloom_intensity(self, intensity: float) -> None:
        self._bloom_intensity = max(0.0, min(1.5, float(intensity)))
