"""Optional native speedups with pure-Python fallback."""

from __future__ import annotations

try:
    try:
        from ._rust_core import (  # type: ignore[attr-defined]
            compute_slider_ball_instances as _rust_compute_slider_ball_instances,
            interpolate_cursor_query as _rust_interpolate_cursor_query,
            keys_index_at as _rust_keys_index_at,
        )
    except Exception:
        from _rust_core import (  # type: ignore[attr-defined]
            compute_slider_ball_instances as _rust_compute_slider_ball_instances,
            interpolate_cursor_query as _rust_interpolate_cursor_query,
            keys_index_at as _rust_keys_index_at,
        )
except Exception:
    _rust_interpolate_cursor_query = None
    _rust_keys_index_at = None
    _rust_compute_slider_ball_instances = None

try:
    from ._cy_gameplay import (  # type: ignore[attr-defined]
        advance_trail_state as _cy_advance_trail_state,
        clip_trail_points as _cy_clip_trail_points,
        compute_slider_ball_instances as _cy_compute_slider_ball_instances,
        interpolate_cursor_query as _cy_interpolate_cursor_query,
        keys_index_at as _cy_keys_index_at,
        prepare_trail_control_points as _cy_prepare_trail_control_points,
        prepare_trail_control_points_ring as _cy_prepare_trail_control_points_ring,
    )
except Exception:
    _cy_advance_trail_state = None
    _cy_clip_trail_points = None
    _cy_interpolate_cursor_query = None
    _cy_keys_index_at = None
    _cy_compute_slider_ball_instances = None
    _cy_prepare_trail_control_points = None
    _cy_prepare_trail_control_points_ring = None


HAS_RUST_CORE = _rust_interpolate_cursor_query is not None
HAS_CYTHON_CORE = _cy_interpolate_cursor_query is not None


def interpolate_cursor_query(times, xs, ys, time_ms: float, anchor: int):
    if _rust_interpolate_cursor_query is not None:
        return _rust_interpolate_cursor_query(times, xs, ys, time_ms, anchor)
    if _cy_interpolate_cursor_query is not None:
        return _cy_interpolate_cursor_query(times, xs, ys, time_ms, anchor)
    return None


def keys_index_at(times, time_ms: float, anchor: int):
    if _rust_keys_index_at is not None:
        return _rust_keys_index_at(times, time_ms, anchor)
    if _cy_keys_index_at is not None:
        return _cy_keys_index_at(times, time_ms, anchor)
    return None


def compute_slider_ball_instances(
    current_time_ms: float,
    circle_radius: float,
    ball_color,
    start_times,
    end_times,
    repeat_counts,
    path_starts,
    path_counts,
    path_points,
    z_values,
    out_buf,
    max_balls: int,
):
    if _rust_compute_slider_ball_instances is not None:
        return _rust_compute_slider_ball_instances(
            current_time_ms,
            circle_radius,
            ball_color,
            start_times,
            end_times,
            repeat_counts,
            path_starts,
            path_counts,
            path_points,
            z_values,
            out_buf,
            max_balls,
        )
    if _cy_compute_slider_ball_instances is not None:
        return _cy_compute_slider_ball_instances(
            current_time_ms,
            circle_radius,
            ball_color,
            start_times,
            end_times,
            repeat_counts,
            path_starts,
            path_counts,
            path_points,
            z_values,
            out_buf,
            max_balls,
        )
    return -1


def prepare_trail_control_points(
    trail_points,
    max_points: int,
    current_time_ms: float,
    trail_lifetime_ms: float,
    raw_buf,
    out_buf,
):
    if _cy_prepare_trail_control_points is not None:
        return _cy_prepare_trail_control_points(
            trail_points,
            max_points,
            current_time_ms,
            trail_lifetime_ms,
            raw_buf,
            out_buf,
        )
    return None


def advance_trail_state(
    trail_buf,
    start_idx: int,
    count: int,
    current_time_ms: float,
    x: float,
    y: float,
    trail_lifetime_ms: float,
    min_sample_dt_ms: float,
    min_sample_dist_sq: float,
):
    if _cy_advance_trail_state is not None:
        return _cy_advance_trail_state(
            trail_buf,
            start_idx,
            count,
            current_time_ms,
            x,
            y,
            trail_lifetime_ms,
            min_sample_dt_ms,
            min_sample_dist_sq,
        )
    return None


def prepare_trail_control_points_ring(
    trail_buf,
    start_idx: int,
    count: int,
    current_time_ms: float,
    trail_lifetime_ms: float,
    out_buf,
):
    if _cy_prepare_trail_control_points_ring is not None:
        return _cy_prepare_trail_control_points_ring(
            trail_buf,
            start_idx,
            count,
            current_time_ms,
            trail_lifetime_ms,
            out_buf,
        )
    return None


def clip_trail_points(
    raw_buf,
    count: int,
    current_time_ms: float,
    trail_lifetime_ms: float,
    min_control_dt_ms: float,
    min_control_dist_sq: float,
    clip_buf,
):
    if _cy_clip_trail_points is not None:
        return _cy_clip_trail_points(
            raw_buf,
            count,
            current_time_ms,
            trail_lifetime_ms,
            min_control_dt_ms,
            min_control_dist_sq,
            clip_buf,
        )
    return None
