"""Danser-style cursor path precomputation.

The goal is fluid, continuous motion where objects feel like they spawned on the
cursor's route instead of being explicitly targeted one by one. The path is
built from sparse waypoints, then a global time spline turns them into a smooth
trajectory.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from osu_map.beatmap import RenderData, SliderRenderData
from replay.spinner import MAX_SPINNER_RPS, SPINNER_CENTER_X, SPINNER_CENTER_Y
from .base import CursorPlaystyle, OSU_W, OSU_H, MARGIN

SAMPLE_MS = 1.0
STACK_DIST = 20.0
STACK_TIME_MS = 260.0
LONG_GAP_MS = 950.0
STREAM_LINE_DOT = 0.992
STREAM_TIME_MS = 125.0
STREAM_MAX_DIST = 72.0
SLIDER_TICK_BLEND = 0.5
SLIDER_CURVE_THRESHOLD = 10.0
PACING_TIME_RATIO = 3.5
PACING_MIN_DT_MS = 220.0
PACING_MAX_GUIDES = 3
SMOOTH_SIGMA = 3.5
RADIUS_MARGIN = 0.5
REFINE_BLEND = 0.7
REFINE_MAX_PASSES = 8
REVERSAL_FIX_MAX_PASSES = 8
WAYPOINT_MATCH_EPS_MS = 1.0
SOFT_CLIP_SCALE = 12.0
REVERSAL_DOT_THRESHOLD = -0.92
REVERSAL_MAX_GUIDES = 12
EDGE_SUPPORT_RATIO = 0.4
EDGE_SUPPORT_MIN = 10.0
EDGE_TURN_DOT_THRESHOLD = 0.9
EDGE_TURN_MAX_GUIDES = 12
MICRO_LOOP_WINDOW_MS = 10.0
MICRO_LOOP_SCAN_RADIUS_MS = 18.0
MICRO_LOOP_DOT_THRESHOLD = -0.55
MICRO_LOOP_MAX_DISP = 1.2
MICRO_LOOP_PREV_GUIDE_TIME_FRAC = 0.24
MICRO_LOOP_GUIDE_TIME_FRAC = 0.32
MICRO_LOOP_PREV_GUIDE_POS_FRAC = 0.24
MICRO_LOOP_GUIDE_POS_FRAC = 0.24
MICRO_LOOP_MIN_DT_MS = 140.0


@dataclass
class _Waypoint:
    time_ms: float
    pos: np.ndarray
    kind: str


def _norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def _unit(vec: np.ndarray) -> np.ndarray:
    length = _norm(vec)
    if length < 1e-6:
        return np.zeros(2, dtype=np.float64)
    return vec / length


def _soft_clip_axis(values: np.ndarray, low: float, high: float, scale: float) -> np.ndarray:
    clipped = values.copy()

    low_mask = clipped < low
    if np.any(low_mask):
        d = low - clipped[low_mask]
        clipped[low_mask] = low - d * np.exp(-d / scale)

    high_mask = clipped > high
    if np.any(high_mask):
        d = clipped[high_mask] - high
        clipped[high_mask] = high + d * np.exp(-d / scale)

    return clipped


def _soft_clip_positions(positions: np.ndarray) -> np.ndarray:
    clipped = positions.copy()
    low = -MARGIN
    high_x = OSU_W + MARGIN
    high_y = OSU_H + MARGIN
    clipped[:, 0] = _soft_clip_axis(clipped[:, 0], low, high_x, SOFT_CLIP_SCALE)
    clipped[:, 1] = _soft_clip_axis(clipped[:, 1], low, high_y, SOFT_CLIP_SCALE)
    return clipped


def _soft_clip_point(point: np.ndarray) -> np.ndarray:
    buf = np.array([[float(point[0]), float(point[1])]], dtype=np.float64)
    return _soft_clip_positions(buf)[0]


def _support_anchor_pos(point: np.ndarray, radius: float) -> np.ndarray:
    """Keep spline support anchors slightly inside the playfield near edges.

    Constraints still use the real object center, but support anchors should
    leave some room for the global spline to curve without skimming the border.
    """
    inset = max(EDGE_SUPPORT_MIN, radius * EDGE_SUPPORT_RATIO)
    return np.array([
        min(max(float(point[0]), inset), OSU_W - inset),
        min(max(float(point[1]), inset), OSU_H - inset),
    ], dtype=np.float64)


def _slider_ball_pos(srd: SliderRenderData, si: int, frac: float) -> np.ndarray:
    ps = srd.path_starts[si]
    pc = srd.path_counts[si]
    fi = frac * (pc - 1)
    idx = min(int(fi), pc - 2)
    f = fi - idx
    return srd.path_points[ps + idx] * (1.0 - f) + srd.path_points[ps + idx + 1] * f


def _merge_waypoints(waypoints: list[_Waypoint]) -> list[_Waypoint]:
    if not waypoints:
        return []

    kind_priority = {
        "intro": 6,
        "outro": 6,
        "circle": 5,
        "slider_anchor": 5,
        "slider_end": 5,
        "refine": 4,
        "stack": 3,
        "guide": 1,
    }

    waypoints.sort(key=lambda w: w.time_ms)
    merged: list[_Waypoint] = [waypoints[0]]
    for wp in waypoints[1:]:
        prev = merged[-1]
        if abs(wp.time_ms - prev.time_ms) < 0.5:
            kind = prev.kind
            if kind_priority.get(wp.kind, 0) > kind_priority.get(prev.kind, 0):
                kind = wp.kind
            merged[-1] = _Waypoint(
                prev.time_ms,
                (prev.pos + wp.pos) * 0.5,
                kind,
            )
        else:
            merged.append(wp)
    return merged


def _flatten_stacks(waypoints: list[_Waypoint]) -> list[_Waypoint]:
    if not waypoints:
        return []

    out: list[_Waypoint] = []
    i = 0
    while i < len(waypoints):
        cluster = [waypoints[i]]
        j = i + 1
        while j < len(waypoints):
            if waypoints[j].kind == "guide":
                break
            if waypoints[j].time_ms - cluster[-1].time_ms > STACK_TIME_MS:
                break
            if _norm(waypoints[j].pos - cluster[0].pos) > STACK_DIST:
                break
            cluster.append(waypoints[j])
            j += 1

        if len(cluster) > 1:
            center = np.mean([wp.pos for wp in cluster], axis=0)
            start_time = float(cluster[0].time_ms)
            end_time = float(cluster[-1].time_ms)
            out.append(_Waypoint(start_time, center.copy(), "stack"))
            if end_time - start_time > 1.0:
                out.append(_Waypoint(end_time, center.copy(), "stack"))
        else:
            out.append(cluster[0])
        i = j
    return out


def _prune_stream_midpoints(waypoints: list[_Waypoint]) -> list[_Waypoint]:
    if len(waypoints) < 3:
        return waypoints

    kept = [waypoints[0]]
    for i in range(1, len(waypoints) - 1):
        prev = kept[-1]
        curr = waypoints[i]
        nxt = waypoints[i + 1]

        if curr.kind == "circle" and prev.kind == "circle" and nxt.kind == "circle":
            dt1 = curr.time_ms - prev.time_ms
            dt2 = nxt.time_ms - curr.time_ms
            v1 = curr.pos - prev.pos
            v2 = nxt.pos - curr.pos
            n1 = _norm(v1)
            n2 = _norm(v2)
            if (
                dt1 <= STREAM_TIME_MS and dt2 <= STREAM_TIME_MS
                and n1 > 1.0 and n2 > 1.0
                and n1 <= STREAM_MAX_DIST and n2 <= STREAM_MAX_DIST
            ):
                if float(np.dot(v1 / n1, v2 / n2)) >= STREAM_LINE_DOT:
                    continue

        kept.append(curr)

    kept.append(waypoints[-1])
    return kept


def _insert_pacing_waypoints(waypoints: list[_Waypoint]) -> list[_Waypoint]:
    """Regularize large timing-density jumps that make global cubics kink."""
    if len(waypoints) < 2:
        return waypoints

    extras: list[_Waypoint] = []
    n = len(waypoints)
    for i in range(n - 1):
        curr = waypoints[i]
        nxt = waypoints[i + 1]
        seg_dt = nxt.time_ms - curr.time_ms
        if seg_dt < PACING_MIN_DT_MS:
            continue

        neighbor_dt = None
        if i > 0:
            neighbor_dt = curr.time_ms - waypoints[i - 1].time_ms
        if i + 2 < n:
            rhs = waypoints[i + 2].time_ms - nxt.time_ms
            neighbor_dt = rhs if neighbor_dt is None else min(neighbor_dt, rhs)
        if neighbor_dt is None or neighbor_dt <= 0.0:
            continue

        ratio = seg_dt / neighbor_dt
        if ratio < PACING_TIME_RATIO:
            continue

        n_guides = min(PACING_MAX_GUIDES, max(1, int(ratio // PACING_TIME_RATIO)))
        for k in range(1, n_guides + 1):
            frac = k / (n_guides + 1)
            pos = curr.pos * (1.0 - frac) + nxt.pos * frac
            extras.append(_Waypoint(curr.time_ms + seg_dt * frac, pos, "guide"))

    return _merge_waypoints(waypoints + extras)


def _insert_gap_guides(waypoints: list[_Waypoint]) -> list[_Waypoint]:
    if len(waypoints) < 2:
        return waypoints

    extras: list[_Waypoint] = []
    center = np.array([OSU_W / 2, OSU_H / 2], dtype=np.float64)

    for prev, nxt in zip(waypoints, waypoints[1:]):
        dt = nxt.time_ms - prev.time_ms
        if dt < LONG_GAP_MS:
            continue

        delta = nxt.pos - prev.pos
        dist = _norm(delta)
        if dist < 1.0:
            continue

        normal = np.array([-delta[1], delta[0]], dtype=np.float64) / dist
        midpoint = (prev.pos + nxt.pos) * 0.5
        to_center = center - midpoint
        if _norm(to_center) > 1e-6 and float(np.dot(normal, _unit(to_center))) < 0.0:
            normal = -normal

        arc_height = min(72.0, max(16.0, dist * 0.14 + (dt - LONG_GAP_MS) * 0.028))
        for frac in (0.33, 0.66):
            base = prev.pos * (1.0 - frac) + nxt.pos * frac
            pos = base + normal * (arc_height * math.sin(math.pi * frac))
            extras.append(_Waypoint(prev.time_ms + dt * frac, pos, "guide"))

    return _merge_waypoints(waypoints + extras)


def _slider_tick_fracs(beatmap, slider) -> list[float]:
    tick_rate = float(getattr(beatmap, "slider_tick_rate", 1.0) or 1.0)
    if tick_rate <= 0.0:
        return []

    base_bl, sv, _, _ = beatmap._timing_at(float(slider.start_time))
    px_per_beat = beatmap.slider_multiplier * 100.0 * sv
    if getattr(beatmap, "file_version", 14) < 8 and sv != 0.0:
        px_per_beat /= sv
    if px_per_beat <= 0.0 or base_bl <= 0.0:
        return []

    one_slide_beats = float(slider.pixel_length) / px_per_beat
    if one_slide_beats <= 0.0:
        return []

    fracs: list[float] = []
    tick_beat = 1.0 / tick_rate
    beat = tick_beat
    while beat < one_slide_beats - 0.1:
        fracs.append(beat / one_slide_beats)
        beat += tick_beat
    return fracs


def _slider_guides_from_ticks(
    beatmap,
    slider,
    srd: SliderRenderData,
    si: int,
    circle_radius: float,
) -> tuple[list[_Waypoint], list[_Waypoint]]:
    repeats = max(1, int(srd.repeat_counts[si]))
    duration = float(slider.end_time) - float(slider.start_time)
    if duration <= 0.0:
        return [], []

    tick_fracs = _slider_tick_fracs(beatmap, slider)
    guides: list[_Waypoint] = []
    constraints: list[_Waypoint] = []
    leg_duration = duration / repeats

    for leg in range(repeats):
        leg_t0 = float(slider.start_time) + leg * leg_duration
        leg_t1 = leg_t0 + leg_duration
        leg_start = _slider_ball_pos(srd, si, 0.0 if leg % 2 == 0 else 1.0).astype(np.float64)
        leg_end = _slider_ball_pos(srd, si, 1.0 if leg % 2 == 0 else 0.0).astype(np.float64)

        if tick_fracs:
            for frac in tick_fracs:
                leg_frac = frac if leg % 2 == 0 else 1.0 - frac
                actual = _slider_ball_pos(srd, si, leg_frac).astype(np.float64)
                chord = leg_start * (1.0 - frac) + leg_end * frac
                pos = chord * (1.0 - SLIDER_TICK_BLEND) + actual * SLIDER_TICK_BLEND
                guides.append(_Waypoint(leg_t0 + leg_duration * frac, pos, "guide"))
        else:
            mid_frac = 0.5 if leg % 2 == 0 else 0.5
            mid_actual = _slider_ball_pos(srd, si, mid_frac).astype(np.float64)
            mid_chord = (leg_start + leg_end) * 0.5
            if _norm(mid_actual - mid_chord) >= SLIDER_CURVE_THRESHOLD and leg_duration >= 80.0:
                pos = mid_chord * (1.0 - SLIDER_TICK_BLEND) + mid_actual * SLIDER_TICK_BLEND
                guides.append(_Waypoint((leg_t0 + leg_t1) * 0.5, pos, "guide"))

        end_frac = 1.0 if leg % 2 == 0 else 0.0
        end_pos = _slider_ball_pos(srd, si, end_frac).astype(np.float64)
        kind = "slider_anchor" if leg < repeats - 1 else "slider_end"
        guides.append(_Waypoint(leg_t1, _support_anchor_pos(end_pos, circle_radius), kind))
        constraints.append(_Waypoint(leg_t1, end_pos, kind))

    return guides, constraints


def _spinner_guides(
    start_time: float,
    end_time: float,
    *,
    entry_pos: np.ndarray,
    circle_radius: float,
) -> list[_Waypoint]:
    duration = end_time - start_time
    if duration <= 1.0:
        return [_Waypoint(start_time, entry_pos.copy(), "guide")]

    center = np.array([SPINNER_CENTER_X, OSU_H - SPINNER_CENTER_Y], dtype=np.float64)
    spin_radius = max(48.0, min(80.0, circle_radius * 1.85))
    total_turns = max(1.0, duration / 1000.0 * MAX_SPINNER_RPS)
    quarter_turns = max(4, int(total_turns * 4.0))

    direction = entry_pos - center
    if _norm(direction) < 1e-3:
        direction = np.array([1.0, 0.0], dtype=np.float64)
    direction = _unit(direction)
    start_angle = math.atan2(direction[1], direction[0])

    guides: list[_Waypoint] = []
    for step in range(quarter_turns + 1):
        frac = step / max(quarter_turns, 1)
        angle = start_angle - frac * total_turns * math.tau
        pos = center + np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * spin_radius
        kind = "slider_anchor" if step in {0, quarter_turns} else "guide"
        guides.append(_Waypoint(start_time + duration * frac, pos, kind))
    return guides


def _evaluate_spline(wp_times: np.ndarray, wp_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cs_x = CubicSpline(wp_times, wp_pos[:, 0], bc_type="not-a-knot")
    cs_y = CubicSpline(wp_times, wp_pos[:, 1], bc_type="not-a-knot")
    sample_times = np.arange(wp_times[0], wp_times[-1] + SAMPLE_MS, SAMPLE_MS)
    px = cs_x(sample_times)
    py = cs_y(sample_times)
    if len(px) > 3:
        px = gaussian_filter1d(px, sigma=SMOOTH_SIGMA, mode="nearest")
        py = gaussian_filter1d(py, sigma=SMOOTH_SIGMA, mode="nearest")
    return sample_times, np.column_stack([px, py])


def _position_at(times: np.ndarray, positions: np.ndarray, time_ms: float) -> np.ndarray:
    idx = int(np.searchsorted(times, time_ms))
    if idx <= 0:
        return positions[0]
    if idx >= len(times):
        return positions[-1]

    t0 = times[idx - 1]
    t1 = times[idx]
    if t1 - t0 < 1e-6:
        return positions[idx]
    frac = (time_ms - t0) / (t1 - t0)
    return positions[idx - 1] * (1.0 - frac) + positions[idx] * frac


def _radius_errors(
    sample_times: np.ndarray,
    sample_pos: np.ndarray,
    constraints: list[_Waypoint],
) -> list[tuple[_Waypoint, float, np.ndarray]]:
    errors: list[tuple[_Waypoint, float, np.ndarray]] = []
    for constraint in constraints:
        path_pos = _position_at(sample_times, sample_pos, constraint.time_ms)
        dist = _norm(constraint.pos - path_pos)
        errors.append((constraint, dist, path_pos))
    return errors


def _reversal_guides(
    waypoints: list[_Waypoint],
    sample_times: np.ndarray,
    sample_pos: np.ndarray,
) -> tuple[list[_Waypoint], bool]:
    if len(sample_pos) < 3 or len(waypoints) < 2:
        return waypoints, False

    deltas = np.diff(sample_pos, axis=0)
    lens = np.linalg.norm(deltas, axis=1)
    good = lens > 1e-3
    idxs = np.flatnonzero(good)
    if len(idxs) < 3:
        return waypoints, False

    dirs = deltas[good] / lens[good][:, np.newaxis]
    dots = np.sum(dirs[:-1] * dirs[1:], axis=1)
    bad = np.where(dots < REVERSAL_DOT_THRESHOLD)[0]
    if len(bad) == 0:
        return waypoints, False

    wp_times = np.array([wp.time_ms for wp in waypoints], dtype=np.float64)
    extras: list[_Waypoint] = []
    seen_times: list[float] = []

    for bad_idx in bad:
        if len(extras) >= REVERSAL_MAX_GUIDES:
            break

        sample_idx = int(idxs[bad_idx + 1])
        time_ms = float(sample_times[sample_idx])
        if any(abs(time_ms - t) < 8.0 for t in seen_times):
            continue

        right = int(np.searchsorted(wp_times, time_ms, side="right"))
        left = right - 1
        if left < 0 or right >= len(waypoints):
            continue

        t0 = wp_times[left]
        t1 = wp_times[right]
        if t1 - t0 < 1e-3:
            continue

        frac = (time_ms - t0) / (t1 - t0)
        frac = max(0.0, min(1.0, frac))
        chord = waypoints[left].pos * (1.0 - frac) + waypoints[right].pos * frac
        pos = chord * 0.85 + sample_pos[sample_idx] * 0.15
        extras.append(_Waypoint(time_ms, pos, "guide"))
        seen_times.append(time_ms)

    if not extras:
        return waypoints, False

    return _merge_waypoints(waypoints + extras), True


def _edge_turn_guides(
    waypoints: list[_Waypoint],
    sample_times: np.ndarray,
    sample_pos: np.ndarray,
    circle_radius: float,
) -> tuple[list[_Waypoint], bool]:
    if len(sample_pos) < 3:
        return waypoints, False

    deltas = np.diff(sample_pos, axis=0)
    lens = np.linalg.norm(deltas, axis=1)
    good = lens > 1e-3
    idxs = np.flatnonzero(good)
    if len(idxs) < 3:
        return waypoints, False

    dirs = deltas[good] / lens[good][:, np.newaxis]
    dots = np.sum(dirs[:-1] * dirs[1:], axis=1)
    inset = max(EDGE_SUPPORT_MIN, circle_radius * EDGE_SUPPORT_RATIO)
    low = inset
    high_x = OSU_W - inset
    high_y = OSU_H - inset

    extras: list[_Waypoint] = []
    seen_times: list[float] = []
    for bad_idx in np.where(dots < EDGE_TURN_DOT_THRESHOLD)[0]:
        if len(extras) >= EDGE_TURN_MAX_GUIDES:
            break

        sample_idx = int(idxs[bad_idx + 1])
        pt = sample_pos[sample_idx]
        near_edge = (
            pt[0] < low or pt[0] > high_x
            or pt[1] < low or pt[1] > high_y
        )
        if not near_edge:
            continue

        time_ms = float(sample_times[sample_idx])
        if any(abs(time_ms - t) < 8.0 for t in seen_times):
            continue

        projected = _support_anchor_pos(pt, circle_radius)
        pos = pt * 0.15 + projected * 0.85
        extras.append(_Waypoint(time_ms, pos, "guide"))
        seen_times.append(time_ms)

    if not extras:
        return waypoints, False

    return _merge_waypoints(waypoints + extras), True


def _micro_loop_guides(
    waypoints: list[_Waypoint],
    sample_times: np.ndarray,
    sample_pos: np.ndarray,
) -> tuple[list[_Waypoint], bool]:
    if len(waypoints) < 3 or len(sample_times) < 8:
        return waypoints, False

    step_ms = float(sample_times[1] - sample_times[0]) if len(sample_times) > 1 else SAMPLE_MS
    if step_ms <= 0.0:
        step_ms = SAMPLE_MS
    window = max(1, int(round(MICRO_LOOP_WINDOW_MS / step_ms)))
    scan = max(window + 1, int(round(MICRO_LOOP_SCAN_RADIUS_MS / step_ms)))
    if len(sample_pos) <= window * 2 + 2:
        return waypoints, False

    extras: list[_Waypoint] = []
    seen_times: list[float] = []
    for prev, curr, nxt in zip(waypoints, waypoints[1:], waypoints[2:]):
        if "guide" in (prev.kind, curr.kind, nxt.kind):
            continue
        dt_prev = curr.time_ms - prev.time_ms
        dt_next = nxt.time_ms - curr.time_ms
        if dt_prev < MICRO_LOOP_MIN_DT_MS or dt_next < MICRO_LOOP_MIN_DT_MS:
            continue

        center_idx = int(np.searchsorted(sample_times, curr.time_ms))
        lo = max(window, center_idx - scan)
        hi = min(len(sample_pos) - window - 1, center_idx + scan)
        if hi <= lo:
            continue

        best_dot = 1.0
        best_disp = 1e9
        for idx in range(lo, hi + 1):
            v1 = sample_pos[idx] - sample_pos[idx - window]
            v2 = sample_pos[idx + window] - sample_pos[idx]
            n1 = _norm(v1)
            n2 = _norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            dot = float(np.dot(v1 / n1, v2 / n2))
            disp = min(n1, n2)
            if dot < best_dot:
                best_dot = dot
                best_disp = disp

        if best_dot > MICRO_LOOP_DOT_THRESHOLD or best_disp > MICRO_LOOP_MAX_DISP:
            continue

        before_time = curr.time_ms - dt_prev * MICRO_LOOP_PREV_GUIDE_TIME_FRAC
        after_time = curr.time_ms + dt_next * MICRO_LOOP_GUIDE_TIME_FRAC
        if any(abs(before_time - t) < 10.0 or abs(after_time - t) < 10.0 for t in seen_times):
            continue
        before_pos = curr.pos * (1.0 - MICRO_LOOP_PREV_GUIDE_POS_FRAC) + prev.pos * MICRO_LOOP_PREV_GUIDE_POS_FRAC
        after_pos = curr.pos * (1.0 - MICRO_LOOP_GUIDE_POS_FRAC) + nxt.pos * MICRO_LOOP_GUIDE_POS_FRAC
        extras.append(_Waypoint(before_time, before_pos, "guide"))
        extras.append(_Waypoint(after_time, after_pos, "guide"))
        seen_times.extend([before_time, after_time])

    if not extras:
        return waypoints, False
    return _merge_waypoints(waypoints + extras), True


def _refine_waypoints(
    waypoints: list[_Waypoint],
    errors: list[tuple[_Waypoint, float, np.ndarray]],
    target_radius: float,
) -> tuple[list[_Waypoint], bool]:
    updated = list(waypoints)
    changed = False

    for constraint, dist, path_pos in errors:
        if dist <= target_radius:
            continue

        required_blend = 1.0 - (target_radius / max(dist, 1e-6))
        blend = min(1.0, max(REFINE_BLEND, required_blend + 0.18))
        new_pos = path_pos * (1.0 - blend) + constraint.pos * blend
        match_idx = None
        for i, wp in enumerate(updated):
            if abs(wp.time_ms - constraint.time_ms) <= WAYPOINT_MATCH_EPS_MS:
                match_idx = i
                break

        if match_idx is None:
            updated.append(_Waypoint(constraint.time_ms, new_pos, "refine"))
        else:
            updated[match_idx] = _Waypoint(
                updated[match_idx].time_ms,
                updated[match_idx].pos * 0.3 + new_pos * 0.7,
                "refine" if updated[match_idx].kind == "guide" else updated[match_idx].kind,
            )
        changed = True

    if changed:
        updated = _merge_waypoints(updated)
    return updated, changed


class DanserPlaystyle(CursorPlaystyle):
    """Sparse, flowing cursor path using a global spline."""

    def __init__(self, beatmap, render_data: RenderData):
        self._times = np.array([], dtype=np.float64)
        self._positions = np.zeros((0, 2), dtype=np.float64)
        self._debug_waypoints: list[dict[str, object]] = []
        self._debug_constraints: list[dict[str, object]] = []
        self._debug_radius_errors: list[dict[str, object]] = []
        self._build(beatmap, render_data)

    def position_at(self, time_ms: float) -> tuple[float, float]:
        n = len(self._times)
        if n == 0:
            return (256.0, OSU_H / 2)
        idx = bisect.bisect_right(self._times, time_ms)
        if idx <= 0:
            p = self._positions[0]
        elif idx >= n:
            p = self._positions[-1]
        else:
            t0, t1 = self._times[idx - 1], self._times[idx]
            dt = t1 - t0
            if dt < 0.01:
                p = self._positions[idx]
            else:
                frac = (time_ms - t0) / dt
                p = self._positions[idx - 1] * (1.0 - frac) + self._positions[idx] * frac
        p = _soft_clip_point(p)
        return float(p[0]), float(p[1])

    def _build(self, beatmap, rd: RenderData):
        from osupyparser.osu.objects import Slider, Spinner

        srd = rd.slider
        slider_obj_map: dict[int, int] = {}
        if srd is not None:
            for i in range(srd.n_sliders):
                slider_obj_map[int(srd.object_indices[i])] = i

        waypoints: list[_Waypoint] = []
        constraints: list[_Waypoint] = []
        center = np.array([256.0, OSU_H / 2], dtype=np.float64)
        first_time: float | None = None

        for obj_idx, obj in enumerate(beatmap.hit_objects):
            pos = np.array([float(obj.pos.x), OSU_H - float(obj.pos.y)], dtype=np.float64)
            support_pos = _support_anchor_pos(pos, float(rd.circle_radius))
            t = float(obj.start_time)
            if first_time is None:
                first_time = t

            if isinstance(obj, Spinner):
                entry_pos = waypoints[-1].pos.copy() if waypoints else center.copy()
                spinner_guides = _spinner_guides(
                    float(obj.start_time),
                    float(obj.end_time),
                    entry_pos=entry_pos,
                    circle_radius=float(rd.circle_radius),
                )
                waypoints.extend(spinner_guides)
                constraints.append(_Waypoint(float(obj.start_time), spinner_guides[0].pos.copy(), "circle"))
                constraints.append(_Waypoint(float(obj.end_time), spinner_guides[-1].pos.copy(), "circle"))
                continue

            circle_wp = _Waypoint(t, support_pos, "circle")
            waypoints.append(circle_wp)
            constraints.append(_Waypoint(t, pos, "circle"))

            if not isinstance(obj, Slider):
                continue

            si = slider_obj_map.get(int(obj_idx), -1)
            if si < 0 or srd is None:
                continue

            slider_guides, slider_constraints = _slider_guides_from_ticks(
                beatmap, obj, srd, si, float(rd.circle_radius)
            )
            waypoints.extend(slider_guides)
            constraints.extend(slider_constraints)

        if not waypoints:
            return

        if first_time is None:
            first_time = waypoints[0].time_ms

        waypoints.insert(0, _Waypoint(first_time - 2000.0, center.copy(), "intro"))
        waypoints.append(_Waypoint(waypoints[-1].time_ms + 2000.0, waypoints[-1].pos.copy(), "outro"))

        waypoints = _merge_waypoints(waypoints)
        constraints = _merge_waypoints(constraints)
        waypoints = _flatten_stacks(waypoints)
        waypoints = _prune_stream_midpoints(waypoints)
        waypoints = _insert_gap_guides(waypoints)
        waypoints = _insert_pacing_waypoints(waypoints)

        target_radius = float(rd.circle_radius) * RADIUS_MARGIN
        current_waypoints = waypoints
        errors: list[tuple[_Waypoint, float, np.ndarray]] = []

        refine_passes = 0
        reversal_passes = 0
        while True:
            wp_times = np.array([wp.time_ms for wp in current_waypoints], dtype=np.float64)
            wp_pos = np.array([wp.pos for wp in current_waypoints], dtype=np.float64)
            for i in range(1, len(wp_times)):
                if wp_times[i] <= wp_times[i - 1]:
                    wp_times[i] = wp_times[i - 1] + 0.01

            sample_times, sample_pos = _evaluate_spline(wp_times, wp_pos)
            sample_pos = _soft_clip_positions(sample_pos)
            current_waypoints, edge_changed = _edge_turn_guides(
                current_waypoints, sample_times, sample_pos, float(rd.circle_radius)
            )
            if edge_changed and reversal_passes < REVERSAL_FIX_MAX_PASSES:
                reversal_passes += 1
                continue
            current_waypoints, reversal_changed = _reversal_guides(current_waypoints, sample_times, sample_pos)
            if reversal_changed and reversal_passes < REVERSAL_FIX_MAX_PASSES:
                reversal_passes += 1
                continue
            current_waypoints, loop_changed = _micro_loop_guides(current_waypoints, sample_times, sample_pos)
            if loop_changed and reversal_passes < REVERSAL_FIX_MAX_PASSES:
                reversal_passes += 1
                continue
            errors = _radius_errors(sample_times, sample_pos, constraints)
            current_waypoints, changed = _refine_waypoints(current_waypoints, errors, target_radius)
            if changed and refine_passes < REFINE_MAX_PASSES:
                refine_passes += 1
                continue
            if not changed or refine_passes >= REFINE_MAX_PASSES:
                break

        wp_times = np.array([wp.time_ms for wp in current_waypoints], dtype=np.float64)
        wp_pos = np.array([wp.pos for wp in current_waypoints], dtype=np.float64)
        for i in range(1, len(wp_times)):
            if wp_times[i] <= wp_times[i - 1]:
                wp_times[i] = wp_times[i - 1] + 0.01
        sample_times, sample_pos = _evaluate_spline(wp_times, wp_pos)
        sample_pos = _soft_clip_positions(sample_pos)
        errors = _radius_errors(sample_times, sample_pos, constraints)

        self._times = sample_times
        self._positions = sample_pos
        self._debug_waypoints = [
            {"time_ms": wp.time_ms, "pos": wp.pos.copy(), "kind": wp.kind}
            for wp in current_waypoints
        ]
        self._debug_constraints = [
            {"time_ms": wp.time_ms, "pos": wp.pos.copy(), "kind": wp.kind}
            for wp in constraints
        ]
        self._debug_radius_errors = [
            {
                "time_ms": constraint.time_ms,
                "distance": dist,
                "target": constraint.pos.copy(),
                "path_pos": path_pos.copy(),
                "kind": constraint.kind,
            }
            for constraint, dist, path_pos in errors
        ]
