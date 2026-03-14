"""Momentum-based cursor playstyle.

Distance-parameterized splines with adaptive easing produce smooth,
purposeful cursor motion.  Sliders are sampled along their actual path
so the cursor follows the curve naturally -- no 180-degree reversals.
"""

from __future__ import annotations

import bisect

import numpy as np
from scipy.interpolate import CubicSpline

from osu_map.beatmap import RenderData, SliderRenderData
from .base import CursorPlaystyle, clamp_to_playfield, OSU_W, OSU_H

SAMPLE_MS = 2.0
EASE_THRESHOLD_MS = 200.0
SLIDER_SAMPLE_INTERVAL_MS = 40.0
LONG_GAP_MS = 2500.0


def _ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 3 * 0.5


def _adaptive_ease(t: float, seg_duration_ms: float) -> float:
    """Linear for dense streams, cubic ease for longer gaps."""
    if seg_duration_ms <= EASE_THRESHOLD_MS:
        return t
    blend = min(1.0, (seg_duration_ms - EASE_THRESHOLD_MS) / 400.0)
    eased = _ease_in_out_cubic(t)
    return t + blend * (eased - t)


# ── slider path sampling ─────────────────────────────────────────────

def _slider_ball_pos(srd: SliderRenderData, si: int, frac: float) -> np.ndarray:
    ps = srd.path_starts[si]
    pc = srd.path_counts[si]
    fi = frac * (pc - 1)
    idx = min(int(fi), pc - 2)
    f = fi - idx
    return srd.path_points[ps + idx] * (1.0 - f) + srd.path_points[ps + idx + 1] * f


def _sample_slider_path(
    srd: SliderRenderData, si: int,
    t_start: float, duration: float, repeats: int,
) -> list[tuple[float, np.ndarray]]:
    """Sample points along a slider's full travel (all repeats).

    Returns evenly-spaced (in time) waypoints that follow the slider ball,
    so the spline tracks the actual slider shape with no 180-degree folds.
    """
    if duration < 1.0:
        return []

    n_samples = max(3, int(duration / SLIDER_SAMPLE_INTERVAL_MS) + 1)
    points: list[tuple[float, np.ndarray]] = []

    for k in range(n_samples + 1):
        frac_total = k / n_samples
        t = t_start + duration * frac_total

        progress = frac_total * repeats
        leg = int(progress)
        leg_frac = progress - leg
        if leg >= repeats:
            leg = repeats - 1
            leg_frac = 1.0

        if leg % 2 == 1:
            leg_frac = 1.0 - leg_frac

        pos = _slider_ball_pos(srd, si, leg_frac).astype(np.float64)
        points.append((t, pos))

    return points


# ── gap drift (keeps cursor on-screen during long breaks) ────────────

def _insert_gap_drifts(
    waypoints: list[tuple[float, np.ndarray]],
) -> list[tuple[float, np.ndarray]]:
    """During long gaps, insert a midpoint pulled toward playfield center."""
    center = np.array([OSU_W / 2, OSU_H / 2], dtype=np.float64)
    extras: list[tuple[float, np.ndarray]] = []

    for k in range(1, len(waypoints)):
        dt = waypoints[k][0] - waypoints[k - 1][0]
        if dt > LONG_GAP_MS:
            p_from = waypoints[k - 1][1]
            p_to = waypoints[k][1]
            mid = (p_from + p_to) * 0.5
            drift = mid * 0.6 + center * 0.4
            extras.append((
                waypoints[k - 1][0] + dt * 0.5,
                drift,
            ))

    waypoints.extend(extras)
    waypoints.sort(key=lambda w: w[0])
    return waypoints


# ── main class ───────────────────────────────────────────────────────

class MomentumPlaystyle(CursorPlaystyle):
    """Distance-parameterized cursor motion with adaptive easing."""

    def __init__(self, beatmap, render_data: RenderData):
        self._times: np.ndarray = np.array([], dtype=np.float64)
        self._positions: np.ndarray = np.zeros((0, 2), dtype=np.float64)
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
        return clamp_to_playfield(float(p[0]), float(p[1]))

    # ── build pipeline ───────────────────────────────────────────────

    def _build(self, beatmap, rd: RenderData):
        srd = rd.slider
        slider_time_map: dict[int, int] = {}
        if srd is not None:
            for i in range(srd.n_sliders):
                slider_time_map[int(srd.start_times[i])] = i

        waypoints = self._collect_waypoints(beatmap, srd, slider_time_map)
        if not waypoints:
            return

        waypoints.sort(key=lambda w: w[0])
        waypoints = _insert_gap_drifts(waypoints)

        first_t = waypoints[0][0]
        last_t = waypoints[-1][0]
        center = np.array([OSU_W / 2, OSU_H / 2], dtype=np.float64)

        waypoints.insert(0, (first_t - 2000.0, center.copy()))
        waypoints.append((last_t + 2000.0, waypoints[-1][1].copy()))

        # Deduplicate close timestamps
        merged: list[tuple[float, np.ndarray]] = [waypoints[0]]
        for t_val, p in waypoints[1:]:
            if abs(t_val - merged[-1][0]) < 0.5:
                merged[-1] = (merged[-1][0], (merged[-1][1] + p) * 0.5)
            else:
                merged.append((t_val, p))
        waypoints = merged

        wp_times = np.array([w[0] for w in waypoints], dtype=np.float64)
        wp_pos = np.array([w[1] for w in waypoints], dtype=np.float64)

        for k in range(1, len(wp_times)):
            if wp_times[k] <= wp_times[k - 1]:
                wp_times[k] = wp_times[k - 1] + 0.01

        # ── distance-parameterized spline ────────────────────────────
        diffs = np.diff(wp_pos, axis=0)
        seg_dists = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        seg_dists = np.maximum(seg_dists, 0.01)
        cum_dist = np.concatenate([[0.0], np.cumsum(seg_dists)])

        cs_x = CubicSpline(cum_dist, wp_pos[:, 0], bc_type='natural')
        cs_y = CubicSpline(cum_dist, wp_pos[:, 1], bc_type='natural')

        t_start = wp_times[0]
        t_end = wp_times[-1]

        sample_times = np.arange(t_start, t_end, SAMPLE_MS)
        sample_dists = np.empty_like(sample_times)

        seg_idx = 0
        n_wp = len(wp_times)
        for i, t in enumerate(sample_times):
            while seg_idx < n_wp - 2 and t >= wp_times[seg_idx + 1]:
                seg_idx += 1
            seg_t0 = wp_times[seg_idx]
            seg_t1 = wp_times[min(seg_idx + 1, n_wp - 1)]
            seg_d0 = cum_dist[seg_idx]
            seg_d1 = cum_dist[min(seg_idx + 1, n_wp - 1)]

            seg_dt = seg_t1 - seg_t0
            if seg_dt < 0.01:
                local_frac = 1.0
            else:
                local_frac = max(0.0, min(1.0, (t - seg_t0) / seg_dt))

            eased = _adaptive_ease(local_frac, seg_dt)
            sample_dists[i] = seg_d0 + (seg_d1 - seg_d0) * eased

        px = cs_x(sample_dists)
        py = cs_y(sample_dists)

        px = np.clip(px, -4.0, OSU_W + 4.0)
        py = np.clip(py, -4.0, OSU_H + 4.0)

        self._times = sample_times
        self._positions = np.column_stack([px, py])

    # ── waypoint collection ──────────────────────────────────────────

    def _collect_waypoints(self, beatmap, srd, slider_time_map):
        from osupyparser.osu.objects import Slider

        waypoints: list[tuple[float, np.ndarray]] = []

        for obj in beatmap.hit_objects:
            pos = np.array(
                [float(obj.pos.x), OSU_H - float(obj.pos.y)], dtype=np.float64
            )
            t = float(obj.start_time)

            if isinstance(obj, Slider):
                si = slider_time_map.get(int(obj.start_time), -1)
                waypoints.append((t, pos))

                if si >= 0 and srd is not None:
                    dur = float(obj.end_time) - t
                    repeats = int(srd.repeat_counts[si])

                    path_pts = _sample_slider_path(srd, si, t, dur, repeats)
                    waypoints.extend(path_pts)
                else:
                    waypoints.append((float(obj.end_time), pos.copy()))
            else:
                waypoints.append((t, pos))

        return waypoints
