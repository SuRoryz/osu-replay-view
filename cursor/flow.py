"""Flow-direction cursor playstyle.

Uses Catmull-Rom / Hermite splines with flow-direction tangents so the curve
naturally "flows through" each waypoint instead of turning sharply.  Eased
segment progress adds organic acceleration.  Same waypoint logic as Danser
(head/reverse/end for sliders, orbital arcs for stacks, gap drift).
"""

from __future__ import annotations

import bisect
import math

import numpy as np

from osu_map.beatmap import RenderData, SliderRenderData
from .base import CursorPlaystyle, clamp_to_playfield, OSU_W, OSU_H

SAMPLE_MS = 2.0
STACK_DIST = 20.0
STACK_ORBIT_RADIUS = 35.0
LONG_GAP_MS = 1500.0
TENSION = 0.5


def _ease_in_out_quad(t: float) -> float:
    """Smooth ease for segment progress."""
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 * 0.5


# ── Hermite basis (Catmull-Rom) ───────────────────────────────────────

def _hermite(p0: np.ndarray, m0: np.ndarray, p1: np.ndarray, m1: np.ndarray,
             t: float) -> np.ndarray:
    """Cubic Hermite interpolation between p0 and p1 with tangents m0, m1."""
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


# ── slider helpers ───────────────────────────────────────────────────

def _slider_ball_pos(srd: SliderRenderData, si: int, frac: float) -> np.ndarray:
    ps = srd.path_starts[si]
    pc = srd.path_counts[si]
    fi = frac * (pc - 1)
    idx = min(int(fi), pc - 2)
    f = fi - idx
    return srd.path_points[ps + idx] * (1.0 - f) + srd.path_points[ps + idx + 1] * f


def _slider_end_pos(srd: SliderRenderData, si: int) -> np.ndarray:
    ps = srd.path_starts[si]
    pc = srd.path_counts[si]
    rc = int(srd.repeat_counts[si])
    if rc % 2 == 0:
        return srd.path_points[ps].copy()
    return srd.path_points[ps + pc - 1].copy()


def _slider_head(srd: SliderRenderData, si: int) -> np.ndarray:
    return srd.path_points[srd.path_starts[si]].copy()


def _slider_tail(srd: SliderRenderData, si: int) -> np.ndarray:
    ps = srd.path_starts[si]
    pc = srd.path_counts[si]
    return srd.path_points[ps + pc - 1].copy()


def _insert_orbit(waypoints: list[tuple[float, np.ndarray]],
                  center: np.ndarray, t_start: float, t_end: float,
                  radius: float = STACK_ORBIT_RADIUS, n_pts: int = 6):
    dur = t_end - t_start
    if dur < 1.0:
        return
    for i in range(n_pts):
        frac = (i + 1) / (n_pts + 1)
        angle = frac * 2.0 * math.pi
        ox = center[0] + radius * math.cos(angle)
        oy = center[1] + radius * math.sin(angle)
        waypoints.append((t_start + dur * frac,
                          np.array([ox, oy], dtype=np.float64)))


def _insert_gap_drift(waypoints: list[tuple[float, np.ndarray]],
                      p_from: np.ndarray, p_to: np.ndarray,
                      t_from: float, t_to: float):
    mid_t = (t_from + t_to) * 0.5
    mid_p = (p_from + p_to) * 0.5
    playfield_center = np.array([OSU_W / 2, OSU_H / 2], dtype=np.float64)
    drift_target = mid_p * 0.6 + playfield_center * 0.4
    waypoints.append((mid_t, drift_target))


# ── main class ───────────────────────────────────────────────────────

class FlowPlaystyle(CursorPlaystyle):
    """Catmull-Rom splines with flow-direction tangents and eased progress."""

    def __init__(self, beatmap, render_data: RenderData):
        self._times = np.array([], dtype=np.float64)
        self._positions = np.zeros((0, 2), dtype=np.float64)
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

    def _build(self, beatmap, rd: RenderData):
        from osupyparser.osu.objects import Slider

        srd = rd.slider
        slider_time_map: dict[int, int] = {}
        if srd is not None:
            for i in range(srd.n_sliders):
                slider_time_map[int(srd.start_times[i])] = i

        waypoints: list[tuple[float, np.ndarray]] = []
        center = np.array([256.0, OSU_H / 2], dtype=np.float64)

        first_time = None
        for obj in beatmap.hit_objects:
            pos = np.array(
                [float(obj.pos.x), OSU_H - float(obj.pos.y)], dtype=np.float64
            )
            t = float(obj.start_time)
            if first_time is None:
                first_time = t

            if isinstance(obj, Slider):
                si = slider_time_map.get(int(obj.start_time), -1)
                waypoints.append((t, pos))

                if si >= 0 and srd is not None:
                    dur = float(obj.end_time) - t
                    repeats = int(srd.repeat_counts[si])
                    head = _slider_head(srd, si).astype(np.float64)
                    tail = _slider_tail(srd, si).astype(np.float64)

                    if repeats > 1 and dur > 10:
                        leg_dur = dur / repeats
                        for r in range(1, repeats):
                            rev_t = t + leg_dur * r
                            rev_pos = tail if r % 2 == 1 else head
                            waypoints.append((rev_t, rev_pos.copy()))
                    elif dur > 60:
                        mid_pos = _slider_ball_pos(srd, si, 0.5)
                        waypoints.append((t + dur * 0.5, mid_pos.astype(np.float64)))

                    end_pos = _slider_end_pos(srd, si)
                    waypoints.append((float(obj.end_time), end_pos.astype(np.float64)))
                else:
                    waypoints.append((float(obj.end_time), pos.copy()))
            else:
                waypoints.append((t, pos))

        if not waypoints:
            return

        waypoints.sort(key=lambda w: w[0])

        if first_time is None:
            first_time = waypoints[0][0]

        # Stack resolution: orbital arcs
        resolved: list[tuple[float, np.ndarray]] = []
        i = 0
        while i < len(waypoints):
            cluster = [waypoints[i]]
            j = i + 1
            while j < len(waypoints):
                dist = np.linalg.norm(waypoints[j][1] - cluster[0][1])
                if dist < STACK_DIST and (waypoints[j][0] - cluster[-1][0]) < 400:
                    cluster.append(waypoints[j])
                    j += 1
                else:
                    break

            if len(cluster) >= 3:
                stack_center = np.mean([c[1] for c in cluster], axis=0)
                t_s, t_e = cluster[0][0], cluster[-1][0]
                resolved.append((t_s, stack_center.copy()))
                _insert_orbit(resolved, stack_center, t_s, t_e)
                resolved.append((t_e, stack_center.copy()))
            else:
                resolved.extend(cluster)
            i = j

        waypoints = resolved

        # Long gaps: drift waypoints
        gap_inserts: list[tuple[float, np.ndarray]] = []
        for k in range(1, len(waypoints)):
            dt = waypoints[k][0] - waypoints[k - 1][0]
            if dt > LONG_GAP_MS:
                _insert_gap_drift(gap_inserts,
                                  waypoints[k - 1][1], waypoints[k][1],
                                  waypoints[k - 1][0], waypoints[k][0])
        waypoints.extend(gap_inserts)
        waypoints.sort(key=lambda w: w[0])

        waypoints.insert(0, (first_time - 2000.0, center.copy()))
        waypoints.append((waypoints[-1][0] + 2000.0, waypoints[-1][1].copy()))

        merged: list[tuple[float, np.ndarray]] = [waypoints[0]]
        for t_val, p in waypoints[1:]:
            if abs(t_val - merged[-1][0]) < 0.5:
                merged[-1] = (merged[-1][0], (merged[-1][1] + p) * 0.5)
            else:
                merged.append((t_val, p))
        waypoints = merged

        wp_times = np.array([w[0] for w in waypoints], dtype=np.float64)
        wp_pos = np.array([w[1] for w in waypoints], dtype=np.float64)
        n = len(waypoints)

        for k in range(1, len(wp_times)):
            if wp_times[k] <= wp_times[k - 1]:
                wp_times[k] = wp_times[k - 1] + 0.01

        # Flow-direction tangents (Catmull-Rom style)
        tangents = np.zeros_like(wp_pos)
        for i in range(n):
            if i == 0:
                tangents[i] = (wp_pos[1] - wp_pos[0]) * TENSION
            elif i == n - 1:
                tangents[i] = (wp_pos[n - 1] - wp_pos[n - 2]) * TENSION
            else:
                tangents[i] = (wp_pos[i + 1] - wp_pos[i - 1]) * TENSION

        t_start = wp_times[0]
        t_end = wp_times[-1]
        sample_times = np.arange(t_start, t_end, SAMPLE_MS)

        px = np.empty(len(sample_times), dtype=np.float64)
        py = np.empty(len(sample_times), dtype=np.float64)

        seg_idx = 0
        for i, t in enumerate(sample_times):
            while seg_idx < n - 2 and t >= wp_times[seg_idx + 1]:
                seg_idx += 1

            seg_t0 = wp_times[seg_idx]
            seg_t1 = wp_times[seg_idx + 1]
            seg_dt = seg_t1 - seg_t0

            if seg_dt < 0.01:
                px[i] = wp_pos[seg_idx + 1, 0]
                py[i] = wp_pos[seg_idx + 1, 1]
                continue

            raw_frac = (t - seg_t0) / seg_dt
            raw_frac = max(0.0, min(1.0, raw_frac))
            frac = _ease_in_out_quad(raw_frac)

            p0 = wp_pos[seg_idx]
            p1 = wp_pos[seg_idx + 1]
            m0 = tangents[seg_idx]
            m1 = tangents[seg_idx + 1]

            pt = _hermite(p0, m0, p1, m1, frac)
            px[i] = pt[0]
            py[i] = pt[1]

        px = np.clip(px, -4.0, OSU_W + 4.0)
        py = np.clip(py, -4.0, OSU_H + 4.0)

        self._times = sample_times
        self._positions = np.column_stack([px, py])
