"""Danser-style cursor path precomputation.

Uses a global CubicSpline fitted over (time -> position) to produce
a C2-continuous trajectory through all hit objects.  Slider bodies use
only head / reverse / end waypoints to avoid spline overshoot while
keeping the motion smooth and fluid.
"""

from __future__ import annotations

import bisect

import numpy as np
from scipy.interpolate import CubicSpline

from osu_map.beatmap import RenderData, SliderRenderData
from speedups import interpolate_cursor_query

SAMPLE_MS = 2.0
OSU_H = 384


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


class CursorPath:
    """Pre-computed danser cursor trajectory."""

    def __init__(self, beatmap, render_data: RenderData):
        self._times = np.array([], dtype=np.float64)
        self._positions = np.zeros((0, 2), dtype=np.float64)
        self._last_idx = 0
        self._build(beatmap, render_data)

    def position_at(self, time_ms: float) -> np.ndarray:
        n = len(self._times)
        if n == 0:
            return np.array([256.0, OSU_H / 2], dtype=np.float32)
        fast_result = interpolate_cursor_query(
            self._times,
            self._positions[:, 0],
            self._positions[:, 1],
            time_ms,
            self._last_idx,
        )
        if fast_result is not None:
            idx, x, y = fast_result
            self._last_idx = int(idx)
            return np.array([x, y], dtype=np.float32)
        idx = max(0, min(self._last_idx, n))
        if idx > 0 and time_ms < self._times[idx - 1]:
            idx = bisect.bisect_right(self._times, time_ms, 0, idx)
        elif idx < n and time_ms >= self._times[idx]:
            idx = bisect.bisect_right(self._times, time_ms, idx, n)
        self._last_idx = idx
        if idx <= 0:
            return self._positions[0].astype(np.float32)
        if idx >= n:
            return self._positions[-1].astype(np.float32)
        t0, t1 = self._times[idx - 1], self._times[idx]
        dt = t1 - t0
        if dt < 0.01:
            return self._positions[idx].astype(np.float32)
        frac = (time_ms - t0) / dt
        result = self._positions[idx - 1] * (1.0 - frac) + self._positions[idx] * frac
        return result.astype(np.float32)

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

        waypoints.insert(0, (first_time - 2000.0, center.copy()))
        waypoints.append((waypoints[-1][0] + 2000.0, waypoints[-1][1].copy()))

        # Deduplicate: merge waypoints at the same time
        merged: list[tuple[float, np.ndarray]] = [waypoints[0]]
        for t_val, p in waypoints[1:]:
            if abs(t_val - merged[-1][0]) < 0.5:
                merged[-1] = (merged[-1][0], (merged[-1][1] + p) * 0.5)
            else:
                merged.append((t_val, p))
        waypoints = merged

        wp_times = np.array([w[0] for w in waypoints], dtype=np.float64)
        wp_x = np.array([w[1][0] for w in waypoints], dtype=np.float64)
        wp_y = np.array([w[1][1] for w in waypoints], dtype=np.float64)

        # Ensure strictly increasing times
        for i in range(1, len(wp_times)):
            if wp_times[i] <= wp_times[i - 1]:
                wp_times[i] = wp_times[i - 1] + 0.01

        cs_x = CubicSpline(wp_times, wp_x, bc_type='natural')
        cs_y = CubicSpline(wp_times, wp_y, bc_type='natural')

        t_start = wp_times[0]
        t_end = wp_times[-1]
        sample_times = np.arange(t_start, t_end, SAMPLE_MS)

        px = cs_x(sample_times)
        py = cs_y(sample_times)

        self._times = sample_times
        self._positions = np.column_stack([px, py])
