"""Hit judgment -- determines hit/miss for each object using replay data."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from osupyparser.osu.objects import Slider, Spinner

from replay.data import KEY_K1, KEY_K2, KEY_M1, KEY_M2, ReplayData, KeyPress
from replay.spinner import (
    MAX_SPINNER_RPS,
    SPINNER_CENTER_X,
    SPINNER_CENTER_Y,
    SpinnerAnalysis,
    SpinnerSpinEvent,
    spinner_required_spins,
    spinner_result,
)

OSU_H = 384


@dataclass
class HitResult:
    """Judgment for a single hit object."""
    obj_index: int
    time_ms: float
    result: str          # "300", "100", "50", "miss"
    hit_time_ms: float   # actual press time, -1 for miss
    spinner: SpinnerAnalysis | None = None


def hit_windows(od: float) -> tuple[float, float, float]:
    """Return (window_300, window_100, window_50) in ms from OD."""
    return (
        math.floor(80.0 - 6.0 * od) - 0.5,
        math.floor(140.0 - 8.0 * od) - 0.5,
        math.floor(200.0 - 10.0 * od) - 0.5,
    )


class HitJudge:
    """Pre-computes hit/miss results for an entire replay against a beatmap.

    Circles and slider heads follow stable hit windows. Spinner judgement is
    reconstructed from replay cursor rotation while an action key is held.
    """

    def __init__(self, hit_objects: list, od: float, circle_radius: float,
                 replay: ReplayData):
        self._objects = hit_objects
        self._od = od
        self._radius = circle_radius
        self._replay = replay

        w300, w100, w50 = hit_windows(od)
        self._w300 = w300
        self._w100 = w100
        self._w50 = w50

        self.results: list[HitResult] = []
        self._results_by_idx: dict[int, HitResult] = {}
        self._missed_indices: set[int] = set()
        self._hit_times: dict[int, float] = {}

        self._judge_all()

    @property
    def missed_indices(self) -> set[int]:
        return self._missed_indices

    def result_for(self, obj_index: int) -> HitResult | None:
        return self._results_by_idx.get(obj_index)

    def _judge_all(self) -> None:
        presses = self._build_action_presses()
        press_idx = 0

        for obj_idx, obj in enumerate(self._objects):
            if isinstance(obj, Spinner):
                analysis = self._analyse_spinner(obj)
                result = spinner_result(analysis.total_spins, analysis.required_spins)
                hit_time = (
                    float(analysis.first_active_time_ms)
                    if analysis.first_active_time_ms is not None and result != "miss"
                    else -1.0
                )
                r = HitResult(
                    obj_idx,
                    float(obj.end_time),
                    result,
                    hit_time,
                    spinner=analysis,
                )
                self.results.append(r)
                self._results_by_idx[obj_idx] = r
                if hit_time >= 0.0:
                    self._hit_times[obj_idx] = hit_time
                if result == "miss":
                    self._missed_indices.add(obj_idx)
                continue

            obj_time = float(obj.start_time)
            obj_x = float(obj.pos.x)
            obj_y = float(obj.pos.y)
            max_window = self._w50
            if isinstance(obj, Slider):
                # Stable slider heads are binary: any hit within the hit50 window counts.
                result_windows = (self._w50, self._w50, self._w50)
            else:
                result_windows = (self._w300, self._w100, self._w50)

            window_start = obj_time - max_window
            window_end = obj_time + max_window

            hit_found = False

            while press_idx < len(presses) and presses[press_idx].time_ms < window_start:
                press_idx += 1

            search_idx = press_idx
            while search_idx < len(presses) and presses[search_idx].time_ms <= window_end:
                p = presses[search_idx]
                px, py = self._replay.position_at(p.time_ms)
                dx = px - obj_x
                dy = py - obj_y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist <= self._radius:
                    dt = abs(p.time_ms - obj_time)
                    if dt <= result_windows[0]:
                        result = "300"
                    elif dt <= result_windows[1]:
                        result = "100"
                    else:
                        result = "50"

                    r = HitResult(obj_idx, obj_time, result, p.time_ms)
                    self.results.append(r)
                    self._results_by_idx[obj_idx] = r
                    self._hit_times[obj_idx] = p.time_ms
                    hit_found = True
                    press_idx = search_idx + 1
                    break

                search_idx += 1

            if not hit_found:
                press_idx = search_idx
                r = HitResult(obj_idx, obj_time, "miss", -1.0)
                self.results.append(r)
                self._results_by_idx[obj_idx] = r
                self._missed_indices.add(obj_idx)

    def _build_action_presses(self) -> list[KeyPress]:
        """Stable osu! clicks are based on left/right action groups, not raw key bits."""
        presses: list[KeyPress] = []
        prev_left = False
        prev_right = False
        for frame in self._replay.frames:
            left = bool(frame.keys & (KEY_M1 | KEY_K1))
            right = bool(frame.keys & (KEY_M2 | KEY_K2))
            new_keys = 0
            if left and not prev_left:
                new_keys |= KEY_M1
            if right and not prev_right:
                new_keys |= KEY_M2
            if new_keys:
                presses.append(KeyPress(frame.time_ms, new_keys))
            prev_left = left
            prev_right = right
        return presses

    def hitsound_events_for_hit(self, obj_index: int) -> float | None:
        """Return the actual hit time for an object, or None if missed."""
        return self._hit_times.get(obj_index)

    def _analyse_spinner(self, obj: Spinner) -> SpinnerAnalysis:
        start_time = float(obj.start_time)
        end_time = float(obj.end_time)
        duration_ms = max(0.0, end_time - start_time)
        required_spins = spinner_required_spins(duration_ms, self._od)
        if duration_ms <= 0.0:
            return SpinnerAnalysis(required_spins=required_spins)

        samples: list[tuple[float, float, float, int]] = [
            (
                start_time,
                *self._replay.position_at(start_time),
                int(self._replay.keys_at(start_time)),
            )
        ]
        for frame in self._replay.frames:
            if start_time < frame.time_ms < end_time:
                samples.append((float(frame.time_ms), float(frame.x), float(frame.y), int(frame.keys)))
        samples.append(
            (
                end_time,
                *self._replay.position_at(end_time),
                int(self._replay.keys_at(end_time)),
            )
        )

        total_rotation_deg = 0.0
        first_active_time_ms: float | None = None
        bonus_events: list[SpinnerSpinEvent] = []

        for (t0, x0, y0, keys0), (t1, x1, y1, _keys1) in zip(samples, samples[1:]):
            dt = t1 - t0
            if dt <= 1e-6 or not (keys0 & (KEY_M1 | KEY_M2 | KEY_K1 | KEY_K2)):
                continue

            if first_active_time_ms is None:
                first_active_time_ms = t0

            a0 = math.degrees(math.atan2(y0 - SPINNER_CENTER_Y, x0 - SPINNER_CENTER_X))
            a1 = math.degrees(math.atan2(y1 - SPINNER_CENTER_Y, x1 - SPINNER_CENTER_X))
            delta_deg = ((a1 - a0 + 180.0) % 360.0) - 180.0
            delta_deg = abs(delta_deg)
            delta_deg = min(delta_deg, MAX_SPINNER_RPS * 360.0 * (dt / 1000.0))
            if delta_deg <= 1e-6:
                continue

            prev_rotation = total_rotation_deg
            total_rotation_deg += delta_deg

            prev_full_spins = int((prev_rotation + 1e-6) // 360.0)
            new_full_spins = int((total_rotation_deg + 1e-6) // 360.0)
            for spin_idx in range(prev_full_spins + 1, new_full_spins + 1):
                threshold_deg = spin_idx * 360.0
                frac = (threshold_deg - prev_rotation) / delta_deg
                event_time = t0 + dt * max(0.0, min(1.0, frac))
                score = 100 if spin_idx <= required_spins else 1100
                bonus_events.append(SpinnerSpinEvent(event_time, score))

        total_spins = total_rotation_deg / 360.0
        return SpinnerAnalysis(
            total_rotation_deg=total_rotation_deg,
            total_spins=total_spins,
            full_spins=int(total_spins + 1e-6),
            required_spins=required_spins,
            bonus_events=bonus_events,
            first_active_time_ms=first_active_time_ms,
        )
