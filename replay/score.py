"""Stable score timeline + live PP helpers for gameplay HUD."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np
import rosu_pp_py as rosu
from osupyparser.osu.objects import Slider, Spinner

from osu_map import Beatmap
from osu_map.curves import sample_curve
from osu_map.mods import normalize_mods, score_multiplier
from replay.data import KEY_ANY, ReplayData
from replay.judge import HitJudge, hit_windows
from replay.spinner import SpinnerAnalysis, build_auto_spinner_analysis

FOLLOW_AREA_MULTIPLIER = 2.4
TAIL_LENIENCY_MS = 36.0
MINIMUM_HEALTH_ERROR = 0.01
MIN_HEALTH_TARGET = 0.99
MID_HEALTH_TARGET = 0.90
MAX_HEALTH_TARGET = 0.40
COMBO_RESULT_NONE = 0
COMBO_RESULT_GOOD = 1
COMBO_RESULT_PERFECT = 2


@dataclass(slots=True)
class ScorePoint:
    """Absolute gameplay state after a scoring-related event."""

    time_ms: float
    score: int
    combo: int
    max_combo: int
    n300: int
    n100: int
    n50: int
    misses: int
    passed_objects: int
    accuracy: float
    pp: float
    last_hit_time_ms: float
    hp: float
    combo_break: bool = False
    eliminated: bool = False


class StablePerformanceTimeline:
    """Precomputed stable-style combo and PP state over wall-clock time."""

    def __init__(
        self,
        points: list[ScorePoint],
        *,
        drain_rate_per_ms: float,
        drain_start_time_ms: float,
        no_drain_periods: list[tuple[float, float]],
    ):
        self.points = points
        self.times = [point.time_ms for point in points]
        self.max_combo = points[-1].max_combo if points else 0
        self.drain_rate_per_ms = max(0.0, float(drain_rate_per_ms))
        self.drain_start_time_ms = float(drain_start_time_ms)
        self.no_drain_periods = [
            (float(start), float(end))
            for start, end in no_drain_periods
            if end > start
        ]

    @classmethod
    def build(
        cls,
        *,
        beatmap: Beatmap,
        beatmap_path: str,
        mods: int,
        clock_rate: float,
        circle_radius: float,
        od: float,
        hp: float,
        replay: ReplayData | None,
        judge: HitJudge | None,
        eliminate_on_miss: bool = False,
    ) -> "StablePerformanceTimeline":
        builder = _StableTimelineBuilder(
            beatmap=beatmap,
            beatmap_path=beatmap_path,
            mods=mods,
            clock_rate=max(0.05, float(clock_rate)),
            circle_radius=float(circle_radius),
            od=float(od),
            hp=float(hp),
            replay=replay,
            judge=judge,
            eliminate_on_miss=bool(eliminate_on_miss),
        )
        return cls(
            builder.build(),
            drain_rate_per_ms=builder.drain_rate_per_ms,
            drain_start_time_ms=builder.drain_start_time_ms,
            no_drain_periods=builder.no_drain_periods,
        )

    def point_at(self, time_ms: float) -> ScorePoint | None:
        idx = bisect.bisect_right(self.times, time_ms) - 1
        if idx < 0:
            return None
        return self.points[idx]

    def hp_at(self, time_ms: float, *, point: ScorePoint | None = None) -> float:
        if time_ms <= self.drain_start_time_ms:
            return 1.0
        if point is None:
            point = self.point_at(time_ms)
        if point is None:
            drained = self._drain_elapsed_between(self.drain_start_time_ms, time_ms) * self.drain_rate_per_ms
            return _clamp_hp(1.0 - drained)
        drained = self._drain_elapsed_between(point.time_ms, time_ms) * self.drain_rate_per_ms
        return _clamp_hp(point.hp - drained)

    def _drain_elapsed_between(self, start_time_ms: float, end_time_ms: float) -> float:
        if end_time_ms <= start_time_ms:
            return 0.0
        drained = end_time_ms - start_time_ms
        for period_start, period_end in self.no_drain_periods:
            overlap_start = max(start_time_ms, period_start)
            overlap_end = min(end_time_ms, period_end)
            if overlap_end > overlap_start:
                drained -= overlap_end - overlap_start
        return max(0.0, drained)


@dataclass(slots=True)
class _SliderEvent:
    time_ms: float
    progress: float
    kind: str


class _StableTimelineBuilder:
    def __init__(
        self,
        *,
        beatmap: Beatmap,
        beatmap_path: str,
        mods: int,
        clock_rate: float,
        circle_radius: float,
        od: float,
        hp: float,
        replay: ReplayData | None,
        judge: HitJudge | None,
        eliminate_on_miss: bool,
    ):
        self._beatmap = beatmap
        self._clock_rate = clock_rate
        self._circle_radius = circle_radius
        self._follow_radius = circle_radius * FOLLOW_AREA_MULTIPLIER
        self._od = od
        self._hp_difficulty = max(0.0, min(10.0, float(hp)))
        self._replay = replay
        self._judge = judge
        self._eliminate_on_miss = eliminate_on_miss
        self._summary_correction_allowed = (
            replay is not None and normalize_mods(replay.mods) == normalize_mods(mods)
        )

        self._points: list[ScorePoint] = []
        self._score = 0
        self._combo = 0
        self._max_combo = 0
        self._n300 = 0
        self._n100 = 0
        self._n50 = 0
        self._misses = 0
        self._passed_objects = 0
        self._pp = 0.0
        self._last_hit_time_ms = -1e9
        self._hp = 1.0
        self._eliminated = False
        self._combo_result = COMBO_RESULT_PERFECT
        self._difficulty_multiplier = self._compute_difficulty_multiplier()
        self._mod_multiplier = score_multiplier(mods)
        self._drain_start_time_ms = self._first_hitobject_time_ms()
        self._no_drain_periods = self._compute_no_drain_periods()
        self._object_last_in_combo = self._compute_last_in_combo_flags()
        self._drain_rate_per_ms = self._calibrate_drain_rate()
        self._last_hp_time_ms = self._drain_start_time_ms

        self._rosu_state = rosu.ScoreState()
        self._rosu_perf = None
        try:
            rosu_map = rosu.Beatmap(path=beatmap_path)
            diff = rosu.Difficulty(mods=mods, lazer=False)
            self._rosu_perf = diff.gradual_performance(rosu_map)
        except Exception:
            self._rosu_perf = None

    @property
    def drain_rate_per_ms(self) -> float:
        return self._drain_rate_per_ms

    @property
    def drain_start_time_ms(self) -> float:
        return self._drain_start_time_ms

    @property
    def no_drain_periods(self) -> list[tuple[float, float]]:
        return self._no_drain_periods

    def build(self) -> list[ScorePoint]:
        for obj_idx, obj in enumerate(self._beatmap.hit_objects):
            if self._eliminated:
                break
            if isinstance(obj, Slider):
                self._process_slider(obj_idx, obj)
            elif isinstance(obj, Spinner):
                self._process_spinner(obj_idx, obj)
            else:
                self._process_circle(obj_idx, obj)

        self._apply_replay_summary_correction()
        return self._points

    def _compute_difficulty_multiplier(self) -> int:
        hit_objects = len(self._beatmap.hit_objects)
        if hit_objects <= 0:
            return 1
        first_time = float(self._beatmap.hit_objects[0].start_time)
        last_obj = self._beatmap.hit_objects[-1]
        last_time = float(getattr(last_obj, "end_time", last_obj.start_time))
        drain_time_s = max(1.0, (last_time - first_time) / 1000.0)
        density = max(0.0, min(hit_objects / drain_time_s * 8.0, 16.0))
        raw = (
            float(self._beatmap.hp)
            + float(self._beatmap.cs)
            + float(self._beatmap.od)
            + density
        ) / 38.0 * 5.0
        return max(1, int(round(raw)))

    def _wall_time(self, original_time_ms: float) -> float:
        return original_time_ms / self._clock_rate

    def _first_hitobject_time_ms(self) -> float:
        if not self._beatmap.hit_objects:
            return 0.0
        return self._wall_time(float(self._beatmap.hit_objects[0].start_time))

    def _compute_no_drain_periods(self) -> list[tuple[float, float]]:
        periods: list[tuple[float, float]] = []
        break_times = getattr(self._beatmap, "break_times", [])
        if not break_times:
            return periods

        hit_objects = self._beatmap.hit_objects
        if not hit_objects:
            return periods

        for break_start, break_end in break_times:
            last_end = float("-inf")
            next_start = float("inf")
            for obj in hit_objects:
                obj_start = float(obj.start_time)
                obj_end = float(getattr(obj, "end_time", obj.start_time))
                if obj_end <= break_start:
                    last_end = max(last_end, obj_end)
                if obj_start >= break_end:
                    next_start = min(next_start, obj_start)
            periods.append((self._wall_time(last_end), self._wall_time(next_start)))
        return periods

    def _compute_last_in_combo_flags(self) -> list[bool]:
        hit_objects = self._beatmap.hit_objects
        count = len(hit_objects)
        if count == 0:
            return []
        flags = [False] * count
        for idx, obj in enumerate(hit_objects):
            if idx == count - 1:
                flags[idx] = True
                continue
            next_obj = hit_objects[idx + 1]
            flags[idx] = (
                not isinstance(next_obj, Spinner)
                and (bool(getattr(next_obj, "new_combo", False)) or isinstance(obj, Spinner))
            )
        return flags

    def _starts_new_combo(self, obj_idx: int) -> bool:
        if obj_idx < 0 or obj_idx >= len(self._beatmap.hit_objects):
            return False
        obj = self._beatmap.hit_objects[obj_idx]
        if isinstance(obj, Spinner):
            return False
        if obj_idx == 0:
            return True
        prev_obj = self._beatmap.hit_objects[obj_idx - 1]
        return bool(getattr(obj, "new_combo", False)) or isinstance(prev_obj, Spinner)

    def _begin_object(self, obj_idx: int) -> None:
        if self._starts_new_combo(obj_idx):
            self._combo_result = COMBO_RESULT_PERFECT

    def _combo_end_hp_bonus(self, obj_idx: int, is_hit: bool) -> float:
        if not is_hit:
            return 0.0
        if not (0 <= obj_idx < len(self._object_last_in_combo)):
            return 0.0
        if not self._object_last_in_combo[obj_idx]:
            return 0.0
        if self._combo_result == COMBO_RESULT_PERFECT:
            return 0.07
        if self._combo_result == COMBO_RESULT_GOOD:
            return 0.05
        return 0.03

    def _update_combo_result_for_object_result(self, result: str) -> None:
        if result == "100":
            self._combo_result = min(self._combo_result, COMBO_RESULT_GOOD)
        elif result in {"50", "miss"}:
            self._combo_result = COMBO_RESULT_NONE

    def _update_combo_result_for_tick(self, *, slider_tail_miss: bool = False, hit: bool = True) -> None:
        if not hit:
            next_result = COMBO_RESULT_GOOD
            if slider_tail_miss:
                next_result = COMBO_RESULT_GOOD
            self._combo_result = min(self._combo_result, next_result)

    def _drain_elapsed_between(self, start_time_ms: float, end_time_ms: float) -> float:
        if end_time_ms <= start_time_ms:
            return 0.0
        drained = end_time_ms - start_time_ms
        for period_start, period_end in self._no_drain_periods:
            overlap_start = max(start_time_ms, period_start)
            overlap_end = min(end_time_ms, period_end)
            if overlap_end > overlap_start:
                drained -= overlap_end - overlap_start
        return max(0.0, drained)

    def _hp_miss(self) -> float:
        return _difficulty_range(self._hp_difficulty, -0.03, -0.125, -0.2)

    def _hp_tick_miss(self) -> float:
        return _difficulty_range(self._hp_difficulty, -0.02, -0.075, -0.14)

    @staticmethod
    def _hp_object_result(result: str) -> float:
        if result == "300":
            return 0.03
        if result == "100":
            return 0.011
        if result == "50":
            return 0.002
        return 0.0

    @staticmethod
    def _hp_slider_part(kind: str) -> float:
        if kind == "tick":
            return 0.015
        return 0.02

    @staticmethod
    def _hp_spinner_spin() -> float:
        return 0.015

    def _ideal_hp_events(self) -> list[tuple[float, float]]:
        events: list[tuple[float, float]] = []
        for obj_idx, obj in enumerate(self._beatmap.hit_objects):
            if isinstance(obj, Slider):
                start_time = float(obj.start_time)
                _base_bl, sv, _, _ = self._beatmap._timing_at(start_time)
                px_per_beat = self._beatmap.slider_multiplier * 100.0 * sv
                if getattr(self._beatmap, "file_version", 14) < 8 and abs(sv) > 1e-6:
                    px_per_beat /= sv
                if px_per_beat <= 0:
                    px_per_beat = 1.0

                span_count = max(int(obj.repeat_count), 1)
                span_duration = float(obj.duration) / span_count
                total_duration = float(obj.duration)
                tail_judge_time = max(
                    start_time + total_duration * 0.5,
                    start_time + total_duration - TAIL_LENIENCY_MS,
                )
                events.append((self._wall_time(start_time), self._hp_slider_part("head")))
                for event in self._slider_nested_events(
                    slider=obj,
                    px_per_beat=px_per_beat,
                    span_count=span_count,
                    span_duration=span_duration,
                    tail_judge_time=tail_judge_time,
                ):
                    part_kind = "tick" if event.kind == "tick" else event.kind
                    events.append((self._wall_time(event.time_ms), self._hp_slider_part(part_kind)))
                delta = self._hp_object_result("300")
                if self._object_last_in_combo[obj_idx]:
                    delta += 0.07
                events.append((self._wall_time(tail_judge_time), delta))
            elif isinstance(obj, Spinner):
                analysis = build_auto_spinner_analysis(
                    float(obj.start_time),
                    float(obj.end_time),
                    self._od,
                )
                for event in analysis.bonus_events:
                    events.append((self._wall_time(event.time_ms), self._hp_spinner_spin()))
                delta = self._hp_object_result("300")
                if self._object_last_in_combo[obj_idx]:
                    delta += 0.07
                events.append((self._wall_time(float(obj.end_time)), delta))
            else:
                delta = self._hp_object_result("300")
                if self._object_last_in_combo[obj_idx]:
                    delta += 0.07
                events.append((self._wall_time(float(obj.start_time)), delta))
        events.sort(key=lambda item: item[0])
        return events

    def _simulate_min_hp(self, events: list[tuple[float, float]], drain_rate_per_ms: float) -> float:
        if not events:
            return 1.0
        hp = 1.0
        min_hp = 1.0
        last_time = self._drain_start_time_ms
        for time_ms, delta in events:
            if time_ms > last_time:
                hp = _clamp_hp(hp - self._drain_elapsed_between(last_time, time_ms) * drain_rate_per_ms)
                last_time = time_ms
                min_hp = min(min_hp, hp)
            hp = _clamp_hp(hp + delta)
            min_hp = min(min_hp, hp)
        return min_hp

    def _calibrate_drain_rate(self) -> float:
        events = self._ideal_hp_events()
        if len(events) <= 1:
            return 0.0
        target_min = _clamp_hp(_difficulty_range(self._hp_difficulty, MIN_HEALTH_TARGET, MID_HEALTH_TARGET, MAX_HEALTH_TARGET))
        adjustment = 1
        result = 1.0
        for _ in range(64):
            lowest_hp = self._simulate_min_hp(events, result)
            if abs(lowest_hp - target_min) <= MINIMUM_HEALTH_ERROR:
                break
            adjustment *= 2
            result = max(0.0, result + math.copysign(1.0 / adjustment, lowest_hp - target_min))
        return result

    def _drain_hp_to(self, time_ms: float) -> None:
        if time_ms <= self._last_hp_time_ms:
            return
        self._hp = _clamp_hp(
            self._hp - self._drain_elapsed_between(self._last_hp_time_ms, time_ms) * self._drain_rate_per_ms
        )
        self._last_hp_time_ms = time_ms

    def _emit(
        self,
        time_ms: float,
        *,
        obj_idx: int | None = None,
        combo_add: int = 0,
        combo_reset: bool = False,
        result: str | None = None,
        passed_object: bool = False,
        mark_hit: bool = False,
        score_bonus: int = 0,
        hp_delta: float | None = None,
        combo_break: bool = False,
        eliminated: bool = False,
    ) -> None:
        self._drain_hp_to(time_ms)
        combo_before = self._combo
        result_value = {"300": 300, "100": 100, "50": 50}.get(result, 0)
        if result_value:
            combo_mul = max(combo_before - 1, 0)
            bonus = result_value * combo_mul * self._difficulty_multiplier * self._mod_multiplier / 25.0
            self._score += int(math.floor(result_value + bonus))
        if score_bonus:
            self._score += int(score_bonus)
        if combo_reset:
            self._combo = 0
        if combo_add:
            self._combo += combo_add
            self._max_combo = max(self._max_combo, self._combo)
        if mark_hit:
            self._last_hit_time_ms = time_ms

        if result == "300":
            self._n300 += 1
        elif result == "100":
            self._n100 += 1
        elif result == "50":
            self._n50 += 1
        elif result == "miss":
            self._misses += 1

        if hp_delta is None and result is not None:
            hp_delta = self._hp_object_result(result)
        if hp_delta:
            self._hp = _clamp_hp(self._hp + hp_delta)

        if passed_object:
            self._passed_objects += 1
            if self._rosu_perf is not None:
                self._rosu_state.max_combo = self._combo
                self._rosu_state.n300 = self._n300
                self._rosu_state.n100 = self._n100
                self._rosu_state.n50 = self._n50
                self._rosu_state.misses = self._misses
                attrs = self._rosu_perf.next(self._rosu_state)
                if attrs is not None:
                    self._pp = float(attrs.pp)
        total_hits = self._n300 + self._n100 + self._n50 + self._misses
        if total_hits > 0:
            accuracy = (
                300.0 * self._n300 + 100.0 * self._n100 + 50.0 * self._n50
            ) / (300.0 * total_hits)
        else:
            accuracy = 1.0
        self._eliminated = self._eliminated or (eliminated and self._eliminate_on_miss)

        self._points.append(
            ScorePoint(
                time_ms=time_ms,
                score=self._score,
                combo=self._combo,
                max_combo=self._max_combo,
                n300=self._n300,
                n100=self._n100,
                n50=self._n50,
                misses=self._misses,
                passed_objects=self._passed_objects,
                accuracy=accuracy,
                pp=self._pp,
                last_hit_time_ms=self._last_hit_time_ms,
                hp=self._hp,
                combo_break=combo_break,
                eliminated=self._eliminated,
            )
        )

    def _apply_replay_summary_correction(self) -> None:
        if self._replay is None or self._eliminate_on_miss or not self._summary_correction_allowed:
            return
        if not self._points:
            return

        target_counts = (
            int(self._replay.n300),
            int(self._replay.n100),
            int(self._replay.n50),
            int(self._replay.nmiss),
        )
        current_counts = (self._n300, self._n100, self._n50, self._misses)
        target_score = int(self._replay.score)
        target_max_combo = int(self._replay.max_combo)

        if (
            current_counts == target_counts
            and self._score == target_score
            and self._max_combo == target_max_combo
        ):
            return

        total_hits = sum(target_counts)
        if total_hits > 0:
            accuracy = (
                300.0 * target_counts[0]
                + 100.0 * target_counts[1]
                + 50.0 * target_counts[2]
            ) / (300.0 * total_hits)
        else:
            accuracy = 1.0

        last_point = self._points[-1]
        corrected_time = max(
            last_point.time_ms + 1e-3,
            self._wall_time(float(getattr(self._beatmap.hit_objects[-1], "end_time", self._beatmap.hit_objects[-1].start_time))) + 1e-3,
        )

        self._score = target_score
        self._n300, self._n100, self._n50, self._misses = target_counts
        self._max_combo = max(self._max_combo, target_max_combo)

        self._points.append(
            ScorePoint(
                time_ms=corrected_time,
                score=target_score,
                combo=last_point.combo,
                max_combo=target_max_combo,
                n300=target_counts[0],
                n100=target_counts[1],
                n50=target_counts[2],
                misses=target_counts[3],
                passed_objects=len(self._beatmap.hit_objects),
                accuracy=accuracy,
                pp=last_point.pp,
                last_hit_time_ms=last_point.last_hit_time_ms,
                hp=last_point.hp,
                combo_break=False,
                eliminated=False,
            )
        )

    def _process_circle(self, obj_idx: int, obj) -> None:
        obj_time = float(obj.start_time)
        w50 = hit_windows(self._od)[2]
        self._begin_object(obj_idx)

        if self._replay is None or self._judge is None:
            self._emit(
                self._wall_time(obj_time),
                obj_idx=obj_idx,
                combo_add=1,
                result="300",
                passed_object=True,
                mark_hit=True,
                hp_delta=self._hp_object_result("300") + self._combo_end_hp_bonus(obj_idx, True),
            )
            return

        hit = self._judge.result_for(obj_idx)
        if hit is not None and hit.result != "miss":
            self._update_combo_result_for_object_result(hit.result)
            self._emit(
                self._wall_time(hit.hit_time_ms),
                obj_idx=obj_idx,
                combo_add=1,
                result=hit.result,
                passed_object=True,
                mark_hit=True,
                hp_delta=self._hp_object_result(hit.result) + self._combo_end_hp_bonus(obj_idx, True),
            )
            return

        self._update_combo_result_for_object_result("miss")
        self._emit(
            self._wall_time(obj_time + w50),
            obj_idx=obj_idx,
            combo_reset=True,
            result="miss",
            passed_object=True,
            mark_hit=False,
            hp_delta=self._hp_miss(),
            combo_break=True,
            eliminated=self._eliminate_on_miss,
        )

    def _process_spinner(self, obj_idx: int, obj: Spinner) -> None:
        start_time = float(obj.start_time)
        end_time = float(obj.end_time)
        self._begin_object(obj_idx)

        analysis: SpinnerAnalysis
        result = "300"
        if self._replay is None or self._judge is None:
            analysis = build_auto_spinner_analysis(start_time, end_time, self._od)
        else:
            judged = self._judge.result_for(obj_idx)
            if judged is None or judged.spinner is None:
                analysis = SpinnerAnalysis(required_spins=0)
                result = "miss"
            else:
                analysis = judged.spinner
                result = judged.result

        self._update_combo_result_for_object_result(result)

        for event in analysis.bonus_events:
            self._emit(
                self._wall_time(event.time_ms),
                obj_idx=obj_idx,
                score_bonus=event.score,
                hp_delta=self._hp_spinner_spin(),
            )

        self._emit(
            self._wall_time(end_time),
            obj_idx=obj_idx,
            combo_add=0 if result == "miss" else 1,
            combo_reset=result == "miss",
            result=result,
            passed_object=True,
            mark_hit=result != "miss",
            hp_delta=(
                self._hp_miss() if result == "miss"
                else self._hp_object_result(result) + self._combo_end_hp_bonus(obj_idx, True)
            ),
            combo_break=result == "miss",
            eliminated=self._eliminate_on_miss and result == "miss",
        )

    def _process_slider(self, obj_idx: int, slider: Slider) -> None:
        start_time = float(slider.start_time)
        self._begin_object(obj_idx)
        head_judge_time = start_time
        _base_bl, sv, _, _ = self._beatmap._timing_at(start_time)
        px_per_beat = self._beatmap.slider_multiplier * 100.0 * sv
        if getattr(self._beatmap, "file_version", 14) < 8 and abs(sv) > 1e-6:
            px_per_beat /= sv
        if px_per_beat <= 0:
            px_per_beat = 1.0

        span_count = max(int(slider.repeat_count), 1)
        span_duration = float(slider.duration) / span_count
        total_duration = float(slider.duration)
        tail_judge_time = max(
            start_time + total_duration * 0.5,
            start_time + total_duration - TAIL_LENIENCY_MS,
        )
        head_window_50 = hit_windows(self._od)[2]

        path = self._slider_path(slider)
        nested_events = self._slider_nested_events(
            slider=slider,
            px_per_beat=px_per_beat,
            span_count=span_count,
            span_duration=span_duration,
            tail_judge_time=tail_judge_time,
        )

        total_parts = 2 + sum(1 for ev in nested_events if ev.kind in {"tick", "repeat"})
        parts_hit = 0

        head_hit = True
        head_hit_time = head_judge_time
        if self._replay is not None and self._judge is not None:
            judged = self._judge.result_for(obj_idx)
            head_hit = judged is not None and judged.result != "miss"
            if head_hit and judged is not None:
                head_hit_time = float(judged.hit_time_ms)

        if head_hit:
            parts_hit += 1
            self._emit(
                self._wall_time(head_hit_time),
                obj_idx=obj_idx,
                combo_add=1,
                mark_hit=True,
                score_bonus=30,
                hp_delta=self._hp_slider_part("head"),
            )
        else:
            self._update_combo_result_for_tick(hit=False)
            self._emit(
                self._wall_time(start_time + head_window_50),
                obj_idx=obj_idx,
                combo_reset=True,
                hp_delta=self._hp_tick_miss(),
                combo_break=True,
                eliminated=self._eliminate_on_miss,
            )

        for event in nested_events:
            hit = True if self._replay is None else self._slider_tracking_ok(
                original_time_ms=event.time_ms,
                follow_pos=self._path_pos(path, event.progress),
            )
            if event.kind == "tail":
                if hit:
                    parts_hit += 1
                    self._emit(
                        self._wall_time(event.time_ms),
                        obj_idx=obj_idx,
                        combo_add=1,
                        mark_hit=True,
                        score_bonus=30,
                        hp_delta=self._hp_slider_part("tail"),
                    )
                else:
                    self._update_combo_result_for_tick(slider_tail_miss=True, hit=False)
                    self._emit(
                        self._wall_time(event.time_ms),
                        obj_idx=obj_idx,
                        hp_delta=self._hp_tick_miss(),
                    )
                continue

            if hit:
                parts_hit += 1
                self._emit(
                    self._wall_time(event.time_ms),
                    obj_idx=obj_idx,
                    combo_add=1,
                    mark_hit=True,
                    score_bonus=30 if event.kind == "repeat" else 10,
                    hp_delta=self._hp_slider_part(event.kind),
                )
            else:
                self._update_combo_result_for_tick(hit=False)
                self._emit(
                    self._wall_time(event.time_ms),
                    obj_idx=obj_idx,
                    combo_reset=True,
                    hp_delta=self._hp_tick_miss(),
                    combo_break=True,
                    eliminated=self._eliminate_on_miss,
                )
                return

        if parts_hit >= total_parts:
            result = "300"
        elif parts_hit * 2 >= total_parts:
            result = "100"
        elif parts_hit > 0:
            result = "50"
        else:
            result = "miss"

        self._update_combo_result_for_object_result(result)
        self._emit(
            self._wall_time(tail_judge_time),
            obj_idx=obj_idx,
            result=result,
            passed_object=True,
            mark_hit=result != "miss",
            hp_delta=(
                self._hp_miss() if result == "miss"
                else self._hp_object_result(result) + self._combo_end_hp_bonus(obj_idx, True)
            ),
            combo_break=result == "miss",
            eliminated=self._eliminate_on_miss and result == "miss",
        )

    def _slider_path(self, slider: Slider):
        ctrl_pts = np.array([
            [float(slider.pos.x), float(slider.pos.y)],
            *[[float(point.x), float(point.y)] for point in slider.points],
        ], dtype=np.float32)
        path = sample_curve(
            slider.curve_type,
            np.nan_to_num(ctrl_pts),
            float(slider.pixel_length),
        )
        if len(path) < 2:
            return np.nan_to_num(ctrl_pts)
        return path

    def _slider_nested_events(
        self,
        *,
        slider: Slider,
        px_per_beat: float,
        span_count: int,
        span_duration: float,
        tail_judge_time: float,
    ) -> list[_SliderEvent]:
        events: list[_SliderEvent] = []
        tick_rate = float(getattr(self._beatmap, "slider_tick_rate", 1.0) or 1.0)
        if tick_rate < 1e-6:
            tick_rate = 1.0

        tick_distance = px_per_beat / tick_rate
        slider_length = float(slider.pixel_length)
        velocity = slider_length / max(span_duration, 1e-6)
        min_distance_from_end = velocity * 10.0

        for span_idx in range(span_count):
            span_start = float(slider.start_time) + span_idx * span_duration
            reversed_span = (span_idx % 2) == 1

            if tick_distance > 0.0:
                dist = tick_distance
                while dist < slider_length - min_distance_from_end:
                    progress = dist / max(slider_length, 1e-6)
                    time_progress = 1.0 - progress if reversed_span else progress
                    events.append(
                        _SliderEvent(
                            time_ms=span_start + time_progress * span_duration,
                            progress=progress,
                            kind="tick",
                        )
                    )
                    dist += tick_distance

            if span_idx < span_count - 1:
                events.append(
                    _SliderEvent(
                        time_ms=span_start + span_duration,
                        progress=0.0 if reversed_span else 1.0,
                        kind="repeat",
                    )
                )

        final_span_index = span_count - 1
        final_span_start = float(slider.start_time) + final_span_index * span_duration
        legacy_progress = (tail_judge_time - final_span_start) / max(span_duration, 1e-6)
        if span_count % 2 == 0:
            legacy_progress = 1.0 - legacy_progress
        legacy_progress = max(0.0, min(1.0, legacy_progress))
        events.append(
            _SliderEvent(
                time_ms=tail_judge_time,
                progress=legacy_progress,
                kind="tail",
            )
        )
        events.sort(key=lambda event: (event.time_ms, 0 if event.kind != "tail" else 1))
        return events

    def _slider_tracking_ok(self, *, original_time_ms: float, follow_pos) -> bool:
        if self._replay is None:
            return True
        if not (self._replay.keys_at(original_time_ms) & KEY_ANY):
            return False
        px, py = self._replay.position_at(original_time_ms)
        dx = float(px) - float(follow_pos[0])
        dy = float(py) - float(follow_pos[1])
        return dx * dx + dy * dy <= self._follow_radius * self._follow_radius

    @staticmethod
    def _path_pos(path, progress: float):
        if len(path) == 0:
            return (256.0, 192.0)
        if len(path) == 1:
            return path[0]
        progress = max(0.0, min(1.0, progress))
        float_idx = progress * (len(path) - 1)
        idx = min(int(float_idx), len(path) - 2)
        frac = float_idx - idx
        return path[idx] * (1.0 - frac) + path[idx + 1] * frac


def _clamp_hp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _difficulty_range(difficulty: float, minimum: float, midpoint: float, maximum: float) -> float:
    difficulty = max(0.0, min(10.0, float(difficulty)))
    if difficulty > 5.0:
        return midpoint + (maximum - midpoint) * ((difficulty - 5.0) / 5.0)
    if difficulty < 5.0:
        return midpoint + (midpoint - minimum) * ((difficulty - 5.0) / 5.0)
    return midpoint
