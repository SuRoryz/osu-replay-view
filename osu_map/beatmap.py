from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np
from osupyparser import OsuFile
from osupyparser.osu.objects import Slider, Spinner

from osu_map.curves import sample_curve

OSU_PLAYFIELD_WIDTH = 512
OSU_PLAYFIELD_HEIGHT = 384

_SAMPLE_SET_MAP = {0: None, 1: "normal", 2: "soft", 3: "drum"}


@dataclass
class HitsoundEvent:
    """A single scheduled hitsound trigger."""
    time_ms: float
    normal_set: str
    addition_set: str
    sound_enum: int
    volume: float


@dataclass
class SliderRenderData:
    """SDF-friendly slider data: all paths concatenated, per-slider metadata."""

    path_points: np.ndarray   # (P, 2) float32 -- all paths end-to-end
    n_sliders: int
    bbox_min: np.ndarray      # (S, 2) float32
    bbox_max: np.ndarray      # (S, 2) float32
    path_starts: np.ndarray   # (S,)   int32
    path_counts: np.ndarray   # (S,)   int32
    colors: np.ndarray        # (S, 3) float32
    start_times: np.ndarray   # (S,)   float32
    end_times: np.ndarray     # (S,)   float32
    z_values: np.ndarray      # (S,)   float32
    repeat_counts: np.ndarray # (S,)   int32
    object_indices: np.ndarray  # (S,) int32


@dataclass
class SpinnerRenderData:
    """Dedicated spinner instances so they are not rendered like circles."""

    positions: np.ndarray       # (S, 2) float32
    radii: np.ndarray           # (S,)   float32
    start_times: np.ndarray     # (S,)   float32
    end_times: np.ndarray       # (S,)   float32
    z_values: np.ndarray        # (S,)   float32
    object_indices: np.ndarray  # (S,)   int32


@dataclass
class RenderData:
    circle_positions: np.ndarray    # (N, 2) float32
    circle_start_times: np.ndarray  # (N,)   float32
    circle_end_times: np.ndarray    # (N,)   float32
    circle_object_indices: np.ndarray  # (N,) int32
    circle_radius: float
    circle_z: np.ndarray            # (N,)   float32
    slider: SliderRenderData | None
    spinner: SpinnerRenderData | None
    preempt: float
    fade_in: float
    hitsound_events: list[HitsoundEvent] = field(default_factory=list)


class Beatmap(OsuFile):
    """Parsed osu! beatmap with coordinate mapping utilities."""

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._sorted_tps: list = []
        self._tp_offsets: list[float] = []

    def parse_file(self):
        result = super().parse_file()
        self._prepare_timing()
        self._fix_slider_durations()
        return result

    # ----------------------------------------------------------- timing points

    def _prepare_timing(self) -> None:
        self._sorted_tps = sorted(self.timing_points, key=lambda tp: tp.offset)
        self._tp_offsets = [tp.offset for tp in self._sorted_tps]

    def _timing_at(self, offset_ms: float) -> tuple[float, float, int, int]:
        """Return (base_beat_length, sv_multiplier, sample_set_id, sample_volume)
        for the given offset using proper binary search."""
        idx = bisect.bisect_right(self._tp_offsets, offset_ms) - 1
        if idx < 0:
            idx = 0

        active = self._sorted_tps[idx]

        base_bl = 500.0  # 120 BPM fallback
        for i in range(idx, -1, -1):
            tp = self._sorted_tps[i]
            if tp.beat_length > 0:
                base_bl = tp.beat_length
                break

        if active.beat_length < 0:
            sv = max(0.1, min(10.0, abs(100.0 / active.beat_length)))
        else:
            sv = 1.0

        return base_bl, sv, active.sample_set_id, active.sample_volume

    def _fix_slider_durations(self) -> None:
        """Recompute slider duration / end_time using correct per-slider timing."""
        for obj in self.hit_objects:
            if not isinstance(obj, Slider):
                continue
            base_bl, sv, _, _ = self._timing_at(float(obj.start_time))
            px_per_beat = self.slider_multiplier * 100.0 * sv
            if px_per_beat <= 0:
                continue
            one_slide_beats = obj.pixel_length / px_per_beat
            one_slide_ms = one_slide_beats * base_bl
            total_ms = one_slide_ms * obj.repeat_count
            obj.duration = max(1, int(math.ceil(total_ms)))
            obj.end_time = obj.start_time + obj.duration

    # ----------------------------------------------------------- difficulty

    def circle_radius_osu(self, cs: float | None = None) -> float:
        """Circle radius in osu! pixels from CircleSize difficulty value."""
        if cs is None:
            cs = self.cs
        return 54.4 - 4.48 * cs

    def approach_preempt(self, ar: float | None = None) -> tuple[float, float]:
        """Convert AR to (preempt_ms, fade_in_ms) using the standard osu! formula."""
        if ar is None:
            ar = self.ar
        if ar < 5:
            preempt = 1200.0 + 600.0 * (5.0 - ar) / 5.0
        elif ar > 5:
            preempt = 1200.0 - 750.0 * (ar - 5.0) / 5.0
        else:
            preempt = 1200.0
        fade_in = preempt * 2.0 / 3.0
        return preempt, fade_in

    # ----------------------------------------------------------- render data

    def build_render_data(self, *, ar: float | None = None,
                          cs: float | None = None,
                          hr_flip: bool = False) -> RenderData:
        """Build GPU-ready arrays for circles and sliders with Z-ordering.

        Parameters let the caller override AR/CS (for mods) and flip Y for HR.
        """
        radius = self.circle_radius_osu(cs)
        preempt, fade_in = self.approach_preempt(ar)

        circles_raw: list[tuple[float, float, float, float, int]] = []
        sliders_raw: list[tuple[Slider, int]] = []
        spinners_raw: list[tuple[float, float, float, float, int]] = []

        for idx, obj in enumerate(self.hit_objects):
            if isinstance(obj, Slider):
                sliders_raw.append((obj, idx))
                end_t = float(obj.end_time)
            elif isinstance(obj, Spinner):
                end_t = float(obj.end_time)
            else:
                end_t = float(obj.start_time)

            oy = float(obj.pos.y)
            if hr_flip:
                world_y = oy
            else:
                world_y = OSU_PLAYFIELD_HEIGHT - oy

            if isinstance(obj, Spinner):
                spinner_radius = min(OSU_PLAYFIELD_WIDTH, OSU_PLAYFIELD_HEIGHT) * 0.47
                spinners_raw.append((
                    OSU_PLAYFIELD_WIDTH * 0.5,
                    OSU_PLAYFIELD_HEIGHT * 0.5,
                    float(obj.start_time),
                    end_t,
                    idx,
                ))
            else:
                circles_raw.append((
                    float(obj.pos.x),
                    world_y,
                    float(obj.start_time),
                    end_t,
                    idx,
                ))

        total_objects = len(self.hit_objects)

        n_circles = len(circles_raw)
        circle_positions = np.empty((n_circles, 2), dtype=np.float32)
        circle_start_times = np.empty(n_circles, dtype=np.float32)
        circle_end_times = np.empty(n_circles, dtype=np.float32)
        circle_object_indices = np.empty(n_circles, dtype=np.int32)
        circle_z = np.empty(n_circles, dtype=np.float32)

        for i, (x, y, t, et, obj_idx) in enumerate(circles_raw):
            circle_positions[i] = [x, y]
            circle_start_times[i] = t
            circle_end_times[i] = et
            circle_object_indices[i] = obj_idx
            circle_z[i] = self._z_from_index(obj_idx, total_objects)

        slider_data = self._build_slider_data(
            sliders_raw, radius, total_objects, hr_flip=hr_flip,
        ) if sliders_raw else None
        spinner_data = self._build_spinner_data(spinners_raw, total_objects) if spinners_raw else None

        hs_events = self._build_hitsound_events()

        return RenderData(
            circle_positions=circle_positions,
            circle_start_times=circle_start_times,
            circle_end_times=circle_end_times,
            circle_object_indices=circle_object_indices,
            circle_radius=radius,
            circle_z=circle_z,
            slider=slider_data,
            spinner=spinner_data,
            preempt=preempt,
            fade_in=fade_in,
            hitsound_events=hs_events,
        )

    def _build_slider_data(
        self,
        sliders_raw: list[tuple[Slider, int]],
        radius: float,
        total_objects: int,
        *,
        hr_flip: bool = False,
    ) -> SliderRenderData:
        all_points: list[np.ndarray] = []
        bbox_mins: list[list[float]] = []
        bbox_maxs: list[list[float]] = []
        path_starts: list[int] = []
        path_counts: list[int] = []
        start_times: list[float] = []
        end_times: list[float] = []
        z_values: list[float] = []
        colors: list[list[float]] = []
        repeat_counts: list[int] = []
        object_indices: list[int] = []
        offset = 0

        def _flip(osu_y: float) -> float:
            return osu_y if hr_flip else OSU_PLAYFIELD_HEIGHT - osu_y

        for slider, obj_idx in sliders_raw:
            ctrl_pts = np.array(
                [[float(slider.pos.x), _flip(float(slider.pos.y))]]
                + [
                    [float(p.x), _flip(float(p.y))]
                    for p in slider.points
                ],
                dtype=np.float32,
            )

            path = sample_curve(
                slider.curve_type, ctrl_pts, float(slider.pixel_length)
            )
            if len(path) < 2:
                continue

            n = len(path)
            all_points.append(path)
            path_starts.append(offset)
            path_counts.append(n)
            offset += n

            p_min = path.min(axis=0)
            p_max = path.max(axis=0)
            bbox_mins.append([p_min[0] - radius, p_min[1] - radius])
            bbox_maxs.append([p_max[0] + radius, p_max[1] + radius])

            start_times.append(float(slider.start_time))
            end_times.append(float(slider.end_time))
            z_values.append(self._z_from_index(obj_idx, total_objects) - 0.005)
            colors.append([1.0, 0.67, 0.2])
            repeat_counts.append(int(slider.repeat_count))
            object_indices.append(int(obj_idx))

        if not all_points:
            return SliderRenderData(
                path_points=np.empty((0, 2), dtype=np.float32),
                n_sliders=0,
                bbox_min=np.empty((0, 2), dtype=np.float32),
                bbox_max=np.empty((0, 2), dtype=np.float32),
                path_starts=np.empty(0, dtype=np.int32),
                path_counts=np.empty(0, dtype=np.int32),
                colors=np.empty((0, 3), dtype=np.float32),
                start_times=np.empty(0, dtype=np.float32),
                end_times=np.empty(0, dtype=np.float32),
                z_values=np.empty(0, dtype=np.float32),
                repeat_counts=np.empty(0, dtype=np.int32),
                object_indices=np.empty(0, dtype=np.int32),
            )

        return SliderRenderData(
            path_points=np.concatenate(all_points),
            n_sliders=len(path_starts),
            bbox_min=np.array(bbox_mins, dtype=np.float32),
            bbox_max=np.array(bbox_maxs, dtype=np.float32),
            path_starts=np.array(path_starts, dtype=np.int32),
            path_counts=np.array(path_counts, dtype=np.int32),
            colors=np.array(colors, dtype=np.float32),
            start_times=np.array(start_times, dtype=np.float32),
            end_times=np.array(end_times, dtype=np.float32),
            z_values=np.array(z_values, dtype=np.float32),
            repeat_counts=np.array(repeat_counts, dtype=np.int32),
            object_indices=np.array(object_indices, dtype=np.int32),
        )

    def _build_spinner_data(
        self,
        spinners_raw: list[tuple[float, float, float, float, int]],
        total_objects: int,
    ) -> SpinnerRenderData:
        n = len(spinners_raw)
        positions = np.empty((n, 2), dtype=np.float32)
        radii = np.empty(n, dtype=np.float32)
        start_times = np.empty(n, dtype=np.float32)
        end_times = np.empty(n, dtype=np.float32)
        z_values = np.empty(n, dtype=np.float32)
        object_indices = np.empty(n, dtype=np.int32)
        spinner_radius = np.float32(min(OSU_PLAYFIELD_WIDTH, OSU_PLAYFIELD_HEIGHT) * 0.47)

        for i, (x, y, start_t, end_t, obj_idx) in enumerate(spinners_raw):
            positions[i] = [x, y]
            radii[i] = spinner_radius
            start_times[i] = start_t
            end_times[i] = end_t
            z_values[i] = self._z_from_index(obj_idx, total_objects) - 0.01
            object_indices[i] = obj_idx

        return SpinnerRenderData(
            positions=positions,
            radii=radii,
            start_times=start_times,
            end_times=end_times,
            z_values=z_values,
            object_indices=object_indices,
        )

    # ----------------------------------------------------------- hitsounds

    def _resolve_sample_set(self, additions, tp_set_id: int) -> tuple[str, str]:
        """Return (normal_set, addition_set) as lowercase strings."""
        default = (self.sample_set or "Soft").lower()
        tp_set = _SAMPLE_SET_MAP.get(tp_set_id) or default

        normal = default
        if additions and additions.normal:
            normal = additions.normal.lower()
        elif tp_set:
            normal = tp_set

        addition = normal
        if additions and additions.additional:
            addition = additions.additional.lower()

        return normal, addition

    @staticmethod
    def _resolve_volume(additions, tp_volume: int) -> float:
        if additions and additions.volume and additions.volume > 0:
            return additions.volume / 100.0
        if tp_volume > 0:
            return tp_volume / 100.0
        return 0.8

    def _build_hitsound_events(self) -> list[HitsoundEvent]:
        events: list[HitsoundEvent] = []

        for obj in self.hit_objects:
            _, _, tp_set_id, tp_volume = self._timing_at(float(obj.start_time))
            additions = getattr(obj, "additions", None)
            normal_set, addition_set = self._resolve_sample_set(additions, tp_set_id)
            volume = self._resolve_volume(additions, tp_volume)

            if isinstance(obj, Slider):
                rc = max(obj.repeat_count, 1)
                for i, edge in enumerate(obj.edges):
                    # duration is TOTAL (all repeats), so per-slide = duration / repeat_count
                    edge_time = float(obj.start_time) + obj.duration * i / rc

                    # Timing point may differ at each edge's time
                    _, _, e_tp_set, e_tp_vol = self._timing_at(edge_time)

                    try:
                        e_sound = int(edge.sound_types) if edge.sound_types else obj.sound_enum
                    except (ValueError, TypeError):
                        e_sound = obj.sound_enum

                    e_adds = edge.additions
                    e_normal, e_addition = self._resolve_sample_set(e_adds, e_tp_set)
                    e_vol = self._resolve_volume(e_adds, e_tp_vol)

                    events.append(HitsoundEvent(
                        edge_time, e_normal, e_addition, e_sound, e_vol,
                    ))
            else:
                events.append(HitsoundEvent(
                    float(obj.start_time), normal_set, addition_set,
                    obj.sound_enum, volume,
                ))

        events.sort(key=lambda e: e.time_ms)
        return events

    @staticmethod
    def _z_from_index(obj_idx: int, total: int) -> float:
        """Earlier objects get higher Z (closer to camera in OpenGL ortho).

        OpenGL ortho negates Z, so higher world-Z = lower depth = closer.
        Maps earliest object to Z=0.9 (closest), latest to Z=0.1 (farthest).
        """
        if total <= 1:
            return 0.5
        return 0.9 - 0.8 * (obj_idx / (total - 1))
