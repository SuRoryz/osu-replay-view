"""Gameplay scene -- renders the beatmap with auto or replay cursor."""

from __future__ import annotations

import bisect
import functools
import hashlib
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import moderngl
import numpy as np
from pyrr import matrix44

from audio import AudioEngine
from audio.hitsounds import HitsoundManager
from cursor import DanserPlaystyle, CursorPlaystyle
from osu_map import Beatmap
from osu_map.beatmap import HitsoundEvent, RenderData
from osu_map.mods import HR, HD, apply_difficulty, normalize_mods, speed_multiplier as _speed_mul
from osu_map.scanner import BeatmapInfo
from profiling import profiler
from replay.data import ReplayData
from replay.judge import HitJudge
from replay.score import StablePerformanceTimeline
from runtime_paths import HITSOUNDS_DIR
from scenes.base import Scene
from skins import SKIN_REGISTRY, Skin
from speedups import clip_trail_points, compute_slider_ball_instances
from ui.menu.layout import Rect

OSU_PLAYFIELD_WIDTH = 512
OSU_PLAYFIELD_HEIGHT = 384
PLAYFIELD_CX = OSU_PLAYFIELD_WIDTH / 2
PLAYFIELD_CY = OSU_PLAYFIELD_HEIGHT / 2

PADDING_H = 0.15
PADDING_V = 0.10

PATH_TEX_WIDTH = 1024
APPROACH_SCALE = 4.0
MAX_SLIDER_BALLS = 32
CURSOR_RADIUS = 4.5
TRAIL_MAX = 256
TRAIL_LIFETIME_MS = 500.0
TRAIL_SUBSAMPLE_TARGET = 192
TRAIL_RIBBON_MAX_VERTS = (TRAIL_SUBSAMPLE_TARGET + 2) * 2
TRAIL_CURVE_MAX = 2048
TRAIL_RESAMPLE_STEP_PX = 1.2
TRAIL_MIN_SAMPLE_DT_MS = 3.0
TRAIL_MIN_SAMPLE_DIST_SQ = 1.0
TRAIL_MIN_CONTROL_DT_MS = 0.05
TRAIL_MIN_CONTROL_DIST_SQ = 0.04
TRAIL_STROKE_RADIUS = CURSOR_RADIUS * 0.92
TRAIL_TAIL_TAPER_FRACTION = 0.12
TRAIL_MITER_LIMIT = 1.5
_QUAD_VERTS = np.array([
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0],
], dtype="f4")

_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype="i4")


FADE_IN_DURATION = 1.2
CURSOR_INTRO_DURATION = 0.8
GAMEPLAY_EXIT_DURATION = 0.4
PAUSE_ITEMS = ["Continue", "Settings", "Back to Menu", "Exit"]
TIMELINE_ENTER_DURATION = 0.45
TIMELINE_CHIP_HEIGHT = 22
TIMELINE_TRACK_HEIGHT = 8
TIMELINE_SCRUB_PAD = 14
VOLUME_BUTTON_SIZE = 26.0
VOLUME_BUTTON_GAP = 8.0
VOLUME_BUTTON_ROW_GAP = 8.0
VOLUME_SLIDER_PANEL_W = 34.0
VOLUME_SLIDER_PANEL_H = 96.0
VOLUME_SLIDER_GAP = 10.0
VOLUME_SLIDER_HIDE_DELAY = 1.5
VOLUME_SLIDER_TRACK_W = 6.0
VOLUME_SLIDER_TRACK_H = 64.0
VOLUME_SLIDER_ANIM_SPEED = 12.0
TIMELINE_KEY_SEEK_STEP_MS = 10_000.0


def _ease_out_back(t: float) -> float:
    c = 1.70158
    return 1.0 + (c + 1.0) * ((t - 1.0) ** 3) + c * ((t - 1.0) ** 2)


def _ease_out_cubic(t: float) -> float:
    return 1.0 - ((1.0 - t) ** 3)


def _smoothstep01(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _centripetal_catmull_rom(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float,
) -> np.ndarray:
    """Evaluate a centripetal Catmull-Rom segment from p1 to p2."""

    def tj(ti: float, a: np.ndarray, b: np.ndarray) -> float:
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        dist = max((dx * dx + dy * dy) ** 0.5, 1e-4)
        return ti + dist ** 0.5

    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)
    u = t1 + (t2 - t1) * max(0.0, min(1.0, float(t)))

    a1 = ((t1 - u) / max(t1 - t0, 1e-5)) * p0 + ((u - t0) / max(t1 - t0, 1e-5)) * p1
    a2 = ((t2 - u) / max(t2 - t1, 1e-5)) * p1 + ((u - t1) / max(t2 - t1, 1e-5)) * p2
    a3 = ((t3 - u) / max(t3 - t2, 1e-5)) * p2 + ((u - t2) / max(t3 - t2, 1e-5)) * p3
    b1 = ((t2 - u) / max(t2 - t0, 1e-5)) * a1 + ((u - t0) / max(t2 - t0, 1e-5)) * a2
    b2 = ((t3 - u) / max(t3 - t1, 1e-5)) * a2 + ((u - t1) / max(t3 - t1, 1e-5)) * a3
    return ((t2 - u) / max(t2 - t1, 1e-5)) * b1 + ((u - t1) / max(t2 - t1, 1e-5)) * b2


@dataclass(slots=True)
class _GameplayParticipant:
    leaderboard_id: str
    name: str
    source_type: str
    effective_mods: int
    replay: ReplayData | None
    visual_replay: ReplayData | None
    cpath: CursorPlaystyle | None
    judge: HitJudge | None
    score_timeline: StablePerformanceTimeline
    missed_circle_indices: set[int]
    hs_events: list[HitsoundEvent]
    auto_key_schedule: list[tuple[int, float, float]]
    color: tuple[float, float, float]
    trail: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=TRAIL_MAX))
    score_point_idx: int = -1
    eliminated: bool = False
    elimination_started_ms: float = -1.0
    elimination_score: int = 0
    current_score: int = 0
    current_accuracy: float = 1.0
    current_combo: int = 0
    current_pp: float = 0.0
    current_hp: float = 1.0
    last_point: object | None = None
    leaderboard_y: float = 0.0
    leaderboard_focus_mix: float = 0.0


@dataclass(slots=True)
class _ActiveRenderWindow:
    full_data: np.ndarray
    appear_times: np.ndarray
    appear_indices: np.ndarray
    disappear_times: np.ndarray
    disappear_indices: np.ndarray
    instance_buf: moderngl.Buffer
    upload_buf: np.ndarray
    active_mask: np.ndarray | None = None
    active_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    start_cursor: int = 0
    end_cursor: int = 0
    active_count: int = 0
    dirty: bool = True
    last_time_ms: float = float("-inf")


def _stable_cursor_color(seed_text: str) -> tuple[float, float, float]:
    digest = hashlib.md5(seed_text.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    sat = 0.55 + digest[1] / 255.0 * 0.25
    val = 0.88 + digest[2] / 255.0 * 0.10
    h6 = hue * 6.0
    i = int(h6) % 6
    f = h6 - int(h6)
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))
    rgb_map = {
        0: (val, t, p),
        1: (q, val, p),
        2: (p, val, t),
        3: (p, q, val),
        4: (t, p, val),
        5: (val, p, q),
    }
    return rgb_map[i]


class GameplayScene(Scene):
    """Full gameplay rendering -- circles, sliders, cursor, audio, hitsounds."""

    def __init__(self, app, beatmap_info: BeatmapInfo,
                 replay_paths: list[str] | None = None,
                 include_danser: bool = False,
                 multi_replay: bool = False,
                 mods: int = 0,
                 mods_overridden: bool = False):
        super().__init__(app)
        self._binfo = beatmap_info
        self._replay_paths = list(replay_paths or [])
        self._include_danser = bool(include_danser)
        self._multi_replay = bool(multi_replay)
        self._mods = normalize_mods(mods)
        self._mods_overridden = bool(mods_overridden)

        self._audio: AudioEngine | None = None
        self._hitsounds: HitsoundManager | None = None
        self._cpath: CursorPlaystyle | None = None
        self._replay: ReplayData | None = None
        self._visual_replay: ReplayData | None = None
        self._judge: HitJudge | None = None
        self._participants: list[_GameplayParticipant] = []
        self._participants_by_id: dict[str, _GameplayParticipant] = {}
        self._danser_participant: _GameplayParticipant | None = None
        self._hud_participant: _GameplayParticipant | None = None
        self._hud_score_participant: _GameplayParticipant | None = None
        self._audio_participant: _GameplayParticipant | None = None
        self._active_gameplay_source_id: str | None = None
        self._leaderboard_rects: list[tuple[float, float, float, float, str]] = []
        self._leaderboard_hover_id: str | None = None
        self._leaderboard_pinned_id: str | None = None
        self._leaderboard_focus_presence: float = 0.0

        self._skins = list(SKIN_REGISTRY)
        self._skin_idx = 0
        self._skin: Skin = self._skins[0]

        self.circle_prog = None
        self.slider_prog = None
        self.approach_prog = None
        self.spinner_prog = None
        self.ball_prog = None
        self.cursor_prog = None

        self.n_circle_instances = 0
        self.circle_vao: moderngl.VertexArray | None = None
        self.approach_vao: moderngl.VertexArray | None = None
        self.spinner_vao: moderngl.VertexArray | None = None
        self.slider_vao: moderngl.VertexArray | None = None
        self.n_spinner_instances = 0
        self.n_slider_instances = 0
        self.path_tex: moderngl.Texture | None = None
        self.path_tex_width = PATH_TEX_WIDTH
        self._circle_quad_vbo: moderngl.Buffer | None = None
        self._circle_index_buf: moderngl.Buffer | None = None
        self._circle_instance_gpu_buf: moderngl.Buffer | None = None
        self._circle_render_window: _ActiveRenderWindow | None = None
        self._spinner_quad_vbo: moderngl.Buffer | None = None
        self._spinner_index_buf: moderngl.Buffer | None = None
        self._spinner_instance_gpu_buf: moderngl.Buffer | None = None
        self._spinner_render_window: _ActiveRenderWindow | None = None
        self._slider_quad_vbo: moderngl.Buffer | None = None
        self._slider_index_buf: moderngl.Buffer | None = None
        self._slider_instance_gpu_buf: moderngl.Buffer | None = None
        self._slider_render_window: _ActiveRenderWindow | None = None

        self.ball_vao: moderngl.VertexArray | None = None
        self._ball_instance_buf: moderngl.Buffer | None = None
        self._ball_color: list[float] = [1.0, 1.0, 1.0]

        self.cursor_vao: moderngl.VertexArray | None = None
        self._cursor_instance_buf: moderngl.Buffer | None = None
        self.trail_prog = None
        self.trail_vao: moderngl.VertexArray | None = None
        self._trail_point_tex: moderngl.Texture | None = None
        self._trail: deque[tuple[float, float, float]] = deque(maxlen=TRAIL_MAX)

        self._stored_data: RenderData | None = None
        self._hs_events: list[HitsoundEvent] = []
        self._hs_times: list[float] = []
        self._hs_cursor: int = 0
        self._prev_time_ms: float = -1e9
        self.projection = np.eye(4, dtype="f4")

        self._missed_circle_indices: set[int] = set()
        self._obj_to_circle_idx: dict[int, int] = {}

        self._paused = False
        self._pause_menu_open = False
        self._pause_selection = 0
        self._fade_timer = 0.0
        self._fade_in_done = False
        self._cursor_intro_timer = 0.0
        self._first_render = True
        self._exiting = False
        self._exit_timer = 0.0
        self._mouse_x = 0
        self._mouse_y = 0
        self._current_time_ms = 0.0
        self._manual_time_ms = 0.0
        self._playback_speed = 1.0
        self._timeline_start_ms = 0.0
        self._timeline_end_ms = 1.0
        self._timeline_total_ms = 1.0
        self._timeline_anim_timer = 0.0
        self._timeline_hovered = False
        self._timeline_hover_time_ms = 0.0
        self._timeline_dragging = False
        self._timeline_drag_resume_playback = False
        self._volume_slider_dragging: str | None = None
        self._volume_slider_visibility = {"sfx": 0.0, "music": 0.0}
        self._volume_slider_hide_timers = {"sfx": 0.0, "music": 0.0}

        # HUD state
        self._combo: int = 0
        self._combo_display: float = 0.0
        self._max_combo: int = 0
        self._hp: float = 1.0
        self._hp_display: float = 1.0
        self._last_hit_time_ms: float = -1e9
        self._combo_anim_timer: float = 0.0
        self._score_timeline: StablePerformanceTimeline | None = None
        self._score_point_idx: int = -1
        self._pp_target: float = 0.0
        self._pp_display: float = 0.0
        self._pp_pulse_timer: float = 0.0
        self._accuracy: float = 1.0
        self._accuracy_display: float = 1.0
        self._leaderboard_display_order: list[str] = []
        self._leaderboard_sorted_cache: list[_GameplayParticipant] = []
        self._leaderboard_last_sort_ms: float = -1e9

        # Key overlay state
        self._k1_count: int = 0
        self._k2_count: int = 0
        self._k1_pressed: bool = False
        self._k2_pressed: bool = False
        self._k1_press_start: float = -1.0
        self._k2_press_start: float = -1.0
        self._k1_release_time: float = -1.0
        self._k2_release_time: float = -1.0
        self._key_bars: list[tuple[int, float, float]] = []  # (key, start_ms, dur_ms)
        self._auto_key_alt: bool = False  # alternates K1/K2 in auto
        self._auto_key_schedule: list[tuple[int, float, float]] = []
        self._auto_key_start_times: list[float] = []
        self._auto_key_cursor: int = 0
        self._replay_key_cursor: int = 0
        self._key_overlay_source_sig: tuple[str, int] | None = None
        self._key_overlay_time_ms: float = 0.0
        self._timeline_layout_cache: dict | None = None
        self._timeline_layout_cache_key: tuple[int, int, int] | None = None
        self._slider_ball_buf = np.empty((MAX_SLIDER_BALLS, 7), dtype="f4")
        self._cursor_head_buf = np.empty((1, 8), dtype="f4")
        self._trail_raw_buf = np.empty((TRAIL_MAX, 3), dtype="f4")
        self._trail_clip_buf = np.empty((TRAIL_MAX + 2, 3), dtype="f4")
        self._trail_gpu_buf = np.empty((TRAIL_MAX + 2, 4), dtype="f4")
        self._prepare_lock = threading.Lock()
        self._prepare_pending: dict | None = None
        self._prepare_error: str | None = None
        self._loading_ready = False

    @property
    def ctx(self) -> moderngl.Context:
        return self.app.ctx

    @property
    def wnd(self):
        return self.app.wnd

    def on_enter(self) -> None:
        self.app.set_settings_button(None, visible=False)
        self._skin.setup(self.ctx)
        self._compile_programs()
        self.projection = self._build_projection()
        self._sync_uniforms()
        self._start_prepare_worker()

    def _start_prepare_worker(self) -> None:
        def _worker() -> None:
            try:
                with profiler.timer("gameplay.prepare_cpu"):
                    beatmap = Beatmap(self._binfo.path)
                    beatmap.parse_file()
                    mods = normalize_mods(self._mods)
                    if not self._mods_overridden:
                        for replay_path in self._replay_paths:
                            if not Path(replay_path).is_file():
                                continue
                            try:
                                mods = normalize_mods(ReplayData.peek_mods(replay_path))
                                break
                            except Exception:
                                continue

                    mod_ar, mod_cs, mod_od, _mod_hp = apply_difficulty(
                        beatmap.ar, beatmap.cs, beatmap.od, beatmap.hp, mods,
                    )
                    speed = _speed_mul(mods)
                    hr_flip = bool(mods & HR)

                    render_data = beatmap.build_render_data(
                        ar=mod_ar, cs=mod_cs, hr_flip=hr_flip,
                    )
                    radius = beatmap.circle_radius_osu(mod_cs)

                    if speed != 1.0:
                        inv = 1.0 / speed
                        render_data.circle_start_times *= inv
                        render_data.circle_end_times *= inv
                        render_data.preempt *= inv
                        render_data.fade_in *= inv
                        if render_data.spinner is not None:
                            render_data.spinner.start_times *= inv
                            render_data.spinner.end_times *= inv
                        if render_data.slider is not None:
                            render_data.slider.start_times *= inv
                            render_data.slider.end_times *= inv
                        for ev in render_data.hitsound_events:
                            ev.time_ms *= inv

                    participants: list[_GameplayParticipant] = []
                    danser_participant: _GameplayParticipant | None = None
                    perfect_participant: _GameplayParticipant | None = None

                    for replay_path in self._replay_paths:
                        if not Path(replay_path).is_file():
                            continue
                        replay = ReplayData.from_file(replay_path)
                        judge = HitJudge(
                            beatmap.hit_objects, mod_od, radius, replay,
                        )
                        hs_events = self._build_replay_hs_events(beatmap, render_data, judge)
                        if speed != 1.0:
                            inv = 1.0 / speed
                            for ev in hs_events:
                                ev.time_ms *= inv
                        participants.append(
                            _GameplayParticipant(
                                leaderboard_id=f"replay:{replay_path}",
                                name=replay.player_name or Path(replay_path).stem,
                                source_type="replay",
                                effective_mods=mods,
                                replay=replay,
                                visual_replay=replay,
                                cpath=None,
                                judge=judge,
                                score_timeline=StablePerformanceTimeline.build(
                                    beatmap=beatmap,
                                    beatmap_path=self._binfo.path,
                                    mods=mods,
                                    clock_rate=speed,
                                    circle_radius=radius,
                                    od=mod_od,
                                    hp=_mod_hp,
                                    replay=replay,
                                    judge=judge,
                                    eliminate_on_miss=self._multi_replay,
                                ),
                                missed_circle_indices=judge.missed_indices,
                                hs_events=hs_events,
                                auto_key_schedule=[],
                                color=_stable_cursor_color(f"replay:{replay_path}"),
                            )
                        )

                    if self._multi_replay or self._include_danser or not participants:
                        perfect_participant = _GameplayParticipant(
                            leaderboard_id="perfect",
                            name="Perfect",
                            source_type="perfect",
                            effective_mods=mods,
                            replay=None,
                            visual_replay=None,
                            cpath=DanserPlaystyle(beatmap, render_data),
                            judge=None,
                            score_timeline=StablePerformanceTimeline.build(
                                beatmap=beatmap,
                                beatmap_path=self._binfo.path,
                                mods=mods,
                                clock_rate=speed,
                                circle_radius=radius,
                                od=mod_od,
                                hp=_mod_hp,
                                replay=None,
                                judge=None,
                                eliminate_on_miss=self._multi_replay,
                            ),
                            missed_circle_indices=set(),
                            hs_events=list(render_data.hitsound_events),
                            auto_key_schedule=self._build_auto_key_schedule(beatmap, speed),
                            color=(1.0, 0.84, 0.28),
                        )
                    if perfect_participant is not None and (self._include_danser or not participants):
                        danser_participant = _GameplayParticipant(
                            leaderboard_id="danser",
                            name="Danser",
                            source_type="danser",
                            effective_mods=perfect_participant.effective_mods,
                            replay=perfect_participant.replay,
                            visual_replay=perfect_participant.visual_replay,
                            cpath=perfect_participant.cpath,
                            judge=perfect_participant.judge,
                            score_timeline=perfect_participant.score_timeline,
                            missed_circle_indices=perfect_participant.missed_circle_indices,
                            hs_events=perfect_participant.hs_events,
                            auto_key_schedule=perfect_participant.auto_key_schedule,
                            color=perfect_participant.color,
                        )
                        participants.append(danser_participant)

                    hud_participant = (
                        perfect_participant
                        if self._multi_replay and perfect_participant is not None
                        else (participants[0] if participants else perfect_participant)
                    )
                    hud_score_participant = (
                        perfect_participant
                        if self._multi_replay and perfect_participant is not None
                        else hud_participant
                    )
                    audio_participant = hud_participant or (participants[0] if participants else None)

                result = {
                    "beatmap": beatmap,
                    "mods": mods,
                    "speed": speed,
                    "hr_flip": hr_flip,
                    "render_data": render_data,
                    "participants": participants,
                    "danser_participant": danser_participant,
                    "hud_participant": hud_participant,
                    "hud_score_participant": hud_score_participant,
                    "audio_participant": audio_participant,
                    "audio_path": self._binfo.audio_path,
                    "audio_lead_in": max(3000.0, float(beatmap.audio_lead_in)),
                }
                with self._prepare_lock:
                    self._prepare_pending = result
            except Exception as exc:
                with self._prepare_lock:
                    self._prepare_error = str(exc)

        threading.Thread(target=_worker, daemon=True, name="gameplay-prepare").start()

    def _apply_prepare_result(self) -> None:
        if self._loading_ready:
            return
        with self._prepare_lock:
            result = self._prepare_pending
            self._prepare_pending = None
            error = self._prepare_error
        if error:
            return
        if result is None:
            return

        self._participants = result["participants"]
        self._participants_by_id = {
            participant.leaderboard_id: participant for participant in self._participants
        }
        self._leaderboard_sorted_cache = []
        self._leaderboard_last_sort_ms = -1e9
        self._danser_participant = result["danser_participant"]
        self._hud_participant = result["hud_participant"]
        self._hud_score_participant = result["hud_score_participant"]
        self._audio_participant = result["audio_participant"]
        self._mods = result["mods"]
        self._speed = result["speed"]
        self._hr_flip = result["hr_flip"]
        self._apply_gameplay_source_participant(self._gameplay_source_participant())
        hud_score_participant = self._hud_score_participant
        self._score_timeline = (
            hud_score_participant.score_timeline if hud_score_participant is not None else None
        )
        self._hitsounds = HitsoundManager(
            HITSOUNDS_DIR,
            volume=self.app.settings.sfx_volume,
            muted=self.app.settings.sfx_muted,
        )

        audio_path = result["audio_path"]
        if audio_path and Path(audio_path).is_file():
            self._audio = AudioEngine(
                audio_path,
                lead_in_ms=result["audio_lead_in"],
                speed=self._speed,
                volume=self.app.settings.music_volume,
                muted=self.app.settings.music_muted,
            )
            threading.Thread(
                target=self._audio.prime_playback_speeds,
                args=([0.5, 1.5, 2.0],),
                daemon=True,
                name="gameplay-audio-prewarm",
            ).start()

        hd = 1 if self._mods & HD else 0
        for prog in (self.circle_prog, self.slider_prog, self.spinner_prog, self.approach_prog):
            if prog is not None and "u_hidden" in prog:
                prog["u_hidden"].value = hd

        self._score_point_idx = -1
        self._pp_target = 0.0
        self._pp_display = 0.0
        self._pp_pulse_timer = 0.0

        render_data = result["render_data"]
        self._upload(render_data)
        self._timeline_start_ms, self._timeline_end_ms = self._compute_timeline_bounds(render_data)
        self._timeline_total_ms = max(1.0, self._timeline_end_ms - self._timeline_start_ms)
        self._invalidate_timeline_layout()
        self._manual_time_ms = self._timeline_start_ms
        self._current_time_ms = self._timeline_start_ms
        self._rebuild_state_for_time(self._timeline_start_ms)
        self._combo_display = float(self._combo)
        self._hp_display = self._hp
        self._accuracy_display = self._accuracy

        if self._audio is not None:
            self._audio.start()
        self._loading_ready = True

    def on_leave(self) -> None:
        if self._audio is not None:
            self._audio.cleanup()
            self._audio = None
        self._skin.cleanup()

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

        spinner_src = self._skin.spinner_shader_source()
        if spinner_src is not None:
            vert, frag = spinner_src
            self.spinner_prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        else:
            self.spinner_prog = None

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

        trail_src = self._skin.trail_shader_source()
        if trail_src is not None:
            if len(trail_src) == 3:
                vert, geom, frag = trail_src
                self.trail_prog = self.ctx.program(
                    vertex_shader=vert,
                    geometry_shader=geom,
                    fragment_shader=frag,
                )
            else:
                vert, frag = trail_src
                self.trail_prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        else:
            self.trail_prog = None

    def _sync_uniforms(self):
        proj_bytes = self.projection.astype("f4").tobytes()
        for prog in (self.circle_prog, self.slider_prog, self.spinner_prog):
            if prog is None:
                continue
            prog["projection"].write(proj_bytes)
            prog["preempt"].value = 0.0
            prog["fade_in"].value = 0.0
            prog["current_time"].value = 0.0
            if "u_hidden" in prog:
                prog["u_hidden"].value = 0
        if self.approach_prog is not None:
            self.approach_prog["projection"].write(proj_bytes)
            self.approach_prog["preempt"].value = 0.0
            self.approach_prog["fade_in"].value = 0.0
            self.approach_prog["current_time"].value = 0.0
            self.approach_prog["approach_scale"].value = APPROACH_SCALE
            if "u_hidden" in self.approach_prog:
                self.approach_prog["u_hidden"].value = 0
        if self.ball_prog is not None:
            self.ball_prog["projection"].write(proj_bytes)
        if self.cursor_prog is not None:
            self.cursor_prog["projection"].write(proj_bytes)
        if self.trail_prog is not None:
            self.trail_prog["projection"].write(proj_bytes)
            self.trail_prog["u_color"].value = (1.0, 0.85, 0.2)
            if "u_trail_points" in self.trail_prog:
                self.trail_prog["u_trail_points"].value = 0
            if "u_radius" in self.trail_prog:
                self.trail_prog["u_radius"].value = TRAIL_STROKE_RADIUS
            if "u_lifetime_ms" in self.trail_prog:
                self.trail_prog["u_lifetime_ms"].value = TRAIL_LIFETIME_MS
            if "u_tail_taper_fraction" in self.trail_prog:
                self.trail_prog["u_tail_taper_fraction"].value = TRAIL_TAIL_TAPER_FRACTION

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
        if not self._hs_events:
            self._hs_events = data.hitsound_events
        self._hs_times = [float(ev.time_ms) for ev in self._hs_events]
        self._hs_cursor = 0
        self._prev_time_ms = -1e9

        proj_bytes = self.projection.astype("f4").tobytes()
        for prog in (self.circle_prog, self.slider_prog, self.spinner_prog):
            if prog is None:
                continue
            prog["preempt"].value = data.preempt
            prog["fade_in"].value = data.fade_in
            prog["projection"].write(proj_bytes)
        if self.approach_prog is not None:
            self.approach_prog["preempt"].value = data.preempt
            self.approach_prog["fade_in"].value = data.fade_in
            self.approach_prog["projection"].write(proj_bytes)
        if self.ball_prog is not None:
            self.ball_prog["projection"].write(proj_bytes)

        self._ball_color = list(self._skin.combo_colors()[0])
        self._upload_circles(data)
        self._upload_spinners(data)
        self._upload_sliders(data)
        self._setup_ball_vao(data)
        self._setup_cursor_vao()

    def _upload_circles(self, data: RenderData):
        n = len(data.circle_positions)
        self.n_circle_instances = n
        self._obj_to_circle_idx = {
            int(obj_idx): idx for idx, obj_idx in enumerate(data.circle_object_indices.tolist())
        }
        if n == 0:
            self.circle_vao = None
            self.approach_vao = None
            self._circle_render_window = None
            return

        color = self._skin.combo_colors()[0]
        miss_color = [0.55, 0.15, 0.15]

        buf = np.empty((n, 9), dtype="f4")
        buf[:, 0:2] = data.circle_positions
        buf[:, 2] = data.circle_radius
        buf[:, 3:6] = color
        buf[:, 6] = data.circle_start_times
        buf[:, 7] = data.circle_end_times
        buf[:, 8] = data.circle_z

        for obj_idx in self._missed_circle_indices:
            circle_idx = self._obj_to_circle_idx.get(int(obj_idx))
            if circle_idx is not None:
                buf[circle_idx, 3:6] = miss_color

        # Sort back-to-front: lowest z first (farthest/latest), highest z last
        # (closest/earliest). Earliest circles render last = on top visually.
        order = np.argsort(buf[:, 8])
        buf = buf[order]
        start_times = data.circle_start_times[order] - data.preempt
        end_times = data.circle_end_times[order] + 150.0

        self._circle_quad_vbo = self.ctx.buffer(_QUAD_VERTS.tobytes())
        self._circle_index_buf = self.ctx.buffer(_QUAD_INDICES.tobytes())
        self._circle_instance_gpu_buf = self.ctx.buffer(reserve=buf.nbytes)
        self._circle_render_window = self._build_active_render_window(
            buf,
            start_times,
            end_times,
            self._circle_instance_gpu_buf,
        )

        self.circle_vao = self.ctx.vertex_array(
            self.circle_prog,
            [
                (self._circle_quad_vbo, "2f", "in_vert"),
                (self._circle_instance_gpu_buf, "2f 1f 3f 1f 1f 1f/i",
                 "in_pos", "in_radius", "in_color", "in_start_time", "in_end_time", "in_z"),
            ],
            index_buffer=self._circle_index_buf,
        )

        if self.approach_prog is not None:
            self.approach_vao = self.ctx.vertex_array(
                self.approach_prog,
                [
                    (self._circle_quad_vbo, "2f", "in_vert"),
                    (self._circle_instance_gpu_buf, "2f 1f 3f 1f 1f 1f/i",
                     "in_pos", "in_radius", "in_color", "in_start_time", "in_end_time", "in_z"),
                ],
                index_buffer=self._circle_index_buf,
            )

    def _upload_spinners(self, data: RenderData):
        sd = data.spinner
        if self.spinner_prog is None or sd is None or len(sd.positions) == 0:
            self.spinner_vao = None
            self.n_spinner_instances = 0
            self._spinner_render_window = None
            return

        n = len(sd.positions)
        color = self._skin.combo_colors()[0]
        buf = np.empty((n, 9), dtype="f4")
        buf[:, 0:2] = sd.positions
        buf[:, 2] = sd.radii
        buf[:, 3:6] = color
        buf[:, 6] = sd.start_times
        buf[:, 7] = sd.end_times
        buf[:, 8] = sd.z_values

        self.n_spinner_instances = n
        self._spinner_quad_vbo = self.ctx.buffer(_QUAD_VERTS.tobytes())
        self._spinner_index_buf = self.ctx.buffer(_QUAD_INDICES.tobytes())
        self._spinner_instance_gpu_buf = self.ctx.buffer(reserve=buf.nbytes)
        self._spinner_render_window = self._build_active_render_window(
            buf,
            sd.start_times - data.preempt,
            sd.end_times + 180.0,
            self._spinner_instance_gpu_buf,
        )

        self.spinner_vao = self.ctx.vertex_array(
            self.spinner_prog,
            [
                (self._spinner_quad_vbo, "2f", "in_vert"),
                (self._spinner_instance_gpu_buf, "2f 1f 3f 1f 1f 1f/i",
                 "in_pos", "in_radius", "in_color", "in_start_time", "in_end_time", "in_z"),
            ],
            index_buffer=self._spinner_index_buf,
        )

    def _upload_sliders(self, data: RenderData):
        sd = data.slider
        if sd is None or sd.n_sliders == 0:
            self.slider_vao = None
            self.n_slider_instances = 0
            self._slider_render_window = None
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

        self._slider_quad_vbo = self.ctx.buffer(_QUAD_VERTS.tobytes())
        self._slider_index_buf = self.ctx.buffer(_QUAD_INDICES.tobytes())
        self._slider_instance_gpu_buf = self.ctx.buffer(reserve=buf.nbytes)
        self._slider_render_window = self._build_active_render_window(
            buf,
            sd.start_times - data.preempt,
            sd.end_times + 50.0,
            self._slider_instance_gpu_buf,
        )

        self.slider_vao = self.ctx.vertex_array(
            self.slider_prog,
            [
                (self._slider_quad_vbo, "2f", "in_vert"),
                (self._slider_instance_gpu_buf, "2f 2f 1f 1f 1f 3f 1f 1f 1f/i",
                 "in_bbox_min", "in_bbox_max", "in_path_start", "in_path_count",
                 "in_radius", "in_color", "in_start_time", "in_end_time", "in_z"),
            ],
            index_buffer=self._slider_index_buf,
        )

    def _build_active_render_window(
        self,
        full_data: np.ndarray,
        appear_times: np.ndarray,
        disappear_times: np.ndarray,
        instance_buf: moderngl.Buffer,
    ) -> _ActiveRenderWindow:
        appear_order = np.argsort(appear_times, kind="mergesort")
        disappear_order = np.argsort(disappear_times, kind="mergesort")
        return _ActiveRenderWindow(
            full_data=full_data,
            appear_times=np.asarray(appear_times[appear_order], dtype=np.float64),
            appear_indices=np.asarray(appear_order, dtype=np.int32),
            disappear_times=np.asarray(disappear_times[disappear_order], dtype=np.float64),
            disappear_indices=np.asarray(disappear_order, dtype=np.int32),
            instance_buf=instance_buf,
            upload_buf=np.empty_like(full_data),
            active_mask=np.zeros(len(full_data), dtype=bool),
        )

    def _rebuild_active_render_window(
        self,
        window: _ActiveRenderWindow | None,
        current_time_ms: float,
    ) -> None:
        if window is None:
            return
        start_cursor = bisect.bisect_right(window.appear_times, current_time_ms)
        end_cursor = bisect.bisect_left(window.disappear_times, current_time_ms)
        active_mask = window.active_mask
        if active_mask is None or len(active_mask) != len(window.full_data):
            active_mask = np.zeros(len(window.full_data), dtype=bool)
            window.active_mask = active_mask
        else:
            active_mask[:] = False
        if start_cursor > 0:
            active_mask[window.appear_indices[:start_cursor]] = True
        if end_cursor > 0:
            active_mask[window.disappear_indices[:end_cursor]] = False
        window.active_indices = np.flatnonzero(active_mask).astype(np.int32, copy=False)
        window.start_cursor = start_cursor
        window.end_cursor = end_cursor
        window.active_count = int(len(window.active_indices))
        window.dirty = True
        window.last_time_ms = current_time_ms

    def _sync_active_render_window(
        self,
        window: _ActiveRenderWindow | None,
        current_time_ms: float,
    ) -> None:
        if window is None:
            return
        if current_time_ms < window.last_time_ms - 0.5:
            self._rebuild_active_render_window(window, current_time_ms)
        else:
            changed = False
            while (
                window.start_cursor < len(window.appear_times)
                and window.appear_times[window.start_cursor] <= current_time_ms
            ):
                idx = int(window.appear_indices[window.start_cursor])
                if window.active_mask is not None:
                    window.active_mask[idx] = True
                window.start_cursor += 1
                changed = True
            while (
                window.end_cursor < len(window.disappear_times)
                and window.disappear_times[window.end_cursor] < current_time_ms
            ):
                idx = int(window.disappear_indices[window.end_cursor])
                if window.active_mask is not None and window.active_mask[idx]:
                    window.active_mask[idx] = False
                    changed = True
                window.end_cursor += 1
            if changed:
                if window.active_mask is not None:
                    window.active_indices = np.flatnonzero(window.active_mask).astype(np.int32, copy=False)
                window.active_count = int(len(window.active_indices))
                window.dirty = True
            window.last_time_ms = current_time_ms

        if not window.dirty:
            return
        count = int(len(window.active_indices))
        window.active_count = count
        if count > 0:
            window.upload_buf[:count] = window.full_data[window.active_indices]
            window.instance_buf.write(memoryview(window.upload_buf[:count]).cast("B"))
        window.dirty = False

    def _setup_ball_vao(self, data: RenderData):
        if self.ball_prog is None:
            self.ball_vao = None
            return
        self._ball_instance_buf = self.ctx.buffer(reserve=MAX_SLIDER_BALLS * 7 * 4)
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
        else:
            self._cursor_instance_buf = self.ctx.buffer(reserve=8 * 4)
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

        if self.trail_prog is None:
            self.trail_vao = None
        else:
            self._trail_point_tex = self.ctx.texture((TRAIL_MAX + 2, 1), 4, dtype="f4")
            self._trail_point_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._trail_point_tex.repeat_x = False
            self._trail_point_tex.repeat_y = False
            self.trail_vao = self.ctx.vertex_array(self.trail_prog, [])

    def _compute_timeline_bounds(self, data: RenderData) -> tuple[float, float]:
        start_ms = -self._audio.lead_in_ms if self._audio is not None else 0.0
        end_candidates = [0.0]

        if len(data.circle_end_times) > 0:
            end_candidates.append(float(np.max(data.circle_end_times)))
        if data.spinner is not None and len(data.spinner.end_times) > 0:
            end_candidates.append(float(np.max(data.spinner.end_times)))
        if data.slider is not None and len(data.slider.end_times) > 0:
            end_candidates.append(float(np.max(data.slider.end_times)))
        if self._hs_events:
            end_candidates.append(float(self._hs_events[-1].time_ms))
        for participant in self._participants:
            if participant.replay is not None and participant.replay.frames:
                end_candidates.append(float(participant.replay.frames[-1].time_ms) / max(0.05, self._speed))
            if participant.score_timeline.points:
                end_candidates.append(float(participant.score_timeline.points[-1].time_ms))
        if self._audio is not None and self._audio.duration_ms is not None:
            end_candidates.append(float(self._audio.duration_ms))

        end_ms = max(start_ms + 1.0, max(end_candidates))
        return start_ms, end_ms

    def _clamp_timeline_time(self, time_ms: float) -> float:
        return max(self._timeline_start_ms, min(self._timeline_end_ms, time_ms))

    def _timeline_elapsed_ms(self, current_time_ms: float) -> float:
        return max(0.0, min(self._timeline_total_ms, current_time_ms - self._timeline_start_ms))

    def _timeline_progress(self, current_time_ms: float) -> float:
        if self._timeline_total_ms <= 1e-6:
            return 0.0
        return self._timeline_elapsed_ms(current_time_ms) / self._timeline_total_ms

    def _invalidate_timeline_layout(self) -> None:
        self._timeline_layout_cache = None
        self._timeline_layout_cache_key = None

    def _gameplay_overlay_visible(self, key: str) -> bool:
        return bool(getattr(self.app.settings, key, True))

    def _gameplay_background_image_enabled(self) -> bool:
        return bool(getattr(self.app.settings, "gameplay_background_image", False))

    def _gameplay_background_path(self) -> str | None:
        if not self._binfo.background_file:
            return None
        bg_path = Path(self._binfo.directory) / self._binfo.background_file
        return str(bg_path) if bg_path.is_file() else None

    def _gameplay_trail_enabled(self) -> bool:
        return bool(getattr(self.app.settings, "gameplay_cursor_trail", True))

    def _gameplay_trail_max_len(self) -> int:
        return max(8, min(TRAIL_MAX, int(getattr(self.app.settings, "gameplay_cursor_trail_max_len", TRAIL_MAX))))

    def _format_timeline_time(self, time_ms: float) -> str:
        total_ms = max(0, int(round(time_ms)))
        minutes = total_ms // 60000
        seconds = (total_ms // 1000) % 60
        millis = total_ms % 1000
        return f"{minutes:02d}:{seconds:02d}:{millis:03d}"

    def _participant_by_id(self, leaderboard_id: str | None) -> _GameplayParticipant | None:
        if leaderboard_id is None:
            return None
        return self._participants_by_id.get(leaderboard_id)

    def _leaderboard_hit_test(self, x: float, y: float) -> str | None:
        if not self._gameplay_overlay_visible("draw_gameplay_leaderboard"):
            return None
        for rx, ry, rw, rh, leaderboard_id in reversed(self._leaderboard_rects):
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return leaderboard_id
        return None

    def _leaderboard_focus_participant(self) -> _GameplayParticipant | None:
        pinned = self._participant_by_id(self._leaderboard_pinned_id)
        if pinned is not None:
            return pinned
        return self._participant_by_id(self._leaderboard_hover_id)

    def _display_score_participant(self) -> _GameplayParticipant | None:
        focused = self._leaderboard_focus_participant()
        if focused is not None:
            return focused
        return self._hud_score_participant

    def _display_key_participant(self) -> _GameplayParticipant | None:
        focused = self._leaderboard_focus_participant()
        if focused is not None:
            return focused
        return self._hud_participant

    def _gameplay_source_participant(self) -> _GameplayParticipant | None:
        focused = self._leaderboard_focus_participant()
        if focused is not None:
            return focused
        return self._hud_participant

    def _apply_gameplay_source_participant(
        self,
        participant: _GameplayParticipant | None,
        *,
        current_time_ms: float | None = None,
    ) -> None:
        source_id = None if participant is None else participant.leaderboard_id
        if source_id == self._active_gameplay_source_id:
            return

        previous_misses = self._missed_circle_indices
        self._active_gameplay_source_id = source_id
        self._replay = participant.replay if participant is not None else None
        self._visual_replay = participant.visual_replay if participant is not None else None
        self._judge = participant.judge if participant is not None else None
        self._missed_circle_indices = (
            participant.missed_circle_indices if participant is not None else set()
        )
        self._cpath = participant.cpath if participant is not None else None

        if self._replay is None and participant is not None:
            self._auto_key_schedule = participant.auto_key_schedule
            self._auto_key_start_times = [start_ms for _, start_ms, _ in self._auto_key_schedule]
        else:
            self._auto_key_schedule = []
            self._auto_key_start_times = []
        self._auto_key_cursor = 0

        audio_source = participant if participant is not None else self._audio_participant
        self._hs_events = audio_source.hs_events if audio_source is not None else []
        self._hs_times = [float(ev.time_ms) for ev in self._hs_events]
        if current_time_ms is None:
            self._hs_cursor = 0
            self._prev_time_ms = -1e9
        else:
            self._hs_cursor = bisect.bisect_right(self._hs_times, current_time_ms)
            self._prev_time_ms = current_time_ms

        if self._stored_data is not None and previous_misses != self._missed_circle_indices:
            self._upload_circles(self._stored_data)

    def _display_participant_label(self) -> str:
        focused = self._leaderboard_focus_participant()
        if focused is not None:
            return focused.name
        if self._hud_score_participant is None:
            return ""
        if self._hud_score_participant.source_type == "perfect":
            return "Perfect"
        return self._hud_score_participant.name

    def _hp_for_point(self, timeline: StablePerformanceTimeline, point, time_ms: float) -> float:
        return timeline.hp_at(time_ms, point=point)

    def _set_paused(self, paused: bool, *, show_menu: bool | None = None) -> None:
        self._paused = paused
        if not paused:
            self._pause_menu_open = False
        elif show_menu is not None:
            self._pause_menu_open = bool(show_menu)
        if paused:
            self._clear_trails()
        if self._audio is not None:
            self._audio.set_paused(paused)

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _volume_control_value(self, key: str) -> float:
        if key == "music":
            return self.app.settings.music_volume
        return self.app.settings.sfx_volume

    def _volume_control_muted(self, key: str) -> bool:
        if key == "music":
            return self.app.settings.music_muted
        return self.app.settings.sfx_muted

    def _refresh_volume_slider_timer(self, key: str) -> None:
        self._volume_slider_hide_timers[key] = VOLUME_SLIDER_HIDE_DELAY

    def _volume_controls_active(self) -> bool:
        return (
            self._volume_slider_dragging is not None
            or any(value > 0.01 for value in self._volume_slider_visibility.values())
            or any(value > 0.0 for value in self._volume_slider_hide_timers.values())
        )

    def _close_volume_sliders(self) -> None:
        self._volume_slider_dragging = None
        for key in self._volume_slider_visibility:
            self._volume_slider_visibility[key] = 0.0
            self._volume_slider_hide_timers[key] = 0.0

    def _set_music_volume(self, value: float) -> None:
        value = self._clamp_unit(value)
        self.app.set_music_volume(value)
        self.app.set_music_muted(value <= 0.0001)
        if self._audio is not None:
            self._audio.set_volume(self.app.settings.music_volume)
            self._audio.set_muted(self.app.settings.music_muted)

    def _set_music_muted(self, muted: bool) -> None:
        self.app.set_music_muted(muted)
        if self._audio is not None:
            self._audio.set_muted(self.app.settings.music_muted)

    def _set_sfx_volume(self, value: float) -> None:
        value = self._clamp_unit(value)
        self.app.set_sfx_volume(value)
        self.app.set_sfx_muted(value <= 0.0001)
        if self._hitsounds is not None:
            self._hitsounds.set_volume(self.app.settings.sfx_volume)
            self._hitsounds.set_muted(self.app.settings.sfx_muted)

    def _set_sfx_muted(self, muted: bool) -> None:
        self.app.set_sfx_muted(muted)
        if self._hitsounds is not None:
            self._hitsounds.set_muted(self.app.settings.sfx_muted)

    def _set_volume_control_value(self, key: str, value: float) -> None:
        if key == "music":
            self._set_music_volume(value)
        else:
            self._set_sfx_volume(value)

    def _toggle_volume_control_muted(self, key: str) -> None:
        if key == "music":
            self._set_music_muted(not self.app.settings.music_muted)
        else:
            self._set_sfx_muted(not self.app.settings.sfx_muted)

    def _reset_key_overlay_state(self) -> None:
        self._k1_count = 0
        self._k2_count = 0
        self._k1_pressed = False
        self._k2_pressed = False
        self._k1_press_start = -1.0
        self._k2_press_start = -1.0
        self._k1_release_time = -1.0
        self._k2_release_time = -1.0
        self._key_bars = []

    def _finalize_rebuilt_key_bar(self, key: int, start_ms: float, end_ms: float) -> None:
        duration_ms = max(0.0, end_ms - start_ms)
        if duration_ms >= 1.2:
            self._key_bars.append((key, start_ms, max(1.2, duration_ms)))
        elif key == 1:
            self._k1_count = max(0, self._k1_count - 1)
        else:
            self._k2_count = max(0, self._k2_count - 1)

    def _build_auto_key_schedule(self, beatmap, speed: float) -> list[tuple[int, float, float]]:
        from osupyparser.osu.objects import Slider as SliderObj
        from osupyparser.osu.objects import Spinner as SpinnerObj

        inv_speed = 1.0 / max(0.05, speed)
        objects = beatmap.hit_objects
        starts = [float(obj.start_time) * inv_speed for obj in objects]
        schedule: list[tuple[int, float, float]] = []
        alt = False

        for idx, obj in enumerate(objects):
            start_ms = starts[idx]
            key = 2 if alt else 1

            if isinstance(obj, SliderObj):
                end_ms = float(obj.end_time) * inv_speed
            elif isinstance(obj, SpinnerObj):
                end_ms = float(obj.end_time) * inv_speed
            else:
                gap_ms = 90.0
                if idx > 0:
                    gap_ms = min(gap_ms, max(20.0, starts[idx] - starts[idx - 1]))
                if idx + 1 < len(starts):
                    gap_ms = min(gap_ms, max(20.0, starts[idx + 1] - starts[idx]))
                end_ms = start_ms + max(32.0, min(90.0, gap_ms * 0.58))

            if end_ms <= start_ms:
                end_ms = start_ms + 32.0
            schedule.append((key, start_ms, end_ms))
            alt = not alt

        return schedule

    @staticmethod
    def _replay_key_groups(keys: int) -> tuple[bool, bool]:
        return bool(keys & (1 | 4)), bool(keys & (2 | 8))

    @staticmethod
    def _key_overlay_source_signature(
        participant: _GameplayParticipant | None,
    ) -> tuple[str, int] | None:
        if participant is None:
            return None
        if participant.replay is not None:
            return ("replay", id(participant.replay))
        if participant.cpath is not None:
            return ("auto", id(participant))
        return None

    def _rebuild_key_overlay_for_source(
        self,
        current_time_ms: float,
        key_source: _GameplayParticipant | None,
    ) -> None:
        self._reset_key_overlay_state()
        self._auto_key_cursor = 0
        self._replay_key_cursor = 0
        self._key_overlay_source_sig = self._key_overlay_source_signature(key_source)
        self._key_overlay_time_ms = current_time_ms
        if key_source is None:
            return
        if key_source.replay is not None:
            self._rebuild_replay_key_overlay(current_time_ms, key_source.replay)
        elif key_source.cpath is not None:
            self._rebuild_auto_key_overlay(current_time_ms, key_source.auto_key_schedule)

    def _rebuild_replay_key_overlay(self, current_time_ms: float, replay: ReplayData | None = None) -> None:
        replay = self._replay if replay is None else replay
        if replay is None:
            return

        replay_target_ms = max(0.0, current_time_ms * max(0.05, self._speed))
        speed = max(0.05, self._speed)
        interval_limit = bisect.bisect_right(replay._key_interval_start_times, replay_target_ms)
        for idx in range(interval_limit):
            key, start_ms, end_ms = replay._key_intervals[idx]
            wall_start_ms = float(start_ms) / speed
            if key == 1:
                self._k1_count += 1
                if end_ms <= replay_target_ms:
                    self._finalize_rebuilt_key_bar(1, wall_start_ms, float(end_ms) / speed)
                else:
                    self._k1_pressed = True
                    self._k1_press_start = wall_start_ms
                    self._k1_release_time = -1.0 if end_ms == float("inf") else float(end_ms) / speed
            else:
                self._k2_count += 1
                if end_ms <= replay_target_ms:
                    self._finalize_rebuilt_key_bar(2, wall_start_ms, float(end_ms) / speed)
                else:
                    self._k2_pressed = True
                    self._k2_press_start = wall_start_ms
                    self._k2_release_time = -1.0 if end_ms == float("inf") else float(end_ms) / speed

        self._replay_key_cursor = bisect.bisect_right(replay._times, replay_target_ms)
        self._key_overlay_time_ms = current_time_ms

    def _advance_replay_key_overlay(self, current_time_ms: float, replay: ReplayData | None = None) -> None:
        replay = self._replay if replay is None else replay
        if replay is None:
            return
        replay_target_ms = max(0.0, current_time_ms * max(0.05, self._speed))
        limit = bisect.bisect_right(replay._times, replay_target_ms)
        if limit < self._replay_key_cursor:
            self._rebuild_replay_key_overlay(current_time_ms, replay)
            return

        speed = max(0.05, self._speed)
        for idx in range(self._replay_key_cursor, limit):
            wall_time_ms = float(replay._times[idx]) / speed
            k1_now, k2_now = self._replay_key_groups(int(replay._keys[idx]))
            if k1_now != self._k1_pressed:
                if k1_now:
                    self._press_overlay_key(1, wall_time_ms)
                else:
                    self._release_overlay_key(1, wall_time_ms)
            if k2_now != self._k2_pressed:
                if k2_now:
                    self._press_overlay_key(2, wall_time_ms)
                else:
                    self._release_overlay_key(2, wall_time_ms)

        self._replay_key_cursor = limit
        self._key_overlay_time_ms = current_time_ms

    def _rebuild_auto_key_overlay(self, current_time_ms: float,
                                  schedule: list[tuple[int, float, float]] | None = None) -> None:
        schedule = self._auto_key_schedule if schedule is None else schedule
        start_times = self._auto_key_start_times if schedule is self._auto_key_schedule else [start for _, start, _ in schedule]
        self._auto_key_cursor = bisect.bisect_right(start_times, current_time_ms)
        for key, start_ms, end_ms in schedule:
            if start_ms > current_time_ms:
                break

            if key == 1:
                self._k1_count += 1
                if end_ms <= current_time_ms:
                    self._finalize_rebuilt_key_bar(1, start_ms, end_ms)
                else:
                    self._k1_pressed = True
                    self._k1_press_start = start_ms
                    self._k1_release_time = end_ms
            else:
                self._k2_count += 1
                if end_ms <= current_time_ms:
                    self._finalize_rebuilt_key_bar(2, start_ms, end_ms)
                else:
                    self._k2_pressed = True
                    self._k2_press_start = start_ms
                    self._k2_release_time = end_ms
        self._key_overlay_time_ms = current_time_ms

    def _advance_auto_key_overlay(self, current_time_ms: float,
                                  schedule: list[tuple[int, float, float]] | None = None) -> None:
        schedule = self._auto_key_schedule if schedule is None else schedule
        start_times = self._auto_key_start_times if schedule is self._auto_key_schedule else [start for _, start, _ in schedule]
        if current_time_ms + 0.5 < self._key_overlay_time_ms:
            self._rebuild_auto_key_overlay(current_time_ms, schedule)
            return

        def release_due(limit_time_ms: float) -> None:
            while True:
                next_key = 0
                next_release = float("inf")
                if self._k1_pressed and self._k1_release_time >= 0.0 and self._k1_release_time <= limit_time_ms:
                    next_key = 1
                    next_release = self._k1_release_time
                if (
                    self._k2_pressed
                    and self._k2_release_time >= 0.0
                    and self._k2_release_time <= limit_time_ms
                    and self._k2_release_time < next_release
                ):
                    next_key = 2
                    next_release = self._k2_release_time
                if next_key == 0:
                    break
                self._release_overlay_key(next_key, next_release)

        limit = bisect.bisect_right(start_times, current_time_ms)
        while self._auto_key_cursor < limit:
            key, start_ms, end_ms = schedule[self._auto_key_cursor]
            release_due(start_ms)
            self._press_overlay_key(key, start_ms, end_ms)
            self._auto_key_cursor += 1

        release_due(current_time_ms)
        self._key_overlay_time_ms = current_time_ms

    def _seed_trail(self, current_time_ms: float, participant: _GameplayParticipant | None = None) -> None:
        trail = self._trail if participant is None else participant.trail
        replay = self._replay if participant is None else participant.replay
        cpath = self._cpath if participant is None else participant.cpath
        trail.clear()
        if cpath is None and replay is None:
            return

        start_ms = max(self._timeline_start_ms, current_time_ms - TRAIL_LIFETIME_MS)
        if current_time_ms <= start_ms:
            cx, cy = self._cursor_pos_at(current_time_ms, participant)
            trail.append((current_time_ms, cx, cy))
            return

        sample_count = min(64, max(12, int((current_time_ms - start_ms) / 10.0)))
        for sample_time in np.linspace(start_ms, current_time_ms, num=sample_count):
            cx, cy = self._cursor_pos_at(float(sample_time), participant)
            trail.append((float(sample_time), cx, cy))

    def _clear_trails(self) -> None:
        self._trail.clear()
        for participant in self._participants:
            participant.trail.clear()

    def _append_trail_point(
        self,
        trail: deque[tuple[float, float, float]],
        time_ms: float,
        x: float,
        y: float,
    ) -> None:
        if not trail:
            trail.append((time_ms, x, y))
            return
        if len(trail) == 1:
            trail.append((time_ms, x, y))
            return

        anchor_t, anchor_x, anchor_y = trail[-2]
        dt = time_ms - anchor_t
        dx = x - anchor_x
        dy = y - anchor_y
        if dt >= TRAIL_MIN_SAMPLE_DT_MS or (dx * dx + dy * dy) >= TRAIL_MIN_SAMPLE_DIST_SQ:
            trail.append((time_ms, x, y))
        else:
            trail[-1] = (time_ms, x, y)

    def _trail_sample_count(
        self,
        participant: _GameplayParticipant | None,
        path_length_px: float,
        duration_ms: float,
    ) -> int:
        target = TRAIL_SUBSAMPLE_TARGET
        if len(self._participants) >= 10 and participant is not None and participant is not self._hud_participant:
            target = min(target, 72)
        elif len(self._participants) >= 6 and participant is not None and participant is not self._hud_participant:
            target = min(target, 104)
        target = max(16, min(target, self._gameplay_trail_max_len()))
        by_length = int(path_length_px / TRAIL_RESAMPLE_STEP_PX) + 1
        by_time = int(max(0.0, duration_ms) / 4.0) + 1
        return min(target, max(16, max(by_length, by_time)))

    def _rebuild_state_for_time(self, current_time_ms: float) -> None:
        current_time_ms = self._clamp_timeline_time(current_time_ms)
        self._current_time_ms = current_time_ms
        self._leaderboard_last_sort_ms = -1e9

        hit_count = bisect.bisect_right(self._hs_times, current_time_ms)
        self._hs_cursor = hit_count
        self._sync_score_state_for_time(current_time_ms, from_seek=True)

        key_source = self._display_key_participant()
        self._rebuild_key_overlay_for_source(current_time_ms, key_source)
        self._rebuild_active_render_window(self._circle_render_window, current_time_ms)
        self._rebuild_active_render_window(self._spinner_render_window, current_time_ms)
        self._rebuild_active_render_window(self._slider_render_window, current_time_ms)

        self._prune_key_bars(current_time_ms)

        for participant in self._participants:
            self._seed_trail(current_time_ms, participant)
        self._prev_time_ms = current_time_ms
        if current_time_ms > self._timeline_start_ms + 5.0:
            self._cursor_intro_timer = CURSOR_INTRO_DURATION

    def _seek_to_time(self, target_ms: float, *, update_audio: bool = True) -> None:
        target_ms = self._clamp_timeline_time(target_ms)
        if abs(target_ms - self._current_time_ms) < 0.5 and (self._audio is None or not update_audio):
            self._manual_time_ms = target_ms
            return
        self._manual_time_ms = target_ms
        if self._audio is not None and update_audio:
            self._audio.seek_ms(target_ms)
        self._rebuild_state_for_time(target_ms)
        self._clear_trails()

    def _set_playback_speed(self, speed: float) -> None:
        speed = max(0.5, min(2.0, speed))
        if abs(speed - self._playback_speed) < 1e-4:
            return

        current_time_ms = self._current_time_ms
        self._playback_speed = speed
        if self._audio is not None:
            self._audio.set_playback_speed(speed)
            self._rebuild_state_for_time(current_time_ms)
        else:
            self._seek_to_time(current_time_ms)

    def _seek_relative(self, delta_ms: float) -> None:
        anchor_ms = self._manual_time_ms if self._timeline_dragging else self._current_time_ms
        self._seek_to_time(anchor_ms + delta_ms, update_audio=not self._timeline_dragging)

    def _begin_timeline_drag(self, target_ms: float) -> None:
        target_ms = self._clamp_timeline_time(target_ms)
        self._timeline_dragging = True
        self._timeline_hovered = True
        self._timeline_hover_time_ms = target_ms
        self._timeline_drag_resume_playback = bool(self._audio is not None and not self._paused)
        if self._timeline_drag_resume_playback and self._audio is not None:
            self._audio.set_paused(True)
        self._seek_to_time(target_ms, update_audio=False)

    def _update_timeline_drag(self, target_ms: float) -> None:
        target_ms = self._clamp_timeline_time(target_ms)
        self._timeline_hovered = True
        self._timeline_hover_time_ms = target_ms
        if abs(target_ms - self._current_time_ms) < 1.0:
            self._manual_time_ms = target_ms
            return
        self._seek_to_time(target_ms, update_audio=False)

    def _finish_timeline_drag(self, target_ms: float | None = None) -> None:
        if not self._timeline_dragging:
            return
        target = self._manual_time_ms if target_ms is None else target_ms
        self._timeline_dragging = False
        self._seek_to_time(target, update_audio=True)
        if self._timeline_drag_resume_playback and self._audio is not None and not self._paused:
            self._audio.set_paused(False)
        self._timeline_drag_resume_playback = False

    # ----------------------------------------------------------- cursor + balls

    def _cursor_pos_at(self, time_ms: float, participant: _GameplayParticipant | None = None) -> tuple[float, float]:
        """Get cursor position from replay or danser auto path.

        *time_ms* is in wall-clock domain; replay frames use original time,
        so we scale by speed to convert back when DT/HT is active.
        """
        if participant is not None and participant.eliminated and participant.elimination_started_ms >= 0.0:
            time_ms = min(time_ms, participant.elimination_started_ms)
        replay = self._visual_replay if participant is None else participant.visual_replay
        cpath = self._cpath if participant is None else participant.cpath
        if replay is not None:
            replay_time = time_ms * getattr(self, "_speed", 1.0)
            ox, oy = replay.position_at(replay_time)
            if getattr(self, "_hr_flip", False):
                return ox, float(oy)
            return ox, OSU_PLAYFIELD_HEIGHT - oy
        if cpath is not None:
            cpath_time = time_ms * getattr(self, "_speed", 1.0)
            p = cpath.position_at(cpath_time)
            return float(p[0]), float(p[1])
        return PLAYFIELD_CX, PLAYFIELD_CY

    def _participant_cursor_alpha(self, participant: _GameplayParticipant, current_time_ms: float) -> float:
        if not participant.eliminated or participant.elimination_started_ms < 0.0:
            alpha = 1.0
        else:
            elapsed = current_time_ms - participant.elimination_started_ms
            alpha = max(0.0, min(1.0, 1.0 - elapsed / 450.0))

        focused = self._leaderboard_focus_participant()
        if focused is None:
            return alpha
        if focused.leaderboard_id == participant.leaderboard_id:
            return alpha
        return alpha * (1.0 - 0.72 * self._leaderboard_focus_presence)

    def _build_trail_ribbon(self, current_time_ms: float,
                            participant: _GameplayParticipant | None = None,
                            alpha_scale: float = 1.0) -> tuple[np.ndarray, int, int]:
        """Build clipped control points for GPU spline evaluation."""
        if not self._gameplay_trail_enabled():
            return self._trail_gpu_buf[:0], 0, 0
        trail = self._trail if participant is None else participant.trail
        count = len(trail)
        if count < 2:
            return self._trail_gpu_buf[:0], 0, 0

        max_points = self._gameplay_trail_max_len()
        if count > max_points:
            trail_points = list(trail)[-max_points:]
            count = len(trail_points)
            raw = self._trail_raw_buf[:count]
            raw[:] = trail_points
        else:
            raw = self._trail_raw_buf[:count]
            raw[:] = trail

        start_time = max(float(raw[0, 0]), current_time_ms - TRAIL_LIFETIME_MS)
        end_time = current_time_ms
        if end_time - start_time <= 1.0:
            return self._trail_gpu_buf[:0], 0, 0

        clip = self._trail_clip_buf
        native_clip_count = clip_trail_points(
            raw,
            count,
            current_time_ms,
            TRAIL_LIFETIME_MS,
            TRAIL_MIN_CONTROL_DT_MS,
            TRAIL_MIN_CONTROL_DIST_SQ,
            clip,
        )
        if native_clip_count is not None:
            clip_count = int(native_clip_count)
        else:
            clip_count = 0
            start_x = float(np.interp(start_time, raw[:, 0], raw[:, 1]))
            start_y = float(np.interp(start_time, raw[:, 0], raw[:, 2]))
            clip[clip_count] = (start_time, start_x, start_y)
            clip_count += 1

            for idx in range(count):
                t = float(raw[idx, 0])
                if start_time < t < end_time:
                    clip[clip_count] = raw[idx]
                    clip_count += 1

            end_x = float(np.interp(end_time, raw[:, 0], raw[:, 1]))
            end_y = float(np.interp(end_time, raw[:, 0], raw[:, 2]))
            if clip_count == 0 or abs(end_x - float(clip[clip_count - 1, 1])) > 1e-4 or abs(end_y - float(clip[clip_count - 1, 2])) > 1e-4:
                clip[clip_count] = (end_time, end_x, end_y)
                clip_count += 1

        if clip_count < 2:
            return self._trail_gpu_buf[:0], 0, 0

        clip_xy = clip[:clip_count, 1:3]
        diffs = np.diff(clip_xy, axis=0)
        seg_lens = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        total_len = float(np.sum(seg_lens))
        sample_count = self._trail_sample_count(participant, total_len, end_time - start_time)
        gpu = self._trail_gpu_buf[:clip_count]
        gpu[:, 0] = clip[:clip_count, 1]
        gpu[:, 1] = clip[:clip_count, 2]
        gpu[:, 2] = clip[:clip_count, 0]
        gpu[:, 3] = 0.0
        return gpu, clip_count, sample_count

    def _prune_key_bars(self, current_time_ms: float) -> None:
        bars = self._key_bars
        if not bars:
            return
        cutoff_time = current_time_ms - 900.0
        first_alive = 0
        for _key, start_ms, duration_ms in bars:
            if start_ms + duration_ms >= cutoff_time:
                break
            first_alive += 1
        if first_alive > 0:
            del bars[:first_alive]

    def _build_cursor_head(self, current_time_ms: float,
                           participant: _GameplayParticipant | None = None,
                           alpha_scale: float = 1.0) -> tuple[np.ndarray, int]:
        """Build single circle instance for the cursor head."""
        replay = self._replay if participant is None else participant.replay
        cpath = self._cpath if participant is None else participant.cpath
        if cpath is None and replay is None:
            return self._cursor_head_buf[:0], 0

        intro_progress = min(1.0, self._cursor_intro_timer / CURSOR_INTRO_DURATION)
        intro_scale = _ease_out_back(intro_progress) if intro_progress < 1.0 else 1.0

        cx, cy = self._cursor_pos_at(current_time_ms, participant)
        trail = self._trail if participant is None else participant.trail
        if self._paused or self._timeline_dragging:
            trail.clear()
        else:
            while trail and current_time_ms - trail[0][0] > TRAIL_LIFETIME_MS:
                trail.popleft()
            self._append_trail_point(trail, current_time_ms, cx, cy)
        color = (1.0, 0.92, 0.3) if participant is None else participant.color
        alpha = min(1.0, intro_scale) * max(0.0, min(1.0, alpha_scale))

        buf = self._cursor_head_buf
        buf[0] = [
            cx,
            cy,
            CURSOR_RADIUS * intro_scale,
            color[0],
            color[1],
            color[2],
            alpha,
            -0.95,
        ]
        return buf, 1

    def _compute_slider_balls(self, current_time_ms: float) -> np.ndarray:
        data = self._stored_data
        if data is None or data.slider is None or data.slider.n_sliders == 0:
            return self._slider_ball_buf[:0]

        sd = data.slider
        native_count = compute_slider_ball_instances(
            current_time_ms,
            float(data.circle_radius),
            self._ball_color,
            sd.start_times,
            sd.end_times,
            sd.repeat_counts,
            sd.path_starts,
            sd.path_counts,
            sd.path_points,
            sd.z_values,
            self._slider_ball_buf,
            MAX_SLIDER_BALLS,
        )
        if native_count >= 0:
            return self._slider_ball_buf[:native_count]
        start_idx = int(np.searchsorted(sd.end_times, current_time_ms, side="left"))
        end_idx = int(np.searchsorted(sd.start_times, current_time_ms, side="right"))
        if end_idx <= start_idx:
            return self._slider_ball_buf[:0]

        ball_count = 0
        for i in range(start_idx, min(end_idx, sd.n_sliders)):
            start = float(sd.start_times[i])
            end = float(sd.end_times[i])
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
            c = self._ball_color
            self._slider_ball_buf[ball_count] = [
                pos[0], pos[1], data.circle_radius, c[0], c[1], c[2], z
            ]
            ball_count += 1

            if ball_count >= MAX_SLIDER_BALLS:
                break

        return self._slider_ball_buf[:ball_count]

    # ----------------------------------------------------------- hitsounds

    def _press_overlay_key(self, key: int, time_ms: float,
                           release_time_ms: float | None = None) -> None:
        if key == 1:
            if self._k1_pressed and self._k1_press_start >= 0.0:
                self._release_overlay_key(1, time_ms, filter_short=False)
            self._k1_count += 1
            self._k1_pressed = True
            self._k1_press_start = time_ms
            self._k1_release_time = -1.0 if release_time_ms is None else release_time_ms
        else:
            if self._k2_pressed and self._k2_press_start >= 0.0:
                self._release_overlay_key(2, time_ms, filter_short=False)
            self._k2_count += 1
            self._k2_pressed = True
            self._k2_press_start = time_ms
            self._k2_release_time = -1.0 if release_time_ms is None else release_time_ms

    def _release_overlay_key(self, key: int, time_ms: float,
                             filter_short: bool = True) -> None:
        if key == 1:
            if not self._k1_pressed or self._k1_press_start < 0.0:
                return
            dur = max(0.0, time_ms - self._k1_press_start)
            if dur >= 1.2 or not filter_short:
                self._key_bars.append((1, self._k1_press_start, max(1.2, dur)))
            else:
                self._k1_count = max(0, self._k1_count - 1)
            self._k1_pressed = False
            self._k1_press_start = -1.0
            self._k1_release_time = -1.0
        else:
            if not self._k2_pressed or self._k2_press_start < 0.0:
                return
            dur = max(0.0, time_ms - self._k2_press_start)
            if dur >= 1.2 or not filter_short:
                self._key_bars.append((2, self._k2_press_start, max(1.2, dur)))
            else:
                self._k2_count = max(0, self._k2_count - 1)
            self._k2_pressed = False
            self._k2_press_start = -1.0
            self._k2_release_time = -1.0

    def _auto_hold_duration_ms(self, event_idx: int) -> float:
        gap_ms = 90.0
        if event_idx > 0:
            prev_gap = self._hs_events[event_idx].time_ms - self._hs_events[event_idx - 1].time_ms
            gap_ms = min(gap_ms, max(20.0, prev_gap))
        if event_idx + 1 < len(self._hs_events):
            next_gap = self._hs_events[event_idx + 1].time_ms - self._hs_events[event_idx].time_ms
            gap_ms = min(gap_ms, max(20.0, next_gap))
        return float(max(32.0, min(90.0, gap_ms * 0.58)))

    def _trigger_hitsounds(self, current_time_ms: float) -> None:
        if not self._hs_events or self._hitsounds is None:
            return

        if self._prev_time_ms < -1e8:
            self._prev_time_ms = current_time_ms
            while (self._hs_cursor < len(self._hs_events)
                   and self._hs_events[self._hs_cursor].time_ms <= current_time_ms):
                self._hs_cursor += 1
            return

        if current_time_ms < self._prev_time_ms - 100:
            self._hs_cursor = 0
            while (self._hs_cursor < len(self._hs_events)
                   and self._hs_events[self._hs_cursor].time_ms <= current_time_ms):
                self._hs_cursor += 1
            self._prev_time_ms = current_time_ms
            return

        while self._hs_cursor < len(self._hs_events):
            event_idx = self._hs_cursor
            ev = self._hs_events[self._hs_cursor]
            if ev.time_ms > current_time_ms:
                break
            self._hitsounds.play(
                ev.normal_set, ev.addition_set, ev.sound_enum, ev.volume,
            )
            self._hs_cursor += 1

    def _build_replay_hs_events(self, beatmap, render_data: RenderData,
                                judge: HitJudge | None = None) -> list[HitsoundEvent]:
        """Build hitsound events timed to actual replay key presses."""
        from osupyparser.osu.objects import Slider as SliderObj

        judge = self._judge if judge is None else judge
        if judge is None:
            return render_data.hitsound_events

        events: list[HitsoundEvent] = []
        hit_obj_set: set[int] = set()
        for r in judge.results:
            if r.result != "miss":
                hit_obj_set.add(r.obj_index)

        for obj_idx, obj in enumerate(beatmap.hit_objects):
            if obj_idx not in hit_obj_set:
                continue

            hit_result = judge.result_for(obj_idx)
            if hit_result is None:
                continue

            _, _, tp_set_id, tp_volume = beatmap._timing_at(float(obj.start_time))
            additions = getattr(obj, "additions", None)
            normal_set, addition_set = beatmap._resolve_sample_set(additions, tp_set_id)
            volume = beatmap._resolve_volume(additions, tp_volume)

            if isinstance(obj, SliderObj):
                events.append(HitsoundEvent(
                    hit_result.hit_time_ms, normal_set, addition_set,
                    obj.sound_enum, volume,
                ))
                rc = max(obj.repeat_count, 1)
                for i, edge in enumerate(obj.edges):
                    if i == 0:
                        continue
                    edge_time = float(obj.start_time) + obj.duration * i / rc
                    _, _, e_tp_set, e_tp_vol = beatmap._timing_at(edge_time)
                    try:
                        e_sound = int(edge.sound_types) if edge.sound_types else obj.sound_enum
                    except (ValueError, TypeError):
                        e_sound = obj.sound_enum
                    e_adds = edge.additions
                    e_normal, e_addition = beatmap._resolve_sample_set(e_adds, e_tp_set)
                    e_vol = beatmap._resolve_volume(e_adds, e_tp_vol)
                    events.append(HitsoundEvent(
                        edge_time, e_normal, e_addition, e_sound, e_vol,
                    ))
            else:
                events.append(HitsoundEvent(
                    hit_result.hit_time_ms, normal_set, addition_set,
                    obj.sound_enum, volume,
                ))

        events.sort(key=lambda e: e.time_ms)
        return events

    def _timeline_layout(self) -> dict:
        w, h = self.wnd.buffer_size
        cache_key = (w, h, int(round(self._timeline_total_ms)))
        if self._timeline_layout_cache_key == cache_key and self._timeline_layout_cache is not None:
            return self._timeline_layout_cache
        text = self.app.text

        line_w = float(max(220, min(430, int(w * 0.28))))
        total_label = self._format_timeline_time(self._timeline_total_ms)
        timer_w, _ = text.measure(f"{total_label}/{total_label}", 14)
        chip_gap = 6.0
        chip_specs = [
            {"kind": "pause", "value": "pause", "label": "Pause"},
            {"kind": "speed", "value": 0.5, "label": "x0.5"},
            {"kind": "speed", "value": 1.0, "label": "x1.0"},
            {"kind": "speed", "value": 1.5, "label": "x1.5"},
            {"kind": "speed", "value": 2.0, "label": "x2.0"},
        ]
        chip_widths: list[float] = []
        for chip in chip_specs:
            chip_text = chip["label"]
            chip_text_w, _ = text.measure(chip_text, 13)
            chip_widths.append(chip_text_w + 18.0)

        chips_total_w = sum(chip_widths) + chip_gap * (len(chip_widths) - 1)
        control_group_gap = 8.0
        controls_total_w = VOLUME_BUTTON_SIZE * 2.0 + control_group_gap
        row_h = max(VOLUME_BUTTON_SIZE, TIMELINE_CHIP_HEIGHT, 30.0)
        total_w = timer_w + 14.0 + line_w + 14.0 + chips_total_w + 12.0 + controls_total_w
        x = w - total_w - 18.0
        y = h - row_h - 18.0
        line_x = x + timer_w + 14.0
        line_y = y + (row_h - TIMELINE_TRACK_HEIGHT) * 0.5
        chip_left = line_x + line_w + 14.0
        chip_y = y + (row_h - TIMELINE_CHIP_HEIGHT) * 0.5
        chips = []
        for chip_spec, chip_w in zip(chip_specs, chip_widths):
            chip_text_w, _ = text.measure(chip_spec["label"], 13)
            chips.append({
                "kind": chip_spec["kind"],
                "value": chip_spec["value"],
                "label": chip_spec["label"],
                "text_w": chip_text_w,
                "x": chip_left,
                "y": chip_y,
                "w": chip_w,
                "h": TIMELINE_CHIP_HEIGHT,
            })
            chip_left += chip_w + chip_gap

        controls_x = chip_left + 8.0
        volume_controls = {}
        for idx, (key, label) in enumerate((("music", "M"), ("sfx", "S"))):
            button_x = controls_x + idx * (VOLUME_BUTTON_SIZE + control_group_gap)
            button_y_i = y + (row_h - VOLUME_BUTTON_SIZE) * 0.5
            slider_x = button_x + (VOLUME_BUTTON_SIZE - VOLUME_SLIDER_PANEL_W) * 0.5
            slider_y = button_y_i - VOLUME_SLIDER_GAP - VOLUME_SLIDER_PANEL_H
            track_x = slider_x + (VOLUME_SLIDER_PANEL_W - VOLUME_SLIDER_TRACK_W) * 0.5
            track_y = slider_y + 14.0
            volume_controls[key] = {
                "key": key,
                "label": label,
                "button": {
                    "x": button_x,
                    "y": button_y_i,
                    "w": VOLUME_BUTTON_SIZE,
                    "h": VOLUME_BUTTON_SIZE,
                },
                "slider": {
                    "x": slider_x,
                    "y": slider_y,
                    "w": VOLUME_SLIDER_PANEL_W,
                    "h": VOLUME_SLIDER_PANEL_H,
                    "track_x": track_x,
                    "track_y": track_y,
                    "track_w": VOLUME_SLIDER_TRACK_W,
                    "track_h": VOLUME_SLIDER_TRACK_H,
                },
            }

        layout = {
            "bounds_x": x,
            "bounds_y": y,
            "bounds_w": total_w,
            "bounds_h": row_h,
            "line_x": line_x,
            "line_y": line_y,
            "line_w": line_w,
            "line_h": float(TIMELINE_TRACK_HEIGHT),
            "timer_x": x,
            "text_y": y + (row_h - 14.0) * 0.5 - 1.0,
            "chips": chips,
            "volume_controls": volume_controls,
        }
        self._timeline_layout_cache_key = cache_key
        self._timeline_layout_cache = layout
        return layout

    def _timeline_time_from_x(self, x: float, layout: dict) -> float:
        progress = (x - layout["line_x"]) / max(1.0, layout["line_w"])
        progress = max(0.0, min(1.0, progress))
        return self._timeline_start_ms + progress * self._timeline_total_ms

    def _timeline_anim_values(self) -> tuple[float, float, float]:
        anim_progress = min(1.0, self._timeline_anim_timer / TIMELINE_ENTER_DURATION)
        eased = _ease_out_cubic(anim_progress)
        return max(0.0, eased), (1.0 - eased) * 60.0, (1.0 - eased) * 18.0

    @staticmethod
    def _point_in_rect(x: float, y: float, rect: dict) -> bool:
        return rect["x"] <= x <= rect["x"] + rect["w"] and rect["y"] <= y <= rect["y"] + rect["h"]

    @staticmethod
    def _volume_key_from_hit(hit_kind: str | None, hit_value: object | None) -> str | None:
        if hit_kind == "volume_button":
            return str(hit_value)
        if hit_kind == "volume_slider" and isinstance(hit_value, tuple):
            return str(hit_value[0])
        return None

    def _volume_slider_value_from_position(self, x: float, y: float, slider: dict) -> float:
        del x
        progress = (slider["track_y"] + slider["track_h"] - y) / max(1.0, slider["track_h"])
        return self._clamp_unit(progress)

    def _timeline_visual_layout(self) -> tuple[dict, float]:
        layout = self._timeline_layout()
        alpha, x_off, y_off = self._timeline_anim_values()
        chips = []
        for chip in layout["chips"]:
            chips.append({**chip, "x": chip["x"] + x_off, "y": chip["y"] + y_off})

        volume_controls = {}
        for key, control in layout["volume_controls"].items():
            visibility = self._volume_slider_visibility[key]
            slider_shift = (1.0 - visibility) * 12.0
            button = {
                **control["button"],
                "x": control["button"]["x"] + x_off,
                "y": control["button"]["y"] + y_off,
            }
            slider = {
                **control["slider"],
                "x": control["slider"]["x"] + x_off,
                "y": control["slider"]["y"] + y_off + slider_shift,
                "track_x": control["slider"]["track_x"] + x_off,
                "track_y": control["slider"]["track_y"] + y_off + slider_shift,
            }
            volume_controls[key] = {
                "key": key,
                "label": control["label"],
                "button": button,
                "slider": slider,
                "visibility": visibility,
            }

        visual_layout = {
            "bounds_x": layout["bounds_x"] + x_off,
            "bounds_y": layout["bounds_y"] + y_off,
            "bounds_w": layout["bounds_w"],
            "bounds_h": layout["bounds_h"],
            "line_x": layout["line_x"] + x_off,
            "line_y": layout["line_y"] + y_off,
            "line_w": layout["line_w"],
            "line_h": layout["line_h"],
            "timer_x": layout["timer_x"] + x_off,
            "text_y": layout["text_y"] + y_off,
            "chips": chips,
            "volume_controls": volume_controls,
        }
        return visual_layout, alpha

    def _update_volume_slider_animations(self, dt: float) -> None:
        hit_kind, hit_value = self._timeline_hit_test(self._mouse_x, self._mouse_y)
        hovered_key = self._volume_key_from_hit(hit_kind, hit_value)
        for key in self._volume_slider_visibility:
            if self._volume_slider_dragging == key or hovered_key == key:
                self._refresh_volume_slider_timer(key)
            elif self._volume_slider_hide_timers[key] > 0.0:
                self._volume_slider_hide_timers[key] = max(0.0, self._volume_slider_hide_timers[key] - dt)

            target = 1.0 if (
                self._volume_slider_dragging == key
                or hovered_key == key
                or self._volume_slider_hide_timers[key] > 0.0
            ) else 0.0
            current = self._volume_slider_visibility[key]
            step = min(1.0, dt * VOLUME_SLIDER_ANIM_SPEED)
            self._volume_slider_visibility[key] = current + (target - current) * step

    def _timeline_hit_test(self, x: float, y: float) -> tuple[str | None, object | None]:
        if not self._gameplay_overlay_visible("draw_gameplay_timeline"):
            return None, None
        layout, _alpha = self._timeline_visual_layout()
        for key, control in layout["volume_controls"].items():
            if self._point_in_rect(x, y, control["button"]):
                return "volume_button", key
            slider_active = (
                control["visibility"] > 0.01
                or self._volume_slider_hide_timers[key] > 0.0
                or self._volume_slider_dragging == key
            )
            if slider_active and self._point_in_rect(x, y, control["slider"]):
                return "volume_slider", (key, self._volume_slider_value_from_position(x, y, control["slider"]))

        for chip in layout["chips"]:
            if self._point_in_rect(x, y, chip):
                return chip["kind"], chip["value"]

        track_x = layout["line_x"]
        track_y = layout["line_y"]
        track_w = layout["line_w"]
        track_h = layout["line_h"]
        progress = self._timeline_progress(self._current_time_ms)
        thumb_x = track_x + track_w * progress

        in_track = (
            track_x - TIMELINE_SCRUB_PAD <= x <= track_x + track_w + TIMELINE_SCRUB_PAD
            and track_y - TIMELINE_SCRUB_PAD <= y <= track_y + track_h + TIMELINE_SCRUB_PAD
        )
        in_thumb = (
            thumb_x - 12.0 <= x <= thumb_x + 12.0
            and track_y - 12.0 <= y <= track_y + track_h + 12.0
        )
        if in_track or in_thumb:
            return "track", self._timeline_time_from_x(x, layout)
        return None, None

    def _update_timeline_hover(self, x: float, y: float) -> None:
        if self._volume_slider_dragging is not None:
            layout, _alpha = self._timeline_visual_layout()
            control = layout["volume_controls"][self._volume_slider_dragging]
            value = self._volume_slider_value_from_position(x, y, control["slider"])
            self._set_volume_control_value(self._volume_slider_dragging, value)
            self._refresh_volume_slider_timer(self._volume_slider_dragging)
            self._timeline_hovered = False
            return

        if self._timeline_dragging:
            layout, _alpha = self._timeline_visual_layout()
            self._timeline_hovered = True
            self._timeline_hover_time_ms = self._timeline_time_from_x(x, layout)
            return

        hit_kind, hit_value = self._timeline_hit_test(x, y)
        hovered_volume_key = self._volume_key_from_hit(hit_kind, hit_value)
        if hovered_volume_key is not None:
            self._refresh_volume_slider_timer(hovered_volume_key)
        if hit_kind == "track" and hit_value is not None:
            self._timeline_hovered = True
            self._timeline_hover_time_ms = hit_value
        else:
            self._timeline_hovered = False

    def _draw_timeline(self, current_time_ms: float) -> None:
        if self._timeline_total_ms <= 1.0:
            return

        panels = self.app.panels
        text = self.app.text
        theme = self.app.menu_context().theme
        colors = theme.colors
        layout, alpha = self._timeline_visual_layout()

        line_x = layout["line_x"]
        line_y = layout["line_y"]
        hover_kind, hover_value = self._timeline_hit_test(self._mouse_x, self._mouse_y)
        hovered_volume_key = self._volume_key_from_hit(hover_kind, hover_value)
        controls_emphasis = 1.0 if (
            self._paused
            or self._timeline_hovered
            or self._timeline_dragging
            or hover_kind is not None
            or self._volume_controls_active()
        ) else 0.68

        progress = self._timeline_progress(current_time_ms)
        elapsed_ms = self._timeline_elapsed_ms(current_time_ms)
        fill_w = layout["line_w"] * progress
        timeline_thumb_x = line_x + fill_w

        for key, control in layout["volume_controls"].items():
            button = control["button"]
            slider = control["slider"]
            visibility = control["visibility"]
            value = self._volume_control_value(key)
            muted = self._volume_control_muted(key)
            button_hovered = hovered_volume_key == key
            slider_alpha = alpha * visibility

            if slider_alpha > 0.01:
                panels.draw(
                    slider["x"], slider["y"], slider["w"], slider["h"],
                    radius=slider["w"] * 0.5,
                    color=(colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.42 * slider_alpha),
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0,
                )
                panels.draw(
                    slider["track_x"], slider["track_y"], slider["track_w"], slider["track_h"],
                    radius=slider["track_w"] * 0.5,
                    color=(colors.surface_dim[0], colors.surface_dim[1], colors.surface_dim[2], 0.24 * slider_alpha),
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0,
                )

                slider_fill_h = slider["track_h"] * value
                fill_y = slider["track_y"] + slider["track_h"] - slider_fill_h
                if slider_fill_h > 1.0:
                    panels.draw(
                        slider["track_x"], fill_y, slider["track_w"], slider_fill_h,
                        radius=slider["track_w"] * 0.5,
                        color=(colors.slider_fill[0], colors.slider_fill[1], colors.slider_fill[2], 0.94 * slider_alpha),
                        border_color=(0.0, 0.0, 0.0, 0.0),
                        border_width=0.0,
                    )

                thumb_size = 12.0 if self._volume_slider_dragging == key else 10.0
                slider_thumb_x = slider["track_x"] - (thumb_size - slider["track_w"]) * 0.5
                slider_thumb_y = fill_y - thumb_size * 0.5
                panels.draw(
                    slider_thumb_x, slider_thumb_y, thumb_size, thumb_size,
                    radius=thumb_size * 0.5,
                    color=(colors.slider_thumb[0], colors.slider_thumb[1], colors.slider_thumb[2], 0.96 * slider_alpha),
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0,
                )

            label_size = 12
            label_w, _ = text.measure(control["label"], label_size)
            if muted:
                button_bg = (colors.tertiary_container[0], colors.tertiary_container[1], colors.tertiary_container[2], 0.84 * alpha)
                button_fg = colors.text_primary
            elif button_hovered or visibility > 0.05:
                button_bg = (colors.surface_variant[0], colors.surface_variant[1], colors.surface_variant[2], 0.70 * alpha)
                button_fg = colors.text_primary
            else:
                button_bg = (colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.34 * alpha)
                button_fg = colors.text_secondary

            panels.draw(
                button["x"], button["y"], button["w"], button["h"],
                radius=button["w"] * 0.5,
                color=button_bg,
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            text.draw(
                control["label"],
                button["x"] + (button["w"] - label_w) * 0.5,
                button["y"] + 4.0,
                label_size,
                color=button_fg,
                alpha=0.98 * alpha * controls_emphasis,
            )

        for chip in layout["chips"]:
            selected = (
                chip["kind"] == "pause" and self._paused
            ) or (
                chip["kind"] == "speed" and abs(float(chip["value"]) - self._playback_speed) < 1e-4
            )
            hovered = hover_kind == chip["kind"] and hover_value == chip["value"]
            if selected:
                bg = (colors.primary_container[0], colors.primary_container[1], colors.primary_container[2], 0.78 * alpha)
                fg = colors.text_primary
            elif hovered:
                bg = (colors.surface_variant[0], colors.surface_variant[1], colors.surface_variant[2], 0.70 * alpha)
                fg = colors.text_primary
            else:
                bg = (colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.30 * alpha)
                fg = colors.text_secondary
            panels.draw(chip["x"], chip["y"], chip["w"], chip["h"], radius=11.0, color=bg, border_color=(0.0, 0.0, 0.0, 0.0), border_width=0.0)
            text.draw(
                chip["label"],
                chip["x"] + (chip["w"] - chip["text_w"]) / 2.0,
                chip["y"] + 1,
                13,
                color=fg,
                alpha=0.98 * alpha * controls_emphasis,
            )

        total_label = self._format_timeline_time(self._timeline_total_ms)
        timer_label = f"{self._format_timeline_time(elapsed_ms)}/{total_label}"
        text.draw(timer_label, layout["timer_x"], layout["text_y"] - 4.0, 14,
                  color=colors.text_primary, alpha=0.84 * alpha)

        panels.draw(
            line_x, line_y, layout["line_w"], layout["line_h"],
            radius=layout["line_h"] / 2.0,
            color=(colors.surface_dim[0], colors.surface_dim[1], colors.surface_dim[2], (0.18 + 0.10 * controls_emphasis) * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )

        if fill_w > 1.0:
            panels.draw(
                line_x, line_y, fill_w, layout["line_h"],
                radius=layout["line_h"] / 2.0,
                color=(colors.slider_fill[0], colors.slider_fill[1], colors.slider_fill[2], 0.96 * alpha),
                border_color=(0, 0, 0, 0),
                border_width=0.0,
            )

        thumb_size = 20.0 if self._timeline_dragging else 16.0 if self._timeline_hovered else 12.0
        panels.draw(
            timeline_thumb_x - (thumb_size - 4.0) * 0.5, line_y - (2.0 if self._timeline_hovered else - 4.0 if self._timeline_dragging else 0.0), thumb_size - 4.0, thumb_size - 4.0,
            radius=(thumb_size - 4.0) * 0.5,
            color=(colors.slider_thumb[0], colors.slider_thumb[1], colors.slider_thumb[2], 0.96 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )

        if self._timeline_hovered or self._timeline_dragging:
            hover_time_ms = self._timeline_hover_time_ms
            hover_text = self._format_timeline_time(hover_time_ms - self._timeline_start_ms)
            hover_progress = self._timeline_progress(hover_time_ms)
            tooltip_w, _ = text.measure(hover_text, 13)
            tooltip_w += 18.0
            tooltip_h = 24.0
            tip_x = line_x + layout["line_w"] * hover_progress - tooltip_w / 2.0
            tip_x = max(layout["bounds_x"] + 10.0, min(layout["bounds_x"] + layout["bounds_w"] - tooltip_w - 10.0, tip_x))
            tip_y = line_y - 24.0
            panels.draw(
                tip_x, tip_y, tooltip_w, tooltip_h,
                radius=10.0,
                color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], 0.82 * alpha),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            text.draw(hover_text, tip_x + 9.0, tip_y + 5.0, 13,
                      color=colors.text_primary, alpha=0.98 * alpha)

    # ----------------------------------------------------------- render

    def on_render(self, time: float, frametime: float):
        profiler.begin_frame("gameplay")
        self.ctx.clear(*self.app.clear_color)
        try:
            if not self._loading_ready:
                with profiler.timer("gameplay.apply_prepare_result"):
                    self._apply_prepare_result()
                if not self._loading_ready:
                    self.ctx.disable(moderngl.DEPTH_TEST)
                    theme = self.app.menu_context().theme
                    self.app.text.begin()
                    self.app.text.draw("Loading gameplay...", 36, 40, 24, color=theme.colors.text_primary)
                    self.app.text.draw(
                        self._prepare_error or "Preparing beatmap, cursor path, and score timeline",
                        36, 72, 14, color=theme.colors.text_secondary
                    )
                    self.app.text.end()
                    return

            dt = min(frametime, 0.05)
            self._timeline_anim_timer += dt

            if not self._fade_in_done:
                if self._first_render:
                    self._first_render = False
                else:
                    self._fade_timer += dt
                    self._cursor_intro_timer += dt
                if self._fade_timer >= FADE_IN_DURATION:
                    self._fade_in_done = True

            if self._exiting:
                self._exit_timer += dt
                if self._exit_timer >= GAMEPLAY_EXIT_DURATION:
                    if self._audio is not None:
                        self._audio.cleanup()
                        self._audio = None
                    from scenes.song_select import SongSelectScene
                    self.app.switch_scene(SongSelectScene(self.app))
                    return

            self.ctx.enable(moderngl.BLEND | moderngl.DEPTH_TEST)
            self.ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ctx.depth_func = '<'

            if self._timeline_dragging:
                current_time_ms = self._manual_time_ms
            elif self._audio is not None:
                if not self._paused:
                    self._audio.tick(dt)
                current_time_ms = self._audio.position_ms
            else:
                if not self._paused:
                    self._manual_time_ms += dt * 1000.0 * self._playback_speed
                current_time_ms = self._manual_time_ms
            current_time_ms = self._clamp_timeline_time(current_time_ms)
            if self._audio is None:
                self._manual_time_ms = current_time_ms
            self._current_time_ms = current_time_ms
            self._apply_gameplay_source_participant(
                self._gameplay_source_participant(),
                current_time_ms=current_time_ms,
            )
            self._update_volume_slider_animations(dt)
            with profiler.timer("gameplay.sync_active_instances"):
                self._sync_active_render_window(self._circle_render_window, current_time_ms)
                self._sync_active_render_window(self._spinner_render_window, current_time_ms)
                self._sync_active_render_window(self._slider_render_window, current_time_ms)

            with profiler.timer("gameplay.skin_pre_render"):
                self._skin.on_pre_render(self.ctx, current_time_ms, frametime)

            background_image_enabled = self._gameplay_background_image_enabled()
            if background_image_enabled:
                self.app.backgrounds.configure(
                    blur_radius=0,
                    dim_factor=1.0 - self._clamp_unit(getattr(self.app.settings, "gameplay_background_dim", 0.65)),
                    motion_enabled=False,
                )
                self.app.backgrounds.load(self._gameplay_background_path())
                self.app.backgrounds.draw(dt)

            self._skin.set_atmosphere_enabled(
                bool(getattr(self.app.settings, "gameplay_background_bloom", True))
                and not background_image_enabled
            )
            self._skin.set_bloom_intensity(getattr(self.app.settings, "gameplay_circle_bloom", 0.50))

            if not self._paused and not self._timeline_dragging:
                self._trigger_hitsounds(current_time_ms)
                self._prev_time_ms = current_time_ms
                with profiler.timer("gameplay.sync_score_state"):
                    self._sync_score_state_for_time(current_time_ms)

            slider_instances = 0 if self._slider_render_window is None else self._slider_render_window.active_count
            if self.slider_vao is not None and slider_instances > 0:
                with profiler.timer("gameplay.render_sliders"):
                    self.ctx.depth_mask = False
                    self.path_tex.use(location=0)
                    self.slider_prog["u_path_tex"].value = 0
                    self.slider_prog["u_path_tex_width"].value = self.path_tex_width
                    self.slider_prog["current_time"].value = current_time_ms
                    self.slider_vao.render(moderngl.TRIANGLES, instances=slider_instances)
                    self.ctx.depth_mask = True

            circle_instances = 0 if self._circle_render_window is None else self._circle_render_window.active_count
            if self.circle_vao is not None and circle_instances > 0:
                with profiler.timer("gameplay.render_circles"):
                    self.ctx.depth_mask = False
                    self.circle_prog["current_time"].value = current_time_ms
                    self.circle_vao.render(moderngl.TRIANGLES, instances=circle_instances)
                    self.ctx.depth_mask = True

            spinner_instances = 0 if self._spinner_render_window is None else self._spinner_render_window.active_count
            if self.spinner_vao is not None and spinner_instances > 0:
                with profiler.timer("gameplay.render_spinners"):
                    self.ctx.depth_mask = False
                    self.spinner_prog["current_time"].value = current_time_ms
                    self.spinner_vao.render(moderngl.TRIANGLES, instances=spinner_instances)
                    self.ctx.depth_mask = True

            self.ctx.disable(moderngl.DEPTH_TEST)

            if self.ball_vao is not None:
                with profiler.timer("gameplay.compute_slider_balls"):
                    ball_data = self._compute_slider_balls(current_time_ms)
                n_balls = len(ball_data)
                if n_balls > 0:
                    self._ball_instance_buf.write(memoryview(ball_data[:n_balls]).cast("B"))
                    self.ball_vao.render(moderngl.TRIANGLES, instances=n_balls)

            if self.approach_vao is not None and circle_instances > 0:
                self.approach_prog["current_time"].value = current_time_ms
                self.approach_vao.render(moderngl.TRIANGLES, instances=circle_instances)

            self.ctx.disable(moderngl.DEPTH_TEST)
            for participant in self._participants:
                alpha = self._participant_cursor_alpha(participant, current_time_ms)
                if alpha <= 0.01:
                    continue
                with profiler.timer("gameplay.build_cursor_head"):
                    head_buf, n_head = self._build_cursor_head(current_time_ms, participant, alpha)

                if self.cursor_vao is not None and n_head > 0:
                    if self._gameplay_trail_enabled() and self.trail_vao is not None:
                        with profiler.timer("gameplay.build_trail_ribbon"):
                            trail_buf, n_trail_ctrl, n_trail_samples = self._build_trail_ribbon(current_time_ms, participant, alpha)
                        if n_trail_ctrl >= 2 and n_trail_samples >= 2:
                            with profiler.timer("gameplay.render_trail"):
                                self._skin.begin_trail_pass(self.ctx)
                                if self._trail_point_tex is not None:
                                    self._trail_point_tex.write(
                                        memoryview(trail_buf[:n_trail_ctrl]).cast("B"),
                                        viewport=(0, 0, n_trail_ctrl, 1),
                                    )
                                    self._trail_point_tex.use(location=0)
                                if self.trail_prog is not None and "u_color" in self.trail_prog:
                                    self.trail_prog["u_color"].value = participant.color
                                    if "u_alpha_scale" in self.trail_prog:
                                        self.trail_prog["u_alpha_scale"].value = alpha
                                    if "u_current_time_ms" in self.trail_prog:
                                        self.trail_prog["u_current_time_ms"].value = current_time_ms
                                    if "u_point_count" in self.trail_prog:
                                        self.trail_prog["u_point_count"].value = n_trail_ctrl
                                    if "u_curve_samples" in self.trail_prog:
                                        self.trail_prog["u_curve_samples"].value = n_trail_samples
                                self.trail_vao.render(moderngl.TRIANGLE_STRIP, vertices=n_trail_samples * 2)
                                self._skin.end_trail_pass(self.ctx)

                    self._cursor_instance_buf.write(memoryview(head_buf[:n_head]).cast("B"))
                    self.cursor_vao.render(moderngl.TRIANGLES, instances=n_head)
            self.ctx.enable(moderngl.DEPTH_TEST)

            with profiler.timer("gameplay.render_atmosphere"):
                self._skin.render_atmosphere(self.ctx, current_time_ms)

            self.ctx.enable(moderngl.DEPTH_TEST)
            with profiler.timer("gameplay.skin_post_render"):
                self._skin.on_post_render(self.ctx, current_time_ms, frametime)

            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

            if not self._paused and not self._timeline_dragging:
                with profiler.timer("gameplay.update_hud"):
                    self._update_hud(current_time_ms, dt)

            # HUD draws (after bloom for crisp UI) - ensure we render to screen
            w, h = self.wnd.buffer_size
            self.ctx.screen.use()
            self.ctx.viewport = (0, 0, w, h)
            self.app.panels.begin_batch()
            self.app.text.begin()
            if self._gameplay_overlay_visible("draw_gameplay_hp"):
                self._draw_healthbar(current_time_ms)
            if self._gameplay_overlay_visible("draw_gameplay_acc_pp"):
                self._draw_stats_cluster(current_time_ms)
            if self._gameplay_overlay_visible("draw_gameplay_combo"):
                self._draw_combo()
            if self._gameplay_overlay_visible("draw_gameplay_keys"):
                self._draw_key_overlay(current_time_ms)
            if self._gameplay_overlay_visible("draw_gameplay_leaderboard"):
                with profiler.timer("gameplay.draw_leaderboard"):
                    self._draw_leaderboard(current_time_ms, dt)
            else:
                self._leaderboard_rects = []
                self._leaderboard_hover_id = None
            if self._gameplay_overlay_visible("draw_gameplay_timeline"):
                with profiler.timer("gameplay.draw_timeline"):
                    self._draw_timeline(current_time_ms)
            self.app.panels.end_batch()
            self.app.text.end()

            if self._pause_menu_open and not self._exiting:
                self._draw_pause_overlay()
                self.app.panels.begin_batch()
                self.app.text.begin()
                if self._gameplay_overlay_visible("draw_gameplay_timeline"):
                    with profiler.timer("gameplay.draw_timeline"):
                        self._draw_timeline(current_time_ms)
                self.app.panels.end_batch()
                self.app.text.end()

            if not self._fade_in_done:
                fade_alpha = 1.0 - min(1.0, self._fade_timer / FADE_IN_DURATION)
                self._draw_fullscreen_overlay((0.0, 0.0, 0.0, fade_alpha))

            if self._exiting:
                exit_alpha = min(1.0, self._exit_timer / GAMEPLAY_EXIT_DURATION)
                self._draw_fullscreen_overlay((0.0, 0.0, 0.0, exit_alpha))
        finally:
            profiler.end_frame()

    # ----------------------------------------------------------- HUD

    def _score_snapshot_for_participant(
        self,
        participant: _GameplayParticipant,
        current_time_ms: float,
        *,
        from_seek: bool = False,
    ) -> tuple[object | None, int]:
        timeline = participant.score_timeline
        if not timeline.points:
            return None, -1

        idx = participant.score_point_idx
        if from_seek or idx < 0:
            idx = bisect.bisect_right(timeline.times, current_time_ms) - 1
        else:
            current_idx_time = timeline.times[idx] if 0 <= idx < len(timeline.times) else -1e9
            if current_time_ms < current_idx_time:
                idx = bisect.bisect_right(timeline.times, current_time_ms) - 1
            else:
                limit = len(timeline.times) - 1
                while idx < limit and timeline.times[idx + 1] <= current_time_ms:
                    idx += 1

        if idx < 0:
            return None, -1
        return timeline.points[idx], idx

    def _sync_score_state_for_time(self, current_time_ms: float, *, from_seek: bool = False) -> None:
        snapshot: list[tuple[_GameplayParticipant, object | None, int]] = []
        for participant in self._participants:
            point, idx = self._score_snapshot_for_participant(
                participant,
                current_time_ms,
                from_seek=from_seek,
            )
            participant.score_point_idx = idx
            participant.last_point = point
            snapshot.append((participant, point, idx))

        survivor_override_id: str | None = None
        if not self._include_danser:
            replay_snapshot = [
                (participant, point, idx)
                for participant, point, idx in snapshot
                if participant.source_type == "replay" and point is not None and idx >= 0
            ]
            if replay_snapshot and all(bool(point.eliminated) for _, point, _ in replay_snapshot):
                best_key = None
                for participant, point, idx in replay_snapshot:
                    elim_idx = idx
                    timeline_points = participant.score_timeline.points
                    while elim_idx > 0 and timeline_points[elim_idx - 1].eliminated:
                        elim_idx -= 1
                    elim_time = float(timeline_points[elim_idx].time_ms)
                    freeze_idx = max(0, elim_idx - 1)
                    freeze_point = timeline_points[freeze_idx]
                    candidate_key = (
                        elim_time,
                        float(freeze_point.score),
                        float(freeze_point.accuracy),
                        participant.name.lower(),
                    )
                    if best_key is None or candidate_key > best_key:
                        best_key = candidate_key
                        survivor_override_id = participant.leaderboard_id

        for participant, point, idx in snapshot:
            if point is None:
                participant.eliminated = False
                participant.elimination_started_ms = -1.0
                participant.elimination_score = 0
                participant.current_score = 0
                participant.current_accuracy = 1.0
                participant.current_combo = 0
                participant.current_pp = 0.0
                participant.current_hp = 1.0
                continue

            ignore_elimination = participant.leaderboard_id == survivor_override_id
            if point.eliminated and not ignore_elimination:
                elim_idx = idx
                timeline_points = participant.score_timeline.points
                while elim_idx > 0 and timeline_points[elim_idx - 1].eliminated:
                    elim_idx -= 1
                freeze_idx = max(0, elim_idx - 1)
                freeze_point = timeline_points[freeze_idx]
                participant.eliminated = True
                participant.elimination_started_ms = timeline_points[elim_idx].time_ms
                participant.elimination_score = freeze_point.score
                participant.current_score = freeze_point.score
                participant.current_accuracy = freeze_point.accuracy
                participant.current_combo = freeze_point.combo
                participant.current_pp = freeze_point.pp
                participant.current_hp = self._hp_for_point(
                    participant.score_timeline,
                    freeze_point,
                    timeline_points[elim_idx].time_ms,
                )
            else:
                participant.eliminated = False
                participant.elimination_started_ms = -1.0
                participant.elimination_score = 0
                participant.current_score = point.score
                participant.current_accuracy = point.accuracy
                participant.current_combo = point.combo
                participant.current_pp = point.pp
                participant.current_hp = self._hp_for_point(participant.score_timeline, point, current_time_ms)

        hud_source = self._display_score_participant()
        point = None
        idx = -1
        if hud_source is not None:
            point = hud_source.last_point
            idx = hud_source.score_point_idx
            # In multi-replay, the default HUD source can be the synthetic
            # perfect play participant, which is not always part of
            # self._participants and therefore is not updated in the loop above.
            if hud_source.leaderboard_id not in self._participants_by_id:
                point, idx = self._score_snapshot_for_participant(
                    hud_source,
                    current_time_ms,
                    from_seek=from_seek,
                )
                hud_source.score_point_idx = idx
                hud_source.last_point = point
        previous_idx = self._score_point_idx
        previous_pp = self._pp_target
        previous_last_hit_time_ms = self._last_hit_time_ms
        previous_combo = self._combo

        if point is None:
            self._combo = 0
            self._max_combo = 0
            self._pp_target = 0.0
            self._accuracy = 1.0
            self._last_hit_time_ms = -1e9
            self._hp = 1.0
            if from_seek or previous_idx >= 0:
                self._pp_display = 0.0
        else:
            rewound = idx < previous_idx
            advanced = idx > previous_idx and not from_seek
            hit_advanced = point.last_hit_time_ms > previous_last_hit_time_ms + 1e-6
            combo_gained = point.combo > previous_combo
            self._combo = point.combo
            self._max_combo = point.max_combo
            self._pp_target = point.pp
            self._accuracy = point.accuracy
            self._last_hit_time_ms = point.last_hit_time_ms

            if from_seek or rewound:
                self._pp_display = point.pp

            if advanced:
                if abs(point.pp - previous_pp) >= 0.01 or hit_advanced or combo_gained:
                    self._pp_pulse_timer = 0.18
                if hit_advanced or combo_gained:
                    self._combo_anim_timer = 0.15

            self._hp = hud_source.score_timeline.hp_at(current_time_ms, point=point)

        self._score_point_idx = idx

    def _update_hud(self, current_time_ms: float, dt: float) -> None:
        """Tick HP drain, combo animation, and replay key tracking."""
        with profiler.timer("gameplay.hud_anim"):
            self._combo_anim_timer = max(0.0, self._combo_anim_timer - dt)
            self._pp_pulse_timer = max(0.0, self._pp_pulse_timer - dt)
            self._pp_display += (self._pp_target - self._pp_display) * min(1.0, dt * 10.0)
            self._combo_display += (float(self._combo) - self._combo_display) * min(1.0, dt * 12.0)
            self._accuracy_display += (self._accuracy - self._accuracy_display) * min(1.0, dt * 10.0)
            self._hp_display += (self._hp - self._hp_display) * min(1.0, dt * 10.0)

        with profiler.timer("gameplay.hud_key_overlay"):
            key_source = self._display_key_participant()
            source_sig = self._key_overlay_source_signature(key_source)
            if (
                source_sig != self._key_overlay_source_sig
                or current_time_ms + 0.5 < self._key_overlay_time_ms
            ):
                self._rebuild_key_overlay_for_source(current_time_ms, key_source)
            elif key_source is not None and key_source.replay is not None:
                self._advance_replay_key_overlay(current_time_ms, key_source.replay)
            elif key_source is not None and key_source.cpath is not None:
                self._advance_auto_key_overlay(current_time_ms, key_source.auto_key_schedule)

        with profiler.timer("gameplay.hud_prune_keys"):
            self._prune_key_bars(current_time_ms)

    def _draw_healthbar(self, current_time_ms: float) -> None:
        w, _h = self.wnd.buffer_size
        panels = self.app.panels
        text = self.app.text
        theme = self.app.menu_context().theme
        colors = theme.colors

        bar_w = min(348.0, w * 0.23)
        bar_h = 10.0
        bar_x = 22.0
        bar_y = 18.0

        since_hit = current_time_ms - self._last_hit_time_ms
        pulse = max(0.0, 1.0 - since_hit / 140.0)
        hp = max(0.0, self._hp_display)
        source_label = self._display_participant_label()
        if hp > 0.60:
            fill_color = colors.success
        elif hp > 0.30:
            fill_color = colors.warning
        else:
            fill_color = colors.error

        tag_w = 58.0
        tag_h = 22.0
        track_h = 9.0
        tag_y = bar_y + 1.0
        track_x = bar_x + 42.0
        track_y = bar_y + 8.0
        under_y = track_y + 10.0

        # Connected chip + bar composition with one secondary rail for rhythm.
        panels.draw(
            bar_x,
            tag_y,
            tag_w,
            tag_h,
            radius=tag_h * 0.5,
            color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], 0.68),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        panels.draw(
            bar_x + 8.0,
            tag_y + 6.0,
            4.0,
            tag_h - 12.0,
            radius=2.0,
            color=(fill_color[0], fill_color[1], fill_color[2], 0.96),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        panels.draw(
            track_x - 12.0,
            track_y + 1.0,
            18.0,
            track_h - 2.0,
            radius=(track_h - 2.0) * 0.5,
            color=(colors.surface_container_low[0], colors.surface_container_low[1], colors.surface_container_low[2], 0.54),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        panels.draw(
            track_x,
            track_y,
            bar_w,
            track_h,
            radius=track_h * 0.5,
            color=(colors.surface_dim[0], colors.surface_dim[1], colors.surface_dim[2], 0.28),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        panels.draw(
            track_x + 26.0,
            under_y,
            bar_w * 0.52,
            4.0,
            radius=2.0,
            color=(colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.32),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )

        fill_w = max(track_h * 1.6, bar_w * hp)
        panels.draw(
            track_x,
            track_y,
            fill_w,
            track_h,
            radius=track_h * 0.5,
            color=(fill_color[0], fill_color[1], fill_color[2], 0.92 + 0.06 * pulse),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        panels.draw(
            track_x + max(0.0, fill_w - 22.0),
            track_y,
            min(22.0, fill_w),
            track_h,
            radius=track_h * 0.5,
            color=(1.0, 1.0, 1.0, 0.08 + 0.08 * pulse),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )

        text.draw("HP", bar_x + 19.0, tag_y + 3.0, 11, color=colors.text_primary, alpha=0.90)

        pct = f"{int(hp * 100)}%"
        pct_w, _ = text.measure(pct, 12)
        text.draw(
            pct,
            track_x + bar_w - pct_w + 40.0,
            tag_y + 3.0,
            12,
            color=colors.text_muted,
            alpha=0.76,
        )
        if source_label:
            text.draw(
                source_label[:18],
                bar_x + 6.0,
                tag_y + 26.0,
                11,
                color=colors.text_secondary,
                alpha=0.62,
            )

    def _draw_stats_cluster(self, current_time_ms: float) -> None:
        panels = self.app.panels
        text = self.app.text
        theme = self.app.menu_context().theme
        colors = theme.colors
        pulse = max(0.0, min(1.0, self._pp_pulse_timer / 0.18))
        pp_rect, acc_rect = self._hud_stat_tiles()
        pp_x, pp_y, pp_w, pp_h = pp_rect

        panels.draw(
            pp_x,
            pp_y,
            pp_w,
            pp_h,
            radius=15.0,
            color=(colors.surface_container_low[0], colors.surface_container_low[1], colors.surface_container_low[2], 0.76),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        panels.draw(
            pp_x + 10.0,
            pp_y + 9.0,
            4.0,
            pp_h - 18.0,
            radius=2.0,
            color=(colors.focus_ring[0], colors.focus_ring[1], colors.focus_ring[2], 0.92),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )

        inner_x = pp_x + 22.0
        inner_y = pp_y + 10.0
        source_label = self._display_participant_label()
        value_text = f"{self._pp_display:.2f}"
        value_size = 25 + int(round(1.5 * pulse))
        text.draw("PP", inner_x, inner_y, 11, color=colors.text_muted, alpha=0.82)
        text.draw(value_text, inner_x, inner_y + 12.0 - pulse * 0.5, value_size, color=colors.text_primary, alpha=0.96)

        if source_label:
            label_w, _ = text.measure(source_label, 11)
            text.draw(
                source_label,
                pp_x + pp_w - label_w - 12.0,
                pp_y + 8.0,
                11,
                color=colors.focus_ring,
                alpha=0.70,
            )
        if acc_rect is None:
            return

        acc_x, acc_y, acc_w, acc_h = acc_rect
        panels.draw(
            acc_x,
            acc_y,
            acc_w,
            acc_h,
            radius=13.0,
            color=(colors.surface_container_low[0], colors.surface_container_low[1], colors.surface_container_low[2], 0.72),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        panels.draw(
            acc_x + 9.0,
            acc_y + 8.0,
            3.0,
            acc_h - 16.0,
            radius=1.5,
            color=(colors.primary_container[0], colors.primary_container[1], colors.primary_container[2], 0.90),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        acc_value = f"{self._accuracy_display * 100.0:.2f}%"
        text.draw("ACC", acc_x + 18.0, acc_y + 8.0, 11, color=colors.text_muted, alpha=0.80)
        text.draw(acc_value, acc_x + 18.0, acc_y + 19.0, 21, color=colors.text_primary, alpha=0.94)

    def _draw_combo(self) -> None:
        w, h = self.wnd.buffer_size
        text = self.app.text

        combo_value = max(0, int(round(self._combo_display)))
        if combo_value < 1:
            return

        scale = 1.0 + min(0.3, self._combo_anim_timer * 2.0)
        combo_str = str(combo_value)
        size = min(48, int(32 * scale))

        tx = 20
        ty = h - 60

        text.draw(combo_str, tx, ty, size, color=(1.0, 1.0, 1.0), alpha=0.9)
        nw, _ = text.measure(combo_str, size)
        text.draw("x", tx + nw + 2, ty + (size - 16), 16,
                  color=(0.8, 0.8, 0.8), alpha=0.7)

    def _draw_accuracy_meter(self) -> None:
        return

    def _hud_stat_tiles(self) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float] | None]:
        w, _ = self.wnd.buffer_size
        pp_w = 156.0
        pp_h = 60.0
        x = w - pp_w - 18.0
        y = 18.0
        pp_rect = (x, y, pp_w, pp_h)
        if self._multi_replay and self._leaderboard_focus_participant() is None:
            return pp_rect, None
        acc_w = 118.0
        acc_h = 52.0
        acc_rect = (x - acc_w - 10.0, y + 8.0, acc_w, acc_h)
        return pp_rect, acc_rect

    def _hud_stats_cluster_rect(self) -> tuple[float, float, float, float]:
        pp_rect, acc_rect = self._hud_stat_tiles()
        x, y, w, h = pp_rect
        if acc_rect is None:
            return pp_rect
        ax, ay, aw, ah = acc_rect
        left = min(x, ax)
        top = min(y, ay)
        right = max(x + w, ax + aw)
        bottom = max(y + h, ay + ah)
        return left, top, right - left, bottom - top

    def _pp_counter_rect(self) -> tuple[float, float, float, float]:
        return self._hud_stats_cluster_rect()

    def _accuracy_meter_rect(self) -> tuple[float, float, float, float]:
        return self._hud_stats_cluster_rect()

    def _draw_key_overlay(self, current_time_ms: float) -> None:
        w, h = self.wnd.buffer_size
        panels = self.app.panels
        text = self.app.text

        theme = self.app.menu_context().theme
        colors = theme.colors
        box_w, box_h = 64.0, 36.0
        gap = 8.0
        rx = w - box_w - 14.0
        ry = h // 2 - box_h - gap * 0.5

        for i, (label, count, pressed) in enumerate([
            ("K1", self._k1_count, self._k1_pressed),
            ("K2", self._k2_count, self._k2_pressed),
        ]):
            y = ry + i * (box_h + gap)
            accent = (0.73, 0.52, 1.0, 0.96) if i == 0 else (0.52, 0.74, 1.0, 0.96)
            bg = (
                colors.surface_container[0],
                colors.surface_container[1],
                colors.surface_container[2],
                0.78 if pressed else 0.54,
            )
            panels.draw(rx, y, box_w, box_h, radius=11.0,
                        color=bg, border_color=(0.0, 0.0, 0.0, 0.0), border_width=0.0)
            panels.draw(rx + 6.0, y + 7.0, 3.0, box_h - 14.0, radius=1.5,
                        color=accent if pressed else (accent[0], accent[1], accent[2], 0.42),
                        border_color=(0.0, 0.0, 0.0, 0.0), border_width=0.0)
            text.draw(label, rx + 15.0, y + 4.0, 12,
                      color=colors.text_primary, alpha=0.92)
            count_text = str(count)
            text.draw(count_text, rx + 15.0, y + 18.0, 12,
                      color=colors.text_secondary if not pressed else colors.text_primary, alpha=0.84)

        fly_speed = 0.25  # px/ms - used for both growth and flying
        spawn_x = rx - 8.0
        fade_width = 800.0 * fly_speed  # 200px - gradient mask "eats" bars from left

        # Growing bars - bar grows left from spawn, gradient mask fades left end
        min_hold_ms = 1.2  # filter tiny blips but keep visible held bars
        for key, press_start in [(1, self._k1_press_start), (2, self._k2_press_start)]:
            pressed = self._k1_pressed if key == 1 else self._k2_pressed
            if not pressed or press_start < 0:
                continue
            hold_time = current_time_ms - press_start
            if hold_time < min_hold_ms:
                continue
            bar_w_px = max(6.0, hold_time * fly_speed)
            bar_y_center = ry + (0 if key == 1 else box_h + gap) + box_h / 2
            bx = spawn_x - bar_w_px
            if bx + bar_w_px < 0:
                continue
            col = (0.7, 0.5, 1.0, 0.9) if key == 1 else (0.5, 0.7, 1.0, 0.9)
            panels.draw_gradient_bar(bx, bar_y_center - 4, bar_w_px, 8,
                                     spawn_x, fade_width, col)

        # Flying bars - same gradient mask, bar flies left and gets "eaten"
        for key, start, dur in self._key_bars:
            age = current_time_ms - (start + dur)
            if age < 0:
                continue
            fly_x = age * fly_speed
            bar_w_px = max(6.0, dur * fly_speed)
            bar_y_center = ry + (0 if key == 1 else box_h + gap) + box_h / 2
            bx = spawn_x - fly_x - bar_w_px
            if bx + bar_w_px < spawn_x - fade_width:
                continue  # fully eaten by mask
            col = (0.7, 0.5, 1.0, 0.9) if key == 1 else (0.5, 0.7, 1.0, 0.9)
            panels.draw_gradient_bar(bx, bar_y_center - 4, bar_w_px, 8,
                                     spawn_x, fade_width, col)

    def _draw_leaderboard(self, current_time_ms: float, dt: float) -> None:
        if not self._participants:
            self._leaderboard_rects = []
            self._leaderboard_hover_id = None
            return
        panels = self.app.panels
        text = self.app.text
        _w, h = self.wnd.buffer_size
        row_h = 50.0
        row_gap = 8.0
        card_w = 258.0
        x = 34.0
        ranked = self._ordered_leaderboard_participants(current_time_ms)[:10]
        self._leaderboard_rects = []
        visible_block_h = len(ranked) * row_h + max(0, len(ranked) - 1) * row_gap
        y = max(72.0, (h - visible_block_h) * 0.5)
        rows: list[tuple[int, _GameplayParticipant, float, float]] = []
        hovered_id: str | None = None
        for idx, participant in enumerate(ranked):
            target_y = y + idx * (row_h + row_gap)
            participant.leaderboard_y += (target_y - participant.leaderboard_y) * min(1.0, dt * 10.0)
            row_y = participant.leaderboard_y if participant.leaderboard_y else target_y
            rows.append((idx, participant, x, row_y))
            if x <= self._mouse_x <= x + card_w and row_y <= self._mouse_y <= row_y + row_h:
                hovered_id = participant.leaderboard_id
        self._leaderboard_hover_id = hovered_id
        focus_participant = self._leaderboard_focus_participant()
        focus_id = None if focus_participant is None else focus_participant.leaderboard_id
        focus_presence_target = 1.0 if focus_id is not None else 0.0
        self._leaderboard_focus_presence += (
            focus_presence_target - self._leaderboard_focus_presence
        ) * min(1.0, dt * 10.0)

        for idx, participant, card_x, row_y in rows:
            alpha = self._participant_cursor_alpha(participant, current_time_ms)
            if participant.eliminated and alpha <= 0.01:
                continue
            focus_target = 1.0 if focus_id == participant.leaderboard_id else 0.0
            participant.leaderboard_focus_mix += (
                focus_target - participant.leaderboard_focus_mix
            ) * min(1.0, dt * 12.0)
            focus_mix = participant.leaderboard_focus_mix
            row_alpha = alpha * (
                1.0 - 0.62 * self._leaderboard_focus_presence * (1.0 - focus_mix)
            )
            color = participant.color
            row_x = card_x - 4.0 * focus_mix
            row_w = card_w + 8.0 * focus_mix
            score_value = participant.elimination_score if participant.eliminated else participant.current_score
            score_text = f"{score_value:08d}"
            combo_text = f"{participant.current_combo}x"
            acc_text = f"{participant.current_accuracy * 100.0:.2f}%"
            if focus_mix > 0.01:
                panels.draw(
                    row_x - 3.0, row_y - 2.0, row_w + 6.0, row_h + 4.0,
                    radius=14.0,
                    color=(color[0], color[1], color[2], 0.10 * focus_mix * row_alpha),
                    border_color=(color[0], color[1], color[2], 0.42 * focus_mix * row_alpha),
                    border_width=1.0,
                )
            panels.draw(
                row_x, row_y, row_w, row_h,
                radius=12.0,
                color=(0.05, 0.07, 0.13, (0.28 + 0.20 * focus_mix) * row_alpha),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            panels.draw(
                row_x + 8.0, row_y + 8.0, 3.0, row_h - 16.0,
                radius=1.5,
                color=(color[0], color[1], color[2], (0.52 + 0.36 * focus_mix) * row_alpha),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            name_text = participant.name[:20]
            score_size = 17
            name_size = 14
            stat_size = 12
            sw, _ = text.measure(score_text, score_size)
            rank_text = f"{idx + 1:02d}"
            text.draw(rank_text, row_x + 18.0, row_y + 7.0, 11,
                      color=(color[0], color[1], color[2]), alpha=(0.68 + 0.14 * focus_mix) * row_alpha)
            text.draw(name_text, row_x + 40.0, row_y + 7.0, name_size,
                      color=(0.96, 0.97, 1.0), alpha=(0.78 + 0.16 * focus_mix) * row_alpha)
            text.draw(score_text, row_x + row_w - sw - 12.0, row_y + 7.0, score_size,
                      color=(0.96, 0.97, 1.0), alpha=(0.78 + 0.16 * focus_mix) * row_alpha)
            meta_text = f"{combo_text}  {acc_text}"
            text.draw(meta_text, row_x + 40.0, row_y + 26.0, stat_size,
                      color=(0.78, 0.84, 0.97), alpha=(0.58 + 0.14 * focus_mix) * row_alpha)
            aw, _ = text.measure(acc_text, stat_size)
            if participant.eliminated:
                text.draw("MISS", row_x + row_w - 40.0 - aw, row_y + 26.0, 11,
                          color=(1.0, 0.46, 0.46), alpha=0.88 * row_alpha)
            self._leaderboard_rects.append((row_x, row_y, row_w, row_h, participant.leaderboard_id))

    def _ordered_leaderboard_participants(self, current_time_ms: float | None = None) -> list[_GameplayParticipant]:
        if (
            self._leaderboard_sorted_cache
            and current_time_ms is not None
            and current_time_ms - self._leaderboard_last_sort_ms < 90.0
        ):
            return self._leaderboard_sorted_cache
        previous_order = {
            leaderboard_id: idx
            for idx, leaderboard_id in enumerate(self._leaderboard_display_order)
        }
        ordered = sorted(
            self._participants,
            key=functools.cmp_to_key(
                lambda a, b: self._compare_leaderboard_participants(a, b, previous_order)
            ),
        )
        self._leaderboard_display_order = [p.leaderboard_id for p in ordered]
        self._leaderboard_sorted_cache = ordered
        if current_time_ms is not None:
            self._leaderboard_last_sort_ms = current_time_ms
        return ordered

    def _compare_leaderboard_participants(
        self,
        a: _GameplayParticipant,
        b: _GameplayParticipant,
        previous_order: dict[str, int],
    ) -> int:
        score_a = a.elimination_score if a.eliminated else a.current_score
        score_b = b.elimination_score if b.eliminated else b.current_score
        acc_a = a.current_accuracy
        acc_b = b.current_accuracy
        combo_a = a.current_combo
        combo_b = b.current_combo

        score_gap = abs(score_a - score_b)
        acc_gap = abs(acc_a - acc_b)
        combo_gap = abs(combo_a - combo_b)
        if score_gap <= 1000 and acc_gap <= 0.0001 and combo_gap <= 1:
            prev_a = previous_order.get(a.leaderboard_id, 10_000)
            prev_b = previous_order.get(b.leaderboard_id, 10_000)
            if prev_a != prev_b:
                return -1 if prev_a < prev_b else 1

        for lhs, rhs in (
            (score_a, score_b),
            (acc_a, acc_b),
            (combo_a, combo_b),
        ):
            if lhs != rhs:
                return -1 if lhs > rhs else 1
        if a.name.lower() != b.name.lower():
            return -1 if a.name.lower() < b.name.lower() else 1
        return 0

    def _draw_pause_overlay(self) -> None:
        w, h = self.wnd.buffer_size
        panels = self.app.panels
        text = self.app.text
        context = self.app.menu_context()
        theme = context.theme
        colors = theme.colors

        panels.begin_batch()
        panels.draw(0, 0, w, h, radius=0.0,
                    color=(colors.scrim[0], colors.scrim[1], colors.scrim[2], 0.55),
                    border_color=(0, 0, 0, 0), border_width=0)

        text.begin()
        sheet_w = min(360.0, w * 0.34)
        btn_w, btn_h = sheet_w - 36.0, 42.0
        start_y_offset = 122.0
        option_gap = 14.0
        bottom_pad = 18.0
        sheet_h = start_y_offset + len(PAUSE_ITEMS) * btn_h + max(0, len(PAUSE_ITEMS) - 1) * option_gap + bottom_pad
        sheet_x = (w - sheet_w) * 0.5
        sheet_y = h * 0.28
        panels.draw(
            sheet_x, sheet_y, sheet_w, sheet_h,
            radius=context.tokens.radius_l,
            color=(colors.surface_container[0], colors.surface_container[1], colors.surface_container[2], 0.88),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        panels.draw(
            sheet_x, sheet_y, sheet_w, 42.0,
            radius=context.tokens.radius_l,
            color=(colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.34),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        pill_w = 74.0
        pill_h = 18.0
        panels.draw(
            sheet_x + 18.0, sheet_y + 12.0, pill_w, pill_h,
            radius=pill_h * 0.5,
            color=(colors.surface_container_low[0], colors.surface_container_low[1], colors.surface_container_low[2], 0.76),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        text.draw("Paused", sheet_x + 18.0, sheet_y + 16.0, 11, color=colors.text_muted)
        text.draw("Menu", sheet_x + 18.0, sheet_y + 46.0, theme.typography.headline, color=colors.text_primary)

        start_y = sheet_y + start_y_offset
        for i, label in enumerate(PAUSE_ITEMS):
            bx = sheet_x + 18.0
            by = start_y + i * (btn_h + option_gap)
            is_hovered = (
                bx <= self._mouse_x <= bx + btn_w
                and by <= self._mouse_y <= by + btn_h
            )
            is_selected = (i == self._pause_selection)
            if is_hovered or is_selected:
                col = (colors.primary_container[0], colors.primary_container[1], colors.primary_container[2], 0.84)
                fg = colors.text_primary
            else:
                col = (colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.54)
                fg = colors.text_secondary

            panels.draw(bx, by, btn_w, btn_h, radius=context.tokens.radius_m,
                        color=col, border_color=(0.0, 0.0, 0.0, 0.0), border_width=0.0)
            if is_selected:
                panels.draw(
                    bx + 8.0, by + 8.0, 3.0, btn_h - 16.0,
                    radius=1.5,
                    color=(colors.focus_ring[0], colors.focus_ring[1], colors.focus_ring[2], 0.92),
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0,
                )
            tw, _ = text.measure(label, 20)
            text.draw(label, bx + 18.0, by + 6.0, 20, color=fg)

        panels.end_batch()
        text.end()

    def _draw_fullscreen_overlay(self, color: tuple) -> None:
        w, h = self.wnd.buffer_size
        self.app.panels.draw(0, 0, w, h, radius=0.0,
                             color=color,
                             border_color=(0, 0, 0, 0), border_width=0)

    def _toggle_pause(self, *, show_menu: bool = False) -> None:
        if self._paused:
            self._set_paused(False)
        else:
            self._set_paused(True, show_menu=show_menu)

    def _open_pause_menu(self) -> None:
        self._pause_selection = 0
        self._set_paused(True, show_menu=True)

    def _handle_pause_action(self, idx: int) -> None:
        if idx == 0:
            self._set_paused(False)
        elif idx == 1:
            self.app.toggle_settings()
        elif idx == 2:
            self._exiting = True
            self._exit_timer = 0.0
        elif idx == 3:
            self.app.wnd.close()

    # ----------------------------------------------------------- events

    def on_resize(self, width: int, height: int):
        self.projection = self._build_projection()
        self._invalidate_timeline_layout()
        proj_bytes = self.projection.astype("f4").tobytes()
        if self.circle_prog:
            self.circle_prog["projection"].write(proj_bytes)
        if self.slider_prog:
            self.slider_prog["projection"].write(proj_bytes)
        if self.approach_prog is not None:
            self.approach_prog["projection"].write(proj_bytes)
        if self.ball_prog is not None:
            self.ball_prog["projection"].write(proj_bytes)
        if self.cursor_prog is not None:
            self.cursor_prog["projection"].write(proj_bytes)
        if self.trail_prog is not None:
            self.trail_prog["projection"].write(proj_bytes)

    def on_key_event(self, key, action, modifiers):
        if not self._loading_ready:
            return
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return

        if self._pause_menu_open:
            if key == keys.ESCAPE:
                self._set_paused(False)
            elif key == keys.SPACE:
                self._set_paused(False)
            elif key == keys.UP:
                self._pause_selection = (self._pause_selection - 1) % len(PAUSE_ITEMS)
            elif key == keys.DOWN:
                self._pause_selection = (self._pause_selection + 1) % len(PAUSE_ITEMS)
            elif key == keys.ENTER:
                self._handle_pause_action(self._pause_selection)
            return

        if key in {keys.LEFT, keys.A}:
            self._seek_relative(-TIMELINE_KEY_SEEK_STEP_MS)
        elif key in {keys.RIGHT, keys.D}:
            self._seek_relative(TIMELINE_KEY_SEEK_STEP_MS)
        elif key == keys.ESCAPE:
            self._open_pause_menu()
        elif key == keys.TAB:
            self._cycle_skin()
        elif key == keys.SPACE:
            self._toggle_pause(show_menu=False)
        elif key == keys.F12:
            self._take_screenshot()

    def _take_screenshot(self, name="screenshot"):
        from PIL import Image
        w, h = self.wnd.buffer_size
        data = self.ctx.screen.read(components=3)
        img = Image.frombytes("RGB", (w, h), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        path = f"{name}.png"
        img.save(path)
        print(f"Screenshot saved to {path}")

    def on_mouse_press(self, x: int, y: int, button: int) -> None:
        if not self._loading_ready:
            return
        self._mouse_x = x
        self._mouse_y = y
        if button != 1:
            return

        hit_kind, hit_value = self._timeline_hit_test(x, y)
        if hit_kind == "volume_button" and hit_value is not None:
            key = str(hit_value)
            self._refresh_volume_slider_timer(key)
            self._toggle_volume_control_muted(key)
            return
        if hit_kind == "volume_slider" and hit_value is not None:
            key, value = hit_value
            key = str(key)
            self._volume_slider_dragging = key
            self._refresh_volume_slider_timer(key)
            self._set_volume_control_value(key, float(value))
            return
        if self._volume_controls_active():
            self._close_volume_sliders()
        if hit_kind == "pause":
            self._toggle_pause(show_menu=False)
            return
        if hit_kind == "speed" and hit_value is not None:
            self._set_playback_speed(float(hit_value))
            return
        if hit_kind == "track" and hit_value is not None:
            self._begin_timeline_drag(float(hit_value))
            return

        if not self._pause_menu_open:
            leaderboard_id = self._leaderboard_hit_test(x, y)
            if leaderboard_id is not None:
                if self._leaderboard_pinned_id == leaderboard_id:
                    self._leaderboard_pinned_id = None
                else:
                    self._leaderboard_pinned_id = leaderboard_id
                return

        if not self._pause_menu_open:
            return
        w, h = self.wnd.buffer_size
        sheet_w = min(360.0, w * 0.34)
        sheet_x = (w - sheet_w) * 0.5
        sheet_y = h * 0.28
        btn_w, btn_h = sheet_w - 36.0, 42.0
        start_y = sheet_y + 122.0
        for i in range(len(PAUSE_ITEMS)):
            bx = sheet_x + 18.0
            by = start_y + i * (btn_h + 14)
            if bx <= x <= bx + btn_w and by <= y <= by + btn_h:
                self._handle_pause_action(i)
                return

    def on_mouse_move(self, x: int, y: int, dx: int, dy: int) -> None:
        if not self._loading_ready:
            return
        self._mouse_x = x
        self._mouse_y = y
        if self._volume_slider_dragging is not None:
            self._update_timeline_hover(x, y)
            return
        if self._timeline_dragging:
            layout, _alpha = self._timeline_visual_layout()
            target_ms = self._timeline_time_from_x(x, layout)
            self._update_timeline_drag(target_ms)
            return
        self._leaderboard_hover_id = self._leaderboard_hit_test(x, y)
        self._update_timeline_hover(x, y)

    def on_mouse_release(self, x: int, y: int, button: int) -> None:
        if not self._loading_ready:
            return
        self._mouse_x = x
        self._mouse_y = y
        if button == 1 and self._volume_slider_dragging is not None:
            self._refresh_volume_slider_timer(self._volume_slider_dragging)
            self._volume_slider_dragging = None
            self._update_timeline_hover(x, y)
            return
        if button == 1 and self._timeline_dragging:
            self._finish_timeline_drag()
            self._update_timeline_hover(x, y)
