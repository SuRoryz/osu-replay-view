"""Song selection menu scene -- osu!-style song select with scrolling cards."""

from __future__ import annotations

import io
import hashlib
import os
import subprocess
import sys
import threading
from pathlib import Path

import moderngl
import pygame.mixer

from osu_map.mods import (
    MOD_FLAG_MAP, MOD_SHORT, FL, incompatible_with, mod_string, normalize_mods,
)
from osu_map.scanner import BeatmapInfo, BeatmapSet
from profiling import profiler
from replay.data import ReplayData, ReplaySummary
from runtime_paths import MAPS_DIR, replay_dir_for_set
from scenes.base import Scene
from social.models import ChatMessagePayload, SharedBeatmap, SharedReplay
from ui.menu.animation import AnimatedFloat
from ui.menu.song_select_view import SongSelectMenuView

# Layout constants
RIGHT_PANEL_X = 0.40
CARD_WIDTH_FRAC = 0.56
CARD_HEIGHT = 94
CARD_GAP = 10
CARD_SKEW = 8.0
CARD_STACK_SHIFT = 14
BOTTOM_BAR_H = 54
BOTTOM_SECTION_GAP = 14
INFO_PANEL_W_FRAC = 0.39
PANEL_MARGIN = 20
SECTION_GAP = 12
PANEL_RADIUS = 18.0
SECTION_RADIUS = 14.0
PLAY_BUTTON_W = 100
SONG_LIST_TOP_PAD = 60
MODS_SUMMARY_W = 150
REPLAY_HEADER_GAP = 16
REPLAY_TOGGLE_W = 78
REPLAY_TOGGLE_H = 22
REPLAY_TOGGLE_GAP = 8
MAX_MULTI_REPLAYS = 10

SECTION_HEADER_SIZE = 16
SECTION_META_SIZE = 13
HEADER_META_DY = 1
HEADER_GAP_X = 10

CONTROL_TEXT_SIZE = 14
CONTROL_TEXT_DY = 10
REPLAY_ROW_H = 44
REPLAY_ROW_GAP = 4
REPLAY_TOP_PAD = 8
REPLAY_BOTTOM_PAD = 8

CARD_RADIUS = 10.0
CARD_BORDER_W = 1.0
CARD_ACCENT_W = 3
CARD_ACCENT_ROUND = 2.0
CARD_CONTENT_X = 18
CARD_CONTENT_TOP = 14
CARD_TITLE_SIZE = 22
CARD_ARTIST_SIZE = 14
CARD_META_SIZE = 13
CARD_TITLE_TO_ARTIST = 26
CARD_ARTIST_TO_META = 20
CARD_BADGE_H = 27
CARD_BADGE_Y = 14
CARD_BADGE_TEXT_SIZE = 12
CARD_BADGE_TEXT_DY = 5

INFO_PAD = 20
DIVIDER_H = 2
DIVIDER_PAD_TOP = 14
DIVIDER_PAD_BOT = 14

TITLE_SIZE = 26
ARTIST_SIZE = 15
MAPPER_SIZE = 12
DIFF_CHIP_H = 22
DIFF_CHIP_TEXT_SIZE = 13
DIFF_CHIP_TEXT_DY = 4

STAT_CHIP_H = 22
STAT_CHIP_TEXT_DY = 5
STAT_LABEL_SIZE = 12
STAT_VALUE_SIZE = 12

DIFF_ROW_PITCH = 24
DIFF_ROW_DOT_R = 3
DIFF_ROW_TEXT_SIZE = 13

BOTTOM_BAR_PAD_Y = 4
BAR_LABEL_SIZE = 13
BAR_LABEL_Y = 12
MOD_CHIP_SIZE = 45
MOD_CHIP_Y_OFF = 0
MOD_CHIP_TEXT_SIZE = 15
MOD_CHIP_GAP = 6
PLAY_BUTTON_H = BOTTOM_BAR_H - BOTTOM_BAR_PAD_Y * 2
PLAY_SUBTITLE_SIZE = 11
PLAY_TITLE_SIZE = 22
SCROLL_SPEED = 60.0
SCROLL_LERP = 0.12
SOUND_SECTION_W = 160
SOUND_SECTION_H = 54
SOUND_BUTTON_SIZE = 26.0
SOUND_BUTTON_GAP = 8.0
SOUND_SLIDER_PANEL_W = 34.0
SOUND_SLIDER_PANEL_H = 96.0
SOUND_SLIDER_GAP = 8.0
SOUND_SLIDER_TRACK_W = 6.0
SOUND_SLIDER_TRACK_H = 58.0
SOUND_SLIDER_HIDE_DELAY = 1.5
SOUND_SLIDER_ANIM_SPEED = 12.0
BOTTOM_SQUARE_SCALE = 0.75

INFO_STATS_SIZE = 12
INFO_DIFF_SIZE = 14
INFO_TITLE_STATS_GAP = 14
INFO_TOP_ROW_INSET = 7
INFO_META_GAP = 6

REPLAY_SECTION_H = 210
DRAG_THRESHOLD = 6

EXIT_DURATION = 0.6
ENTER_DURATION = 0.5

MOMENTUM_DECAY = 0.92
MOMENTUM_MIN = 0.5


def _ease_in_quad(t: float) -> float:
    return t * t


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _truncate_text(text, value: str, size: int, max_width: float) -> str:
    if hasattr(text, "truncate"):
        return text.truncate(value, size, max_width)
    if max_width <= 0:
        return ""
    if text.measure(value, size)[0] <= max_width:
        return value
    ellipsis = "..."
    trimmed = value
    while trimmed:
        trimmed = trimmed[:-1]
        candidate = trimmed.rstrip() + ellipsis
        if text.measure(candidate, size)[0] <= max_width:
            return candidate
    return ellipsis


def _normalize_replay_path(path: str | None) -> str | None:
    if not path:
        return None
    return os.path.normcase(os.path.normpath(str(path)))


def _draw_header_pair(text, label: str, meta: str, x: float, y: float,
                      *, label_size: int = SECTION_HEADER_SIZE,
                      meta_size: int = SECTION_META_SIZE,
                      label_color=None,
                      meta_color=None) -> None:
    if label_color is None:
        label_color = COL_TEXT
    if meta_color is None:
        meta_color = COL_TEXT_MUTED
    text.draw(label, x, y, label_size, color=label_color)
    if not meta:
        return
    label_w, _ = text.measure(label, label_size)
    text.draw(meta, x + label_w + HEADER_GAP_X, y + HEADER_META_DY, meta_size, color=meta_color)


# Colors
COL_ACCENT = (0.90, 0.42, 0.95)
COL_ACCENT_SOFT = (0.64, 0.32, 0.80)
COL_TEXT = (0.98, 0.98, 1.0)
COL_TEXT_DIM = (0.79, 0.80, 0.88)
COL_TEXT_MUTED = (0.60, 0.62, 0.72)

COL_INFO_BG = (0.030, 0.032, 0.070, 0.78)
COL_SECTION_BORDER = (0.22, 0.20, 0.34, 0.56)

COL_CARD_BG = (0.055, 0.050, 0.105, 0.80)
COL_CARD_HOVER = (0.090, 0.075, 0.145, 0.88)
COL_CARD_SELECTED = (0.29, 0.16, 0.42, 0.88)
COL_CARD_BORDER = (0.22, 0.20, 0.34, 0.62)
COL_CARD_BORDER_SEL = (0.92, 0.50, 0.96, 0.96)

COL_BOTTOM_BG = (0.045, 0.045, 0.085, 0.94)
COL_MOD_ACTIVE = (0.92, 0.46, 0.96, 0.95)
COL_MOD_INACTIVE = (0.10, 0.10, 0.16, 0.94)
COL_MOD_LOCKED = (0.16, 0.14, 0.18, 0.92)
COL_MOD_BORDER_ACTIVE = (0.97, 0.58, 0.99, 1.0)
COL_MOD_BORDER_INACTIVE = (0.24, 0.22, 0.34, 0.82)
COL_MOD_BORDER_LOCKED = (0.75, 0.62, 0.26, 0.85)

_MOD_LABELS = ["Easy", "NoFail", "HalfTime", "HardRock", "Hidden", "DoubleTime", "Flashlight"]


class SongSelectScene(Scene):
    """osu!-style song selection screen."""

    def __init__(self, app):
        super().__init__(app)
        self._menu_view = SongSelectMenuView()
        self._debug_layout: bool = os.environ.get("OSU_DEBUG_LAYOUT", "").lower() in ("1", "true", "yes", "on")
        self.mod_flag_map = MOD_FLAG_MAP
        self.mod_short = MOD_SHORT
        self.mod_string = mod_string
        self.fl_flag = FL
        self._mod_labels = _MOD_LABELS
        self._sets: list[BeatmapSet] = []
        self._selected_idx: int = 0
        self._selected_diff_idx: int = 0
        self._hover_idx: int = -1
        self._mouse_x: int = 0
        self._mouse_y: int = 0
        self._preview_playing: bool = False
        self._mixer_inited: bool = False
        self._pending_info: BeatmapInfo | None = None
        self._pending_replay_paths: list[str] = []
        self._pending_include_danser: bool = False
        self._pending_multi_replay: bool = False
        self._pending_mods_overridden: bool = False
        self._selected_replay: str | None = None
        self._selected_multi_replays: list[str] = []
        self._multi_replay_enabled: bool = False
        self._danser_replay_enabled: bool = False
        self._replay_rects: list[tuple[int, int, int, int, str]] = []
        self._replay_action_rects: list[tuple[float, float, float, float, str]] = []
        self._multi_toggle_rect: tuple[int, int, int, int] | None = None
        self._danser_toggle_rect: tuple[int, int, int, int] | None = None
        self._replay_local_tab_rect: tuple[float, float, float, float] | None = None
        self._replay_online_tab_rect: tuple[float, float, float, float] | None = None
        self._replay_source_tab: str = "local"
        self._selected_online_replay_id: str | None = None
        self._replay_context_menu_rect: tuple[float, float, float, float] | None = None
        self._replay_context_menu_options: list[tuple[float, float, float, float, str]] = []
        self._replay_context_target: str | None = None
        self._replay_context_menu_anim = AnimatedFloat(0.0, 0.0, 16.0)
        self._songs_open_btn_rect: tuple[int, int, int, int] | None = None
        self._song_list_interact_rect: object | None = None
        self._song_card_rects: list[tuple[float, float, float, float, int]] = []

        # Song list scroll
        self._scroll_target: float = 0.0
        self._scroll_current: float = 0.0
        self._dragging_songs: bool = False
        self._drag_start_y: int = 0
        self._drag_accum: float = 0.0
        self._song_velocity: float = 0.0

        # Replay section scroll
        self._replay_scroll_target: float = 0.0
        self._replay_scroll_current: float = 0.0
        self._dragging_replays: bool = False
        self._replay_drag_start_y: int = 0
        self._replay_drag_accum: float = 0.0
        self._replay_velocity: float = 0.0
        self._replay_section_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._replay_total_h: float = 0.0

        # Mods
        self._active_mods: int = 0
        self._mods_overridden: bool = False
        self._mod_rects: list[tuple[int, int, int, int, int]] = []
        self._mods_trigger_rect: tuple[float, float, float, float] | None = None
        self._mods_surface_rect: tuple[float, float, float, float] | None = None
        self._mods_palette_locked: bool = False
        self._mods_palette_anim: float = 0.0

        # Replay list cache (avoids per-frame filesystem I/O)
        self._replay_cache: dict[str, list[str]] = {}
        self._replay_cache_mtime: dict[str, int | None] = {}
        self._replay_summary_cache: dict[str, ReplaySummary] = {}
        self._replay_summary_loading: set[str] = set()
        self._replay_summary_failed: set[str] = set()
        self._replay_entity_cache: dict[str, tuple[str, int, str] | None] = {}
        self._replay_entity_index_key: tuple[str, int | None] | None = None
        self._replay_entity_index: dict[str, str] = {}
        self._flattened_list_cache: list[tuple[int, int]] = []
        self._flattened_list_dirty: bool = True
        self._beatmap_lookup: dict[str, tuple[int, int]] = {}

        # Async peek_mods (avoids blocking when selecting replay)
        self._replay_summary_generation: int = 0
        self._peek_mods_pending: list[tuple[int, str, ReplaySummary | None]] = []
        self._peek_mods_lock = threading.Lock()

        # Deferred audio loading (avoids blocking when changing songs)
        self._audio_load_pending: tuple[io.BytesIO, str, float] | None = None
        self._audio_load_lock = threading.Lock()
        self._audio_load_path: str | None = None  # path we're loading (for stale check)
        self._preview_cache_lock = threading.Lock()
        self._preview_audio_cache: dict[str, tuple[bytes, str]] = {}
        self._preview_cache_order: list[str] = []
        self._preview_loading_paths: set[str] = set()

        # Animation
        self._anim_state: str = "idle"
        self._last_cards_x_off: float = 0.0
        self._anim_timer: float = 0.0
        self._debug_regions: list[tuple[float, float, float, float, tuple[float, float, float, float]]] = []
        self._sound_slider_dragging: str | None = None
        self._sound_slider_visibility = {"sfx": 0.0, "music": 0.0}
        self._sound_slider_hide_timers = {"sfx": 0.0, "music": 0.0}

    def on_enter(self) -> None:
        self._sets = self.app.scanner.sets
        self._rebuild_beatmap_lookup()
        self._invalidate_replay_cache()
        self._mark_flattened_list_dirty()
        self._anim_state = "entering"
        self._anim_timer = 0.0
        if self._sets:
            self._selected_idx = 0
            self._selected_diff_idx = 0
            self._center_scroll()
            self._start_preview()

    def on_leave(self) -> None:
        self._stop_preview()

    def _rebuild_beatmap_lookup(self) -> None:
        self._beatmap_lookup = {}
        for set_idx, bset in enumerate(self._sets):
            for diff_idx, info in enumerate(bset.maps):
                if info.beatmap_md5:
                    self._beatmap_lookup.setdefault(info.beatmap_md5, (set_idx, diff_idx))

    def on_global_audio_settings_changed(self) -> None:
        self._apply_preview_volume()

    # ----------------------------------------------------------- audio preview

    def _ensure_mixer(self) -> None:
        if not self._mixer_inited:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                pygame.mixer.music.set_volume(self.app.effective_music_volume)
            except Exception:
                pass
            self._mixer_inited = True

    def _start_preview(self) -> None:
        if not self._sets:
            return
        bset = self._sets[self._selected_idx]
        audio = bset.audio_path
        if audio is None or not Path(audio).is_file():
            return

        self._ensure_mixer()
        start_sec = max(0, bset.preview_time) / 1000.0

        cached = self._preview_audio_cache_get(audio)
        if cached is not None:
            data, namehint = cached
            with self._audio_load_lock:
                self._audio_load_pending = (io.BytesIO(data), namehint, start_sec)
                self._audio_load_path = audio
            self._prefetch_adjacent_previews()
            return

        def _audio_load_worker(preview_path: str, requested_start_sec: float) -> None:
            try:
                with open(preview_path, "rb") as f:
                    data = f.read()
                ext = Path(preview_path).suffix.lstrip(".").lower() or "ogg"
                namehint = ext if ext in ("mp3", "ogg", "wav", "mid", "mod") else "ogg"
                self._preview_audio_cache_store(preview_path, data, namehint)
                with self._audio_load_lock:
                    self._audio_load_pending = (io.BytesIO(data), namehint, requested_start_sec)
                    self._audio_load_path = preview_path
            except Exception:
                pass
            finally:
                with self._preview_cache_lock:
                    self._preview_loading_paths.discard(preview_path)

        with self._audio_load_lock:
            self._audio_load_pending = None
            self._audio_load_path = audio
        with self._preview_cache_lock:
            if audio in self._preview_loading_paths:
                return
            self._preview_loading_paths.add(audio)
        threading.Thread(target=_audio_load_worker, args=(audio, start_sec), daemon=True).start()
        self._prefetch_adjacent_previews()

    def _stop_preview(self) -> None:
        if self._preview_playing:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            self._preview_playing = False

    def _sound_control_value(self, key: str) -> float:
        if key == "music":
            return self.app.settings.music_volume
        return self.app.settings.sfx_volume

    def _sound_control_muted(self, key: str) -> bool:
        if key == "music":
            return self.app.settings.music_muted
        return self.app.settings.sfx_muted

    def _refresh_sound_slider_timer(self, key: str) -> None:
        self._sound_slider_hide_timers[key] = SOUND_SLIDER_HIDE_DELAY

    def _sound_controls_active(self) -> bool:
        return (
            self._sound_slider_dragging is not None
            or any(value > 0.01 for value in self._sound_slider_visibility.values())
            or any(value > 0.0 for value in self._sound_slider_hide_timers.values())
        )

    def _close_sound_sliders(self) -> None:
        self._sound_slider_dragging = None
        for key in self._sound_slider_visibility:
            self._sound_slider_visibility[key] = 0.0
            self._sound_slider_hide_timers[key] = 0.0

    def _apply_preview_volume(self) -> None:
        try:
            pygame.mixer.music.set_volume(self.app.effective_music_volume)
        except Exception:
            pass

    def _set_music_volume(self, value: float) -> None:
        value = _clamp(value, 0.0, 1.0)
        self.app.set_music_volume(value)
        self.app.set_music_muted(value <= 0.0001)
        self._apply_preview_volume()

    def _set_music_muted(self, muted: bool) -> None:
        self.app.set_music_muted(muted)
        self._apply_preview_volume()

    def _set_sfx_volume(self, value: float) -> None:
        value = _clamp(value, 0.0, 1.0)
        self.app.set_sfx_volume(value)
        self.app.set_sfx_muted(value <= 0.0001)

    def _set_sfx_muted(self, muted: bool) -> None:
        self.app.set_sfx_muted(muted)

    def _set_sound_control_value(self, key: str, value: float) -> None:
        if key == "music":
            self._set_music_volume(value)
        else:
            self._set_sfx_volume(value)

    def _toggle_sound_control_muted(self, key: str) -> None:
        if key == "music":
            self._set_music_muted(not self.app.settings.music_muted)
        else:
            self._set_sfx_muted(not self.app.settings.sfx_muted)

    def _sound_layout(self, w: int, h: int, *, y_offset: float = 0.0) -> dict:
        bar_y = h - BOTTOM_BAR_H + int(y_offset)
        play_h = max(24.0, round(float(MOD_CHIP_SIZE) * BOTTOM_SQUARE_SCALE))
        play_y = bar_y + (BOTTOM_BAR_H - play_h) * 0.5
        play_x = w - PLAY_BUTTON_W - PANEL_MARGIN
        button_size = play_h
        buttons_total_w = button_size * 2.0 + SOUND_BUTTON_GAP
        section_x = play_x - SECTION_GAP - buttons_total_w
        section_y = float(play_y)

        controls = {}
        for idx, (key, label) in enumerate((("sfx", "S"), ("music", "M"))):
            button_x = section_x + idx * (button_size + SOUND_BUTTON_GAP)
            slider_x = button_x + (button_size - SOUND_SLIDER_PANEL_W) * 0.5
            slider_y = section_y - SOUND_SLIDER_PANEL_H - SOUND_SLIDER_GAP
            track_x = slider_x + (SOUND_SLIDER_PANEL_W - SOUND_SLIDER_TRACK_W) * 0.5
            track_y = slider_y + 12.0
            controls[key] = {
                "key": key,
                "label": label,
                "button": {"x": button_x, "y": section_y, "w": button_size, "h": button_size},
                "slider": {
                    "x": slider_x,
                    "y": slider_y,
                    "w": SOUND_SLIDER_PANEL_W,
                    "h": SOUND_SLIDER_PANEL_H,
                    "track_x": track_x,
                    "track_y": track_y,
                    "track_w": SOUND_SLIDER_TRACK_W,
                    "track_h": SOUND_SLIDER_TRACK_H,
                },
            }
        return {
            "bar_y": bar_y,
            "section_x": section_x,
            "section_y": section_y,
            "section_w": buttons_total_w,
            "button_size": button_size,
            "play_x": play_x,
            "play_y": play_y,
            "play_h": play_h,
            "controls": controls,
        }

    @staticmethod
    def _point_in_rect(x: float, y: float, rect: dict | tuple[int, int, int, int]) -> bool:
        if isinstance(rect, tuple):
            rx, ry, rw, rh = rect
            return rx <= x <= rx + rw and ry <= y <= ry + rh
        return rect["x"] <= x <= rect["x"] + rect["w"] and rect["y"] <= y <= rect["y"] + rect["h"]

    def _sound_slider_value_from_y(self, y: float, slider: dict) -> float:
        progress = (slider["track_y"] + slider["track_h"] - y) / max(1.0, slider["track_h"])
        return _clamp(progress, 0.0, 1.0)

    def _sound_hit_test(self, x: float, y: float, *, y_offset: float = 0.0) -> tuple[str | None, object | None]:
        layout = self._sound_layout(*self.app.wnd.buffer_size, y_offset=y_offset)
        for key, control in layout["controls"].items():
            if self._point_in_rect(x, y, control["button"]):
                return "sound_button", key
            if (
                control["key"] == self._sound_slider_dragging
                or self._sound_slider_visibility[key] > 0.01
                or self._sound_slider_hide_timers[key] > 0.0
            ) and self._point_in_rect(x, y, control["slider"]):
                return "sound_slider", (key, self._sound_slider_value_from_y(y, control["slider"]))
        return None, None

    def _update_sound_slider_animations(self, dt: float, *, y_offset: float = 0.0) -> None:
        hit_kind, hit_value = self._sound_hit_test(self._mouse_x, self._mouse_y, y_offset=y_offset)
        hovered_key = None
        if hit_kind == "sound_button":
            hovered_key = str(hit_value)
        elif hit_kind == "sound_slider" and isinstance(hit_value, tuple):
            hovered_key = str(hit_value[0])

        for key in self._sound_slider_visibility:
            if self._sound_slider_dragging == key or hovered_key == key:
                self._refresh_sound_slider_timer(key)
            elif self._sound_slider_hide_timers[key] > 0.0:
                self._sound_slider_hide_timers[key] = max(0.0, self._sound_slider_hide_timers[key] - dt)

            target = 1.0 if (
                self._sound_slider_dragging == key
                or hovered_key == key
                or self._sound_slider_hide_timers[key] > 0.0
            ) else 0.0
            current = self._sound_slider_visibility[key]
            step = min(1.0, dt * SOUND_SLIDER_ANIM_SPEED)
            self._sound_slider_visibility[key] = current + (target - current) * step

    def _preview_audio_cache_get(self, audio_path: str) -> tuple[bytes, str] | None:
        with self._preview_cache_lock:
            cached = self._preview_audio_cache.get(audio_path)
            if cached is not None:
                if audio_path in self._preview_cache_order:
                    self._preview_cache_order.remove(audio_path)
                self._preview_cache_order.append(audio_path)
            return cached

    def _preview_audio_cache_store(self, audio_path: str, data: bytes, namehint: str) -> None:
        with self._preview_cache_lock:
            self._preview_audio_cache[audio_path] = (data, namehint)
            if audio_path in self._preview_cache_order:
                self._preview_cache_order.remove(audio_path)
            self._preview_cache_order.append(audio_path)
            while len(self._preview_cache_order) > 4:
                old_path = self._preview_cache_order.pop(0)
                self._preview_audio_cache.pop(old_path, None)

    def _prefetch_preview(self, audio_path: str) -> None:
        if not audio_path or not Path(audio_path).is_file():
            return
        with self._preview_cache_lock:
            if audio_path in self._preview_audio_cache or audio_path in self._preview_loading_paths:
                return
            self._preview_loading_paths.add(audio_path)

        def _worker(preview_path: str) -> None:
            try:
                with open(preview_path, "rb") as f:
                    data = f.read()
                ext = Path(preview_path).suffix.lstrip(".").lower() or "ogg"
                namehint = ext if ext in ("mp3", "ogg", "wav", "mid", "mod") else "ogg"
                self._preview_audio_cache_store(preview_path, data, namehint)
            except Exception:
                pass
            finally:
                with self._preview_cache_lock:
                    self._preview_loading_paths.discard(preview_path)

        threading.Thread(target=_worker, args=(audio_path,), daemon=True).start()

    def _prefetch_adjacent_previews(self) -> None:
        if not self._sets:
            return
        for idx in (self._selected_idx - 1, self._selected_idx + 1):
            if 0 <= idx < len(self._sets):
                audio_path = self._sets[idx].audio_path
                if audio_path:
                    self._prefetch_preview(audio_path)

    # ----------------------------------------------------------- helpers

    def _selected_info(self) -> BeatmapInfo | None:
        if not self._sets:
            return None
        bset = self._sets[self._selected_idx]
        di = min(self._selected_diff_idx, len(bset.maps) - 1)
        return bset.maps[di] if bset.maps else None

    def _selected_set(self) -> BeatmapSet | None:
        if not self._sets:
            return None
        return self._sets[self._selected_idx]

    def _refresh_selected_map_state(
        self,
        *,
        clear_replays: bool = True,
        restart_preview: bool = True,
    ) -> None:
        self._mark_flattened_list_dirty()
        if clear_replays:
            self._clear_replay_selection()
        self._center_scroll()
        if restart_preview:
            self._stop_preview()
            self._start_preview()
        bset = self._selected_set()
        if bset is not None:
            self.app.backgrounds.load(bset.background_path)

    def _apply_selected_indices(self, set_idx: int, diff_idx: int, *, clear_replays: bool = True) -> bool:
        if not (0 <= set_idx < len(self._sets)):
            return False
        bset = self._sets[set_idx]
        if not bset.maps:
            return False
        next_diff = max(0, min(diff_idx, len(bset.maps) - 1))
        if set_idx == self._selected_idx and next_diff == self._selected_diff_idx:
            if clear_replays:
                self._clear_replay_selection()
            return True
        same_set = (set_idx == self._selected_idx)
        self._selected_idx = set_idx
        self._selected_diff_idx = next_diff
        self._refresh_selected_map_state(
            clear_replays=clear_replays,
            restart_preview=not same_set,
        )
        return True

    def _find_selection_for_md5(self, beatmap_md5: str) -> tuple[int, int] | None:
        if not beatmap_md5:
            return None
        return self._beatmap_lookup.get(beatmap_md5)

    def _replay_base_label(self, replay_name: str, full_path: str) -> str:
        summary = self._replay_summary_cache.get(full_path)
        if summary is not None and summary.player_name:
            return f"{summary.player_name}  {mod_string(summary.mods) or '+NM'}"
        stem = Path(replay_name).stem
        if full_path in self._replay_summary_loading:
            return f"{stem}  Loading..."
        if full_path in self._replay_summary_failed:
            return f"{stem}  Read failed"
        return stem

    def _replay_row_labels(self, replay_name: str, full_path: str, duplicate_labels: set[str]) -> tuple[str, str]:
        primary = self._replay_base_label(replay_name, full_path)
        stem = Path(replay_name).stem or replay_name
        if primary in duplicate_labels:
            primary = f"{primary}  [{Path(replay_name).suffix.lower() or 'osr'}]"
        secondary = stem
        if secondary == primary:
            secondary = replay_name
        return primary, secondary

    def _effective_selected_replays(self) -> list[str]:
        values = list(self._selected_multi_replays) if self._multi_replay_enabled else ([self._selected_replay] if self._selected_replay else [])
        deduped: list[str] = []
        seen_entities: set[str] = set()
        seen_paths: set[str] = set()
        for path in values:
            normalized_path = _normalize_replay_path(path)
            if normalized_path is None or normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            entity_info = self._replay_entity_info(path)
            entity_key = entity_info[0] if entity_info is not None else normalized_path
            if entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)
            deduped.append(path)
        return deduped

    def _primary_selected_replay(self) -> str | None:
        replays = self._effective_selected_replays()
        return replays[0] if replays else None

    def _selected_replay_summary(self) -> ReplaySummary | None:
        primary = self._primary_selected_replay()
        if primary is None:
            return None
        return self._replay_summary_cache.get(primary)

    def _load_replay_summary_sync(self, path: str, *, force: bool = False) -> ReplaySummary | None:
        if not force:
            summary = self._replay_summary_cache.get(path)
            if summary is not None:
                return summary
            if path in self._replay_summary_loading:
                return None
        try:
            summary = ReplayData.peek_summary(path)
        except Exception:
            self._replay_summary_failed.add(path)
            self._replay_summary_loading.discard(path)
            return None
        self._replay_summary_cache[path] = summary
        self._replay_summary_loading.discard(path)
        self._replay_summary_failed.discard(path)
        return summary

    def _ensure_replay_summary(self, path: str) -> ReplaySummary | None:
        summary = self._replay_summary_cache.get(path)
        if summary is not None:
            return summary
        return self._load_replay_summary_sync(path, force=True)

    def _drain_pending_replay_summaries(self) -> None:
        with self._peek_mods_lock:
            pending = list(self._peek_mods_pending)
            self._peek_mods_pending.clear()
        generation = self._replay_summary_generation
        for entry_generation, path, summary in pending:
            if entry_generation != generation:
                continue
            self._replay_summary_loading.discard(path)
            if summary is None:
                self._replay_summary_failed.add(path)
                continue
            self._replay_summary_cache[path] = summary
            self._replay_summary_failed.discard(path)
            if path == self._primary_selected_replay():
                self._align_diff_for_summary(summary)
                if not self._mods_overridden:
                    self._active_mods = normalize_mods(summary.mods)

    def _selected_replay_count(self) -> int:
        return len(self._effective_selected_replays())

    def selected_replay_entity_keys(self) -> set[str]:
        keys: set[str] = set()
        for replay_path in self._effective_selected_replays():
            entity_info = self._replay_entity_info(replay_path)
            if entity_info is not None:
                keys.add(entity_info[0])
                continue
            normalized_path = _normalize_replay_path(replay_path)
            if normalized_path is not None:
                keys.add(normalized_path)
        return keys

    @staticmethod
    def _display_title(info: BeatmapInfo) -> str:
        return info.title_unicode or info.title or "Unknown map"

    def _shared_beatmap_from_info(self, info: BeatmapInfo) -> SharedBeatmap:
        return SharedBeatmap(
            map_md5=info.beatmap_md5,
            title=self._display_title(info),
            version=info.version or "?",
            mods=normalize_mods(self._active_mods),
            beatmap_id=self._online_beatmap_id_for_md5(info.beatmap_md5),
        )

    def _shared_replay_from_online(self, replay) -> SharedReplay:
        return SharedReplay(
            replay_id=replay.replay_id,
            player_name=replay.player_name or Path(replay.original_filename).stem,
            mods=normalize_mods(replay.mods),
            map_md5=replay.map_md5,
            beatmap_id=replay.beatmap_id,
            replay_hash=replay.replay_hash,
        )

    def _shared_replay_for_local_path(self, replay_path: str) -> SharedReplay | None:
        summary = self._ensure_replay_summary(replay_path)
        if summary is None:
            return None
        replay_id = None
        replay_hash = ""
        entity_info = self._replay_entity_info(replay_path)
        if entity_info is not None:
            _entity_key, beatmap_id, replay_hash = entity_info
            state = self.app.social_client.online_replays(beatmap_id)
            for item in state.items:
                if item.replay_hash == replay_hash:
                    replay_id = item.replay_id
                    break
        beatmap_id = self._online_beatmap_id_for_md5(summary.map_md5)
        return SharedReplay(
            replay_id=replay_id,
            player_name=summary.player_name or Path(replay_path).stem,
            mods=normalize_mods(summary.mods),
            map_md5=summary.map_md5,
            beatmap_id=beatmap_id,
            replay_hash=replay_hash,
        )

    def build_chat_share(self, command: str) -> tuple[str, ChatMessagePayload] | None:
        info = self._selected_info()
        if info is None:
            return None
        beatmap = self._shared_beatmap_from_info(info)
        if command == "/np":
            content = f"{beatmap.title} [{beatmap.version}] {mod_string(beatmap.mods) or '+NM'}"
            return content, ChatMessagePayload(kind="now_playing", beatmap=beatmap)
        if command != "/npr":
            return None
        replays: list[SharedReplay] = []
        seen_entities: set[str] = set()
        online_replay = self._selected_online_replay()
        if online_replay is not None:
            replay = self._shared_replay_from_online(online_replay)
            entity_key = self._shared_replay_entity_key(replay)
            if entity_key is not None:
                seen_entities.add(entity_key)
            replays.append(replay)
        else:
            for replay_path in self._effective_selected_replays():
                replay = self._shared_replay_for_local_path(replay_path)
                if replay is not None:
                    entity_key = self._shared_replay_entity_key(replay)
                    if entity_key is not None and entity_key in seen_entities:
                        continue
                    if entity_key is not None:
                        seen_entities.add(entity_key)
                    replays.append(replay)
        if not replays:
            return None
        replay_text = ", ".join(
            f"{replay.player_name or 'Replay'} {mod_string(replay.mods) or '+NM'}"
            for replay in replays
        )
        content = f"{beatmap.title} [{beatmap.version}] - replays ({replay_text})"
        return content, ChatMessagePayload(kind="now_playing_replays", beatmap=beatmap, replays=replays)

    def force_select_shared_beatmap(self, beatmap: SharedBeatmap | None) -> bool:
        if beatmap is None:
            return False
        target = self._find_selection_for_md5(beatmap.map_md5)
        if target is None:
            return False
        set_idx, diff_idx = target
        same_set = (set_idx == self._selected_idx)
        self._selected_idx = set_idx
        self._selected_diff_idx = diff_idx
        self._refresh_selected_map_state(clear_replays=True, restart_preview=not same_set)
        self._active_mods = normalize_mods(beatmap.mods)
        self._mods_overridden = True
        return True

    def selected_download_dir(self) -> str | None:
        bset = self._selected_set()
        if bset is None:
            return None
        return str(self._download_dir_for_set(bset))

    def _shared_replay_entity_key(self, replay: SharedReplay | None) -> str | None:
        if replay is None or replay.beatmap_id is None or not replay.replay_hash:
            return None
        return self._replay_entity_key(int(replay.beatmap_id), replay.replay_hash)

    def apply_shared_replay_selection(self, replays: list[SharedReplay], replay_paths: list[str]) -> bool:
        self._invalidate_replay_cache()
        local_index = dict(self._local_replay_entity_index_for_selected_set())
        for replay_path in replay_paths:
            if not replay_path or not Path(replay_path).is_file():
                continue
            entity_info = self._replay_entity_info(replay_path)
            if entity_info is None:
                continue
            entity_key, _beatmap_id, _replay_hash = entity_info
            local_index.setdefault(entity_key, replay_path)
        valid_paths: list[str] = []
        seen_paths: set[str] = set()
        seen_entities: set[str] = set()
        for replay in replays:
            entity_key = self._shared_replay_entity_key(replay)
            if entity_key is None or entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)
            local_path = local_index.get(entity_key)
            normalized_local = _normalize_replay_path(local_path)
            if not local_path or not Path(local_path).is_file() or normalized_local in seen_paths:
                continue
            seen_paths.add(normalized_local)
            valid_paths.append(normalized_local)
        if not valid_paths:
            for replay_path in replay_paths:
                normalized_path = _normalize_replay_path(replay_path)
                if normalized_path and Path(normalized_path).is_file() and normalized_path not in seen_paths:
                    seen_paths.add(normalized_path)
                    valid_paths.append(normalized_path)
        if not valid_paths:
            return False
        self._set_replay_source_tab("local")
        self._selected_online_replay_id = None
        if len(valid_paths) == 1:
            self._multi_replay_enabled = False
            self._selected_multi_replays = []
            self._danser_replay_enabled = False
            self._set_single_selected_replay(valid_paths[0])
            self._clamp_replay_scroll()
            return True
        self._multi_replay_enabled = True
        self._danser_replay_enabled = False
        self._selected_multi_replays = list(valid_paths[:MAX_MULTI_REPLAYS])
        self._selected_replay = self._selected_multi_replays[0] if self._selected_multi_replays else None
        first_summary = self._ensure_replay_summary(self._selected_multi_replays[0])
        self._align_diff_for_summary(first_summary)
        if first_summary is not None and not self._mods_overridden:
            self._active_mods = normalize_mods(first_summary.mods)
        self._clamp_replay_scroll()
        return True

    def _selected_online_beatmap_id(self) -> int | None:
        info = self._selected_info()
        if info is None or not info.beatmap_md5:
            return None
        return self._online_beatmap_id_for_md5(info.beatmap_md5)

    def _online_beatmap_id_for_md5(self, beatmap_md5: str) -> int | None:
        if not beatmap_md5:
            return None
        digest = hashlib.sha1(beatmap_md5.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) & 0x7FFFFFFF

    @staticmethod
    def _replay_entity_key(beatmap_id: int, replay_hash: str) -> str:
        return f"{beatmap_id}:{replay_hash}"

    def _replay_entity_info(self, path: str) -> tuple[str, int, str] | None:
        cached = self._replay_entity_cache.get(path)
        if cached is not None or path in self._replay_entity_cache:
            return cached
        summary = self._load_replay_summary_sync(path, force=path in self._replay_summary_failed)
        if summary is None or not summary.map_md5:
            return None
        beatmap_id = self._online_beatmap_id_for_md5(summary.map_md5)
        if beatmap_id is None:
            self._replay_entity_cache[path] = None
            return None
        try:
            payload = Path(path).read_bytes()
        except OSError:
            self._replay_entity_cache[path] = None
            return None
        replay_hash = hashlib.sha256(payload + f":{beatmap_id}".encode("utf-8")).hexdigest()
        info = (self._replay_entity_key(beatmap_id, replay_hash), beatmap_id, replay_hash)
        self._replay_entity_cache[path] = info
        return info

    def _local_replay_entity_index_for_selected_set(self) -> dict[str, str]:
        if not self._sets:
            return {}
        bset = self._sets[self._selected_idx]
        cache_key = (bset.directory, self._replay_cache_mtime.get(bset.directory))
        if self._replay_entity_index_key == cache_key:
            return self._replay_entity_index
        replay_dir = replay_dir_for_set(bset.directory)
        index: dict[str, str] = {}
        for replay_name in self._replays_for_set(bset):
            full_path = str(replay_dir / replay_name)
            info = self._replay_entity_info(full_path)
            if info is None:
                continue
            entity_key, _beatmap_id, _replay_hash = info
            index.setdefault(entity_key, full_path)
        self._replay_entity_index_key = cache_key
        self._replay_entity_index = index
        return index

    def _local_replay_path_for_online_replay(self, replay) -> str | None:
        entity_key = self._replay_entity_key(replay.beatmap_id, replay.replay_hash)
        local_path = self._local_replay_entity_index_for_selected_set().get(entity_key)
        if local_path and Path(local_path).is_file():
            return local_path
        fallback_path = replay.local_path or self.app.social_client.downloaded_path_for_replay(replay.replay_id)
        if fallback_path and Path(fallback_path).is_file():
            return fallback_path
        return None

    def _sync_online_replay_local_state(self, replay) -> None:
        local_path = self._local_replay_path_for_online_replay(replay)
        replay.local_path = local_path
        replay.is_downloaded = local_path is not None
        if replay.is_downloaded:
            replay.status_text = "Downloaded" if not replay.is_downloading else replay.status_text
        elif not replay.is_downloading:
            replay.status_text = ""

    def _download_dir_for_set(self, bset: BeatmapSet) -> Path:
        return replay_dir_for_set(bset.directory)

    def _online_replay_state(self):
        beatmap_id = self._selected_online_beatmap_id()
        if beatmap_id is None:
            return None
        return self.app.social_client.online_replays(beatmap_id)

    def _selected_online_replay(self):
        state = self._online_replay_state()
        if state is None:
            return None
        for item in state.items:
            if item.replay_id == self._selected_online_replay_id:
                return item
        return None

    def _set_replay_source_tab(self, tab: str) -> None:
        if tab not in {"local", "online"} or self._replay_source_tab == tab:
            return
        self._replay_source_tab = tab
        self._replay_scroll_target = 0.0
        self._replay_scroll_current = 0.0
        self._close_replay_context_menu(instant=True)
        if tab == "online":
            self._multi_replay_enabled = False
            self._danser_replay_enabled = False
            self._selected_multi_replays = []
            beatmap_id = self._selected_online_beatmap_id()
            if beatmap_id is not None:
                self.app.social_client.fetch_online_replays(beatmap_id, force=False)
        else:
            self._selected_online_replay_id = None

    def _close_replay_context_menu(self, *, instant: bool = False) -> None:
        self._replay_context_target = None
        if instant:
            self._replay_context_menu_rect = None
            self._replay_context_menu_options = []
            self._replay_context_menu_anim.snap(0.0)
            return
        self._replay_context_menu_anim.set_target(0.0)

    def _open_replay_context_menu(self, token: str, x: float, y: float) -> None:
        options = self._context_actions_for_token(token)
        if not options:
            self._close_replay_context_menu()
            return
        row_h = 28.0 * self.app.menu_context().density
        menu_w = 164.0 * self.app.menu_context().density
        self._replay_context_target = token
        self._replay_context_menu_rect = (x, y, menu_w, row_h * len(options))
        self._replay_context_menu_options = [
            (x + 4.0, y + idx * row_h, menu_w - 8.0, row_h, action)
            for idx, action in enumerate(options)
        ]
        self._replay_context_menu_anim.snap(0.0)
        self._replay_context_menu_anim.set_target(1.0)

    def _context_actions_for_token(self, token: str) -> list[str]:
        if token.startswith("online:"):
            replay = self.app.social_client._find_replay_by_id(token.split(":", 1)[1])
            if replay is None:
                return []
            self._sync_online_replay_local_state(replay)
            actions: list[str] = []
            if not replay.is_downloaded:
                actions.append("download")
            if replay.is_downloaded:
                actions.append("delete")
            return actions
        if not Path(token).is_file():
            return []
        actions = []
        if not self.is_uploaded_local_replay(token):
            actions.append("upload")
        if Path(token).is_file():
            actions.append("delete")
        return actions

    def is_uploaded_local_replay(self, replay_path: str) -> bool:
        entity_info = self._replay_entity_info(replay_path)
        if entity_info is None:
            return False
        _entity_key, beatmap_id, replay_hash = entity_info
        self.app.social_client.fetch_online_replays(beatmap_id, force=False)
        state = self.app.social_client.online_replays(beatmap_id)
        for item in state.items:
            if item.beatmap_id == beatmap_id and item.replay_hash == replay_hash:
                return True
        return self.app.social_client.is_uploaded_path(replay_path)

    def _activate_replay_context_action(self, action: str, token: str) -> None:
        if token.startswith("online:"):
            replay_id = token.split(":", 1)[1]
            replay = self.app.social_client._find_replay_by_id(replay_id)
            bset = self._sets[self._selected_idx] if self._sets else None
            if replay is None or bset is None:
                return
            self._sync_online_replay_local_state(replay)
            if action == "download":
                self.app.social_client.download_replay(replay_id, str(self._download_dir_for_set(bset)))
            elif action == "delete":
                deleted_path = replay.local_path
                self.app.social_client.delete_downloaded_replay(replay_id)
                self._invalidate_replay_cache()
                if deleted_path and self._selected_replay == deleted_path:
                    self._clear_replay_selection()
            return
        if action == "upload":
            summary = self._replay_summary_cache.get(token)
            if summary is None:
                try:
                    summary = ReplayData.peek_summary(token)
                except Exception:
                    self.app.social_client.push_system_message("Failed to read replay metadata for upload.")
                    return
                self._replay_summary_cache[token] = summary
            matched_idx = self._match_diff_idx_for_md5(summary.map_md5)
            if matched_idx is None:
                self.app.social_client.push_system_message("Replay does not match any difficulty in the selected beatmap set.")
                return
            self._align_diff_for_summary(summary)
            beatmap_id = self._online_beatmap_id_for_md5(summary.map_md5)
            if beatmap_id is not None:
                self.app.social_client.upload_local_replay(token, beatmap_id)
        elif action == "delete":
            try:
                os.remove(token)
            except OSError:
                pass
            self._invalidate_replay_cache()
            self._clear_replay_selection()

    def _handle_online_replay_activate(self, replay_id: str) -> None:
        replay = self.app.social_client._find_replay_by_id(replay_id)
        if replay is None:
            return
        self._sync_online_replay_local_state(replay)
        if replay.is_downloaded and replay.local_path:
            self._set_replay_source_tab("local")
            self._set_single_selected_replay(replay.local_path)
            return
        if self._selected_online_replay_id == replay_id:
            bset = self._sets[self._selected_idx] if self._sets else None
            if bset is not None:
                self.app.social_client.download_replay(replay_id, str(self._download_dir_for_set(bset)))
        else:
            self._selected_online_replay_id = replay_id

    def _clear_replay_selection(self) -> None:
        self._selected_replay = None
        self._selected_multi_replays = []
        self._selected_online_replay_id = None
        self._active_mods = 0
        self._mods_overridden = False

    def _toggle_multi_replay(self) -> None:
        self._multi_replay_enabled = not self._multi_replay_enabled
        if self._multi_replay_enabled:
            self._selected_multi_replays = [self._selected_replay] if self._selected_replay else []
        else:
            self._selected_replay = self._selected_multi_replays[0] if self._selected_multi_replays else self._selected_replay
            self._selected_multi_replays = []
            self._danser_replay_enabled = False
        self._sync_replay_mod_defaults()
        self._clamp_replay_scroll()

    def _toggle_danser_replay(self) -> None:
        if not self._multi_replay_enabled:
            return
        self._danser_replay_enabled = not self._danser_replay_enabled

    def _match_diff_idx_for_md5(self, beatmap_md5: str) -> int | None:
        if not beatmap_md5 or not self._sets:
            return None
        bset = self._sets[self._selected_idx]
        for diff_idx, info in enumerate(bset.maps):
            if info.beatmap_md5 == beatmap_md5:
                return diff_idx
        return None

    def _align_diff_for_summary(self, summary: ReplaySummary | None) -> None:
        if summary is None:
            return
        matched_idx = self._match_diff_idx_for_md5(summary.map_md5)
        if matched_idx is None or matched_idx == self._selected_diff_idx:
            return
        self._selected_diff_idx = matched_idx
        self._mark_flattened_list_dirty()
        self._center_scroll()

    def _visible_replays_for_set(self, bset: BeatmapSet) -> list[str]:
        replays = self._replays_for_set(bset)
        if not self._multi_replay_enabled or not self._selected_multi_replays:
            return replays
        primary = self._replay_summary_cache.get(self._selected_multi_replays[0])
        if primary is None or not primary.map_md5:
            return replays
        visible: list[str] = []
        replay_dir = replay_dir_for_set(bset.directory)
        for rp in replays:
            full_path = str(replay_dir / rp)
            summary = self._replay_summary_cache.get(full_path)
            if summary is None:
                self._queue_replay_summary(full_path)
                visible.append(rp)
                continue
            if summary.map_md5 == primary.map_md5:
                visible.append(rp)
        return visible

    def _queue_replay_summary(self, path: str, *, force: bool = False) -> None:
        if not force and (path in self._replay_summary_cache or path in self._replay_summary_loading):
            return
        generation = self._replay_summary_generation
        self._replay_summary_loading.add(path)
        self._replay_summary_failed.discard(path)

        def _peek_worker() -> None:
            try:
                summary = ReplayData.peek_summary(path)
                with self._peek_mods_lock:
                    self._peek_mods_pending.append((generation, path, summary))
            except Exception:
                with self._peek_mods_lock:
                    self._peek_mods_pending.append((generation, path, None))

        threading.Thread(target=_peek_worker, daemon=True).start()

    def _sync_replay_mod_defaults(self) -> None:
        summary = self._selected_replay_summary()
        if summary is not None and not self._mods_overridden:
            self._active_mods = normalize_mods(summary.mods)

    def _set_single_selected_replay(self, path: str) -> None:
        normalized_path = _normalize_replay_path(path) or path
        self._selected_replay = normalized_path
        summary = self._ensure_replay_summary(normalized_path)
        if summary is not None:
            self._align_diff_for_summary(summary)
            if not self._mods_overridden:
                self._active_mods = normalize_mods(summary.mods)

    def _toggle_single_selected_replay(self, path: str) -> None:
        normalized_path = _normalize_replay_path(path) or path
        if self._selected_replay == normalized_path:
            self._clear_replay_selection()
            return
        self._set_single_selected_replay(normalized_path)

    def _toggle_multi_selected_replay(self, path: str) -> None:
        normalized_path = _normalize_replay_path(path) or path
        self._ensure_replay_summary(normalized_path)
        entity_info = self._replay_entity_info(normalized_path)
        entity_key = entity_info[0] if entity_info is not None else normalized_path
        remaining: list[str] = []
        removed = False
        for replay_path in self._selected_multi_replays:
            other_info = self._replay_entity_info(replay_path)
            other_key = other_info[0] if other_info is not None else (_normalize_replay_path(replay_path) or replay_path)
            if other_key == entity_key:
                removed = True
                continue
            remaining.append(replay_path)
        if removed:
            self._selected_multi_replays = remaining
        elif len(remaining) < MAX_MULTI_REPLAYS:
            remaining.append(normalized_path)
            self._selected_multi_replays = remaining
        if self._selected_multi_replays:
            first_summary = self._ensure_replay_summary(self._selected_multi_replays[0])
            self._align_diff_for_summary(first_summary)
            if first_summary is not None and not self._mods_overridden:
                self._active_mods = normalize_mods(first_summary.mods)
        else:
            self._active_mods = 0
            self._mods_overridden = False

    def _build_flattened_list(self) -> list[tuple[int, int]]:
        """Build list of (set_idx, diff_idx) for each visible row.
        Selected mapset with multiple diffs expands; others are collapsed."""
        if not self._flattened_list_dirty:
            return self._flattened_list_cache
        result: list[tuple[int, int]] = []
        for si, bset in enumerate(self._sets):
            if si == self._selected_idx and len(bset.maps) > 1:
                for di in range(len(bset.maps)):
                    result.append((si, di))
            else:
                result.append((si, 0))
        self._flattened_list_cache = result
        self._flattened_list_dirty = False
        return result

    def _mark_flattened_list_dirty(self) -> None:
        self._flattened_list_dirty = True

    def _selection_to_flat_idx(self) -> int:
        """Convert (selected_idx, selected_diff_idx) to flattened list index."""
        flat = 0
        for si, bset in enumerate(self._sets):
            n = len(bset.maps) if (si == self._selected_idx and len(bset.maps) > 1) else 1
            if si == self._selected_idx:
                return flat + (self._selected_diff_idx if len(bset.maps) > 1 else 0)
            flat += n
        return 0

    def _flat_idx_to_selection(self, flat_idx: int) -> tuple[int, int]:
        """Convert flattened list index to (set_idx, diff_idx)."""
        flat = 0
        for si, bset in enumerate(self._sets):
            n = len(bset.maps) if (si == self._selected_idx and len(bset.maps) > 1) else 1
            if flat_idx < flat + n:
                di = (flat_idx - flat) if (si == self._selected_idx and len(bset.maps) > 1) else 0
                return (si, di)
            flat += n
        return (max(0, len(self._sets) - 1), 0)

    def _song_list_metrics(self, w: int, h: int, *, x_offset: float = 0.0) -> tuple[int, int, int, int, int]:
        metrics = self._menu_view.song_list_metrics(self, x_offset=x_offset)
        visible_rect = metrics["visible_rect"]
        return (
            int(metrics["list_x"]),
            int(metrics["card_w"]),
            int(visible_rect.y),
            int(visible_rect.bottom),
            int(metrics["item_h"]),
        )

    def _song_card_x(self, list_x: int, cy: int, visible_top: int, list_h: int,
                     *, set_idx: int, diff_idx: int) -> int:
        scale = self.app.content_layout()["scale"]
        center_y = visible_top + list_h / 2 if list_h else visible_top
        depth = min(1.0, abs((cy + CARD_HEIGHT * scale / 2) - center_y) / max(1.0, list_h / 2))
        card_x = list_x + int(CARD_SKEW * scale * (1.0 - depth))
        if 0 <= set_idx < len(self._sets):
            bset = self._sets[set_idx]
            if set_idx == self._selected_idx and len(bset.maps) > 1:
                card_x += int(CARD_STACK_SHIFT * scale)
        return card_x

    def _center_scroll(self) -> None:
        metrics = self._menu_view.song_list_metrics(self)
        visible_h = metrics["visible_rect"].h
        item_h = metrics["item_h"]
        flat_idx = self._selection_to_flat_idx()
        self._scroll_target = flat_idx * item_h - visible_h / 2 + item_h / 2

    def _replays_for_set(self, bset: BeatmapSet) -> list[str]:
        key = bset.directory
        replay_dir = replay_dir_for_set(bset.directory)
        dir_mtime = replay_dir.stat().st_mtime_ns if replay_dir.is_dir() else None
        if key in self._replay_cache and self._replay_cache_mtime.get(key) == dir_mtime:
            return self._replay_cache[key]
        if not replay_dir.is_dir():
            self._replay_cache[key] = []
            self._replay_cache_mtime[key] = None
            return []
        replays = sorted(
            f.name for f in replay_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".osr"
        )
        self._replay_cache[key] = replays
        self._replay_cache_mtime[key] = dir_mtime
        for replay_name in replays:
            self._queue_replay_summary(str(replay_dir / replay_name))
        return replays

    def _invalidate_replay_cache(self) -> None:
        """Clear replay cache when selection changes or returning from gameplay."""
        self._replay_summary_generation += 1
        self._replay_cache.clear()
        self._replay_cache_mtime.clear()
        self._replay_summary_cache.clear()
        self._replay_summary_loading.clear()
        self._replay_summary_failed.clear()
        self._replay_entity_cache.clear()
        self._replay_entity_index_key = None
        self._replay_entity_index.clear()
        with self._peek_mods_lock:
            self._peek_mods_pending.clear()

    def _open_replays_folder(self, bset: BeatmapSet) -> None:
        replay_dir = replay_dir_for_set(bset.directory)
        replay_dir.mkdir(parents=True, exist_ok=True)
        folder = str(replay_dir.resolve())
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    def _clamp_song_scroll(self) -> None:
        metrics = self._menu_view.song_list_metrics(self)
        flat_len = len(self._build_flattened_list())
        visible_h = metrics["visible_rect"].h
        density = self.app.menu_context().density
        max_scroll = max(0, flat_len * metrics["item_h"] - visible_h)
        self._scroll_target = _clamp(self._scroll_target, -80 * density, max_scroll + 40 * density)

    def _hit_test_song_list(self, x: int, y: int, *, x_offset: float = 0.0) -> int:
        for rx, ry, rw, rh, flat_idx in self._song_card_rects:
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return flat_idx
        return -1

    def _clamp_replay_scroll(self) -> None:
        section_h = self._replay_section_rect[3] if self._replay_section_rect else REPLAY_SECTION_H
        max_s = max(0.0, self._replay_total_h - section_h)
        self._replay_scroll_target = max(0.0, min(self._replay_scroll_target, max_s))

    def _is_in_rect(self, x: int, y: int, rect: tuple) -> bool:
        rx, ry, rw, rh = rect[:4]
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def _update_replay_mods(self) -> None:
        """Refresh the default mods from the current replay selection."""
        primary = self._primary_selected_replay()
        if primary is None:
            return
        self._queue_replay_summary(primary)
        summary = self._replay_summary_cache.get(primary)
        if summary is not None and not self._mods_overridden:
            self._align_diff_for_summary(summary)
            self._active_mods = normalize_mods(summary.mods)

    def _debug_rect(self, x: float, y: float, w: float, h: float,
                    color: tuple[float, float, float, float]) -> None:
        if self._debug_layout:
            self._debug_regions.append((x, y, w, h, color))

    def _draw_toggle_chip(
        self,
        text,
        panels,
        *,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        enabled: bool,
    ) -> tuple[int, int, int, int]:
        bg = (0.34, 0.18, 0.52, 0.96) if enabled else (0.08, 0.08, 0.13, 0.78)
        border = (0.82, 0.64, 0.98, 0.92) if enabled else (0.28, 0.30, 0.40, 0.72)
        text_col = COL_TEXT if enabled else COL_TEXT_MUTED
        panels.draw(x, y, w, h, radius=11.0,
                    color=bg, border_color=border, border_width=1.0)
        tw, _ = text.measure(label, 12)
        text.draw(label, x + (w - tw) / 2, y + 3, 12, color=text_col)
        return int(x), int(y), int(w), int(h)

    def _draw_debug_regions(self, panels) -> None:
        if not self._debug_layout:
            return
        for x, y, w, h, color in self._debug_regions:
            panels.draw(x, y, w, h, radius=0.0,
                        color=(0.0, 0.0, 0.0, 0.0),
                        border_color=color, border_width=1.0)

    def _bottom_bar_y_offset(self) -> float:
        scale = self.app.menu_context().density
        if self._anim_state == "entering":
            progress = min(1.0, self._anim_timer / ENTER_DURATION)
            t = _ease_out_cubic(progress)
            return (1.0 - t) * BOTTOM_BAR_H * scale
        if self._anim_state == "exiting":
            progress = min(1.0, self._anim_timer / EXIT_DURATION)
            t = _ease_in_quad(progress)
            return t * BOTTOM_BAR_H * scale
        return 0.0

    # ----------------------------------------------------------- render

    def on_render(self, time: float, frametime: float) -> None:
        profiler.begin_frame("song_select")
        w, h = self.app.wnd.buffer_size
        ctx = self.app.ctx
        self._debug_regions = []

        try:
            dt = min(frametime, 0.05)
            self._replay_context_menu_anim.update(dt)
            if (
                self._replay_context_menu_anim.target <= 0.001
                and self._replay_context_menu_anim.value <= 0.001
                and self._replay_context_menu_rect is not None
            ):
                self._replay_context_menu_rect = None
                self._replay_context_menu_options = []

            # Apply all pending replay metadata results from background threads.
            self._drain_pending_replay_summaries()
            beatmap_id = self._selected_online_beatmap_id()
            if beatmap_id is not None:
                self.app.social_client.fetch_online_replays(beatmap_id, force=False)

            # Apply pending audio load from background thread (must run on main thread)
            with self._audio_load_lock:
                audio_pending = self._audio_load_pending
                audio_path = self._audio_load_path
                self._audio_load_pending = None
                self._audio_load_path = None
            if audio_pending is not None:
                with profiler.timer("song_select.audio_apply"):
                    buffer, namehint, start_sec = audio_pending
                    expected = (
                        self._sets[self._selected_idx].audio_path
                        if self._sets
                        else None
                    )
                    if expected and audio_path == expected:
                        try:
                            buffer.seek(0)
                            target_volume = self.app.effective_music_volume
                            pygame.mixer.music.stop()
                            pygame.mixer.music.load(buffer, namehint)
                            pygame.mixer.music.set_volume(target_volume)
                            pygame.mixer.music.play(loops=-1, start=start_sec)
                            pygame.mixer.music.set_volume(target_volume)
                            self._preview_playing = True
                        except Exception:
                            self._preview_playing = False

            # Poll for async scan completion
            if not self._sets:
                self._sets = self.app.scanner.sets
                if self._sets:
                    self._selected_idx = 0
                    self._selected_diff_idx = 0
                    self._mark_flattened_list_dirty()
                    self._center_scroll()
                    self._start_preview()

            if self._anim_state == "entering":
                self._anim_timer += dt
                if self._anim_timer >= ENTER_DURATION:
                    self._anim_timer = ENTER_DURATION
                    self._anim_state = "idle"
            elif self._anim_state == "exiting":
                self._anim_timer += dt
                if self._anim_timer >= EXIT_DURATION:
                    self._stop_preview()
                    from scenes.gameplay import GameplayScene
                    self.app.switch_scene(GameplayScene(
                        self.app, self._pending_info,
                        replay_paths=self._pending_replay_paths,
                        include_danser=self._pending_include_danser,
                        multi_replay=self._pending_multi_replay,
                        mods=self._active_mods,
                        mods_overridden=self._pending_mods_overridden,
                    ))
                    return

            info_x_off = 0.0
            cards_x_off = 0.0
            bar_y_off = self._bottom_bar_y_offset()
            dim_alpha = 0.0

            if self._anim_state == "entering":
                progress = min(1.0, self._anim_timer / ENTER_DURATION)
                t = _ease_out_cubic(progress)
                panel_w = int(w * INFO_PANEL_W_FRAC)
                info_x_off = -(1.0 - t) * panel_w
                cards_x_off = (1.0 - t) * w * 0.6
                dim_alpha = (1.0 - t) * 0.6
            elif self._anim_state == "exiting":
                progress = min(1.0, self._anim_timer / EXIT_DURATION)
                t = _ease_in_quad(progress)
                panel_w = int(w * INFO_PANEL_W_FRAC)
                info_x_off = -t * panel_w
                cards_x_off = t * w * 0.6
                dim_alpha = t

            # Scroll lerp + momentum
            self._scroll_current += (self._scroll_target - self._scroll_current) * min(1.0, SCROLL_LERP * dt * 60)
            self._replay_scroll_current += (self._replay_scroll_target - self._replay_scroll_current) * min(1.0, SCROLL_LERP * dt * 60)

            if not self._dragging_songs and abs(self._song_velocity) > MOMENTUM_MIN:
                self._scroll_target += self._song_velocity
                self._song_velocity *= MOMENTUM_DECAY
                self._clamp_song_scroll()
            else:
                self._song_velocity = 0.0

            if not self._dragging_replays and abs(self._replay_velocity) > MOMENTUM_MIN:
                self._replay_scroll_target += self._replay_velocity
                self._replay_velocity *= MOMENTUM_DECAY
                self._clamp_replay_scroll()
            else:
                self._replay_velocity = 0.0

            mods_hovered = (
                (self._mods_trigger_rect is not None and self._is_in_rect(self._mouse_x, self._mouse_y, self._mods_trigger_rect))
                or (self._mods_surface_rect is not None and self._is_in_rect(self._mouse_x, self._mouse_y, self._mods_surface_rect))
            )
            mods_target = 1.0 if (self._mods_palette_locked or mods_hovered) else 0.0
            self._mods_palette_anim += (mods_target - self._mods_palette_anim) * min(1.0, dt * 10.0)
            if mods_target <= 0.0 and self._mods_palette_anim < 0.01:
                self._mods_palette_anim = 0.0

            self._update_sound_slider_animations(dt, y_offset=bar_y_off)

            # Background
            bset = self._sets[self._selected_idx] if self._sets else None
            bg_path = bset.background_path if bset else None
            self.app.backgrounds.configure(
                blur_radius=18,
                dim_factor=0.35,
                motion_enabled=True,
            )
            self.app.backgrounds.load(bg_path)
            self.app.backgrounds.draw(dt)

            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            ctx.disable(moderngl.DEPTH_TEST)

            text = self.app.text
            panels = self.app.panels
            with profiler.timer("song_select.draw_menu_view"):
                self._menu_view.draw(
                    self,
                    info_x_off=info_x_off,
                    cards_x_off=cards_x_off,
                    bar_y_off=bar_y_off,
                )
            self._draw_debug_regions(panels)
            self._last_cards_x_off = cards_x_off

            if dim_alpha > 0.001:
                panels.draw(0, 0, w, h, radius=0.0,
                            color=(0.0, 0.0, 0.0, dim_alpha),
                            border_color=(0, 0, 0, 0), border_width=0)
        finally:
            profiler.end_frame()

    def _draw_song_list(self, w: int, h: int, text, panels,
                        *, x_offset: float = 0.0) -> None:
        scale = self.app.content_layout()["scale"]
        list_x, card_w, visible_top, visible_bot, item_h = self._song_list_metrics(w, h, x_offset=x_offset)
        list_h = max(0, visible_bot - visible_top)
        self._debug_rect(list_x, visible_top, card_w, list_h, (0.25, 0.85, 1.0, 0.85))

        self._songs_open_btn_rect = None
        if self._sets:
            _draw_header_pair(text, "Songs", f"{len(self._sets)} mapsets",
                              list_x + 4, visible_top - 30 * scale,
                              label_color=COL_TEXT_DIM, meta_color=COL_TEXT_MUTED)
        else:
            text.draw("Songs", list_x + 4, visible_top - 30 * scale, SECTION_HEADER_SIZE, color=COL_TEXT_DIM)
        songs_btn_w = max(120.0 * scale, text.measure("Open folder", 12)[0] + 28.0 * scale)
        songs_btn_h = 24.0 * scale
        songs_btn_x = list_x + card_w - songs_btn_w
        songs_btn_y = visible_top - 34.0 * scale
        panels.draw(
            songs_btn_x, songs_btn_y, songs_btn_w, songs_btn_h,
            radius=10.0,
            color=(0.08, 0.08, 0.13, 0.82),
            border_color=(0.28, 0.30, 0.40, 0.72),
            border_width=1.0,
        )
        label = "Open folder"
        lw, _ = text.measure(label, 12)
        text.draw(label, songs_btn_x + (songs_btn_w - lw) * 0.5, songs_btn_y + 4.0 * scale, 12, color=COL_TEXT_MUTED)
        self._songs_open_btn_rect = (int(songs_btn_x), int(songs_btn_y), int(songs_btn_w), int(songs_btn_h))

        self._hover_idx = -1
        flattened = self._build_flattened_list()

        for flat_i, (set_idx, diff_idx) in enumerate(flattened):
            card_h = CARD_HEIGHT * scale
            cy = int(flat_i * item_h - self._scroll_current + visible_top)
            if cy + card_h < visible_top or cy > visible_bot:
                continue

            bset = self._sets[set_idx]
            info = bset.maps[diff_idx] if bset.maps else None
            is_expanded_row = (set_idx == self._selected_idx and len(bset.maps) > 1)
            is_selected = (set_idx == self._selected_idx and diff_idx == self._selected_diff_idx)
            show_badge = len(bset.maps) > 1 and (not is_expanded_row or diff_idx == 0)
            badge_text = str(len(bset.maps)) if show_badge else ""
            badge_w = 0
            if show_badge:
                badge_w = max(int(CARD_BADGE_H * 1.35), int(text.measure(badge_text, CARD_BADGE_TEXT_SIZE)[0] + 24))
            text_max_w = max(80, card_w - CARD_CONTENT_X - (badge_w + 24 if show_badge else 18))
            title = _truncate_text(text, bset.display_title, CARD_TITLE_SIZE, text_max_w)
            artist = _truncate_text(text, bset.display_artist, CARD_ARTIST_SIZE, text_max_w)
            if is_expanded_row and info:
                meta_text = f"[{info.version}]  mapped by {info.creator}"
            elif bset.maps:
                shown = " | ".join(m.version for m in bset.maps[:3])
                extra = f" +{len(bset.maps) - 3}" if len(bset.maps) > 3 else ""
                meta_text = shown + extra
            else:
                meta_text = "No difficulties found"
            meta_text = _truncate_text(text, meta_text, CARD_META_SIZE, text_max_w)

            card_x = self._song_card_x(list_x, cy, visible_top, list_h, set_idx=set_idx, diff_idx=diff_idx)
            is_hovered = (
                not self._dragging_songs
                and self._mouse_x >= card_x and self._mouse_x <= card_x + card_w
                and self._mouse_y >= cy and self._mouse_y <= cy + card_h
            )
            if is_hovered:
                self._hover_idx = flat_i

            if is_selected:
                bg = COL_CARD_SELECTED
                border = COL_CARD_BORDER_SEL
                accent_col = COL_CARD_BORDER_SEL
            elif is_hovered:
                bg = COL_CARD_HOVER
                border = COL_CARD_BORDER
                accent_col = COL_ACCENT_SOFT
            else:
                bg = COL_CARD_BG
                border = COL_CARD_BORDER
                accent_col = COL_ACCENT_SOFT

            panels.draw(card_x, cy, card_w, card_h,
                        radius=CARD_RADIUS, color=bg, border_color=border, border_width=CARD_BORDER_W)

            panels.draw(card_x + 1, cy + 6 * scale, CARD_ACCENT_W, card_h - 12 * scale,
                        radius=CARD_ACCENT_ROUND,
                        color=(accent_col[0], accent_col[1], accent_col[2], 0.85),
                        border_color=(0, 0, 0, 0), border_width=0)

            self._debug_rect(card_x, cy, card_w, card_h, (0.88, 0.38, 0.95, 0.7))
            if show_badge:
                badge_x = card_x + card_w - badge_w - 12
                badge_y = cy + CARD_BADGE_Y - 2
                panels.draw(badge_x, badge_y, badge_w, CARD_BADGE_H, radius=max(6.0, CARD_RADIUS - 2.0),
                            color=(0.08, 0.07, 0.14, 0.92),
                            border_color=(accent_col[0], accent_col[1], accent_col[2], 0.40),
                            border_width=0.8)
                badge_label = _truncate_text(text, badge_text, CARD_BADGE_TEXT_SIZE, badge_w - 12)
                bw, _ = text.measure(badge_label, CARD_BADGE_TEXT_SIZE)
                text.draw(badge_label, badge_x + (badge_w - bw) / 2, badge_y + CARD_BADGE_TEXT_DY + 1,
                          CARD_BADGE_TEXT_SIZE, color=COL_TEXT_MUTED)

            text.draw(title, card_x + CARD_CONTENT_X, cy + CARD_CONTENT_TOP, CARD_TITLE_SIZE, color=COL_TEXT)
            text.draw(artist, card_x + CARD_CONTENT_X, cy + CARD_CONTENT_TOP + CARD_TITLE_TO_ARTIST,
                      CARD_ARTIST_SIZE, color=COL_TEXT_DIM)
            text.draw(meta_text, card_x + CARD_CONTENT_X, cy + CARD_CONTENT_TOP + CARD_TITLE_TO_ARTIST + CARD_ARTIST_TO_META,
                      CARD_META_SIZE,
                      color=COL_TEXT_DIM if is_selected else COL_TEXT_MUTED,
                      alpha=0.95 if is_selected else 0.80)

    def _draw_divider(self, panels, x: float, y: float, w: float) -> None:
        panels.draw_gradient_bar(
            x, y, w, DIVIDER_H,
            spawn_x=x + w,
            fade_width=max(60.0, w * 0.65),
            color=(COL_ACCENT[0], COL_ACCENT[1], COL_ACCENT[2], 0.22),
        )

    def _draw_info_panel(self, w: int, h: int, text, panels,
                         *, x_offset: float = 0.0) -> None:
        content = self.app.content_layout()
        scale = content["scale"]
        panel_w = int(content["w"] * INFO_PANEL_W_FRAC) - int(PANEL_MARGIN * scale)
        panel_h = int(content["h"] - (BOTTOM_BAR_H + PANEL_MARGIN + BOTTOM_SECTION_GAP) * scale)
        panel_x = int(content["x"] + x_offset + PANEL_MARGIN * scale)
        panel_y = int(content["y"] + PANEL_MARGIN * scale)
        self._debug_rect(panel_x, panel_y, panel_w, panel_h, (0.25, 1.0, 0.35, 0.85))

        panels.draw(panel_x, panel_y, panel_w, panel_h, radius=PANEL_RADIUS,
                    color=COL_INFO_BG, border_color=(0.0, 0.0, 0.0, 0.0), border_width=0.0)

        info = self._selected_info()
        bset = self._sets[self._selected_idx] if self._sets else None

        if info is None:
            if not self.app.scanner.scan_complete:
                text.draw("Loading maps...", panel_x + INFO_PAD, panel_y + 34, 26, color=COL_TEXT)
            else:
                text.draw("No maps found", panel_x + INFO_PAD, panel_y + 34, 26, color=COL_TEXT)
                text.draw(f"Place mapset folders in {MAPS_DIR.name}/",
                          panel_x + INFO_PAD, panel_y + 68, 14, color=COL_TEXT_DIM)
            return

        lx = panel_x + INFO_PAD
        lw = panel_w - INFO_PAD * 2
        y = panel_y + INFO_PAD

        # -- Song title + inline stats --
        title_raw = info.title_unicode or info.title
        artist_raw = info.artist_unicode or info.artist

        stats_text = f"CS {info.cs:.1f}  AR {info.ar:.1f}  OD {info.od:.1f}  HP {info.hp:.1f}"
        stats_w, _ = text.measure(stats_text, INFO_STATS_SIZE)
        stats_x = lx + max(0.0, lw - stats_w)
        title_max_w = max(120.0, stats_x - lx - INFO_TITLE_STATS_GAP)
        title = _truncate_text(text, title_raw, TITLE_SIZE, title_max_w)
        text.draw(title, lx, y, TITLE_SIZE, color=COL_TEXT)
        text.draw(stats_text, stats_x, y + INFO_TOP_ROW_INSET, INFO_STATS_SIZE, color=COL_TEXT_MUTED)

        diff_label = _truncate_text(text, f"[{info.version}]", INFO_DIFF_SIZE, stats_w)
        diff_w, _ = text.measure(diff_label, INFO_DIFF_SIZE)
        text.draw(diff_label, stats_x + max(0.0, stats_w - diff_w), y + INFO_TOP_ROW_INSET + INFO_STATS_SIZE + 3,
                  INFO_DIFF_SIZE, color=COL_TEXT_DIM)
        top_row_h = max(TITLE_SIZE, INFO_TOP_ROW_INSET + INFO_STATS_SIZE + 3 + INFO_DIFF_SIZE)
        y += top_row_h + INFO_META_GAP

        artist = _truncate_text(text, artist_raw, ARTIST_SIZE, lw)
        text.draw(artist, lx, y, ARTIST_SIZE, color=COL_TEXT_DIM)
        y += ARTIST_SIZE + 3

        mapper = _truncate_text(text, f"Mapped by {info.creator}", MAPPER_SIZE, lw)
        text.draw(mapper, lx, y, MAPPER_SIZE, color=COL_TEXT_MUTED)
        y += MAPPER_SIZE + DIVIDER_PAD_TOP

        self._draw_divider(panels, lx, y, lw)
        y += DIVIDER_H + DIVIDER_PAD_BOT

        # -- Replays --
        all_replays = self._replays_for_set(bset) if bset else []
        replays = self._visible_replays_for_set(bset) if bset else []
        replay_meta = f"{len(replays)} shown"
        if len(replays) != len(all_replays):
            replay_meta = f"{len(replays)} shown / {len(all_replays)} saved"
        _draw_header_pair(text, "Replays", replay_meta,
                          lx, y, label_color=COL_TEXT_DIM, meta_color=COL_TEXT_MUTED)
        toggle_y = y - 3
        toggle_x = lx + lw - REPLAY_TOGGLE_W
        self._multi_toggle_rect = self._draw_toggle_chip(
            text, panels,
            x=toggle_x,
            y=toggle_y,
            w=REPLAY_TOGGLE_W,
            h=REPLAY_TOGGLE_H,
            label="Multi" if self._multi_replay_enabled else "Single",
            enabled=self._multi_replay_enabled,
        )
        if self._multi_replay_enabled:
            toggle_x -= REPLAY_TOGGLE_W + REPLAY_TOGGLE_GAP
            self._danser_toggle_rect = self._draw_toggle_chip(
                text, panels,
                x=toggle_x,
                y=toggle_y,
                w=REPLAY_TOGGLE_W,
                h=REPLAY_TOGGLE_H,
                label="Danser",
                enabled=self._danser_replay_enabled,
            )
        else:
            self._danser_toggle_rect = None
        y += SECTION_HEADER_SIZE + REPLAY_HEADER_GAP

        footer_h = int(28 * scale)
        button_y = panel_y + panel_h - footer_h
        section_h = max(60, button_y - y - 4)
        section_x = lx
        section_y = y
        section_w = lw
        self._replay_section_rect = (section_x, section_y, section_w, section_h)
        self._clamp_replay_scroll()
        self._debug_rect(section_x, section_y, section_w, section_h, (1.0, 0.65, 0.25, 0.85))

        self._replay_rects = []

        panels.draw(section_x, section_y, section_w, section_h,
                    radius=10.0,
                    color=(0.035, 0.035, 0.075, 0.55),
                    border_color=(0.18, 0.16, 0.28, 0.30), border_width=0.8)

        with profiler.timer("song_select.replay_section"):
            text.end()

            ctx = self.app.ctx
            _, wnd_h = self.app.wnd.buffer_size
            scx = max(0, section_x)
            scy = max(0, wnd_h - (section_y + section_h))
            scw = max(1, section_w)
            sch = max(1, section_h)
            old_scissor = ctx.scissor
            ctx.scissor = (scx, scy, scw, sch)

            text.begin()

            if replays:
                replay_dir = replay_dir_for_set(bset.directory)
                full_paths = [str(replay_dir / rp) for rp in replays]
                label_counts: dict[str, int] = {}
                for rp, full_path in zip(replays, full_paths):
                    base_label = self._replay_base_label(rp, full_path)
                    label_counts[base_label] = label_counts.get(base_label, 0) + 1
                duplicate_labels = {
                    label for label, count in label_counts.items()
                    if count > 1
                }
                row_h = REPLAY_ROW_H
                row_gap = REPLAY_ROW_GAP
                ry = section_y + REPLAY_TOP_PAD - int(self._replay_scroll_current)
                for rp, full_path in zip(replays, full_paths):
                    is_sel = full_path in self._effective_selected_replays()
                    row_color = (0.12, 0.09, 0.19, 0.92) if is_sel else (0.06, 0.055, 0.10, 0.70)
                    row_border = (COL_ACCENT[0], COL_ACCENT[1], COL_ACCENT[2], 0.35) if is_sel else (0, 0, 0, 0)
                    panels.draw(section_x + 4, ry, section_w - 8, row_h,
                                radius=8.0, color=row_color, border_color=row_border, border_width=0.8)
                    primary_text, secondary_text = self._replay_row_labels(rp, full_path, duplicate_labels)
                    primary = _truncate_text(text, primary_text, CONTROL_TEXT_SIZE, section_w - 28)
                    secondary = _truncate_text(text, secondary_text, 11, section_w - 28)
                    text.draw(primary, section_x + 14, ry + 6, CONTROL_TEXT_SIZE,
                              color=COL_TEXT if is_sel else COL_TEXT_DIM)
                    text.draw(secondary, section_x + 14, ry + 24, 11,
                              color=COL_TEXT_MUTED, alpha=0.82)
                    self._replay_rects.append((section_x + 4, ry, section_w - 8, row_h, full_path))
                    ry += row_h + row_gap
                self._replay_total_h = REPLAY_TOP_PAD + len(replays) * row_h + max(0, len(replays) - 1) * row_gap + REPLAY_BOTTOM_PAD
            else:
                text.draw("No replays yet", section_x + 12, section_y + 14, 14, color=COL_TEXT_MUTED)
                text.draw("Drop .osr files into the mapset's replay folder.",
                          section_x + 12, section_y + 34, 12, color=COL_TEXT_MUTED, alpha=0.65)
                self._replay_total_h = 0.0

            text.end()
            ctx.scissor = old_scissor
            text.begin()

        # -- Footer: open folder link --
        link_label = "Open replays folder"
        link_w, _ = text.measure(link_label, 12)
        text.draw(link_label, lx + (lw - link_w) / 2, button_y + 4, 12, color=COL_TEXT_MUTED)
        self._open_btn_rect = (lx, button_y - 4, lw, footer_h + 4)

    def _draw_bottom_bar(self, w: int, h: int, text, panels,
                         *, y_offset: float = 0.0) -> None:
        content = self.app.content_layout()
        scale = content["scale"]
        bar_h = int(BOTTOM_BAR_H * scale)
        bar_x = int(content["x"])
        bar_w = int(content["w"])
        bar_y = int(content["y"] + content["h"] - bar_h + y_offset)
        self._debug_rect(bar_x, bar_y, bar_w, bar_h, (1.0, 0.25, 0.25, 0.85))
        panels.draw(bar_x, bar_y, bar_w, bar_h, radius=0.0,
                    color=COL_BOTTOM_BG, border_color=(0, 0, 0, 0), border_width=0)

        panels.draw_gradient_bar(
            bar_x, bar_y, bar_w, 2,
            spawn_x=float(bar_x + bar_w),
            fade_width=max(200.0 * scale, bar_w * 0.5),
            color=(COL_ACCENT[0], COL_ACCENT[1], COL_ACCENT[2], 0.18),
        )

        play_button_w = PLAY_BUTTON_W * scale
        play_h = max(24.0 * scale, round(float(MOD_CHIP_SIZE) * BOTTOM_SQUARE_SCALE * scale))
        play_y = bar_y + (bar_h - play_h) * 0.5
        play_x = bar_x + bar_w - play_button_w - PANEL_MARGIN * scale

        mods_x = bar_x + PANEL_MARGIN * scale
        mods_limit_x = play_x - SECTION_GAP * scale

        self._mod_rects = []
        mx = mods_x
        chip_sz = int(round(play_h))
        chip_y = play_y + (play_h - chip_sz) / 2
        chip_gap = MOD_CHIP_GAP

        for label in _MOD_LABELS:
            flag = MOD_FLAG_MAP.get(label, 0)
            short = MOD_SHORT.get(flag, label[:2].upper())
            is_active = bool(self._active_mods & flag)
            is_fl = (flag == FL)

            if is_fl:
                bg_col = (0.07, 0.07, 0.10, 0.50)
                border_col = (0.18, 0.18, 0.24, 0.30)
                text_col = (0.32, 0.32, 0.40)
            elif is_active:
                bg_col = COL_MOD_ACTIVE
                border_col = COL_MOD_BORDER_ACTIVE
                text_col = (0.10, 0.07, 0.16)
            else:
                bg_col = COL_MOD_INACTIVE
                border_col = COL_MOD_BORDER_INACTIVE
                text_col = COL_TEXT_DIM

            if mx + chip_sz > mods_limit_x:
                break

            panels.draw(mx, chip_y, chip_sz, chip_sz, radius=8.0,
                        color=bg_col, border_color=border_col, border_width=1.0)
            tw, _ = text.measure(short, MOD_CHIP_TEXT_SIZE)
            text_dy = (chip_sz - MOD_CHIP_TEXT_SIZE) // 2 - 4
            text.draw(short, mx + 2 + (chip_sz - tw) / 2, chip_y + text_dy,
                      MOD_CHIP_TEXT_SIZE, color=text_col)
            self._mod_rects.append((mx, chip_y, chip_sz, chip_sz, flag))
            mx += chip_sz + chip_gap

        summary = mod_string(self._active_mods) or "No mod"
        if self._multi_replay_enabled:
            summary = f"{self._selected_replay_count()} selected  {summary}"
        summary_x = mx + 12
        summary_max_w = max(60, mods_limit_x - summary_x)
        summary = _truncate_text(text, summary, BAR_LABEL_SIZE, summary_max_w)
        text.draw(summary, summary_x, play_y + (play_h - BAR_LABEL_SIZE) / 2 - 4, BAR_LABEL_SIZE,
                  color=COL_TEXT_MUTED)

        play_hover = (
            self._mouse_x >= play_x and self._mouse_x <= play_x + play_button_w
            and self._mouse_y >= play_y and self._mouse_y <= play_y + play_h
        )
        play_col = (0.42, 0.18, 0.58, 0.98) if play_hover else (0.30, 0.14, 0.46, 0.98)
        panels.draw(play_x, play_y, play_button_w, play_h, radius=SECTION_RADIUS,
                    color=play_col,
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0)

        play_text_w, _ = text.measure("PLAY", PLAY_TITLE_SIZE)
        center_y = play_y + (play_h - PLAY_TITLE_SIZE) / 2 - 4
        text.draw("PLAY", play_x + (play_button_w - play_text_w) / 2, center_y,
                  PLAY_TITLE_SIZE, color=COL_TEXT)
        self._play_btn_rect = (play_x, play_y, play_button_w, play_h)

    # ----------------------------------------------------------- input

    def on_key_event(self, key, action, modifiers) -> None:
        keys = self.app.wnd.keys
        if action != keys.ACTION_PRESS:
            return

        if key == keys.ESCAPE:
            if self._mods_palette_locked:
                self._mods_palette_locked = False
                return
            self.app.wnd.close()
        elif key == keys.UP:
            self._move_diff(-1)
        elif key == keys.DOWN:
            self._move_diff(1)
        elif key == keys.LEFT:
            self._move_mapset(-1)
        elif key == keys.RIGHT:
            self._move_mapset(1)
        elif key == keys.ENTER:
            self._play_selected()

    def on_mouse_press(self, x: int, y: int, button: int) -> None:
        if button in {2, 4}:
            if self._is_in_rect(x, y, self._replay_section_rect):
                for ax, ay, aw, ah, token in self._replay_action_rects:
                    if ax <= x <= ax + aw and ay <= y <= ay + ah:
                        self._open_replay_context_menu(token, x, y)
                        return
                for rx, ry, rw, rh, token in self._replay_rects:
                    if rx <= x <= rx + rw and ry <= y <= ry + rh:
                        self._open_replay_context_menu(token, x, y)
                        return
            return
        if button != 1:
            return

        bar_y_off = self._bottom_bar_y_offset()

        if self._replay_context_menu_rect is not None:
            mx, my, mw, mh = self._replay_context_menu_rect
            if mx <= x <= mx + mw and my <= y <= my + mh:
                for ox, oy, ow, oh, action in self._replay_context_menu_options:
                    if ox <= x <= ox + ow and oy <= y <= oy + oh and self._replay_context_target:
                        self._activate_replay_context_action(action, self._replay_context_target)
                        self._close_replay_context_menu()
                        return
            else:
                self._close_replay_context_menu()

        clicked_mods_trigger = self._mods_trigger_rect and self._is_in_rect(x, y, self._mods_trigger_rect)
        clicked_mods_surface = self._mods_surface_rect and self._is_in_rect(x, y, self._mods_surface_rect)
        if clicked_mods_trigger:
            self._mods_palette_locked = not self._mods_palette_locked
            return
        if self._mods_palette_locked and not clicked_mods_surface:
            self._mods_palette_locked = False

        # Check play button
        if hasattr(self, "_play_btn_rect"):
            px, py, pw, ph = self._play_btn_rect
            if px <= x <= px + pw and py <= y <= py + ph:
                self._play_selected()
                return

        if hasattr(self, "_settings_btn_rect"):
            sx, sy, sw, sh = self._settings_btn_rect
            if sx <= x <= sx + sw and sy <= y <= sy + sh:
                self.app.toggle_settings()
                return

        if self._songs_open_btn_rect and self._is_in_rect(x, y, self._songs_open_btn_rect):
            self.app.open_maps_folder()
            return

        if self._replay_local_tab_rect and self._is_in_rect(x, y, self._replay_local_tab_rect):
            self._set_replay_source_tab("local")
            return
        if self._replay_online_tab_rect and self._is_in_rect(x, y, self._replay_online_tab_rect):
            self._set_replay_source_tab("online")
            return

        if self._multi_toggle_rect and self._is_in_rect(x, y, self._multi_toggle_rect):
            self._toggle_multi_replay()
            return
        if self._danser_toggle_rect and self._is_in_rect(x, y, self._danser_toggle_rect):
            self._toggle_danser_replay()
            return

        for ax, ay, aw, ah, token in self._replay_action_rects:
            if ax <= x <= ax + aw and ay <= y <= ay + ah:
                self._open_replay_context_menu(token, ax, ay + ah)
                return

        # Check mod button clicks
        for mx, my, mw, mh, flag in self._mod_rects:
            if mx <= x <= mx + mw and my <= y <= my + mh:
                if flag == FL:
                    return
                if self._active_mods & flag:
                    self._active_mods &= ~flag
                else:
                    self._active_mods &= ~incompatible_with(flag)
                    self._active_mods |= flag
                self._active_mods = normalize_mods(self._active_mods)
                self._mods_overridden = True
                self._mods_palette_locked = True
                return

        # Check if click is in replay section for drag
        if self._is_in_rect(x, y, self._replay_section_rect):
            self._dragging_replays = True
            self._replay_drag_start_y = y
            self._replay_drag_accum = 0.0
            self._replay_velocity = 0.0

            # Check replay entry clicks (only if not dragging yet -- will confirm on release)
            return

        # Check open replays button
        if hasattr(self, "_open_btn_rect") and self._sets:
            ox, oy, ow, oh = self._open_btn_rect
            if ox <= x <= ox + ow and oy <= y <= oy + oh:
                if self._replay_source_tab == "online":
                    beatmap_id = self._selected_online_beatmap_id()
                    if beatmap_id is not None:
                        self.app.social_client.fetch_online_replays(beatmap_id, force=True)
                else:
                    bset = self._sets[self._selected_idx]
                    self._open_replays_folder(bset)
                return

        # Song list area: start drag
        if self._song_list_interact_rect and self._song_list_interact_rect.contains(x, y):
            self._dragging_songs = True
            self._drag_start_y = y
            self._drag_accum = 0.0
            self._song_velocity = 0.0
            return

    def on_mouse_release(self, x: int, y: int, button: int) -> None:
        if button != 1:
            return

        was_dragging_songs = self._dragging_songs
        was_dragging_replays = self._dragging_replays
        song_drag_dist = abs(self._drag_accum)
        replay_drag_dist = abs(self._replay_drag_accum)

        self._dragging_songs = False
        self._dragging_replays = False

        # If releasing after a replay section click (no significant drag) -> handle click
        if was_dragging_replays and replay_drag_dist < DRAG_THRESHOLD:
            for rx, ry, rw, rh, rpath in self._replay_rects:
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    if rpath.startswith("online:"):
                        self._handle_online_replay_activate(rpath.split(":", 1)[1])
                        return
                    if self._multi_replay_enabled:
                        self._toggle_multi_selected_replay(rpath)
                    else:
                        self._toggle_single_selected_replay(rpath)
                    return
            return

        # If releasing after a song list click (no significant drag) -> handle click
        if was_dragging_songs and song_drag_dist < DRAG_THRESHOLD:
            flat_idx = self._hit_test_song_list(x, y, x_offset=self._last_cards_x_off)
            if flat_idx >= 0:
                set_idx, diff_idx = self._flat_idx_to_selection(flat_idx)
                is_same = (set_idx == self._selected_idx and diff_idx == self._selected_diff_idx)
                if is_same:
                    self._play_selected()
                else:
                    self._apply_selected_indices(set_idx, diff_idx)

    def on_mouse_scroll(self, x_offset: float, y_offset: float) -> None:
        if self._mods_surface_rect and self._is_in_rect(self._mouse_x, self._mouse_y, self._mods_surface_rect):
            return
        if self._is_in_rect(self._mouse_x, self._mouse_y, self._replay_section_rect):
            self._replay_scroll_target -= y_offset * SCROLL_SPEED * 0.5
            self._clamp_replay_scroll()
        else:
            self._scroll_target -= y_offset * SCROLL_SPEED
            self._clamp_song_scroll()

    def on_mouse_move(self, x: int, y: int, dx: int, dy: int) -> None:
        self._mouse_x = x
        self._mouse_y = y

        if self._dragging_songs:
            delta = y - self._drag_start_y
            self._scroll_target -= delta
            self._drag_accum += delta
            self._song_velocity = -delta * 0.5
            self._drag_start_y = y
            self._clamp_song_scroll()

        if self._dragging_replays:
            delta = y - self._replay_drag_start_y
            self._replay_scroll_target -= delta
            self._replay_drag_accum += delta
            self._replay_velocity = -delta * 0.5
            self._replay_drag_start_y = y
            self._clamp_replay_scroll()

    # ----------------------------------------------------------- navigation

    def _move_selection(self, delta: int) -> None:
        if not self._sets:
            return
        flattened = self._build_flattened_list()
        flat_idx = self._selection_to_flat_idx()
        new_flat = max(0, min(flat_idx + delta, len(flattened) - 1))
        set_idx, diff_idx = self._flat_idx_to_selection(new_flat)
        self._apply_selected_indices(set_idx, diff_idx)

    def _move_mapset(self, delta: int) -> None:
        if not self._sets or delta == 0:
            return
        new_set_idx = max(0, min(self._selected_idx + delta, len(self._sets) - 1))
        if new_set_idx == self._selected_idx:
            return
        self._apply_selected_indices(new_set_idx, 0)

    def _move_diff(self, delta: int) -> None:
        self._move_selection(delta)

    def _play_selected(self) -> None:
        if self._anim_state == "exiting":
            return
        info = self._selected_info()
        if info is None:
            return
        replay_paths = self._effective_selected_replays()
        include_danser = self._multi_replay_enabled and self._danser_replay_enabled
        if self._multi_replay_enabled and not replay_paths and not include_danser:
            return
        self._pending_info = info
        self._pending_replay_paths = replay_paths
        self._pending_include_danser = include_danser
        self._pending_multi_replay = self._multi_replay_enabled
        self._pending_mods_overridden = self._mods_overridden
        self._anim_state = "exiting"
        self._anim_timer = 0.0
        if self._preview_playing:
            try:
                pygame.mixer.music.fadeout(int(EXIT_DURATION * 1000))
            except Exception:
                pass
