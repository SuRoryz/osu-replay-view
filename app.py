"""Main application window -- manages scenes and shared resources."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time as time_module
from dataclasses import asdict, dataclass, field
from pathlib import Path

import moderngl_window as mglw
import pyglet

from build_version import get_display_version
from osu_map.scanner import BeatmapScanner
from runtime_paths import APP_ICON_PATH, APP_ROOT, MAPS_DIR, SETTINGS_PATH, ensure_runtime_dirs
from skins import SKIN_REGISTRY
from skins.default import DefaultSkinVisualSettings, make_default_skin_visual_settings
from social import SocialClient
from social.models import ChatMessagePayload, SharedReplay
from ui.alert_overlay import AlertOverlay
from ui.background import BackgroundRenderer
from ui.elements import PanelRenderer
from ui.menu import SettingsOverlay, SocialOverlay, build_layout_context
from ui.text import TextRenderer


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _shape_volume(value: float, exponent: float) -> float:
    value = _clamp01(value)
    return value ** exponent


def _clamp_rgb(value, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return tuple(_clamp01(channel) for channel in value)
        except (TypeError, ValueError):
            pass
    return default


BUILD_VERSION = get_display_version()
DESIGN_WIDTH = 1920.0
DESIGN_HEIGHT = 1080.0
COMMON_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (1024, 768),
    (1280, 960),
    (1400, 1050),
    (1280, 720),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (2560, 1080),
    (3440, 1440),
)
SCREEN_MODES = ("windowed", "borderless", "fullscreen")
FPS_LIMIT_OPTIONS = (0, 60, 120, 144, 180)
_DEFAULT_SKIN_VISUALS = make_default_skin_visual_settings()


@dataclass(slots=True)
class AppSettings:
    music_volume: float = 0.35
    music_muted: bool = False
    sfx_volume: float = 1.0
    sfx_muted: bool = False
    nickname: str = ""
    screen_mode: str = "windowed"
    resolution_width: int = 1920
    resolution_height: int = 1080
    fps_limit: int = 0
    draw_gameplay_timeline: bool = True
    draw_gameplay_acc_pp: bool = True
    draw_gameplay_leaderboard: bool = True
    draw_gameplay_combo: bool = True
    draw_gameplay_hp: bool = True
    draw_gameplay_keys: bool = True
    gameplay_background_bloom: bool = True
    gameplay_background_image: bool = False
    gameplay_background_dim: float = 0.65
    gameplay_cursor_trail: bool = True
    gameplay_cursor_trail_max_len: int = 256
    gameplay_circle_bloom: float = 0.50
    skin_circle_fill_color: tuple[float, float, float] = _DEFAULT_SKIN_VISUALS.circle_fill_color
    skin_circle_fill_opacity: float = _DEFAULT_SKIN_VISUALS.circle_fill_opacity
    skin_circle_border_color: tuple[float, float, float] = _DEFAULT_SKIN_VISUALS.circle_border_color
    skin_circle_border_width: float = _DEFAULT_SKIN_VISUALS.circle_border_width
    skin_circle_bloom: float = _DEFAULT_SKIN_VISUALS.circle_bloom
    skin_circle_bloom_color: tuple[float, float, float] = _DEFAULT_SKIN_VISUALS.circle_bloom_color
    skin_slider_use_circle_head: bool = _DEFAULT_SKIN_VISUALS.slider_use_circle_head
    skin_slider_head_fill_color: tuple[float, float, float] = _DEFAULT_SKIN_VISUALS.slider_head_fill_color
    skin_slider_head_fill_opacity: float = _DEFAULT_SKIN_VISUALS.slider_head_fill_opacity
    skin_slider_head_border_color: tuple[float, float, float] = _DEFAULT_SKIN_VISUALS.slider_head_border_color
    skin_slider_head_border_width: float = _DEFAULT_SKIN_VISUALS.slider_head_border_width
    skin_slider_head_bloom: float = _DEFAULT_SKIN_VISUALS.slider_head_bloom
    skin_slider_head_bloom_color: tuple[float, float, float] = _DEFAULT_SKIN_VISUALS.slider_head_bloom_color
    skin_slider_path_fill_color: tuple[float, float, float] = _DEFAULT_SKIN_VISUALS.slider_path_fill_color
    skin_slider_path_fill_opacity: float = _DEFAULT_SKIN_VISUALS.slider_path_fill_opacity
    skin_slider_path_border_color: tuple[float, float, float] = _DEFAULT_SKIN_VISUALS.slider_path_border_color
    skin_slider_path_border_width: float = _DEFAULT_SKIN_VISUALS.slider_path_border_width
    skin_slider_ball_fill_color: tuple[float, float, float] = _DEFAULT_SKIN_VISUALS.slider_ball_fill_color
    skin_slider_ball_fill_opacity: float = _DEFAULT_SKIN_VISUALS.slider_ball_fill_opacity
    skin_slider_ball_border_color: tuple[float, float, float] = _DEFAULT_SKIN_VISUALS.slider_ball_border_color
    skin_slider_ball_border_width: float = _DEFAULT_SKIN_VISUALS.slider_ball_border_width
    skin_slider_ball_bloom: float = _DEFAULT_SKIN_VISUALS.slider_ball_bloom
    skin_slider_ball_bloom_color: tuple[float, float, float] = _DEFAULT_SKIN_VISUALS.slider_ball_bloom_color
    skin_cursor_color: tuple[float, float, float] = _DEFAULT_SKIN_VISUALS.cursor_color
    skin_cursor_size: float = _DEFAULT_SKIN_VISUALS.cursor_size

    @classmethod
    def load(cls, path: Path) -> "AppSettings":
        defaults = cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
            return defaults
        return cls(
            music_volume=_clamp01(data.get("music_volume", defaults.music_volume)),
            music_muted=bool(data.get("music_muted", defaults.music_muted)),
            sfx_volume=_clamp01(data.get("sfx_volume", defaults.sfx_volume)),
            sfx_muted=bool(data.get("sfx_muted", defaults.sfx_muted)),
            nickname=str(data.get("nickname", defaults.nickname))[:16],
            screen_mode=(
                str(data.get("screen_mode", defaults.screen_mode))
                if str(data.get("screen_mode", defaults.screen_mode)) in SCREEN_MODES
                else defaults.screen_mode
            ),
            resolution_width=max(640, int(data.get("resolution_width", defaults.resolution_width))),
            resolution_height=max(480, int(data.get("resolution_height", defaults.resolution_height))),
            fps_limit=(
                int(data.get("fps_limit", defaults.fps_limit))
                if int(data.get("fps_limit", defaults.fps_limit)) in FPS_LIMIT_OPTIONS
                else defaults.fps_limit
            ),
            draw_gameplay_timeline=bool(data.get("draw_gameplay_timeline", defaults.draw_gameplay_timeline)),
            draw_gameplay_acc_pp=bool(data.get("draw_gameplay_acc_pp", defaults.draw_gameplay_acc_pp)),
            draw_gameplay_leaderboard=bool(data.get("draw_gameplay_leaderboard", defaults.draw_gameplay_leaderboard)),
            draw_gameplay_combo=bool(data.get("draw_gameplay_combo", defaults.draw_gameplay_combo)),
            draw_gameplay_hp=bool(data.get("draw_gameplay_hp", defaults.draw_gameplay_hp)),
            draw_gameplay_keys=bool(data.get("draw_gameplay_keys", defaults.draw_gameplay_keys)),
            gameplay_background_bloom=bool(data.get("gameplay_background_bloom", defaults.gameplay_background_bloom)),
            gameplay_background_image=bool(data.get("gameplay_background_image", defaults.gameplay_background_image)),
            gameplay_background_dim=_clamp01(data.get("gameplay_background_dim", defaults.gameplay_background_dim)),
            gameplay_cursor_trail=bool(data.get("gameplay_cursor_trail", defaults.gameplay_cursor_trail)),
            gameplay_cursor_trail_max_len=max(8, min(256, int(data.get("gameplay_cursor_trail_max_len", defaults.gameplay_cursor_trail_max_len)))),
            gameplay_circle_bloom=_clamp01(data.get("gameplay_circle_bloom", defaults.gameplay_circle_bloom)),
            skin_circle_fill_color=_clamp_rgb(data.get("skin_circle_fill_color", defaults.skin_circle_fill_color), defaults.skin_circle_fill_color),
            skin_circle_fill_opacity=_clamp01(data.get("skin_circle_fill_opacity", defaults.skin_circle_fill_opacity)),
            skin_circle_border_color=_clamp_rgb(data.get("skin_circle_border_color", defaults.skin_circle_border_color), defaults.skin_circle_border_color),
            skin_circle_border_width=_clamp01(data.get("skin_circle_border_width", defaults.skin_circle_border_width)),
            skin_circle_bloom=_clamp01(data.get("skin_circle_bloom", defaults.skin_circle_bloom)),
            skin_circle_bloom_color=_clamp_rgb(data.get("skin_circle_bloom_color", defaults.skin_circle_bloom_color), defaults.skin_circle_bloom_color),
            skin_slider_use_circle_head=bool(data.get("skin_slider_use_circle_head", defaults.skin_slider_use_circle_head)),
            skin_slider_head_fill_color=_clamp_rgb(data.get("skin_slider_head_fill_color", defaults.skin_slider_head_fill_color), defaults.skin_slider_head_fill_color),
            skin_slider_head_fill_opacity=_clamp01(data.get("skin_slider_head_fill_opacity", defaults.skin_slider_head_fill_opacity)),
            skin_slider_head_border_color=_clamp_rgb(data.get("skin_slider_head_border_color", defaults.skin_slider_head_border_color), defaults.skin_slider_head_border_color),
            skin_slider_head_border_width=_clamp01(data.get("skin_slider_head_border_width", defaults.skin_slider_head_border_width)),
            skin_slider_head_bloom=_clamp01(data.get("skin_slider_head_bloom", defaults.skin_slider_head_bloom)),
            skin_slider_head_bloom_color=_clamp_rgb(data.get("skin_slider_head_bloom_color", defaults.skin_slider_head_bloom_color), defaults.skin_slider_head_bloom_color),
            skin_slider_path_fill_color=_clamp_rgb(data.get("skin_slider_path_fill_color", defaults.skin_slider_path_fill_color), defaults.skin_slider_path_fill_color),
            skin_slider_path_fill_opacity=_clamp01(data.get("skin_slider_path_fill_opacity", defaults.skin_slider_path_fill_opacity)),
            skin_slider_path_border_color=_clamp_rgb(data.get("skin_slider_path_border_color", defaults.skin_slider_path_border_color), defaults.skin_slider_path_border_color),
            skin_slider_path_border_width=_clamp01(data.get("skin_slider_path_border_width", defaults.skin_slider_path_border_width)),
            skin_slider_ball_fill_color=_clamp_rgb(data.get("skin_slider_ball_fill_color", defaults.skin_slider_ball_fill_color), defaults.skin_slider_ball_fill_color),
            skin_slider_ball_fill_opacity=_clamp01(data.get("skin_slider_ball_fill_opacity", defaults.skin_slider_ball_fill_opacity)),
            skin_slider_ball_border_color=_clamp_rgb(data.get("skin_slider_ball_border_color", defaults.skin_slider_ball_border_color), defaults.skin_slider_ball_border_color),
            skin_slider_ball_border_width=_clamp01(data.get("skin_slider_ball_border_width", defaults.skin_slider_ball_border_width)),
            skin_slider_ball_bloom=_clamp01(data.get("skin_slider_ball_bloom", defaults.skin_slider_ball_bloom)),
            skin_slider_ball_bloom_color=_clamp_rgb(data.get("skin_slider_ball_bloom_color", defaults.skin_slider_ball_bloom_color), defaults.skin_slider_ball_bloom_color),
            skin_cursor_color=_clamp_rgb(data.get("skin_cursor_color", defaults.skin_cursor_color), defaults.skin_cursor_color),
            skin_cursor_size=_clamp01(data.get("skin_cursor_size", defaults.skin_cursor_size)),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass(slots=True)
class _ReplayBatchDownload:
    beatmap_id: int
    replay_ids: list[str]
    target_dir: str
    requested_replays: list[SharedReplay] = field(default_factory=list)
    current_index: int = 0
    started_current: bool = False
    selected_paths: list[str] = field(default_factory=list)


class App(mglw.WindowConfig):
    gl_version = (4, 3)
    window_size = (1920, 1080)
    aspect_ratio = None
    title = "osu! replay"
    vsync = False
    samples = 4
    clear_color = (0.02, 0.02, 0.06, 1.0)
    settings_path = SETTINGS_PATH

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ensure_runtime_dirs()
        self._apply_window_icon()
        self.wnd.exit_key = None
        self.wnd.fixed_aspect_ratio = None
        self.settings = AppSettings.load(self.settings_path)
        self._apply_skin_visual_settings(notify_scene=False)
        self._menu_context_cache = None
        self._last_buffer_size = tuple(int(v) for v in self.wnd.buffer_size)
        self._pending_framebuffer_sync = False
        self._restart_pending = False
        self._restart_spawned = False
        self._settings_button_rect: tuple[float, float, float, float] | None = None
        self._settings_button_visible = False
        self._mouse_cursor_style = "default"
        self._last_frame_limit_tick: float | None = None
        self._windowed_size = (
            int(self.settings.resolution_width),
            int(self.settings.resolution_height),
        )

        self.text = TextRenderer(self.ctx)
        self.panels = PanelRenderer(self.ctx)
        self.backgrounds = BackgroundRenderer(self.ctx)
        self.scanner = BeatmapScanner(str(MAPS_DIR), scan_immediately=False)
        scan_thread = threading.Thread(target=self.scanner.scan, daemon=True)
        scan_thread.start()

        w, h = self.wnd.buffer_size
        self.text.set_projection(w, h)
        self.panels.set_projection(w, h)
        self.backgrounds.set_projection(w, h)
        self.social_client = SocialClient()
        self.social_overlay = SocialOverlay(self, self.social_client)
        self.settings_overlay = SettingsOverlay(self)
        self.alert_overlay = AlertOverlay(self)
        self._shared_replay_batch: _ReplayBatchDownload | None = None
        self.social_client.command_context_provider = self._social_command_context
        self.social_client.chat_payload_handler = self._handle_chat_message_payload
        self.social_client.download_event_handler = self._handle_download_event
        from scenes.song_select import SongSelectScene
        self._scene = SongSelectScene(self)
        self._scene.on_enter()
        self._apply_window_settings()
        self._sync_framebuffer_state()

    def _apply_window_icon(self) -> None:
        if not APP_ICON_PATH.is_file():
            return
        try:
            image = pyglet.image.load(str(APP_ICON_PATH))
        except Exception:
            return
        for attr in ("set_icon",):
            setter = getattr(self.wnd, attr, None)
            if callable(setter):
                try:
                    setter(image)
                    return
                except Exception:
                    pass
        native_window = getattr(self.wnd, "_window", None)
        setter = getattr(native_window, "set_icon", None)
        if callable(setter):
            try:
                setter(image)
            except Exception:
                pass

    def switch_scene(self, scene) -> None:
        self._scene.on_leave()
        self.set_settings_button(None, visible=False)
        self._scene = scene
        self._scene.on_enter()

    def on_render(self, time: float, frametime: float):
        self._apply_frame_limit()
        if self._restart_pending and not self._restart_spawned:
            self._perform_restart()
            return
        current_buffer_size = self._refresh_window_metrics()
        if self._pending_framebuffer_sync or current_buffer_size != self._last_buffer_size:
            self._sync_framebuffer_state()
        self.ctx.scissor = None
        self.social_client.update(self.settings.nickname, self._social_presence_status_text())
        self._update_shared_replay_batch()
        self.alert_overlay.update(frametime)
        self._scene.on_render(time, frametime)
        self.social_overlay.draw(frametime)
        self.settings_overlay.draw(frametime)
        self.alert_overlay.draw()
        self._update_mouse_cursor()

    def on_resize(self, width: int, height: int):
        self._pending_framebuffer_sync = True
        self._sync_framebuffer_state()

    def _refresh_window_metrics(self) -> tuple[int, int]:
        width, height = (int(v) for v in self.wnd.size)
        buffer_w, buffer_h = (int(v) for v in self.wnd.buffer_size)
        raw_window = getattr(self.wnd, "_window", None)
        if raw_window is not None:
            try:
                raw_size = raw_window.get_size()
                if raw_size:
                    width, height = int(raw_size[0]), int(raw_size[1])
            except Exception:
                pass
            try:
                raw_fb_size = raw_window.get_framebuffer_size()
                if raw_fb_size:
                    buffer_w, buffer_h = int(raw_fb_size[0]), int(raw_fb_size[1])
            except Exception:
                buffer_w, buffer_h = width, height
        if hasattr(self.wnd, "_width"):
            self.wnd._width = width
        if hasattr(self.wnd, "_height"):
            self.wnd._height = height
        if hasattr(self.wnd, "_buffer_width"):
            self.wnd._buffer_width = buffer_w
        if hasattr(self.wnd, "_buffer_height"):
            self.wnd._buffer_height = buffer_h
        return (buffer_w, buffer_h)

    def _sync_framebuffer_state(self) -> None:
        bw, bh = self._refresh_window_metrics()
        self.wnd.set_default_viewport()
        self.ctx.viewport = (0, 0, bw, bh)
        self.ctx.scissor = None
        self.text.set_projection(bw, bh)
        self.panels.set_projection(bw, bh)
        self.backgrounds.set_projection(bw, bh)
        self._menu_context_cache = None
        self.settings_overlay.invalidate()
        self._scene.on_resize(bw, bh)
        self._last_buffer_size = (int(bw), int(bh))
        self._pending_framebuffer_sync = False

    def on_key_event(self, key, action, modifiers):
        if self.settings_overlay.handle_key_event(key, action):
            return
        if self.social_overlay.handle_key_event(key, action):
            return
        self._scene.on_key_event(key, action, modifiers)

    def on_unicode_char_entered(self, char: str):
        if self.settings_overlay.handle_text(char):
            return
        if self.social_overlay.handle_text(char):
            return
        self._scene.on_unicode_char_entered(char)

    def on_mouse_press_event(self, x: int, y: int, button: int):
        if self.settings_overlay.handle_mouse_press(x, y, button):
            return
        if self.social_overlay.handle_mouse_press(x, y, button):
            return
        self._scene.on_mouse_press(x, y, button)

    def on_mouse_release_event(self, x: int, y: int, button: int):
        if self.settings_overlay.handle_mouse_release(button):
            return
        if self.social_overlay.handle_mouse_release(button):
            return
        self._scene.on_mouse_release(x, y, button)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        if self.settings_overlay.handle_scroll(y_offset):
            return
        if self.social_overlay.handle_scroll(y_offset):
            return
        self._scene.on_mouse_scroll(x_offset, y_offset)

    def on_mouse_position_event(self, x: int, y: int, dx: int, dy: int):
        if self.settings_overlay.handle_mouse_move(x, y):
            return
        if self.social_overlay.handle_mouse_move(x, y):
            return
        self._scene.on_mouse_move(x, y, dx, dy)

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        if self.settings_overlay.handle_mouse_move(x, y):
            return
        if self.social_overlay.handle_mouse_move(x, y):
            return
        self._scene.on_mouse_move(x, y, dx, dy)

    def on_close(self):
        self.social_client.shutdown()
        self._scene.on_leave()

    def _set_mouse_cursor_style(self, style: str) -> None:
        style = style if style in {"default", "hand", "text"} else "default"
        if style == self._mouse_cursor_style:
            return
        window = getattr(self.wnd, "_window", None)
        if window is None:
            return
        getter = getattr(window, "get_system_mouse_cursor", None)
        setter = getattr(window, "set_mouse_cursor", None)
        if not callable(setter):
            return
        try:
            if style == "default":
                setter(None)
            else:
                cursor_name = getattr(
                    pyglet.window.Window,
                    "CURSOR_TEXT" if style == "text" else "CURSOR_HAND",
                    None,
                )
                if cursor_name is None or not callable(getter):
                    return
                setter(getter(cursor_name))
        except Exception:
            return
        self._mouse_cursor_style = style

    def _update_mouse_cursor(self) -> None:
        style = "default"
        if self.settings_overlay.wants_text_cursor():
            style = "text"
        elif self.social_overlay.wants_text_cursor():
            style = "text"
        elif self.settings_overlay.wants_hand_cursor():
            style = "hand"
        elif self.social_overlay.wants_hand_cursor():
            style = "hand"
        else:
            scene_cursor = getattr(self._scene, "wants_hand_cursor", None)
            if callable(scene_cursor) and scene_cursor():
                style = "hand"
        self._set_mouse_cursor_style(style)

    def _social_command_context(self):
        if hasattr(self._scene, "build_chat_share"):
            return self._scene
        return None

    def _social_presence_status_text(self) -> str:
        if type(self._scene).__name__ != "GameplayScene":
            return ""
        beatmap = getattr(self._scene, "_binfo", None)
        if beatmap is None:
            return ""
        title = str(getattr(beatmap, "title_unicode", "") or getattr(beatmap, "title", "") or "Unknown map").strip()
        version = str(getattr(beatmap, "version", "") or "?").strip()
        return f"Listening {title} - {version}"

    def _ensure_song_select_scene(self):
        from scenes.song_select import SongSelectScene

        if isinstance(self._scene, SongSelectScene):
            return self._scene
        self.switch_scene(SongSelectScene(self))
        if isinstance(self._scene, SongSelectScene):
            return self._scene
        return None

    def _handle_chat_message_payload(self, payload: ChatMessagePayload) -> None:
        scene = self._ensure_song_select_scene()
        if scene is None or not scene.force_select_shared_beatmap(payload.beatmap):
            self.alert_overlay.show_message("You don't have this map.")
            return
        if payload.kind != "now_playing_replays":
            return
        beatmap_id = None
        if payload.beatmap is not None and payload.beatmap.beatmap_id is not None:
            beatmap_id = int(payload.beatmap.beatmap_id)
        elif payload.replays:
            for replay in payload.replays:
                if replay.beatmap_id is not None:
                    beatmap_id = int(replay.beatmap_id)
                    break
        if beatmap_id is None:
            self.alert_overlay.show_message("Replays aren't uploaded")
            return
        target_dir = scene.selected_download_dir()
        if not target_dir:
            self.alert_overlay.show_message("Replays aren't uploaded")
            return
        replay_ids: list[str] = []
        seen_entities: set[tuple[int | None, str]] = set()
        unique_replays: list[SharedReplay] = []
        for replay in payload.replays:
            entity_key = (replay.beatmap_id, replay.replay_hash)
            if replay.replay_hash and entity_key in seen_entities:
                continue
            if replay.replay_hash:
                seen_entities.add(entity_key)
            unique_replays.append(replay)
            if replay.replay_id:
                replay_ids.append(replay.replay_id)
        if not replay_ids:
            self.alert_overlay.show_message("Replays aren't uploaded")
            return
        self.social_client.fetch_online_replays(beatmap_id, force=True)
        self._shared_replay_batch = _ReplayBatchDownload(
            beatmap_id=beatmap_id,
            replay_ids=list(dict.fromkeys(replay_ids)),
            target_dir=target_dir,
            requested_replays=unique_replays,
        )
        self.alert_overlay.hide_progress()

    def _handle_download_event(self, event_type: str, payload: dict) -> None:
        batch = self._shared_replay_batch
        if batch is None or batch.current_index >= len(batch.replay_ids):
            return
        replay_id = str(payload.get("replay_id") or "")
        current_id = batch.replay_ids[batch.current_index]
        if replay_id != current_id:
            return
        progress = float(payload.get("progress") or 0.0)
        label = f"Downloading replay ({batch.current_index + 1}/{len(batch.replay_ids)})"
        if event_type in {"started", "progress", "finished"}:
            self.alert_overlay.show_progress(label, progress)
        elif event_type == "failed":
            self._shared_replay_batch = None
            self.alert_overlay.hide_progress()
            self.alert_overlay.show_message("Failed to download replay.")

    def _update_shared_replay_batch(self) -> None:
        batch = self._shared_replay_batch
        if batch is None:
            return
        state = self.social_client.online_replays(batch.beatmap_id)
        if state.loading:
            return
        if state.error:
            self._shared_replay_batch = None
            self.alert_overlay.hide_progress()
            self.alert_overlay.show_message("Replay server unavailable.")
            return
        replay_map = {item.replay_id: item for item in state.items}
        available_ids = [replay_id for replay_id in batch.replay_ids if replay_id in replay_map]
        if not available_ids:
            self._shared_replay_batch = None
            self.alert_overlay.hide_progress()
            self.alert_overlay.show_message("Replays aren't uploaded")
            return
        batch.replay_ids = available_ids
        while batch.current_index < len(batch.replay_ids):
            replay = replay_map[batch.replay_ids[batch.current_index]]
            if replay.local_path and replay.local_path in batch.selected_paths:
                # Repair old duplicate replay-id -> path mappings by forcing a fresh unique download.
                replay.local_path = None
                replay.is_downloaded = False
                replay.download_progress = 0.0
                replay.status_text = ""
            if replay.local_path and Path(replay.local_path).is_file():
                replay.is_downloaded = True
                replay.is_downloading = False
                replay.download_progress = 1.0
                if replay.local_path not in batch.selected_paths:
                    batch.selected_paths.append(replay.local_path)
                batch.current_index += 1
                batch.started_current = False
                continue
            label = f"Downloading replay ({batch.current_index + 1}/{len(batch.replay_ids)})"
            if replay.is_downloading:
                self.alert_overlay.show_progress(label, replay.download_progress)
                return
            if batch.started_current:
                return
            self.social_client.download_replay(replay.replay_id, batch.target_dir)
            batch.started_current = True
            self.alert_overlay.show_progress(label, 0.0)
            return
        scene = self._ensure_song_select_scene()
        if scene is not None and batch.selected_paths:
            scene.apply_shared_replay_selection(batch.requested_replays, batch.selected_paths)
        self._shared_replay_batch = None
        self.alert_overlay.hide_progress()

    @property
    def effective_music_volume(self) -> float:
        if self.settings.music_muted:
            return 0.0
        return _shape_volume(self.settings.music_volume, 3.5)

    @property
    def effective_sfx_volume(self) -> float:
        if self.settings.sfx_muted:
            return 0.0
        return _shape_volume(self.settings.sfx_volume, 2.2)

    def _save_settings(self) -> None:
        try:
            self.settings.save(self.settings_path)
        except OSError:
            pass

    def _apply_frame_limit(self) -> None:
        limit = int(getattr(self.settings, "fps_limit", 0) or 0)
        now = time_module.perf_counter()
        if limit <= 0:
            self._last_frame_limit_tick = now
            return
        if self._last_frame_limit_tick is None:
            self._last_frame_limit_tick = now
            return
        target_dt = 1.0 / max(1, limit)
        elapsed = now - self._last_frame_limit_tick
        if elapsed < target_dt:
            time_module.sleep(target_dt - elapsed)
        self._last_frame_limit_tick = time_module.perf_counter()

    def _schedule_restart(self) -> None:
        self._restart_pending = True
        self.close_settings()

    def _perform_restart(self) -> None:
        if self._restart_spawned:
            return
        self._restart_spawned = True
        try:
            if getattr(sys, "frozen", False):
                command = [sys.executable]
            else:
                command = [sys.executable, str(APP_ROOT / "main.py")]
            subprocess.Popen(command, cwd=str(APP_ROOT))
        except OSError:
            self._restart_spawned = False
            self._restart_pending = False
            return
        self.wnd.close()

    @property
    def build_version(self) -> str:
        return BUILD_VERSION

    @property
    def common_resolutions(self) -> tuple[tuple[int, int], ...]:
        return COMMON_RESOLUTIONS

    def set_music_volume(self, volume: float, *, persist: bool = True) -> None:
        volume = _clamp01(volume)
        if abs(self.settings.music_volume - volume) < 1e-6:
            return
        self.settings.music_volume = volume
        if persist:
            self._save_settings()
        self._notify_audio_settings_changed()

    def set_music_muted(self, muted: bool, *, persist: bool = True) -> None:
        muted = bool(muted)
        if self.settings.music_muted == muted:
            return
        self.settings.music_muted = muted
        if persist:
            self._save_settings()
        self._notify_audio_settings_changed()

    def set_sfx_volume(self, volume: float, *, persist: bool = True) -> None:
        volume = _clamp01(volume)
        if abs(self.settings.sfx_volume - volume) < 1e-6:
            return
        self.settings.sfx_volume = volume
        if persist:
            self._save_settings()
        self._notify_audio_settings_changed()

    def set_sfx_muted(self, muted: bool, *, persist: bool = True) -> None:
        muted = bool(muted)
        if self.settings.sfx_muted == muted:
            return
        self.settings.sfx_muted = muted
        if persist:
            self._save_settings()
        self._notify_audio_settings_changed()

    def set_gameplay_overlay_visible(self, key: str, visible: bool) -> None:
        if not hasattr(self.settings, key):
            return
        visible = bool(visible)
        if bool(getattr(self.settings, key)) == visible:
            return
        setattr(self.settings, key, visible)
        self._save_settings()

    def set_graphics_setting(self, key: str, value, *, persist: bool = True) -> None:
        if not hasattr(self.settings, key):
            return
        current = getattr(self.settings, key)
        if isinstance(current, bool):
            normalized = bool(value)
        elif isinstance(current, int):
            normalized = int(round(float(value)))
        elif isinstance(current, float):
            normalized = float(value)
        else:
            normalized = value
        if current == normalized:
            return
        setattr(self.settings, key, normalized)
        if persist:
            self._save_settings()

    def skin_visual_settings(self) -> DefaultSkinVisualSettings:
        return DefaultSkinVisualSettings(
            circle_fill_color=self.settings.skin_circle_fill_color,
            circle_fill_opacity=self.settings.skin_circle_fill_opacity,
            circle_border_color=self.settings.skin_circle_border_color,
            circle_border_width=self.settings.skin_circle_border_width,
            circle_bloom=self.settings.skin_circle_bloom,
            circle_bloom_color=self.settings.skin_circle_bloom_color,
            slider_use_circle_head=self.settings.skin_slider_use_circle_head,
            slider_head_fill_color=self.settings.skin_slider_head_fill_color,
            slider_head_fill_opacity=self.settings.skin_slider_head_fill_opacity,
            slider_head_border_color=self.settings.skin_slider_head_border_color,
            slider_head_border_width=self.settings.skin_slider_head_border_width,
            slider_head_bloom=self.settings.skin_slider_head_bloom,
            slider_head_bloom_color=self.settings.skin_slider_head_bloom_color,
            slider_path_fill_color=self.settings.skin_slider_path_fill_color,
            slider_path_fill_opacity=self.settings.skin_slider_path_fill_opacity,
            slider_path_border_color=self.settings.skin_slider_path_border_color,
            slider_path_border_width=self.settings.skin_slider_path_border_width,
            slider_ball_fill_color=self.settings.skin_slider_ball_fill_color,
            slider_ball_fill_opacity=self.settings.skin_slider_ball_fill_opacity,
            slider_ball_border_color=self.settings.skin_slider_ball_border_color,
            slider_ball_border_width=self.settings.skin_slider_ball_border_width,
            slider_ball_bloom=self.settings.skin_slider_ball_bloom,
            slider_ball_bloom_color=self.settings.skin_slider_ball_bloom_color,
            cursor_color=self.settings.skin_cursor_color,
            cursor_size=self.settings.skin_cursor_size,
        ).normalized()

    def _apply_skin_visual_settings(self, *, notify_scene: bool = True) -> None:
        visuals = self.skin_visual_settings()
        for skin in SKIN_REGISTRY:
            setter = getattr(skin, "set_visual_settings", None)
            if callable(setter):
                setter(visuals)
        if notify_scene:
            callback = getattr(self._scene, "on_skin_settings_changed", None)
            if callable(callback):
                callback()

    def set_skin_setting(self, key: str, value, *, persist: bool = True) -> None:
        if not hasattr(self.settings, key):
            return
        current = getattr(self.settings, key)
        if isinstance(current, bool):
            normalized = bool(value)
        elif isinstance(current, tuple) and len(current) == 3:
            normalized = _clamp_rgb(value, current)
        else:
            normalized = _clamp01(value)
        if current == normalized:
            return
        setattr(self.settings, key, normalized)
        if persist:
            self._save_settings()
        self._apply_skin_visual_settings()

    def sync_skin_group_from_circle(self, target: str, *, persist: bool = True) -> None:
        if target == "slider_head":
            self.settings.skin_slider_head_fill_color = self.settings.skin_circle_fill_color
            self.settings.skin_slider_head_fill_opacity = self.settings.skin_circle_fill_opacity
            self.settings.skin_slider_head_border_color = self.settings.skin_circle_border_color
            self.settings.skin_slider_head_border_width = self.settings.skin_circle_border_width
            self.settings.skin_slider_head_bloom = self.settings.skin_circle_bloom
            self.settings.skin_slider_head_bloom_color = self.settings.skin_circle_bloom_color
        elif target == "slider_ball":
            self.settings.skin_slider_ball_fill_color = self.settings.skin_circle_fill_color
            self.settings.skin_slider_ball_fill_opacity = self.settings.skin_circle_fill_opacity
            self.settings.skin_slider_ball_border_color = self.settings.skin_circle_border_color
            self.settings.skin_slider_ball_border_width = self.settings.skin_circle_border_width
            self.settings.skin_slider_ball_bloom = self.settings.skin_circle_bloom
            self.settings.skin_slider_ball_bloom_color = self.settings.skin_circle_bloom_color
        else:
            return
        if persist:
            self._save_settings()
        self._apply_skin_visual_settings()

    def sync_skin_bloom_color_from_fill(self, target: str, *, persist: bool = True) -> None:
        if target == "circle":
            self.settings.skin_circle_bloom_color = self.settings.skin_circle_fill_color
        elif target == "slider_head":
            self.settings.skin_slider_head_bloom_color = self.settings.skin_slider_head_fill_color
        elif target == "slider_ball":
            self.settings.skin_slider_ball_bloom_color = self.settings.skin_slider_ball_fill_color
        else:
            return
        if persist:
            self._save_settings()
        self._apply_skin_visual_settings()

    def _notify_audio_settings_changed(self) -> None:
        callback = getattr(self._scene, "on_global_audio_settings_changed", None)
        if callable(callback):
            callback()

    def menu_context(self):
        width, height = self.wnd.buffer_size
        key = (width, height)
        if self._menu_context_cache is None or self._menu_context_cache[0] != key:
            self._menu_context_cache = (key, build_layout_context(width, height))
        return self._menu_context_cache[1]

    def content_layout(self) -> dict[str, float]:
        context = self.menu_context()
        return {
            "scale": context.density,
            "x": context.content_rect.x,
            "y": context.content_rect.y,
            "w": context.content_rect.w,
            "h": context.content_rect.h,
        }

    def open_maps_folder(self) -> None:
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        folder = str(MAPS_DIR.resolve())
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    def toggle_settings(self) -> None:
        self.settings_overlay.toggle()

    def close_settings(self) -> None:
        self.settings_overlay.close()

    def set_settings_button(
        self,
        rect: tuple[float, float, float, float] | None,
        *,
        visible: bool,
    ) -> None:
        self._settings_button_rect = rect
        self._settings_button_visible = bool(visible and rect is not None)
        self.settings_overlay.invalidate()

    def _set_nickname(self, nickname: str) -> None:
        nickname = nickname[:16]
        if self.settings.nickname == nickname:
            return
        self.settings.nickname = nickname
        self._save_settings()

    def _set_screen_mode(self, mode: str) -> None:
        if mode not in SCREEN_MODES or self.settings.screen_mode == mode:
            return
        self.settings.screen_mode = mode
        self._save_settings()
        self._schedule_restart()

    def _set_resolution(self, width: int, height: int) -> None:
        width = max(640, int(width))
        height = max(480, int(height))
        if (
            self.settings.resolution_width == width
            and self.settings.resolution_height == height
        ):
            return
        self.settings.resolution_width = width
        self.settings.resolution_height = height
        self._windowed_size = (width, height)
        self._save_settings()
        self._schedule_restart()

    def _set_fps_limit(self, limit: int) -> None:
        limit = int(limit)
        if limit not in FPS_LIMIT_OPTIONS or self.settings.fps_limit == limit:
            return
        self.settings.fps_limit = limit
        self._save_settings()

    def _set_borderless_windowed(self, enabled: bool) -> bool:
        # The raw GLFW undecorated-window path turned out to be unstable on some setups.
        # Keep this as a no-op and let borderless use the safer fullscreen-sized fallback.
        return False

    def _apply_window_settings(self) -> None:
        mode = self.settings.screen_mode
        resolution = (
            int(self.settings.resolution_width),
            int(self.settings.resolution_height),
        )
        raw_window = getattr(self.wnd, "_window", None)
        if mode in {"fullscreen", "borderless"}:
            self._set_borderless_windowed(False)
            if raw_window is not None and hasattr(raw_window, "set_fullscreen"):
                raw_window.set_fullscreen(True, width=resolution[0], height=resolution[1])
                if hasattr(self.wnd, "_fullscreen"):
                    self.wnd._fullscreen = True
            elif not self.wnd.fullscreen:
                self.wnd.fullscreen = True
            self._pending_framebuffer_sync = True
            return

        if self.wnd.fullscreen:
            if raw_window is not None and hasattr(raw_window, "set_fullscreen"):
                raw_window.set_fullscreen(False)
                if hasattr(self.wnd, "_fullscreen"):
                    self.wnd._fullscreen = False
            else:
                self.wnd.fullscreen = False
            self._pending_framebuffer_sync = True

        self._set_borderless_windowed(False)
        self.wnd.resizable = True
        if raw_window is not None and hasattr(raw_window, "set_size"):
            raw_window.set_size(*resolution)
            if hasattr(self.wnd, "_width"):
                self.wnd._width = int(resolution[0])
            if hasattr(self.wnd, "_height"):
                self.wnd._height = int(resolution[1])
        else:
            self.wnd.size = resolution
        self._pending_framebuffer_sync = True
