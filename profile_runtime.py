"""Automated runtime profiler for menu/gameplay benchmarking."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyglet

from app import App, AppSettings
from profiling import profiler
from scenes.gameplay import GameplayScene
from scenes.song_select import SongSelectScene


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class ProfileRuntimeApp(App):
    _settings = AppSettings.load(App.settings_path)
    window_size = (
        _env_int("OSU_BENCH_WIDTH", int(_settings.resolution_width)),
        _env_int("OSU_BENCH_HEIGHT", int(_settings.resolution_height)),
    )
    fullscreen = (
        _settings.screen_mode in {"fullscreen", "borderless"}
        and hasattr(pyglet, "canvas")
    )
    samples = _env_int("OSU_BENCH_SAMPLES", App.samples)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bench_mode = os.environ.get("OSU_BENCH_MODE", "menu").strip().lower()
        self._bench_label = os.environ.get("OSU_BENCH_LABEL", "").strip()
        self._menu_seconds = max(1.0, _env_float("OSU_BENCH_MENU_SECONDS", 8.0))
        self._gameplay_seconds = max(1.0, _env_float("OSU_BENCH_GAMEPLAY_SECONDS", 12.0))
        self._post_ready_warmup_seconds = max(0.0, _env_float("OSU_BENCH_WARMUP_SECONDS", 1.5))
        self._auto_play_started = False
        self._menu_ready_at: float | None = None
        self._gameplay_ready_at: float | None = None
        self._bench_started_at: float | None = None
        self._bench_finished = False
        self._disable_bloom = os.environ.get("OSU_BENCH_DISABLE_BLOOM", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._disable_frame_limit = os.environ.get("OSU_BENCH_KEEP_FPS_LIMIT", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._bench_output = os.environ.get("OSU_BENCH_OUTPUT", "").strip()
        if self._disable_frame_limit:
            self.settings.fps_limit = 0
        print(
            f"[bench] mode={self._bench_mode} size={self.window_size} "
            f"fullscreen={self.fullscreen} samples={self.samples} bloom_off={self._disable_bloom} "
            f"fps_limit={self.settings.fps_limit}"
        )

    def on_render(self, time: float, frametime: float):
        if self._bench_mode == "gameplay" and not self._auto_play_started:
            scene = self._scene
            if isinstance(scene, SongSelectScene) and self.scanner.scan_complete:
                if scene._selected_info() is not None:
                    scene._play_selected()
                    self._auto_play_started = True
                    print("[bench] requested gameplay transition")

        super().on_render(time, frametime)
        scene = self._scene

        if self._bench_mode == "menu":
            if isinstance(scene, SongSelectScene) and scene._selected_info() is not None and scene._anim_state == "idle":
                if self._menu_ready_at is None:
                    self._menu_ready_at = time
                    self._bench_started_at = time + self._post_ready_warmup_seconds
                    print("[bench] menu ready")
                elif self._bench_started_at is not None and time >= self._bench_started_at + self._menu_seconds:
                    print("[bench] closing after menu benchmark")
                    self._finish_benchmark(time)
                    self.wnd.close()
            return

        if isinstance(scene, GameplayScene):
            if self._disable_bloom and hasattr(scene, "_skin"):
                setattr(scene._skin, "_bloom_enabled", False)
            if scene._loading_ready:
                if self._gameplay_ready_at is None:
                    self._gameplay_ready_at = time
                    self._bench_started_at = time + self._post_ready_warmup_seconds
                    print("[bench] gameplay ready")
                elif self._bench_started_at is not None and time >= self._bench_started_at + self._gameplay_seconds:
                    print("[bench] closing after gameplay benchmark")
                    self._finish_benchmark(time)
                    self.wnd.close()

    def _finish_benchmark(self, time_value: float) -> None:
        if self._bench_finished:
            return
        self._bench_finished = True
        summary = profiler.report_now(reset=True)
        duration_s = 0.0
        if self._bench_started_at is not None:
            duration_s = max(0.0, time_value - self._bench_started_at)
        frame_key = f"{self._bench_mode}.frame"
        frame_stats = summary.get(frame_key, {})
        avg_frame_ms = float(frame_stats.get("avg_ms", 0.0))
        fps = 1000.0 / avg_frame_ms if avg_frame_ms > 1e-6 else 0.0
        result = {
            "label": self._bench_label,
            "mode": self._bench_mode,
            "window_size": list(self.window_size),
            "fullscreen": bool(self.fullscreen),
            "samples": int(self.samples),
            "duration_s": duration_s,
            "fps_limit": int(getattr(self.settings, "fps_limit", 0) or 0),
            "bloom_disabled": self._disable_bloom,
            "warmup_s": self._post_ready_warmup_seconds,
            "avg_fps": fps,
            "frame_stats": frame_stats,
            "profile": summary,
            "gl_version_code": int(getattr(self.ctx, "version_code", 0) or 0),
            "gl_version": getattr(self.ctx, "info", {}).get("GL_VERSION", ""),
            "gl_renderer": getattr(self.ctx, "info", {}).get("GL_RENDERER", ""),
            "gl_vendor": getattr(self.ctx, "info", {}).get("GL_VENDOR", ""),
        }
        print(
            f"[bench] summary mode={self._bench_mode} duration={duration_s:.2f}s "
            f"avg_frame={avg_frame_ms:.3f}ms avg_fps={fps:.2f}"
        )
        if self._bench_output:
            output_path = Path(self._bench_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"[bench] wrote summary to {output_path}")


if __name__ == "__main__":
    ProfileRuntimeApp.run()
