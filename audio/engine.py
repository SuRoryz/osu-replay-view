"""Audio playback engine backed by pygame.mixer."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pygame.mixer
from runtime_paths import bundled_binary_path


def _pydub_hidden_popen(original_popen):
    if getattr(original_popen, "_osu_replay_hidden_wrapper", False):
        return original_popen

    def wrapped(*args, **kwargs):
        if sys.platform == "win32":
            creationflags = int(kwargs.get("creationflags") or 0)
            creationflags |= int(getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0)
            kwargs["creationflags"] = creationflags
            if kwargs.get("startupinfo") is None and hasattr(subprocess, "STARTUPINFO"):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= int(getattr(subprocess, "STARTF_USESHOWWINDOW", 0) or 0)
                if hasattr(subprocess, "SW_HIDE"):
                    startupinfo.wShowWindow = subprocess.SW_HIDE
                kwargs["startupinfo"] = startupinfo
        return original_popen(*args, **kwargs)

    wrapped._osu_replay_hidden_wrapper = True
    return wrapped


class _SubprocessProxy:
    def __init__(self, module):
        self._module = module
        self.Popen = _pydub_hidden_popen(module.Popen)

    def __getattr__(self, name):
        return getattr(self._module, name)


def _configure_pydub_binaries() -> None:
    try:
        from pydub import AudioSegment
        from pydub import audio_segment as pydub_audio_segment
        from pydub import utils as pydub_utils
    except ImportError:
        return

    if not getattr(pydub_audio_segment, "_osu_replay_hidden_subprocess", False):
        pydub_audio_segment.subprocess = _SubprocessProxy(pydub_audio_segment.subprocess)
        pydub_audio_segment._osu_replay_hidden_subprocess = True
    if not getattr(pydub_utils, "_osu_replay_hidden_subprocess", False):
        pydub_utils.Popen = _pydub_hidden_popen(pydub_utils.Popen)
        pydub_utils._osu_replay_hidden_subprocess = True

    ffmpeg_path = bundled_binary_path("ffmpeg")
    if ffmpeg_path is not None:
        resolved = str(ffmpeg_path)
        AudioSegment.converter = resolved
        AudioSegment.ffmpeg = resolved

    ffprobe_path = bundled_binary_path("ffprobe")
    if ffprobe_path is not None:
        AudioSegment.ffprobe = str(ffprobe_path)


class AudioEngine:
    """Plays a music file and exposes the current playback position in ms.

    Supports a lead-in period where position_ms returns negative values
    before the audio actually starts, keeping gameplay and music in sync.

    Base map speed still uses pitch-shifted audio to match DT/HT, while
    runtime playback speed changes try to preserve pitch like osu! stable.
    """

    def __init__(self, audio_path: str, lead_in_ms: float = 0.0,
                 speed: float = 1.0,
                 volume: float = 1.0,
                 muted: bool = False):
        _configure_pydub_binaries()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self._audio_path = audio_path
        self._base_speed = max(0.05, float(speed))
        self._playback_speed = 1.0
        self._volume = max(0.0, min(float(volume), 1.0))
        self._muted = bool(muted)
        self._tmp_paths: dict[tuple[float, float], str] = {}
        self._tmp_paths_lock = threading.Lock()
        self._duration_ms = self._probe_duration_ms(audio_path)

        actual_path = self._path_for_speed()
        pygame.mixer.music.load(actual_path)
        self._apply_music_volume()
        self._lead_in_ms = max(0.0, lead_in_ms)
        self._elapsed_ms: float = 0.0
        self._music_anchor_ms: float = 0.0
        self._music_playing = False
        self._paused = False
        self._started = False
        self._finished = False

    def _effective_volume(self) -> float:
        if self._muted:
            return 0.0
        return self._volume ** 3.5

    def _apply_music_volume(self) -> None:
        try:
            pygame.mixer.music.set_volume(self._effective_volume())
        except Exception:
            pass

    @staticmethod
    def _atempo_filter(speed: float) -> str:
        factors: list[float] = []
        remaining = max(0.05, float(speed))
        while remaining < 0.5:
            factors.append(0.5)
            remaining /= 0.5
        while remaining > 2.0:
            factors.append(2.0)
            remaining /= 2.0
        if abs(remaining - 1.0) >= 1e-4:
            factors.append(remaining)
        return ",".join(f"atempo={factor:.6f}" for factor in factors)

    @classmethod
    def _make_speed_file(cls, audio_path: str, base_speed: float, playback_speed: float) -> str:
        """Create a temp WAV for the current map speed and playback speed."""
        try:
            from pydub import AudioSegment
        except ImportError:
            return audio_path

        base_speed = max(0.05, float(base_speed))
        playback_speed = max(0.05, float(playback_speed))
        combined_speed = base_speed * playback_speed

        seg = AudioSegment.from_file(audio_path)
        if abs(base_speed - 1.0) >= 1e-4:
            new_rate = max(1000, int(round(seg.frame_rate * base_speed)))
            seg = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate})
        seg = seg.set_frame_rate(44100)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            if abs(playback_speed - 1.0) >= 1e-4:
                seg.export(
                    tmp.name,
                    format="wav",
                    parameters=["-filter:a", cls._atempo_filter(playback_speed)],
                )
            else:
                seg.export(tmp.name, format="wav")
        except Exception:
            new_rate = max(1000, int(round(seg.frame_rate * playback_speed)))
            fallback = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(44100)
            fallback.export(tmp.name, format="wav")
        tmp.close()
        return tmp.name

    def _probe_duration_ms(self, audio_path: str) -> float | None:
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(audio_path)
            return float(len(seg)) / self._base_speed
        except Exception:
            try:
                return float(pygame.mixer.Sound(audio_path).get_length() * 1000.0) / self._base_speed
            except Exception:
                return None

    def _path_for_speed(self, *, playback_speed: float | None = None) -> str:
        base_key = round(float(self._base_speed), 4)
        playback_key = round(float(self._playback_speed if playback_speed is None else playback_speed), 4)
        if abs(base_key - 1.0) < 1e-4 and abs(playback_key - 1.0) < 1e-4:
            return self._audio_path
        cache_key = (base_key, playback_key)
        with self._tmp_paths_lock:
            cached = self._tmp_paths.get(cache_key)
            if cached is not None:
                return cached
            path = self._make_speed_file(self._audio_path, base_key, playback_key)
            self._tmp_paths[cache_key] = path
            return path

    def _load_music_for_current_speed(self) -> None:
        pygame.mixer.music.load(self._path_for_speed())
        self._apply_music_volume()

    def _source_position_ms(self, logical_time_ms: float) -> float:
        return max(0.0, logical_time_ms / max(0.05, self._playback_speed))

    def _sync_elapsed_from_mixer(self) -> None:
        if not self._music_playing:
            return
        busy = pygame.mixer.music.get_busy()
        pos = pygame.mixer.music.get_pos()
        if pos < 0 and not busy and not self._paused:
            self._music_playing = False
            self._finished = True
            if self._duration_ms is not None:
                self._elapsed_ms = max(self._elapsed_ms, self._duration_ms + self._lead_in_ms)
            else:
                self._elapsed_ms = max(self._elapsed_ms, self._music_anchor_ms + self._lead_in_ms)
            return
        if pos < 0:
            pos = 0
        logical_pos = self._music_anchor_ms + float(pos) * self._playback_speed
        if self._duration_ms is not None:
            logical_pos = min(logical_pos, self._duration_ms)
        self._elapsed_ms = logical_pos + self._lead_in_ms

    def _start_music(self, logical_pos_ms: float) -> None:
        self._music_anchor_ms = max(0.0, logical_pos_ms)
        source_pos_ms = self._source_position_ms(self._music_anchor_ms)
        start_sec = max(0.0, source_pos_ms / 1000.0)
        try:
            pygame.mixer.music.play(start=start_sec)
        except TypeError:
            pygame.mixer.music.play()
            try:
                pygame.mixer.music.set_pos(start_sec)
            except Exception:
                pass
        except Exception:
            pygame.mixer.music.play()
            try:
                pygame.mixer.music.set_pos(start_sec)
            except Exception:
                pass
        self._apply_music_volume()
        self._music_playing = True
        self._finished = False

    def start(self) -> None:
        self._started = True
        self._elapsed_ms = 0.0
        self._music_anchor_ms = 0.0
        self._finished = False
        if self._lead_in_ms <= 0:
            self._start_music(0.0)

    def tick(self, dt_seconds: float) -> None:
        """Advance the internal timer. Call once per frame with frametime."""
        if not self._started or self._paused:
            return
        self._elapsed_ms += dt_seconds * 1000.0 * self._playback_speed
        if not self._music_playing and not self._finished and self._elapsed_ms >= self._lead_in_ms:
            self._start_music(self.position_ms)

    @property
    def position_ms(self) -> float:
        if not self._started:
            return -self._lead_in_ms
        if self._paused:
            return self._elapsed_ms - self._lead_in_ms
        if not self._music_playing:
            return self._elapsed_ms - self._lead_in_ms
        self._sync_elapsed_from_mixer()
        return self._elapsed_ms - self._lead_in_ms

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def lead_in_ms(self) -> float:
        return self._lead_in_ms

    @property
    def duration_ms(self) -> float | None:
        return self._duration_ms

    @property
    def playback_speed(self) -> float:
        return self._playback_speed

    def set_paused(self, paused: bool) -> None:
        if paused == self._paused:
            return
        if paused:
            self._sync_elapsed_from_mixer()
            self._paused = True
            if self._music_playing:
                pygame.mixer.music.pause()
            return
        self._paused = False
        if self._music_playing:
            pygame.mixer.music.unpause()

    def toggle_pause(self) -> None:
        self.set_paused(not self._paused)

    def seek_ms(self, position_ms: float) -> None:
        if not self._started:
            return

        if self._duration_ms is not None:
            position_ms = min(position_ms, self._duration_ms)
        position_ms = max(-self._lead_in_ms, position_ms)
        self._elapsed_ms = position_ms + self._lead_in_ms
        self._finished = False

        if self._music_playing:
            pygame.mixer.music.stop()
            self._music_playing = False

        if position_ms < 0.0:
            return
        if self._duration_ms is not None and position_ms >= self._duration_ms - 1e-3:
            self._finished = True
            return

        self._load_music_for_current_speed()
        self._start_music(position_ms)
        if self._paused:
            pygame.mixer.music.pause()

    def set_playback_speed(self, speed: float) -> None:
        speed = max(0.05, float(speed))
        if abs(speed - self._playback_speed) < 1e-4:
            return

        logical_pos = self.position_ms if self._started else -self._lead_in_ms
        self._playback_speed = speed

        if self._started:
            self.seek_ms(logical_pos)
        else:
            self._load_music_for_current_speed()

    def prime_playback_speeds(self, speeds: list[float]) -> None:
        for speed in speeds:
            speed = max(0.05, float(speed))
            if abs(speed - self._playback_speed) < 1e-4:
                continue
            self._path_for_speed(playback_speed=speed)

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(float(volume), 1.0))
        self._apply_music_volume()

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        self._apply_music_volume()

    def cleanup(self) -> None:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.stop()
        except Exception:
            pass
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        for path in self._tmp_paths.values():
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
