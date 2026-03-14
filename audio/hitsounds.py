"""Hitsound playback manager -- loads WAV/OGG files from a skin directory."""

from __future__ import annotations

from pathlib import Path

import pygame.mixer

_SAMPLE_SET_NAMES = ("normal", "soft", "drum")
_HIT_TYPES = ("normal", "whistle", "finish", "clap")

MAX_CHANNELS = 48


class HitsoundManager:
    """Loads hitsound files from a directory.  No files = no sounds."""

    def __init__(self, hitsound_dir: str | Path | None = None,
                 volume: float = 1.0,
                 muted: bool = False):
        pygame.mixer.set_num_channels(MAX_CHANNELS)
        self._volume = max(0.0, min(float(volume), 1.0))
        self._muted = bool(muted)
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        if hitsound_dir is not None:
            self._load(Path(hitsound_dir))

    def _effective_volume(self) -> float:
        if self._muted:
            return 0.0
        return self._volume ** 2.2

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(float(volume), 1.0))

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)

    def play(self, normal_set: str, addition_set: str, sound_enum: int, volume: float) -> None:
        vol = max(0.0, min(volume, 1.0)) * self._effective_volume()
        if vol <= 0.0:
            return
        self._play_one(normal_set, "normal", vol)
        if sound_enum & 2:
            self._play_one(addition_set, "whistle", vol)
        if sound_enum & 4:
            self._play_one(addition_set, "finish", vol)
        if sound_enum & 8:
            self._play_one(addition_set, "clap", vol)

    def _play_one(self, sample_set: str, hit_type: str, volume: float) -> None:
        key = f"{sample_set}-hit{hit_type}"
        sound = self._sounds.get(key)
        if sound is None:
            key = f"normal-hit{hit_type}"
            sound = self._sounds.get(key)
        if sound is None:
            return
        ch = pygame.mixer.find_channel(True)
        if ch is None:
            ch = sound.play()
        else:
            ch.set_volume(volume)
            ch.play(sound)
        if ch is not None:
            ch.set_volume(volume)

    def _load(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for ss in _SAMPLE_SET_NAMES:
            for ht in _HIT_TYPES:
                key = f"{ss}-hit{ht}"
                for ext in (".wav", ".ogg", ".mp3"):
                    path = directory / f"{key}{ext}"
                    if path.is_file():
                        try:
                            self._sounds[key] = pygame.mixer.Sound(str(path))
                        except Exception:
                            pass
                        break
