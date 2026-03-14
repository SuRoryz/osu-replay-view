"""Helpers for resolving portable runtime paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_ROOT = app_root()
ENV_FILE = APP_ROOT / ".env"
STATIC_DIR = APP_ROOT / "static"
APP_ICON_PATH = STATIC_DIR / "icon.ico"
SETTINGS_PATH = APP_ROOT / "app_settings.json"
MAPS_DIR = APP_ROOT / "maps"
REPLAYS_DIR = APP_ROOT / "replays"
SKINS_DIR = APP_ROOT / "skins"
HITSOUNDS_DIR = SKINS_DIR / "hitsounds"
FFMPEG_DIR = APP_ROOT / "ffmpeg"


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    target = Path(path) if path is not None else ENV_FILE
    if not target.is_file():
        return {}

    loaded: dict[str, str] = {}
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = os.environ.get(key, value)
    return loaded


def replay_dir_for_set(mapset_directory: str | Path) -> Path:
    return REPLAYS_DIR / Path(mapset_directory).name


def bundled_binary_path(stem: str) -> Path | None:
    suffix = ".exe" if sys.platform == "win32" else ""
    filename = f"{stem}{suffix}"
    candidates = (
        APP_ROOT / filename,
        FFMPEG_DIR / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def ensure_runtime_dirs() -> None:
    for path in (MAPS_DIR, REPLAYS_DIR, SKINS_DIR, HITSOUNDS_DIR):
        path.mkdir(parents=True, exist_ok=True)


load_env_file()
