from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from runtime_paths import load_env_file

load_env_file()

ENV_SERVER_URL = os.getenv("OSU_REPLAY_SERVER_URL", "").rstrip("/")
DEFAULT_SERVER_URL = ENV_SERVER_URL or "http://127.0.0.1:8000"
LEGACY_SERVER_URLS = {
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://0.0.0.0:8000",
}


def social_appdata_dir() -> Path:
    root = os.getenv("APPDATA")
    if root:
        return Path(root) / "osu_replay_v2"
    return Path.home() / ".osu_replay_v2"


@dataclass(slots=True)
class SocialLocalState:
    player_uuid: str
    friends: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    replay_downloads: dict[str, str] = field(default_factory=dict)
    replay_uploads: dict[str, str] = field(default_factory=dict)
    server_url: str = DEFAULT_SERVER_URL

    @classmethod
    def path(cls) -> Path:
        return social_appdata_dir() / "social_state.json"

    @classmethod
    def uuid_path(cls) -> Path:
        return social_appdata_dir() / "player_uuid.txt"

    @classmethod
    def _fallback_player_uuid(cls) -> str | None:
        try:
            value = cls.uuid_path().read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None

    @classmethod
    def load(cls) -> "SocialLocalState":
        path = cls.path()
        fallback_uuid = cls._fallback_player_uuid() or str(uuid4())
        defaults = cls(player_uuid=fallback_uuid)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            defaults.save()
            return defaults
        server_url = str(data.get("server_url") or defaults.server_url).rstrip("/")
        if ENV_SERVER_URL:
            server_url = ENV_SERVER_URL
        elif server_url in LEGACY_SERVER_URLS:
            server_url = DEFAULT_SERVER_URL

        player_uuid = str(data.get("player_uuid") or "").strip() or fallback_uuid
        state = cls(
            player_uuid=player_uuid,
            friends=[str(item) for item in data.get("friends", [])],
            blocked=[str(item) for item in data.get("blocked", [])],
            replay_downloads={str(key): str(value) for key, value in dict(data.get("replay_downloads", {})).items()},
            replay_uploads={str(key): str(value) for key, value in dict(data.get("replay_uploads", {})).items()},
            server_url=server_url,
        )
        state.save()
        return state

    def save(self) -> None:
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        self.uuid_path().write_text(self.player_uuid.strip(), encoding="utf-8")

    def set_friend(self, player_uuid: str, enabled: bool) -> None:
        current = set(self.friends)
        if enabled:
            current.add(player_uuid)
        else:
            current.discard(player_uuid)
        self.friends = sorted(current)
        self.save()

    def set_blocked(self, player_uuid: str, enabled: bool) -> None:
        current = set(self.blocked)
        if enabled:
            current.add(player_uuid)
        else:
            current.discard(player_uuid)
        self.blocked = sorted(current)
        self.save()

    def remember_download(self, replay_id: str, path: str) -> None:
        self.replay_downloads[replay_id] = path
        self.save()

    def forget_download(self, replay_id: str) -> None:
        self.replay_downloads.pop(replay_id, None)
        self.save()

    def remember_upload(self, replay_path: str, replay_id: str) -> None:
        self.replay_uploads[replay_path] = replay_id
        self.save()
