from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def official_osu_appdata_dir() -> Path:
    root = os.getenv("APPDATA")
    if root:
        return Path(root) / "osu_replay_v2"
    return Path.home() / ".osu_replay_v2"


@dataclass(slots=True)
class OfficialOsuLocalState:
    score_downloads: dict[str, str] = field(default_factory=dict)

    @classmethod
    def path(cls) -> Path:
        return official_osu_appdata_dir() / "official_osu_state.json"

    @classmethod
    def load(cls) -> "OfficialOsuLocalState":
        path = cls.path()
        defaults = cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            defaults.save()
            return defaults
        state = cls(
            score_downloads={
                str(key): str(value)
                for key, value in dict(data.get("score_downloads", {})).items()
            },
        )
        state.save()
        return state

    def save(self) -> None:
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def remember_download(self, score_id: int, path: str) -> None:
        self.score_downloads[str(int(score_id))] = str(path)
        self.save()

    def forget_download(self, score_id: int) -> None:
        self.score_downloads.pop(str(int(score_id)), None)
        self.save()

    def downloaded_path_for_score(self, score_id: int) -> str | None:
        return self.score_downloads.get(str(int(score_id)))
