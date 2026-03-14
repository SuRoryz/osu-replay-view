from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import URL


@dataclass(slots=True, frozen=True)
class Settings:
    database_url: str
    storage_root: Path
    cors_origins: tuple[str, ...]

    @classmethod
    def load(cls) -> "Settings":
        root = Path(__file__).resolve().parents[1]
        storage_root = Path(os.getenv("OSU_SERVER_STORAGE", root / "storage" / "replays"))
        origins = tuple(
            part.strip()
            for part in os.getenv("OSU_SERVER_CORS", "*").split(",")
            if part.strip()
        )
        database_url = os.getenv("OSU_SERVER_DATABASE_URL", "").strip()
        if not database_url:
            database_url = URL.create(
                "postgresql+psycopg",
                username=os.getenv("OSU_SERVER_DB_USER") or os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("OSU_SERVER_DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD", "postgres"),
                host=os.getenv("OSU_SERVER_DB_HOST", "localhost"),
                port=int(os.getenv("OSU_SERVER_DB_PORT", "5432")),
                database=os.getenv("OSU_SERVER_DB_NAME") or os.getenv("POSTGRES_DB", "osu_replay_v2"),
            )
        return cls(
            database_url=database_url,
            storage_root=storage_root,
            cors_origins=origins or ("*",),
        )
