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
    osu_client_id: str
    osu_client_secret: str
    osu_redirect_uri: str
    osu_api_base_url: str
    osu_authorize_url: str
    osu_token_url: str
    osu_auth_root: Path
    osu_oauth_scopes: tuple[str, ...]
    map_mirror_download_templates: tuple[str, ...]
    map_mirror_timeout_seconds: float

    @classmethod
    def load(cls) -> "Settings":
        root = Path(__file__).resolve().parents[1]
        storage_root = Path(os.getenv("OSU_SERVER_STORAGE", root / "storage" / "replays"))
        osu_auth_root = Path(os.getenv("OSU_SERVER_OSU_AUTH_STORAGE", root / "storage" / "osu_auth"))
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
            osu_client_id=os.getenv("OSU_OFFICIAL_CLIENT_ID", "").strip(),
            osu_client_secret=os.getenv("OSU_OFFICIAL_CLIENT_SECRET", "").strip(),
            osu_redirect_uri=os.getenv("OSU_OFFICIAL_REDIRECT_URI", "http://127.0.0.1:8000/osu/auth/callback").strip(),
            osu_api_base_url=os.getenv("OSU_OFFICIAL_API_BASE_URL", "https://osu.ppy.sh/api/v2").rstrip("/"),
            osu_authorize_url=os.getenv("OSU_OFFICIAL_AUTHORIZE_URL", "https://osu.ppy.sh/oauth/authorize").strip(),
            osu_token_url=os.getenv("OSU_OFFICIAL_TOKEN_URL", "https://osu.ppy.sh/oauth/token").strip(),
            osu_auth_root=osu_auth_root,
            osu_oauth_scopes=tuple(
                part.strip()
                for part in os.getenv("OSU_OFFICIAL_SCOPES", "public identify").split()
                if part.strip()
            ),
            map_mirror_download_templates=tuple(
                part.strip()
                for part in os.getenv(
                    "OSU_MIRROR_DOWNLOAD_TEMPLATES",
                    "https://api.chimu.moe/v1/download/{beatmapset_id},https://osu.direct/d/{beatmapset_id}",
                ).split(",")
                if part.strip()
            ),
            map_mirror_timeout_seconds=max(5.0, float(os.getenv("OSU_MIRROR_TIMEOUT_SECONDS", "60"))),
        )
