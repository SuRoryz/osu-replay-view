from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OsuAccount:
    user_id: int
    username: str
    avatar_url: str = ""
    country_code: str = ""
    cover_url: str = ""


@dataclass(slots=True)
class OsuAuthStatus:
    enabled: bool = False
    linked: bool = False
    scopes: list[str] = field(default_factory=list)
    account: OsuAccount | None = None
    expires_at: float | None = None
    loading: bool = False
    login_in_progress: bool = False
    error: str | None = None


@dataclass(slots=True)
class OsuBeatmapLookup:
    beatmap_id: int
    beatmapset_id: int | None
    checksum: str
    mode: str = "osu"
    title: str = ""
    artist: str = ""
    creator: str = ""
    version: str = ""


@dataclass(slots=True)
class OsuBeatmapLookupState:
    item: OsuBeatmapLookup | None = None
    loading: bool = False
    error: str | None = None
    requested_at: float = 0.0
    loaded_at: float = 0.0


@dataclass(slots=True)
class OsuBeatmapsetSearchItem:
    beatmapset_id: int
    artist: str
    title: str
    creator: str
    status: str = ""
    favourite_count: int = 0
    play_count: int = 0
    source: str = ""
    tags: str = ""
    cover_url: str = ""
    preview_url: str = ""
    beatmaps: list[dict] = field(default_factory=list)
    is_downloading: bool = False
    download_progress: float = 0.0
    download_status: str = ""
    downloaded_dir: str | None = None


@dataclass(slots=True)
class OsuMapSearchState:
    query: str = ""
    status_filter: str = ""
    mode_filter: str = "osu"
    cursor_string: str = ""
    items: list[OsuBeatmapsetSearchItem] = field(default_factory=list)
    loading: bool = False
    error: str | None = None
    requested_at: float = 0.0
    loaded_at: float = 0.0


@dataclass(slots=True)
class OsuOfficialScore:
    score_id: int
    legacy_score_id: int = 0
    username: str = ""
    avatar_url: str = ""
    country_code: str = ""
    mods: list[str] = field(default_factory=list)
    total_score: int = 0
    max_combo: int = 0
    accuracy: float = 0.0
    passed: bool = True
    perfect: bool = False
    has_replay: bool = False
    ended_at: str = ""
    rank: str = ""
    ruleset: str = "osu"
    count_300: int = 0
    count_100: int = 0
    count_50: int = 0
    count_miss: int = 0
    local_path: str | None = None
    is_downloaded: bool = False
    is_downloading: bool = False
    download_progress: float = 0.0
    status_text: str = ""

    @property
    def mods_text(self) -> str:
        return "".join(self.mods) if self.mods else "NM"


@dataclass(slots=True)
class OsuOfficialScoreState:
    items: list[OsuOfficialScore] = field(default_factory=list)
    loading: bool = False
    error: str | None = None
    requested_at: float = 0.0
    loaded_at: float = 0.0
