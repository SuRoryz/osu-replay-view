from __future__ import annotations

from pydantic import BaseModel, Field


class OsuAuthStartResponse(BaseModel):
    auth_url: str
    state: str


class OsuAccountResponse(BaseModel):
    user_id: int
    username: str
    avatar_url: str = ""
    country_code: str = ""
    cover_url: str = ""


class OsuAuthStatusResponse(BaseModel):
    linked: bool = False
    enabled: bool = False
    scopes: list[str] = Field(default_factory=list)
    account: OsuAccountResponse | None = None
    expires_at: float | None = None


class OsuBeatmapInfoResponse(BaseModel):
    beatmap_id: int
    version: str = ""
    difficulty_rating: float = 0.0
    mode: str = "osu"
    checksum: str = ""


class OsuBeatmapsetSearchItemResponse(BaseModel):
    beatmapset_id: int
    artist: str = ""
    title: str = ""
    creator: str = ""
    status: str = ""
    favourite_count: int = 0
    play_count: int = 0
    source: str = ""
    tags: str = ""
    cover_url: str = ""
    preview_url: str = ""
    beatmaps: list[OsuBeatmapInfoResponse] = Field(default_factory=list)


class OsuBeatmapsetSearchResponse(BaseModel):
    cursor_string: str = ""
    beatmapsets: list[OsuBeatmapsetSearchItemResponse] = Field(default_factory=list)


class OsuBeatmapLookupResponse(BaseModel):
    beatmap_id: int
    beatmapset_id: int | None = None
    checksum: str = ""
    mode: str = "osu"
    title: str = ""
    artist: str = ""
    creator: str = ""
    version: str = ""


class OsuOfficialScoreResponse(BaseModel):
    score_id: int
    legacy_score_id: int = 0
    username: str = ""
    avatar_url: str = ""
    country_code: str = ""
    mods: list[str] = Field(default_factory=list)
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
