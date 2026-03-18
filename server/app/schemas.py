from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PlayerIdentifyRequest(BaseModel):
    player_uuid: str = Field(min_length=8, max_length=64)
    nickname: str = Field(default="", max_length=16)
    status_text: str = Field(default="", max_length=512)


class PlayerPresenceResponse(BaseModel):
    player_uuid: str
    nickname: str
    online: bool
    status_text: str = ""
    last_seen: datetime


class PlayerStatusUpdateRequest(BaseModel):
    player_uuid: str = Field(min_length=8, max_length=64)
    status_text: str = Field(default="", max_length=512)


class ChannelCreateRequest(BaseModel):
    player_uuid: str
    name: str = Field(min_length=1, max_length=128)


class DirectMessageRequest(BaseModel):
    player_uuid: str
    target_player_uuid: str


class RelationshipRequest(BaseModel):
    player_uuid: str
    target_player_uuid: str
    enabled: bool = True


class ChannelResponse(BaseModel):
    channel_id: str
    name: str
    topic: str
    kind: str


class SharedBeatmapPayload(BaseModel):
    map_md5: str = Field(default="", max_length=64)
    title: str = Field(default="", max_length=512)
    version: str = Field(default="", max_length=256)
    mods: int = 0
    beatmap_id: int | None = None


class SharedReplayPayload(BaseModel):
    replay_id: str | None = Field(default=None, max_length=64)
    player_name: str = Field(default="", max_length=64)
    mods: int = 0
    map_md5: str = Field(default="", max_length=64)
    beatmap_id: int | None = None
    replay_hash: str = Field(default="", max_length=128)


class MessagePayload(BaseModel):
    kind: str = Field(min_length=1, max_length=64)
    beatmap: SharedBeatmapPayload | None = None
    replays: list[SharedReplayPayload] = Field(default_factory=list)


class MessageCreateRequest(BaseModel):
    player_uuid: str
    content: str = Field(min_length=1, max_length=2000)
    is_action: bool = False
    payload: MessagePayload | None = None


class MessageResponse(BaseModel):
    message_id: str
    channel_id: str
    sender_uuid: str | None
    sender_name: str
    content: str
    is_action: bool
    payload: MessagePayload | None = None
    created_at: datetime


class ReplayResponse(BaseModel):
    replay_id: str
    beatmap_id: int
    replay_hash: str
    player_name: str
    mods: int
    score: int
    max_combo: int
    n300: int
    n100: int
    n50: int
    nmiss: int
    map_md5: str
    views: int
    original_filename: str
    created_at: datetime


class IdentifyResponse(BaseModel):
    player_uuid: str
    nickname: str
    channels: list[ChannelResponse]
    presence: list[PlayerPresenceResponse]


class ReplayUploadResponse(BaseModel):
    replay: ReplayResponse
    duplicate: bool
