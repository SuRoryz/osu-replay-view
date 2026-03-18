from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from osu_map.mods import mod_string


@dataclass(slots=True)
class OnlineUser:
    player_uuid: str
    nickname: str
    online: bool
    status_text: str = ""
    last_seen: datetime | None = None
    is_friend: bool = False
    is_blocked: bool = False


@dataclass(slots=True)
class ChatChannel:
    channel_id: str
    name: str
    topic: str = ""
    kind: str = "room"


@dataclass(slots=True)
class SharedBeatmap:
    map_md5: str
    title: str = ""
    version: str = ""
    mods: int = 0
    beatmap_id: int | None = None


@dataclass(slots=True)
class SharedReplay:
    replay_id: str | None
    player_name: str
    mods: int = 0
    map_md5: str = ""
    beatmap_id: int | None = None
    replay_hash: str = ""


@dataclass(slots=True)
class ChatMessagePayload:
    kind: str
    beatmap: SharedBeatmap | None = None
    replays: list[SharedReplay] = field(default_factory=list)


def chat_payload_to_dict(payload: ChatMessagePayload | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "kind": payload.kind,
        "beatmap": None if payload.beatmap is None else {
            "map_md5": payload.beatmap.map_md5,
            "title": payload.beatmap.title,
            "version": payload.beatmap.version,
            "mods": int(payload.beatmap.mods),
            "beatmap_id": payload.beatmap.beatmap_id,
        },
        "replays": [
            {
                "replay_id": replay.replay_id,
                "player_name": replay.player_name,
                "mods": int(replay.mods),
                "map_md5": replay.map_md5,
                "beatmap_id": replay.beatmap_id,
                "replay_hash": replay.replay_hash,
            }
            for replay in payload.replays
        ],
    }


def chat_payload_from_dict(value: Any) -> ChatMessagePayload | None:
    if not isinstance(value, dict):
        return None
    beatmap_value = value.get("beatmap")
    beatmap = None
    if isinstance(beatmap_value, dict):
        beatmap = SharedBeatmap(
            map_md5=str(beatmap_value.get("map_md5") or ""),
            title=str(beatmap_value.get("title") or ""),
            version=str(beatmap_value.get("version") or ""),
            mods=int(beatmap_value.get("mods") or 0),
            beatmap_id=(
                int(beatmap_value["beatmap_id"])
                if beatmap_value.get("beatmap_id") is not None
                else None
            ),
        )
    replays: list[SharedReplay] = []
    for replay_value in value.get("replays") or ():
        if not isinstance(replay_value, dict):
            continue
        replays.append(
            SharedReplay(
                replay_id=(
                    str(replay_value["replay_id"])
                    if replay_value.get("replay_id") is not None
                    else None
                ),
                player_name=str(replay_value.get("player_name") or ""),
                mods=int(replay_value.get("mods") or 0),
                map_md5=str(replay_value.get("map_md5") or ""),
                beatmap_id=(
                    int(replay_value["beatmap_id"])
                    if replay_value.get("beatmap_id") is not None
                    else None
                ),
                replay_hash=str(replay_value.get("replay_hash") or ""),
            )
        )
    kind = str(value.get("kind") or "").strip()
    if not kind:
        return None
    return ChatMessagePayload(kind=kind, beatmap=beatmap, replays=replays)


def format_shared_beatmap(beatmap: SharedBeatmap | None) -> str:
    if beatmap is None:
        return ""
    title = beatmap.title or "Unknown map"
    version = beatmap.version or "?"
    mods = mod_string(beatmap.mods) or "+NM"
    return f"{title} [{version}] {mods}"


def format_shared_replays(replays: list[SharedReplay]) -> str:
    parts = [
        f"{replay.player_name or 'Replay'} {mod_string(replay.mods) or '+NM'}"
        for replay in replays
    ]
    return ", ".join(parts)


def format_chat_payload(payload: ChatMessagePayload | None) -> str:
    if payload is None:
        return ""
    beatmap_text = format_shared_beatmap(payload.beatmap)
    if payload.kind == "now_playing_replays":
        replay_text = format_shared_replays(payload.replays)
        if replay_text:
            return f"{beatmap_text} - replays ({replay_text})"
    return beatmap_text


@dataclass(slots=True)
class ChatMessage:
    message_id: str
    channel_id: str
    sender_uuid: str | None
    sender_name: str
    content: str
    is_action: bool
    payload: ChatMessagePayload | None = None
    created_at: datetime | None = None
    local_only: bool = False


@dataclass(slots=True)
class OnlineReplayMetadata:
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
    created_at: datetime | None = None
    local_path: str | None = None
    is_downloaded: bool = False
    is_downloading: bool = False
    download_progress: float = 0.0
    upload_state: str = "unknown"
    status_text: str = ""


@dataclass(slots=True)
class ReplayTabState:
    items: list[OnlineReplayMetadata] = field(default_factory=list)
    loading: bool = False
    error: str | None = None
    loaded_at: float = 0.0
