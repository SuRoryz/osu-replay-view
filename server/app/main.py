from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import delete, select

from .chat_commands import dm_channel_name, normalize_channel_name
from .config import Settings
from .db import init_db, session_scope, build_session_factory
from .models import BlockEdge, Channel, ChannelMembership, FriendEdge, Message, Player, Replay, ReplayUploadEvent, utcnow
from .schemas import (
    ChannelCreateRequest,
    ChannelResponse,
    DirectMessageRequest,
    IdentifyResponse,
    MessageCreateRequest,
    MessagePayload,
    MessageResponse,
    PlayerIdentifyRequest,
    PlayerPresenceResponse,
    PlayerStatusUpdateRequest,
    RelationshipRequest,
    ReplayResponse,
    ReplayUploadResponse,
)


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._status_text: dict[str, str] = {}

    async def connect(self, player_uuid: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[player_uuid].add(websocket)

    def disconnect(self, player_uuid: str, websocket: WebSocket) -> None:
        sockets = self._connections.get(player_uuid)
        if sockets is None:
            return
        sockets.discard(websocket)
        if not sockets:
            self._connections.pop(player_uuid, None)
            self._status_text.pop(player_uuid, None)

    def online_ids(self) -> set[str]:
        return set(self._connections.keys())

    def status_text_for(self, player_uuid: str) -> str:
        return self._status_text.get(player_uuid, "")

    def set_status_text(self, player_uuid: str, status_text: str) -> bool:
        normalized = status_text[:512].strip()
        current = self._status_text.get(player_uuid, "")
        if current == normalized:
            return False
        if normalized:
            self._status_text[player_uuid] = normalized
        else:
            self._status_text.pop(player_uuid, None)
        return True

    async def send_to_player(self, player_uuid: str, payload: dict) -> None:
        encoded_payload = jsonable_encoder(payload)
        for websocket in list(self._connections.get(player_uuid, ())):
            await websocket.send_json(encoded_payload)

    async def broadcast(self, payload: dict) -> None:
        encoded_payload = jsonable_encoder(payload)
        for sockets in list(self._connections.values()):
            for websocket in list(sockets):
                await websocket.send_json(encoded_payload)


settings = Settings.load()
engine, SessionLocal = build_session_factory(settings)
settings.storage_root.mkdir(parents=True, exist_ok=True)
init_db(engine)

app = FastAPI(title="osu replay social server", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.manager = ConnectionManager()


@app.on_event("startup")
def startup() -> None:
    init_db(engine)
    with session_scope(SessionLocal) as session:
        lobby = session.scalar(select(Channel).where(Channel.name == "#lobby"))
        if lobby is None:
            session.add(Channel(name="#lobby", topic="Global chat", kind="room"))


def replay_to_schema(replay: Replay) -> ReplayResponse:
    return ReplayResponse(
        replay_id=replay.replay_id,
        beatmap_id=replay.beatmap_id,
        replay_hash=replay.replay_hash,
        player_name=replay.player_name,
        mods=replay.mods,
        score=replay.score,
        max_combo=replay.max_combo,
        n300=replay.n300,
        n100=replay.n100,
        n50=replay.n50,
        nmiss=replay.nmiss,
        map_md5=replay.map_md5,
        views=replay.views,
        original_filename=replay.original_filename,
        created_at=replay.created_at,
    )


def channel_to_schema(channel: Channel) -> ChannelResponse:
    return ChannelResponse(
        channel_id=channel.channel_id,
        name=channel.name,
        topic=channel.topic,
        kind=channel.kind,
    )


def message_to_schema(message: Message) -> MessageResponse:
    payload = None
    if message.metadata_json:
        try:
            payload = MessagePayload.model_validate(json.loads(message.metadata_json))
        except Exception:
            payload = None
    return MessageResponse(
        message_id=message.message_id,
        channel_id=message.channel_id,
        sender_uuid=message.sender_uuid,
        sender_name=message.sender_name,
        content=message.content,
        is_action=message.is_action,
        payload=payload,
        created_at=message.created_at,
    )


def normalize_beatmap_id(value: int) -> int:
    return int(value) & 0x7FFFFFFF


def presence_snapshot(session) -> list[PlayerPresenceResponse]:
    online = app.state.manager.online_ids()
    players = session.scalars(select(Player).order_by(Player.nickname.asc(), Player.player_uuid.asc())).all()
    return [
        PlayerPresenceResponse(
            player_uuid=player.player_uuid,
            nickname=player.nickname,
            online=player.player_uuid in online,
            status_text=app.state.manager.status_text_for(player.player_uuid),
            last_seen=player.last_seen,
        )
        for player in players
    ]


def player_channels(session, player_uuid: str) -> list[Channel]:
    memberships = session.scalars(
        select(Channel)
        .join(ChannelMembership, ChannelMembership.channel_id == Channel.channel_id)
        .where(ChannelMembership.player_uuid == player_uuid)
    ).all()
    public_rooms = session.scalars(select(Channel).where(Channel.kind == "room")).all()
    seen: set[str] = set()
    result: list[Channel] = []
    for channel in [*public_rooms, *memberships]:
        if channel.channel_id not in seen:
            seen.add(channel.channel_id)
            result.append(channel)
    return sorted(result, key=lambda item: (item.kind != "room", item.name.lower()))


def ensure_player(session, player_uuid: str, nickname: str) -> Player:
    nickname = nickname[:16].strip()
    player = session.get(Player, player_uuid)
    if player is None:
        player = Player(player_uuid=player_uuid, nickname=nickname)
        session.add(player)
    else:
        player.last_seen = utcnow()
        if nickname:
            player.nickname = nickname
    if not player.nickname:
        player.nickname = f"user-{player_uuid[:8]}"
    # Flush immediately so later inserts that reference this player_uuid
    # satisfy PostgreSQL foreign key checks in the same request.
    session.flush()
    return player


def ensure_membership(session, channel_id: str, player_uuid: str) -> None:
    membership = session.scalar(
        select(ChannelMembership).where(
            ChannelMembership.channel_id == channel_id,
            ChannelMembership.player_uuid == player_uuid,
        )
    )
    if membership is None:
        session.add(ChannelMembership(channel_id=channel_id, player_uuid=player_uuid))


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/session/identify", response_model=IdentifyResponse)
async def identify(body: PlayerIdentifyRequest):
    with session_scope(SessionLocal) as session:
        player = ensure_player(session, body.player_uuid, body.nickname)
        app.state.manager.set_status_text(player.player_uuid, body.status_text)
        lobby = session.scalar(select(Channel).where(Channel.name == "#lobby"))
        if lobby is None:
            lobby = Channel(name="#lobby", topic="Global chat", kind="room")
            session.add(lobby)
            session.flush()
        ensure_membership(session, lobby.channel_id, player.player_uuid)
        session.flush()
        channels = [channel_to_schema(channel) for channel in player_channels(session, player.player_uuid)]
        presence = presence_snapshot(session)
        response = IdentifyResponse(
            player_uuid=player.player_uuid,
            nickname=player.nickname,
            channels=channels,
            presence=presence,
        )
    await app.state.manager.broadcast({"type": "presence_changed"})
    return response


@app.get("/social/presence", response_model=list[PlayerPresenceResponse])
def social_presence():
    with session_scope(SessionLocal) as session:
        return presence_snapshot(session)


@app.post("/session/status")
async def session_status(body: PlayerStatusUpdateRequest):
    changed = app.state.manager.set_status_text(body.player_uuid, body.status_text)
    if changed:
        await app.state.manager.broadcast({"type": "presence_changed"})
    return {"ok": True}


@app.get("/social/channels", response_model=list[ChannelResponse])
def social_channels(player_uuid: str = Query(...)):
    with session_scope(SessionLocal) as session:
        return [channel_to_schema(channel) for channel in player_channels(session, player_uuid)]


@app.post("/social/channels", response_model=ChannelResponse)
async def create_channel(body: ChannelCreateRequest):
    with session_scope(SessionLocal) as session:
        ensure_player(session, body.player_uuid, "")
        name = normalize_channel_name(body.name)
        channel = session.scalar(select(Channel).where(Channel.name == name))
        if channel is None:
            channel = Channel(name=name, topic="Custom room", kind="room", owner_uuid=body.player_uuid)
            session.add(channel)
            session.flush()
        ensure_membership(session, channel.channel_id, body.player_uuid)
        payload = channel_to_schema(channel)
    await app.state.manager.broadcast({"type": "channel_created", "channel": payload.model_dump()})
    return payload


@app.post("/social/dm", response_model=ChannelResponse)
async def open_dm(body: DirectMessageRequest):
    with session_scope(SessionLocal) as session:
        ensure_player(session, body.player_uuid, "")
        ensure_player(session, body.target_player_uuid, "")
        name = dm_channel_name(body.player_uuid, body.target_player_uuid)
        channel = session.scalar(select(Channel).where(Channel.name == name))
        if channel is None:
            channel = Channel(name=name, topic="Direct message", kind="dm", owner_uuid=body.player_uuid)
            session.add(channel)
            session.flush()
        ensure_membership(session, channel.channel_id, body.player_uuid)
        ensure_membership(session, channel.channel_id, body.target_player_uuid)
        payload = channel_to_schema(channel)
    await app.state.manager.send_to_player(body.player_uuid, {"type": "channel_created", "channel": payload.model_dump()})
    await app.state.manager.send_to_player(body.target_player_uuid, {"type": "channel_created", "channel": payload.model_dump()})
    return payload


@app.get("/social/channels/{channel_id}/messages", response_model=list[MessageResponse])
def channel_messages(channel_id: str, limit: int = Query(100, ge=1, le=300)):
    with session_scope(SessionLocal) as session:
        rows = session.scalars(
            select(Message)
            .where(Message.channel_id == channel_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        ).all()
        rows.reverse()
        return [message_to_schema(row) for row in rows]


@app.post("/social/channels/{channel_id}/messages", response_model=MessageResponse)
async def post_message(channel_id: str, body: MessageCreateRequest):
    with session_scope(SessionLocal) as session:
        player = session.get(Player, body.player_uuid)
        if player is None:
            raise HTTPException(status_code=404, detail="Player not found.")
        channel = session.get(Channel, channel_id)
        if channel is None:
            raise HTTPException(status_code=404, detail="Channel not found.")
        ensure_membership(session, channel.channel_id, player.player_uuid)
        message = Message(
            channel_id=channel.channel_id,
            sender_uuid=player.player_uuid,
            sender_name=player.nickname,
            content=body.content,
            is_action=body.is_action,
            metadata_json=(
                json.dumps(body.payload.model_dump(mode="json"))
                if body.payload is not None
                else None
            ),
        )
        session.add(message)
        session.flush()
        payload = message_to_schema(message)
    await app.state.manager.broadcast({"type": "message_created", "message": payload.model_dump()})
    return payload


@app.post("/social/friends")
async def set_friend(body: RelationshipRequest):
    with session_scope(SessionLocal) as session:
        ensure_player(session, body.player_uuid, "")
        ensure_player(session, body.target_player_uuid, "")
        if body.enabled:
            existing = session.scalar(
                select(FriendEdge).where(
                    FriendEdge.player_uuid == body.player_uuid,
                    FriendEdge.target_player_uuid == body.target_player_uuid,
                )
            )
            if existing is None:
                session.add(FriendEdge(player_uuid=body.player_uuid, target_player_uuid=body.target_player_uuid))
        else:
            session.execute(
                delete(FriendEdge).where(
                    FriendEdge.player_uuid == body.player_uuid,
                    FriendEdge.target_player_uuid == body.target_player_uuid,
                )
            )
    return {"ok": True}


@app.post("/social/blocks")
async def set_block(body: RelationshipRequest):
    with session_scope(SessionLocal) as session:
        ensure_player(session, body.player_uuid, "")
        ensure_player(session, body.target_player_uuid, "")
        if body.enabled:
            existing = session.scalar(
                select(BlockEdge).where(
                    BlockEdge.player_uuid == body.player_uuid,
                    BlockEdge.target_player_uuid == body.target_player_uuid,
                )
            )
            if existing is None:
                session.add(BlockEdge(player_uuid=body.player_uuid, target_player_uuid=body.target_player_uuid))
        else:
            session.execute(
                delete(BlockEdge).where(
                    BlockEdge.player_uuid == body.player_uuid,
                    BlockEdge.target_player_uuid == body.target_player_uuid,
                )
            )
    return {"ok": True}


def _parse_replay_summary(path: str) -> dict[str, int | str]:
    try:
        from osupyparser.osr.osr_parser import ReplayFile

        replay = ReplayFile.from_file(path)
        return {
            "player_name": replay.player_name or "",
            "mods": int(replay.mods),
            "score": int(replay.score),
            "max_combo": int(replay.max_combo),
            "n300": int(replay.n300),
            "n100": int(replay.n100),
            "n50": int(replay.n50),
            "nmiss": int(replay.nmiss),
            "map_md5": replay.map_md5 or "",
        }
    except Exception:
        filename = Path(path).stem
        return {
            "player_name": filename[:64],
            "mods": 0,
            "score": 0,
            "max_combo": 0,
            "n300": 0,
            "n100": 0,
            "n50": 0,
            "nmiss": 0,
            "map_md5": "",
        }


@app.get("/replays", response_model=list[ReplayResponse])
def list_replays(beatmap_id: int = Query(...)):
    beatmap_id = normalize_beatmap_id(beatmap_id)
    with session_scope(SessionLocal) as session:
        rows = session.scalars(
            select(Replay).where(Replay.beatmap_id == beatmap_id).order_by(Replay.views.desc(), Replay.created_at.desc())
        ).all()
        return [replay_to_schema(row) for row in rows]


@app.post("/replays/upload", response_model=ReplayUploadResponse)
async def upload_replay(
    beatmap_id: int = Query(...),
    player_uuid: str = Query(""),
    file: UploadFile = File(...),
):
    beatmap_id = normalize_beatmap_id(beatmap_id)
    suffix = Path(file.filename or "upload.osr").suffix or ".osr"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        payload = await file.read()
        temp_file.write(payload)
        temp_path = temp_file.name
    replay_hash = hashlib.sha256(payload + f":{beatmap_id}".encode("utf-8")).hexdigest()
    summary = _parse_replay_summary(temp_path)
    duplicate = False
    with session_scope(SessionLocal) as session:
        if player_uuid:
            ensure_player(session, player_uuid, "")
        existing = session.scalar(
            select(Replay).where(Replay.beatmap_id == beatmap_id, Replay.replay_hash == replay_hash)
        )
        if existing is not None:
            duplicate = True
            session.add(
                ReplayUploadEvent(
                    replay_id=existing.replay_id,
                    beatmap_id=beatmap_id,
                    replay_hash=replay_hash,
                    player_uuid=player_uuid or None,
                    duplicate=True,
                )
            )
            replay_schema = replay_to_schema(existing)
        else:
            replay_id = os.path.splitext(os.path.basename(temp_path))[0]
            storage_path = settings.storage_root / f"{replay_hash}{suffix.lower()}"
            storage_path.write_bytes(payload)
            replay = Replay(
                beatmap_id=beatmap_id,
                replay_hash=replay_hash,
                player_uuid=player_uuid or None,
                original_filename=file.filename or storage_path.name,
                storage_path=str(storage_path),
                **summary,
            )
            session.add(replay)
            session.flush()
            session.add(
                ReplayUploadEvent(
                    replay_id=replay.replay_id,
                    beatmap_id=beatmap_id,
                    replay_hash=replay_hash,
                    player_uuid=player_uuid or None,
                    duplicate=False,
                )
            )
            replay_schema = replay_to_schema(replay)
    try:
        os.unlink(temp_path)
    except OSError:
        pass
    await app.state.manager.broadcast({"type": "replay_changed", "beatmap_id": beatmap_id})
    return ReplayUploadResponse(replay=replay_schema, duplicate=duplicate)


@app.get("/replays/{replay_id}/download")
def download_replay(replay_id: str):
    with session_scope(SessionLocal) as session:
        replay = session.get(Replay, replay_id)
        if replay is None:
            raise HTTPException(status_code=404, detail="Replay not found.")
        replay.views += 1
        path = Path(replay.storage_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Replay file missing.")
        filename = replay.original_filename or f"{replay.replay_hash}.osr"
        return FileResponse(path, filename=filename, media_type="application/octet-stream")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, player_uuid: str = Query(...)):
    await app.state.manager.connect(player_uuid, websocket)
    await app.state.manager.broadcast({"type": "presence_changed"})
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        app.state.manager.disconnect(player_uuid, websocket)
        await app.state.manager.broadcast({"type": "presence_changed"})
