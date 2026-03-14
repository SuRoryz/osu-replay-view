from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Player(Base):
    __tablename__ = "players"

    player_uuid: Mapped[str] = mapped_column(String(64), primary_key=True)
    nickname: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class FriendEdge(Base):
    __tablename__ = "friend_edges"
    __table_args__ = (UniqueConstraint("player_uuid", "target_player_uuid", name="uq_friend_edge"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    player_uuid: Mapped[str] = mapped_column(ForeignKey("players.player_uuid", ondelete="CASCADE"), nullable=False)
    target_player_uuid: Mapped[str] = mapped_column(ForeignKey("players.player_uuid", ondelete="CASCADE"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class BlockEdge(Base):
    __tablename__ = "block_edges"
    __table_args__ = (UniqueConstraint("player_uuid", "target_player_uuid", name="uq_block_edge"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    player_uuid: Mapped[str] = mapped_column(ForeignKey("players.player_uuid", ondelete="CASCADE"), nullable=False)
    target_player_uuid: Mapped[str] = mapped_column(ForeignKey("players.player_uuid", ondelete="CASCADE"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Channel(Base):
    __tablename__ = "channels"
    __table_args__ = (UniqueConstraint("name", name="uq_channel_name"),)

    channel_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    topic: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="room")
    owner_uuid: Mapped[str | None] = mapped_column(ForeignKey("players.player_uuid", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    members: Mapped[list["ChannelMembership"]] = relationship(back_populates="channel", cascade="all, delete-orphan")


class ChannelMembership(Base):
    __tablename__ = "channel_memberships"
    __table_args__ = (UniqueConstraint("channel_id", "player_uuid", name="uq_channel_membership"),)

    membership_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    channel_id: Mapped[str] = mapped_column(ForeignKey("channels.channel_id", ondelete="CASCADE"), nullable=False)
    player_uuid: Mapped[str] = mapped_column(ForeignKey("players.player_uuid", ondelete="CASCADE"), nullable=False)
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    channel: Mapped[Channel] = relationship(back_populates="members")


class Message(Base):
    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    channel_id: Mapped[str] = mapped_column(ForeignKey("channels.channel_id", ondelete="CASCADE"), nullable=False, index=True)
    sender_uuid: Mapped[str | None] = mapped_column(ForeignKey("players.player_uuid", ondelete="SET NULL"))
    sender_name: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_action: Mapped[bool] = mapped_column(default=False, nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)


class Replay(Base):
    __tablename__ = "replays"
    __table_args__ = (UniqueConstraint("beatmap_id", "replay_hash", name="uq_replay_hash_per_beatmap"),)

    replay_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    beatmap_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    replay_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    player_uuid: Mapped[str | None] = mapped_column(ForeignKey("players.player_uuid", ondelete="SET NULL"))
    player_name: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    mods: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_combo: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n300: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n100: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n50: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    nmiss: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    map_md5: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    views: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ReplayUploadEvent(Base):
    __tablename__ = "replay_upload_events"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    replay_id: Mapped[str | None] = mapped_column(ForeignKey("replays.replay_id", ondelete="SET NULL"))
    beatmap_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    replay_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    player_uuid: Mapped[str | None] = mapped_column(ForeignKey("players.player_uuid", ondelete="SET NULL"))
    duplicate: Mapped[bool] = mapped_column(nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
