from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4
from urllib.parse import urlencode, urlparse, urlunparse

import requests
from websocket import WebSocketApp

from social.commands import execute_social_command
from social.models import (
    ChatChannel,
    ChatMessage,
    ChatMessagePayload,
    OnlineReplayMetadata,
    OnlineUser,
    ReplayTabState,
    chat_payload_from_dict,
    chat_payload_to_dict,
)
from social.storage import SocialLocalState


def _parse_datetime(value) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


class SocialClient:
    def __init__(self) -> None:
        self.local_state = SocialLocalState.load()
        self.player_uuid = (self.local_state.player_uuid or str(uuid4())).strip()
        if self.local_state.player_uuid != self.player_uuid:
            self.local_state.player_uuid = self.player_uuid
            self.local_state.save()
        self.base_url = self.local_state.server_url.rstrip("/")
        self.ws_url = self._build_ws_url(self.base_url)
        self.nickname = ""
        self.connected = False
        self.connecting = False
        self.connection_error: str | None = None
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._ws_app = None
        self._presence_refresh_at = 0.0
        self._message_loads: set[str] = set()
        self._replay_loads: set[int] = set()
        self._current_request_threads: set[threading.Thread] = set()
        self.users: dict[str, OnlineUser] = {}
        self.channels: dict[str, ChatChannel] = {}
        self.channel_order: list[str] = []
        self.messages: dict[str, list[ChatMessage]] = defaultdict(list)
        self.active_channel_id: str | None = None
        self.replay_tabs: dict[int, ReplayTabState] = {}
        self._pending_room_focus: str | None = None
        self._pending_dm_message: tuple[str, str] | None = None
        self.command_context_provider: Callable[[], object | None] | None = None
        self.chat_payload_handler: Callable[[ChatMessagePayload], None] | None = None
        self.download_event_handler: Callable[[str, dict], None] | None = None
        self._presence_status_text = ""
        self._last_synced_presence_status_text = ""
        self._presence_status_sync_in_flight = False

    @staticmethod
    def _build_ws_url(base_url: str) -> str:
        parsed = urlparse(base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunparse((scheme, parsed.netloc, "/ws", "", "", ""))

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._ws_app is not None:
            try:
                self._ws_app.close()
            except Exception:
                pass

    def update(self, nickname: str, status_text: str = "") -> None:
        nickname = (nickname or "").strip()[:16] or f"user-{self.player_uuid[:8]}"
        normalized_status = str(status_text or "").strip()[:512]
        self.nickname = nickname
        self._presence_status_text = normalized_status
        self._drain_events()
        if not self.connected and not self.connecting:
            self._identify()
        elif time.time() >= self._presence_refresh_at and not self.connecting:
            self.refresh_presence()
            self._presence_refresh_at = time.time() + 20.0
        if (
            self.connected
            and not self.connecting
            and not self._presence_status_sync_in_flight
            and self._presence_status_text != self._last_synced_presence_status_text
        ):
            self._sync_presence_status()

    def push_system_message(self, text: str, channel_id: str | None = None) -> None:
        with self._lock:
            target = channel_id or self.active_channel_id
            if target is None:
                target = self._ensure_local_channel("#lobby", "room")
            message = ChatMessage(
                message_id=f"local:{uuid4()}",
                channel_id=target,
                sender_uuid=None,
                sender_name="system",
                content=text,
                is_action=False,
                created_at=datetime.utcnow(),
                local_only=True,
            )
            self.messages[target].append(message)

    def _request_json(self, method: str, path: str, *, params=None, json_body=None, timeout: float = 10.0):
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            params=params,
            json=json_body,
            timeout=timeout,
        )
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()

    def _spawn(self, event_name: str, worker) -> None:
        thread = threading.Thread(target=self._run_worker, args=(event_name, worker), daemon=True)
        self._current_request_threads.add(thread)
        thread.start()

    def _run_worker(self, event_name: str, worker) -> None:
        try:
            payload = worker()
            self._queue.put((event_name, payload))
        except Exception as exc:  # noqa: BLE001
            self._queue.put(("error", (event_name, str(exc))))

    def _notify_download_event(self, event_type: str, **payload) -> None:
        if self.download_event_handler is None:
            return
        self.download_event_handler(event_type, payload)

    def _identify(self) -> None:
        self.connecting = True

        def worker():
            return self._request_json(
                "POST",
                "/session/identify",
                json_body={
                    "player_uuid": self.player_uuid,
                    "nickname": self.nickname,
                    "status_text": self._presence_status_text,
                },
            )

        self._spawn("identify", worker)

    def _sync_presence_status(self) -> None:
        self._presence_status_sync_in_flight = True
        requested_status = self._presence_status_text

        def worker():
            self._request_json(
                "POST",
                "/session/status",
                json_body={"player_uuid": self.player_uuid, "status_text": requested_status},
            )
            return requested_status

        self._spawn("presence_status", worker)

    def refresh_presence(self) -> None:
        self._spawn("presence", lambda: self._request_json("GET", "/social/presence"))

    def refresh_channels(self) -> None:
        self._spawn(
            "channels",
            lambda: self._request_json("GET", "/social/channels", params={"player_uuid": self.player_uuid}),
        )

    def create_room(self, name: str) -> None:
        self._spawn(
            "create_room",
            lambda: self._request_json(
                "POST",
                "/social/channels",
                json_body={"player_uuid": self.player_uuid, "name": name},
            ),
        )

    def open_dm(self, target_player_uuid: str) -> None:
        self._spawn(
            "open_dm",
            lambda: self._request_json(
                "POST",
                "/social/dm",
                json_body={"player_uuid": self.player_uuid, "target_player_uuid": target_player_uuid},
            ),
        )

    def open_dm_by_name(self, nickname: str) -> None:
        user = self._find_user_by_name(nickname)
        if user is None:
            self.push_system_message(f"User not found: {nickname}")
            return
        self.open_dm(user.player_uuid)

    def send_private_message(self, nickname: str, text: str) -> None:
        user = self._find_user_by_name(nickname)
        if user is None:
            self.push_system_message(f"User not found: {nickname}")
            return
        self._pending_dm_message = (user.player_uuid, text)
        self.open_dm(user.player_uuid)

    def send_message_input(self, text: str) -> None:
        if execute_social_command(self, text) is not None:
            return
        self.send_message(text)

    def send_message(self, content: str, *, is_action: bool = False, payload: ChatMessagePayload | None = None) -> None:
        channel_id = self.active_channel_id
        if not channel_id:
            self.push_system_message("No active chat channel selected.")
            return
        payload_dict = {
            "player_uuid": self.player_uuid,
            "content": content,
            "is_action": is_action,
            "payload": chat_payload_to_dict(payload),
        }
        self._spawn(
            "message_sent",
            lambda: self._request_json("POST", f"/social/channels/{channel_id}/messages", json_body=payload_dict),
        )

    def activate_message_payload(self, payload: ChatMessagePayload | None) -> bool:
        if payload is None or self.chat_payload_handler is None:
            return False
        self.chat_payload_handler(payload)
        return True

    def fetch_channel_messages(self, channel_id: str) -> None:
        if channel_id in self._message_loads:
            return
        self._message_loads.add(channel_id)
        self._spawn(
            f"messages:{channel_id}",
            lambda: self._request_json("GET", f"/social/channels/{channel_id}/messages", params={"limit": 120}),
        )

    def select_channel(self, channel_id: str) -> None:
        with self._lock:
            if channel_id not in self.channels:
                return
            self.active_channel_id = channel_id
        self.fetch_channel_messages(channel_id)

    def close_active_channel(self) -> None:
        with self._lock:
            channel_id = self.active_channel_id
            if channel_id is None:
                return
            channel = self.channels.get(channel_id)
            if channel is not None and channel.kind == "dm":
                self.channels.pop(channel_id, None)
                self.channel_order = [item for item in self.channel_order if item != channel_id]
                self.messages.pop(channel_id, None)
            self.active_channel_id = self.channel_order[0] if self.channel_order else None

    def clear_active_channel_messages(self) -> None:
        with self._lock:
            if self.active_channel_id:
                self.messages[self.active_channel_id] = []

    def set_friend(self, target_player_uuid: str, enabled: bool) -> None:
        self.local_state.set_friend(target_player_uuid, enabled)
        user = self.users.get(target_player_uuid)
        if user is not None:
            user.is_friend = enabled
        self._spawn(
            "friend_set",
            lambda: self._request_json(
                "POST",
                "/social/friends",
                json_body={"player_uuid": self.player_uuid, "target_player_uuid": target_player_uuid, "enabled": enabled},
            ),
        )

    def set_blocked(self, target_player_uuid: str, enabled: bool) -> None:
        self.local_state.set_blocked(target_player_uuid, enabled)
        user = self.users.get(target_player_uuid)
        if user is not None:
            user.is_blocked = enabled
        self._spawn(
            "block_set",
            lambda: self._request_json(
                "POST",
                "/social/blocks",
                json_body={"player_uuid": self.player_uuid, "target_player_uuid": target_player_uuid, "enabled": enabled},
            ),
        )

    def filtered_users(self, tab: str, query: str) -> list[OnlineUser]:
        query_norm = query.strip().lower()
        values = list(self.users.values())
        if tab == "friends":
            values = [item for item in values if item.is_friend]
        if query_norm:
            values = [item for item in values if query_norm in item.nickname.lower()]
        values.sort(key=lambda item: (not item.online, item.nickname.lower()))
        return values

    def fetch_online_replays(self, beatmap_id: int, *, force: bool = False) -> ReplayTabState:
        state = self.replay_tabs.setdefault(beatmap_id, ReplayTabState())
        if state.loading or (state.items and not force):
            return state
        state.loading = True
        state.error = None
        self._replay_loads.add(beatmap_id)
        self._spawn(
            f"replays:{beatmap_id}",
            lambda: self._request_json("GET", "/replays", params={"beatmap_id": beatmap_id}),
        )
        return state

    def online_replays(self, beatmap_id: int) -> ReplayTabState:
        return self.replay_tabs.setdefault(beatmap_id, ReplayTabState())

    def download_replay(self, replay_id: str, target_dir: str) -> None:
        with self._lock:
            replay = self._find_replay_by_id(replay_id)
            if replay is None or replay.is_downloading:
                return
            replay.is_downloading = True
            replay.download_progress = 0.0
            replay.status_text = "Downloading..."
            self._notify_download_event(
                "started",
                replay_id=replay_id,
                progress=0.0,
                replay=replay,
            )

        def worker():
            response = requests.get(f"{self.base_url}/replays/{replay_id}/download", stream=True, timeout=30)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            filename = response.headers.get("content-disposition", "")
            raw_name = replay.original_filename or f"{replay_id}.osr"
            if "filename=" in filename:
                raw_name = filename.split("filename=")[-1].strip("\"'")
            safe_name = raw_name.replace("/", "_").replace("\\", "_")
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            name_path = Path(safe_name)
            suffix = name_path.suffix or ".osr"
            stem = name_path.stem or replay_id
            unique_name = f"{stem} [{replay_id[:8]}]{suffix}"
            local_path = str(Path(target_dir) / unique_name)
            downloaded = 0
            with open(local_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    progress = 0.0 if total <= 0 else min(1.0, downloaded / total)
                    self._queue.put(("download_progress", (replay_id, progress)))
            return replay_id, local_path

        self._spawn(f"download:{replay_id}", worker)

    def delete_downloaded_replay(self, replay_id: str) -> None:
        replay = self._find_replay_by_id(replay_id)
        if replay is None or not replay.local_path:
            return
        try:
            os.remove(replay.local_path)
        except OSError:
            pass
        replay.local_path = None
        replay.is_downloaded = False
        replay.download_progress = 0.0
        replay.status_text = ""
        self.local_state.forget_download(replay_id)

    def upload_local_replay(self, replay_path: str, beatmap_id: int) -> None:
        replay_path = str(replay_path)
        self._spawn(
            f"upload:{replay_path}",
            lambda: self._upload_worker(replay_path, beatmap_id),
        )

    def _upload_worker(self, replay_path: str, beatmap_id: int):
        with open(replay_path, "rb") as handle:
            response = requests.post(
                f"{self.base_url}/replays/upload",
                params={"beatmap_id": beatmap_id, "player_uuid": self.player_uuid},
                files={"file": (Path(replay_path).name, handle, "application/octet-stream")},
                timeout=60,
            )
        response.raise_for_status()
        return replay_path, response.json()

    def is_uploaded_path(self, replay_path: str) -> bool:
        return replay_path in self.local_state.replay_uploads

    def downloaded_path_for_replay(self, replay_id: str) -> str | None:
        return self.local_state.replay_downloads.get(replay_id)

    def _find_user_by_name(self, nickname: str) -> OnlineUser | None:
        needle = nickname.strip().lower()
        if not needle:
            return None
        for user in self.users.values():
            if user.nickname.lower() == needle:
                return user
        return None

    def _find_replay_by_id(self, replay_id: str) -> OnlineReplayMetadata | None:
        for state in self.replay_tabs.values():
            for item in state.items:
                if item.replay_id == replay_id:
                    return item
        return None

    def _ensure_local_channel(self, name: str, kind: str) -> str:
        for channel_id, channel in self.channels.items():
            if channel.name == name:
                return channel_id
        channel_id = f"local:{uuid4()}"
        self.channels[channel_id] = ChatChannel(channel_id=channel_id, name=name, kind=kind)
        self.channel_order.append(channel_id)
        if self.active_channel_id is None:
            self.active_channel_id = channel_id
        return channel_id

    def _start_ws(self) -> None:
        if self._ws_thread is not None and self._ws_thread.is_alive():
            return
        query = urlencode({"player_uuid": self.player_uuid})
        ws_url = f"{self.ws_url}?{query}"

        def on_message(_ws, message):
            try:
                payload = json.loads(message)
            except ValueError:
                return
            self._queue.put(("ws_event", payload))

        def on_error(_ws, error):
            self._queue.put(("error", f"websocket: {error}"))

        def on_close(_ws, *_args):
            self._queue.put(("ws_closed", None))

        self._ws_app = WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close)
        self._ws_thread = threading.Thread(target=self._ws_app.run_forever, daemon=True)
        self._ws_thread.start()

    def _drain_events(self) -> None:
        while True:
            try:
                name, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if name == "identify":
                self.connected = True
                self.connecting = False
                self.connection_error = None
                self._apply_identify(payload)
                self._start_ws()
            elif name == "presence":
                self.connection_error = None
                self._apply_presence(payload)
            elif name == "presence_status":
                self.connection_error = None
                self._last_synced_presence_status_text = str(payload or "")
                self._presence_status_sync_in_flight = False
            elif name == "channels":
                self.connection_error = None
                self._apply_channels(payload)
            elif name == "create_room":
                self.connection_error = None
                self._upsert_channel(payload, focus=True)
            elif name == "open_dm":
                self.connection_error = None
                channel = self._upsert_channel(payload, focus=True)
                if self._pending_dm_message is not None:
                    target_uuid, text = self._pending_dm_message
                    if channel is not None and f":{target_uuid}" in channel.name:
                        self.active_channel_id = channel.channel_id
                        self.send_message(text)
                        self._pending_dm_message = None
            elif name.startswith("messages:"):
                self.connection_error = None
                channel_id = name.split(":", 1)[1]
                self._message_loads.discard(channel_id)
                self._apply_messages(channel_id, payload)
            elif name == "message_sent":
                self.connection_error = None
                self._append_message(payload)
            elif name.startswith("replays:"):
                self.connection_error = None
                beatmap_id = int(name.split(":", 1)[1])
                self._replay_loads.discard(beatmap_id)
                self._apply_replays(beatmap_id, payload)
            elif name == "download_progress":
                replay_id, progress = payload
                replay = self._find_replay_by_id(replay_id)
                if replay is not None:
                    replay.download_progress = progress
                    replay.status_text = f"{int(progress * 100):d}%"
                    self._notify_download_event(
                        "progress",
                        replay_id=replay_id,
                        progress=progress,
                        replay=replay,
                    )
            elif name.startswith("download:"):
                self.connection_error = None
                replay_id, local_path = payload
                replay = self._find_replay_by_id(replay_id)
                if replay is not None:
                    replay.local_path = local_path
                    replay.is_downloaded = True
                    replay.is_downloading = False
                    replay.download_progress = 1.0
                    replay.status_text = "Downloaded"
                    replay.views += 1
                    self.local_state.remember_download(replay_id, local_path)
                    self._notify_download_event(
                        "finished",
                        replay_id=replay_id,
                        progress=1.0,
                        local_path=local_path,
                        replay=replay,
                    )
            elif name.startswith("upload:"):
                self.connection_error = None
                replay_path, response_payload = payload
                replay_id = response_payload["replay"]["replay_id"]
                self.local_state.remember_upload(replay_path, replay_id)
                beatmap_id = int(response_payload["replay"]["beatmap_id"])
                self.fetch_online_replays(beatmap_id, force=True)
            elif name == "ws_event":
                self._handle_ws_event(payload)
            elif name == "ws_closed":
                self.connected = False
                self.connecting = False
                self._presence_status_sync_in_flight = False
            elif name == "friend_set" or name == "block_set":
                pass
            elif name == "error":
                event_name = ""
                error_text = str(payload)
                if isinstance(payload, tuple):
                    event_name = str(payload[0])
                    error_text = str(payload[1])
                self.connection_error = f"{event_name}: {error_text}" if event_name else error_text
                self.connecting = False
                if event_name in {"identify", "presence_status"}:
                    self._presence_status_sync_in_flight = False
                if event_name.startswith("replays:"):
                    beatmap_id = int(event_name.split(":", 1)[1])
                    self._replay_loads.discard(beatmap_id)
                    state = self.replay_tabs.setdefault(beatmap_id, ReplayTabState())
                    state.loading = False
                    state.error = error_text
                elif event_name.startswith("download:"):
                    replay_id = event_name.split(":", 1)[1]
                    replay = self._find_replay_by_id(replay_id)
                    if replay is not None:
                        replay.is_downloading = False
                        replay.download_progress = 0.0
                        replay.status_text = "Download failed"
                        self._notify_download_event(
                            "failed",
                            replay_id=replay_id,
                            error=error_text,
                            replay=replay,
                        )

    def _apply_identify(self, payload: dict) -> None:
        self._apply_channels(payload.get("channels", []))
        self._apply_presence(payload.get("presence", []))
        if self.active_channel_id is None and self.channel_order:
            self.active_channel_id = self.channel_order[0]
        if self.active_channel_id:
            self.fetch_channel_messages(self.active_channel_id)

    def _presence_row_priority(self, row: dict) -> tuple[int, int, float, str]:
        player_uuid = str(row.get("player_uuid") or "")
        online = 1 if bool(row.get("online")) else 0
        parsed_last_seen = _parse_datetime(row.get("last_seen"))
        last_seen_ts = parsed_last_seen.timestamp() if parsed_last_seen is not None else float("-inf")
        is_self = 1 if player_uuid == self.player_uuid else 0
        return (is_self, online, last_seen_ts, player_uuid)

    def _dedupe_presence_rows(self, rows: list[dict]) -> list[dict]:
        best_by_nickname: dict[str, dict] = {}
        for row in rows:
            nickname = str(row.get("nickname") or "").strip().casefold()
            if not nickname:
                continue
            current = best_by_nickname.get(nickname)
            if current is None or self._presence_row_priority(row) > self._presence_row_priority(current):
                best_by_nickname[nickname] = row

        deduped: list[dict] = []
        for row in rows:
            nickname = str(row.get("nickname") or "").strip().casefold()
            if not nickname:
                deduped.append(row)
                continue
            if best_by_nickname.get(nickname) is row:
                deduped.append(row)
        return deduped

    def _apply_presence(self, rows: list[dict]) -> None:
        rows = self._dedupe_presence_rows(rows)
        with self._lock:
            next_users: dict[str, OnlineUser] = {}
            for row in rows:
                player_uuid = str(row["player_uuid"])
                next_users[player_uuid] = OnlineUser(
                    player_uuid=player_uuid,
                    nickname=str(row.get("nickname") or f"user-{player_uuid[:8]}"),
                    online=bool(row.get("online")),
                    status_text=str(row.get("status_text") or ""),
                    last_seen=_parse_datetime(row.get("last_seen")),
                    is_friend=player_uuid in self.local_state.friends,
                    is_blocked=player_uuid in self.local_state.blocked,
                )
            self.users = next_users

    def _apply_channels(self, rows: list[dict]) -> None:
        for row in rows:
            self._upsert_channel(row)

    def _upsert_channel(self, row: dict, *, focus: bool = False) -> ChatChannel:
        channel = ChatChannel(
            channel_id=str(row["channel_id"]),
            name=str(row["name"]),
            topic=str(row.get("topic") or ""),
            kind=str(row.get("kind") or "room"),
        )
        self.channels[channel.channel_id] = channel
        if channel.channel_id not in self.channel_order:
            self.channel_order.append(channel.channel_id)
        self.channel_order.sort(key=lambda item: (self.channels[item].kind != "room", self.channels[item].name.lower()))
        if self.active_channel_id is None or focus:
            self.active_channel_id = channel.channel_id
        self.fetch_channel_messages(channel.channel_id)
        return channel

    def _apply_messages(self, channel_id: str, rows: list[dict]) -> None:
        self.messages[channel_id] = [self._message_from_row(row) for row in rows]

    def _append_message(self, row: dict) -> None:
        message = self._message_from_row(row)
        bucket = self.messages[message.channel_id]
        if any(item.message_id == message.message_id for item in bucket):
            return
        bucket.append(message)
        if len(bucket) > 200:
            del bucket[:-200]

    def _message_from_row(self, row: dict) -> ChatMessage:
        return ChatMessage(
            message_id=str(row["message_id"]),
            channel_id=str(row["channel_id"]),
            sender_uuid=row.get("sender_uuid"),
            sender_name=str(row.get("sender_name") or "unknown"),
            content=str(row.get("content") or ""),
            is_action=bool(row.get("is_action")),
            payload=chat_payload_from_dict(row.get("payload")),
            created_at=_parse_datetime(row.get("created_at")),
        )

    def _apply_replays(self, beatmap_id: int, rows: list[dict]) -> None:
        state = self.replay_tabs.setdefault(beatmap_id, ReplayTabState())
        state.loading = False
        state.error = None
        state.loaded_at = time.time()
        items: list[OnlineReplayMetadata] = []
        seen_local_paths: set[str] = set()
        for row in rows:
            replay_id = str(row["replay_id"])
            local_path = self.local_state.replay_downloads.get(replay_id)
            if local_path and (not Path(local_path).is_file() or local_path in seen_local_paths):
                local_path = None
            elif local_path:
                seen_local_paths.add(local_path)
            items.append(
                OnlineReplayMetadata(
                    replay_id=replay_id,
                    beatmap_id=int(row["beatmap_id"]),
                    replay_hash=str(row["replay_hash"]),
                    player_name=str(row.get("player_name") or ""),
                    mods=int(row.get("mods") or 0),
                    score=int(row.get("score") or 0),
                    max_combo=int(row.get("max_combo") or 0),
                    n300=int(row.get("n300") or 0),
                    n100=int(row.get("n100") or 0),
                    n50=int(row.get("n50") or 0),
                    nmiss=int(row.get("nmiss") or 0),
                    map_md5=str(row.get("map_md5") or ""),
                    views=int(row.get("views") or 0),
                    original_filename=str(row.get("original_filename") or f"{replay_id}.osr"),
                    created_at=_parse_datetime(row.get("created_at")),
                    local_path=local_path,
                    is_downloaded=bool(local_path and Path(local_path).is_file()),
                    upload_state="uploaded",
                    status_text="Downloaded" if local_path and Path(local_path).is_file() else "",
                )
            )
        state.items = items

    def _handle_ws_event(self, payload: dict) -> None:
        event_type = payload.get("type")
        if event_type == "presence_changed":
            self.refresh_presence()
        elif event_type == "channel_created":
            self._upsert_channel(payload.get("channel", {}))
        elif event_type == "message_created":
            self._append_message(payload.get("message", {}))
        elif event_type == "replay_changed":
            beatmap_id = int(payload.get("beatmap_id") or 0)
            if beatmap_id:
                self.fetch_online_replays(beatmap_id, force=True)

