from __future__ import annotations

import json
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class OsuIdentity:
    user_id: int
    username: str
    avatar_url: str = ""
    country_code: str = ""
    cover_url: str = ""


@dataclass(slots=True)
class OsuSession:
    access_token: str
    refresh_token: str
    token_type: str
    expires_at: float
    scopes: tuple[str, ...] = field(default_factory=tuple)
    identity: OsuIdentity | None = None

    def is_expired(self, *, skew_seconds: float = 60.0) -> bool:
        return time.time() >= max(0.0, self.expires_at - skew_seconds)


class OsuAuthStore:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._sessions_dir = self._root / "sessions"
        self._states_dir = self._root / "states"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._states_dir.mkdir(parents=True, exist_ok=True)

    def issue_state(self, player_uuid: str) -> str:
        state = secrets.token_urlsafe(32)
        payload = {"player_uuid": player_uuid, "created_at": time.time()}
        self._state_path(state).write_text(json.dumps(payload), encoding="utf-8")
        return state

    def consume_state(self, state: str, *, max_age_seconds: float = 900.0) -> str | None:
        path = self._state_path(state)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        finally:
            try:
                path.unlink()
            except OSError:
                pass
        created_at = float(payload.get("created_at") or 0.0)
        if not payload.get("player_uuid") or time.time() - created_at > max_age_seconds:
            return None
        return str(payload["player_uuid"])

    def save_session(self, player_uuid: str, session: OsuSession) -> None:
        payload = asdict(session)
        payload["scopes"] = list(session.scopes)
        self._session_path(player_uuid).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_session(self, player_uuid: str) -> OsuSession | None:
        path = self._session_path(player_uuid)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        identity_value = payload.get("identity")
        identity = None
        if isinstance(identity_value, dict):
            try:
                identity = OsuIdentity(
                    user_id=int(identity_value.get("user_id") or 0),
                    username=str(identity_value.get("username") or ""),
                    avatar_url=str(identity_value.get("avatar_url") or ""),
                    country_code=str(identity_value.get("country_code") or ""),
                    cover_url=str(identity_value.get("cover_url") or ""),
                )
            except (TypeError, ValueError):
                identity = None
        access_token = str(payload.get("access_token") or "")
        refresh_token = str(payload.get("refresh_token") or "")
        token_type = str(payload.get("token_type") or "Bearer")
        if not access_token or not refresh_token:
            return None
        return OsuSession(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type=token_type,
            expires_at=float(payload.get("expires_at") or 0.0),
            scopes=tuple(str(item) for item in payload.get("scopes") or ()),
            identity=identity,
        )

    def clear_session(self, player_uuid: str) -> None:
        try:
            self._session_path(player_uuid).unlink()
        except OSError:
            pass

    def _session_path(self, player_uuid: str) -> Path:
        safe_name = "".join(ch for ch in player_uuid if ch.isalnum() or ch in {"-", "_"})
        return self._sessions_dir / f"{safe_name}.json"

    def _state_path(self, state: str) -> Path:
        return self._states_dir / f"{state}.json"
