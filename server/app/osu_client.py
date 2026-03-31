from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import Settings
from .osu_auth import OsuAuthStore, OsuIdentity, OsuSession


class OsuApiError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail


@dataclass(slots=True)
class OsuBinaryResponse:
    payload: bytes
    content_type: str
    filename: str


class OsuApiClient:
    def __init__(self, settings: Settings, auth_store: OsuAuthStore) -> None:
        self.settings = settings
        self.auth_store = auth_store

    @property
    def enabled(self) -> bool:
        return bool(
            self.settings.osu_client_id
            and self.settings.osu_client_secret
            and self.settings.osu_redirect_uri
        )

    def ensure_enabled(self) -> None:
        if self.enabled:
            return
        raise OsuApiError(503, "Official osu OAuth is not configured on the server.")

    def build_authorize_url(self, player_uuid: str) -> tuple[str, str]:
        self.ensure_enabled()
        state = self.auth_store.issue_state(player_uuid)
        params = urlencode(
            {
                "client_id": self.settings.osu_client_id,
                "redirect_uri": self.settings.osu_redirect_uri,
                "response_type": "code",
                "scope": " ".join(self.settings.osu_oauth_scopes),
                "state": state,
            }
        )
        return f"{self.settings.osu_authorize_url}?{params}", state

    def exchange_code(self, code: str) -> OsuSession:
        self.ensure_enabled()
        payload = self._token_request(
            {
                "client_id": self.settings.osu_client_id,
                "client_secret": self.settings.osu_client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.settings.osu_redirect_uri,
            }
        )
        session = self._session_from_token_payload(payload)
        session.identity = self.fetch_me(session)
        return session

    def refresh_session(self, session: OsuSession) -> OsuSession:
        self.ensure_enabled()
        payload = self._token_request(
            {
                "client_id": self.settings.osu_client_id,
                "client_secret": self.settings.osu_client_secret,
                "grant_type": "refresh_token",
                "refresh_token": session.refresh_token,
                "redirect_uri": self.settings.osu_redirect_uri,
            }
        )
        refreshed = self._session_from_token_payload(payload, fallback_refresh_token=session.refresh_token)
        refreshed.identity = self.fetch_me(refreshed)
        return refreshed

    def current_session(self, player_uuid: str, *, refresh_if_needed: bool = True) -> OsuSession:
        session = self.auth_store.load_session(player_uuid)
        if session is None:
            raise OsuApiError(401, "No osu account is linked for this player.")
        if not refresh_if_needed or not session.is_expired():
            return session
        refreshed = self.refresh_session(session)
        self.auth_store.save_session(player_uuid, refreshed)
        return refreshed

    def fetch_me(self, session: OsuSession) -> OsuIdentity:
        payload = self._request_json("GET", "/me", session=session)
        cover_url = ""
        cover_value = payload.get("cover")
        if isinstance(cover_value, dict):
            cover_url = str(
                cover_value.get("custom_url")
                or cover_value.get("url")
                or ""
            )
        return OsuIdentity(
            user_id=int(payload.get("id") or 0),
            username=str(payload.get("username") or ""),
            avatar_url=str(payload.get("avatar_url") or ""),
            country_code=str(payload.get("country_code") or ""),
            cover_url=cover_url,
        )

    def search_beatmapsets(
        self,
        player_uuid: str,
        *,
        query: str = "",
        cursor_string: str = "",
        status: str = "",
        mode: str = "osu",
    ) -> dict[str, Any]:
        session = self.current_session(player_uuid)
        params: dict[str, Any] = {}
        if query.strip():
            params["q"] = query.strip()
        if cursor_string.strip():
            params["cursor_string"] = cursor_string.strip()
        if status.strip():
            params["s"] = status.strip()
        if mode.strip():
            params["m"] = mode.strip()
        payload = self._request_json("GET", "/beatmapsets/search", params=params, session=session)
        beatmapsets = payload.get("beatmapsets")
        if not isinstance(beatmapsets, list):
            beatmapsets = []
        cursor = payload.get("cursor_string")
        return {
            "cursor_string": str(cursor or ""),
            "beatmapsets": [self._normalize_beatmapset(item) for item in beatmapsets if isinstance(item, dict)],
        }

    def lookup_beatmap(
        self,
        player_uuid: str,
        *,
        checksum: str = "",
        beatmap_id: int | None = None,
    ) -> dict[str, Any]:
        session = self.current_session(player_uuid)
        params: dict[str, Any] = {}
        if checksum.strip():
            params["checksum"] = checksum.strip()
        if beatmap_id is not None:
            params["id"] = int(beatmap_id)
        if not params:
            raise OsuApiError(400, "Beatmap lookup requires a checksum or beatmap id.")
        payload = self._request_json("GET", "/beatmaps/lookup", params=params, session=session)
        beatmapset_value = payload.get("beatmapset")
        beatmapset_id = None
        if isinstance(beatmapset_value, dict) and beatmapset_value.get("id") is not None:
            beatmapset_id = int(beatmapset_value["id"])
        return {
            "beatmap_id": int(payload.get("id") or 0),
            "beatmapset_id": beatmapset_id,
            "checksum": str(payload.get("checksum") or ""),
            "mode": str(payload.get("mode") or "osu"),
            "title": str(payload.get("beatmapset", {}).get("title") or ""),
            "artist": str(payload.get("beatmapset", {}).get("artist") or ""),
            "creator": str(payload.get("beatmapset", {}).get("creator") or ""),
            "version": str(payload.get("version") or ""),
        }

    def list_scores(self, player_uuid: str, beatmap_id: int, *, ruleset: str = "osu") -> list[dict[str, Any]]:
        session = self.current_session(player_uuid)
        payload = self._request_json(
            "GET",
            f"/beatmaps/{int(beatmap_id)}/scores",
            params={"ruleset": ruleset},
            session=session,
        )
        scores_value = payload.get("scores")
        rows = scores_value if isinstance(scores_value, list) else payload
        if not isinstance(rows, list):
            rows = []
        return [self._normalize_score(item, default_ruleset=ruleset) for item in rows if isinstance(item, dict)]

    def download_score(
        self,
        player_uuid: str,
        score_id: int,
        *,
        ruleset: str = "osu",
        legacy_score_id: int | None = None,
    ) -> OsuBinaryResponse:
        session = self.current_session(player_uuid)
        attempts: list[tuple[str, str]] = []
        if legacy_score_id:
            attempts.append(
                (
                    f"/scores/{ruleset}/{int(legacy_score_id)}/download",
                    f"score-{int(legacy_score_id)}.osr",
                )
            )
        attempts.append((f"/scores/{int(score_id)}/download", f"score-{int(score_id)}.osr"))
        if not legacy_score_id or int(legacy_score_id) != int(score_id):
            attempts.append(
                (
                    f"/scores/{ruleset}/{int(score_id)}/download",
                    f"score-{int(score_id)}.osr",
                )
            )

        last_error: OsuApiError | None = None
        for path, fallback_filename in attempts:
            try:
                return self._request_binary(
                    "GET",
                    path,
                    session=session,
                    fallback_filename=fallback_filename,
                )
            except OsuApiError as exc:
                last_error = exc
                if exc.status_code not in {400, 404, 422}:
                    raise
        if last_error is not None:
            raise last_error
        raise OsuApiError(404, "Replay download is unavailable for this score.")

    def _normalize_beatmapset(self, payload: dict[str, Any]) -> dict[str, Any]:
        beatmaps: list[dict[str, Any]] = []
        for item in payload.get("beatmaps") or ():
            if not isinstance(item, dict):
                continue
            beatmaps.append(
                {
                    "beatmap_id": int(item.get("id") or 0),
                    "version": str(item.get("version") or ""),
                    "difficulty_rating": float(item.get("difficulty_rating") or 0.0),
                    "mode": str(item.get("mode") or "osu"),
                    "checksum": str(item.get("checksum") or ""),
                }
            )
        covers = payload.get("covers")
        cover_url = ""
        if isinstance(covers, dict):
            cover_url = str(covers.get("cover@2x") or covers.get("cover") or "")
        return {
            "beatmapset_id": int(payload.get("id") or 0),
            "artist": str(payload.get("artist") or ""),
            "title": str(payload.get("title") or ""),
            "creator": str(payload.get("creator") or ""),
            "status": str(payload.get("status") or ""),
            "favourite_count": int(payload.get("favourite_count") or 0),
            "play_count": int(payload.get("play_count") or 0),
            "source": str(payload.get("source") or ""),
            "tags": str(payload.get("tags") or ""),
            "cover_url": cover_url,
            "preview_url": str(payload.get("preview_url") or ""),
            "beatmaps": beatmaps,
        }

    def _normalize_score(self, payload: dict[str, Any], *, default_ruleset: str) -> dict[str, Any]:
        user_value = payload.get("user")
        username = ""
        avatar_url = ""
        country_code = ""
        if isinstance(user_value, dict):
            username = str(user_value.get("username") or "")
            avatar_url = str(user_value.get("avatar_url") or "")
            country_code = str(user_value.get("country_code") or "")
        score_id = int(payload.get("id") or 0)
        legacy_score_id = int(payload.get("legacy_score_id") or 0)
        statistics = payload.get("statistics")
        accuracy = float(payload.get("accuracy") or 0.0)
        ruleset_name = str(payload.get("mode") or payload.get("ruleset") or default_ruleset or "osu")
        has_replay_value = payload.get("has_replay")
        has_replay = bool(has_replay_value) if has_replay_value is not None else bool(score_id or legacy_score_id)
        return {
            "score_id": score_id,
            "legacy_score_id": legacy_score_id,
            "username": username,
            "avatar_url": avatar_url,
            "country_code": country_code,
            "mods": [str(item) for item in payload.get("mods") or ()],
            "total_score": int(payload.get("total_score") or payload.get("score") or 0),
            "max_combo": int(payload.get("max_combo") or 0),
            "accuracy": accuracy,
            "passed": bool(payload.get("passed", True)),
            "perfect": bool(payload.get("perfect", False)),
            "has_replay": has_replay,
            "ended_at": str(payload.get("ended_at") or payload.get("created_at") or ""),
            "rank": str(payload.get("rank") or ""),
            "ruleset": ruleset_name,
            "count_300": int((statistics or {}).get("great") or (statistics or {}).get("count_300") or 0),
            "count_100": int((statistics or {}).get("ok") or (statistics or {}).get("count_100") or 0),
            "count_50": int((statistics or {}).get("meh") or (statistics or {}).get("count_50") or 0),
            "count_miss": int((statistics or {}).get("miss") or (statistics or {}).get("count_miss") or 0),
        }

    def _token_request(self, form_data: dict[str, Any]) -> dict[str, Any]:
        payload = urlencode(form_data).encode("utf-8")
        request = Request(
            self.settings.osu_token_url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        )
        return self._read_json(request)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        session: OsuSession | None = None,
    ) -> dict[str, Any]:
        request = self._make_request(method, path, params=params, session=session)
        return self._read_json(request)

    def _request_binary(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        session: OsuSession | None = None,
        fallback_filename: str,
    ) -> OsuBinaryResponse:
        request = self._make_request(method, path, params=params, session=session)
        try:
            with urlopen(request, timeout=30.0) as response:
                content_type = str(response.headers.get_content_type() or "application/octet-stream")
                filename = response.headers.get_filename() or fallback_filename
                return OsuBinaryResponse(
                    payload=response.read(),
                    content_type=content_type,
                    filename=filename,
                )
        except HTTPError as exc:
            raise self._http_error(exc) from exc
        except URLError as exc:
            raise OsuApiError(502, f"Failed to reach osu servers: {exc.reason}") from exc

    def _make_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        session: OsuSession | None = None,
    ) -> Request:
        query = ""
        if params:
            encoded = {key: value for key, value in params.items() if value not in (None, "")}
            if encoded:
                query = f"?{urlencode(encoded, doseq=True)}"
        headers = {"Accept": "application/json"}
        if session is not None:
            headers["Authorization"] = f"{session.token_type or 'Bearer'} {session.access_token}"
        return Request(f"{self.settings.osu_api_base_url}{path}{query}", method=method, headers=headers)

    def _read_json(self, request: Request) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=30.0) as response:
                raw_payload = response.read()
        except HTTPError as exc:
            raise self._http_error(exc) from exc
        except URLError as exc:
            raise OsuApiError(502, f"Failed to reach osu servers: {exc.reason}") from exc
        try:
            value = json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise OsuApiError(502, "osu returned an invalid JSON response.") from exc
        if not isinstance(value, dict):
            raise OsuApiError(502, "osu returned an unexpected response shape.")
        return value

    def _session_from_token_payload(
        self,
        payload: dict[str, Any],
        *,
        fallback_refresh_token: str = "",
    ) -> OsuSession:
        access_token = str(payload.get("access_token") or "")
        refresh_token = str(payload.get("refresh_token") or fallback_refresh_token)
        expires_in = max(1, int(payload.get("expires_in") or 3600))
        token_type = str(payload.get("token_type") or "Bearer")
        if not access_token or not refresh_token:
            raise OsuApiError(502, "osu OAuth response is missing tokens.")
        return OsuSession(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type=token_type,
            expires_at=time.time() + expires_in,
            scopes=tuple(str(item) for item in payload.get("scope", "").split() if item),
        )

    def _http_error(self, exc: HTTPError) -> OsuApiError:
        body = b""
        try:
            body = exc.read()
        except Exception:
            body = b""
        detail = f"osu request failed with status {exc.code}."
        if body:
            try:
                parsed = json.loads(body.decode("utf-8"))
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                detail = str(parsed.get("error") or parsed.get("message") or detail)
            else:
                try:
                    detail = body.decode("utf-8").strip() or detail
                except Exception:
                    detail = detail
        return OsuApiError(exc.code, detail)
