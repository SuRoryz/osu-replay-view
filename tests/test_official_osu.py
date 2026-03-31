import io
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

from server.app.config import Settings
from server.app.map_mirrors import MirrorResolver, MirrorResolverError
from server.app.osu_auth import OsuAuthStore, OsuIdentity, OsuSession
from server.app.osu_client import OsuApiClient, OsuApiError, OsuBinaryResponse


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url="sqlite://",
        storage_root=tmp_path / "replays",
        cors_origins=("*",),
        osu_client_id="client-id",
        osu_client_secret="client-secret",
        osu_redirect_uri="http://127.0.0.1:8000/osu/auth/callback",
        osu_api_base_url="https://osu.ppy.sh/api/v2",
        osu_authorize_url="https://osu.ppy.sh/oauth/authorize",
        osu_token_url="https://osu.ppy.sh/oauth/token",
        osu_auth_root=tmp_path / "auth",
        osu_oauth_scopes=("public", "identify"),
        map_mirror_download_templates=(
            "https://primary.invalid/download/{beatmapset_id}",
            "https://fallback.invalid/download/{beatmapset_id}",
        ),
        map_mirror_timeout_seconds=15.0,
    )


def test_osu_auth_store_round_trip(tmp_path: Path) -> None:
    store = OsuAuthStore(tmp_path / "auth")
    state = store.issue_state("player-123")

    assert store.consume_state(state) == "player-123"
    assert store.consume_state(state) is None

    session = OsuSession(
        access_token="access",
        refresh_token="refresh",
        token_type="Bearer",
        expires_at=12345.0,
        scopes=("public", "identify"),
        identity=OsuIdentity(user_id=5, username="ri4ka", avatar_url="avatar", country_code="UA", cover_url="cover"),
    )
    store.save_session("player-123", session)

    loaded = store.load_session("player-123")

    assert loaded is not None
    assert loaded.access_token == "access"
    assert loaded.refresh_token == "refresh"
    assert loaded.identity is not None
    assert loaded.identity.username == "ri4ka"

    store.clear_session("player-123")
    assert store.load_session("player-123") is None


def test_osu_api_client_normalizes_score_payload(tmp_path: Path) -> None:
    client = OsuApiClient(_settings(tmp_path), OsuAuthStore(tmp_path / "auth"))

    normalized = client._normalize_score(
        {
            "id": 42,
            "legacy_score_id": 1337,
            "accuracy": 0.9875,
            "max_combo": 512,
            "mods": ["HD", "HR"],
            "passed": True,
            "perfect": False,
            "rank": "A",
            "ended_at": "2026-03-28T12:00:00+00:00",
            "user": {
                "username": "ri4ka",
                "avatar_url": "https://a.ppy.sh/1",
                "country_code": "UA",
            },
            "statistics": {
                "great": 1234,
                "ok": 56,
                "meh": 7,
                "miss": 1,
            },
        },
        default_ruleset="osu",
    )

    assert normalized["score_id"] == 42
    assert normalized["legacy_score_id"] == 1337
    assert normalized["username"] == "ri4ka"
    assert normalized["mods"] == ["HD", "HR"]
    assert normalized["count_300"] == 1234
    assert normalized["count_100"] == 56
    assert normalized["count_50"] == 7
    assert normalized["count_miss"] == 1
    assert normalized["ruleset"] == "osu"
    assert normalized["has_replay"] is True


def test_osu_api_client_download_score_falls_back_between_endpoints(tmp_path: Path, monkeypatch) -> None:
    client = OsuApiClient(_settings(tmp_path), OsuAuthStore(tmp_path / "auth"))
    dummy_session = OsuSession(
        access_token="access",
        refresh_token="refresh",
        token_type="Bearer",
        expires_at=12345.0,
        scopes=("public", "identify"),
    )
    attempted_paths: list[str] = []

    monkeypatch.setattr(client, "current_session", lambda player_uuid: dummy_session)

    def fake_request_binary(method, path, *, params=None, session=None, fallback_filename):
        attempted_paths.append(path)
        if path == "/scores/osu/1337/download":
            raise OsuApiError(404, "legacy endpoint missed")
        return OsuBinaryResponse(payload=b"osr", content_type="application/octet-stream", filename="score.osr")

    monkeypatch.setattr(client, "_request_binary", fake_request_binary)

    response = client.download_score("player-123", 42, ruleset="osu", legacy_score_id=1337)

    assert attempted_paths == [
        "/scores/osu/1337/download",
        "/scores/42/download",
    ]
    assert response.filename == "score.osr"
    assert response.payload == b"osr"


class _FakeResponse:
    def __init__(self, payload: bytes, *, content_type: str = "application/octet-stream", filename: str | None = None) -> None:
        headers = Message()
        headers.add_header("Content-Type", content_type)
        if filename is not None:
            headers.add_header("Content-Disposition", "attachment", filename=filename)
        self.headers = headers
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_mirror_resolver_falls_back_to_next_provider(tmp_path: Path, monkeypatch) -> None:
    resolver = MirrorResolver(_settings(tmp_path))
    requests: list[str] = []

    def fake_urlopen(request, timeout=0):
        requests.append(request.full_url)
        if "primary.invalid" in request.full_url:
            raise HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=io.BytesIO(b""))
        return _FakeResponse(b"PK\x03\x04mirror-archive", filename="fallback.osz")

    monkeypatch.setattr("server.app.map_mirrors.urlopen", fake_urlopen)

    response = resolver.download_beatmapset(1426912)

    assert requests == [
        "https://primary.invalid/download/1426912",
        "https://fallback.invalid/download/1426912",
    ]
    assert response.filename == "fallback.osz"
    assert response.payload.startswith(b"PK")


def test_mirror_resolver_rejects_non_archive_payload(tmp_path: Path, monkeypatch) -> None:
    resolver = MirrorResolver(_settings(tmp_path))

    def fake_urlopen(request, timeout=0):
        return _FakeResponse(b"<html>blocked</html>", content_type="text/html")

    monkeypatch.setattr("server.app.map_mirrors.urlopen", fake_urlopen)

    try:
        resolver.download_beatmapset(42)
    except MirrorResolverError as exc:
        assert exc.status_code == 502
        assert "returned HTML instead of an .osz archive" in exc.detail
    else:
        raise AssertionError("Expected mirror resolver to reject HTML payloads.")
