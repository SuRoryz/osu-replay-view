import threading
from types import SimpleNamespace

from social.client import SocialClient


def test_dedupe_presence_rows_keeps_highest_priority_row() -> None:
    client = SocialClient.__new__(SocialClient)
    client.player_uuid = "current-uuid"

    rows = [
        {
            "player_uuid": "older-uuid",
            "nickname": "SuRory",
            "online": False,
            "last_seen": "2026-03-14T10:00:00+00:00",
        },
        {
            "player_uuid": "current-uuid",
            "nickname": "SuRory",
            "online": True,
            "last_seen": "2026-03-14T11:00:00+00:00",
        },
        {
            "player_uuid": "other-user",
            "nickname": "Other",
            "online": False,
            "last_seen": "2026-03-14T09:00:00+00:00",
        },
    ]

    deduped = client._dedupe_presence_rows(rows)

    assert len(deduped) == 2
    assert deduped[0]["player_uuid"] == "current-uuid"
    assert deduped[1]["player_uuid"] == "other-user"


def test_apply_presence_keeps_status_text() -> None:
    client = SocialClient.__new__(SocialClient)
    client.player_uuid = "current-uuid"
    client._lock = threading.RLock()
    client.local_state = SimpleNamespace(friends=[], blocked=[])
    client.users = {}

    client._apply_presence(
        [
            {
                "player_uuid": "other-user",
                "nickname": "Other",
                "online": True,
                "status_text": "Listening Camellia - Expert",
                "last_seen": "2026-03-14T09:00:00+00:00",
            }
        ]
    )

    assert client.users["other-user"].status_text == "Listening Camellia - Expert"


def test_existing_download_path_discards_stale_mapping(tmp_path) -> None:
    replay_id = "deadbeef-1234"
    stale_path = tmp_path / "stale.osr"
    stale_path.write_bytes(b"stale")
    valid_path = tmp_path / "valid [deadbeef].osr"
    valid_path.write_bytes(b"valid")

    forgotten: list[str] = []

    def forget_download(target_replay_id: str) -> None:
        forgotten.append(target_replay_id)
        local_state.replay_downloads.pop(target_replay_id, None)

    local_state = SimpleNamespace(
        replay_downloads={replay_id: str(stale_path)},
        forget_download=forget_download,
    )

    client = SocialClient.__new__(SocialClient)
    client.local_state = local_state
    client._download_path_matches_replay = (
        lambda local_path, *, beatmap_id, replay_hash: local_path == str(valid_path)
    )

    resolved = client._existing_download_path_for_replay(
        replay_id,
        str(tmp_path),
        beatmap_id=123,
        replay_hash="expected-hash",
    )

    assert resolved == str(valid_path)
    assert forgotten == [replay_id]
    assert replay_id not in local_state.replay_downloads
