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
