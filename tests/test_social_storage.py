import json

from social.storage import SocialLocalState


def test_social_local_state_load_reuses_uuid_backup(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))
    state_dir = tmp_path / "osu_replay_v2"
    state_dir.mkdir()
    (state_dir / "player_uuid.txt").write_text("stable-uuid", encoding="utf-8")
    (state_dir / "social_state.json").write_text("{broken", encoding="utf-8")

    state = SocialLocalState.load()

    assert state.player_uuid == "stable-uuid"
    payload = json.loads((state_dir / "social_state.json").read_text(encoding="utf-8"))
    assert payload["player_uuid"] == "stable-uuid"


def test_social_local_state_save_writes_uuid_backup(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))
    state = SocialLocalState(player_uuid="persisted-uuid")

    state.save()

    assert SocialLocalState.uuid_path().read_text(encoding="utf-8") == "persisted-uuid"
