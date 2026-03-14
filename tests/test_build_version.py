from pathlib import Path

import build_version


def test_get_display_version_for_dev_uses_next_build(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "version_state.json").write_text(
        '{\n  "major": 0,\n  "minor": 2,\n  "build": 4\n}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OSU_REPLAY_BRANCH_NAME", "feature/test")

    version = build_version.get_display_version(
        frozen=False,
        repo_root=tmp_path,
        app_root=tmp_path,
    )

    assert version == "0.2.5-DEV-feature_test"


def test_prepare_and_finalize_build_persist_next_prod_version(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "version_state.json").write_text(
        '{\n  "major": 1,\n  "minor": 3,\n  "build": 8\n}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OSU_REPLAY_BRANCH_NAME", "release/main")

    prepared = build_version.prepare_build(repo_root=tmp_path)
    finalized = build_version.finalize_build(repo_root=tmp_path)
    persisted = build_version.load_version_state(tmp_path / "version_state.json")

    assert prepared.version == "1.3.9-PROD-release_main"
    assert finalized.version == prepared.version
    assert persisted.build == 9
