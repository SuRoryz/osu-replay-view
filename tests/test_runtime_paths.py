from pathlib import Path

import runtime_paths


def test_load_env_file_sets_process_environment(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment\n"
        "OSU_REPLAY_SERVER_URL=http://example.test:8000\n"
        "EXTRA_VALUE='quoted value'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OSU_REPLAY_SERVER_URL", raising=False)
    monkeypatch.delenv("EXTRA_VALUE", raising=False)

    loaded = runtime_paths.load_env_file(env_path, override=True)

    assert loaded["OSU_REPLAY_SERVER_URL"] == "http://example.test:8000"
    assert loaded["EXTRA_VALUE"] == "quoted value"


def test_replay_dir_for_set_uses_final_path_segment() -> None:
    target = runtime_paths.replay_dir_for_set(r"C:\Songs\Artist - Title")
    assert target == runtime_paths.REPLAYS_DIR / "Artist - Title"


def test_bundled_binary_path_prefers_app_root(monkeypatch, tmp_path: Path) -> None:
    ffmpeg_dir = tmp_path / "ffmpeg"
    ffmpeg_dir.mkdir()

    suffix = ".exe" if runtime_paths.sys.platform == "win32" else ""
    app_binary = tmp_path / f"ffmpeg{suffix}"
    nested_binary = ffmpeg_dir / f"ffmpeg{suffix}"
    app_binary.write_bytes(b"")
    nested_binary.write_bytes(b"")

    monkeypatch.setattr(runtime_paths, "APP_ROOT", tmp_path)
    monkeypatch.setattr(runtime_paths, "FFMPEG_DIR", ffmpeg_dir)

    assert runtime_paths.bundled_binary_path("ffmpeg") == app_binary


def test_ensure_runtime_dirs_creates_expected_tree(monkeypatch, tmp_path: Path) -> None:
    maps_dir = tmp_path / "maps"
    replays_dir = tmp_path / "replays"
    skins_dir = tmp_path / "skins"
    hitsounds_dir = skins_dir / "hitsounds"

    monkeypatch.setattr(runtime_paths, "MAPS_DIR", maps_dir)
    monkeypatch.setattr(runtime_paths, "REPLAYS_DIR", replays_dir)
    monkeypatch.setattr(runtime_paths, "SKINS_DIR", skins_dir)
    monkeypatch.setattr(runtime_paths, "HITSOUNDS_DIR", hitsounds_dir)

    runtime_paths.ensure_runtime_dirs()

    assert maps_dir.is_dir()
    assert replays_dir.is_dir()
    assert skins_dir.is_dir()
    assert hitsounds_dir.is_dir()
