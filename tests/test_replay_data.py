import pytest

from osu_map.mods import HR
from replay.data import KEY_K1, KEY_K2, ReplayData, RFrame


def build_replay() -> ReplayData:
    return ReplayData(
        frames=[
            RFrame(time_ms=0.0, x=0.0, y=0.0, keys=0),
            RFrame(time_ms=10.0, x=10.0, y=20.0, keys=KEY_K1),
            RFrame(time_ms=20.0, x=20.0, y=40.0, keys=KEY_K1),
            RFrame(time_ms=30.0, x=30.0, y=60.0, keys=0),
            RFrame(time_ms=40.0, x=40.0, y=80.0, keys=KEY_K2),
        ],
        mods=0,
    )


def test_position_at_interpolates_between_frames() -> None:
    replay = build_replay()

    x, y = replay.position_at(15.0)

    assert x == pytest.approx(15.0)
    assert y == pytest.approx(30.0)


def test_keys_at_returns_active_key_state() -> None:
    replay = build_replay()

    assert replay.keys_at(5.0) == 0
    assert replay.keys_at(10.0) == KEY_K1
    assert replay.keys_at(35.0) == 0
    assert replay.keys_at(45.0) == KEY_K2


def test_new_presses_between_returns_rising_edges_only() -> None:
    replay = build_replay()

    presses = replay.new_presses_between(0.0, 50.0)

    assert [(press.time_ms, press.keys) for press in presses] == [
        (10.0, KEY_K1),
        (40.0, KEY_K2),
    ]


def test_with_target_mods_flips_vertical_axis_for_hr() -> None:
    replay = build_replay()

    flipped = replay.with_target_mods(HR)

    assert flipped is not replay
    assert flipped.frames[0].y == pytest.approx(384.0)
    assert flipped.frames[1].y == pytest.approx(364.0)
