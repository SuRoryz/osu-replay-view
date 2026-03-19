from types import SimpleNamespace

from cursor.danser import DanserPlaystyle
from osu_map.beatmap import RenderData


def _render_data() -> RenderData:
    return RenderData(
        circle_positions=[],
        circle_start_times=[],
        circle_end_times=[],
        circle_object_indices=[],
        circle_radius=32.0,
        circle_z=[],
        slider=None,
        spinner=None,
        preempt=600.0,
        fade_in=400.0,
    )


def _beatmap(y: float):
    return SimpleNamespace(
        hit_objects=[
            SimpleNamespace(
                pos=SimpleNamespace(x=128.0, y=y),
                start_time=1000.0,
            )
        ]
    )


def test_danser_constraints_use_normal_playfield_orientation() -> None:
    playstyle = DanserPlaystyle(_beatmap(96.0), _render_data(), hr_flip=False)

    assert playstyle._debug_constraints[0]["pos"][1] == 288.0


def test_danser_constraints_follow_hr_flipped_orientation() -> None:
    playstyle = DanserPlaystyle(_beatmap(96.0), _render_data(), hr_flip=True)

    assert playstyle._debug_constraints[0]["pos"][1] == 96.0
