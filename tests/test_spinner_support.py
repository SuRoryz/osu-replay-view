import math

from osupyparser.osu.objects import Position, Spinner

from replay.data import KEY_K1, ReplayData, RFrame
from replay.judge import HitJudge
from replay.score import StablePerformanceTimeline


class _SpinnerBeatmap:
    def __init__(self, hit_objects, *, od: float = 5.0):
        self.hit_objects = hit_objects
        self.hp = 5.0
        self.cs = 4.0
        self.od = od
        self.ar = 9.0
        self.slider_multiplier = 1.4
        self.slider_tick_rate = 1.0
        self.break_times = []
        self.file_version = 14

    def _timing_at(self, _offset_ms: float):
        return 500.0, 1.0, 1, 80


def _spinner(start_ms: int = 0, end_ms: int = 2000) -> Spinner:
    return Spinner(
        pos=Position(256, 192),
        start_time=start_ms,
        new_combo=False,
        sound_enum=0,
        end_time=end_ms,
        additions=None,
    )


def _spinner_replay(*, start_ms: int, end_ms: int, spins_per_second: float, radius: float = 80.0) -> ReplayData:
    frames = [RFrame(time_ms=float(start_ms - 1), x=256.0 + radius, y=192.0, keys=0)]
    step_ms = 10
    for time_ms in range(start_ms, end_ms + 1, step_ms):
        phase = (time_ms - start_ms) / 1000.0
        angle = phase * spins_per_second * math.tau
        keys = KEY_K1 if time_ms < end_ms else 0
        frames.append(
            RFrame(
                time_ms=float(time_ms),
                x=256.0 + math.cos(angle) * radius,
                y=192.0 + math.sin(angle) * radius,
                keys=keys,
            )
        )
    return ReplayData(frames=frames)


def test_hit_judge_reconstructs_spinner_spins_and_bonus_ticks() -> None:
    spinner = _spinner()
    replay = _spinner_replay(start_ms=0, end_ms=2000, spins_per_second=5.0)

    judge = HitJudge([spinner], od=5.0, circle_radius=64.0, replay=replay)
    result = judge.result_for(0)

    assert result is not None
    assert result.result == "300"
    assert abs(result.hit_time_ms - 0.0) <= 1e-6
    assert result.spinner is not None
    assert result.spinner.required_spins == 5
    assert result.spinner.full_spins == 10
    assert [event.score for event in result.spinner.bonus_events] == [100] * 5 + [1100] * 5


def test_spinner_auto_timeline_includes_stable_bonus_score() -> None:
    spinner = _spinner()
    beatmap = _SpinnerBeatmap([spinner], od=5.0)

    timeline = StablePerformanceTimeline.build(
        beatmap=beatmap,
        beatmap_path="missing.osu",
        mods=0,
        clock_rate=1.0,
        circle_radius=36.48,
        od=5.0,
        hp=5.0,
        replay=None,
        judge=None,
    )

    final_point = timeline.points[-1]
    assert final_point.n300 == 1
    assert final_point.combo == 1
    assert final_point.score == 11800
