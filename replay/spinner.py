"""Shared osu!stable-style spinner math helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

SPINNER_CENTER_X = 256.0
SPINNER_CENTER_Y = 192.0
MAX_SPINNER_RPM = 477.0
MAX_SPINNER_RPS = MAX_SPINNER_RPM / 60.0


@dataclass(slots=True)
class SpinnerSpinEvent:
    """One scored full-spin boundary crossed during a spinner."""

    time_ms: float
    score: int


@dataclass(slots=True)
class SpinnerAnalysis:
    """Derived spinner state reconstructed from replay or auto-play."""

    total_rotation_deg: float = 0.0
    total_spins: float = 0.0
    full_spins: int = 0
    required_spins: int = 0
    bonus_events: list[SpinnerSpinEvent] = field(default_factory=list)
    first_active_time_ms: float | None = None


def spinner_clear_spins_per_second(od: float) -> float:
    """Stable spinner clear requirement in full spins per second."""
    od = max(0.0, min(10.0, float(od)))
    if od < 5.0:
        return 1.5 + 0.2 * od
    return 1.25 + 0.25 * od


def spinner_required_spins(duration_ms: float, od: float) -> int:
    """Stable spinner requirement rounded like the wiki formula."""
    duration_s = max(0.0, float(duration_ms)) / 1000.0
    return max(0, int(duration_s * spinner_clear_spins_per_second(od) + 0.5))


def spinner_result(total_spins: float, required_spins: int) -> str:
    """Stable spinner judgement from total spins performed."""
    total_spins = max(0.0, float(total_spins))
    required_spins = max(0, int(required_spins))
    if required_spins <= 0:
        return "300"
    if total_spins >= required_spins:
        return "300"
    if total_spins >= max(0, required_spins - 1):
        return "100"
    if total_spins >= required_spins * 0.25:
        return "50"
    return "miss"


def spinner_score_for_spin(spin_index: int, required_spins: int) -> int:
    """Stable ScoreV1 bonus for the given completed full spin."""
    if spin_index <= max(0, int(required_spins)):
        return 100
    return 1100


def build_auto_spinner_analysis(start_time_ms: float, end_time_ms: float, od: float) -> SpinnerAnalysis:
    """Perfect auto-play spinner analysis using the stable RPM cap."""
    duration_ms = max(0.0, float(end_time_ms) - float(start_time_ms))
    duration_s = duration_ms / 1000.0
    total_spins = duration_s * MAX_SPINNER_RPS
    full_spins = int(total_spins + 1e-6)
    required_spins = spinner_required_spins(duration_ms, od)
    bonus_events = [
        SpinnerSpinEvent(
            time_ms=float(start_time_ms) + spin_idx / MAX_SPINNER_RPS * 1000.0,
            score=spinner_score_for_spin(spin_idx, required_spins),
        )
        for spin_idx in range(1, full_spins + 1)
    ]
    return SpinnerAnalysis(
        total_rotation_deg=total_spins * 360.0,
        total_spins=total_spins,
        full_spins=full_spins,
        required_spins=required_spins,
        bonus_events=bonus_events,
        first_active_time_ms=float(start_time_ms),
    )
