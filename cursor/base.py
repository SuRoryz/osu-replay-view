"""Abstract base for cursor playstyles."""

from __future__ import annotations

from abc import ABC, abstractmethod

from osu_map.beatmap import RenderData

OSU_W = 512
OSU_H = 384
MARGIN = 4.0


def clamp_to_playfield(x: float, y: float) -> tuple[float, float]:
    return (
        max(-MARGIN, min(OSU_W + MARGIN, x)),
        max(-MARGIN, min(OSU_H + MARGIN, y)),
    )


class CursorPlaystyle(ABC):
    """Interface every cursor playstyle must implement."""

    @abstractmethod
    def __init__(self, beatmap, render_data: RenderData) -> None: ...

    @abstractmethod
    def position_at(self, time_ms: float) -> tuple[float, float]:
        """Return (x, y) in osu! playfield coordinates (Y-up)."""
