"""Interaction-state helpers for desktop UI components."""

from __future__ import annotations

from enum import Enum


class InteractionState(str, Enum):
    REST = "rest"
    HOVER = "hover"
    FOCUS = "focus"
    PRESSED = "pressed"
    SELECTED = "selected"
    DISABLED = "disabled"


def rgba(color: tuple[float, float, float], alpha: float = 1.0) -> tuple[float, float, float, float]:
    return (color[0], color[1], color[2], alpha)
