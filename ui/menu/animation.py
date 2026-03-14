"""Small animation helpers for menu/UI motion."""

from __future__ import annotations

from dataclasses import dataclass


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def ease_out_cubic(value: float) -> float:
    t = clamp01(value)
    return 1.0 - (1.0 - t) ** 3


def ease_in_quad(value: float) -> float:
    t = clamp01(value)
    return t * t


def ease_in_out_cubic(value: float) -> float:
    t = clamp01(value)
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 3) * 0.5


def ease_out_back(value: float) -> float:
    t = clamp01(value)
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * ((t - 1.0) ** 3) + c1 * ((t - 1.0) ** 2)


def stagger(value: float, index: int, step: float = 0.12) -> float:
    t = clamp01((clamp01(value) - index * step) / max(0.001, 1.0 - index * step))
    return ease_out_cubic(t)


def approach(current: float, target: float, speed: float, dt: float) -> float:
    if dt <= 0.0:
        return current
    step = min(1.0, dt * speed)
    return current + (target - current) * step


@dataclass(slots=True)
class AnimatedFloat:
    value: float = 0.0
    target: float = 0.0
    speed: float = 10.0

    def set_target(self, target: float) -> None:
        self.target = float(target)

    def snap(self, value: float) -> None:
        self.value = float(value)
        self.target = float(value)

    def update(self, dt: float) -> float:
        self.value = approach(self.value, self.target, self.speed, dt)
        return self.value

