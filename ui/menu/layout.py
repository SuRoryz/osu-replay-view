"""Adaptive layout primitives for menu/UI rendering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import floor

from ui.design.theme import build_desktop_theme
from ui.design.tokens import DesktopTheme


class LayoutMode(str, Enum):
    NARROW = "narrow"
    STANDARD = "standard"
    WIDE = "wide"
    ULTRAWIDE = "ultrawide"


@dataclass(frozen=True, slots=True)
class Insets:
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def all(cls, value: float) -> "Insets":
        return cls(value, value, value, value)

    @classmethod
    def symmetric(cls, horizontal: float, vertical: float) -> "Insets":
        return cls(horizontal, vertical, horizontal, vertical)


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def center_x(self) -> float:
        return self.x + self.w * 0.5

    @property
    def center_y(self) -> float:
        return self.y + self.h * 0.5

    def inset(self, value: float | Insets) -> "Rect":
        if isinstance(value, Insets):
            inset = value
        else:
            inset = Insets.all(value)
        return Rect(
            self.x + inset.left,
            self.y + inset.top,
            max(0.0, self.w - inset.left - inset.right),
            max(0.0, self.h - inset.top - inset.bottom),
        )

    def translate(self, dx: float = 0.0, dy: float = 0.0) -> "Rect":
        return Rect(self.x + dx, self.y + dy, self.w, self.h)

    def clamp_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            min(max(x, self.x), self.right),
            min(max(y, self.y), self.bottom),
        )

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.right and self.y <= y <= self.bottom

    def snap(self) -> "Rect":
        return Rect(floor(self.x), floor(self.y), floor(self.w), floor(self.h))

    def tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)


@dataclass(frozen=True, slots=True)
class TypographyScale:
    title_l: int
    title_m: int
    title_s: int
    body_l: int
    body_m: int
    body_s: int
    caption: int


@dataclass(frozen=True, slots=True)
class MotionScale:
    fast: float
    normal: float
    slow: float


@dataclass(frozen=True, slots=True)
class LayoutTokens:
    radius_l: float
    radius_m: float
    radius_s: float
    gap_xl: float
    gap_l: float
    gap_m: float
    gap_s: float
    gutter: float
    panel_pad: float
    content_max_width: float
    drawer_width: float
    drawer_min_width: float
    bottom_bar_height: float
    typography: TypographyScale
    motion: MotionScale


@dataclass(frozen=True, slots=True)
class LayoutContext:
    viewport: Rect
    safe_area: Rect
    content_rect: Rect
    mode: LayoutMode
    density: float
    tokens: LayoutTokens
    theme: DesktopTheme


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mode_for_viewport(width: float, height: float) -> LayoutMode:
    aspect = width / max(1.0, height)
    if width < 1180 or height < 760:
        return LayoutMode.NARROW
    if aspect >= 2.1:
        return LayoutMode.ULTRAWIDE
    if aspect >= 1.72:
        return LayoutMode.WIDE
    return LayoutMode.STANDARD


def _content_rect(viewport: Rect, mode: LayoutMode, density: float) -> Rect:
    gutter = 24.0 * density
    max_width = {
        LayoutMode.NARROW: viewport.w - gutter * 2.0,
        LayoutMode.STANDARD: min(viewport.w - gutter * 2.0, 1540.0 * density),
        LayoutMode.WIDE: min(viewport.w - gutter * 2.0, 1760.0 * density),
        LayoutMode.ULTRAWIDE: min(viewport.w - gutter * 2.0, 1840.0 * density),
    }[mode]
    content_w = max(320.0, max_width)
    return Rect(
        viewport.x + max(0.0, (viewport.w - content_w) * 0.5),
        viewport.y,
        content_w,
        viewport.h,
    )


def build_layout_context(width: int, height: int) -> LayoutContext:
    viewport = Rect(0.0, 0.0, float(max(1, width)), float(max(1, height)))
    density = clamp(min(viewport.w / 1920.0, viewport.h / 1080.0), 0.72, 1.35)
    mode = _mode_for_viewport(viewport.w, viewport.h)
    safe_gutter = 16.0 * density
    safe_area = viewport.inset(Insets.all(safe_gutter))
    content_rect = _content_rect(safe_area, mode, density)
    typography = TypographyScale(
        title_l=max(24, int(round(32.0 * density))),
        title_m=max(20, int(round(26.0 * density))),
        title_s=max(16, int(round(20.0 * density))),
        body_l=max(15, int(round(18.0 * density))),
        body_m=max(13, int(round(15.0 * density))),
        body_s=max(12, int(round(13.0 * density))),
        caption=max(11, int(round(12.0 * density))),
    )
    tokens = LayoutTokens(
        radius_l=18.0 * density,
        radius_m=12.0 * density,
        radius_s=8.0 * density,
        gap_xl=28.0 * density,
        gap_l=20.0 * density,
        gap_m=14.0 * density,
        gap_s=8.0 * density,
        gutter=24.0 * density,
        panel_pad=20.0 * density,
        content_max_width=content_rect.w,
        drawer_width=min(content_rect.w * 0.44, 720.0 * density),
        drawer_min_width=360.0 * density,
        bottom_bar_height=60.0 * density,
        typography=typography,
        motion=MotionScale(
            fast=14.0,
            normal=10.0,
            slow=6.0,
        ),
    )
    return LayoutContext(
        viewport=viewport,
        safe_area=safe_area,
        content_rect=content_rect,
        mode=mode,
        density=density,
        tokens=tokens,
        theme=build_desktop_theme(
            density,
            typography_scale=typography,
            motion_scale=tokens.motion,
        ),
    )


def split_columns(rect: Rect, *, ratio: float, gap: float) -> tuple[Rect, Rect]:
    left_w = max(0.0, rect.w * ratio - gap * 0.5)
    right_x = rect.x + left_w + gap
    right_w = max(0.0, rect.right - right_x)
    return (
        Rect(rect.x, rect.y, left_w, rect.h),
        Rect(right_x, rect.y, right_w, rect.h),
    )


def stack_vertical(rect: Rect, heights: list[float], gap: float) -> list[Rect]:
    y = rect.y
    out: list[Rect] = []
    for idx, item_h in enumerate(heights):
        out.append(Rect(rect.x, y, rect.w, max(0.0, item_h)))
        y += item_h
        if idx != len(heights) - 1:
            y += gap
    return out


def flow_rows(
    rect: Rect,
    item_widths: list[float],
    *,
    row_height: float,
    gap_x: float,
    gap_y: float,
) -> list[Rect]:
    x = rect.x
    y = rect.y
    row_bottom = y + row_height
    result: list[Rect] = []
    for width in item_widths:
        if result and x + width > rect.right:
            x = rect.x
            y = row_bottom + gap_y
            row_bottom = y + row_height
        result.append(Rect(x, y, min(width, rect.w), row_height))
        x += width + gap_x
    return result

