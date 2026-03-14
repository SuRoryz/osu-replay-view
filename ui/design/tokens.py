"""Design-system token dataclasses for the desktop UI."""

from __future__ import annotations

from dataclasses import dataclass

Color3 = tuple[float, float, float]
Color4 = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ColorRoles:
    primary: Color3
    primary_soft: Color3
    on_primary: Color3
    primary_container: Color3
    primary_container_hover: Color3
    on_primary_container: Color3
    secondary_container: Color3
    secondary_container_hover: Color3
    tertiary_container: Color3
    surface: Color3
    surface_dim: Color3
    surface_container: Color3
    surface_container_high: Color3
    surface_container_low: Color3
    surface_variant: Color3
    surface_variant_soft: Color3
    surface_selected: Color3
    surface_drawer: Color3
    surface_bottom_bar: Color3
    surface_input: Color3
    outline: Color3
    outline_variant: Color3
    outline_strong: Color3
    text_primary: Color3
    text_secondary: Color3
    text_muted: Color3
    text_disabled: Color3
    scrim: Color3
    focus_ring: Color3
    slider_fill: Color3
    slider_thumb: Color3
    success: Color3
    warning: Color3
    error: Color3
    info: Color3


@dataclass(frozen=True, slots=True)
class TypographyRoles:
    display: int
    headline: int
    title_l: int
    title_m: int
    body_l: int
    body_m: int
    label_l: int
    label_m: int
    caption: int


@dataclass(frozen=True, slots=True)
class SpacingScale:
    xxs: float
    xs: float
    sm: float
    md: float
    lg: float
    xl: float
    xxl: float


@dataclass(frozen=True, slots=True)
class ShapeScale:
    corner_s: float
    corner_m: float
    corner_l: float
    pill: float


@dataclass(frozen=True, slots=True)
class ElevationScale:
    border_subtle: float
    border_normal: float
    border_strong: float
    overlay_soft: float
    overlay_strong: float


@dataclass(frozen=True, slots=True)
class MotionRoles:
    fast: float
    normal: float
    emphasis: float


@dataclass(frozen=True, slots=True)
class BackgroundGradient:
    top: Color3
    bottom: Color3


@dataclass(frozen=True, slots=True)
class DesktopTheme:
    colors: ColorRoles
    typography: TypographyRoles
    spacing: SpacingScale
    shape: ShapeScale
    elevation: ElevationScale
    motion: MotionRoles
    background: BackgroundGradient
