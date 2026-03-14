"""Material-inspired theme builder adapted for the desktop renderer."""

from __future__ import annotations

from ui.design.tokens import (
    BackgroundGradient,
    ColorRoles,
    DesktopTheme,
    ElevationScale,
    MotionRoles,
    ShapeScale,
    SpacingScale,
    TypographyRoles,
)

DEFAULT_BACKGROUND_GRADIENT = BackgroundGradient(
    top=(0.08, 0.05, 0.15),
    bottom=(0.02, 0.02, 0.06),
)


def _typography_from_scale(scale) -> TypographyRoles:
    return TypographyRoles(
        display=max(scale.title_l + 8, scale.title_l),
        headline=scale.title_l,
        title_l=scale.title_m,
        title_m=scale.title_s,
        body_l=scale.body_l,
        body_m=scale.body_m,
        label_l=scale.body_s,
        label_m=scale.caption,
        caption=scale.caption,
    )


def _motion_from_scale(scale) -> MotionRoles:
    return MotionRoles(
        fast=float(scale.fast),
        normal=float(scale.normal),
        emphasis=float(scale.slow),
    )


def build_desktop_theme(density: float, *, typography_scale=None, motion_scale=None) -> DesktopTheme:
    spacing = SpacingScale(
        xxs=4.0 * density,
        xs=8.0 * density,
        sm=12.0 * density,
        md=16.0 * density,
        lg=20.0 * density,
        xl=24.0 * density,
        xxl=32.0 * density,
    )
    shape = ShapeScale(
        corner_s=8.0 * density,
        corner_m=12.0 * density,
        corner_l=18.0 * density,
        pill=999.0,
    )
    elevation = ElevationScale(
        border_subtle=0.8,
        border_normal=1.0,
        border_strong=1.5,
        overlay_soft=0.18,
        overlay_strong=0.55,
    )
    if typography_scale is None:
        typography = TypographyRoles(
            display=max(28, int(round(40.0 * density))),
            headline=max(24, int(round(32.0 * density))),
            title_l=max(20, int(round(26.0 * density))),
            title_m=max(16, int(round(20.0 * density))),
            body_l=max(15, int(round(18.0 * density))),
            body_m=max(13, int(round(15.0 * density))),
            label_l=max(12, int(round(13.0 * density))),
            label_m=max(11, int(round(12.0 * density))),
            caption=max(11, int(round(12.0 * density))),
        )
    else:
        typography = _typography_from_scale(typography_scale)
    if motion_scale is None:
        motion = MotionRoles(fast=14.0, normal=10.0, emphasis=6.0)
    else:
        motion = _motion_from_scale(motion_scale)

    colors = ColorRoles(
        primary=(0.90, 0.42, 0.95),
        primary_soft=(0.64, 0.32, 0.80),
        on_primary=(0.10, 0.07, 0.16),
        primary_container=(0.30, 0.14, 0.46),
        primary_container_hover=(0.42, 0.18, 0.58),
        on_primary_container=(0.98, 0.98, 1.0),
        secondary_container=(0.12, 0.13, 0.20),
        secondary_container_hover=(0.22, 0.18, 0.36),
        tertiary_container=(0.12, 0.08, 0.22),
        surface=(0.030, 0.032, 0.070),
        surface_dim=(0.020, 0.020, 0.060),
        surface_container=(0.055, 0.050, 0.105),
        surface_container_high=(0.090, 0.075, 0.145),
        surface_container_low=(0.045, 0.045, 0.085),
        surface_variant=(0.080, 0.080, 0.130),
        surface_variant_soft=(0.035, 0.035, 0.075),
        surface_selected=(0.29, 0.16, 0.42),
        surface_drawer=(0.045, 0.050, 0.090),
        surface_bottom_bar=(0.045, 0.045, 0.085),
        surface_input=(0.12, 0.14, 0.22),
        outline=(0.28, 0.30, 0.40),
        outline_variant=(0.22, 0.20, 0.34),
        outline_strong=(0.92, 0.50, 0.96),
        text_primary=(0.98, 0.98, 1.0),
        text_secondary=(0.79, 0.80, 0.88),
        text_muted=(0.60, 0.62, 0.72),
        text_disabled=(0.32, 0.32, 0.40),
        scrim=(0.0, 0.0, 0.0),
        focus_ring=(0.72, 0.86, 1.0),
        slider_fill=(0.94, 0.30, 0.84),
        slider_thumb=(0.98, 0.98, 1.0),
        success=(0.34, 0.88, 0.56),
        warning=(1.0, 0.84, 0.28),
        error=(1.0, 0.46, 0.46),
        info=(0.45, 0.92, 1.0),
    )
    return DesktopTheme(
        colors=colors,
        typography=typography,
        spacing=spacing,
        shape=shape,
        elevation=elevation,
        motion=motion,
        background=DEFAULT_BACKGROUND_GRADIENT,
    )
