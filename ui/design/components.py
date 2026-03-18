"""Reusable desktop component drawing helpers."""

from __future__ import annotations

from ui.design.state import InteractionState, rgba


def draw_surface(
    commands,
    theme,
    rect,
    *,
    role: str,
    state: InteractionState = InteractionState.REST,
    radius: float,
    alpha: float = 1.0,
    border_width: float = 1.0,
) -> None:
    colors = theme.colors
    fill = colors.surface_container
    border = colors.outline_variant
    if role == "panel":
        fill = colors.surface
        border = colors.outline_variant
    elif role == "card":
        fill = colors.surface_selected if state == InteractionState.SELECTED else (
            colors.surface_container_high if state == InteractionState.HOVER else colors.surface_container
        )
        border = colors.outline_strong if state == InteractionState.SELECTED else colors.outline_variant
    elif role == "badge":
        fill = colors.surface_variant
        border = colors.primary_soft
    elif role == "section":
        fill = colors.surface_variant_soft
        border = colors.outline_variant
    elif role == "drawer":
        fill = colors.surface_drawer
        border = colors.outline
    elif role == "bottom_bar":
        fill = colors.surface_bottom_bar
        border = colors.outline_variant
    elif role == "toolbar":
        fill = colors.surface_container_low
        border = colors.outline_variant
    elif role == "hud":
        fill = colors.surface_container_low
        border = colors.outline_variant
    elif role == "overlay_button":
        fill = colors.secondary_container_hover if state == InteractionState.HOVER else colors.secondary_container
        border = colors.focus_ring if state == InteractionState.HOVER else colors.outline
    elif role == "input":
        fill = colors.surface_input
        border = colors.outline_strong if state == InteractionState.FOCUS else colors.outline
    elif role == "slider_panel":
        fill = colors.surface_dim
        border = colors.outline
    elif role == "slider_track":
        fill = colors.secondary_container
        border = colors.outline
    elif role == "scrim":
        fill = colors.scrim
        border = colors.scrim
        border_width = 0.0
    commands.panel(
        rect,
        radius=radius,
        color=rgba(fill, alpha),
        border_color=rgba(border, alpha if border_width > 0.0 else 0.0),
        border_width=border_width,
    )


def draw_button(
    commands,
    text,
    theme,
    rect,
    *,
    label: str,
    size: int,
    variant: str = "secondary",
    state: InteractionState = InteractionState.REST,
    alpha: float = 1.0,
    radius: float | None = None,
    border_width: float | None = None,
) -> None:
    colors = theme.colors
    shape = theme.shape
    radius = shape.corner_m if radius is None else radius
    if variant == "primary":
        fill = colors.primary_container_hover if state == InteractionState.HOVER else colors.primary_container
        border = colors.primary_container_hover
        fg = colors.text_primary
        border_width = 0.0
    elif variant == "chip":
        active = state == InteractionState.SELECTED
        fill = colors.primary_container if active else (
            colors.secondary_container_hover if state == InteractionState.HOVER else colors.surface_variant
        )
        border = colors.outline_strong if active else colors.outline
        fg = colors.text_primary if active else colors.text_muted
        border_width = 0.0
    elif variant == "danger":
        fill = colors.tertiary_container if state != InteractionState.SELECTED else colors.error
        border = colors.error if state == InteractionState.SELECTED else colors.outline
        fg = colors.text_primary
        computed_border_width = theme.elevation.border_normal
    elif variant == "quiet":
        if state == InteractionState.SELECTED:
            fill = colors.primary_container
            border = colors.outline_strong
            fg = colors.text_primary
        else:
            fill = colors.surface_variant if state == InteractionState.HOVER else colors.surface_variant_soft
            border = colors.focus_ring if state == InteractionState.HOVER else colors.outline_variant
            fg = colors.text_primary if state == InteractionState.HOVER else colors.text_secondary
        computed_border_width = 0.0
    else:
        fill = colors.surface_variant if state == InteractionState.HOVER else colors.surface_variant_soft
        border = colors.focus_ring if state == InteractionState.HOVER else colors.outline_variant
        fg = colors.text_primary if state == InteractionState.HOVER else colors.text_secondary
        computed_border_width = 0.0
    if variant == "primary":
        computed_border_width = 0.0
    elif variant == "chip":
        computed_border_width = theme.elevation.border_normal
    applied_border_width = computed_border_width if border_width is None else border_width
    commands.panel(
        rect,
        radius=radius,
        color=rgba(fill, alpha),
        border_color=rgba(border, alpha if applied_border_width > 0.0 else 0.0),
        border_width=applied_border_width,
    )
    tw, _ = text.measure(label, size)
    commands.text(
        label,
        rect.x + (rect.w - tw) * 0.5,
        rect.y + (rect.h - size) * 0.5 - 4.0,
        size,
        color=fg,
        alpha=alpha,
    )


def draw_chip(
    commands,
    text,
    theme,
    rect,
    *,
    label: str,
    size: int,
    selected: bool = False,
    hovered: bool = False,
    disabled: bool = False,
    alpha: float = 1.0,
    style: str = "default",
) -> None:
    state = InteractionState.DISABLED if disabled else (
        InteractionState.SELECTED if selected else (
            InteractionState.HOVER if hovered else InteractionState.REST
        )
    )
    variant = "danger" if disabled else ("quiet" if style == "quiet" else "chip")
    draw_button(
        commands,
        text,
        theme,
        rect,
        label=label,
        size=size,
        variant=variant,
        state=state,
        alpha=alpha,
        radius=min(theme.shape.corner_s, rect.h * 0.5),
        border_width=0.0 if style == "quiet" else None,
    )


def draw_tab(
    commands,
    text,
    theme,
    rect,
    *,
    label: str,
    size: int,
    selected: bool = False,
    hovered: bool = False,
    alpha: float = 1.0,
) -> None:
    colors = theme.colors
    fill_alpha = alpha * (0.28 if selected else (0.16 if hovered else 0.08))
    commands.panel(
        rect,
        radius=min(theme.shape.corner_s, rect.h * 0.5),
        color=rgba(colors.surface_container, fill_alpha),
        border_color=(0.0, 0.0, 0.0, 0.0),
        border_width=0.0,
    )
    if selected:
        underline_w = max(18.0, rect.w - 20.0)
        underline_rect = type(rect)(
            rect.center_x - underline_w * 0.5,
            rect.bottom + 2.0,
            underline_w,
            2.0,
        )
        commands.panel(
            underline_rect,
            radius=1.0,
            color=rgba(colors.focus_ring, 0.86 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
    label_w, _ = text.measure(label, size)
    commands.text(
        label,
        rect.x + (rect.w - label_w) * 0.5,
        rect.y + (rect.h - size) * 0.5 - 1.0,
        size,
        color=colors.text_primary if selected else (colors.text_secondary if hovered else colors.text_muted),
        alpha=alpha,
    )


def draw_slider(
    commands,
    theme,
    track_rect,
    *,
    value: float,
    density: float,
    alpha: float = 1.0,
    thumb_scale: float = 1.0,
    panel_rect=None,
    panel_border_width: float | None = None,
    track_border_width: float | None = None,
    thumb_border_width: float | None = None,
) -> None:
    if panel_rect is not None:
        draw_surface(
            commands,
            theme,
            panel_rect,
            role="slider_panel",
            radius=max(theme.shape.corner_m, panel_rect.h * 0.5),
            alpha=0.9 * alpha,
            border_width=0.0,
        )
    draw_surface(
        commands,
        theme,
        track_rect,
        role="slider_track",
        radius=track_rect.h * 0.5,
        alpha=0.98 * alpha,
        border_width=0.0,
    )
    fill_width = max(track_rect.h, track_rect.w * value)
    fill_rect = type(track_rect)(track_rect.x, track_rect.y, fill_width, track_rect.h)
    commands.panel(
        fill_rect,
        radius=track_rect.h * 0.5,
        color=rgba(theme.colors.slider_fill, 0.96 * alpha),
        border_color=(0.0, 0.0, 0.0, 0.0),
        border_width=0.0,
    )
    thumb_w = 14.0 * density * thumb_scale
    thumb_rect = type(track_rect)(
        track_rect.x + track_rect.w * value - thumb_w * 0.5,
        track_rect.y - 4.0 * density,
        thumb_w,
        track_rect.h + 8.0 * density,
    )
    commands.panel(
        thumb_rect,
        radius=max(theme.shape.corner_s, thumb_rect.w * 0.5),
        color=rgba(theme.colors.slider_thumb, 0.96 * alpha),
        border_color=rgba(theme.colors.focus_ring, 0.90 * alpha if (thumb_border_width is None or thumb_border_width > 0.0) else 0.0),
        border_width=theme.elevation.border_normal if thumb_border_width is None else thumb_border_width,
    )


def draw_text_field(
    commands,
    text,
    theme,
    rect,
    *,
    value: str,
    placeholder: bool,
    focused: bool | float,
    size: int,
    density: float,
    alpha: float = 1.0,
    border_width: float | None = None,
) -> None:
    colors = theme.colors
    radius = theme.shape.corner_m
    applied_border_width = 0.0 if border_width is None else border_width
    focus_amount = max(0.0, min(1.0, float(focused)))
    if focus_amount > 0.001:
        glow_rect = type(rect)(
            rect.x - 2.0,
            rect.y - 2.0,
            rect.w + 4.0,
            rect.h + 4.0,
        )
        commands.panel(
            glow_rect,
            radius=radius + 2.0,
            color=rgba(colors.focus_ring, 0.12 * alpha * focus_amount),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
    commands.panel(
        rect,
        radius=radius,
        color=rgba(
            colors.surface_variant_soft if focus_amount > 0.5 else colors.surface_container_low,
            (0.60 + 0.18 * focus_amount) * alpha,
        ),
        border_color=(0.0, 0.0, 0.0, 0.0),
        border_width=applied_border_width,
    )
    inner_rect = type(rect)(
        rect.x + 1.0,
        rect.y + 1.0,
        max(0.0, rect.w - 2.0),
        max(0.0, rect.h - 2.0),
    )
    commands.panel(
        inner_rect,
        radius=max(0.0, radius - 1.0),
        color=rgba(colors.surface, (0.04 + 0.02 * focus_amount) * alpha),
        border_color=(0.0, 0.0, 0.0, 0.0),
        border_width=0.0,
    )
    accent_rect = type(rect)(
        rect.x + 8.0 * density,
        rect.y + max(7.0 * density, (rect.h - 14.0 * density) * 0.5),
        max(3.0, 3.0 * density),
        14.0 * density,
    )
    commands.panel(
        accent_rect,
        radius=accent_rect.w * 0.5,
        color=rgba(colors.focus_ring, (0.22 + 0.56 * focus_amount) * alpha),
        border_color=(0.0, 0.0, 0.0, 0.0),
        border_width=0.0,
    )
    color = colors.text_muted if placeholder else colors.text_primary
    text_x = rect.x + 22.0 * density
    text_clip_rect = type(rect)(
        text_x,
        rect.y + 2.0,
        max(0.0, rect.right - text_x - 10.0 * density),
        max(0.0, rect.h - 4.0),
    )
    display_value = value
    if hasattr(text, "truncate"):
        display_value = text.truncate(value, size, text_clip_rect.w)
    commands.clip_push(text_clip_rect)
    commands.text(
        display_value,
        text_x,
        rect.y + (rect.h - size) * 0.5 - 4.0,
        size,
        color=color,
        alpha=alpha,
    )
    commands.clip_pop()


def draw_supporting_text(
    commands,
    theme,
    value: str,
    x: float,
    y: float,
    size: int,
    *,
    alpha: float = 1.0,
    tone: str = "muted",
) -> None:
    colors = theme.colors
    color = colors.text_muted if tone == "muted" else colors.text_secondary
    commands.text(value, x, y, size, color=color, alpha=alpha)


def draw_select_field(
    commands,
    text,
    theme,
    rect,
    *,
    value: str,
    size: int,
    density: float,
    hovered: bool = False,
    focused: bool = False,
    open_progress: float = 0.0,
    placeholder: bool = False,
    alpha: float = 1.0,
    border_width: float | None = None,
) -> None:
    colors = theme.colors
    radius = theme.shape.corner_m
    is_active = focused or open_progress > 0.25
    applied_border_width = 0.0 if border_width is None else border_width
    if is_active:
        glow_rect = type(rect)(
            rect.x - 2.0,
            rect.y - 2.0,
            rect.w + 4.0,
            rect.h + 4.0,
        )
        commands.panel(
            glow_rect,
            radius=radius + 2.0,
            color=rgba(colors.focus_ring, 0.12 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
    commands.panel(
        rect,
        radius=radius,
        color=rgba(
            colors.surface_variant_soft if is_active else colors.surface_container_low,
            (0.78 if is_active else 0.60 + 0.06 * (1.0 if hovered else 0.0)) * alpha,
        ),
        border_color=(0.0, 0.0, 0.0, 0.0),
        border_width=applied_border_width,
    )
    inner_rect = type(rect)(
        rect.x + 1.0,
        rect.y + 1.0,
        max(0.0, rect.w - 2.0),
        max(0.0, rect.h - 2.0),
    )
    commands.panel(
        inner_rect,
        radius=max(0.0, radius - 1.0),
        color=rgba(colors.surface, (0.06 if is_active else 0.04) * alpha),
        border_color=(0.0, 0.0, 0.0, 0.0),
        border_width=0.0,
    )
    accent_rect = type(rect)(
        rect.x + 8.0 * density,
        rect.y + max(7.0 * density, (rect.h - 14.0 * density) * 0.5),
        max(3.0, 3.0 * density),
        14.0 * density,
    )
    commands.panel(
        accent_rect,
        radius=accent_rect.w * 0.5,
        color=rgba(colors.focus_ring, (0.78 if is_active else 0.24 + 0.08 * (1.0 if hovered else 0.0)) * alpha),
        border_color=(0.0, 0.0, 0.0, 0.0),
        border_width=0.0,
    )
    text_color = colors.text_muted if placeholder else colors.text_primary
    text_x = rect.x + 22.0 * density
    text_clip_rect = type(rect)(
        text_x,
        rect.y + 2.0,
        max(0.0, rect.right - text_x - 28.0 * density),
        max(0.0, rect.h - 4.0),
    )
    display_value = value
    if hasattr(text, "truncate"):
        display_value = text.truncate(value, size, text_clip_rect.w)
    commands.clip_push(text_clip_rect)
    commands.text(
        display_value,
        text_x,
        rect.y + (rect.h - size) * 0.5 - 4.0,
        size,
        color=text_color,
        alpha=alpha,
    )
    commands.clip_pop()
    caret = "^" if open_progress > 0.55 else "v"
    caret_size = max(12, size - 1)
    caret_w, _ = text.measure(caret, caret_size)
    commands.text(
        caret,
        rect.right - 12.0 * density - caret_w,
        rect.y + (rect.h - caret_size) * 0.5 - 1.0 - open_progress * density,
        caret_size,
        color=colors.focus_ring if open_progress > 0.25 else colors.text_secondary,
        alpha=alpha,
    )


def draw_dropdown_menu(
    commands,
    text,
    theme,
    menu_rect,
    *,
    option_rects: list,
    labels: list[str],
    selected_index: int,
    hovered_index: int,
    size: int,
    alpha: float = 1.0,
    progress: float = 1.0,
    item_gap: float = 2.0,
    edge_padding: float = 0.0,
    menu_border_width: float | None = None,
) -> None:
    progress = max(0.0, min(1.0, float(progress)))
    draw_surface(
        commands,
        theme,
        menu_rect,
        role="section",
        radius=theme.shape.corner_m,
        alpha=alpha * progress,
        border_width=theme.elevation.border_normal if menu_border_width is None else menu_border_width,
    )
    colors = theme.colors
    for idx, rect in enumerate(option_rects):
        item_progress = min(1.0, max(0.0, progress * 1.16 - idx * 0.07))
        if item_progress <= 0.01:
            continue
        option_rect = rect
        if edge_padding > 0.0:
            top_pad = edge_padding if idx == 0 else 0.0
            bottom_pad = edge_padding if idx == len(option_rects) - 1 else 0.0
            option_rect = type(rect)(
                rect.x,
                rect.y + top_pad,
                rect.w,
                max(0.0, rect.h - top_pad - bottom_pad),
            )
        if item_gap > 0.0:
            option_rect = type(option_rect)(
                option_rect.x,
                option_rect.y + item_gap * 0.5,
                option_rect.w,
                max(0.0, option_rect.h - item_gap),
            )
        option_rect = type(option_rect)(
            option_rect.x,
            option_rect.y + (1.0 - item_progress) * 8.0,
            option_rect.w,
            option_rect.h,
        )
        if idx == selected_index:
            fill = rgba(colors.primary_container, 0.90 * alpha * item_progress)
            fg = colors.text_primary
        elif idx == hovered_index:
            fill = rgba(colors.secondary_container_hover, 0.88 * alpha * item_progress)
            fg = colors.text_primary
        else:
            fill = rgba(colors.surface_variant_soft, 0.56 * alpha * item_progress)
            fg = colors.text_secondary
        commands.panel(
            option_rect,
            radius=min(theme.shape.corner_s, option_rect.h * 0.5),
            color=fill,
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        commands.text(
            labels[idx],
            option_rect.x + 12.0,
            option_rect.y + (option_rect.h - size) * 0.5 - 4.0,
            size,
            color=fg,
            alpha=0.96 * alpha * item_progress,
        )


def draw_linear_progress(
    commands,
    theme,
    rect,
    *,
    value: float,
    fill_color=None,
    alpha: float = 1.0,
    track_alpha: float = 0.42,
    track_border_width: float = 0.0,
    show_thumb: bool = False,
    thumb_scale: float = 1.0,
) -> None:
    colors = theme.colors
    value = max(0.0, min(1.0, float(value)))
    fill_color = colors.slider_fill if fill_color is None else fill_color
    draw_surface(
        commands,
        theme,
        rect,
        role="slider_track",
        radius=rect.h * 0.5,
        alpha=track_alpha * alpha,
        border_width=track_border_width,
    )
    fill_w = max(rect.h, rect.w * value)
    fill_rect = type(rect)(rect.x, rect.y, fill_w, rect.h)
    commands.panel(
        fill_rect,
        radius=rect.h * 0.5,
        color=rgba(fill_color, 0.96 * alpha),
        border_color=(0.0, 0.0, 0.0, 0.0),
        border_width=0.0,
    )
    if show_thumb:
        thumb_size = max(rect.h + 4.0, rect.h * 1.4 * thumb_scale)
        thumb_rect = type(rect)(
            rect.x + fill_w - thumb_size * 0.5,
            rect.y - (thumb_size - rect.h) * 0.5,
            thumb_size,
            thumb_size,
        )
        commands.panel(
            thumb_rect,
            radius=thumb_size * 0.5,
            color=rgba(colors.slider_thumb, 0.96 * alpha),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )


def draw_overlay_band(
    commands,
    theme,
    rect,
    *,
    alpha: float = 1.0,
    radius: float | None = None,
) -> None:
    radius = theme.shape.corner_l if radius is None else radius
    commands.panel(
        rect,
        radius=radius,
        color=rgba(theme.colors.surface_container_low, 0.66 * alpha),
        border_color=(0.0, 0.0, 0.0, 0.0),
        border_width=0.0,
    )


def draw_text_stat(
    commands,
    text,
    theme,
    *,
    label: str,
    value: str,
    x: float,
    y: float,
    label_size: int,
    value_size: int,
    value_color=None,
    caption: str | None = None,
    caption_size: int | None = None,
    alpha: float = 1.0,
) -> tuple[float, float]:
    colors = theme.colors
    value_color = colors.text_primary if value_color is None else value_color
    commands.text(label, x, y, label_size, color=colors.text_muted, alpha=0.82 * alpha)
    commands.text(value, x, y + label_size + 2.0, value_size, color=value_color, alpha=0.96 * alpha)
    bottom = y + label_size + 2.0 + value_size
    width = max(
        text.measure(label, label_size)[0],
        text.measure(value, value_size)[0],
    )
    if caption:
        cap_size = value_size if caption_size is None else caption_size
        commands.text(
            caption,
            x,
            bottom + 4.0,
            cap_size,
            color=colors.text_secondary,
            alpha=0.72 * alpha,
        )
        bottom += 4.0 + cap_size
        width = max(width, text.measure(caption, cap_size)[0])
    return (width, bottom - y)


def draw_settings_section(
    commands,
    text,
    theme,
    rect,
    *,
    title: str,
    subtitle: str = "",
    title_size: int,
    subtitle_size: int,
    padding: float,
    alpha: float = 1.0,
):
    draw_surface(
        commands,
        theme,
        rect,
        role="section",
        radius=theme.shape.corner_l,
        alpha=alpha,
        border_width=theme.elevation.border_normal,
    )
    colors = theme.colors
    title_x = rect.x + padding
    title_y = rect.y + padding
    commands.text(title, title_x, title_y, title_size, color=colors.text_primary, alpha=alpha)
    subtitle_bottom = title_y + title_size
    if subtitle:
        subtitle_y = title_y + title_size + 4.0
        commands.text(
            subtitle,
            title_x,
            subtitle_y,
            subtitle_size,
            color=colors.text_muted,
            alpha=alpha * 0.92,
        )
        subtitle_bottom = subtitle_y + subtitle_size
    top_gap = max(padding + 4.0, subtitle_bottom - rect.y + padding)
    return type(rect)(
        rect.x + padding,
        rect.y + top_gap,
        max(0.0, rect.w - padding * 2.0),
        max(0.0, rect.h - top_gap - padding),
    )


def draw_divider(commands, theme, rect, *, alpha: float = 0.22) -> None:
    commands.gradient_bar(
        rect,
        spawn_x=rect.right,
        fade_width=max(60.0, rect.w * 0.65),
        color=rgba(theme.colors.primary, alpha),
    )
