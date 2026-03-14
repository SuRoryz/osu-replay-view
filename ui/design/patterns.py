"""Higher-level reusable text and layout patterns."""

from __future__ import annotations


def draw_header_pair(
    commands,
    text,
    theme,
    *,
    label: str,
    meta: str,
    x: float,
    y: float,
    label_size: int,
    meta_size: int,
    label_color=None,
    meta_color=None,
) -> None:
    colors = theme.colors
    label_color = colors.text_primary if label_color is None else label_color
    meta_color = colors.text_muted if meta_color is None else meta_color
    commands.text(label, x, y, label_size, color=label_color)
    if not meta:
        return
    label_w, _ = text.measure(label, label_size)
    commands.text(meta, x + label_w + 10.0, y + 5.0, meta_size, color=meta_color)
