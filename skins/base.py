"""Abstract base class for visual skins."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Skin(ABC):
    name: str = "unnamed"

    @abstractmethod
    def circle_shader_source(self) -> tuple[str, str]:
        """Return (vertex_glsl, fragment_glsl) for hit circles."""

    @abstractmethod
    def slider_shader_source(self) -> tuple[str, str]:
        """Return (vertex_glsl, fragment_glsl) for slider bodies."""

    @abstractmethod
    def combo_colors(self) -> list[tuple[float, float, float]]:
        """RGB combo colour palette."""

    def circle_fill_color(self, *, slider_head: bool = False) -> tuple[float, float, float]:
        """Return the active fill colour for circles or slider heads."""
        colors = self.combo_colors()
        return colors[0] if colors else (1.0, 1.0, 1.0)

    def slider_fill_color(self) -> tuple[float, float, float]:
        """Return the active slider body fill colour."""
        return self.circle_fill_color()

    def slider_ball_fill_color(self) -> tuple[float, float, float]:
        """Return the active slider ball fill colour."""
        return self.circle_fill_color()

    def sync_object_uniforms(
        self,
        *,
        circle_prog=None,
        approach_prog=None,
        slider_prog=None,
        spinner_prog=None,
        ball_prog=None,
    ) -> None:
        """Push live style uniforms into already-compiled gameplay programs."""

    def set_visual_settings(self, settings) -> None:
        """Apply runtime-editable visual settings to the active skin."""

    def approach_shader_source(self) -> tuple[str, str] | None:
        """Return (vertex_glsl, fragment_glsl) for approach circles, or None to skip."""
        return None

    def spinner_shader_source(self) -> tuple[str, str] | None:
        """Return (vertex_glsl, fragment_glsl) for spinners, or None to skip."""
        return None

    def slider_ball_shader_source(self) -> tuple[str, str] | None:
        """Return (vertex_glsl, fragment_glsl) for slider ball, or None to skip."""
        return None

    def cursor_shader_source(self) -> tuple[str, str] | None:
        """Return (vertex_glsl, fragment_glsl) for cursor + trail, or None to skip."""
        return None

    def trail_shader_source(self) -> tuple[str, ...] | None:
        """Return (vertex_glsl, fragment_glsl) or (vertex, geometry, fragment) for trail."""
        return None

    def begin_trail_pass(self, ctx) -> None:
        """Prepare an isolated pass for trail rendering, if desired."""

    def end_trail_pass(self, ctx) -> None:
        """Composite the isolated trail pass back into the active target, if used."""

    def setup(self, ctx) -> None:
        """Create custom GPU resources when skin is activated."""

    def cleanup(self) -> None:
        """Release GPU resources when skin is deactivated."""

    def on_pre_render(self, ctx, time_ms: float, frametime: float) -> None:
        """Hook called before object rendering."""

    def on_post_render(self, ctx, time_ms: float, frametime: float) -> None:
        """Hook called after object rendering."""

    def render_atmosphere(self, ctx, time_ms: float) -> None:
        """Render atmospheric effects into the scene FBO before bloom."""

    def set_atmosphere_enabled(self, enabled: bool) -> None:
        """Enable or disable atmospheric background effects."""

    def set_bloom_intensity(self, intensity: float) -> None:
        """Set additive bloom intensity used during post-processing."""
