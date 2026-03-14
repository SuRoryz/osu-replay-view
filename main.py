import pyglet

from app import App, AppSettings

if __name__ == "__main__":
    settings = AppSettings.load(App.settings_path)
    App.window_size = (
        int(settings.resolution_width),
        int(settings.resolution_height),
    )
    # moderngl_window's pyglet backend still expects pyglet.canvas during
    # fullscreen window creation, which is missing on newer pyglet builds.
    # Only request fullscreen at startup when that API exists.
    App.fullscreen = (
        settings.screen_mode in {"fullscreen", "borderless"}
        and hasattr(pyglet, "canvas")
    )
    App.run()
