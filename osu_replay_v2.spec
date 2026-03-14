# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_root = Path.cwd()

# moderngl-window resolves the pyglet backend by module name at runtime,
# so we collect that backend package explicitly for frozen builds.
hiddenimports = sorted(
    set(
        collect_submodules("moderngl_window.context.pyglet")
        + collect_submodules("pyglet")
        + collect_submodules("speedups")
        + [
            "_rust_core",
            "moderngl_window.context.pyglet",
            "moderngl_window.context.pyglet.window",
            "moderngl_window.context.pyglet.keys",
        ]
    )
)

datas = collect_data_files("moderngl_window")

excludes = [
    "matplotlib",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "tkinter",
]


a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="osu_replay",
    icon=str(project_root / "static" / "icon.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="osu_replay",
)
