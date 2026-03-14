"""Build optional native speedups in-place."""

from __future__ import annotations

from pathlib import Path

import numpy
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - build-time script
    raise SystemExit("Cython is required to build native speedups. Install it first.") from exc


ROOT = Path(__file__).resolve().parent


extensions = [
    Extension(
        "speedups._cy_gameplay",
        [str(ROOT / "speedups" / "_cy_gameplay.pyx")],
        include_dirs=[numpy.get_include()],
    ),
]


setup(
    name="osu-replay-native-speedups",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
