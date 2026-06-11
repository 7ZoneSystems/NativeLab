from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


BUILD_NATIVE = _env_truthy("NATIVELAB_BUILD_NATIVE")
NATIVE_ARTIFACT_PATTERNS = (
    "_native_core*.so",
    "_native_core*.pyd",
    "_native_core*.dylib",
    "_native_core*.dll",
    "libnativelab_rust.so",
    "libnativelab_rust.dylib",
    "nativelab_rust.dll",
)


class PureBuildPy(build_py):
    """Keep default wheels pure even if a previous local native build exists."""

    def run(self):
        super().run()
        if BUILD_NATIVE:
            return
        native_dir = Path(self.build_lib) / "nativelab" / "native"
        for pattern in NATIVE_ARTIFACT_PATTERNS:
            for path in native_dir.glob(pattern):
                try:
                    path.unlink()
                except OSError:
                    pass


class OptionalBuildExt(build_ext):
    """Build native helpers when possible without making installs compiler-bound."""

    def run(self):
        try:
            super().run()
        except Exception as exc:
            self.warn(f"Native helper build skipped: {exc}")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            self.warn(f"Native helper '{ext.name}' skipped: {exc}")


setup_kwargs = {
    "cmdclass": {"build_py": PureBuildPy},
}

if BUILD_NATIVE:
    setup_kwargs["ext_modules"] = [
        Extension(
            "nativelab.native._native_core",
            ["nativelab/native/_core.c"],
        ),
    ]
    setup_kwargs["cmdclass"]["build_ext"] = OptionalBuildExt


setup(**setup_kwargs)
