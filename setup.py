from __future__ import annotations

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


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


setup(
    ext_modules=[
        Extension(
            "nativelab.native._native_core",
            ["nativelab/native/_core.c"],
        ),
    ],
    cmdclass={"build_ext": OptionalBuildExt},
)
