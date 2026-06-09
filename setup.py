from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class NativeBuildExt(build_ext):
    """Build C extensions and the optional dependency-free Rust cdylib."""

    def run(self):
        super().run()
        self._build_rust_helpers()

    def _rust_library_name(self) -> str:
        system = platform.system()
        if system == "Windows":
            return "nativelab_rust.dll"
        if system == "Darwin":
            return "libnativelab_rust.dylib"
        return "libnativelab_rust.so"

    def _build_rust_helpers(self) -> None:
        rustc = shutil.which("rustc")
        src = Path(__file__).resolve().parent / "nativelab" / "native" / "rust_model.rs"
        if not src.exists():
            return
        if not rustc:
            self.announce("rustc not found; Rust model helpers will use Python fallback", level=2)
            return

        dest_dirs = [Path(self.build_lib) / "nativelab" / "native"]
        if self.inplace:
            dest_dirs.append(Path(__file__).resolve().parent / "nativelab" / "native")

        for dest_dir in dest_dirs:
            dest_dir.mkdir(parents=True, exist_ok=True)
            out = dest_dir / self._rust_library_name()
            cmd = [
                rustc,
                "--crate-type",
                "cdylib",
                "-O",
                str(src),
                "-o",
                str(out),
            ]
            try:
                subprocess.check_call(cmd)
            except Exception as exc:
                self.announce(f"Rust helper build skipped: {exc}", level=2)
                return


setup(
    ext_modules=[
        Extension(
            "nativelab.native._native_core",
            ["nativelab/native/_core.c"],
            optional=True,
        )
    ],
    cmdclass={"build_ext": NativeBuildExt},
)
