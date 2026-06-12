from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class PhonoLabBuildPy(build_py):
    """Keep the Python preview wheel separate from the Android Studio project."""

    def run(self):
        super().run()
        package_root = Path(self.build_lib) / "PhonoLab"
        setup_copy = package_root / "setup.py"
        if setup_copy.exists():
            setup_copy.unlink()
        for name in ("android", "build", "tests"):
            target = package_root / name
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)


setup(cmdclass={"build_py": PhonoLabBuildPy})
