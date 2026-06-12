from __future__ import annotations

import sys
import types
from pathlib import Path

if __package__ in {None, ""}:
    package_dir = Path(__file__).resolve().parent
    package = types.ModuleType("PhonoLab")
    package.__path__ = [str(package_dir)]
    sys.modules.setdefault("PhonoLab", package)
    __package__ = "PhonoLab"

def main() -> int:
    try:
        from .ui.kivy_app import run
    except Exception as exc:
        print(
            "PhonoLab is GUI-only. Install Kivy for the Python preview or open "
            f"PhonoLab/android in Android Studio. Details: {exc}",
            file=sys.stderr,
        )
        return 1
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
