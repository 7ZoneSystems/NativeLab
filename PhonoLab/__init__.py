"""PhonoLab mobile node for NativeLab.

The mobile node is intentionally small: it keeps a llama.cpp-only runtime,
mobile-safe model registry, resumable downloads, and a Kivy UI that delegates
all blocking work to background threads.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
