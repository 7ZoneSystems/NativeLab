"""NativeLab terminal client.

Routed from `__main__.py` whenever the user passes `--cli`. Re-uses the same
`LabEndpoints` abstraction the GUI's Labs tab is built on, so the CLI and the
desktop app share a single backend surface.
"""
from .app import run as run_cli

__all__ = ["run_cli"]
