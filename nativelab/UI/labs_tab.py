"""
Backward-compat shim.

The labs feature lives under `nativelab.labs` now. This module re-exports the
public symbols so any older import paths (`from nativelab.UI.labs_tab import …`)
keep working.
"""
from nativelab.labs import (
    LabEndpoints,
    LabsTab,
    LAB_FEATURES,
    PyToDocPanel,
    PyToDocWorker,
    DEFAULT_OVERVIEW_PROMPT,
    DEFAULT_CLASS_PROMPT,
    DEFAULT_FUNC_PROMPT,
    parse_python_file,
)

__all__ = [
    "LabEndpoints",
    "LabsTab", "LAB_FEATURES",
    "PyToDocPanel", "PyToDocWorker",
    "DEFAULT_OVERVIEW_PROMPT", "DEFAULT_CLASS_PROMPT", "DEFAULT_FUNC_PROMPT",
    "parse_python_file",
]
