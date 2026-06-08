"""
Labs - experimentation layer for NativeLab.

Adding a new lab feature
------------------------
1. Create `nativelab/labs/<feature>.py`.
2. Define a `QWidget` subclass with `LAB_NAME`, `LAB_ICON`, and a
   `set_endpoints(endpoints: LabEndpoints)` method.
3. Use `endpoints` for all engine reads (status, model_path, ctx_value),
   LLM calls (`endpoints.call_llm(...)`), and reverse routing
   (`endpoints.request_context(...)`, `endpoints.request_load_model(...)`).
4. Register the panel in `LAB_FEATURES` in `labs_tab.py`.

Doing so means new labs never touch `MainWindow` or engine internals.
"""
from .endpoints import LabEndpoints

__all__ = [
    "LabEndpoints",
    "LabsTab", "LAB_FEATURES",
    "CodeEditPanel", "CodeEditWorker",
    "PyToDocPanel", "PyToDocWorker",
    "DEFAULT_OVERVIEW_PROMPT", "DEFAULT_CLASS_PROMPT", "DEFAULT_FUNC_PROMPT",
    "parse_python_file",
]


def __getattr__(name: str):
    if name in {"LabsTab", "LAB_FEATURES"}:
        from .labs_tab import LAB_FEATURES, LabsTab
        return {"LabsTab": LabsTab, "LAB_FEATURES": LAB_FEATURES}[name]
    if name in {"CodeEditPanel", "CodeEditWorker"}:
        from .codeedit import CodeEditPanel, CodeEditWorker
        return {"CodeEditPanel": CodeEditPanel, "CodeEditWorker": CodeEditWorker}[name]
    if name in {
        "PyToDocPanel",
        "PyToDocWorker",
        "DEFAULT_OVERVIEW_PROMPT",
        "DEFAULT_CLASS_PROMPT",
        "DEFAULT_FUNC_PROMPT",
        "parse_python_file",
    }:
        from .pytodoc import (
            DEFAULT_CLASS_PROMPT,
            DEFAULT_FUNC_PROMPT,
            DEFAULT_OVERVIEW_PROMPT,
            PyToDocPanel,
            PyToDocWorker,
            parse_python_file,
        )
        return {
            "PyToDocPanel": PyToDocPanel,
            "PyToDocWorker": PyToDocWorker,
            "DEFAULT_OVERVIEW_PROMPT": DEFAULT_OVERVIEW_PROMPT,
            "DEFAULT_CLASS_PROMPT": DEFAULT_CLASS_PROMPT,
            "DEFAULT_FUNC_PROMPT": DEFAULT_FUNC_PROMPT,
            "parse_python_file": parse_python_file,
        }[name]
    raise AttributeError(name)
