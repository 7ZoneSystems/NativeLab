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
from .labs_tab  import LabsTab, LAB_FEATURES
from .codeedit import CodeEditPanel, CodeEditWorker
from .pytodoc   import (
    PyToDocPanel, PyToDocWorker,
    DEFAULT_OVERVIEW_PROMPT, DEFAULT_CLASS_PROMPT, DEFAULT_FUNC_PROMPT,
    parse_python_file,
)

__all__ = [
    "LabEndpoints",
    "LabsTab", "LAB_FEATURES",
    "CodeEditPanel", "CodeEditWorker",
    "PyToDocPanel", "PyToDocWorker",
    "DEFAULT_OVERVIEW_PROMPT", "DEFAULT_CLASS_PROMPT", "DEFAULT_FUNC_PROMPT",
    "parse_python_file",
]
