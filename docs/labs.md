# Labs - the experimentation layer

The `nativelab/labs/` package is where new features land. It exists to let you ship experiments without touching `MainWindow`, engine internals, or the streaming workers.

> **TL;DR** - drop a `QWidget` panel into `nativelab/labs/<feature>.py`, register it in `LAB_FEATURES`, and you get engine status, model swap, context change, and synchronous LLM calls for free.

---

## What's in the box

```
nativelab/labs/
├── __init__.py        re-exports LabEndpoints, LabsTab, LAB_FEATURES, …
├── endpoints.py       LabEndpoints - the shared surface
├── labs_tab.py        sidebar + stacked panels (LabsTab + LAB_FEATURES list)
└── pytodoc.py         first feature: py-to-doc README generator
```

Labs now live under **Dev > Labs**. The Dev tab swaps the normal chat-history
sidebar for a vertical developer sidebar containing Labs, Logs, Integrations,
Pipeline, MCP, and Skills.

### py-to-doc project recovery

In py-to-doc Project mode, NativeLab writes a restart-safe checkpoint to:

```text
localllm/temp
```

The checkpoint is updated after each generated file section, including class and
function documentation. If the app is paused, closed, or the machine shuts down,
select the same project root and output folder, keep the same settings, and run
Generate again. NativeLab verifies the checkpoint and existing markdown files,
then resumes from the last completed step.

The **Resume Project** button requires a matching checkpoint and stops with a
clear message if the selected project/output/settings do not match the saved
state.

The py-to-doc panel also shows saved project tasks from `localllm/temp` and
`localllm/pytodoc_jobs`. Selecting one restores the project/output/settings
when available and resumes from the saved checkpoint automatically.

Project mode scans Python files recursively, applies the selected project
root's `.gitignore`, and creates a matching directory tree in the output folder
before writing generated markdown files.

py-to-doc context can run in three modes: no reset, fixed reset after classes
or functions, or auto budget reset. Auto budget reset uses approximate tokens
for py-to-doc's carried history. For local GGUF models, selecting an auto
budget reloads the local model/server to the same context window before
generation begins. Auto mode can also reload the active local model after the
current file when available RAM drops below the configured GB/MB threshold,
clearing backend caches before the next file starts. If a class/function/file
generation crosses the budget, that generation finishes and the next section
starts with fresh carried context.

### structured-edit lab

The structured-edit lab lives under **Dev > Labs**. It can attach an existing
code file or start from an empty temp workspace, then ask the active model to
return structured edit operations instead of rewriting the whole file.

The working copy is saved continuously to:

```text
localllm/temp_code_edit.json
localllm/temp_code_edit_file
```

The original file is not changed until **Save** or **Save As** is clicked. The
lab shows the parsed file structure, including detected functions, line ranges,
arguments, return expressions, and local variables where available.

---

## The contract for a lab panel

A panel is any `QWidget` subclass that:

1. Defines `LAB_NAME` and `LAB_ICON` class attributes.
2. Implements `set_endpoints(endpoints: LabEndpoints)`.
3. Uses `endpoints` for all engine reads, LLM calls, and reverse routing.

That's it. No `MainWindow` import, no engine import, no streaming worker plumbing.

---

## The endpoint surface

```python
from nativelab.labs import LabEndpoints
```

### Read state

```python
endpoints.status_text     # "🟢 Server  :8612"
endpoints.is_loaded       # bool
endpoints.mode            # "server" | "cli" | "api" | "unloaded"
endpoints.model_path      # absolute GGUF path
endpoints.model_name      # filename only
endpoints.ctx_value       # int
endpoints.server_port     # int
endpoints.is_api_active   # bool
endpoints.is_local_active # bool
endpoints.is_loading      # bool
endpoints.snapshot()      # all of the above as a dict
endpoints.model_family()  # ModelFamily template (BOS/EOS, prefixes, stops)
```

### Synchronous LLM call

Auto-routes API → server → CLI based on which engine is active. Safe to call from a `QThread`.

```python
reply = endpoints.call_llm(
    messages=[
        {"role": "user", "content": "Summarise this:\n" + code},
    ],
    system_prompt="You are a senior engineer.",
    n_predict=512,
    temperature=0.3,
)
```

### Reverse routing - ask the host to change state

```python
endpoints.request_load_model("/path/to/other.gguf")  # → True/False
endpoints.request_context(8192)                      # → True/False
endpoints.request_active_model_reload()              # → True/False
endpoints.wait_until_loaded()                        # → True/False
endpoints.request_unload()                           # → None
endpoints.ensure_server(log_cb=lambda m: print(m))   # → True/False
```

The host (`MainWindow` for the GUI, `cli/chat.py` for the CLI) wires these hooks at startup. Same panel works under both.

### Signals

```python
endpoints.engine_changed.connect(self._refresh)
endpoints.status_changed.connect(self._update_label)   # str
endpoints.log_msg.connect(self._on_log)                # (level, message)
```

---

## How to add a new lab feature

### 1. Create the module

```
nativelab/labs/codereview.py
```

```python
from __future__ import annotations
from typing import Optional

from nativelab.imports.import_global import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton,
)
from .endpoints import LabEndpoints


class CodeReviewPanel(QWidget):
    LAB_NAME = "code-review"
    LAB_ICON = "🔍"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._endpoints: Optional[LabEndpoints] = None
        self._build()

    def set_endpoints(self, endpoints: LabEndpoints):
        self._endpoints = endpoints
        endpoints.status_changed.connect(self.lbl_status.setText)
        self.lbl_status.setText(endpoints.status_text)

    def _build(self):
        root = QVBoxLayout(self)
        root.addWidget(QLabel(f"{self.LAB_ICON}  {self.LAB_NAME}"))
        self.lbl_status = QLabel("…")
        root.addWidget(self.lbl_status)

        self.input = QTextEdit()
        root.addWidget(self.input, 1)

        self.btn = QPushButton("Review")
        self.btn.clicked.connect(self._run)
        root.addWidget(self.btn)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        root.addWidget(self.output, 1)

    def _run(self):
        if not self._endpoints or not self._endpoints.is_loaded:
            self.output.setPlainText("No engine loaded.")
            return
        code = self.input.toPlainText()
        try:
            reply = self._endpoints.call_llm(
                prompt=f"Review this code and list issues:\n\n{code}",
                system_prompt="You are a senior code reviewer.",
            )
        except Exception as exc:
            reply = f"[error: {exc}]"
        self.output.setPlainText(reply)
```

### 2. Register it

In `nativelab/labs/labs_tab.py`:

```python
from .codereview import CodeReviewPanel

LAB_FEATURES: list[Type[QWidget]] = [
    PyToDocPanel,
    CodeReviewPanel,    # ← here
]
```

### 3. That's the whole change

Restart the app. Your panel shows up in **Dev > Labs** with the icon and name from the class attributes. It receives a `LabEndpoints` instance automatically.

---

## Why this exists

Before Labs, adding a feature touched `MainWindow` (engine refs), `tabs.py` (UI definition), the streaming workers (custom calls), and probably the model registry too. Removing one feature meant chasing references through every layer.

The labs surface is intentionally narrow:

- **Read state** - no state mutation through reads.
- **`call_llm`** - one synchronous call, one return value, no QThread choreography.
- **Reverse routing** - three explicit hooks; no direct engine handles leak out.
- **Signals** - engine changes notify panels without panels polling.

If you find yourself wanting to import an engine class, a streaming worker, or `MainWindow` from inside a lab feature - stop. The feature should either work through `endpoints.call_llm` or the surface needs a new method. PRs to extend `LabEndpoints` are welcome and easy to review precisely *because* the surface is small.

---

## How the CLI uses the same surface

`nativelab/cli/chat.py` builds a `LabEndpoints` exactly like `MainWindow` does - different reverse-route hooks (synchronous CLI behavior instead of GUI dialog flows), same call semantics. That's why `/load`, `/ctx`, and `/unload` from the REPL behave identically to a lab feature requesting the same change in the GUI: they're literally the same code path.

```python
# nativelab/cli/chat.py - the relevant snippet
endpoints.bind_engines(
    llama_provider=lambda: eng,
    api_provider  =lambda: api,
)
endpoints.bind_reverse_routes(
    on_context=on_context,
    on_model  =on_model,
    on_unload =on_unload,
)
```

---

## Inventory

| Feature       | Status   | Module              |
| ------------- | -------- | ------------------- |
| py-to-doc     | ✅ shipped | `labs/pytodoc.py`   |

More are being prototyped. If you ship one, add a row.
