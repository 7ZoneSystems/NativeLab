"""
Lab feature: py-to-doc.

Generates structured README-style documentation for a Python file by feeding
its overview, classes, and functions through the active LLM. The feature talks
to the rest of the app exclusively through `LabEndpoints`, so it works
identically against a local llama-server, llama-cli, or a remote API model.
"""
from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from nativelab.imports.import_global import (
    QThread, pyqtSignal,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTextEdit, QFrame, QScrollArea, QCheckBox, QFileDialog, QMessageBox,
    QFont, Qt,
)

from .endpoints import LabEndpoints


# ─────────────────────────────────────────────────────────────────────────────
#  Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OVERVIEW_PROMPT = (
    "Generate a structured overview of this Python file. "
    "Describe its purpose, key components, and overall design. "
    "Keep to 2–4 sentences. Use plain prose - do not add headings."
)
DEFAULT_CLASS_PROMPT = (
    "Describe this Python class concisely. Cover its purpose, key attributes, "
    "and what it is responsible for. Do NOT list methods - they will be "
    "documented separately. Keep to 2–4 sentences."
)
DEFAULT_FUNC_PROMPT = (
    "Document this Python function. Include: purpose, parameters (with types "
    "if inferrable), return value, and any important behaviour or side-effects. "
    "Be concise but precise."
)


# ─────────────────────────────────────────────────────────────────────────────
#  AST helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_source(node: ast.AST, source_lines: list[str]) -> str:
    try:
        return ast.get_source_segment("\n".join(source_lines), node) or ""
    except Exception:
        start = getattr(node, "lineno", 1) - 1
        end   = getattr(node, "end_lineno", start + 1)
        return "\n".join(source_lines[start:end])


def parse_python_file(path: str) -> dict:
    src   = Path(path).read_text(encoding="utf-8")
    tree  = ast.parse(src)
    lines = src.splitlines()

    classes:   list[dict] = []
    functions: list[dict] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                {"name": m.name, "node": m}
                for m in ast.walk(node)
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and m is not node
            ]
            classes.append({"name": node.name, "node": node, "methods": methods})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({"name": node.name, "node": node})

    return {"source": src, "lines": lines,
            "classes": classes, "functions": functions}


# ─────────────────────────────────────────────────────────────────────────────
#  Worker
# ─────────────────────────────────────────────────────────────────────────────

class PyToDocWorker(QThread):
    log_msg = pyqtSignal(str)
    chunk   = pyqtSignal(str)
    done    = pyqtSignal()
    error   = pyqtSignal(str)

    def __init__(
        self,
        file_path:           str,
        out_path:            str,
        out_name:            str,
        include_globals:     bool,
        reset_per_function:  bool,
        reset_per_class:     bool,
        prompt_overview:     str,
        prompt_class:        str,
        prompt_function:     str,
        endpoints:           LabEndpoints,
        parent=None,
    ):
        super().__init__(parent)
        self.file_path          = file_path
        self.out_path           = out_path
        self.out_name           = out_name
        self.include_globals    = include_globals
        self.reset_per_function = reset_per_function
        self.reset_per_class    = reset_per_class
        self.prompt_overview    = prompt_overview or DEFAULT_OVERVIEW_PROMPT
        self.prompt_class       = prompt_class    or DEFAULT_CLASS_PROMPT
        self.prompt_function    = prompt_function or DEFAULT_FUNC_PROMPT
        self.endpoints          = endpoints
        self._abort             = False
        self._history: List[dict] = []

    def abort(self):
        self._abort = True

    # ── helpers ──────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_msg.emit(f"[{ts}]  {msg}")

    def _write(self, fh, text: str):
        fh.write(text)
        fh.flush()
        self.chunk.emit(text)

    def _reset_context(self):
        self._history.clear()

    def _call_llm(self, system_prompt: str, code: str) -> str:
        if self.endpoints is None or not self.endpoints.is_loaded:
            return "(no LLM engine loaded - load a model first)"

        user_content = f"{system_prompt}\n\n```python\n{code}\n```"
        self._history.append({"role": "user", "content": user_content})

        try:
            text = self.endpoints.call_llm(messages=self._history).strip()
        except Exception as exc:
            text = f"[LLM error: {exc}]"

        self._history.append({"role": "assistant", "content": text})
        return text

    # ── pipeline ─────────────────────────────────────────────────────────────
    def run(self):
        try:
            self._pipeline()
        except Exception as exc:
            self.error.emit(str(exc))

    def _pipeline(self):
        if self._abort:
            return

        self._log(f"Parsing: {Path(self.file_path).name}")
        parsed = parse_python_file(self.file_path)
        src    = parsed["source"]
        lines  = parsed["lines"]
        fname  = Path(self.file_path).stem

        out_dir = Path(self.out_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / self.out_name
        fh = out_file.open("w", encoding="utf-8")

        try:
            self._run_pipeline(fh, fname, src, lines, parsed)
        finally:
            fh.close()

        self._log(f"README saved → {out_file}")
        self.done.emit()

    def _run_pipeline(self, fh, fname, src, lines, parsed):
        self._write(fh, f"## Doc for {fname}\n---\n\n")
        self._log("README initialized")

        if self._abort: return
        self._log("Generating overview…")
        overview_text = self._call_llm(self.prompt_overview, src)
        self._write(fh, f"{overview_text}\n\n---\n\n")
        self._log("Overview generated")

        for cls in parsed["classes"]:
            if self._abort: return
            if self.reset_per_class:
                self._reset_context()

            class_code = _get_source(cls["node"], lines)
            self._log(f"Processing class: {cls['name']}")

            class_desc = self._call_llm(self.prompt_class, class_code)
            self._write(fh, f"### {cls['name']}\n{class_desc}\n\n")

            self._write(fh, "### Functions\n\n")
            for method in cls["methods"]:
                if self._abort: return
                if self.reset_per_function:
                    self._reset_context()

                fn_code = _get_source(method["node"], lines)
                fn_out  = self._call_llm(self.prompt_function, fn_code)
                self._write(
                    fh,
                    f"#### `{method['name']}`\n\n"
                    f"{fn_out.strip()}\n\n"
                    f"---\n\n"
                )
                self._log(f"Function processed: {method['name']}")

            self._log(f"Class processed: {cls['name']}")
            self._write(fh, "\n")

        if self.include_globals and parsed["functions"]:
            if self._abort: return
            self._write(fh, "---\n\n## Global Functions\n\n### Functions\n\n")
            for fn in parsed["functions"]:
                if self._abort: return
                if self.reset_per_function:
                    self._reset_context()

                fn_code = _get_source(fn["node"], lines)
                fn_out  = self._call_llm(self.prompt_function, fn_code)
                self._write(
                    fh,
                    f"#### `{fn['name']}`\n\n"
                    f"{fn_out.strip()}\n\n"
                    f"---\n\n"
                )
                self._log(f"Function processed: {fn['name']}")


# ─────────────────────────────────────────────────────────────────────────────
#  Panel
# ─────────────────────────────────────────────────────────────────────────────

class PyToDocPanel(QWidget):
    """UI panel for the py-to-doc lab."""

    LAB_NAME = "py-to-doc"
    LAB_ICON = "📄"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker:    Optional[PyToDocWorker] = None
        self._endpoints: Optional[LabEndpoints]  = None
        self._build()

    # ── endpoint wiring ──────────────────────────────────────────────────────
    def set_endpoints(self, endpoints: LabEndpoints):
        self._endpoints = endpoints
        endpoints.status_changed.connect(self._on_status_changed)
        self._on_status_changed(endpoints.status_text)

    def _on_status_changed(self, status: str):
        if hasattr(self, "lbl_engine"):
            self.lbl_engine.setText(f"Active engine: {status}")

    # ── build ────────────────────────────────────────────────────────────────
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("chat_scroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        inner.setObjectName("chat_container")
        root = QVBoxLayout(inner)
        root.setContentsMargins(22, 18, 22, 22)
        root.setSpacing(0)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # Header
        hdr = QLabel(f"{self.LAB_ICON}  {self.LAB_NAME}")
        hdr.setObjectName("labs_panel_header")
        root.addWidget(hdr)

        sub = QLabel(
            "Convert a Python file into structured README-style documentation "
            "using your loaded LLM. Classes and functions are fed individually "
            "for precise, well-scoped output."
        )
        sub.setWordWrap(True)
        sub.setObjectName("txt2_small")
        sub.setStyleSheet("margin-bottom:18px;")
        root.addWidget(sub)

        # Active engine indicator
        self.lbl_engine = QLabel("Active engine: ⚪ No Engine")
        self.lbl_engine.setObjectName("txt2_small")
        self.lbl_engine.setStyleSheet("margin-bottom:10px;")
        root.addWidget(self.lbl_engine)

        # File settings
        root.addWidget(self._section_label("FILE SETTINGS"))
        file_card = self._card()
        fc = QVBoxLayout(file_card)
        fc.setContentsMargins(16, 14, 16, 14)
        fc.setSpacing(10)

        self.inp_src = QLineEdit()
        self.inp_src.setPlaceholderText("Select Python file to document…")
        self.inp_src.setReadOnly(True)
        self.inp_src.setFixedHeight(30)
        btn_browse_src = QPushButton("Browse…")
        btn_browse_src.setFixedHeight(30)
        btn_browse_src.setFixedWidth(80)
        btn_browse_src.clicked.connect(self._browse_src)
        fc.addLayout(self._field_row("Python file:", self.inp_src, btn_browse_src))

        self.inp_out_dir = QLineEdit()
        self.inp_out_dir.setPlaceholderText("Select output folder…")
        self.inp_out_dir.setReadOnly(True)
        self.inp_out_dir.setFixedHeight(30)
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.setFixedHeight(30)
        btn_browse_out.setFixedWidth(80)
        btn_browse_out.clicked.connect(self._browse_out)
        fc.addLayout(self._field_row("Output folder:", self.inp_out_dir, btn_browse_out))

        self.inp_out_name = QLineEdit("README.md")
        self.inp_out_name.setFixedHeight(30)
        fc.addLayout(self._field_row("Output filename:", self.inp_out_name))

        root.addWidget(file_card)
        root.addSpacing(14)

        # Options
        root.addWidget(self._section_label("OPTIONS"))
        opt_card = self._card()
        oc = QVBoxLayout(opt_card)
        oc.setContentsMargins(16, 14, 16, 14)
        oc.setSpacing(8)

        self.chk_globals    = QCheckBox("Include module-level (global) functions")
        self.chk_reset_fn   = QCheckBox("Reset LLM context after each function")
        self.chk_reset_cls  = QCheckBox("Reset LLM context after each class")
        self.chk_reset_fn.setChecked(True)
        for chk in (self.chk_globals, self.chk_reset_fn, self.chk_reset_cls):
            oc.addWidget(chk)

        root.addWidget(opt_card)
        root.addSpacing(14)

        # Custom prompts
        root.addWidget(self._section_label("CUSTOM PROMPTS  (leave blank for defaults)"))
        prompt_card = self._card()
        pc = QVBoxLayout(prompt_card)
        pc.setContentsMargins(16, 14, 16, 14)
        pc.setSpacing(10)

        self.inp_prompt_overview = self._prompt_edit(
            "Overview prompt…", DEFAULT_OVERVIEW_PROMPT)
        self.inp_prompt_class    = self._prompt_edit(
            "Class description prompt…", DEFAULT_CLASS_PROMPT)
        self.inp_prompt_function = self._prompt_edit(
            "Function description prompt…", DEFAULT_FUNC_PROMPT)

        for label, widget in (
            ("Overview prompt:", self.inp_prompt_overview),
            ("Class prompt:",    self.inp_prompt_class),
            ("Function prompt:", self.inp_prompt_function),
        ):
            lbl = QLabel(label)
            lbl.setObjectName("txt2_small")
            pc.addWidget(lbl)
            pc.addWidget(widget)

        root.addWidget(prompt_card)
        root.addSpacing(18)

        # Generate / Abort
        btn_row = QHBoxLayout()
        self.btn_generate = QPushButton("⚙️  Generate Documentation")
        self.btn_generate.setObjectName("labs_generate_btn")
        self.btn_generate.setMinimumHeight(38)
        self.btn_generate.clicked.connect(self._run_py_to_doc)
        self.btn_abort = QPushButton("⏹  Abort")
        self.btn_abort.setObjectName("btn_stop")
        self.btn_abort.setFixedHeight(38)
        self.btn_abort.setVisible(False)
        self.btn_abort.clicked.connect(self._abort)
        btn_row.addWidget(self.btn_generate, 1)
        btn_row.addWidget(self.btn_abort)
        root.addLayout(btn_row)
        root.addSpacing(14)

        # Log
        root.addWidget(self._section_label("PIPELINE LOG"))
        log_card = self._card()
        lc = QVBoxLayout(log_card)
        lc.setContentsMargins(14, 10, 14, 12)
        self.log_te = QTextEdit()
        self.log_te.setObjectName("log_te")
        self.log_te.setReadOnly(True)
        self.log_te.setFixedHeight(130)
        self.log_te.setFont(QFont("Consolas", 10))
        lc.addWidget(self.log_te)
        root.addWidget(log_card)
        root.addSpacing(14)

        # Preview
        root.addWidget(self._section_label("LIVE OUTPUT PREVIEW"))
        prev_card = self._card()
        pvc = QVBoxLayout(prev_card)
        pvc.setContentsMargins(14, 10, 14, 12)
        self.preview_te = QTextEdit()
        self.preview_te.setObjectName("labs_preview_te")
        self.preview_te.setReadOnly(True)
        self.preview_te.setMinimumHeight(260)
        self.preview_te.setFont(QFont("Consolas", 10))
        self.preview_te.setPlaceholderText(
            "Generated documentation will stream here in real time…")
        pvc.addWidget(self.preview_te)

        btn_copy = QPushButton("📋  Copy to Clipboard")
        btn_copy.setFixedHeight(28)
        btn_copy.clicked.connect(
            lambda: self.preview_te.selectAll() or self.preview_te.copy())
        pvc.addWidget(btn_copy)
        root.addWidget(prev_card)

        root.addStretch()

    # ── widget helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-size:12px;font-weight:bold;"
            "letter-spacing:0.5px;padding:0;margin-bottom:2px;"
        )
        return lbl

    @staticmethod
    def _card() -> QFrame:
        f = QFrame()
        f.setObjectName("tab_card")
        return f

    @staticmethod
    def _field_row(label_text: str, widget, btn=None) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        lbl = QLabel(label_text)
        lbl.setObjectName("txt2")
        lbl.setFixedWidth(110)
        row.addWidget(lbl)
        row.addWidget(widget, 1)
        if btn:
            row.addWidget(btn)
        return row

    @staticmethod
    def _prompt_edit(placeholder: str, default_text: str) -> QTextEdit:
        te = QTextEdit()
        te.setPlaceholderText(placeholder)
        te.setPlainText(default_text)
        te.setFixedHeight(68)
        return te

    # ── actions ──────────────────────────────────────────────────────────────
    def _browse_src(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Python File",
            str(Path(self.inp_src.text()).parent
                if self.inp_src.text() else Path.home()),
            "Python Files (*.py);;All Files (*)"
        )
        if path:
            self.inp_src.setText(path)
            if not self.inp_out_dir.text():
                self.inp_out_dir.setText(str(Path(path).parent))

    def _browse_out(self):
        p = QFileDialog.getExistingDirectory(
            self, "Select Output Folder",
            self.inp_out_dir.text() or str(Path.home())
        )
        if p:
            self.inp_out_dir.setText(p)

    def _log(self, msg: str):
        self.log_te.append(msg)
        self.log_te.verticalScrollBar().setValue(
            self.log_te.verticalScrollBar().maximum())

    def _on_chunk(self, text: str):
        cursor = self.preview_te.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.preview_te.setTextCursor(cursor)
        self.preview_te.verticalScrollBar().setValue(
            self.preview_te.verticalScrollBar().maximum())

    def _abort(self):
        if self._worker:
            self._worker.abort()
        self._log("[aborted by user]")
        self._set_running(False)

    def _set_running(self, running: bool):
        self.btn_generate.setEnabled(not running)
        self.btn_abort.setVisible(running)

    def _run_py_to_doc(self):
        src      = self.inp_src.text().strip()
        out_dir  = self.inp_out_dir.text().strip()
        out_name = self.inp_out_name.text().strip()

        if not src or not Path(src).is_file():
            QMessageBox.warning(self, "Missing File",
                                "Please select a valid Python source file.")
            return
        if not out_dir:
            QMessageBox.warning(self, "Missing Output Folder",
                                "Please select an output folder.")
            return
        if not out_name:
            out_name = "README.md"
            self.inp_out_name.setText(out_name)

        if self._endpoints is None or not self._endpoints.is_loaded:
            QMessageBox.warning(
                self, "No Engine",
                "No LLM engine is loaded. Load a model (or connect an API "
                "model) from the main tabs first.")
            return

        # Make sure the local server is up before we start streaming requests.
        if self._endpoints.llama_engine and self._endpoints.api_engine \
                and not self._endpoints.api_engine.is_loaded:
            self._endpoints.ensure_server(log_cb=lambda m: self._log(m))

        self.log_te.clear()
        self.preview_te.clear()
        self._set_running(True)

        self._worker = PyToDocWorker(
            file_path          = src,
            out_path           = out_dir,
            out_name           = out_name,
            include_globals    = self.chk_globals.isChecked(),
            reset_per_function = self.chk_reset_fn.isChecked(),
            reset_per_class    = self.chk_reset_cls.isChecked(),
            prompt_overview    = self.inp_prompt_overview.toPlainText().strip(),
            prompt_class       = self.inp_prompt_class.toPlainText().strip(),
            prompt_function    = self.inp_prompt_function.toPlainText().strip(),
            endpoints          = self._endpoints,
        )
        self._worker.log_msg.connect(self._log)
        self._worker.chunk.connect(self._on_chunk)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self):
        self._set_running(False)
        out = Path(self.inp_out_dir.text()) / self.inp_out_name.text()
        self._log(f"✅  Done  →  {out}")
        QMessageBox.information(
            self, "Documentation Generated",
            f"README saved to:\n{out}"
        )
        self._worker = None

    def _on_error(self, msg: str):
        self._set_running(False)
        self._log(f"❌  Error: {msg}")
        QMessageBox.critical(self, "Pipeline Error", msg)
        self._worker = None
