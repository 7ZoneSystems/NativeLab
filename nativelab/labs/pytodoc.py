"""
Lab feature: py-to-doc.

Generates structured README-style documentation for a Python file by feeding
its overview, classes, and functions through the active LLM. The feature talks
to the rest of the app exclusively through `LabEndpoints`, so it works
identically against a local llama-server, llama-cli, or a remote API model.
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from nativelab.imports.import_global import (
    QThread, pyqtSignal,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTextEdit, QFrame, QScrollArea, QCheckBox, QFileDialog, QMessageBox,
    QFont, Qt,
)
from nativelab.UI.icons import set_button_icon, set_label_icon, set_status_label

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

PYTODOC_TEMP_STATE = Path("./localllm/temp")


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
    paused  = pyqtSignal(str)
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
        file_list:           Optional[List[str]] = None,
        project_root:        Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.file_list    = file_list or []
        self.project_root = project_root
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
        self._pause             = False
        self._history: List[dict] = []
        self._state: Optional[dict[str, Any]] = None
        self._resuming = False
        self._state_path = PYTODOC_TEMP_STATE

    def abort(self):
        self._abort = True

    def pause(self):
        self._pause = True

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

    def _project_enabled(self) -> bool:
        return bool(self.project_root and self.file_list)

    def _settings_fingerprint(self) -> str:
        payload = {
            "include_globals": self.include_globals,
            "reset_per_function": self.reset_per_function,
            "reset_per_class": self.reset_per_class,
            "prompt_overview": self.prompt_overview,
            "prompt_class": self.prompt_class,
            "prompt_function": self.prompt_function,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _project_key(self) -> str:
        raw = json.dumps({
            "project_root": str(Path(self.project_root or "").resolve()),
            "out_path": str(Path(self.out_path).resolve()),
            "settings": self._settings_fingerprint(),
        }, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_project_state(self) -> Optional[dict[str, Any]]:
        if not self._project_enabled() or not self._state_path.exists():
            return None
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._log(f"Recovery checkpoint ignored: could not read temp ({exc})")
            return None
        if state.get("feature") != "py_to_doc" or state.get("project_key") != self._project_key():
            self._log("Recovery checkpoint ignored: project, output folder, or settings changed")
            return None
        return state

    def _fresh_project_state(self) -> dict[str, Any]:
        return {
            "feature": "py_to_doc",
            "version": 1,
            "status": "running",
            "project_key": self._project_key(),
            "project_root": str(Path(self.project_root or "").resolve()),
            "out_path": str(Path(self.out_path).resolve()),
            "settings_fingerprint": self._settings_fingerprint(),
            "file_list": [str(Path(p).resolve()) for p in self.file_list],
            "current_file": "",
            "completed_files": [],
            "files": {},
            "history": [],
            "updated_at": "",
        }

    def _save_project_state(self, status: Optional[str] = None):
        if not self._project_enabled() or self._state is None:
            return
        if status:
            self._state["status"] = status
        self._state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._state["history"] = list(self._history)
        self._state["file_list"] = [str(Path(p).resolve()) for p in self.file_list]
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_name(self._state_path.name + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._state_path)

    def _prepare_project_state(self):
        if not self._project_enabled():
            return
        state = self._load_project_state()
        if state:
            self._state = state
            self._resuming = True
            self._history = list(state.get("history", []))
            self._log(f"Resume checkpoint found -> {self._state_path}")
            self._verify_project_state()
        else:
            self._state = self._fresh_project_state()
            self._save_project_state("running")
            self._log(f"Project checkpoint created -> {self._state_path}")

    def _verify_project_state(self):
        if self._state is None:
            return
        completed = set(self._state.get("completed_files", []))
        for file_path, info in list(self._state.get("files", {}).items()):
            out_file = Path(info.get("out_file", ""))
            if file_path in completed and not out_file.exists():
                completed.discard(file_path)
                info["completed"] = False
                info["steps"] = []
                info["offset"] = 0
                self._log(f"Checkpoint repaired: missing output will regenerate {Path(file_path).name}")
            elif not out_file.exists():
                info["steps"] = []
                info["offset"] = 0
                info["completed"] = False
        self._state["completed_files"] = sorted(completed)
        self._save_project_state("running")

    def _file_state(self, file_path: str, out_file: Path) -> dict[str, Any]:
        source_sig = self._source_signature(file_path)
        if self._state is None:
            return {
                "steps": [],
                "offset": 0,
                "completed": False,
                "out_file": str(out_file),
                "source": source_sig,
            }
        key = str(Path(file_path).resolve())
        files = self._state.setdefault("files", {})
        info = files.setdefault(key, {
            "steps": [],
            "offset": 0,
            "completed": False,
            "out_file": str(out_file),
            "source": source_sig,
        })
        info["out_file"] = str(out_file)
        if info.get("source") != source_sig:
            info["steps"] = []
            info["offset"] = 0
            info["completed"] = False
            info["source"] = source_sig
            completed = set(self._state.get("completed_files", []))
            completed.discard(key)
            self._state["completed_files"] = sorted(completed)
            self._log(f"Source changed since checkpoint; regenerating {Path(file_path).name}")
            self._save_project_state("running")
        return info

    @staticmethod
    def _source_signature(file_path: str) -> dict[str, int]:
        try:
            st = Path(file_path).stat()
            return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
        except Exception:
            return {"size": 0, "mtime_ns": 0}

    def _step_done(self, file_path: str, step_id: str) -> bool:
        if self._state is None:
            return False
        key = str(Path(file_path).resolve())
        info = self._state.get("files", {}).get(key, {})
        return step_id in set(info.get("steps", []))

    def _mark_step(self, file_path: str, step_id: str, fh):
        if self._state is None:
            return
        key = str(Path(file_path).resolve())
        info = self._state["files"][key]
        steps = list(info.get("steps", []))
        if step_id not in steps:
            steps.append(step_id)
        info["steps"] = steps
        info["offset"] = fh.tell()
        info["completed"] = False
        self._state["current_file"] = key
        self._save_project_state("running")

    def _mark_file_complete(self, file_path: str, fh):
        if self._state is None:
            return
        key = str(Path(file_path).resolve())
        info = self._state["files"][key]
        info["completed"] = True
        info["offset"] = fh.tell()
        completed = set(self._state.get("completed_files", []))
        completed.add(key)
        self._state["completed_files"] = sorted(completed)
        self._state["current_file"] = ""
        self._save_project_state("running")

    def _should_continue(self) -> bool:
        if self._abort:
            self._save_project_state("aborted")
            return False
        if self._pause:
            self._save_project_state("paused")
            self.paused.emit(str(self._state_path))
            return False
        return True

    def _run_step(self, fh, file_path: str, step_id: str, build_text) -> bool:
        if self._step_done(file_path, step_id):
            return True
        if not self._should_continue():
            return False
        history_len = len(self._history)
        text = build_text()
        if self._abort:
            self._history = self._history[:history_len]
            self._save_project_state("aborted")
            return False
        self._write(fh, text)
        self._mark_step(file_path, step_id, fh)
        return self._should_continue()

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
        files = self.file_list if self.file_list else [self.file_path]
        self._prepare_project_state()
        for file_path in files:
            if not self._should_continue():
                return
            self._process_one(file_path)
            if self._pause:
                return
        self._save_project_state("complete")
        self.done.emit()

    def _process_one(self, file_path: str):
        if not self._should_continue():
            return

        self._log(f"Parsing: {Path(file_path).name}")
        parsed = parse_python_file(file_path)
        src    = parsed["source"]
        lines  = parsed["lines"]
        fname  = Path(file_path).stem

        out_dir = Path(self.out_path)
        if self.project_root:
            rel      = Path(file_path).relative_to(self.project_root)
            out_file = out_dir / rel.with_suffix(".md")
            out_file.parent.mkdir(parents=True, exist_ok=True)
        elif self.file_list:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{fname}.md"
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / self.out_name

        file_state = self._file_state(file_path, out_file) if self._project_enabled() else None
        if file_state and file_state.get("completed") and out_file.exists():
            self._log(f"Already complete, verified from temp -> {Path(file_path).name}")
            return
        if file_state and not out_file.exists():
            file_state["steps"] = []
            file_state["offset"] = 0
            file_state["completed"] = False
            self._save_project_state("running")

        mode = "r+" if file_state and out_file.exists() else "w"
        fh = out_file.open(mode, encoding="utf-8")
        try:
            if file_state:
                offset = int(file_state.get("offset", 0) or 0)
                try:
                    fh.seek(0, 2)
                    end = fh.tell()
                    if 0 <= offset <= end:
                        fh.seek(offset)
                        fh.truncate()
                    else:
                        file_state["steps"] = []
                        file_state["offset"] = 0
                        file_state["completed"] = False
                        fh.seek(0)
                        fh.truncate()
                        self._save_project_state("running")
                except Exception:
                    file_state["steps"] = []
                    file_state["offset"] = 0
                    file_state["completed"] = False
                    fh.seek(0)
                    fh.truncate()
                    self._save_project_state("running")
            if self._run_pipeline(fh, file_path, fname, src, lines, parsed):
                self._mark_file_complete(file_path, fh)
        finally:
            fh.close()

        if not self._pause and not self._abort:
            self._log(f"README saved -> {out_file}")

    def _run_pipeline(self, fh, file_path, fname, src, lines, parsed) -> bool:
        if not self._run_step(
            fh, file_path, "header",
            lambda: f"## Doc for {fname}\n---\n\n"
        ):
            return False
        self._log("README initialized")

        if not self._should_continue(): return False
        self._log("Generating overview…")
        if not self._run_step(
            fh, file_path, "overview",
            lambda: f"{self._call_llm(self.prompt_overview, src)}\n\n---\n\n"
        ):
            return False
        self._log("Overview generated")

        for cls in parsed["classes"]:
            if not self._should_continue(): return False

            class_code = _get_source(cls["node"], lines)
            self._log(f"Processing class: {cls['name']}")

            class_step = f"class:{cls['name']}:{getattr(cls['node'], 'lineno', 0)}"
            if self.reset_per_class and not self._step_done(file_path, class_step):
                self._reset_context()
            if not self._run_step(
                fh, file_path, class_step,
                lambda cls=cls, class_code=class_code: (
                    f"### {cls['name']}\n"
                    f"{self._call_llm(self.prompt_class, class_code)}\n\n"
                )
            ):
                return False

            functions_step = f"class-functions:{cls['name']}:{getattr(cls['node'], 'lineno', 0)}"
            if not self._run_step(fh, file_path, functions_step, lambda: "### Functions\n\n"):
                return False
            for method in cls["methods"]:
                if not self._should_continue(): return False

                fn_code = _get_source(method["node"], lines)
                method_step = (
                    f"method:{cls['name']}.{method['name']}:"
                    f"{getattr(method['node'], 'lineno', 0)}"
                )
                if self.reset_per_function and not self._step_done(file_path, method_step):
                    self._reset_context()
                if not self._run_step(
                    fh, file_path, method_step,
                    lambda method=method, fn_code=fn_code: (
                        f"#### `{method['name']}`\n\n"
                        f"{self._call_llm(self.prompt_function, fn_code).strip()}\n\n"
                        f"---\n\n"
                    )
                ):
                    return False
                self._log(f"Function processed: {method['name']}")

            self._log(f"Class processed: {cls['name']}")
            class_end_step = f"class-end:{cls['name']}:{getattr(cls['node'], 'lineno', 0)}"
            if not self._run_step(fh, file_path, class_end_step, lambda: "\n"):
                return False

        if self.include_globals and parsed["functions"]:
            if not self._should_continue(): return False
            if not self._run_step(
                fh, file_path, "global-functions-header",
                lambda: "---\n\n## Global Functions\n\n### Functions\n\n"
            ):
                return False
            for fn in parsed["functions"]:
                if not self._should_continue(): return False

                fn_code = _get_source(fn["node"], lines)
                fn_step = f"global:{fn['name']}:{getattr(fn['node'], 'lineno', 0)}"
                if self.reset_per_function and not self._step_done(file_path, fn_step):
                    self._reset_context()
                if not self._run_step(
                    fh, file_path, fn_step,
                    lambda fn=fn, fn_code=fn_code: (
                        f"#### `{fn['name']}`\n\n"
                        f"{self._call_llm(self.prompt_function, fn_code).strip()}\n\n"
                        f"---\n\n"
                    )
                ):
                    return False
                self._log(f"Function processed: {fn['name']}")
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  Panel
# ─────────────────────────────────────────────────────────────────────────────

class PyToDocPanel(QWidget):
    """UI panel for the py-to-doc lab."""

    LAB_NAME = "py-to-doc"
    LAB_ICON = "file-text"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker:    Optional[PyToDocWorker] = None
        self._endpoints: Optional[LabEndpoints]  = None
        self._mode = "single"
        self._queue_files: list[str] = []
        self._build()

    # ── endpoint wiring ──────────────────────────────────────────────────────
    def set_endpoints(self, endpoints: LabEndpoints):
        self._endpoints = endpoints
        endpoints.status_changed.connect(self._on_status_changed)
        self._on_status_changed(endpoints.status_text)

    def _on_status_changed(self, status: str):
        if hasattr(self, "lbl_engine"):
            state = "idle" if "No Engine" in status or "Not Loaded" in status else "ok"
            set_status_label(self.lbl_engine, f"Active engine: {status}", state)

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
        hdr = QLabel(self.LAB_NAME)
        set_label_icon(hdr, "file-text", self.LAB_NAME, 18)
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
        self.lbl_engine = QLabel("")
        set_status_label(self.lbl_engine, "Active engine: No Engine", "idle")
        self.lbl_engine.setObjectName("txt2_small")
        self.lbl_engine.setStyleSheet("margin-bottom:10px;")
        root.addWidget(self.lbl_engine)

        root.addWidget(self._section_label("MODE"))
        mode_card = self._card()
        mc = QHBoxLayout(mode_card)
        mc.setContentsMargins(16, 10, 16, 10)
        mc.setSpacing(6)
        self.btn_mode_single  = QPushButton("Single File")
        self.btn_mode_queue   = QPushButton("Queue")
        self.btn_mode_project = QPushButton("Project")
        set_button_icon(self.btn_mode_single, "file-text", "Single File")
        set_button_icon(self.btn_mode_queue, "files", "Queue")
        set_button_icon(self.btn_mode_project, "folder", "Project")
        for _b in (self.btn_mode_single, self.btn_mode_queue, self.btn_mode_project):
            _b.setCheckable(True)
            _b.setFixedHeight(30)
            mc.addWidget(_b)
        self.btn_mode_single.setChecked(True)
        self.btn_mode_single.clicked.connect(lambda: self._set_mode("single"))
        self.btn_mode_queue.clicked.connect(lambda: self._set_mode("queue"))
        self.btn_mode_project.clicked.connect(lambda: self._set_mode("project"))
        root.addWidget(mode_card)
        root.addSpacing(8)

        # File settings
        root.addWidget(self._section_label("FILE SETTINGS"))
        file_card = self._card()
        fc = QVBoxLayout(file_card)
        fc.setContentsMargins(16, 14, 16, 14)
        fc.setSpacing(10)

        # ── single-file row ───────────────────────────────────────────────
        self._wgt_single_row = QWidget()
        _sl = QHBoxLayout(self._wgt_single_row)
        _sl.setContentsMargins(0, 0, 0, 0); _sl.setSpacing(8)
        _lbl_s = QLabel("Python file:"); _lbl_s.setObjectName("txt2"); _lbl_s.setFixedWidth(110)
        self.inp_src = QLineEdit()
        self.inp_src.setPlaceholderText("Select Python file to document…")
        self.inp_src.setReadOnly(True); self.inp_src.setFixedHeight(30)
        btn_browse_src = QPushButton("Browse…")
        btn_browse_src.setFixedHeight(30); btn_browse_src.setFixedWidth(80)
        btn_browse_src.clicked.connect(self._browse_src)
        _sl.addWidget(_lbl_s); _sl.addWidget(self.inp_src, 1); _sl.addWidget(btn_browse_src)
        fc.addWidget(self._wgt_single_row)

        # ── queue rows ────────────────────────────────────────────────────
        self._wgt_queue_row = QWidget()
        _qlayout = QVBoxLayout(self._wgt_queue_row)
        _qlayout.setContentsMargins(0, 0, 0, 0); _qlayout.setSpacing(4)
        _qh = QHBoxLayout(); _qh.setSpacing(8)
        _lbl_q = QLabel("Queued files:"); _lbl_q.setObjectName("txt2"); _lbl_q.setFixedWidth(110)
        self.lst_queue = QTextEdit()
        self.lst_queue.setReadOnly(True); self.lst_queue.setFixedHeight(72)
        self.lst_queue.setPlaceholderText("No files queued yet…")
        _qh.addWidget(_lbl_q); _qh.addWidget(self.lst_queue, 1)
        _qlayout.addLayout(_qh)
        _qb = QHBoxLayout(); _qb.addSpacing(118)
        btn_q_add = QPushButton("Add Files"); set_button_icon(btn_q_add, "plus", "Add Files"); btn_q_add.setFixedHeight(26)
        btn_q_add.clicked.connect(self._browse_src_queue)
        btn_q_clr = QPushButton("Clear"); set_button_icon(btn_q_clr, "delete", "Clear"); btn_q_clr.setFixedHeight(26)
        btn_q_clr.clicked.connect(self._clear_queue)
        _qb.addWidget(btn_q_add); _qb.addWidget(btn_q_clr); _qb.addStretch()
        _qlayout.addLayout(_qb)
        fc.addWidget(self._wgt_queue_row)
        self._wgt_queue_row.setVisible(False)

        # ── project source row ────────────────────────────────────────────
        self._wgt_project_row = QWidget()
        _pl = QHBoxLayout(self._wgt_project_row)
        _pl.setContentsMargins(0, 0, 0, 0); _pl.setSpacing(8)
        _lbl_p = QLabel("Project root:"); _lbl_p.setObjectName("txt2"); _lbl_p.setFixedWidth(110)
        self.inp_project_src = QLineEdit()
        self.inp_project_src.setPlaceholderText("Select project root directory…")
        self.inp_project_src.setReadOnly(True); self.inp_project_src.setFixedHeight(30)
        btn_browse_proj = QPushButton("Browse…")
        btn_browse_proj.setFixedHeight(30); btn_browse_proj.setFixedWidth(80)
        btn_browse_proj.clicked.connect(self._browse_src_project)
        _pl.addWidget(_lbl_p); _pl.addWidget(self.inp_project_src, 1); _pl.addWidget(btn_browse_proj)
        fc.addWidget(self._wgt_project_row)
        self._wgt_project_row.setVisible(False)

        # ── output folder (always visible) ────────────────────────────────
        self.inp_out_dir = QLineEdit()
        self.inp_out_dir.setPlaceholderText("Select output folder…")
        self.inp_out_dir.setReadOnly(True); self.inp_out_dir.setFixedHeight(30)
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.setFixedHeight(30); btn_browse_out.setFixedWidth(80)
        btn_browse_out.clicked.connect(self._browse_out)
        fc.addLayout(self._field_row("Output folder:", self.inp_out_dir, btn_browse_out))

        # ── output filename (single mode only) ────────────────────────────
        self._wgt_outname_row = QWidget()
        _ol = QHBoxLayout(self._wgt_outname_row)
        _ol.setContentsMargins(0, 0, 0, 0); _ol.setSpacing(8)
        _lbl_o = QLabel("Output filename:"); _lbl_o.setObjectName("txt2"); _lbl_o.setFixedWidth(110)
        self.inp_out_name = QLineEdit("README.md"); self.inp_out_name.setFixedHeight(30)
        _ol.addWidget(_lbl_o); _ol.addWidget(self.inp_out_name, 1)
        fc.addWidget(self._wgt_outname_row)

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
        self.btn_generate = QPushButton("Generate Documentation")
        set_button_icon(self.btn_generate, "settings", "Generate Documentation")
        self.btn_generate.setObjectName("labs_generate_btn")
        self.btn_generate.setMinimumHeight(38)
        self.btn_generate.clicked.connect(self._run_py_to_doc)
        self.btn_abort = QPushButton("Abort")
        set_button_icon(self.btn_abort, "stop-circle", "Abort")
        self.btn_abort.setObjectName("btn_stop")
        self.btn_abort.setFixedHeight(38)
        self.btn_abort.setVisible(False)
        self.btn_abort.clicked.connect(self._abort)
        self.btn_pause = QPushButton("Pause")
        set_button_icon(self.btn_pause, "circle-pause", "Pause")
        self.btn_pause.setFixedHeight(38)
        self.btn_pause.setVisible(False)
        self.btn_pause.clicked.connect(self._pause_project)
        btn_row.addWidget(self.btn_generate, 1)
        btn_row.addWidget(self.btn_pause)
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

        btn_copy = QPushButton("Copy to Clipboard")
        set_button_icon(btn_copy, "copy", "Copy to Clipboard")
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

    def _set_mode(self, mode: str):
        self._mode = mode
        self.btn_mode_single.setChecked(mode == "single")
        self.btn_mode_queue.setChecked(mode == "queue")
        self.btn_mode_project.setChecked(mode == "project")
        self._wgt_single_row.setVisible(mode == "single")
        self._wgt_queue_row.setVisible(mode == "queue")
        self._wgt_project_row.setVisible(mode == "project")
        self._wgt_outname_row.setVisible(mode == "single")

    def _browse_src_queue(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Python Files",
            str(Path(self._queue_files[-1]).parent if self._queue_files else Path.home()),
            "Python Files (*.py);;All Files (*)"
        )
        for p in paths:
            if p not in self._queue_files:
                self._queue_files.append(p)
        self._update_queue_display()
        if self._queue_files and not self.inp_out_dir.text():
            self.inp_out_dir.setText(str(Path(self._queue_files[0]).parent))

    def _clear_queue(self):
        self._queue_files.clear()
        self._update_queue_display()

    def _update_queue_display(self):
        self.lst_queue.setPlainText("\n".join(self._queue_files))

    def _browse_src_project(self):
        p = QFileDialog.getExistingDirectory(
            self, "Select Project Root",
            self.inp_project_src.text() or str(Path.home())
        )
        if p:
            self.inp_project_src.setText(p)
            if not self.inp_out_dir.text():
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

    def _pause_project(self):
        if self._worker:
            self._worker.pause()
            self._log(f"[pause requested] checkpoint will save to {PYTODOC_TEMP_STATE}")

    def _set_running(self, running: bool):
        self.btn_generate.setEnabled(not running)
        self.btn_abort.setVisible(running)
        self.btn_pause.setVisible(running and self._mode == "project")

    def _run_py_to_doc(self):
        mode     = self._mode
        out_dir  = self.inp_out_dir.text().strip()
        out_name = self.inp_out_name.text().strip() or "README.md"
        file_list:    list[str]    = []
        project_root: Optional[str] = None

        if mode == "single":
            src = self.inp_src.text().strip()
            if not src or not Path(src).is_file():
                QMessageBox.warning(self, "Missing File",
                                    "Please select a valid Python source file.")
                return

        elif mode == "queue":
            if not self._queue_files:
                QMessageBox.warning(self, "Empty Queue",
                                    "Add at least one Python file to the queue.")
                return
            src       = self._queue_files[0]
            file_list = list(self._queue_files)

        else:  # project
            proj = self.inp_project_src.text().strip()
            if not proj or not Path(proj).is_dir():
                QMessageBox.warning(self, "Missing Project Root",
                                    "Please select a valid project root directory.")
                return
            file_list = [str(p) for p in sorted(Path(proj).rglob("*.py"))]
            if not file_list:
                QMessageBox.warning(self, "No Python Files",
                                    "No .py files found in the selected directory.")
                return
            src          = file_list[0]
            project_root = proj

        if not out_dir:
            QMessageBox.warning(self, "Missing Output Folder",
                                "Please select an output folder.")
            return

        if self._endpoints is None or not self._endpoints.is_loaded:
            QMessageBox.warning(
                self, "No Engine",
                "No LLM engine is loaded. Load a model (or connect an API "
                "model) from the main tabs first.")
            return

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
            file_list          = file_list,
            project_root       = project_root,
        )
        self._worker.log_msg.connect(self._log)
        self._worker.chunk.connect(self._on_chunk)
        self._worker.done.connect(self._on_done)
        self._worker.paused.connect(self._on_paused)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_paused(self, state_path: str):
        self._set_running(False)
        self._log(f"Paused. Resume by selecting the same project/output and clicking Generate. Temp: {state_path}")
        self._worker = None

    def _on_done(self):
        self._set_running(False)
        if self._mode == "single":
            out = Path(self.inp_out_dir.text()) / self.inp_out_name.text()
            self._log(f"Done  ->  {out}")
            QMessageBox.information(self, "Documentation Generated",
                                    f"README saved to:\n{out}")
        else:
            out_dir = self.inp_out_dir.text()
            self._log(f"Done  ->  {out_dir}")
            QMessageBox.information(self, "Documentation Generated",
                                    f"All docs saved to:\n{out_dir}")
        self._worker = None

    def _on_error(self, msg: str):
        self._set_running(False)
        self._log(f"Error: {msg}")
        QMessageBox.critical(self, "Pipeline Error", msg)
        self._worker = None
