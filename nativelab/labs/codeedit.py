"""
Lab feature: structured code edit.

Keeps an editable temp workspace in localllm, asks the active model for
structured edit operations, then applies those operations locally so iteration
does not require rewriting whole files.
"""
from __future__ import annotations

import ast
import difflib
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from nativelab.imports.import_global import (
    QThread, pyqtSignal,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTextEdit, QFrame, QScrollArea, QFileDialog, QMessageBox,
    QFont, Qt, QSplitter, QSizePolicy, QDialog, QColor,
)
from nativelab.UI.icons import set_button_icon, set_label_icon, set_status_label
from nativelab.UI.buildUI import prepare_adaptive_window
from nativelab.UI.toggle import ToggleSwitch

from .endpoints import LabEndpoints


EDIT_TEMP_FILE = Path("./localllm/temp_code_edit.json")
EDIT_TEMP_CODE_FILE = Path("./localllm/temp_code_edit_file")
CODE_EXTS = "Code Files (*.py *.c *.h *.cpp *.hpp *.cc *.js *.ts *.tsx *.java *.rs *.go *.cs *.php *.rb *.swift *.kt *.m);;All Files (*)"


EDIT_SYSTEM_PROMPT = """
You are NativeLab's structured edit skill.
Return only JSON. Do not wrap it in markdown.

Schema:
{
  "summary": "short explanation",
  "operations": [
    {"op": "replace_function", "name": "function_or_method_name", "code": "complete replacement function code"},
    {"op": "replace_span", "start_line": 10, "end_line": 18, "code": "replacement lines"},
    {"op": "insert_after", "line": 25, "code": "new lines"},
    {"op": "append", "code": "new lines"},
    {"op": "full_replace", "code": "complete file content"}
  ]
}

Rules:
- Prefer replace_function for function-level changes.
- Prefer replace_span only for small non-function edits such as imports/constants.
- Use full_replace only for initial generation or when explicitly requested.
- Preserve unrelated code, comments, imports, names, public APIs, and formatting.
- If the request is ambiguous, make the smallest reasonable edit.
""".strip()


def detect_language(path: str, code: str = "") -> str:
    ext = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".c": "c",
        ".h": "c/c++ header",
        ".cpp": "c++",
        ".hpp": "c++ header",
        ".cc": "c++",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript/react",
        ".java": "java",
        ".rs": "rust",
        ".go": "go",
        ".cs": "c#",
        ".php": "php",
        ".rb": "ruby",
        ".swift": "swift",
        ".kt": "kotlin",
    }.get(ext, "code" if code else "unknown")


def structure_for_code(code: str, path: str = "") -> dict[str, Any]:
    lang = detect_language(path, code)
    if lang == "python":
        return _python_structure(code, path)
    return _generic_structure(code, path, lang)


def _python_structure(code: str, path: str) -> dict[str, Any]:
    rows: dict[str, Any] = {"language": "python", "path": path, "functions": [], "classes": []}
    try:
        tree = ast.parse(code)
    except Exception as exc:
        rows["parse_error"] = str(exc)
        return rows
    lines = code.splitlines()

    def fn_info(node, owner: str = "") -> dict[str, Any]:
        args = []
        for arg in list(getattr(node.args, "posonlyargs", [])) + list(node.args.args) + list(node.args.kwonlyargs):
            args.append(arg.arg)
        returns = []
        variables = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                try:
                    returns.append(ast.unparse(child.value))
                except Exception:
                    returns.append("<return>")
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        variables.add(target.id)
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                variables.add(child.target.id)
        return {
            "name": f"{owner}.{node.name}" if owner else node.name,
            "short_name": node.name,
            "start_line": int(getattr(node, "lineno", 1)),
            "end_line": int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
            "args": args,
            "returns": returns,
            "variables": sorted(variables),
            "preview": "\n".join(lines[getattr(node, "lineno", 1) - 1:getattr(node, "lineno", 1) + 3]),
        }

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rows["functions"].append(fn_info(node))
        elif isinstance(node, ast.ClassDef):
            methods = [
                fn_info(child, node.name)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            rows["classes"].append({
                "name": node.name,
                "start_line": int(getattr(node, "lineno", 1)),
                "end_line": int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
                "methods": methods,
            })
    return rows


def _generic_structure(code: str, path: str, lang: str) -> dict[str, Any]:
    rows: dict[str, Any] = {"language": lang, "path": path, "functions": []}
    lines = code.splitlines()
    signature = re.compile(
        r"^\s*(?:[\w:<>\*\&\[\],]+\s+)+(?P<name>[A-Za-z_]\w*)\s*\([^;]*\)\s*(?:\{|$)"
    )
    for i, line in enumerate(lines, start=1):
        m = signature.match(line)
        if not m:
            continue
        end = _brace_end(lines, i)
        rows["functions"].append({
            "name": m.group("name"),
            "short_name": m.group("name"),
            "start_line": i,
            "end_line": end,
            "args": [],
            "returns": [],
            "variables": [],
            "preview": "\n".join(lines[i - 1:min(len(lines), i + 3)]),
        })
    return rows


def _brace_end(lines: list[str], start_line: int) -> int:
    depth = 0
    seen = False
    for idx in range(start_line, len(lines) + 1):
        line = lines[idx - 1]
        depth += line.count("{")
        if "{" in line:
            seen = True
        depth -= line.count("}")
        if seen and depth <= 0:
            return idx
    return start_line


def extract_json_object(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?", "", clean).strip()
        clean = re.sub(r"```$", "", clean).strip()
    try:
        return json.loads(clean)
    except Exception:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            return json.loads(clean[start:end + 1])
    raise ValueError("Model did not return valid JSON edit operations")


def apply_operations(code: str, operations: list[dict[str, Any]], structure: dict[str, Any]) -> str:
    updated = code
    for op in operations:
        kind = op.get("op", "")
        if kind == "full_replace":
            updated = str(op.get("code", ""))
        elif kind == "replace_function":
            updated = _replace_function(updated, str(op.get("name", "")), str(op.get("code", "")), structure)
        elif kind == "replace_span":
            updated = _replace_span(updated, int(op.get("start_line", 1)), int(op.get("end_line", 1)), str(op.get("code", "")))
        elif kind == "insert_after":
            updated = _insert_after(updated, int(op.get("line", 0)), str(op.get("code", "")))
        elif kind == "append":
            extra = str(op.get("code", ""))
            updated = updated.rstrip() + "\n\n" + extra.strip() + "\n"
    return updated


def _replace_function(code: str, name: str, replacement: str, structure: dict[str, Any]) -> str:
    candidates = list(structure.get("functions", []))
    for cls in structure.get("classes", []):
        candidates.extend(cls.get("methods", []))
    short = name.split(".")[-1]
    for fn in candidates:
        if fn.get("name") == name or fn.get("short_name") == short:
            return _replace_span(code, int(fn["start_line"]), int(fn["end_line"]), replacement)
    raise ValueError(f"Function not found for structured edit: {name}")


def _replace_span(code: str, start_line: int, end_line: int, replacement: str) -> str:
    lines = code.splitlines()
    start = max(1, start_line) - 1
    end = max(start + 1, end_line)
    new_lines = replacement.rstrip("\n").splitlines()
    return "\n".join(lines[:start] + new_lines + lines[end:]) + "\n"


def _insert_after(code: str, line: int, new_code: str) -> str:
    lines = code.splitlines()
    idx = max(0, min(line, len(lines)))
    new_lines = new_code.rstrip("\n").splitlines()
    return "\n".join(lines[:idx] + new_lines + lines[idx:]) + "\n"


def render_diff_html(before: str, after: str) -> tuple[str, int, int]:
    diff = list(difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile="before",
        tofile="after",
        lineterm="",
    ))
    added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    deleted = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    if not diff:
        return (
            "<div style='font-family:Consolas,monospace;'>No line changes.</div>",
            0,
            0,
        )

    rows = [
        "<div style='font-family:Consolas,monospace;font-size:12px;white-space:pre-wrap;'>",
    ]
    for line in diff:
        safe = html.escape(line)
        if line.startswith("+") and not line.startswith("+++"):
            rows.append(f"<span style='color:#16a34a;font-weight:600;'>{safe}</span>")
        elif line.startswith("-") and not line.startswith("---"):
            rows.append(f"<span style='color:#dc2626;font-weight:600;'>{safe}</span>")
        elif line.startswith("@@"):
            rows.append(f"<span style='color:#2563eb;font-weight:600;'>{safe}</span>")
        elif line.startswith(("---", "+++")):
            rows.append(f"<span style='color:#6b7280;font-weight:600;'>{safe}</span>")
        else:
            rows.append(f"<span>{safe}</span>")
    rows.append("</div>")
    return "<br>".join(rows), added, deleted


class CodeEditWorker(QThread):
    log_msg = pyqtSignal(str)
    model_response = pyqtSignal(str)
    diff_ready = pyqtSignal(str, int, int)
    done = pyqtSignal(str, str, str, str, int, int)
    error = pyqtSignal(str)

    def __init__(self, endpoints: LabEndpoints, code: str, path: str, instruction: str, generate: bool = False):
        super().__init__()
        self.endpoints = endpoints
        self.code = code
        self.path = path
        self.instruction = instruction
        self.generate = generate

    def run(self):
        try:
            structure = structure_for_code(self.code, self.path)
            payload = {
                "path": self.path,
                "language": structure.get("language", "code"),
                "structure": structure,
                "current_code": self.code,
                "request": self.instruction,
                "mode": "generate" if self.generate else "edit",
            }
            self.log_msg.emit("Asking model for structured edit operations...")
            raw = self.endpoints.call_llm(
                system_prompt=EDIT_SYSTEM_PROMPT,
                prompt=json.dumps(payload, indent=2, ensure_ascii=False),
                n_predict=1800,
                temperature=0.15,
            )
            self.model_response.emit(raw)
            data = extract_json_object(raw)
            operations = data.get("operations", [])
            if not isinstance(operations, list) or not operations:
                raise ValueError("Model returned no edit operations")
            if self.generate and not self.code.strip():
                if operations[0].get("op") != "full_replace":
                    operations = [{"op": "full_replace", "code": data.get("code", "") or operations[0].get("code", "")}]
            updated = apply_operations(self.code, operations, structure)
            diff_view, added, deleted = render_diff_html(self.code, updated)
            self.diff_ready.emit(diff_view, added, deleted)
            self.done.emit(
                updated,
                data.get("summary", "Structured edit applied."),
                raw,
                diff_view,
                added,
                deleted,
            )
        except Exception as exc:
            self.error.emit(str(exc))


class CodeEditPanel(QWidget):
    LAB_NAME = "structured-edit"
    LAB_ICON = "code-2"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._endpoints: Optional[LabEndpoints] = None
        self._worker: Optional[CodeEditWorker] = None
        self._source_path = ""
        self._history: list[dict[str, Any]] = []
        self._pending_index: Optional[int] = None
        self._lint_line: Optional[int] = None
        self._build()
        self._load_temp()

    def set_endpoints(self, endpoints: LabEndpoints):
        self._endpoints = endpoints
        endpoints.status_changed.connect(self._on_status_changed)
        self._on_status_changed(endpoints.status_text)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(2)
        root.addWidget(self.main_splitter, 1)

        left_scroll = QScrollArea()
        left_scroll.setObjectName("chat_scroll")
        left_scroll.setWidgetResizable(True)
        left_body = QWidget()
        left_body.setObjectName("chat_container")
        layout = QVBoxLayout(left_body)
        layout.setContentsMargins(22, 18, 18, 22)
        layout.setSpacing(10)
        left_scroll.setWidget(left_body)
        self.main_splitter.addWidget(left_scroll)

        hdr = QLabel(self.LAB_NAME)
        set_label_icon(hdr, "code-2", self.LAB_NAME, 18)
        hdr.setObjectName("labs_panel_header")
        layout.addWidget(hdr)

        self.lbl_engine = QLabel("")
        set_status_label(self.lbl_engine, "Active engine: No Engine", "idle")
        self.lbl_engine.setObjectName("txt2_small")
        layout.addWidget(self.lbl_engine)

        file_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Optional source file. Work stays in temp until Save.")
        self.btn_open = QPushButton("Attach File")
        self.btn_new = QPushButton("New Temp")
        self.btn_save = QPushButton("Save")
        self.btn_save_as = QPushButton("Save As")
        for btn, ico in (
            (self.btn_open, "folder-open"),
            (self.btn_new, "plus"),
            (self.btn_save, "save"),
            (self.btn_save_as, "export"),
        ):
            set_button_icon(btn, ico, btn.text())
            btn.setFixedHeight(30)
        self.btn_open.clicked.connect(self._open_file)
        self.btn_new.clicked.connect(self._new_temp)
        self.btn_save.clicked.connect(self._save_to_source)
        self.btn_save_as.clicked.connect(self._save_as)
        file_row.addWidget(self.path_edit, 1)
        file_row.addWidget(self.btn_open)
        file_row.addWidget(self.btn_new)
        file_row.addWidget(self.btn_save)
        file_row.addWidget(self.btn_save_as)
        layout.addLayout(file_row)

        chat_hdr = QLabel("Edit Chat")
        chat_hdr.setStyleSheet("font-size:13px;font-weight:bold;")
        layout.addWidget(chat_hdr)

        self.history_scroll = QScrollArea()
        self.history_scroll.setObjectName("chat_scroll")
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setMinimumHeight(330)
        self.history_body = QWidget()
        self.history_body.setObjectName("chat_container")
        self.history_layout = QVBoxLayout(self.history_body)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.setSpacing(8)
        self.history_scroll.setWidget(self.history_body)
        layout.addWidget(self.history_scroll, 1)

        self.prompt_te = QTextEdit()
        self.prompt_te.setPlaceholderText("Ask for a structured edit, or describe a file to generate...")
        self.prompt_te.setFixedHeight(86)
        layout.addWidget(self.prompt_te)

        actions = QHBoxLayout()
        self.chk_generate = ToggleSwitch("Generate if temp is empty")
        self.chk_generate.setChecked(True)
        self.btn_apply = QPushButton("Apply Structured Edit")
        set_button_icon(self.btn_apply, "replace", "Apply Structured Edit")
        self.btn_apply.setObjectName("labs_generate_btn")
        self.btn_apply.setFixedHeight(36)
        self.btn_apply.clicked.connect(self._apply_edit)
        actions.addWidget(self.chk_generate)
        actions.addWidget(self.btn_apply, 1)
        layout.addLayout(actions)

        right_scroll = QScrollArea()
        right_scroll.setObjectName("chat_scroll")
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_panel = QWidget()
        right_panel.setObjectName("chat_container")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(18, 18, 22, 22)
        right_layout.setSpacing(10)
        right_scroll.setWidget(right_panel)
        self.main_splitter.addWidget(right_scroll)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([760, 520])

        preview_hdr = QLabel("Live File")
        set_label_icon(preview_hdr, "code-2", "Live File", 18)
        preview_hdr.setObjectName("labs_panel_header")
        right_layout.addWidget(preview_hdr)

        self.lint_lbl = QLabel("Lint: waiting for code")
        self.lint_lbl.setObjectName("txt2_small")
        right_layout.addWidget(self.lint_lbl)

        self.code_te = QTextEdit()
        self.code_te.setObjectName("labs_preview_te")
        self.code_te.setFont(QFont("Consolas", 10))
        self.code_te.setMinimumHeight(420)
        self.code_te.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.code_te.textChanged.connect(self._on_code_changed)
        right_layout.addWidget(self.code_te, 1)

        self.problems_te = QTextEdit()
        self.problems_te.setReadOnly(True)
        self.problems_te.setObjectName("log_te")
        self.problems_te.setFixedHeight(82)
        self.problems_te.setFont(QFont("Consolas", 9))
        self.problems_te.setPlaceholderText("Python diagnostics appear here.")
        right_layout.addWidget(self.problems_te)

        self.structure_te = QTextEdit()
        self.structure_te.setReadOnly(True)
        self.structure_te.setObjectName("log_te")
        self.structure_te.setFixedHeight(118)
        self.structure_te.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self.structure_te)

        self.log_te = QTextEdit()
        self.log_te.setReadOnly(True)
        self.log_te.setObjectName("log_te")
        self.log_te.setFixedHeight(96)
        right_layout.addWidget(self.log_te)
        right_layout.addStretch(1)

        self._render_history()

    def _on_status_changed(self, status: str):
        state = "idle" if "No Engine" in status or "Not Loaded" in status else "ok"
        set_status_label(self.lbl_engine, f"Active engine: {status}", state)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Attach Code File", str(Path.home()), CODE_EXTS)
        if not path:
            return
        self._source_path = path
        self.path_edit.setText(path)
        self.code_te.setPlainText(Path(path).read_text(encoding="utf-8", errors="replace"))
        self._history.clear()
        self._pending_index = None
        self._save_temp()
        self._refresh_structure()
        self._render_history()

    def _new_temp(self):
        self._source_path = ""
        self.path_edit.clear()
        self.code_te.clear()
        self.prompt_te.clear()
        self._history.clear()
        self._pending_index = None
        self._save_temp()
        self._refresh_structure()
        self._render_history()

    def _save_to_source(self):
        if not self._source_path:
            self._save_as()
            return
        Path(self._source_path).write_text(self.code_te.toPlainText(), encoding="utf-8")
        self._log(f"Saved -> {self._source_path}")

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Code File", self._source_path or str(Path.home()), CODE_EXTS)
        if not path:
            return
        self._source_path = path
        self.path_edit.setText(path)
        Path(path).write_text(self.code_te.toPlainText(), encoding="utf-8")
        self._save_temp()
        self._log(f"Saved -> {path}")

    def _apply_edit(self):
        if self._endpoints is None or not self._endpoints.is_loaded:
            QMessageBox.warning(self, "No Engine", "Load a model or API model before editing.")
            return
        instruction = self.prompt_te.toPlainText().strip()
        if not instruction:
            QMessageBox.warning(self, "Missing Request", "Describe the edit or file to generate.")
            return
        code = self.code_te.toPlainText()
        generate = bool(self.chk_generate.isChecked() and not code.strip())
        self.btn_apply.setEnabled(False)
        self._pending_index = self._append_pending_turn(instruction, code)
        self._worker = CodeEditWorker(self._endpoints, code, self._source_path, instruction, generate)
        self._worker.log_msg.connect(self._log)
        self._worker.model_response.connect(self._on_model_response)
        self._worker.diff_ready.connect(self._on_diff_ready)
        self._worker.done.connect(self._on_edit_done)
        self._worker.error.connect(self._on_edit_error)
        self._worker.start()

    def _on_model_response(self, response: str):
        if self._pending_index is not None and 0 <= self._pending_index < len(self._history):
            self._history[self._pending_index]["raw_response"] = response
            self._render_history()
            self._save_temp()
        self._log("Model structured response received.")

    def _on_diff_ready(self, diff_html: str, added: int, deleted: int):
        if self._pending_index is not None and 0 <= self._pending_index < len(self._history):
            self._history[self._pending_index].update({
                "diff_html": diff_html,
                "added": added,
                "deleted": deleted,
            })
            self._render_history()
            self._save_temp()
        self._log(f"Structured diff ready: +{added} / -{deleted}")

    def _on_edit_done(self, updated: str, summary: str, raw: str, diff_html: str, added: int, deleted: int):
        self.code_te.blockSignals(True)
        self.code_te.setPlainText(updated)
        self.code_te.blockSignals(False)
        self._refresh_structure()
        idx = self._pending_index if self._pending_index is not None else len(self._history)
        turn = self._history[idx] if 0 <= idx < len(self._history) else {}
        turn.update({
            "pending": False,
            "summary": summary,
            "raw_response": raw,
            "diff_html": diff_html,
            "added": added,
            "deleted": deleted,
            "code_after": updated,
            "source_path": self._source_path,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        if not (0 <= idx < len(self._history)):
            self._history.append(turn)
        self._pending_index = None
        self._render_history()
        self._save_temp()
        self._log(summary)
        self.btn_apply.setEnabled(True)
        self._worker = None

    def _on_edit_error(self, msg: str):
        self._log(f"Error: {msg}")
        if self._pending_index is not None and 0 <= self._pending_index < len(self._history):
            self._history[self._pending_index]["pending"] = False
            self._history[self._pending_index]["error"] = msg
            self._render_history()
            self._save_temp()
        self._pending_index = None
        QMessageBox.critical(self, "Structured Edit Error", msg)
        self.btn_apply.setEnabled(True)
        self._worker = None

    def _on_code_changed(self):
        self._refresh_structure()
        self._refresh_lint()
        self._save_temp()

    def _refresh_structure(self):
        structure = structure_for_code(self.code_te.toPlainText(), self._source_path)
        self.structure_te.setPlainText(json.dumps(structure, indent=2, ensure_ascii=False)[:12000])
        self._refresh_lint()

    def _refresh_lint(self):
        code = self.code_te.toPlainText()
        lang = detect_language(self._source_path, code)
        if not code.strip():
            self.lint_lbl.setText("Lint: waiting for code")
            self.lint_lbl.setStyleSheet("")
            self.problems_te.clear()
            self._set_lint_highlight(None)
            return
        if lang != "python":
            self.lint_lbl.setText(f"Lint: python syntax check skipped for {lang}")
            self.lint_lbl.setStyleSheet("")
            self.problems_te.setPlainText("No Python diagnostics for this file type.")
            self._set_lint_highlight(None)
            return
        try:
            compile(code, self._source_path or "<structured-edit>", "exec")
            issues = self._basic_python_lints(code)
            if issues:
                self.lint_lbl.setText(f"Python diagnostics: {len(issues)} warning(s)")
                self.lint_lbl.setStyleSheet("color:#e8971a;font-weight:600;")
                self.problems_te.setPlainText("\n".join(issues))
                self._set_lint_highlight(self._line_from_issue(issues[0]))
                return
            self.lint_lbl.setText("Python diagnostics: clean")
            self.lint_lbl.setStyleSheet("color:#16a34a;font-weight:600;")
            self.problems_te.setPlainText("No problems detected.")
            self._set_lint_highlight(None)
        except SyntaxError as exc:
            line = int(exc.lineno or 1)
            col = int(exc.offset or 0)
            self.lint_lbl.setText(f"Python error: line {line}, col {col}: {exc.msg}")
            self.lint_lbl.setStyleSheet("color:#dc2626;font-weight:600;")
            self.problems_te.setPlainText(f"line {line}:{col}  SyntaxError: {exc.msg}")
            self._set_lint_highlight(line)

    @staticmethod
    def _basic_python_lints(code: str) -> list[str]:
        issues: list[str] = []
        lines = code.splitlines()
        for i, line in enumerate(lines, start=1):
            if len(line) > 120:
                issues.append(f"line {i}:1  Warning: line is {len(line)} characters")
            if line.rstrip("\n\r") != line.rstrip():
                issues.append(f"line {i}:{len(line.rstrip()) + 1}  Warning: trailing whitespace")
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not ast.get_docstring(node) and len(getattr(node, "body", [])) > 6:
                    issues.append(f"line {node.lineno}:1  Hint: function '{node.name}' has no docstring")
            elif isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append(f"line {node.lineno}:1  Warning: bare except")
        return issues[:60]

    @staticmethod
    def _line_from_issue(issue: str) -> Optional[int]:
        m = re.search(r"line\s+(\d+)", issue)
        return int(m.group(1)) if m else None

    def _set_lint_highlight(self, line_no: Optional[int]):
        self._lint_line = line_no
        selections = []
        if line_no is not None and line_no > 0:
            try:
                from PyQt6.QtGui import QTextCharFormat, QTextCursor, QTextFormat
                selection = QTextEdit.ExtraSelection()
                selection.format = QTextCharFormat()
                selection.format.setBackground(QColor(220, 38, 38, 46))
                selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
                cursor = self.code_te.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.Start)
                cursor.movePosition(QTextCursor.MoveOperation.Down, QTextCursor.MoveMode.MoveAnchor, line_no - 1)
                cursor.select(QTextCursor.SelectionType.LineUnderCursor)
                selection.cursor = cursor
                selections.append(selection)
            except Exception:
                selections = []
        try:
            self.code_te.setExtraSelections(selections)
        except Exception:
            pass

    def _load_temp(self):
        if not EDIT_TEMP_FILE.exists():
            return
        try:
            data = json.loads(EDIT_TEMP_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        self._source_path = data.get("source_path", "")
        self.path_edit.setText(self._source_path)
        code = data.get("code", "")
        if EDIT_TEMP_CODE_FILE.exists():
            try:
                code = EDIT_TEMP_CODE_FILE.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
        self.code_te.blockSignals(True)
        self.code_te.setPlainText(code)
        self.code_te.blockSignals(False)
        self.prompt_te.setPlainText(data.get("last_prompt", ""))
        self._history = list(data.get("history", []))
        self._pending_index = None
        self._refresh_structure()
        self._render_history()
        self._log(f"Loaded temp workspace -> {EDIT_TEMP_FILE}")

    def _save_temp(self):
        EDIT_TEMP_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "source_path": self._source_path,
            "last_prompt": self.prompt_te.toPlainText() if hasattr(self, "prompt_te") else "",
            "code": self.code_te.toPlainText() if hasattr(self, "code_te") else "",
            "temp_code_file": str(EDIT_TEMP_CODE_FILE),
            "history": self._history,
        }
        EDIT_TEMP_CODE_FILE.write_text(payload["code"], encoding="utf-8")
        EDIT_TEMP_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_pending_turn(self, prompt: str, code_before: str) -> int:
        if self._pending_index is not None and 0 <= self._pending_index < len(self._history):
            self._history = self._history[:self._pending_index]
        turn = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "prompt": prompt,
            "summary": "Thinking...",
            "raw_response": "",
            "diff_html": "",
            "added": 0,
            "deleted": 0,
            "code_before": code_before,
            "code_after": code_before,
            "source_path": self._source_path,
            "pending": True,
        }
        self._history.append(turn)
        self._render_history()
        self._save_temp()
        return len(self._history) - 1

    def _render_history(self):
        if not hasattr(self, "history_layout"):
            return
        while self.history_layout.count():
            item = self.history_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        if not self._history:
            empty = QLabel("No edits yet. Send a request to generate the first checkpoint.")
            empty.setObjectName("txt2_small")
            empty.setWordWrap(True)
            self.history_layout.addWidget(empty)
        for idx, turn in enumerate(self._history):
            self.history_layout.addWidget(self._history_card(idx, turn))
        self.history_layout.addStretch(1)

    def _history_card(self, idx: int, turn: dict[str, Any]) -> QFrame:
        card = QFrame()
        card.setObjectName("tab_card")
        box = QVBoxLayout(card)
        box.setContentsMargins(10, 8, 10, 8)
        box.setSpacing(6)

        stamp = str(turn.get("updated_at") or turn.get("created_at") or "")
        title = QLabel(f"Edit {idx + 1}  {stamp}")
        title.setObjectName("txt2_small")
        box.addWidget(title)

        prompt_lbl = QLabel("You")
        prompt_lbl.setStyleSheet("font-weight:bold;")
        box.addWidget(prompt_lbl)
        prompt_edit = QTextEdit()
        prompt_edit.setObjectName("log_te")
        prompt_edit.setFont(QFont("Consolas", 9))
        prompt_edit.setFixedHeight(68)
        prompt_edit.setPlainText(str(turn.get("prompt", "")))
        box.addWidget(prompt_edit)

        user_actions = QHBoxLayout()
        btn_save_prompt = QPushButton("Update Prompt")
        btn_branch = QPushButton("Edit From Here")
        set_button_icon(btn_save_prompt, "save", "Update Prompt")
        set_button_icon(btn_branch, "replace", "Edit From Here")
        btn_save_prompt.clicked.connect(lambda _=False, i=idx, te=prompt_edit: self._update_turn_prompt(i, te.toPlainText()))
        btn_branch.clicked.connect(lambda _=False, i=idx, te=prompt_edit: self._branch_from_turn(i, te.toPlainText()))
        user_actions.addWidget(btn_save_prompt)
        user_actions.addWidget(btn_branch)
        user_actions.addStretch(1)
        box.addLayout(user_actions)

        if turn.get("error"):
            err = QLabel(f"Error: {turn.get('error')}")
            err.setWordWrap(True)
            err.setStyleSheet("color:#dc2626;font-weight:600;")
            box.addWidget(err)

        summary = QLabel(str(turn.get("summary", "Thinking...")))
        summary.setWordWrap(True)
        summary.setStyleSheet("font-weight:bold;")
        box.addWidget(summary)

        result_row = QHBoxLayout()
        diff_btn = QPushButton(f"+{int(turn.get('added', 0))}  -{int(turn.get('deleted', 0))}  View diff")
        set_button_icon(diff_btn, "list", diff_btn.text())
        diff_btn.setEnabled(bool(turn.get("diff_html")))
        diff_btn.clicked.connect(lambda _=False, i=idx: self._show_diff_dialog(i))
        result_row.addWidget(diff_btn)
        result_row.addStretch(1)
        box.addLayout(result_row)

        assistant_lbl = QLabel("Assistant response")
        assistant_lbl.setStyleSheet("font-weight:bold;")
        box.addWidget(assistant_lbl)
        raw_edit = QTextEdit()
        raw_edit.setObjectName("log_te")
        raw_edit.setFont(QFont("Consolas", 9))
        raw_edit.setFixedHeight(92)
        raw_edit.setPlainText(str(turn.get("raw_response", "")))
        raw_edit.setPlaceholderText("Assistant structured JSON response")
        raw_edit.setReadOnly(bool(turn.get("pending")))
        box.addWidget(raw_edit)

        assistant_actions = QHBoxLayout()
        btn_restore = QPushButton("Restore File")
        btn_apply_raw = QPushButton("Apply Edited Response")
        set_button_icon(btn_restore, "refresh-cw", "Restore File")
        set_button_icon(btn_apply_raw, "replace", "Apply Edited Response")
        btn_restore.setEnabled(not bool(turn.get("pending")) and bool(turn.get("code_after", "")))
        btn_apply_raw.setEnabled(not bool(turn.get("pending")) and bool(turn.get("raw_response", "")))
        btn_restore.clicked.connect(lambda _=False, i=idx: self._restore_turn(i))
        btn_apply_raw.clicked.connect(lambda _=False, i=idx, te=raw_edit: self._apply_edited_response(i, te.toPlainText()))
        assistant_actions.addWidget(btn_restore)
        assistant_actions.addWidget(btn_apply_raw)
        assistant_actions.addStretch(1)
        box.addLayout(assistant_actions)
        return card

    def _show_diff_dialog(self, idx: int):
        if not (0 <= idx < len(self._history)):
            return
        turn = self._history[idx]
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit {idx + 1} Diff")
        prepare_adaptive_window(dlg, 920, 680, min_width=620, min_height=420)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        hdr = QLabel(str(turn.get("summary", f"Edit {idx + 1}")))
        hdr.setWordWrap(True)
        hdr.setStyleSheet("font-size:14px;font-weight:bold;")
        layout.addWidget(hdr)

        stats = QLabel(
            f"<span style='color:#16a34a;font-weight:600;'>+{int(turn.get('added', 0))} lines added</span>"
            f"    <span style='color:#dc2626;font-weight:600;'>-{int(turn.get('deleted', 0))} lines deleted</span>"
        )
        layout.addWidget(stats)

        diff = QTextEdit()
        diff.setReadOnly(True)
        diff.setObjectName("log_te")
        diff.setFont(QFont("Consolas", 10))
        diff.setHtml(str(turn.get("diff_html", "")) or "No diff recorded.")
        layout.addWidget(diff, 1)

        row = QHBoxLayout()
        btn_restore = QPushButton("Restore File To This Edit")
        btn_close = QPushButton("Close")
        set_button_icon(btn_restore, "refresh-cw", "Restore File To This Edit")
        set_button_icon(btn_close, "error", "Close")
        btn_restore.clicked.connect(lambda: (self._restore_turn(idx), dlg.accept()))
        btn_close.clicked.connect(dlg.accept)
        row.addWidget(btn_restore)
        row.addStretch(1)
        row.addWidget(btn_close)
        layout.addLayout(row)
        dlg.exec()

    def _update_turn_prompt(self, idx: int, prompt: str):
        if 0 <= idx < len(self._history):
            self._history[idx]["prompt"] = prompt.strip()
            self._save_temp()
            self._log(f"Updated prompt for edit {idx + 1}")

    def _branch_from_turn(self, idx: int, prompt: str):
        if not (0 <= idx < len(self._history)):
            return
        turn = self._history[idx]
        self._history = self._history[:idx]
        self._pending_index = None
        self.code_te.blockSignals(True)
        self.code_te.setPlainText(str(turn.get("code_before", "")))
        self.code_te.blockSignals(False)
        self.prompt_te.setPlainText(prompt.strip())
        self._refresh_structure()
        self._render_history()
        self._save_temp()
        self._log(f"Restored file before edit {idx + 1}; ready to re-edit.")

    def _restore_turn(self, idx: int):
        if not (0 <= idx < len(self._history)):
            return
        turn = self._history[idx]
        self._history = self._history[:idx + 1]
        self._pending_index = None
        self.code_te.blockSignals(True)
        self.code_te.setPlainText(str(turn.get("code_after", "")))
        self.code_te.blockSignals(False)
        self._refresh_structure()
        self._render_history()
        self._save_temp()
        self._log(f"Restored edit {idx + 1}")

    def _apply_edited_response(self, idx: int, raw: str):
        if not (0 <= idx < len(self._history)):
            return
        try:
            turn = self._history[idx]
            code_before = str(turn.get("code_before", ""))
            structure = structure_for_code(code_before, str(turn.get("source_path", self._source_path)))
            data = extract_json_object(raw)
            operations = data.get("operations", [])
            if not isinstance(operations, list) or not operations:
                raise ValueError("Edited response has no operations")
            if not code_before.strip() and operations[0].get("op") != "full_replace":
                operations = [{"op": "full_replace", "code": data.get("code", "") or operations[0].get("code", "")}]
            updated = apply_operations(code_before, operations, structure)
            diff_view, added, deleted = render_diff_html(code_before, updated)
            turn.update({
                "summary": data.get("summary", "Edited response applied."),
                "raw_response": raw,
                "diff_html": diff_view,
                "added": added,
                "deleted": deleted,
                "code_after": updated,
                "pending": False,
                "error": "",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })
            self._history = self._history[:idx + 1]
            self._pending_index = None
            self.code_te.blockSignals(True)
            self.code_te.setPlainText(updated)
            self.code_te.blockSignals(False)
            self._refresh_structure()
            self._render_history()
            self._save_temp()
            self._log(f"Applied edited response for edit {idx + 1}")
        except Exception as exc:
            QMessageBox.critical(self, "Edited Response Error", str(exc))
            self._log(f"Edited response error: {exc}")

    def _log(self, msg: str):
        self.log_te.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
