from __future__ import annotations

import ast
from pathlib import Path
from datetime import datetime
from typing import Optional

from nativelab.imports.import_global import (
    QThread,pyqtSignal
)
from nativelab.core.engine_global import ApiEngine


# ─────────────────────────────────────────────────────────────────────────────
#  AST helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_source(node: ast.AST, source_lines: list[str]) -> str:
    """Return the raw source text for an AST node."""
    try:
        return ast.get_source_segment("\n".join(source_lines), node) or ""
    except Exception:
        start = getattr(node, "lineno", 1) - 1
        end   = getattr(node, "end_lineno", start + 1)
        return "\n".join(source_lines[start:end])


def parse_python_file(path: str) -> dict:
    """
    Parse a Python file with ast and return:
      {
        "source":   full source string,
        "classes":  [ { "name", "node", "methods": [{"name", "node"}] } ],
        "functions":[ { "name", "node" } ],   # module-level only
      }
    """
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()

    classes   = []
    functions = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                {"name": m.name, "node": m}
                for m in ast.walk(node)
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and m is not node           # skip nested classes
            ]
            classes.append({"name": node.name, "node": node, "methods": methods})

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({"name": node.name, "node": node})

    return {"source": src, "lines": lines, "classes": classes, "functions": functions}


# ─────────────────────────────────────────────────────────────────────────────
#  Worker
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OVERVIEW_PROMPT = (
    "Generate a structured overview of this Python file. "
    "Describe its purpose, key components, and overall design. "
    "Keep to 2–4 sentences. Use plain prose — do not add headings."
)
DEFAULT_CLASS_PROMPT = (
    "Describe this Python class concisely. Cover its purpose, key attributes, "
    "and what it is responsible for. Do NOT list methods — they will be documented separately. "
    "Keep to 2–4 sentences."
)
DEFAULT_FUNC_PROMPT = (
    "Document this Python function. Include: purpose, parameters (with types if inferrable), "
    "return value, and any important behaviour or side-effects. "
    "Be concise but precise."
)


class PyToDocWorker(QThread):
    """
    Runs the full py-to-doc pipeline off the main thread.

    Signals
    -------
    log_msg(str)   — progress messages for the log panel
    chunk(str)     — incremental text appended to the README (for live preview)
    done()         — pipeline completed successfully
    error(str)     — pipeline failed with this message
    """

    log_msg = pyqtSignal(str)
    chunk   = pyqtSignal(str)
    done    = pyqtSignal()
    error   = pyqtSignal(str)

    def __init__(
        self,
        file_path:     str,
        out_path:      str,
        out_name:      str,
        include_globals:       bool,
        reset_per_function:    bool,
        reset_per_class:       bool,
        prompt_overview:       str,
        prompt_class:          str,
        prompt_function:       str,
        engine:                Optional[ApiEngine],
        parent=None,
    ):
        super().__init__(parent)
        self.file_path          = file_path
        self.out_path           = out_path
        self.out_name           = out_name
        self.include_globals    = include_globals
        self.reset_per_function = reset_per_function
        self.reset_per_class    = reset_per_class
        self.prompt_overview    = prompt_overview  or DEFAULT_OVERVIEW_PROMPT
        self.prompt_class       = prompt_class     or DEFAULT_CLASS_PROMPT
        self.prompt_function    = prompt_function  or DEFAULT_FUNC_PROMPT
        self.engine             = engine
        self._abort             = False

        # Conversation history for context management
        self._history: list[dict] = []

    # ── public ───────────────────────────────────────────────────────────────
    def abort(self):
        self._abort = True

    # ── internals ────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_msg.emit(f"[{ts}]  {msg}")

    def _write(self, fh, text: str):
        """Write text to file and emit as a chunk for live preview."""
        fh.write(text)
        fh.flush()
        self.chunk.emit(text)

    def _reset_context(self):
        self._history.clear()

    def _call_llm(self, system_prompt: str, code: str) -> str:
        """
        Send a request to the loaded engine and return the response text.
        Falls back to a placeholder if no engine is loaded.
        """
        if self.engine is None:
            return "(no LLM engine loaded — wire up engine in _run_py_to_doc)"

        user_content = f"{system_prompt}\n\n```python\n{code}\n```"
        self._history.append({"role": "user", "content": user_content})

        try:
            response = self.engine.chat(
                messages=self._history,
                stream=False,
            )
            text = response.strip()
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

        # ── parse ─────────────────────────────────────────────────────────────
        self._log(f"Parsing: {Path(self.file_path).name}")
        parsed = parse_python_file(self.file_path)
        src    = parsed["source"]
        lines  = parsed["lines"]
        fname  = Path(self.file_path).stem

        # ── open output file ──────────────────────────────────────────────────
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
        # ── Step 1: header ────────────────────────────────────────────────────
        self._write(fh, f"## Doc for {fname}\n---\n\n")
        self._log("README initialized")

        # ── Step 2: overview ──────────────────────────────────────────────────
        if self._abort: return
        self._log("Generating overview…")
        overview_text = self._call_llm(self.prompt_overview, src)
        self._write(fh, f"{overview_text}\n\n---\n\n")
        self._log("Overview generated")

        # ── Step 3: classes ───────────────────────────────────────────────────
        for cls in parsed["classes"]:
            if self._abort: return

            if self.reset_per_class:
                self._reset_context()

            class_code = _get_source(cls["node"], lines)
            self._log(f"Processing class: {cls['name']}")

            # 3.1 — class description
            class_desc = self._call_llm(self.prompt_class, class_code)
            self._write(fh, f"### {cls['name']}\n{class_desc}\n\n")

            # 3.2 — methods
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

        # ── Step 4: global functions (optional) ───────────────────────────────
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
