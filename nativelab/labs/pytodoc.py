"""
Lab feature: py-to-doc.

Generates structured README-style documentation for a Python file by feeding
its overview, classes, and functions through the active LLM. The feature talks
to the rest of the app exclusively through `LabEndpoints`, so it works
identically against a local llama-server, llama-cli, or a remote API model.
"""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

from nativelab.imports.import_global import (
    QThread, pyqtSignal,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTextEdit, QFrame, QScrollArea, QFileDialog, QMessageBox,
    QListWidget, QListWidgetItem,
    QFont, Qt, QSlider, QSpinBox,
)
from nativelab.UI.icons import refresh_widget_icons, set_button_icon, set_label_icon, set_status_label
from nativelab.UI.toggle import ToggleSwitch
from nativelab.GlobalConfig.const import LONG_TIMEOUT_MS, MAX_CONTEXT_TOKENS

from .endpoints import ContextWindowExceededError, LabEndpoints

try:
    import psutil as _psutil
except Exception:
    _psutil = None


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
PYTODOC_JOBS_DIR = Path("./localllm/pytodoc_jobs")
CONTEXT_POLICY_NONE = "none"
CONTEXT_POLICY_FIXED = "fixed"
CONTEXT_POLICY_AUTO = "auto"
AUTO_CONTEXT_MIN = 512
AUTO_CONTEXT_MAX = MAX_CONTEXT_TOKENS
AUTO_CONTEXT_DEFAULT = 4096
AUTO_RELOAD_RAM_GB_DEFAULT = 1
AUTO_RELOAD_RAM_MB_DEFAULT = 0
AUTO_RELOAD_REARM_MARGIN_MB = 512
AUTO_RELOAD_DROP_MARGIN_MB = 768
AUTO_RELOAD_MIN_STEPS_BETWEEN = 2
AUTO_RELOAD_MIN_SECONDS_BETWEEN = 45
AUTO_RELOAD_MAX_FAILURES = 2

DEFAULT_PROJECT_IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "build",
    "dist",
    ".eggs",
    ".cache",
}


class _GitignorePattern:
    def __init__(self, pattern: str, *, negated: bool, anchored: bool, directory_only: bool):
        self.pattern = pattern
        self.negated = negated
        self.anchored = anchored
        self.directory_only = directory_only


class _ProjectIgnoreMatcher:
    """Small stdlib-only matcher for the project-root .gitignore subset we need."""

    def __init__(self, project_root: Path):
        self.root = project_root.resolve()
        self.patterns = self._load_gitignore()

    def is_ignored(self, path: Path, *, is_dir: bool) -> bool:
        rel = self._rel(path)
        if not rel:
            return False
        parts = rel.split("/")
        if any(part in DEFAULT_PROJECT_IGNORE_DIRS for part in parts):
            return True

        ignored = False
        for spec in self.patterns:
            if self._matches(spec, rel, is_dir):
                ignored = not spec.negated
        return ignored

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except Exception:
            return ""

    def _load_gitignore(self) -> list[_GitignorePattern]:
        gitignore = self.root / ".gitignore"
        if not gitignore.exists():
            return []
        patterns: list[_GitignorePattern] = []
        for raw in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            negated = line.startswith("!")
            if negated:
                line = line[1:].strip()
            if not line:
                continue
            anchored = line.startswith("/")
            line = line.lstrip("/")
            directory_only = line.endswith("/")
            line = line.rstrip("/")
            if line:
                patterns.append(_GitignorePattern(
                    line,
                    negated=negated,
                    anchored=anchored,
                    directory_only=directory_only,
                ))
        return patterns

    def _matches(self, spec: _GitignorePattern, rel: str, is_dir: bool) -> bool:
        pattern = spec.pattern
        if spec.directory_only:
            if spec.anchored or "/" in pattern:
                return rel == pattern or rel.startswith(pattern + "/")
            return any(part == pattern or fnmatch.fnmatchcase(part, pattern) for part in rel.split("/"))

        if spec.anchored or "/" in pattern:
            return fnmatch.fnmatchcase(rel, pattern)
        name = Path(rel).name
        return fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(rel, pattern)


def _walk_project(project_root: Union[str, Path], output_root: Optional[Union[str, Path]] = None):
    root = Path(project_root).expanduser().resolve()
    matcher = _ProjectIgnoreMatcher(root)
    output_rel = None
    if output_root:
        try:
            output_rel = Path(output_root).expanduser().resolve().relative_to(root).as_posix()
        except Exception:
            output_rel = None

    def is_output_path(candidate: Path) -> bool:
        if not output_rel:
            return False
        try:
            rel = candidate.resolve().relative_to(root).as_posix()
        except Exception:
            return False
        return rel == output_rel or rel.startswith(output_rel + "/")

    for current, dirs, files in os.walk(root):
        current_path = Path(current)

        if is_output_path(current_path):
            dirs[:] = []
            continue

        dirs[:] = sorted(
            d for d in dirs
            if not matcher.is_ignored(current_path / d, is_dir=True)
            and not is_output_path(current_path / d)
        )
        yield current_path, dirs, sorted(files), matcher


def discover_project_python_files(project_root: Union[str, Path]) -> list[str]:
    """Return recursive Python files under project_root, filtered by project .gitignore."""
    files: list[str] = []
    for current_path, _, names, matcher in _walk_project(project_root):
        for name in names:
            path = current_path / name
            if path.suffix == ".py" and not matcher.is_ignored(path, is_dir=False):
                files.append(str(path))
    return sorted(files)


def mirror_project_directories(project_root: Union[str, Path], output_root: Union[str, Path]) -> int:
    """Create a directory-only mirror of project_root inside output_root."""
    root = Path(project_root).expanduser().resolve()
    out = Path(output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    created = 0
    for current_path, dirs, _, _ in _walk_project(root, out):
        for dirname in dirs:
            src_dir = current_path / dirname
            rel = src_dir.resolve().relative_to(root)
            dst = out / rel
            existed = dst.exists()
            dst.mkdir(parents=True, exist_ok=True)
            if not existed:
                created += 1
    return created


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
                {"name": child.name, "node": child}
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            methods.sort(key=lambda item: getattr(item["node"], "lineno", 0))
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
        context_policy:      str,
        fixed_reset_per_function: bool,
        fixed_reset_per_class: bool,
        auto_context_tokens: int,
        auto_model_reload: bool,
        auto_reload_free_ram_mb: int,
        prompt_overview:     str,
        prompt_class:        str,
        prompt_function:     str,
        endpoints:           LabEndpoints,
        file_list:           Optional[List[str]] = None,
        project_root:        Optional[str] = None,
        resume_required:     bool = False,
        state_path:          Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.file_list    = file_list or []
        self.project_root = project_root
        self.resume_required = resume_required
        self.file_path          = file_path
        self.out_path           = out_path
        self.out_name           = out_name
        self.include_globals    = include_globals
        self.context_policy     = context_policy if context_policy in {
            CONTEXT_POLICY_NONE,
            CONTEXT_POLICY_FIXED,
            CONTEXT_POLICY_AUTO,
        } else CONTEXT_POLICY_FIXED
        self.reset_per_function = fixed_reset_per_function
        self.reset_per_class    = fixed_reset_per_class
        self.auto_context_tokens = max(AUTO_CONTEXT_MIN, int(auto_context_tokens or AUTO_CONTEXT_DEFAULT))
        self.auto_model_reload = bool(auto_model_reload)
        self.auto_reload_free_ram_mb = max(0, int(auto_reload_free_ram_mb or 0))
        self.prompt_overview    = prompt_overview or DEFAULT_OVERVIEW_PROMPT
        self.prompt_class       = prompt_class    or DEFAULT_CLASS_PROMPT
        self.prompt_function    = prompt_function or DEFAULT_FUNC_PROMPT
        self.endpoints          = endpoints
        self._abort             = False
        self._pause             = False
        self._history: List[dict] = []
        self._state: Optional[dict[str, Any]] = None
        self._resuming = False
        self._state_path = Path(state_path) if state_path else PYTODOC_TEMP_STATE
        self._state_path_explicit = bool(state_path)
        self._context_overflow_pending = False
        self._budget_warned = False
        self._ram_watch_warned = False
        self._ram_reload_pending = False
        self._ram_reload_pending_free_mb: Optional[int] = None
        self._ram_reload_pending_after = ""
        self._ram_reload_armed = True
        self._ram_reload_steps_since_reload = AUTO_RELOAD_MIN_STEPS_BETWEEN
        self._ram_reload_last_at = 0.0
        self._ram_reload_post_free_mb: Optional[int] = None
        self._ram_reload_failures = 0
        self._ram_reload_last_skip_reason = ""

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
        self._context_overflow_pending = False

    @staticmethod
    def _estimate_tokens_for_messages(messages: List[dict]) -> int:
        chars = sum(len(str(m.get("content", ""))) for m in messages)
        return max(0, (chars + 3) // 4)

    def _history_token_estimate(self) -> int:
        return self._estimate_tokens_for_messages(self._history)

    def _effective_auto_context_budget(self) -> int:
        budget = self.auto_context_tokens
        try:
            loaded_ctx = int(getattr(self.endpoints, "ctx_value", budget))
        except Exception:
            loaded_ctx = budget
        if loaded_ctx > 0:
            budget = min(budget, loaded_ctx)
        return max(AUTO_CONTEXT_MIN, budget)

    def _prepare_context_for_llm_call(self, upcoming_user_content: str):
        if self.context_policy != CONTEXT_POLICY_AUTO:
            return
        budget = self._effective_auto_context_budget()
        upcoming = self._estimate_tokens_for_messages([
            {"role": "user", "content": upcoming_user_content}
        ])
        current = self._history_token_estimate()
        combined = current + upcoming
        if self._context_overflow_pending or combined >= budget:
            if current:
                self._log(
                    "Auto context refresh: cleared carried py-to-doc history "
                    f"before next section (~{combined:,}/{budget:,} tokens)"
                )
            self._reset_context()
            if upcoming >= budget:
                self._log(
                    "Warning: current section alone is larger than the loaded "
                    f"context (~{upcoming:,}/{budget:,} tokens); increase the "
                    "loaded model context to avoid llama-server rejection"
                )

    def _update_context_budget_after_call(self):
        if self.context_policy != CONTEXT_POLICY_AUTO:
            return
        budget = self._effective_auto_context_budget()
        current = self._history_token_estimate()
        if current >= budget:
            self._context_overflow_pending = True
            self._log(
                "Auto context budget reached "
                f"(~{current:,}/{budget:,} tokens); "
                "next section will start with fresh py-to-doc context"
            )

    def _log_context_policy(self):
        if self.context_policy == CONTEXT_POLICY_NONE:
            self._log("Context policy: no reset")
        elif self.context_policy == CONTEXT_POLICY_FIXED:
            parts = []
            if self.reset_per_function:
                parts.append("function")
            if self.reset_per_class:
                parts.append("class")
            self._log(f"Context policy: fixed reset ({', '.join(parts) or 'manual only'})")
        else:
            self._log(
                f"Context policy: auto budget reset at ~{self.auto_context_tokens:,} tokens"
            )
            if self.auto_model_reload and self.auto_reload_free_ram_mb > 0:
                self._log(
                    "Auto model reload: enabled when free RAM drops below "
                    f"{self.auto_reload_free_ram_mb:,} MB after a completed section"
                )
            try:
                loaded_ctx = int(getattr(self.endpoints, "ctx_value", AUTO_CONTEXT_DEFAULT))
            except Exception:
                loaded_ctx = AUTO_CONTEXT_DEFAULT
            if self.auto_context_tokens > loaded_ctx and not self._budget_warned:
                self._log(
                    "Warning: auto budget is above loaded model context "
                    f"({self.auto_context_tokens:,} > {loaded_ctx:,}); using "
                    "loaded context as the reset threshold"
                )
                self._budget_warned = True

    def _project_enabled(self) -> bool:
        return bool(self.project_root and self.file_list)

    def _settings_payload(self) -> dict[str, Any]:
        return {
            "include_globals": self.include_globals,
            "context_policy": self.context_policy,
            "fixed_reset_per_function": self.reset_per_function,
            "fixed_reset_per_class": self.reset_per_class,
            "auto_context_tokens": self.auto_context_tokens,
            "auto_model_reload": self.auto_model_reload,
            "auto_reload_free_ram_mb": self.auto_reload_free_ram_mb,
            "prompt_overview": self.prompt_overview,
            "prompt_class": self.prompt_class,
            "prompt_function": self.prompt_function,
        }

    def _settings_fingerprint(self) -> str:
        payload = self._settings_payload()
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _project_key(self) -> str:
        raw = json.dumps({
            "project_root": str(Path(self.project_root or "").resolve()),
            "out_path": str(Path(self.out_path).resolve()),
            "settings": self._settings_fingerprint(),
        }, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _job_state_path(self) -> Path:
        return PYTODOC_JOBS_DIR / f"{self._project_key()}.json"

    def _candidate_state_paths(self) -> list[Path]:
        paths = [self._state_path]
        if self._project_enabled():
            paths.append(self._job_state_path())
        paths.append(PYTODOC_TEMP_STATE)
        out: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path.resolve())
            if key not in seen:
                out.append(path)
                seen.add(key)
        return out

    @staticmethod
    def _write_state_file(path: Path, state: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(state, indent=2, ensure_ascii=False)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(raw, encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _text_file_end(path: Path) -> int:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, 2)
            return int(fh.tell())

    @classmethod
    def _repair_resume_state_outputs(
        cls,
        state: dict[str, Any],
        *,
        log_cb=None,
    ) -> bool:
        files = state.get("files")
        if not isinstance(files, dict):
            return False
        changed = False
        completed = set(state.get("completed_files", []) or [])
        file_list = [str(Path(p).resolve()) for p in (state.get("file_list", []) or [])]

        for raw_key, info in list(files.items()):
            if not isinstance(info, dict):
                continue
            key = str(Path(raw_key).resolve())
            if key != raw_key:
                files[key] = files.pop(raw_key)
                changed = True
            source_sig = cls._source_signature(key)
            if info.get("source") and source_sig != info.get("source"):
                info["steps"] = []
                info["offset"] = 0
                info["completed"] = False
                info["source"] = source_sig
                completed.discard(key)
                changed = True
                if log_cb:
                    log_cb(f"Recovered checkpoint: source changed, regenerating {Path(key).name}")
                continue

            out_file = Path(str(info.get("out_file", "")))
            if not out_file.exists():
                if info.get("steps") or info.get("offset") or info.get("completed"):
                    info["steps"] = []
                    info["offset"] = 0
                    info["completed"] = False
                    completed.discard(key)
                    changed = True
                    if log_cb:
                        log_cb(f"Recovered checkpoint: missing output will regenerate {Path(key).name}")
                continue

            try:
                end = cls._text_file_end(out_file)
            except Exception:
                end = 0
            try:
                offset = int(info.get("offset", 0) or 0)
            except Exception:
                offset = 0
            if offset < 0 or offset > end:
                info["offset"] = max(0, min(offset, end))
                info["completed"] = False
                completed.discard(key)
                changed = True
                if log_cb:
                    log_cb(f"Recovered checkpoint: repaired invalid offset for {Path(key).name}")
            if end <= 0 and (info.get("steps") or info.get("completed")):
                info["steps"] = []
                info["offset"] = 0
                info["completed"] = False
                completed.discard(key)
                changed = True
            elif info.get("completed"):
                completed.add(key)

        completed = {p for p in completed if p in files}
        if sorted(completed) != sorted(state.get("completed_files", []) or []):
            state["completed_files"] = sorted(completed)
            changed = True
        current = str(state.get("current_file", "") or "")
        if current and current not in files:
            state["current_file"] = ""
            changed = True
        if not state.get("current_file"):
            for path in file_list:
                info = files.get(path, {})
                if not info.get("completed"):
                    state["current_file"] = path
                    changed = True
                    break
        return changed

    @classmethod
    def recover_state_file_for_resume(
        cls,
        path: Path,
        state: dict[str, Any],
        *,
        recover_running: bool = True,
        log_cb=None,
    ) -> tuple[dict[str, Any], bool]:
        changed = cls._repair_resume_state_outputs(state, log_cb=log_cb)
        status = str(state.get("status", "") or "")
        if recover_running and status == "running":
            state["status"] = "paused"
            state["recovered_from_status"] = "running"
            state["recovered_at"] = datetime.now().isoformat(timespec="seconds")
            state["recovery_reason"] = (
                "Previous py-to-doc run ended without writing a paused, "
                "complete, aborted, or failed status."
            )
            changed = True
            if log_cb:
                log_cb(f"Recovered stale running checkpoint -> {path}")
        if state.get("status") == "complete":
            file_list = state.get("file_list", []) or []
            completed = set(state.get("completed_files", []) or [])
            if len(completed) < len(file_list):
                state["status"] = "paused"
                state["recovered_from_status"] = "complete"
                state["recovered_at"] = datetime.now().isoformat(timespec="seconds")
                state["recovery_reason"] = "Complete checkpoint was missing completed files."
                changed = True
        if changed:
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            cls._write_state_file(path, state)
        return state, changed

    def _load_project_state(self) -> Optional[dict[str, Any]]:
        if not self._project_enabled():
            return None
        found_any = False
        for path in self._candidate_state_paths():
            if not path.exists():
                continue
            found_any = True
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._log(f"Recovery checkpoint ignored: could not read {path} ({exc})")
                continue
            if state.get("feature") != "py_to_doc":
                continue
            if state.get("project_key") == self._project_key():
                self._state_path = path
                state, _ = self.recover_state_file_for_resume(
                    path,
                    state,
                    recover_running=True,
                    log_cb=self._log,
                )
                return state
            if (
                self.resume_required
                and self._state_path_explicit
                and path.resolve() == self._state_path.resolve()
                and str(Path(state.get("project_root", "")).resolve()) == str(Path(self.project_root or "").resolve())
                and str(Path(state.get("out_path", "")).resolve()) == str(Path(self.out_path).resolve())
            ):
                self._log("Selected checkpoint loaded; settings metadata did not match current controls")
                self._state_path = path
                state, _ = self.recover_state_file_for_resume(
                    path,
                    state,
                    recover_running=True,
                    log_cb=self._log,
                )
                return state
        if found_any:
            self._log("Recovery checkpoint ignored: project, output folder, or settings changed")
        return None

    def _fresh_project_state(self) -> dict[str, Any]:
        return {
            "feature": "py_to_doc",
            "version": 1,
            "status": "running",
            "project_key": self._project_key(),
            "project_root": str(Path(self.project_root or "").resolve()),
            "out_path": str(Path(self.out_path).resolve()),
            "settings_fingerprint": self._settings_fingerprint(),
            "settings": self._settings_payload(),
            "file_list": [str(Path(p).resolve()) for p in self.file_list],
            "current_file": "",
            "completed_files": [],
            "files": {},
            "history": [],
            "context_overflow_pending": False,
            "updated_at": "",
        }

    def _save_project_state(self, status: Optional[str] = None):
        if not self._project_enabled() or self._state is None:
            return
        if status:
            self._state["status"] = status
        self._state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._state["history"] = list(self._history)
        self._state["context_overflow_pending"] = bool(self._context_overflow_pending)
        self._state["file_list"] = [str(Path(p).resolve()) for p in self.file_list]
        targets = [self._state_path, PYTODOC_TEMP_STATE, self._job_state_path()]
        seen: set[str] = set()
        for target in targets:
            key = str(target.resolve())
            if key in seen:
                continue
            seen.add(key)
            self._write_state_file(target, self._state)

    def _prepare_project_state(self):
        if not self._project_enabled():
            return
        state = self._load_project_state()
        if state:
            self._state = state
            self._resuming = True
            self._history = list(state.get("history", []))
            self._context_overflow_pending = bool(state.get("context_overflow_pending", False))
            self._log(f"Resume checkpoint found -> {self._state_path}")
            self._verify_project_state()
        elif self.resume_required:
            raise RuntimeError(
                "No matching py-to-doc checkpoint found. Select the same "
                "project root, output folder, and settings used before pausing."
            )
        else:
            self._state_path = self._job_state_path()
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

    def _expected_steps(self, parsed: dict) -> set[str]:
        steps = {"header", "overview"}
        for cls in parsed.get("classes", []):
            class_name = cls["name"]
            class_lineno = getattr(cls["node"], "lineno", 0)
            methods = sorted(
                cls.get("methods", []),
                key=lambda item: getattr(item["node"], "lineno", 0),
            )
            steps.add(f"class:{class_name}:{class_lineno}")
            if methods:
                steps.add(f"class-functions:{class_name}:{class_lineno}")
            for method in methods:
                steps.add(
                    f"method:{class_name}.{method['name']}:"
                    f"{getattr(method['node'], 'lineno', 0)}"
                )
            steps.add(f"class-end:{class_name}:{class_lineno}")
        if self.include_globals and parsed.get("functions"):
            steps.add("global-functions-header")
            for fn in parsed["functions"]:
                steps.add(f"global:{fn['name']}:{getattr(fn['node'], 'lineno', 0)}")
        return steps

    def _repair_incomplete_complete_file(self, file_path: str, info: dict[str, Any], parsed: dict):
        if self._state is None or not info.get("completed"):
            return
        missing = self._expected_steps(parsed).difference(set(info.get("steps", [])))
        if not missing:
            return
        key = str(Path(file_path).resolve())
        info["steps"] = []
        info["offset"] = 0
        info["completed"] = False
        completed = set(self._state.get("completed_files", []))
        completed.discard(key)
        self._state["completed_files"] = sorted(completed)
        self._log(
            f"Checkpoint repaired: {Path(file_path).name} was marked complete "
            f"with {len(missing)} missing sections"
        )
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

    def _save_failed_state_for_resume(self, exc: Exception):
        if not self._project_enabled() or self._state is None:
            return
        self._state["last_error"] = str(exc)
        self._state["failed_at"] = datetime.now().isoformat(timespec="seconds")
        self._state["recovery_reason"] = "Worker error was converted into a resumable pause."
        self._save_project_state("paused")

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
        self._maybe_schedule_auto_reload_after_step(file_path, step_id)
        return self._should_continue()

    def _wait_for_engine_ready(self) -> bool:
        if self.endpoints is None:
            return False
        if self.endpoints.is_loaded:
            return True
        self._log("Model is still loading; waiting before writing this section")
        try:
            return bool(self.endpoints.wait_until_loaded(LONG_TIMEOUT_MS))
        except Exception as exc:
            self._log(f"Model wait failed: {exc}")
            return False

    def _call_llm(self, system_prompt: str, code: str) -> str:
        if not self._reload_active_model_before_next_llm():
            raise RuntimeError("py-to-doc stopped before the next LLM section")
        if not self._wait_for_engine_ready():
            raise RuntimeError(
                "No LLM engine is loaded. Load a model first, then rerun py-to-doc."
            )

        user_content = f"{system_prompt}\n\n```python\n{code}\n```"
        self._prepare_context_for_llm_call(user_content)
        return self._call_llm_with_retry(user_content)

    @staticmethod
    def _is_engine_not_ready_error(raw: str) -> bool:
        text = raw.lower()
        return any(needle in text for needle in (
            "no engine loaded",
            "not loaded",
            "model is not loaded",
            "no local engine loaded",
            "api config not loaded",
        ))

    @staticmethod
    def _available_ram_mb() -> Optional[int]:
        if _psutil is None:
            return None
        try:
            return int(_psutil.virtual_memory().available // (1024 * 1024))
        except Exception:
            return None

    def _auto_model_reload_enabled(self) -> bool:
        return (
            self.context_policy == CONTEXT_POLICY_AUTO
            and self.auto_model_reload
            and self.auto_reload_free_ram_mb > 0
        )

    def _clear_pending_auto_reload(self):
        self._ram_reload_pending = False
        self._ram_reload_pending_free_mb = None
        self._ram_reload_pending_after = ""

    def _auto_reload_rearm_margin_mb(self) -> int:
        return max(
            AUTO_RELOAD_REARM_MARGIN_MB,
            min(2048, int(self.auto_reload_free_ram_mb * 0.10)),
        )

    def _auto_reload_drop_margin_mb(self, baseline_mb: Optional[int] = None) -> int:
        margin = max(
            AUTO_RELOAD_DROP_MARGIN_MB,
            min(2048, int(self.auto_reload_free_ram_mb * 0.08)),
        )
        if baseline_mb is not None and baseline_mb > 0:
            margin = min(margin, max(256, baseline_mb // 2))
        return margin

    def _auto_reload_cooldown_ready(self) -> bool:
        if self._ram_reload_steps_since_reload < AUTO_RELOAD_MIN_STEPS_BETWEEN:
            return False
        if self._ram_reload_last_at <= 0:
            return True
        return (time.monotonic() - self._ram_reload_last_at) >= AUTO_RELOAD_MIN_SECONDS_BETWEEN

    def _log_auto_reload_skip_once(self, reason: str):
        if reason and reason != self._ram_reload_last_skip_reason:
            self._log(reason)
            self._ram_reload_last_skip_reason = reason

    def _reset_auto_reload_skip_reason(self):
        self._ram_reload_last_skip_reason = ""

    def _update_auto_reload_arming(self, free_mb: int):
        if self._ram_reload_armed:
            return
        rearm_mb = self.auto_reload_free_ram_mb + self._auto_reload_rearm_margin_mb()
        if free_mb >= rearm_mb:
            self._ram_reload_armed = True
            self._ram_reload_post_free_mb = None
            self._reset_auto_reload_skip_reason()
            self._log(
                "Auto model reload re-armed after RAM recovered "
                f"to {free_mb:,} MB"
            )
            return
        baseline = self._ram_reload_post_free_mb
        if baseline is None:
            return
        drop_margin = self._auto_reload_drop_margin_mb(baseline)
        if free_mb <= max(0, baseline - drop_margin):
            self._ram_reload_armed = True
            self._reset_auto_reload_skip_reason()
            self._log(
                "Auto model reload re-armed because free RAM dropped "
                f"from post-reload baseline {baseline:,} MB to {free_mb:,} MB"
            )

    @staticmethod
    def _step_label(file_path: str, step_id: str) -> str:
        file_name = Path(file_path).name
        parts = step_id.split(":")
        if step_id == "overview":
            return f"overview in {file_name}"
        if step_id == "header":
            return f"header in {file_name}"
        if step_id == "global-functions-header":
            return f"global functions header in {file_name}"
        if parts[0] == "class" and len(parts) >= 2:
            return f"class {parts[1]} in {file_name}"
        if parts[0] == "method" and len(parts) >= 2:
            return f"function {parts[1]} in {file_name}"
        if parts[0] == "global" and len(parts) >= 2:
            return f"function {parts[1]} in {file_name}"
        if parts[0] == "class-end" and len(parts) >= 2:
            return f"class {parts[1]} footer in {file_name}"
        if step_id == "file-complete":
            return f"file {file_name}"
        return f"{step_id} in {file_name}"

    def _can_auto_reload_active_model(self) -> bool:
        if self.endpoints is None:
            return False
        try:
            if getattr(self.endpoints, "is_loading", False):
                self._log_auto_reload_skip_once(
                    "Auto model reload skipped: model load is already in progress"
                )
                return False
        except Exception:
            pass
        if getattr(self.endpoints, "can_reload_active_model", False):
            return True
        if not self._ram_watch_warned:
            self._log("Auto model reload skipped: active backend cannot be reloaded by Labs")
            self._ram_watch_warned = True
        return False

    def _maybe_schedule_auto_reload_after_step(self, file_path: str, step_id: str):
        if (
            self._ram_reload_pending
            or not self._auto_model_reload_enabled()
        ):
            return
        self._ram_reload_steps_since_reload += 1
        if not self._can_auto_reload_active_model():
            return
        free_mb = self._available_ram_mb()
        if free_mb is None:
            if not self._ram_watch_warned:
                self._log("Auto model reload skipped: RAM stats are not available")
                self._ram_watch_warned = True
            return
        if free_mb >= self.auto_reload_free_ram_mb:
            self._update_auto_reload_arming(free_mb)
            self._reset_auto_reload_skip_reason()
            return
        self._update_auto_reload_arming(free_mb)
        if not self._ram_reload_armed:
            baseline = self._ram_reload_post_free_mb
            if baseline is None:
                self._log_auto_reload_skip_once(
                    "Auto model reload held: waiting for RAM recovery before reloading again"
                )
            else:
                drop_margin = self._auto_reload_drop_margin_mb(baseline)
                self._log_auto_reload_skip_once(
                    "Auto model reload held: RAM is still below threshold after "
                    f"the last reload; next reload needs recovery or a drop below "
                    f"{max(0, baseline - drop_margin):,} MB"
                )
            return
        if not self._auto_reload_cooldown_ready():
            self._log_auto_reload_skip_once(
                "Auto model reload held: waiting for cooldown after the last reload"
            )
            return

        self._ram_reload_pending = True
        self._ram_reload_pending_free_mb = free_mb
        self._ram_reload_pending_after = self._step_label(file_path, step_id)
        self._reset_auto_reload_skip_reason()
        self._log(
            f"Free RAM {free_mb:,} MB is below threshold "
            f"{self.auto_reload_free_ram_mb:,} MB after "
            f"{self._ram_reload_pending_after}; active task finished, "
            "model will fully unload/reload before the next LLM section"
        )

    def _reload_active_model_before_next_llm(self) -> bool:
        if not self._ram_reload_pending:
            return True
        if not self._should_continue():
            return False
        if not self._auto_model_reload_enabled():
            self._clear_pending_auto_reload()
            return True
        if not self._can_auto_reload_active_model():
            self._clear_pending_auto_reload()
            return True

        free_mb = self._available_ram_mb()
        if free_mb is not None:
            if free_mb >= self.auto_reload_free_ram_mb:
                self._log(
                    "Auto model reload cancelled: free RAM recovered to "
                    f"{free_mb:,} MB before the next section"
                )
                self._clear_pending_auto_reload()
                self._update_auto_reload_arming(free_mb)
                return True
            self._update_auto_reload_arming(free_mb)
            if not self._ram_reload_armed:
                self._clear_pending_auto_reload()
                return True
        if not self._auto_reload_cooldown_ready():
            self._clear_pending_auto_reload()
            return True
        if free_mb is not None:
            free_label = f"{free_mb:,} MB"
        elif self._ram_reload_pending_free_mb is not None:
            free_label = f"{self._ram_reload_pending_free_mb:,} MB"
        else:
            free_label = "unknown"
        self._log(
            "Auto model reload starting before next py-to-doc section "
            f"(free RAM {free_label}; threshold "
            f"{self.auto_reload_free_ram_mb:,} MB)"
        )
        self._reset_context()
        self._save_project_state("running")
        try:
            ok = bool(self.endpoints.request_active_model_reload())
        except Exception as exc:
            self._ram_reload_failures += 1
            self._clear_pending_auto_reload()
            if self.endpoints is not None and getattr(self.endpoints, "is_loaded", False):
                self._log(f"Auto model reload failed but current model is still ready: {exc}")
                if self._ram_reload_failures >= AUTO_RELOAD_MAX_FAILURES:
                    self.auto_model_reload = False
                    self._log("Auto model reload disabled after repeated failures")
                return True
            raise RuntimeError(f"Auto model reload failed: {exc}") from exc
        if not ok or not self._wait_for_engine_ready():
            self._ram_reload_failures += 1
            self._clear_pending_auto_reload()
            if self.endpoints is not None and getattr(self.endpoints, "is_loaded", False):
                self._log("Auto model reload failed; continuing with current model")
                if self._ram_reload_failures >= AUTO_RELOAD_MAX_FAILURES:
                    self.auto_model_reload = False
                    self._log("Auto model reload disabled after repeated failures")
                return True
            raise RuntimeError("Auto model reload failed; active model is not ready")
        self._clear_pending_auto_reload()
        self._ram_reload_failures = 0
        self._ram_reload_steps_since_reload = 0
        self._ram_reload_last_at = time.monotonic()
        post_free_mb = self._available_ram_mb()
        if post_free_mb is not None and post_free_mb < self.auto_reload_free_ram_mb:
            self._ram_reload_armed = False
            self._ram_reload_post_free_mb = post_free_mb
            self._log(
                "Auto model reload complete; free RAM is still below threshold "
                f"({post_free_mb:,}/{self.auto_reload_free_ram_mb:,} MB), "
                "so further reloads are paused until RAM recovers or drops again"
            )
        else:
            self._ram_reload_armed = True
            self._ram_reload_post_free_mb = post_free_mb
        self._save_project_state("running")
        self._log("Auto model reload complete; continuing py-to-doc")
        return True

    def _maybe_auto_reload_after_file(self, file_path: str, *, has_more_files: bool):
        if (
            not has_more_files
            or not self._auto_model_reload_enabled()
        ):
            return
        if not self._ram_reload_pending:
            self._maybe_schedule_auto_reload_after_step(file_path, "file-complete")
        self._reload_active_model_before_next_llm()

    def _selected_context_value(self) -> int:
        if self.context_policy == CONTEXT_POLICY_AUTO:
            return max(AUTO_CONTEXT_MIN, int(self.auto_context_tokens or AUTO_CONTEXT_DEFAULT))
        try:
            return int(getattr(self.endpoints, "ctx_value", AUTO_CONTEXT_DEFAULT))
        except Exception:
            return AUTO_CONTEXT_DEFAULT

    def _recover_from_context_exceeded(self, exc: ContextWindowExceededError) -> bool:
        target_ctx = self._selected_context_value()
        self._log(
            "Context exceeded while generating current section; clearing "
            f"py-to-doc history and reloading context {target_ctx:,}"
        )
        self._reset_context()
        try:
            ok = self.endpoints.request_context(target_ctx)
        except Exception as reload_exc:
            self._log(f"Context reload failed: {reload_exc}")
            ok = False
        if not ok:
            self._log(f"Context reload was not available after backend error: {exc}")
        return bool(ok)

    def _call_llm_with_retry(self, user_content: str) -> str:
        for attempt in range(2):
            self._history.append({"role": "user", "content": user_content})
            try:
                text = self.endpoints.call_llm(messages=self._history).strip()
            except ContextWindowExceededError as exc:
                self._history.pop()
                self._context_overflow_pending = True
                if attempt == 0 and self._recover_from_context_exceeded(exc):
                    continue
                return f"[LLM error: {exc}]"
            except Exception as exc:
                self._history.pop()
                if self._is_engine_not_ready_error(str(exc)):
                    if attempt == 0 and self._wait_for_engine_ready():
                        continue
                    raise RuntimeError(
                        "LLM engine is not ready; no documentation was written for this section"
                    ) from exc
                text = f"[LLM error: {exc}]"
                self._context_overflow_pending = True
                return text

            self._history.append({"role": "assistant", "content": text})
            self._update_context_budget_after_call()
            return text
        return "[LLM error: context retry failed]"

    # ── pipeline ─────────────────────────────────────────────────────────────
    def run(self):
        try:
            self._pipeline()
        except Exception as exc:
            self._save_failed_state_for_resume(exc)
            self.error.emit(str(exc))

    def _pipeline(self):
        files = self.file_list if self.file_list else [self.file_path]
        self._prepare_project_state()
        self._log_context_policy()
        for index, file_path in enumerate(files):
            if not self._should_continue():
                return
            self._process_one(file_path)
            if self._pause or self._abort:
                return
            self._maybe_auto_reload_after_file(
                file_path,
                has_more_files=(index < len(files) - 1),
            )
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
        if file_state:
            self._repair_incomplete_complete_file(file_path, file_state, parsed)
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
            methods = sorted(
                cls.get("methods", []),
                key=lambda item: getattr(item["node"], "lineno", 0),
            )
            self._log(f"Processing class: {cls['name']} ({len(methods)} methods)")

            class_step = f"class:{cls['name']}:{getattr(cls['node'], 'lineno', 0)}"
            if (
                self.context_policy == CONTEXT_POLICY_FIXED
                and self.reset_per_class
                and not self._step_done(file_path, class_step)
            ):
                self._reset_context()
            if not self._run_step(
                fh, file_path, class_step,
                lambda cls=cls, class_code=class_code: (
                    f"### {cls['name']}\n"
                    f"{self._call_llm(self.prompt_class, class_code)}\n\n"
                )
            ):
                return False

            if not methods:
                self._log(f"No direct methods found for class: {cls['name']}")
            else:
                functions_step = f"class-functions:{cls['name']}:{getattr(cls['node'], 'lineno', 0)}"
                if not self._run_step(fh, file_path, functions_step, lambda: "### Functions\n\n"):
                    return False
            for method in methods:
                if not self._should_continue(): return False

                fn_code = _get_source(method["node"], lines)
                method_step = (
                    f"method:{cls['name']}.{method['name']}:"
                    f"{getattr(method['node'], 'lineno', 0)}"
                )
                if (
                    self.context_policy == CONTEXT_POLICY_FIXED
                    and self.reset_per_function
                    and not self._step_done(file_path, method_step)
                ):
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

            self._log(f"Class processed: {cls['name']} ({len(methods)} methods)")
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
                if (
                    self.context_policy == CONTEXT_POLICY_FIXED
                    and self.reset_per_function
                    and not self._step_done(file_path, fn_step)
                ):
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
        self._resume_state_path: str = ""
        self._build()

    # ── endpoint wiring ──────────────────────────────────────────────────────
    def set_endpoints(self, endpoints: LabEndpoints):
        self._endpoints = endpoints
        endpoints.status_changed.connect(self._on_status_changed)
        self._on_status_changed(endpoints.status_text)
        if not getattr(self, "_budget_user_touched", False):
            self._set_auto_budget_value(getattr(endpoints, "ctx_value", AUTO_CONTEXT_DEFAULT), mark_touched=False)

    def refresh_icons(self):
        refresh_widget_icons(self)

    def refresh_theme(self):
        self.refresh_icons()

    def _on_status_changed(self, status: str):
        if hasattr(self, "lbl_engine"):
            state = "idle" if "No Engine" in status or "Not Loaded" in status else "ok"
            set_status_label(self.lbl_engine, f"Active engine: {status}", state)
        if (
            self._endpoints is not None
            and not getattr(self, "_budget_user_touched", False)
            and hasattr(self, "ctx_budget_input")
        ):
            self._set_auto_budget_value(getattr(self._endpoints, "ctx_value", AUTO_CONTEXT_DEFAULT), mark_touched=False)

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

        root.addWidget(self._section_label("SAVED PROJECT TASKS"))
        resume_card = self._card()
        rc = QVBoxLayout(resume_card)
        rc.setContentsMargins(16, 12, 16, 12)
        rc.setSpacing(8)
        self.resume_jobs = QListWidget()
        self.resume_jobs.setObjectName("labs_resume_jobs")
        self.resume_jobs.setFixedHeight(96)
        rc.addWidget(self.resume_jobs)
        resume_row = QHBoxLayout()
        resume_row.setSpacing(8)
        self.btn_refresh_jobs = QPushButton("Refresh")
        set_button_icon(self.btn_refresh_jobs, "refresh-cw", "Refresh")
        self.btn_refresh_jobs.setFixedHeight(28)
        self.btn_refresh_jobs.clicked.connect(self._refresh_resume_jobs)
        self.btn_resume_selected = QPushButton("Resume Selected")
        set_button_icon(self.btn_resume_selected, "rotate-ccw", "Resume Selected")
        self.btn_resume_selected.setFixedHeight(28)
        self.btn_resume_selected.clicked.connect(self._resume_selected_job)
        resume_row.addWidget(self.btn_refresh_jobs)
        resume_row.addWidget(self.btn_resume_selected)
        resume_row.addStretch()
        rc.addLayout(resume_row)
        root.addWidget(resume_card)
        root.addSpacing(14)

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

        self.chk_globals = ToggleSwitch("Include module-level (global) functions")
        oc.addWidget(self.chk_globals)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self.btn_ctx_none = QPushButton("No reset")
        self.btn_ctx_fixed = QPushButton("Fixed reset")
        self.btn_ctx_auto = QPushButton("Auto budget")
        for btn in (self.btn_ctx_none, self.btn_ctx_fixed, self.btn_ctx_auto):
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            mode_row.addWidget(btn)
        mode_row.addStretch()
        oc.addLayout(mode_row)

        self._context_policy = CONTEXT_POLICY_FIXED
        self._budget_user_touched = False
        self.btn_ctx_none.clicked.connect(lambda: self._set_context_policy(CONTEXT_POLICY_NONE))
        self.btn_ctx_fixed.clicked.connect(lambda: self._set_context_policy(CONTEXT_POLICY_FIXED))
        self.btn_ctx_auto.clicked.connect(lambda: self._set_context_policy(CONTEXT_POLICY_AUTO))

        self._fixed_context_box = QWidget()
        fixed_layout = QVBoxLayout(self._fixed_context_box)
        fixed_layout.setContentsMargins(0, 0, 0, 0)
        fixed_layout.setSpacing(6)
        self.chk_reset_fn = ToggleSwitch("Reset LLM context after each function")
        self.chk_reset_cls = ToggleSwitch("Reset LLM context after each class")
        self.chk_reset_fn.setChecked(True)
        fixed_layout.addWidget(self.chk_reset_fn)
        fixed_layout.addWidget(self.chk_reset_cls)
        oc.addWidget(self._fixed_context_box)

        self._auto_context_box = QWidget()
        auto_layout = QVBoxLayout(self._auto_context_box)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        auto_layout.setSpacing(6)
        auto_row = QHBoxLayout()
        auto_row.setSpacing(8)
        auto_lbl = QLabel("Context budget:")
        auto_lbl.setObjectName("txt2")
        auto_lbl.setFixedWidth(110)
        self.ctx_budget_slider = QSlider(Qt.Orientation.Horizontal)
        self.ctx_budget_slider.setRange(AUTO_CONTEXT_MIN, AUTO_CONTEXT_MAX)
        self.ctx_budget_slider.setSingleStep(512)
        self.ctx_budget_slider.setPageStep(4096)
        self.ctx_budget_input = QLineEdit(str(AUTO_CONTEXT_DEFAULT))
        self.ctx_budget_input.setFixedHeight(30)
        self.ctx_budget_input.setFixedWidth(96)
        self.ctx_budget_slider.valueChanged.connect(self._sync_budget_from_slider)
        self.ctx_budget_input.editingFinished.connect(self._sync_budget_from_input)
        auto_row.addWidget(auto_lbl)
        auto_row.addWidget(self.ctx_budget_slider, 1)
        auto_row.addWidget(self.ctx_budget_input)
        auto_layout.addLayout(auto_row)
        auto_note = QLabel("Approximate tokens, py-to-doc history only")
        auto_note.setObjectName("txt2_small")
        auto_layout.addWidget(auto_note)

        self.chk_auto_reload_model = ToggleSwitch("Auto model reload when free RAM is below")
        self.chk_auto_reload_model.toggled.connect(self._sync_reload_ram_enabled)
        auto_layout.addWidget(self.chk_auto_reload_model)

        reload_row = QHBoxLayout()
        reload_row.setSpacing(8)
        reload_lbl = QLabel("Reload below:")
        reload_lbl.setObjectName("txt2")
        reload_lbl.setFixedWidth(110)
        self.reload_ram_gb = QSpinBox()
        self.reload_ram_gb.setRange(0, 1024)
        self.reload_ram_gb.setSuffix(" GB")
        self.reload_ram_gb.setValue(AUTO_RELOAD_RAM_GB_DEFAULT)
        self.reload_ram_gb.setFixedHeight(30)
        self.reload_ram_mb = QSpinBox()
        self.reload_ram_mb.setRange(0, 1023)
        self.reload_ram_mb.setSuffix(" MB")
        self.reload_ram_mb.setValue(AUTO_RELOAD_RAM_MB_DEFAULT)
        self.reload_ram_mb.setFixedHeight(30)
        reload_row.addWidget(reload_lbl)
        reload_row.addWidget(self.reload_ram_gb)
        reload_row.addWidget(self.reload_ram_mb)
        reload_row.addStretch()
        auto_layout.addLayout(reload_row)
        self._sync_reload_ram_enabled(False)
        oc.addWidget(self._auto_context_box)
        self._set_auto_budget_value(AUTO_CONTEXT_DEFAULT, mark_touched=False)
        self._set_context_policy(CONTEXT_POLICY_FIXED)

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
        self.btn_resume = QPushButton("Resume Project")
        set_button_icon(self.btn_resume, "rotate-ccw", "Resume Project")
        self.btn_resume.setFixedHeight(38)
        self.btn_resume.setVisible(False)
        self.btn_resume.clicked.connect(lambda: self._run_py_to_doc(resume_requested=True))
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
        btn_row.addWidget(self.btn_resume)
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
        self._refresh_resume_jobs()

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

    # ── saved project task helpers ───────────────────────────────────────────
    def _active_resume_state_paths(self) -> set[str]:
        worker = getattr(self, "_worker", None)
        if worker is None:
            return set()
        try:
            if not worker.isRunning():
                return set()
        except Exception:
            return set()
        paths = []
        for attr in ("_state_path",):
            value = getattr(worker, attr, None)
            if value:
                paths.append(Path(value))
        try:
            if worker._project_enabled():
                paths.append(worker._job_state_path())
        except Exception:
            pass
        paths.append(PYTODOC_TEMP_STATE)
        out: set[str] = set()
        for path in paths:
            try:
                out.add(str(path.resolve()))
            except Exception:
                pass
        return out

    def _read_resume_state(
        self,
        path: Path,
        *,
        recover_running: bool = True,
    ) -> Optional[dict[str, Any]]:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if state.get("feature") != "py_to_doc":
            return None
        if recover_running:
            try:
                active_paths = self._active_resume_state_paths()
                state, _ = PyToDocWorker.recover_state_file_for_resume(
                    path,
                    state,
                    recover_running=str(path.resolve()) not in active_paths,
                )
            except Exception as exc:
                self._log(f"Resume recovery skipped for {path.name}: {exc}")
        state["_state_path"] = str(path)
        return state

    def _saved_job_states(self) -> list[dict[str, Any]]:
        paths: list[Path] = []
        if PYTODOC_TEMP_STATE.exists():
            paths.append(PYTODOC_TEMP_STATE)
        if PYTODOC_JOBS_DIR.exists():
            paths.extend(sorted(PYTODOC_JOBS_DIR.glob("*.json")))
        states: list[dict[str, Any]] = []
        seen: set[str] = set()
        for path in paths:
            state = self._read_resume_state(path)
            if not state:
                continue
            key = state.get("project_key") or str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            states.append(state)
        states.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return states

    @staticmethod
    def _resume_job_label(state: dict[str, Any]) -> str:
        root = Path(state.get("project_root", "")).name or "project"
        out = Path(state.get("out_path", "")).name or "output"
        status = state.get("status", "unknown")
        if status == "paused" and state.get("recovered_at"):
            status = "paused/recovered"
        elif status == "paused" and state.get("last_error"):
            status = "paused/error"
        files = state.get("file_list", []) or []
        complete = len(state.get("completed_files", []) or [])
        updated = state.get("updated_at", "")
        return f"{root} -> {out}  [{status}]  {complete}/{len(files)} files  {updated}"

    def _refresh_resume_jobs(self):
        if not hasattr(self, "resume_jobs"):
            return
        self.resume_jobs.clear()
        states = self._saved_job_states()
        if not states:
            item = QListWidgetItem("No saved py-to-doc project tasks")
            item.setData(Qt.ItemDataRole.UserRole, "")
            self.resume_jobs.addItem(item)
            self.btn_resume_selected.setEnabled(False)
            return
        self.btn_resume_selected.setEnabled(True)
        for state in states:
            item = QListWidgetItem(self._resume_job_label(state))
            item.setData(Qt.ItemDataRole.UserRole, state.get("_state_path", ""))
            item.setToolTip(
                f"Project: {state.get('project_root', '')}\n"
                f"Output: {state.get('out_path', '')}"
            )
            self.resume_jobs.addItem(item)
        self.resume_jobs.setCurrentRow(0)

    def _apply_resume_state_to_ui(self, state: dict[str, Any]):
        self._set_mode("project")
        self.inp_project_src.setText(state.get("project_root", ""))
        self.inp_out_dir.setText(state.get("out_path", ""))
        settings = state.get("settings") or {}
        if settings:
            self.chk_globals.setChecked(bool(settings.get("include_globals", False)))
            self._set_context_policy(settings.get("context_policy", CONTEXT_POLICY_FIXED))
            self.chk_reset_fn.setChecked(bool(settings.get("fixed_reset_per_function", True)))
            self.chk_reset_cls.setChecked(bool(settings.get("fixed_reset_per_class", False)))
            self._set_auto_budget_value(
                int(settings.get("auto_context_tokens", AUTO_CONTEXT_DEFAULT) or AUTO_CONTEXT_DEFAULT),
                mark_touched=True,
            )
            threshold_mb = int(settings.get("auto_reload_free_ram_mb", 0) or 0)
            self.chk_auto_reload_model.setChecked(bool(settings.get("auto_model_reload", False)) and threshold_mb > 0)
            self.reload_ram_gb.setValue(max(0, threshold_mb // 1024))
            self.reload_ram_mb.setValue(max(0, threshold_mb % 1024))
            self._sync_reload_ram_enabled(self.chk_auto_reload_model.isChecked())
            self.inp_prompt_overview.setPlainText(settings.get("prompt_overview", DEFAULT_OVERVIEW_PROMPT))
            self.inp_prompt_class.setPlainText(settings.get("prompt_class", DEFAULT_CLASS_PROMPT))
            self.inp_prompt_function.setPlainText(settings.get("prompt_function", DEFAULT_FUNC_PROMPT))

    def _resume_selected_job(self):
        item = self.resume_jobs.currentItem() if hasattr(self, "resume_jobs") else None
        if not item:
            return
        state_path = item.data(Qt.ItemDataRole.UserRole) or ""
        if not state_path:
            return
        state = self._read_resume_state(Path(state_path), recover_running=True)
        if not state:
            QMessageBox.warning(self, "Resume Failed", "Could not read the selected py-to-doc task.")
            self._refresh_resume_jobs()
            return
        self._resume_state_path = state_path
        self._apply_resume_state_to_ui(state)
        self._run_py_to_doc(resume_requested=True)

    # ── context option helpers ───────────────────────────────────────────────
    def _set_context_policy(self, policy: str):
        if policy not in {CONTEXT_POLICY_NONE, CONTEXT_POLICY_FIXED, CONTEXT_POLICY_AUTO}:
            policy = CONTEXT_POLICY_FIXED
        self._context_policy = policy
        self.btn_ctx_none.setChecked(policy == CONTEXT_POLICY_NONE)
        self.btn_ctx_fixed.setChecked(policy == CONTEXT_POLICY_FIXED)
        self.btn_ctx_auto.setChecked(policy == CONTEXT_POLICY_AUTO)
        self._fixed_context_box.setVisible(policy == CONTEXT_POLICY_FIXED)
        self._auto_context_box.setVisible(policy == CONTEXT_POLICY_AUTO)

    def _set_auto_budget_value(self, value: int, *, mark_touched: bool = True):
        value = max(AUTO_CONTEXT_MIN, int(value or AUTO_CONTEXT_DEFAULT))
        slider_value = min(value, AUTO_CONTEXT_MAX)
        self.ctx_budget_input.blockSignals(True)
        self.ctx_budget_slider.blockSignals(True)
        self.ctx_budget_input.setText(str(value))
        self.ctx_budget_slider.setValue(slider_value)
        self.ctx_budget_input.blockSignals(False)
        self.ctx_budget_slider.blockSignals(False)
        if mark_touched:
            self._budget_user_touched = True

    def _sync_budget_from_slider(self, value: int):
        self._set_auto_budget_value(int(value), mark_touched=True)

    def _sync_budget_from_input(self):
        raw = self.ctx_budget_input.text().strip()
        try:
            value = int(raw)
        except Exception:
            value = AUTO_CONTEXT_DEFAULT
        self._set_auto_budget_value(value, mark_touched=True)

    def _context_budget_value(self) -> int:
        try:
            return max(AUTO_CONTEXT_MIN, int(self.ctx_budget_input.text().strip()))
        except Exception:
            self._set_auto_budget_value(AUTO_CONTEXT_DEFAULT, mark_touched=True)
            return AUTO_CONTEXT_DEFAULT

    def _sync_reload_ram_enabled(self, enabled: bool):
        if hasattr(self, "reload_ram_gb"):
            self.reload_ram_gb.setEnabled(bool(enabled))
        if hasattr(self, "reload_ram_mb"):
            self.reload_ram_mb.setEnabled(bool(enabled))

    def _auto_reload_threshold_mb(self) -> int:
        if (
            self._context_policy != CONTEXT_POLICY_AUTO
            or not getattr(self, "chk_auto_reload_model", None)
            or not self.chk_auto_reload_model.isChecked()
        ):
            return 0
        gb = int(self.reload_ram_gb.value()) if hasattr(self, "reload_ram_gb") else 0
        mb = int(self.reload_ram_mb.value()) if hasattr(self, "reload_ram_mb") else 0
        return max(0, (gb * 1024) + mb)

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
        if hasattr(self, "btn_resume"):
            self.btn_resume.setVisible(mode == "project" and not bool(self._worker))

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
        self.btn_resume.setVisible((not running) and self._mode == "project")
        self.btn_abort.setVisible(running)
        self.btn_pause.setVisible(running and self._mode == "project")

    def _run_py_to_doc(self, resume_requested: bool = False):
        mode     = self._mode
        out_dir  = self.inp_out_dir.text().strip()
        out_name = self.inp_out_name.text().strip() or "README.md"
        file_list:    list[str]    = []
        project_root: Optional[str] = None
        mirrored_dirs = 0
        state_path = self._resume_state_path if resume_requested else ""
        if not resume_requested:
            self._resume_state_path = ""

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
            file_list = discover_project_python_files(proj)
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

        if self._endpoints is None:
            QMessageBox.warning(
                self, "No Engine",
                "No LLM engine is loaded. Load a model (or connect an API "
                "model) from the main tabs first.")
            return
        if not self._endpoints.is_loaded and not self._endpoints.is_loading:
            QMessageBox.warning(
                self, "No Engine",
                "No LLM engine is loaded. Load a model (or connect an API "
                "model) from the main tabs first.")
            return
        waiting_for_initial_load = self._endpoints.is_loading

        auto_reload_threshold_mb = self._auto_reload_threshold_mb()
        auto_reload_enabled = (
            self._context_policy == CONTEXT_POLICY_AUTO
            and self.chk_auto_reload_model.isChecked()
        )
        if auto_reload_enabled and auto_reload_threshold_mb <= 0:
            QMessageBox.warning(
                self,
                "Invalid RAM Threshold",
                "Set the auto model reload RAM threshold above 0 MB."
            )
            return

        if mode == "project":
            try:
                mirrored_dirs = mirror_project_directories(project_root or "", out_dir)
            except Exception as exc:
                QMessageBox.warning(
                    self, "Output Mirror Failed",
                    f"Could not prepare the output directory structure:\n{exc}")
                return

        if (
            self._context_policy == CONTEXT_POLICY_AUTO
            and self._endpoints.is_local_active
        ):
            target_ctx = self._context_budget_value()
            current_ctx = int(getattr(self._endpoints, "ctx_value", 0) or 0)
            if target_ctx != current_ctx:
                self._log(
                    f"Reloading local model/server for py-to-doc context: "
                    f"{current_ctx:,} -> {target_ctx:,}"
                )
                if not self._endpoints.request_context(target_ctx):
                    QMessageBox.warning(
                        self, "Context Reload Failed",
                        "Could not reload the local model/server with the "
                        f"selected py-to-doc context ({target_ctx:,})."
                    )
                    return
                self._log("Context reload complete")

        if self._endpoints.is_local_active:
            self._endpoints.ensure_server(log_cb=lambda m: self._log(m))

        self.log_te.clear()
        self.preview_te.clear()
        self._set_running(True)
        if waiting_for_initial_load:
            self._log("Model is still loading; py-to-doc will wait before the first LLM call")
        if mode == "project":
            self._log(f"Project files selected: {len(file_list)}")
            self._log(f"Output directory structure prepared: {mirrored_dirs} directories")

        self._worker = PyToDocWorker(
            file_path          = src,
            out_path           = out_dir,
            out_name           = out_name,
            include_globals    = self.chk_globals.isChecked(),
            context_policy     = self._context_policy,
            fixed_reset_per_function = self.chk_reset_fn.isChecked(),
            fixed_reset_per_class    = self.chk_reset_cls.isChecked(),
            auto_context_tokens = self._context_budget_value(),
            auto_model_reload = auto_reload_enabled,
            auto_reload_free_ram_mb = auto_reload_threshold_mb,
            prompt_overview    = self.inp_prompt_overview.toPlainText().strip(),
            prompt_class       = self.inp_prompt_class.toPlainText().strip(),
            prompt_function    = self.inp_prompt_function.toPlainText().strip(),
            endpoints          = self._endpoints,
            file_list          = file_list,
            project_root       = project_root,
            resume_required    = resume_requested,
            state_path         = state_path,
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
        self._refresh_resume_jobs()

    def _on_done(self):
        self._set_running(False)
        self._refresh_resume_jobs()
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
        self._refresh_resume_jobs()
