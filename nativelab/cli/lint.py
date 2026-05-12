"""Lightweight linting for the CLI.

Order of preference (whichever is installed first wins):

  1. `pyflakes`  - fast, no config
  2. `flake8`    - broader checks
  3. `pylint`    - full static analysis

Falls back to `python -m py_compile` for raw syntax checks if no linter is
available. Always supplements with `compile()` to catch syntax errors that
some linters silently ignore.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def _run(cmd: list[str]) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, ""
    except Exception as e:
        return 1, f"[lint runner error: {e}]"


def _which_linter() -> str:
    if shutil.which("pyflakes"): return "pyflakes"
    if shutil.which("flake8"):   return "flake8"
    if shutil.which("pylint"):   return "pylint"
    return ""


def _syntax_check(path: Path) -> List[str]:
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        compile(src, str(path), "exec")
        return []
    except SyntaxError as e:
        return [f"{path}:{e.lineno}:{e.offset or 0}: SyntaxError: {e.msg}"]


def lint_file(path: str) -> dict:
    """Return {"linter": <name>, "issues": [str], "ok": bool}."""
    p = Path(path).expanduser().resolve()
    if not p.exists() or p.suffix != ".py":
        return {"linter": "n/a", "issues": [f"Not a .py file: {p}"], "ok": False}

    issues = _syntax_check(p)
    if issues:
        return {"linter": "compile", "issues": issues, "ok": False}

    linter = _which_linter()
    if not linter:
        return {"linter": "compile", "issues": [], "ok": True}

    cmd: list[str]
    if linter == "pyflakes":
        cmd = ["pyflakes", str(p)]
    elif linter == "flake8":
        cmd = ["flake8", "--max-line-length=120", str(p)]
    else:  # pylint
        cmd = ["pylint", "--disable=C0114,C0115,C0116", "--score=no", str(p)]

    code, out = _run(cmd)
    raw_issues = [ln for ln in out.splitlines() if ln.strip()]
    # pylint exits non-zero even for warnings; treat empty stdout as clean
    has_issues = bool(raw_issues) and not (linter == "pylint" and code == 0)
    return {
        "linter": linter,
        "issues": raw_issues,
        "ok":     not has_issues,
    }


def lint_paths(paths: List[str]) -> int:
    """Lint each path; print results; return overall exit code (0 = clean)."""
    overall = 0
    for path in paths:
        result = lint_file(path)
        header = f"[{result['linter']}]  {path}"
        if result["ok"]:
            print(f"\033[32mOK\033[0m  {header}  - clean")
            continue
        overall = 1
        print(f"\033[33mWARN\033[0m  {header}")
        for line in result["issues"]:
            print(f"    {line}")
    return overall


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: nativelab --cli lint <file.py> [<file.py> ...]",
              file=sys.stderr)
        return 2
    return lint_paths(argv)
