"""Helpers for checking and installing the HF Transformers runtime stack."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
from typing import Iterable, Sequence

from nativelab.imports.import_global import QThread, pyqtSignal


HF_TRANSFORMERS_DEP_PACKAGES = (
    "transformers>=5.0.0",
    "torch>=2.2.0",
    "safetensors>=0.4.0",
    "accelerate>=0.30.0",
    "sentencepiece>=0.2.0",
    "pillow>=12.2.0",
    "requests>=2.33.0",
    "urllib3>=2.7.0",
    "filelock>=3.20.3",
)

HF_TRANSFORMERS_DEP_IMPORTS = (
    ("transformers", "transformers"),
    ("torch", "torch"),
    ("safetensors", "safetensors"),
    ("accelerate", "accelerate"),
    ("sentencepiece", "sentencepiece"),
    ("pillow", "PIL"),
    ("requests", "requests"),
    ("urllib3", "urllib3"),
    ("filelock", "filelock"),
)


def build_hf_transformers_install_command(
    python_executable: str = "",
    packages: Sequence[str] | None = None,
) -> list[str]:
    """Return the exact pip command used by the in-app installer."""
    exe = str(python_executable or sys.executable).strip() or sys.executable
    deps = tuple(packages or HF_TRANSFORMERS_DEP_PACKAGES)
    return [exe, "-m", "pip", "install", "-U", *deps]


def hf_transformers_dependency_report() -> dict:
    """Return installed/missing status for modules required by the HF backend."""
    installed = []
    missing = []
    for package, module in HF_TRANSFORMERS_DEP_IMPORTS:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, AttributeError, ValueError):
            found = False
        (installed if found else missing).append(package)
    return {
        "ok": not missing,
        "installed": installed,
        "missing": missing,
        "packages": list(HF_TRANSFORMERS_DEP_PACKAGES),
    }


def _command_for_log(cmd: Iterable[str]) -> str:
    parts = []
    for value in cmd:
        text = str(value)
        if not text or any(ch.isspace() for ch in text):
            text = '"' + text.replace('"', '\\"') + '"'
        parts.append(text)
    return " ".join(parts)


class HfTransformersDepsWorker(QThread):
    """Install the Python libraries needed by the HF Transformers backend."""

    status = pyqtSignal(str)
    output = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, python_executable: str = "", packages: Sequence[str] | None = None):
        super().__init__()
        self._python_executable = python_executable
        self._packages = tuple(packages or HF_TRANSFORMERS_DEP_PACKAGES)
        self._abort = False
        self._proc = None
        self._lock = threading.Lock()

    def abort(self):
        self._abort = True
        with self._lock:
            proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def run(self):
        cmd = build_hf_transformers_install_command(self._python_executable, self._packages)
        self.status.emit("Installing Hugging Face Transformers libraries...")
        self.output.emit("$ " + _command_for_log(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._proc = proc
            if self._abort and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            if proc.stdout:
                for line in proc.stdout:
                    if self._abort:
                        break
                    text = line.rstrip()
                    if text:
                        self.output.emit(text)
            code = proc.wait()
            if self._abort:
                self.done.emit(False, "HF Transformers library installation cancelled.")
                return
            if code != 0:
                self.done.emit(False, f"pip exited with code {code}.")
                return
            report = hf_transformers_dependency_report()
            if report["ok"]:
                self.done.emit(True, "HF Transformers libraries are installed.")
            else:
                self.done.emit(
                    False,
                    "pip finished, but these imports are still missing: "
                    + ", ".join(report["missing"]),
                )
        except Exception as exc:
            self.done.emit(False, f"Could not install HF Transformers libraries: {exc}")
        finally:
            with self._lock:
                self._proc = None
