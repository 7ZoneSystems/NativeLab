from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from PhonoLab.config import MobileConfig, load_config
from PhonoLab.llama_cpp_setup import find_llama_cli
from PhonoLab.registry import MobileModelConfig, get_registry
from PhonoLab.safety import SafetyError, explain_error, guard_prompt, sampler_args, validate_model_path


TokenCallback = Callable[[str], None]


class MobileLlamaCppEngine:
    def __init__(self, config: MobileConfig | None = None):
        self.config = config or load_config()
        self.model: MobileModelConfig | None = None
        self.llama_cli: Path | None = None
        self._proc: subprocess.Popen | None = None
        self._abort = threading.Event()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.llama_cli is not None

    def load(self, model_path: str = "") -> bool:
        target = model_path or self.config.active_model
        if not target:
            raise SafetyError("No mobile model is selected.")
        path = validate_model_path(target)
        cli = find_llama_cli(self.config)
        if cli is None:
            raise SafetyError("llama.cpp runtime is missing. Run PhonoLab setup first.")
        registry = get_registry()
        cfg = registry.get(str(path)) or registry.add(path, set_active=True)
        self.model = cfg
        self.llama_cli = cli
        self.config.active_model = str(path)
        self.config.ctx = cfg.ctx
        self.config.threads = cfg.threads
        self.config.n_predict = cfg.n_predict
        return True

    def unload(self) -> None:
        self.abort()
        self.model = None

    def abort(self) -> None:
        self._abort.set()
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _command(self, prompt: str) -> list[str]:
        if self.model is None or self.llama_cli is None:
            raise SafetyError("No model is loaded.")
        args = sampler_args(self.config)
        return [
            str(self.llama_cli),
            "-m",
            self.model.path,
            "-t",
            str(args["threads"]),
            "--ctx-size",
            str(args["ctx"]),
            "-n",
            str(args["n_predict"]),
            "--temp",
            str(args["temperature"]),
            "--top-p",
            str(args["top_p"]),
            "--repeat-penalty",
            str(args["repeat_penalty"]),
            "--no-display-prompt",
            "--no-escape",
            "-p",
            prompt,
        ]

    def generate(self, prompt: str, token_cb: TokenCallback | None = None) -> str:
        guard_prompt(prompt, self.config)
        self._abort.clear()
        cmd = self._command(prompt)
        started = time.monotonic()
        pieces: list[str] = []
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert self._proc.stdout is not None
            while True:
                if self._abort.is_set():
                    raise SafetyError("Generation was cancelled.")
                if time.monotonic() - started > self.config.stream_timeout_seconds:
                    raise TimeoutError("llama.cpp generation timed out.")
                chunk = self._proc.stdout.read(1)
                if chunk:
                    pieces.append(chunk)
                    if token_cb:
                        token_cb(chunk)
                    continue
                if self._proc.poll() is not None:
                    break
                time.sleep(0.01)
            stderr = ""
            if self._proc.stderr is not None:
                stderr = self._proc.stderr.read()
            code = self._proc.wait(timeout=2)
            if code != 0:
                notice = explain_error(stderr or f"llama.cpp exited with code {code}")
                raise RuntimeError(notice.user_message)
            return "".join(pieces).strip()
        except Exception:
            self.abort()
            raise
        finally:
            self._proc = None
