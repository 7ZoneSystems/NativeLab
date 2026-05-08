"""
Labs ⇄ App endpoint layer.

This module is the *only* surface that lab features should use to talk to the
rest of the application. Every lab panel receives a `LabEndpoints` instance and
goes through it for:

  • Read state - server status, current model, ctx size, mode, family
  • LLM calls  - synchronous `call_llm(...)` that picks API > server > CLI
  • Reverse routing - request the host app to change context, load a model,
    unload, ensure the server is up, etc.

Lab features therefore stay decoupled from `MainWindow` internals: they don't
need to know whether the active backend is a local llama-server, llama-cli, or
a remote API provider. New labs simply read `endpoints.snapshot()` and call
`endpoints.call_llm(...)`.
"""
from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from nativelab.GlobalConfig.config_global import (
    DEFAULT_CTX, DEFAULT_THREADS, DEFAULT_N_PRED,
)
from nativelab.components.components_global import detect_model_family
from nativelab.Model.model_global import FAMILY_TEMPLATES
from nativelab.core.engine_global import LlamaEngine, ApiEngine


# ─────────────────────────────────────────────────────────────────────────────
#  LabEndpoints
# ─────────────────────────────────────────────────────────────────────────────

class LabEndpoints(QObject):
    """Shared experimentation surface for the Labs tab.

    Wired once by the host application (typically `MainWindow`) via
    `bind_engines(...)` and `bind_reverse_routes(...)`. Lab panels then receive
    the same instance through `LabsTab.set_endpoints(endpoints)` and never need
    to import engines directly.
    """

    engine_changed = pyqtSignal()
    status_changed = pyqtSignal(str)
    log_msg        = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._llama_provider: Callable[[], Optional[LlamaEngine]] = lambda: None
        self._api_provider:   Callable[[], Optional[ApiEngine]]   = lambda: None
        self._on_context_request: Callable[[int], bool] = lambda c: False
        self._on_model_request:   Callable[[str], bool] = lambda p: False
        self._on_unload_request:  Callable[[], None]    = lambda: None

    # ── wiring (host app) ────────────────────────────────────────────────────
    def bind_engines(
        self,
        llama_provider: Callable[[], Optional[LlamaEngine]],
        api_provider:   Callable[[], Optional[ApiEngine]],
    ) -> None:
        self._llama_provider = llama_provider
        self._api_provider   = api_provider
        self.notify_engine_changed()

    def bind_reverse_routes(
        self,
        on_context: Optional[Callable[[int], bool]] = None,
        on_model:   Optional[Callable[[str], bool]] = None,
        on_unload:  Optional[Callable[[], None]]    = None,
    ) -> None:
        if on_context is not None: self._on_context_request = on_context
        if on_model   is not None: self._on_model_request   = on_model
        if on_unload  is not None: self._on_unload_request  = on_unload

    def notify_engine_changed(self) -> None:
        self.engine_changed.emit()
        self.status_changed.emit(self.status_text)

    # ── engines / state ──────────────────────────────────────────────────────
    @property
    def llama_engine(self) -> Optional[LlamaEngine]:
        return self._llama_provider()

    @property
    def api_engine(self) -> Optional[ApiEngine]:
        return self._api_provider()

    def active_engine(self):
        api = self.api_engine
        if api is not None and api.is_loaded:
            return api
        return self.llama_engine

    @property
    def is_loaded(self) -> bool:
        eng = self.active_engine()
        return bool(eng and eng.is_loaded)

    @property
    def status_text(self) -> str:
        eng = self.active_engine()
        return eng.status_text if eng else "⚪ No Engine"

    @property
    def model_path(self) -> str:
        eng = self.active_engine()
        return getattr(eng, "model_path", "") if eng else ""

    @property
    def model_name(self) -> str:
        p = self.model_path
        return Path(p).name if p else ""

    @property
    def mode(self) -> str:
        eng = self.active_engine()
        return getattr(eng, "mode", "unloaded") if eng else "unloaded"

    @property
    def ctx_value(self) -> int:
        eng = self.active_engine()
        return getattr(eng, "ctx_value", DEFAULT_CTX()) if eng else DEFAULT_CTX()

    @property
    def server_port(self) -> int:
        eng = self.active_engine()
        return getattr(eng, "server_port", 0) if eng else 0

    def model_family(self):
        return detect_model_family(self.model_path) if self.model_path \
               else FAMILY_TEMPLATES.get("mistral")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "status":      self.status_text,
            "is_loaded":   self.is_loaded,
            "mode":        self.mode,
            "model_path":  self.model_path,
            "model_name":  self.model_name,
            "ctx_value":   self.ctx_value,
            "server_port": self.server_port,
            "backend":     "api" if (self.api_engine and self.api_engine.is_loaded)
                           else self.mode,
        }

    # ── reverse routing ──────────────────────────────────────────────────────
    def request_context(self, new_ctx: int) -> bool:
        return bool(self._on_context_request(int(new_ctx)))

    def request_load_model(self, model_path: str) -> bool:
        return bool(self._on_model_request(str(model_path)))

    def request_unload(self) -> None:
        self._on_unload_request()

    def ensure_server(self, log_cb=None) -> bool:
        eng = self.active_engine()
        if eng is None:
            return False
        if hasattr(eng, "ensure_server"):
            return bool(eng.ensure_server(log_cb=log_cb))
        return bool(eng.is_loaded)

    # ── LLM call (sync; safe to invoke from a QThread) ───────────────────────
    def call_llm(
        self,
        messages:       Optional[List[Dict[str, str]]] = None,
        prompt:         Optional[str] = None,
        system_prompt:  Optional[str] = None,
        n_predict:      int   = DEFAULT_N_PRED,
        temperature:    float = 0.3,
        top_p:          float = 0.9,
        repeat_penalty: float = 1.15,
    ) -> str:
        msgs: List[Dict[str, str]] = list(messages or [])
        if system_prompt:
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        if prompt is not None:
            msgs.append({"role": "user", "content": prompt})
        if not msgs:
            raise RuntimeError("call_llm: no messages or prompt provided")

        api = self.api_engine
        if api and api.is_loaded:
            return self._call_api(api, msgs, n_predict, temperature)

        llama = self.llama_engine
        if llama and llama.is_loaded:
            if llama.mode == "server":
                return self._call_server(
                    llama, msgs, n_predict, temperature, top_p, repeat_penalty
                )
            return self._call_cli(
                llama, msgs, n_predict, temperature, repeat_penalty
            )

        raise RuntimeError("No engine loaded")

    # ── internals ────────────────────────────────────────────────────────────
    def _build_text_prompt(self, messages: List[Dict[str, str]],
                           model_path: str) -> str:
        fam = detect_model_family(model_path) if model_path \
              else FAMILY_TEMPLATES.get("mistral")
        out: List[str] = []
        sys_buf = ""
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                sys_buf += content + "\n"
            elif role == "user":
                u = (sys_buf + content) if sys_buf else content
                sys_buf = ""
                out.append(getattr(fam, "user_prefix", "") + u
                           + getattr(fam, "user_suffix", ""))
            elif role == "assistant":
                out.append(getattr(fam, "assistant_prefix", "") + content
                           + getattr(fam, "assistant_suffix", ""))
        out.append(getattr(fam, "assistant_prefix", ""))
        return getattr(fam, "bos", "") + "".join(out)

    def _call_server(self, eng, messages, n_predict,
                     temperature, top_p, repeat_penalty) -> str:
        import http.client
        prompt_text = self._build_text_prompt(messages, eng.model_path)
        fam = detect_model_family(eng.model_path)
        body = json.dumps({
            "prompt":         prompt_text,
            "n_predict":      n_predict,
            "stream":         False,
            "temperature":    temperature,
            "top_p":          top_p,
            "repeat_penalty": repeat_penalty,
            "stop":           getattr(fam, "stop_tokens", []),
        })
        timeout = min(120 + len(prompt_text) // 100, 900)
        conn = http.client.HTTPConnection(
            "127.0.0.1", eng.server_port, timeout=timeout
        )
        conn.request("POST", "/completion", body,
                     {"Content-Type": "application/json"})
        r = conn.getresponse()
        if r.status != 200:
            raise RuntimeError(f"llama-server HTTP {r.status}")
        d = json.loads(r.read().decode("utf-8", errors="replace"))
        return (d.get("content") or "").strip()

    def _call_cli(self, eng, messages, n_predict,
                  temperature, repeat_penalty) -> str:
        import nativelab.GlobalConfig.binaryResolve as _binres
        from nativelab.Server.server_global import SERVER_CONFIG

        prompt_text = self._build_text_prompt(messages, eng.model_path)
        cli_bin = SERVER_CONFIG.cli_path or _binres.LLAMA_CLI
        cmd = [
            cli_bin, "-m", eng.model_path,
            "-t", str(DEFAULT_THREADS()),
            "--ctx-size", str(getattr(eng, "ctx_value", DEFAULT_CTX())),
            "-n", str(n_predict),
            "--no-display-prompt", "--no-escape",
            "--temp", str(temperature),
            "--repeat-penalty", str(repeat_penalty),
            "-p", prompt_text,
        ]
        timeout = min(120 + len(prompt_text) // 100, 900)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
        return result.stdout.decode("utf-8", errors="replace").strip()

    def _call_api(self, api: ApiEngine, messages, n_predict,
                  temperature) -> str:
        cfg = api._config
        if cfg is None:
            raise RuntimeError("API config not loaded")
        max_tokens = min(n_predict, cfg.max_tokens)

        if cfg.api_format == "anthropic":
            sys_text = ""
            chat: List[Dict[str, str]] = []
            for m in messages:
                if m["role"] == "system":
                    sys_text += m["content"] + "\n"
                else:
                    chat.append({"role": m["role"], "content": m["content"]})
            payload = {
                "model":       cfg.model_id,
                "messages":    chat or [{"role": "user", "content": ""}],
                "max_tokens":  max_tokens,
                "temperature": temperature,
                "stream":      False,
            }
            if sys_text.strip():
                payload["system"] = sys_text.strip()
            req = urllib.request.Request(
                f"{cfg.base_url.rstrip('/')}/v1/messages",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type":      "application/json",
                    "x-api-key":         cfg.api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read().decode("utf-8", errors="replace"))
            content = d.get("content") or []
            if isinstance(content, list):
                return "".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                ).strip()
            return str(content).strip()

        # OpenAI-compatible
        payload = {
            "model":       cfg.model_id,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      False,
        }
        req = urllib.request.Request(
            f"{cfg.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read().decode("utf-8", errors="replace"))
        choices = d.get("choices") or []
        if choices:
            return (choices[0].get("message", {}).get("content") or "").strip()
        return ""
