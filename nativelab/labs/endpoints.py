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
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from nativelab.imports.qt_compat import QObject, pyqtSignal
from nativelab.GlobalConfig.config_global import (
    DEFAULT_CTX, DEFAULT_THREADS, DEFAULT_N_PRED, LONG_TIMEOUT_SECONDS,
)
from nativelab.components.components_global import detect_model_family
from nativelab.Model.model_global import FAMILY_TEMPLATES, model_ref_display_name, model_ref_payload
from nativelab.core.context_meter import context_meter
from nativelab.core.engine_status import active_engine, active_engine_status, engine_snapshot
from nativelab.core.engine_global import LlamaEngine, ApiEngine
from nativelab.native.engine_helpers import (
    build_text_prompt as native_build_text_prompt,
    error_message as native_error_message,
    is_context_error as native_is_context_error,
)


class LabEndpointError(RuntimeError):
    """Base error raised by the Labs endpoint layer."""


class ContextWindowExceededError(LabEndpointError):
    """Raised when the active backend rejects a prompt for exceeding context."""

    def __init__(self, message: str, *, status: int = 0, raw: str = ""):
        super().__init__(message)
        self.status = int(status or 0)
        self.raw = raw


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
        self._on_reload_request:  Optional[Callable[[], bool]] = None
        self._on_wait_loaded:     Optional[Callable[[int], bool]] = None
        self._on_is_loading:      Optional[Callable[[], bool]] = None
        self._skill_context_provider: Callable[[], str] = lambda: ""

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
        on_reload:  Optional[Callable[[], bool]]    = None,
        on_wait_loaded: Optional[Callable[[int], bool]] = None,
        on_is_loading: Optional[Callable[[], bool]] = None,
    ) -> None:
        if on_context is not None: self._on_context_request = on_context
        if on_model   is not None: self._on_model_request   = on_model
        if on_unload  is not None: self._on_unload_request  = on_unload
        if on_reload  is not None: self._on_reload_request  = on_reload
        if on_wait_loaded is not None: self._on_wait_loaded = on_wait_loaded
        if on_is_loading is not None: self._on_is_loading = on_is_loading

    def set_skill_context_provider(self, provider: Callable[[], str]) -> None:
        self._skill_context_provider = provider or (lambda: "")

    def notify_engine_changed(self) -> None:
        self.engine_changed.emit()
        self.status_changed.emit(self.status_text)

    def _log(self, level: str, msg: str) -> None:
        print(f"[LAB][{level}] {msg}", flush=True)
        self.log_msg.emit(level, msg)

    # ── engines / state ──────────────────────────────────────────────────────
    @property
    def llama_engine(self) -> Optional[LlamaEngine]:
        return self._llama_provider()

    @property
    def api_engine(self) -> Optional[ApiEngine]:
        return self._api_provider()

    def active_engine(self):
        return active_engine(self.llama_engine, self.api_engine)

    @property
    def engine_status(self):
        return active_engine_status(
            self.llama_engine,
            self.api_engine,
            is_loading=self.is_loading,
        )

    @property
    def is_loaded(self) -> bool:
        eng = self.active_engine()
        return bool(eng and eng.is_loaded)

    @property
    def is_api_active(self) -> bool:
        api = self.api_engine
        return bool(api and api.is_loaded)

    @property
    def is_local_active(self) -> bool:
        llama = self.llama_engine
        return bool(llama and llama.is_loaded and not self.is_api_active)

    @property
    def can_reload_active_model(self) -> bool:
        return self.is_local_active

    @property
    def is_loading(self) -> bool:
        if self.is_loaded:
            return False
        if self._on_is_loading is None:
            return False
        try:
            return bool(self._on_is_loading())
        except Exception:
            return False

    @property
    def status_text(self) -> str:
        return self.engine_status.status_text

    @property
    def model_path(self) -> str:
        eng = self.active_engine()
        return getattr(eng, "model_path", "") if eng else ""

    @property
    def model_name(self) -> str:
        p = self.model_path
        return model_ref_display_name(p) if p else ""

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
        return detect_model_family(model_ref_payload(self.model_path) or self.model_path) if self.model_path \
               else FAMILY_TEMPLATES.get("mistral")

    def snapshot(self) -> Dict[str, Any]:
        snap = engine_snapshot(
            self.llama_engine,
            self.api_engine,
            is_loading=self.is_loading,
        )
        snap["can_reload_active_model"] = self.can_reload_active_model
        return snap

    # ── reverse routing ──────────────────────────────────────────────────────
    def request_context(self, new_ctx: int) -> bool:
        return bool(self._on_context_request(int(new_ctx)))

    def request_load_model(self, model_path: str) -> bool:
        return bool(self._on_model_request(str(model_path)))

    def request_unload(self) -> None:
        self._on_unload_request()

    def request_active_model_reload(self) -> bool:
        """Ask the host to restart the active local model, keeping its context."""
        if self.is_api_active:
            return True
        if not self.is_local_active:
            return False
        if self._on_reload_request is not None:
            return bool(self._on_reload_request())
        return bool(self.request_context(self.ctx_value))

    def wait_until_loaded(self, timeout_ms: int = 0) -> bool:
        """Wait for an in-flight host model load to finish without exposing engines."""
        if self.is_loaded:
            return True
        if self._on_wait_loaded is not None:
            try:
                if self._on_wait_loaded(int(timeout_ms or 0)):
                    return True
            except Exception as exc:
                self._log("WARN", f"wait_until_loaded route failed: {exc}")
        return self._poll_loaded(timeout_ms)

    def ensure_server(self, log_cb=None) -> bool:
        eng = self.active_engine()
        if eng is None:
            return False
        if hasattr(eng, "ensure_server"):
            return bool(eng.ensure_server(log_cb=log_cb))
        return bool(eng.is_loaded)

    def _poll_loaded(self, timeout_ms: int = 0) -> bool:
        deadline = None
        if timeout_ms and timeout_ms > 0:
            deadline = time.monotonic() + (timeout_ms / 1000.0)
        while True:
            if self.is_loaded:
                return True
            if deadline is None or time.monotonic() >= deadline:
                return False
            time.sleep(0.25)

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
        top_k:          int   = 40,
        min_p:          float = 0.0,
        typical_p:      float = 1.0,
        seed:           int   = -1,
        image_data:      Optional[List[Dict[str, Any]]] = None,
        context_source:  str = "Labs",
    ) -> str:
        msgs: List[Dict[str, str]] = list(messages or [])
        if system_prompt:
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        if prompt is not None:
            msgs.append({"role": "user", "content": prompt})
        skill_context = self._skill_context_provider()
        if skill_context:
            msgs = [{"role": "system", "content": skill_context}] + msgs
        if not msgs:
            raise RuntimeError("call_llm: no messages or prompt provided")

        api = self.api_engine
        if api and api.is_loaded:
            self._log("INFO", f"Routing → API  {api.model_path}")
            context_meter.report_messages(
                source=context_source,
                engine=api,
                messages=msgs,
                n_predict=n_predict,
            )
            result = self._call_api(api, msgs, n_predict, temperature)
            context_meter.append_output(result)
            return result

        llama = self.llama_engine
        if llama and llama.is_loaded:
            if hasattr(llama, "generate_sync"):
                self._log("INFO", f"Routing → local {getattr(llama, 'mode', 'engine')}  {Path(getattr(llama, 'model_path', '')).name}")
                try:
                    return llama.generate_sync(
                        messages=msgs,
                        image_data=image_data,
                        n_predict=n_predict,
                        temperature=temperature,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        top_k=top_k,
                        min_p=min_p,
                        typical_p=typical_p,
                        seed=seed,
                        context_source=context_source,
                    )
                except Exception as exc:
                    raw = str(exc)
                    if self._is_context_error(raw):
                        raise ContextWindowExceededError(
                            f"{getattr(llama, 'mode', 'local')} context exceeded: {raw}",
                            raw=raw,
                        )
                    raise
            if llama.mode == "server":
                self._log("INFO", f"Routing → llama-server  port={llama.server_port}")
                return self._call_server(
                    llama, msgs, n_predict, temperature, top_p, repeat_penalty,
                    top_k, min_p, typical_p, seed, context_source,
                )
            self._log("INFO", f"Routing → llama-cli  {Path(llama.model_path).name}")
            return self._call_cli(
                llama, msgs, n_predict, temperature, top_p, repeat_penalty,
                top_k, min_p, typical_p, seed, context_source,
            )

        self._log("ERROR", "call_llm: no engine loaded")
        raise RuntimeError("No engine loaded")

    # ── internals ────────────────────────────────────────────────────────────
    def _build_text_prompt(self, messages: List[Dict[str, str]],
                           model_path: str) -> str:
        fam = detect_model_family(model_ref_payload(model_path) or model_path) if model_path \
              else FAMILY_TEMPLATES.get("mistral")
        return native_build_text_prompt(messages, fam)

    def _call_server(self, eng, messages, n_predict,
                     temperature, top_p, repeat_penalty,
                     top_k=40, min_p=0.0, typical_p=1.0, seed=-1,
                     context_source: str = "Labs") -> str:
        import http.client
        prompt_text = self._build_text_prompt(messages, eng.model_path)
        context_meter.report_prompt(
            source=context_source,
            engine=eng,
            prompt=prompt_text,
            n_predict=n_predict,
        )
        fam = detect_model_family(eng.model_path)
        body_obj = {
            "prompt":         prompt_text,
            "n_predict":      n_predict,
            "stream":         False,
            "temperature":    temperature,
            "top_p":          top_p,
            "repeat_penalty": repeat_penalty,
            "stop":           getattr(fam, "stop_tokens", []),
        }
        body_obj.update(LlamaEngine._sampler_payload(top_k=top_k, min_p=min_p, typical_p=typical_p, seed=seed))
        body = json.dumps(body_obj)
        conn = http.client.HTTPConnection(
            "127.0.0.1", eng.server_port, timeout=LONG_TIMEOUT_SECONDS
        )
        conn.request("POST", "/completion", body,
                     {"Content-Type": "application/json"})
        r = conn.getresponse()

        raw = r.read().decode("utf-8", errors="replace")

        if r.status != 200:
            self._log("ERROR", f"llama-server HTTP {r.status} — {raw[:120]}")
            self._raise_backend_error(r.status, raw, "llama-server")

        try:
            d = json.loads(raw)
        except Exception:
            raise RuntimeError(
                f"Invalid JSON returned by llama-server:\n\n{raw}"
            )

        result = (d.get("content") or "").strip()
        context_meter.append_output(result)
        return result

    @staticmethod
    def _error_message(raw: str) -> str:
        return native_error_message(raw)

    @staticmethod
    def _is_context_error(raw: str) -> bool:
        return native_is_context_error(raw)

    def _raise_backend_error(self, status: int, raw: str, backend: str):
        message = self._error_message(raw)
        if self._is_context_error(raw):
            raise ContextWindowExceededError(
                f"{backend} context exceeded: {message}",
                status=status,
                raw=raw,
            )
        raise LabEndpointError(
            f"{backend} HTTP {status}\n\n"
            f"Response body:\n{raw}"
        )

    def _call_cli(self, eng, messages, n_predict,
                  temperature, top_p, repeat_penalty,
                  top_k=40, min_p=0.0, typical_p=1.0, seed=-1,
                  context_source: str = "Labs") -> str:
        import nativelab.GlobalConfig.binaryResolve as _binres
        from nativelab.Server.server_global import SERVER_CONFIG

        prompt_text = self._build_text_prompt(messages, eng.model_path)
        context_meter.report_prompt(
            source=context_source,
            engine=eng,
            prompt=prompt_text,
            n_predict=n_predict,
        )
        cli_bin = SERVER_CONFIG.cli_path or _binres.LLAMA_CLI
        cmd = [
            cli_bin, "-m", eng.model_path,
            "-t", str(DEFAULT_THREADS()),
            "--ctx-size", str(getattr(eng, "ctx_value", DEFAULT_CTX())),
            "-n", str(n_predict),
            "--no-display-prompt", "--no-escape",
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "--repeat-penalty", str(repeat_penalty),
            "-p", prompt_text,
        ]
        LlamaEngine._append_cli_sampler_args(cmd, top_k=top_k, min_p=min_p, typical_p=typical_p, seed=seed)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            timeout=LONG_TIMEOUT_SECONDS,
        )
        text = result.stdout.decode("utf-8", errors="replace").strip()
        context_meter.append_output(text)
        return text

    def _call_api(self, api: ApiEngine, messages, n_predict,
                  temperature) -> str:
        cfg = api._config
        if cfg is None:
            raise RuntimeError("API config not loaded")
        max_tokens = int(n_predict or 0)
        self._log("INFO",
                  f"API call → {cfg.api_format.upper()}  "
                  f"model={cfg.model_id}"
                  + (f"  max_tokens={max_tokens}" if max_tokens > 0 else "  max_tokens=provider default"))

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
                "max_tokens":  max_tokens if max_tokens > 0 else 8192,
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
            try:
                with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
                    d = json.loads(r.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                self._raise_backend_error(exc.code, raw, "API")
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
            "temperature": temperature,
            "stream":      False,
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        req = urllib.request.Request(
            f"{cfg.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
                d = json.loads(r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            self._raise_backend_error(exc.code, raw, "API")
        choices = d.get("choices") or []
        if choices:
            return (choices[0].get("message", {}).get("content") or "").strip()
        return ""
