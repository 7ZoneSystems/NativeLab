"""
Centralized backend facade for all NativeLab operations.

Provides a unified interface for:
- Model operations (load, unload, list, configure)
- API model operations (register, list, call)
- Device operations (scan, register, manage)
- Generation (local, API, device)
- Server operations (start, stop, status)

All network operations go through NativeLabHttpClient.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from nativelab.core.http_client import HttpError, NativeLabHttpClient, get_http_client
from nativelab.core.engine_status import (
    EngineStatus,
    active_engine_status,
    engine_status,
    engine_snapshot,
)
from nativelab.core.llm_errors import LlmErrorNotice, explain_llm_error
from nativelab.Model.APImodels import (
    ApiConfig,
    api_model_ref,
    api_model_name_from_ref,
    getapi_registry,
    is_api_model_ref,
    is_phonolab_device,
    device_spec_label,
)
from nativelab.Model.ModelRegistry import get_model_registry

# ── Device types ──────────────────────────────────────────────────

@dataclass
class DeviceInfo:
    """A discovered PhonoLab device."""
    ip: str
    port: int = 8787
    name: str = ""
    model: str = ""
    status: str = "unknown"
    is_vision: bool = False
    cpu_cores: int = 0
    ram_mb: int = 0
    android_version: str = ""
    api_key: str = ""
    last_seen: float = 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self.ip}:{self.port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1"

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.ip})" if self.name else self.ip


@dataclass
class BackendResult:
    """Standardized result from backend operations."""
    ok: bool
    data: Any = None
    error: str = ""
    error_notice: Optional[LlmErrorNotice] = None

    @classmethod
    def success(cls, data: Any = None) -> "BackendResult":
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, error: str, notice: Optional[LlmErrorNotice] = None) -> "BackendResult":
        return cls(ok=False, error=error, error_notice=notice)


# ── Backend Facade ────────────────────────────────────────────────

class NativeLabBackend:
    """
    Unified backend for all NativeLab operations.

    Usage:
        backend = NativeLabBackend()
        result = backend.load_model("/path/to/model.gguf")
        result = backend.generate(messages=[...])
        devices = backend.scan_network()
    """

    def __init__(self, http_client: Optional[NativeLabHttpClient] = None):
        self._http = http_client or get_http_client()
        self._llama_engine = None
        self._api_engine = None

    def set_engines(self, llama_engine: Any = None, api_engine: Any = None):
        """Set the active engines (called from MainWindow)."""
        self._llama_engine = llama_engine
        self._api_engine = api_engine

    # ── Model Operations ──────────────────────────────────────────

    def load_model(
        self,
        model_path: str,
        threads: int = 4,
        ctx: int = 2048,
        n_predict: int = 384,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> BackendResult:
        """Load a local GGUF model."""
        registry = get_model_registry()
        engine = self._llama_engine
        if engine is None:
            return BackendResult.failure("No llama engine available")

        try:
            ok = engine.load(model_path, threads=threads, ctx=ctx)
            if ok:
                return BackendResult.success({"model_path": model_path})
            return BackendResult.failure("Model load returned false")
        except Exception as e:
            notice = explain_llm_error(e, source="Model loader")
            return BackendResult.failure(str(e), notice)

    def load_api_model(self, config: ApiConfig) -> BackendResult:
        """Load an API model."""
        from nativelab.core.engines.apiengine import ApiEngine
        engine = ApiEngine()
        try:
            ok = engine.load(config)
            if ok:
                self._api_engine = engine
                return BackendResult.success({"model": config.model_id})
            return BackendResult.failure("API model verification failed")
        except Exception as e:
            return BackendResult.failure(str(e))

    def unload_model(self) -> BackendResult:
        """Unload the current model."""
        engine = self._llama_engine
        if engine:
            try:
                engine.shutdown()
            except Exception:
                pass
        api = self._api_engine
        if api:
            try:
                api.shutdown()
            except Exception:
                pass
        return BackendResult.success()

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available local models."""
        return get_model_registry().all_models()

    def list_api_models(self) -> List[ApiConfig]:
        """List all registered API models."""
        return getapi_registry().all()

    def register_api_model(self, config: ApiConfig) -> BackendResult:
        """Register a new API model."""
        try:
            getapi_registry().add(config)
            return BackendResult.success({"name": config.name})
        except Exception as e:
            return BackendResult.failure(str(e))

    def remove_api_model(self, name: str) -> BackendResult:
        """Remove an API model."""
        try:
            getapi_registry().remove(name)
            return BackendResult.success()
        except Exception as e:
            return BackendResult.failure(str(e))

    # ── Generation ────────────────────────────────────────────────

    def generate(
        self,
        messages: List[Dict[str, str]],
        model_ref: str = "",
        n_predict: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = "",
        on_token: Optional[Callable[[str], None]] = None,
    ) -> BackendResult:
        """
        Generate text from messages.
        Routes to the appropriate backend (local, API, or device).
        """
        if is_api_model_ref(model_ref):
            return self._generate_api(messages, model_ref, n_predict, temperature, top_p, on_token)

        engine = self._llama_engine
        if engine is None or not getattr(engine, "is_loaded", False):
            return BackendResult.failure("No model loaded")

        try:
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages

            if on_token and hasattr(engine, "generate_sync"):
                # Streaming via engine
                text = engine.generate_sync(
                    messages=messages,
                    n_predict=n_predict,
                    temperature=temperature,
                    top_p=top_p,
                    on_token=on_token,
                )
            else:
                text = engine.generate_sync(
                    messages=messages,
                    n_predict=n_predict,
                    temperature=temperature,
                    top_p=top_p,
                )
            return BackendResult.success(text.strip() if text else "")
        except Exception as e:
            notice = explain_llm_error(e, source="LLM engine")
            return BackendResult.failure(str(e), notice)

    def _generate_api(
        self,
        messages: List[Dict[str, str]],
        model_ref: str,
        n_predict: int,
        temperature: float,
        top_p: float,
        on_token: Optional[Callable[[str], None]],
    ) -> BackendResult:
        """Generate via an API model."""
        name = api_model_name_from_ref(model_ref)
        registry = getapi_registry()
        cfg = registry.get(name)
        if cfg is None:
            return BackendResult.failure(f"API model not found: {name}")

        base_url = cfg.base_url.rstrip("/")
        api_key = cfg.api_key

        try:
            if cfg.api_format == "anthropic":
                url = f"{base_url}/v1/messages"
                text = self._http.post_anthropic_stream(
                    url=url,
                    messages=messages,
                    model=cfg.model_id,
                    api_key=api_key,
                    max_tokens=n_predict,
                    temperature=temperature,
                    on_token=on_token,
                )
            else:
                url = f"{base_url}/chat/completions"
                text = self._http.post_openai_stream(
                    url=url,
                    messages=messages,
                    model=cfg.model_id,
                    api_key=api_key,
                    max_tokens=n_predict,
                    temperature=temperature,
                    top_p=top_p,
                    on_token=on_token,
                )
            return BackendResult.success(text)
        except HttpError as e:
            notice = explain_llm_error(e, source=f"API ({name})")
            return BackendResult.failure(str(e), notice)
        except Exception as e:
            return BackendResult.failure(str(e))

    # ── Device Operations ─────────────────────────────────────────

    def scan_network(
        self,
        subnet: str = "",
        port: int = 8787,
        on_found: Optional[Callable[[DeviceInfo], None]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> List[DeviceInfo]:
        """Scan local network for PhonoLab devices."""
        from nativelab.api_server.device_discovery import scan_network, DiscoveredDevice

        def _convert(d: DiscoveredDevice) -> DeviceInfo:
            return DeviceInfo(
                ip=d.ip, port=d.port, name=d.name, model=d.model,
                status=d.status, is_vision=d.is_vision,
                cpu_cores=d.cpu_cores, ram_mb=d.ram_mb,
                android_version=d.android_version, api_key=d.api_key,
                last_seen=d.last_seen,
            )

        found = scan_network(
            subnet=subnet,
            port=port,
            on_found=lambda d: on_found(_convert(d)) if on_found else None,
            progress_cb=progress_cb,
        )
        return [_convert(d) for d in found]

    def test_device(self, device: DeviceInfo, api_key: str = "") -> BackendResult:
        """Test connection to a PhonoLab device."""
        key = api_key or device.api_key or ""
        try:
            resp = self._http.get(
                f"{device.base_url}/health",
                auth_token=key,
                timeout=5,
            )
            if resp.ok and isinstance(resp.data, dict) and resp.data.get("ok"):
                return BackendResult.success(resp.data)
            return BackendResult.failure(resp.error_message or "Connection failed")
        except Exception as e:
            return BackendResult.failure(str(e))

    def get_device_status(self, device: DeviceInfo) -> BackendResult:
        """Get full status from a device."""
        try:
            resp = self._http.get(f"{device.base_url}/status", auth_token=device.api_key, timeout=5)
            if resp.ok:
                return BackendResult.success(resp.data)
            return BackendResult.failure(resp.error_message)
        except Exception as e:
            return BackendResult.failure(str(e))

    def get_device_runtime_config(self, device: DeviceInfo) -> BackendResult:
        """Get live runtime config (model, ctx, vision, status) from a device."""
        try:
            resp = self._http.get(f"{device.base_url}/runtime", auth_token=device.api_key, timeout=5)
            if resp.ok and isinstance(resp.data, dict):
                return BackendResult.success(resp.data)
            return BackendResult.failure(resp.error_message)
        except Exception as e:
            return BackendResult.failure(str(e))

    def get_device_models(self, device: DeviceInfo) -> BackendResult:
        """Get model list from a device."""
        try:
            resp = self._http.get(f"{device.base_url}/v1/models", auth_token=device.api_key, timeout=5)
            if resp.ok and isinstance(resp.data, dict):
                return BackendResult.success(resp.data.get("data", []))
            return BackendResult.failure(resp.error_message)
        except Exception as e:
            return BackendResult.failure(str(e))

    def load_model_on_device(self, device: DeviceInfo, model_path: str) -> BackendResult:
        """Trigger model load on a remote device."""
        try:
            resp = self._http.post(
                f"{device.base_url}/load",
                body={"model_path": model_path},
                auth_token=device.api_key,
                timeout=10,
            )
            if resp.ok and isinstance(resp.data, dict) and resp.data.get("ok"):
                return BackendResult.success(resp.data)
            return BackendResult.failure(resp.error_message or "Load failed")
        except Exception as e:
            return BackendResult.failure(str(e))

    def update_device_config(self, device: DeviceInfo, config: Dict[str, Any]) -> BackendResult:
        """Update model config on a remote device."""
        try:
            resp = self._http.post(
                f"{device.base_url}/config",
                body=config,
                auth_token=device.api_key,
                timeout=5,
            )
            if resp.ok and isinstance(resp.data, dict) and resp.data.get("ok"):
                return BackendResult.success(resp.data)
            return BackendResult.failure(resp.error_message or "Config update failed")
        except Exception as e:
            return BackendResult.failure(str(e))

    def register_device_as_model(self, device: DeviceInfo) -> BackendResult:
        """Register a discovered device as an API model."""
        cfg = ApiConfig(
            name=device.display_name,
            provider="PhonoLab",
            model_id=device.model or "phonolab-active",
            api_key=device.api_key,
            base_url=device.api_url,
            api_format="openai",
            max_tokens=512,
            temperature=0.7,
            custom_provider_name="PhonoLab",
            cpu_cores=device.cpu_cores,
            ram_mb=device.ram_mb,
            is_vision=device.is_vision,
            ctx_limit=2048,
            android_ver=device.android_version,
            device_status=device.status,
        )
        return self.register_api_model(cfg)

    # ── Server Operations ─────────────────────────────────────────

    def get_engine_status(self) -> EngineStatus:
        """Get current engine status."""
        return active_engine_status(
            llama_engine=self._llama_engine,
            api_engine=self._api_engine,
        )

    def get_engine_snapshot(self) -> Dict[str, Any]:
        """Get full engine snapshot as dict."""
        return engine_snapshot(
            llama_engine=self._llama_engine,
            api_engine=self._api_engine,
        )

    def ensure_server(self) -> bool:
        """Ensure the local llama-server is running."""
        engine = self._llama_engine
        if engine and hasattr(engine, "ensure_server"):
            return engine.ensure_server()
        return False

    # ── HuggingFace Operations ────────────────────────────────────

    def fetch_hf_gguf_files(self, repo: str) -> BackendResult:
        """List GGUF files in a HuggingFace repo."""
        try:
            resp = self._http.get(
                f"https://huggingface.co/api/models/{repo}",
                timeout=30,
            )
            if not resp.ok:
                return BackendResult.failure(f"HF API error: {resp.status}")
            data = resp.data if isinstance(resp.data, dict) else json.loads(resp.raw_text)
            siblings = data.get("siblings", [])
            files = [
                {"name": s.get("rfilename", ""), "size": s.get("size", 0)}
                for s in siblings
                if s.get("rfilename", "").lower().endswith(".gguf")
            ]
            return BackendResult.success(files)
        except Exception as e:
            return BackendResult.failure(str(e))

    def fetch_llama_cpp_releases(self) -> BackendResult:
        """Fetch latest llama.cpp releases from GitHub."""
        try:
            resp = self._http.get(
                "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
                timeout=15,
            )
            if not resp.ok:
                return BackendResult.failure(f"GitHub API error: {resp.status}")
            data = resp.data if isinstance(resp.data, dict) else json.loads(resp.raw_text)
            return BackendResult.success(data)
        except Exception as e:
            return BackendResult.failure(str(e))


# ── Singleton accessor ────────────────────────────────────────────

_backend: Optional[NativeLabBackend] = None

def get_backend() -> NativeLabBackend:
    """Get the global backend instance."""
    global _backend
    if _backend is None:
        _backend = NativeLabBackend()
    return _backend
