from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from nativelab.GlobalConfig.config_global import DEFAULT_CTX
from nativelab.Model.model_global import (
    api_model_name_from_ref,
    detect_model_family,
    detect_quant_type,
    is_api_model_ref,
    is_external_model_ref,
    is_hf_model_ref,
    is_ollama_model_ref,
    model_ref_backend,
    model_ref_display_name,
    model_ref_payload,
)


UNLOADED_MODE = "unloaded"
API_MODE = "api"
SERVER_MODE = "server"
CLI_MODE = "cli"
OLLAMA_MODE = "ollama"
HF_TRANSFORMERS_MODE = "hf_transformers"

READY_MODES = {API_MODE, SERVER_MODE, OLLAMA_MODE, HF_TRANSFORMERS_MODE}
WARN_READY_MODES = {CLI_MODE}


@dataclass(frozen=True)
class EngineStatus:
    """One normalized, read-only view of an engine's display/runtime state."""

    role: str = ""
    mode: str = UNLOADED_MODE
    backend: str = UNLOADED_MODE
    state: str = "idle"
    status_text: str = "No Engine"
    model_path: str = ""
    model_name: str = ""
    ctx_value: int = field(default_factory=lambda: int(DEFAULT_CTX()))
    server_port: int = 0
    family: str = ""
    quant_type: str = ""
    is_loaded: bool = False
    is_api: bool = False
    is_local: bool = False
    can_reload: bool = False
    is_loading: bool = False

    @property
    def family_quant_tag(self) -> str:
        if self.family and self.quant_type:
            return f"{self.family} · {self.quant_type}"
        return self.family or self.quant_type

    @property
    def mode_tag(self) -> str:
        return self.mode if self.is_loaded else ""

    @property
    def manager_label(self) -> str:
        model = self.model_name or "not loaded"
        mode = f"  [{self.mode_tag}]" if self.mode_tag else ""
        meta = f"  [{self.family_quant_tag}]" if self.family_quant_tag else ""
        return f"  {self.role:<22}  {model}{mode}{meta}"

    @property
    def active_label(self) -> str:
        return f"Active engine: {self.status_text}"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def active_engine(llama_engine: Any = None, api_engine: Any = None) -> Optional[Any]:
    """Return the engine that should be presented as active to users."""
    if _is_loaded(api_engine):
        return api_engine
    return llama_engine


def engine_status(
    engine: Any,
    *,
    role: str = "",
    is_loading: bool = False,
    none_text: str = "No Engine",
) -> EngineStatus:
    if engine is None:
        return EngineStatus(
            role=role,
            status_text="Loading model..." if is_loading else none_text,
            state="loading" if is_loading else "idle",
            is_loading=bool(is_loading),
        )

    mode = str(getattr(engine, "mode", UNLOADED_MODE) or UNLOADED_MODE)
    loaded = _is_loaded(engine)
    model_path = str(getattr(engine, "model_path", "") or "")
    model_name = _model_name(engine, model_path)
    server_port = _int_attr(engine, "server_port", 0)
    ctx_value = _int_attr(engine, "ctx_value", int(DEFAULT_CTX()))
    backend = _backend_for(mode, model_path)
    state = _state_for(mode, loaded, is_loading)
    family, quant = _model_meta(model_path, mode)

    return EngineStatus(
        role=role,
        mode=mode,
        backend=backend,
        state=state,
        status_text=_status_text(
            engine,
            mode,
            loaded,
            model_path,
            model_name,
            server_port,
            none_text,
            is_loading,
        ),
        model_path=model_path,
        model_name=model_name,
        ctx_value=ctx_value,
        server_port=server_port,
        family=family,
        quant_type=quant,
        is_loaded=loaded,
        is_api=(mode == API_MODE),
        is_local=loaded and mode != API_MODE,
        can_reload=loaded and mode != API_MODE,
        is_loading=bool(is_loading),
    )


def active_engine_status(
    llama_engine: Any = None,
    api_engine: Any = None,
    *,
    is_loading: bool = False,
    none_text: str = "No Engine",
) -> EngineStatus:
    return engine_status(
        active_engine(llama_engine, api_engine),
        is_loading=is_loading,
        none_text=none_text,
    )


def engine_snapshot(
    llama_engine: Any = None,
    api_engine: Any = None,
    *,
    is_loading: bool = False,
) -> Dict[str, Any]:
    status = active_engine_status(
        llama_engine,
        api_engine,
        is_loading=is_loading,
        none_text="No Engine",
    )
    data = status.as_dict()
    data["status"] = status.status_text
    return data


def _is_loaded(engine: Any) -> bool:
    if engine is None:
        return False
    try:
        return bool(getattr(engine, "is_loaded", False))
    except Exception:
        return False


def _int_attr(engine: Any, attr: str, default: int) -> int:
    try:
        return int(getattr(engine, attr, default) or 0)
    except Exception:
        return int(default)


def _model_name(engine: Any, model_path: str) -> str:
    cfg = getattr(engine, "_config", None)
    if cfg is not None:
        provider = getattr(cfg, "custom_provider_name", "") or getattr(cfg, "provider", "")
        model_id = getattr(cfg, "model_id", "")
        if provider and model_id:
            return f"{provider} · {model_id}"
        return model_id or getattr(cfg, "name", "") or model_path
    if is_api_model_ref(model_path):
        return api_model_name_from_ref(model_path)
    return model_ref_display_name(model_path) if model_path else ""


def _backend_for(mode: str, model_path: str) -> str:
    if mode == API_MODE or is_api_model_ref(model_path):
        return API_MODE
    if mode in READY_MODES or mode in WARN_READY_MODES:
        return mode
    if model_path:
        return model_ref_backend(model_path)
    return UNLOADED_MODE


def _state_for(mode: str, loaded: bool, is_loading: bool) -> str:
    if is_loading and not loaded:
        return "loading"
    if not loaded:
        return "idle"
    if mode in WARN_READY_MODES:
        return "warn"
    return "ok"


def _status_text(
    engine: Any,
    mode: str,
    loaded: bool,
    model_path: str,
    model_name: str,
    server_port: int,
    none_text: str,
    is_loading: bool,
) -> str:
    if is_loading and not loaded:
        return "Loading model..."
    if mode == API_MODE or _looks_like_api_engine(engine):
        cfg = getattr(engine, "_config", None)
        if cfg is not None:
            provider = getattr(cfg, "custom_provider_name", "") or getattr(cfg, "provider", "")
            model_id = getattr(cfg, "model_id", "")
            return f"{provider}  ·  {model_id}" if provider and model_id else (model_id or "API Connected")
        return "API Not Connected" if not loaded else (model_name or "API Connected")
    if mode == SERVER_MODE:
        return f"Server  :{server_port}" if server_port else "Server"
    if mode == CLI_MODE:
        return "CLI Mode"
    if mode == OLLAMA_MODE:
        return f"Ollama  ·  {model_name or model_ref_display_name(model_path)}"
    if mode == HF_TRANSFORMERS_MODE:
        return f"Transformers  ·  {model_name or model_ref_display_name(model_path)}"
    return "Not Loaded" if engine is not None else none_text


def _looks_like_api_engine(engine: Any) -> bool:
    return engine is not None and (
        engine.__class__.__name__ == "ApiEngine"
        or hasattr(engine, "_config") and not hasattr(engine, "server_proc")
    )


def _model_meta(model_path: str, mode: str) -> tuple[str, str]:
    if not model_path or is_api_model_ref(model_path):
        return "", "API" if is_api_model_ref(model_path) else ""
    payload = model_ref_payload(model_path) or model_path
    try:
        family = detect_model_family(payload).name
    except Exception:
        family = ""
    if is_ollama_model_ref(model_path) or mode == OLLAMA_MODE:
        quant = "OLLAMA"
    elif is_hf_model_ref(model_path) or mode == HF_TRANSFORMERS_MODE:
        quant = "TRANSFORMERS"
    elif is_external_model_ref(model_path):
        quant = model_ref_backend(model_path).upper()
    else:
        try:
            quant = detect_quant_type(payload)
        except Exception:
            quant = ""
    return family, quant
