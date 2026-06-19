"""Core module — centralized backend, HTTP client, engine status, and error handling."""

from .http_client import NativeLabHttpClient, HttpError, get_http_client
from .backend import NativeLabBackend, BackendResult, DeviceInfo, get_backend
from .engine_status import EngineStatus, engine_status, active_engine_status
from .llm_errors import LlmErrorNotice, explain_llm_error

__all__ = [
    "NativeLabHttpClient",
    "HttpError",
    "get_http_client",
    "NativeLabBackend",
    "BackendResult",
    "DeviceInfo",
    "get_backend",
    "EngineStatus",
    "engine_status",
    "active_engine_status",
    "LlmErrorNotice",
    "explain_llm_error",
]
