from .config import ACTIVE_MODEL_REF, ApiServerConfig
from .server import NativeLabApiServer

__all__ = [
    "ACTIVE_MODEL_REF",
    "ApiServerConfig",
    "ApiServerTab",
    "NativeLabApiServer",
]


def __getattr__(name: str):
    if name == "ApiServerTab":
        from .tab import ApiServerTab
        return ApiServerTab
    raise AttributeError(name)
