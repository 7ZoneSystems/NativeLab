from .config import ACTIVE_MODEL_REF, ApiServerConfig
from .server import NativeLabApiServer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .devices_tab import DevicesTab
    from .tab import ApiServerTab

__all__ = [
    "ACTIVE_MODEL_REF",
    "ApiServerConfig",
    "ApiServerTab",
    "DevicesTab",
    "NativeLabApiServer",
]


def __getattr__(name: str):
    if name == "ApiServerTab":
        from .tab import ApiServerTab
        return ApiServerTab
    if name == "DevicesTab":
        from .devices_tab import DevicesTab
        return DevicesTab
    raise AttributeError(name)
