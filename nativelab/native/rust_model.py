from __future__ import annotations

import ctypes
import platform
from pathlib import Path
from typing import Optional


_LIB = None
_LOAD_ATTEMPTED = False


def _library_names() -> list[str]:
    system = platform.system()
    if system == "Windows":
        return ["nativelab_rust.dll"]
    if system == "Darwin":
        return ["libnativelab_rust.dylib"]
    return ["libnativelab_rust.so"]


def _load_library():
    global _LIB, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LIB
    _LOAD_ATTEMPTED = True
    root = Path(__file__).resolve().parent
    for name in _library_names():
        path = root / name
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            lib.nl_detect_family_key.argtypes = [ctypes.c_char_p]
            lib.nl_detect_family_key.restype = ctypes.c_void_p
            lib.nl_detect_quant_type.argtypes = [ctypes.c_char_p]
            lib.nl_detect_quant_type.restype = ctypes.c_void_p
            lib.nl_free_string.argtypes = [ctypes.c_void_p]
            lib.nl_free_string.restype = None
            _LIB = lib
            return _LIB
        except Exception:
            continue
    return None


def rust_model_available() -> bool:
    return _load_library() is not None


def _call_string(func_name: str, value: str) -> Optional[str]:
    lib = _load_library()
    if lib is None:
        return None
    func = getattr(lib, func_name)
    ptr = func(str(value or "").encode("utf-8", errors="replace"))
    if not ptr:
        return None
    try:
        return ctypes.string_at(ptr).decode("utf-8", errors="replace")
    finally:
        lib.nl_free_string(ptr)


def detect_family_key(filename: str) -> Optional[str]:
    return _call_string("nl_detect_family_key", filename)


def detect_quant_type(filename: str) -> Optional[str]:
    return _call_string("nl_detect_quant_type", filename)
