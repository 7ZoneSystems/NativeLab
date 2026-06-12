from __future__ import annotations

import ctypes
import platform
from pathlib import Path

from PhonoLab.paths import RUNTIME_DIR


def _library_names() -> tuple[str, ...]:
    system = platform.system().lower()
    if system == "windows":
        return ("phonolab_mobile_core.dll",)
    if system == "darwin":
        return ("libphonolab_mobile_core.dylib",)
    return ("libphonolab_mobile_core.so",)


def _load_lib():
    for name in _library_names():
        candidate = RUNTIME_DIR / "lib" / name
        if not candidate.exists():
            candidate = Path(__file__).resolve().parent / name
        if not candidate.exists():
            continue
        try:
            lib = ctypes.CDLL(str(candidate))
            lib.phonolab_estimate_tokens.argtypes = [ctypes.c_char_p]
            lib.phonolab_estimate_tokens.restype = ctypes.c_size_t
            lib.phonolab_context_fits.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            lib.phonolab_context_fits.restype = ctypes.c_int
            return lib
        except Exception:
            continue
    return None


_LIB = _load_lib()


def _fallback_estimate_tokens(text: str) -> int:
    raw = str(text or "")
    if not raw:
        return 0
    # Conservative enough for mobile guardrails without requiring tokenizer deps.
    byte_cost = max(1, (len(raw.encode("utf-8")) + 3) // 4)
    word_cost = max(1, len(raw.split()))
    return max(byte_cost, word_cost)


def estimate_tokens(text: str) -> int:
    if _LIB is None:
        return _fallback_estimate_tokens(text)
    try:
        return int(_LIB.phonolab_estimate_tokens(str(text or "").encode("utf-8")))
    except Exception:
        return _fallback_estimate_tokens(text)


def prompt_fits_context(text: str, context_tokens: int, reserved_output_tokens: int) -> bool:
    ctx = int(context_tokens or 0)
    reserved = int(reserved_output_tokens or 0)
    if _LIB is not None:
        try:
            return bool(_LIB.phonolab_context_fits(str(text or "").encode("utf-8"), ctx, reserved))
        except Exception:
            pass
    return estimate_tokens(text) + max(0, reserved) <= max(0, ctx)
