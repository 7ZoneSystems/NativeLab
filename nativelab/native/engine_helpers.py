from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    from . import _native_core as _core
except Exception:
    _core = None


def native_core_available() -> bool:
    return _core is not None


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def build_text_prompt(messages: Iterable[Dict[str, Any]], family: Any) -> str:
    """Build a llama.cpp text prompt using the detected family template."""
    msg_list = list(messages or [])
    if _core is not None:
        try:
            return _core.build_text_prompt(msg_list, family)
        except Exception:
            pass

    out: List[str] = []
    sys_buf = ""
    for msg in msg_list:
        role = msg.get("role", "user")
        content = _message_content_to_text(msg.get("content", ""))
        if role == "system":
            sys_buf += content + "\n"
        elif role == "user":
            user_text = (sys_buf + content) if sys_buf else content
            sys_buf = ""
            out.append(
                getattr(family, "user_prefix", "")
                + user_text
                + getattr(family, "user_suffix", "")
            )
        elif role == "assistant":
            out.append(
                getattr(family, "assistant_prefix", "")
                + content
                + getattr(family, "assistant_suffix", "")
            )
    out.append(getattr(family, "assistant_prefix", ""))
    return getattr(family, "bos", "") + "".join(out)


def raw_prompt_text(
    messages: Iterable[Dict[str, Any]],
    prompt_builder: Callable[[Iterable[Dict[str, Any]]], str],
) -> str:
    msg_list = list(messages or [])
    if len(msg_list) == 1:
        content = msg_list[0].get("content", "")
        if isinstance(content, str):
            return content
    return prompt_builder(msg_list)


def sampler_payload(
    top_k: int = 40,
    min_p: float = 0.0,
    typical_p: float = 1.0,
    seed: int = -1,
) -> Dict[str, Any]:
    if _core is not None:
        try:
            return dict(_core.sampler_payload(top_k, min_p, typical_p, seed))
        except Exception:
            pass

    out: Dict[str, Any] = {}
    try:
        out["top_k"] = max(0, int(top_k))
    except (TypeError, ValueError):
        pass
    try:
        value = float(min_p)
        if value > 0:
            out["min_p"] = value
    except (TypeError, ValueError):
        pass
    try:
        value = float(typical_p)
        if 0 < value < 1:
            out["typical_p"] = value
    except (TypeError, ValueError):
        pass
    try:
        value = int(seed)
        if value >= 0:
            out["seed"] = value
    except (TypeError, ValueError):
        pass
    return out


def ollama_sampler_options(
    top_k: int = 40,
    min_p: float = 0.0,
    seed: int = -1,
) -> Dict[str, Any]:
    payload = sampler_payload(top_k=top_k, min_p=min_p, typical_p=1.0, seed=seed)
    payload.pop("typical_p", None)
    return payload


def hf_sampler_kwargs(
    top_k: int = 40,
    min_p: float = 0.0,
    typical_p: float = 1.0,
) -> Dict[str, Any]:
    payload = sampler_payload(top_k=top_k, min_p=min_p, typical_p=typical_p, seed=-1)
    payload.pop("seed", None)
    return payload


def append_cli_sampler_args(
    cmd: List[str],
    top_k: int = 40,
    min_p: float = 0.0,
    typical_p: float = 1.0,
    seed: int = -1,
) -> None:
    if _core is not None:
        try:
            _core.append_cli_sampler_args(cmd, top_k, min_p, typical_p, seed)
            return
        except Exception:
            pass

    payload = sampler_payload(top_k=top_k, min_p=min_p, typical_p=typical_p, seed=seed)
    if "top_k" in payload:
        cmd.extend(["--top-k", str(payload["top_k"])])
    if "min_p" in payload:
        cmd.extend(["--min-p", str(payload["min_p"])])
    if "typical_p" in payload:
        cmd.extend(["--typical", str(payload["typical_p"])])
    if "seed" in payload:
        cmd.extend(["--seed", str(payload["seed"])])


def image_b64_list(image_data: Optional[Iterable[Dict[str, Any]]] = None) -> List[str]:
    out: List[str] = []
    for item in image_data or []:
        if isinstance(item, dict):
            data = item.get("data") or ""
            if isinstance(data, str) and data:
                out.append(data.split(";base64,", 1)[-1])
    return out


_CONTEXT_NEEDLES = (
    "context size has been exceeded",
    "exceeds the available context size",
    "exceed_context_size",
    "context window",
    "n_ctx",
)


def is_context_error(raw: str) -> bool:
    if _core is not None:
        try:
            return bool(_core.is_context_error(str(raw)))
        except Exception:
            pass
    text = str(raw).lower()
    return any(needle in text for needle in _CONTEXT_NEEDLES)


def error_message(raw: str) -> str:
    try:
        data = json.loads(raw)
    except Exception:
        return raw
    err = data.get("error", data)
    if isinstance(err, dict):
        return str(err.get("message") or err.get("type") or raw)
    return str(err or raw)


def build_reference_chunks(text: str, step: int, overlap: int = 80) -> List[str]:
    if _core is not None:
        try:
            return list(_core.build_reference_chunks(str(text), int(step), int(overlap)))
        except Exception:
            pass
    step = int(step)
    if step <= 0:
        raise ValueError("step must be greater than zero")
    overlap = max(0, int(overlap))
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i: i + step + overlap])
        i += step
    return chunks
