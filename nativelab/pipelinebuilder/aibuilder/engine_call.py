from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from nativelab.GlobalConfig.config_global import LONG_TIMEOUT_SECONDS
from nativelab.core.context_meter import context_meter
from nativelab.core.streamerworker.apistreamer import ANTHROPIC_UNCAPPED_FALLBACK_TOKENS


def _apply_api_custom_prompt(messages: List[Dict[str, Any]], cfg: Any) -> List[Dict[str, Any]]:
    if not getattr(cfg, "use_custom_prompt", False):
        return list(messages)
    wrapped: List[Dict[str, Any]] = []
    system_prompt = str(getattr(cfg, "system_prompt", "") or "")
    user_prefix = str(getattr(cfg, "user_prefix", "") or "")
    user_suffix = str(getattr(cfg, "user_suffix", "") or "")
    assistant_prefix = str(getattr(cfg, "assistant_prefix", "") or "")
    if system_prompt:
        wrapped.append({"role": "system", "content": system_prompt})
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = msg.get("content", "")
        if role == "user":
            wrapped.append({"role": role, "content": f"{user_prefix}{content}{user_suffix}"})
        elif role == "assistant":
            wrapped.append({"role": role, "content": f"{assistant_prefix}{content}"})
        else:
            wrapped.append({"role": role, "content": content})
    return wrapped


def _sync_api_generate(
    engine: Any,
    messages: List[Dict[str, Any]],
    *,
    n_predict: int,
    temperature: float,
) -> str:
    import urllib.error
    import urllib.request

    cfg = getattr(engine, "_config", None)
    if cfg is None:
        raise RuntimeError("API model is not configured.")

    messages = _apply_api_custom_prompt(messages, cfg)
    context_meter.report_messages(
        source="AI Pipeline Builder",
        engine=engine,
        messages=messages,
        n_predict=n_predict,
        mode="api",
    )

    api_format = str(getattr(cfg, "api_format", "openai") or "openai")
    base_url = str(getattr(cfg, "base_url", "") or "").rstrip("/")
    model_id = str(getattr(cfg, "model_id", "") or "")
    api_key = str(getattr(cfg, "api_key", "") or "")
    if not base_url or not model_id:
        raise RuntimeError("API model is missing base URL or model id.")

    try:
        if api_format == "anthropic":
            system_msg = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            user_messages = [m for m in messages if m.get("role") != "system"]
            max_tokens = int(n_predict or 0) if int(n_predict or 0) > 0 else ANTHROPIC_UNCAPPED_FALLBACK_TOKENS
            body = json.dumps({
                "model": model_id,
                "messages": user_messages,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **({"system": system_msg} if system_msg else {}),
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/v1/messages",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            parts = payload.get("content", [])
            text = "".join(
                str(part.get("text", ""))
                for part in parts
                if isinstance(part, dict) and part.get("type", "text") == "text"
            )
        else:
            payload = {
                "model": model_id,
                "messages": messages,
                "stream": False,
                "temperature": temperature,
            }
            if int(n_predict or 0) > 0:
                payload["max_tokens"] = int(n_predict)
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            text = str(payload.get("choices", [{}])[0].get("message", {}).get("content") or "")
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            raw = str(getattr(exc, "reason", "") or "")
        raise RuntimeError(f"API HTTP {exc.code}: {raw}") from exc

    context_meter.append_output(text)
    return text


def generate_pipeline_response(
    engine: Any,
    messages: List[Dict[str, Any]],
    *,
    n_predict: int,
    temperature: float = 0.15,
    abort_cb: Optional[Any] = None,
) -> str:
    if engine is None or not getattr(engine, "is_loaded", False):
        raise RuntimeError("Load a model before using AI Pipeline Builder.")

    if hasattr(engine, "generate_sync"):
        return engine.generate_sync(
            messages=messages,
            n_predict=n_predict,
            temperature=temperature,
            top_p=0.85,
            repeat_penalty=1.08,
            abort_cb=abort_cb,
            context_source="AI Pipeline Builder",
        )

    if getattr(engine, "mode", "") == "api" and getattr(engine, "_config", None) is not None:
        return _sync_api_generate(
            engine,
            messages,
            n_predict=n_predict,
            temperature=temperature,
        )

    raise RuntimeError("The loaded backend does not support synchronous AI builder requests.")
