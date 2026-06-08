from __future__ import annotations

import time
import uuid
from typing import Any


def _data_url_payload(value: str) -> tuple[str, str]:
    text = str(value or "")
    if text.startswith("data:") and ";base64," in text:
        header, data = text.split(";base64,", 1)
        media_type = header[5:] or "image/png"
        return media_type, data
    return "", ""


def _append_image(images: list[dict[str, Any]], data: str, media_type: str = "image/png") -> None:
    if data:
        images.append({
            "id": len(images) + 1,
            "data": data,
            "media_type": media_type or "image/png",
        })


def normalize_openai_messages(payload: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    messages: list[dict[str, str]] = []
    images: list[dict[str, Any]] = []
    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append(str(part))
                    continue
                ptype = str(part.get("type") or "")
                if ptype == "text":
                    parts.append(str(part.get("text") or ""))
                elif ptype == "image_url":
                    image_url = part.get("image_url") or {}
                    url = image_url.get("url") if isinstance(image_url, dict) else image_url
                    media_type, data = _data_url_payload(str(url or ""))
                    if data:
                        _append_image(images, data, media_type)
                    else:
                        parts.append("[Image URL omitted: only inline base64 data URLs are supported locally.]")
            content = "\n".join(p for p in parts if p)
        messages.append({"role": role, "content": str(content or "")})
    if not messages and payload.get("prompt") is not None:
        messages.append({"role": "user", "content": str(payload.get("prompt") or "")})
    return messages, images


def normalize_anthropic_messages(payload: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    messages: list[dict[str, str]] = []
    images: list[dict[str, Any]] = []
    system = payload.get("system")
    if system:
        messages.append({"role": "system", "content": str(system)})
    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append(str(part))
                    continue
                ptype = str(part.get("type") or "")
                if ptype == "text":
                    parts.append(str(part.get("text") or ""))
                elif ptype == "image":
                    source = part.get("source") or {}
                    if isinstance(source, dict):
                        if source.get("type") == "base64":
                            _append_image(
                                images,
                                str(source.get("data") or ""),
                                str(source.get("media_type") or "image/png"),
                            )
                        else:
                            parts.append("[Image omitted: only base64 image sources are supported locally.]")
            content = "\n".join(p for p in parts if p)
        messages.append({"role": role, "content": str(content or "")})
    return messages, images


def sampling_options(payload: dict[str, Any]) -> dict[str, Any]:
    max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens", payload.get("n_predict", 512)))
    return {
        "n_predict": int(max_tokens or 512),
        "temperature": float(payload.get("temperature", 0.7) if payload.get("temperature") is not None else 0.7),
        "top_p": float(payload.get("top_p", 0.9) if payload.get("top_p") is not None else 0.9),
        "repeat_penalty": float(payload.get("repeat_penalty", 1.1) if payload.get("repeat_penalty") is not None else 1.1),
        "top_k": int(payload.get("top_k", 40) if payload.get("top_k") is not None else 40),
        "min_p": float(payload.get("min_p", 0.0) if payload.get("min_p") is not None else 0.0),
        "typical_p": float(payload.get("typical_p", 1.0) if payload.get("typical_p") is not None else 1.0),
        "seed": int(payload.get("seed", -1) if payload.get("seed") is not None else -1),
    }


def openai_chat_response(model: str, text: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def openai_completion_response(model: str, text: str) -> dict[str, Any]:
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def anthropic_message_response(model: str, text: str) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def error_payload(message: str, code: str = "server_error", status: int = 500) -> dict[str, Any]:
    return {
        "error": {
            "message": str(message),
            "type": code,
            "code": code,
            "status": int(status),
        }
    }
