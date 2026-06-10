from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from nativelab.native import error_message as native_error_message
from nativelab.native import is_context_error as native_is_context_error


@dataclass(frozen=True)
class LlmErrorNotice:
    category: str
    title: str
    summary: str
    action: str
    technical_detail: str
    prompt_tokens: int = 0
    context_tokens: int = 0

    @property
    def user_message(self) -> str:
        return f"{self.summary}\n\nWhat to do:\n{self.action}"


def _raw_text(raw: Any) -> str:
    if isinstance(raw, BaseException):
        return f"{type(raw).__name__}: {raw}"
    return str(raw or "")


def _json_payload(text: str) -> Any:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            payload, _end = decoder.raw_decode(text[idx:])
            return payload
        except Exception:
            continue
    return None


def _error_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and isinstance(payload.get("error"), (dict, str)):
        return payload["error"]
    return payload


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _extract_token_counts(text: str, payload: Any) -> tuple[int, int]:
    err = _error_payload(payload)
    prompt_tokens = 0
    context_tokens = 0
    if isinstance(err, dict):
        prompt_tokens = (
            _coerce_int(err.get("n_prompt_tokens"))
            or _coerce_int(err.get("prompt_tokens"))
            or _coerce_int(err.get("n_tokens"))
        )
        context_tokens = (
            _coerce_int(err.get("n_ctx"))
            or _coerce_int(err.get("context_size"))
            or _coerce_int(err.get("ctx"))
        )
    if not prompt_tokens:
        match = re.search(r"request\s*\((\d+)\s+tokens?\)", text, re.IGNORECASE)
        if match:
            prompt_tokens = _coerce_int(match.group(1))
    if not context_tokens:
        match = re.search(r"context size\s*\((\d+)\s+tokens?\)", text, re.IGNORECASE)
        if match:
            context_tokens = _coerce_int(match.group(1))
    if not context_tokens:
        match = re.search(r'"n_ctx"\s*:\s*(\d+)', text, re.IGNORECASE)
        if match:
            context_tokens = _coerce_int(match.group(1))
    return prompt_tokens, context_tokens


def _backend_message(text: str, payload: Any) -> str:
    if payload is None:
        return text
    try:
        return native_error_message(json.dumps(payload))
    except Exception:
        return text


def explain_llm_error(raw: Any, *, source: str = "LLM engine") -> LlmErrorNotice:
    text = _raw_text(raw).strip()
    payload = _json_payload(text)
    backend_message = _backend_message(text, payload)
    lower = f"{text}\n{backend_message}".lower()
    prompt_tokens, context_tokens = _extract_token_counts(text, payload)
    source_label = str(source or "LLM engine").strip() or "LLM engine"

    if native_is_context_error(lower):
        if prompt_tokens and context_tokens:
            summary = (
                f"{source_label} could not run because the request is about "
                f"{prompt_tokens:,} prompt tokens, but the loaded model/server "
                f"currently has only {context_tokens:,} context tokens available."
            )
        else:
            summary = (
                f"{source_label} could not run because the request is larger "
                "than the model's current context window."
            )
        action = (
            "- Send less text, remove some reference material, or shorten the pipeline input.\n"
            "- If this happened in a pipeline, reduce earlier block output before the model block.\n"
            "- Increase the model context size in NativeLab model/server settings, then reload the model.\n"
            "- Use a model/runtime with a larger context window if your RAM/VRAM can handle it."
        )
        return LlmErrorNotice(
            category="context_size",
            title="Model Context Is Full",
            summary=summary,
            action=action,
            technical_detail=text or backend_message,
            prompt_tokens=prompt_tokens,
            context_tokens=context_tokens,
        )

    if any(word in lower for word in ("timed out", "timeout", "stream stalled", "no tokens for")):
        return LlmErrorNotice(
            category="timeout",
            title="Model Response Timed Out",
            summary=f"{source_label} stopped because the model did not return tokens in time.",
            action=(
                "- Try the request again.\n"
                "- Reduce prompt size or output length if the request is large.\n"
                "- Reload the model if the server appears stuck."
            ),
            technical_detail=text,
        )

    if any(word in lower for word in ("connection refused", "connection reset", "connection aborted", "broken pipe")):
        return LlmErrorNotice(
            category="connection",
            title="Model Server Connection Failed",
            summary=f"{source_label} could not reach the local model server.",
            action=(
                "- Reload the model from the Models tab.\n"
                "- Check that llama-server/Ollama/HF backend is still running.\n"
                "- If this keeps happening, restart NativeLab and load the model again."
            ),
            technical_detail=text,
        )

    if "http 400" in lower or "bad request" in lower:
        return LlmErrorNotice(
            category="bad_request",
            title="Model Request Was Rejected",
            summary=f"{source_label} sent a request that the backend rejected.",
            action=(
                "- Try a shorter prompt first.\n"
                "- Check model settings such as context size, output tokens, and sampler values.\n"
                "- Reload the model after changing backend settings."
            ),
            technical_detail=text,
        )

    if "not installed" in lower or "dependencies missing" in lower or "modulenotfounderror" in lower:
        return LlmErrorNotice(
            category="missing_dependency",
            title="Model Backend Dependency Missing",
            summary=f"{source_label} cannot run because a backend library is missing.",
            action=(
                "- Open Downloads or Help and install the required backend libraries.\n"
                "- For Hugging Face models, install the Transformers dependencies from inside NativeLab.\n"
                "- Restart NativeLab after installation if the import still fails."
            ),
            technical_detail=text,
        )

    return LlmErrorNotice(
        category="generic",
        title="LLM Engine Error",
        summary=f"{source_label} failed while generating a response.",
        action=(
            "- Try the request again with less input text.\n"
            "- Reload the active model if the backend may be stuck.\n"
            "- Check the technical details below if this repeats."
        ),
        technical_detail=text,
    )
