from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from nativelab.GlobalConfig.config_global import DEFAULT_CTX
from nativelab.imports.qt_compat import QObject, pyqtSignal


def estimate_tokens(text: str) -> int:
    """Fast cross-backend estimate used for UI context fill when no tokenizer is exposed."""
    text = str(text or "")
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content or "")


def messages_text(messages: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        lines.append(f"{role}: {_content_text(msg.get('content', ''))}")
    return "\n".join(lines)


def _engine_limit(engine: Any = None, fallback: int = 0) -> int:
    try:
        value = int(getattr(engine, "ctx_value", 0) or 0)
    except Exception:
        value = 0
    if value <= 0:
        try:
            value = int(fallback or 0)
        except Exception:
            value = 0
    return value if value > 0 else int(DEFAULT_CTX())


def _engine_model(engine: Any = None) -> str:
    if engine is None:
        return ""
    cfg = getattr(engine, "_config", None)
    if cfg is not None:
        provider = str(getattr(cfg, "custom_provider_name", "") or getattr(cfg, "provider", "") or "")
        model_id = str(getattr(cfg, "model_id", "") or "")
        return f"{provider} · {model_id}" if provider and model_id else model_id
    model_path = str(getattr(engine, "model_path", "") or "")
    if not model_path:
        return ""
    if ":" in model_path and not Path(model_path).exists():
        return model_path.split(":", 1)[-1].strip() or model_path
    return Path(model_path).name


class ContextMeter(QObject):
    updated = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._snapshot: Dict[str, Any] = {}
        self._output_chars = 0

    @property
    def snapshot(self) -> Dict[str, Any]:
        return dict(self._snapshot)

    def report_prompt(
        self,
        *,
        source: str,
        engine: Any = None,
        prompt: str = "",
        n_predict: int = 0,
        limit_tokens: int = 0,
        mode: str = "",
    ) -> Dict[str, Any]:
        input_tokens = estimate_tokens(prompt)
        return self._set_snapshot(
            source=source,
            engine=engine,
            input_tokens=input_tokens,
            prompt_chars=len(str(prompt or "")),
            n_predict=n_predict,
            limit_tokens=limit_tokens,
            mode=mode,
        )

    def report_messages(
        self,
        *,
        source: str,
        engine: Any = None,
        messages: Iterable[Dict[str, Any]],
        n_predict: int = 0,
        limit_tokens: int = 0,
        mode: str = "",
    ) -> Dict[str, Any]:
        text = messages_text(messages)
        return self.report_prompt(
            source=source,
            engine=engine,
            prompt=text,
            n_predict=n_predict,
            limit_tokens=limit_tokens,
            mode=mode,
        )

    def report_session(
        self,
        *,
        source: str,
        used_tokens: int,
        limit_tokens: int,
        engine: Any = None,
    ) -> Dict[str, Any]:
        return self._set_snapshot(
            source=source,
            engine=engine,
            input_tokens=max(0, int(used_tokens or 0)),
            prompt_chars=0,
            n_predict=0,
            limit_tokens=limit_tokens,
            mode="session",
        )

    def append_output(self, text: str) -> Dict[str, Any]:
        if not self._snapshot:
            return {}
        self._output_chars += len(str(text or ""))
        output_tokens = estimate_tokens("x" * self._output_chars)
        snap = dict(self._snapshot)
        snap["output_tokens"] = output_tokens
        snap["used_tokens"] = int(snap.get("input_tokens", 0)) + output_tokens
        self._snapshot = snap
        self.updated.emit(dict(snap))
        return snap

    def _set_snapshot(
        self,
        *,
        source: str,
        engine: Any,
        input_tokens: int,
        prompt_chars: int,
        n_predict: int,
        limit_tokens: int,
        mode: str,
    ) -> Dict[str, Any]:
        self._output_chars = 0
        try:
            reserved = max(0, int(n_predict or 0))
        except Exception:
            reserved = 0
        limit = _engine_limit(engine, limit_tokens)
        input_tokens = max(0, int(input_tokens or 0))
        projected = input_tokens + reserved
        snap = {
            "source": str(source or "LLM"),
            "mode": mode or str(getattr(engine, "mode", "") or ""),
            "model": _engine_model(engine),
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "reserved_tokens": reserved,
            "projected_tokens": projected,
            "used_tokens": input_tokens,
            "limit_tokens": limit,
            "prompt_chars": int(prompt_chars or 0),
        }
        self._snapshot = snap
        self.updated.emit(dict(snap))
        return snap


context_meter = ContextMeter()
