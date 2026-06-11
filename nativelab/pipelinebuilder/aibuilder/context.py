from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from nativelab.GlobalConfig.config_global import MODELS_DIR


AI_BUILDER_HISTORY_DIR = MODELS_DIR / "pipeline_builder_history"
MAX_HISTORY_ITEMS = 40
RECENT_HISTORY_ITEMS = 8


@dataclass(frozen=True)
class SmartContextRequest:
    command: str
    model_request: str
    user_request: str
    notice: str = ""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_session_id(session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(session_id or "default")).strip(".-")
    return safe[:80] or "default"


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def history_dir(root: Optional[Any] = None):
    path = Path(root) if root is not None else AI_BUILDER_HISTORY_DIR
    if root is not None and not path.is_absolute():
        path = MODELS_DIR / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_canvas_empty(canvas_state: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(canvas_state, dict):
        return True
    return not bool(canvas_state.get("blocks"))


def format_canvas_state(canvas_state: Optional[Dict[str, Any]]) -> str:
    state = canvas_state if isinstance(canvas_state, dict) else {}
    return _json({
        "tool": "/get_data",
        "result": state,
    })


def pipeline_digest(data: Optional[Dict[str, Any]]) -> str:
    if not isinstance(data, dict):
        return "Pipeline saved."
    blocks = data.get("blocks") if isinstance(data.get("blocks"), list) else []
    conns = data.get("connections") if isinstance(data.get("connections"), list) else []
    labels = []
    for block in blocks[:12]:
        if not isinstance(block, dict):
            continue
        label = str(block.get("label") or block.get("btype") or "block")
        btype = str(block.get("btype") or "")
        labels.append(f"{label} [{btype}]")
    suffix = "..." if len(blocks) > 12 else ""
    return (
        f"Saved pipeline with {len(blocks)} block(s), {len(conns)} connection(s). "
        f"Blocks: {', '.join(labels)}{suffix}"
    )


class AiBuilderHistoryStore:
    def __init__(self, session_id: str = "default", root: Optional[Any] = None):
        self.session_id = _safe_session_id(session_id)
        self.root = history_dir(root)
        self.path = self.root / f"{self.session_id}.json"
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "summary": "", "messages": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "summary": "", "messages": []}
        if not isinstance(data, dict):
            return {"version": 1, "summary": "", "messages": []}
        if not isinstance(data.get("messages"), list):
            data["messages"] = []
        data.setdefault("version", 1)
        data.setdefault("summary", "")
        return data

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(_json(self._data), encoding="utf-8")

    @property
    def summary(self) -> str:
        return str(self._data.get("summary") or "")

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return [m for m in self._data.get("messages", []) if isinstance(m, dict)]

    def is_empty(self) -> bool:
        return not self.summary.strip() and not self.messages

    def append(self, role: str, content: str, **extra: Any):
        item = {
            "ts": _now(),
            "role": str(role or "user"),
            "content": str(content or ""),
        }
        item.update(extra)
        messages = self.messages + [item]
        if len(messages) > MAX_HISTORY_ITEMS:
            messages = messages[-MAX_HISTORY_ITEMS:]
        self._data["messages"] = messages
        self.save()

    def append_user(self, content: str):
        self.append("user", content)

    def append_assistant(self, content: str, **extra: Any):
        self.append("assistant", content, **extra)

    def clear(self):
        self._data = {"version": 1, "summary": "", "messages": []}
        self.save()

    def compact(self, summary: str, keep_recent: int = 2):
        keep = max(0, int(keep_recent or 0))
        self._data["summary"] = str(summary or "").strip()
        self._data["messages"] = self.messages[-keep:] if keep else []
        self.save()

    def recent_text(self, limit: int = RECENT_HISTORY_ITEMS) -> str:
        rows = self.messages[-max(0, int(limit or 0)):]
        if not rows:
            return ""
        lines = []
        for row in rows:
            role = str(row.get("role") or "user")
            content = str(row.get("content") or "").strip()
            if len(content) > 1200:
                content = content[:1200] + "\n...[truncated]"
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)


def build_smart_context_request(
    user_request: str,
    *,
    history: AiBuilderHistoryStore,
    canvas_state: Optional[Dict[str, Any]] = None,
) -> SmartContextRequest:
    raw = str(user_request or "").strip()
    command = raw.split(None, 1)[0].lower() if raw.startswith("/") else ""
    if command == "/get_data":
        return SmartContextRequest(
            command="get_data",
            user_request=raw,
            model_request="",
            notice=format_canvas_state(canvas_state),
        )
    if command == "/context":
        return SmartContextRequest(
            command="context",
            user_request=raw,
            model_request="",
            notice="Compacting AI Builder history.",
        )

    canvas_empty = is_canvas_empty(canvas_state)
    if history.is_empty() and canvas_empty:
        return SmartContextRequest(command="", user_request=raw, model_request=raw)

    parts = [
        "You are editing a NativeLab pipeline over multiple AI Builder prompts.",
        "Return the full updated NativeLab pipeline JSON, not a patch.",
        f"Current user request:\n{raw}",
    ]
    if history.summary.strip():
        parts.append(f"Compact prior AI Builder context:\n{history.summary.strip()}")
    recent = history.recent_text()
    if recent:
        parts.append(f"Recent AI Builder turns:\n{recent}")
    if not canvas_empty:
        first = (
            "This is the user's first AI Builder prompt for an existing canvas."
            if history.is_empty()
            else "The current canvas may already include edits from earlier prompts."
        )
        parts.append(
            f"{first}\n"
            "Pseudo-tool available to the active model: /get_data.\n"
            "The /get_data result is provided below so you can edit the current pipeline efficiently:\n"
            f"{format_canvas_state(canvas_state)}"
        )
    return SmartContextRequest(
        command="",
        user_request=raw,
        model_request="\n\n---\n\n".join(parts),
        notice="Smart context attached current canvas/history." if not canvas_empty or not history.is_empty() else "",
    )


def deterministic_compact(history: AiBuilderHistoryStore, canvas_state: Optional[Dict[str, Any]] = None) -> str:
    canvas = "empty canvas" if is_canvas_empty(canvas_state) else "canvas has existing blocks"
    recent = history.recent_text(limit=6) or "(no recent turns)"
    return f"AI Builder context compacted locally. Current canvas: {canvas}.\n\nRecent turns:\n{recent}"
