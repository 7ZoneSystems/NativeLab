from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


INTEGRATIONS_DIR = Path("./localllm/integrations")
WHATSAPP_BOTS_FILE = INTEGRATIONS_DIR / "whatsapp_bots.json"

DEFAULT_WHATSAPP_SYSTEM_PROMPT = (
    "You are NativeLab replying inside WhatsApp. Keep answers concise."
)

DEFAULT_WHATSAPP_BOT: Dict[str, Any] = {
    "name": "",
    "enabled": True,
    "access_token": "",
    "phone_number_id": "",
    "business_account_id": "",
    "verify_token": "nativelab-whatsapp",
    "endpoint_url": "http://127.0.0.1:8765",
    "webhook_host": "127.0.0.1",
    "webhook_port": 8770,
    "webhook_path": "/webhook",
    "reply": {
        "max_chars": 3500,
        "direct_messages": True,
    },
    "queue": {
        "enabled": True,
        "max_concurrent": 1,
        "max_queued": 12,
    },
    "capabilities": {
        "ask_model": True,
        "runtime": True,
        "pipelines": True,
        "labs": True,
        "models": True,
    },
    "system_prompt": DEFAULT_WHATSAPP_SYSTEM_PROMPT,
    "notes": "",
}


def command_catalog(config: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    cfg = config or DEFAULT_WHATSAPP_BOT
    caps = cfg.get("capabilities", {})
    rows = [{"command": "/help", "description": "Show available NativeLab WhatsApp commands."}]
    if cfg.get("reply", {}).get("direct_messages", True) and caps.get("ask_model", True):
        rows.append({"command": "<message>", "description": "Ask the active model with any normal message."})
    if caps.get("ask_model", True):
        rows.append({"command": "/ask <prompt>", "description": "Ask the active NativeLab model."})
    if caps.get("runtime", True):
        rows.append({"command": "/status", "description": "Show active backend, loaded model, and status."})
    if caps.get("pipelines", True):
        rows.extend([
            {"command": "/pipelines", "description": "List saved NativeLab pipelines."},
            {"command": "/pipeline <name>", "description": "Show a saved pipeline summary."},
        ])
    if caps.get("labs", True):
        rows.extend([
            {"command": "/labs", "description": "List available Labs routes."},
            {"command": "/lab <name>", "description": "Show metadata for a Labs route."},
        ])
    if caps.get("models", True):
        rows.append({"command": "/models", "description": "List local and API models visible to integrations."})
    return rows


def load_whatsapp_bots() -> List[Dict[str, Any]]:
    if not WHATSAPP_BOTS_FILE.exists():
        return []
    try:
        raw = json.loads(WHATSAPP_BOTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = raw.get("bots", raw if isinstance(raw, list) else [])
    return [_merge_defaults(item) for item in items if isinstance(item, dict)]


def save_whatsapp_bots(bots: List[Dict[str, Any]]) -> None:
    INTEGRATIONS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"bots": [_merge_defaults(bot) for bot in bots]}
    WHATSAPP_BOTS_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def upsert_whatsapp_bot(config: Dict[str, Any]) -> Dict[str, Any]:
    clean = _merge_defaults(config)
    name = str(clean.get("name", "")).strip()
    if not name:
        raise ValueError("WhatsApp bot profile name is required")
    clean["name"] = name

    bots = load_whatsapp_bots()
    for i, bot in enumerate(bots):
        if bot.get("name") == name:
            bots[i] = clean
            save_whatsapp_bots(bots)
            return clean
    bots.append(clean)
    save_whatsapp_bots(bots)
    return clean


def delete_whatsapp_bot(name: str) -> None:
    bots = [b for b in load_whatsapp_bots() if b.get("name") != name]
    save_whatsapp_bots(bots)


def get_whatsapp_bot(name: str | None = None) -> Dict[str, Any] | None:
    bots = load_whatsapp_bots()
    if not bots:
        return None
    if not name:
        return bots[0]
    for bot in bots:
        if bot.get("name") == name:
            return bot
    return None


def _merge_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_WHATSAPP_BOT)
    for key, value in (config or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged

