from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


INTEGRATIONS_DIR = Path("./localllm/integrations")
DISCORD_BOTS_FILE = INTEGRATIONS_DIR / "discord_bots.json"

DISCORD_PERMISSION_BITS = {
    "view_channels": 1024,
    "send_messages": 2048,
    "embed_links": 16384,
    "attach_files": 32768,
    "read_message_history": 65536,
    "use_external_emojis": 262144,
}

DEFAULT_DISCORD_SYSTEM_PROMPT = (
    "You are NativeLab replying inside Discord. Keep answers concise."
)

DEFAULT_DISCORD_BOT: Dict[str, Any] = {
    "name": "",
    "enabled": True,
    "token": "",
    "application_id": "",
    "guild_id": "",
    "endpoint_url": "http://127.0.0.1:8765",
    "reply": {
        "mode": "interaction_reply",
        "ephemeral": False,
        "max_chars": 1900,
        "direct_mentions": False,
    },
    "queue": {
        "enabled": True,
        "max_concurrent": 1,
        "max_queued": 12,
    },
    "privileges": {
        "slash_commands": True,
        "message_content_intent": False,
        "view_channels": True,
        "send_messages": True,
        "embed_links": True,
        "attach_files": False,
        "read_message_history": True,
        "use_external_emojis": False,
    },
    "capabilities": {
        "ask_model": True,
        "runtime": True,
        "pipelines": True,
        "labs": True,
        "models": True,
    },
    "system_prompt": DEFAULT_DISCORD_SYSTEM_PROMPT,
    "notes": "",
}


def command_catalog(config: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    caps = (config or DEFAULT_DISCORD_BOT).get("capabilities", {})
    reply = (config or DEFAULT_DISCORD_BOT).get("reply", {})
    rows = [{"command": "/help", "description": "Show available NativeLab bot commands."}]
    if reply.get("direct_mentions", False) and caps.get("ask_model", True):
        rows.append({"command": "@Bot <message>", "description": "Ask the active model by mentioning the bot."})
    if caps.get("ask_model", True):
        rows.append({"command": "/ask prompt:<text>", "description": "Send a prompt to the active NativeLab model."})
    if caps.get("runtime", True):
        rows.append({"command": "/status", "description": "Show active backend, loaded model, and status."})
    if caps.get("pipelines", True):
        rows.extend([
            {"command": "/pipelines", "description": "List saved NativeLab pipelines."},
            {"command": "/pipeline name:<pipeline>", "description": "Show a saved pipeline definition summary."},
        ])
    if caps.get("labs", True):
        rows.extend([
            {"command": "/labs", "description": "List available Labs routes."},
            {"command": "/lab name:<lab>", "description": "Show metadata for a Labs route such as py_to_doc."},
        ])
    if caps.get("models", True):
        rows.append({"command": "/models", "description": "List local and API models visible to integrations."})
    return rows


def load_discord_bots() -> List[Dict[str, Any]]:
    if not DISCORD_BOTS_FILE.exists():
        return []
    try:
        raw = json.loads(DISCORD_BOTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = raw.get("bots", raw if isinstance(raw, list) else [])
    return [_merge_defaults(item) for item in items if isinstance(item, dict)]


def save_discord_bots(bots: List[Dict[str, Any]]) -> None:
    INTEGRATIONS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"bots": [_merge_defaults(bot) for bot in bots]}
    DISCORD_BOTS_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def upsert_discord_bot(config: Dict[str, Any]) -> Dict[str, Any]:
    clean = _merge_defaults(config)
    name = str(clean.get("name", "")).strip()
    if not name:
        raise ValueError("Discord bot profile name is required")
    clean["name"] = name

    bots = load_discord_bots()
    for i, bot in enumerate(bots):
        if bot.get("name") == name:
            bots[i] = clean
            save_discord_bots(bots)
            return clean
    bots.append(clean)
    save_discord_bots(bots)
    return clean


def delete_discord_bot(name: str) -> None:
    bots = [b for b in load_discord_bots() if b.get("name") != name]
    save_discord_bots(bots)


def get_discord_bot(name: str | None = None) -> Dict[str, Any] | None:
    bots = load_discord_bots()
    if not bots:
        return None
    if not name:
        return bots[0]
    for bot in bots:
        if bot.get("name") == name:
            return bot
    return None


def invite_url(config: Dict[str, Any]) -> str:
    app_id = str(config.get("application_id", "")).strip()
    if not app_id:
        return ""
    permissions = 0
    priv = config.get("privileges", {})
    for key, bit in DISCORD_PERMISSION_BITS.items():
        if priv.get(key, False):
            permissions |= bit
    scopes = "bot%20applications.commands"
    return (
        "https://discord.com/api/oauth2/authorize"
        f"?client_id={app_id}&permissions={permissions}&scope={scopes}"
    )


def _merge_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_DISCORD_BOT)
    for key, value in (config or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged
