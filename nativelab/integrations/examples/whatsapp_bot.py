"""
WhatsApp Cloud API webhook runtime for saved NativeLab connector profiles.

Create a profile in NativeLab > Integrations > WhatsApp Bot, expose the webhook
port with a public tunnel, configure the Meta webhook, then run from the app or:
    WHATSAPP_BOT_PROFILE=bot1 python -m nativelab.integrations.examples.whatsapp_bot
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict

if __package__ in {None, ""}:
    for parent in Path(__file__).resolve().parents:
        if (parent / "nativelab").is_dir():
            sys.path.insert(0, str(parent))
            break

try:
    import aiohttp
    from aiohttp import web
except ModuleNotFoundError as e:
    missing = e.name or "aiohttp"
    raise SystemExit(
        f"Missing WhatsApp bot dependency: {missing}. "
        "Install NativeLab package dependencies or run `python -m pip install aiohttp`."
    ) from e

from nativelab.integrations.whatsapp_connector import (
    DEFAULT_WHATSAPP_BOT,
    DEFAULT_WHATSAPP_SYSTEM_PROMPT,
    command_catalog,
    get_whatsapp_bot,
)


PROFILE_NAME = os.getenv("WHATSAPP_BOT_PROFILE", "")
CONFIG = get_whatsapp_bot(PROFILE_NAME) or dict(DEFAULT_WHATSAPP_BOT)
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", CONFIG.get("access_token", ""))
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", CONFIG.get("phone_number_id", ""))
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", CONFIG.get("verify_token", "nativelab-whatsapp"))
ENDPOINT_URL = os.getenv(
    "NATIVELAB_INTEGRATION_URL",
    CONFIG.get("endpoint_url", "http://127.0.0.1:8765"),
).rstrip("/")

WEBHOOK_HOST = os.getenv("WHATSAPP_WEBHOOK_HOST", CONFIG.get("webhook_host", "127.0.0.1"))
WEBHOOK_PORT = int(os.getenv("WHATSAPP_WEBHOOK_PORT", str(CONFIG.get("webhook_port", 8770))))
WEBHOOK_PATH = os.getenv("WHATSAPP_WEBHOOK_PATH", CONFIG.get("webhook_path", "/webhook"))
if not WEBHOOK_PATH.startswith("/"):
    WEBHOOK_PATH = "/" + WEBHOOK_PATH

REPLY = CONFIG.get("reply", {})
QUEUE = CONFIG.get("queue", {})
CAPS = CONFIG.get("capabilities", {})
SYSTEM_PROMPT = CONFIG.get("system_prompt") or DEFAULT_WHATSAPP_SYSTEM_PROMPT

MAX_REPLY_CHARS = int(REPLY.get("max_chars", 3500))
DIRECT_MESSAGES = bool(REPLY.get("direct_messages", True))
QUEUE_ENABLED = bool(QUEUE.get("enabled", True))
MAX_CONCURRENT = max(1, int(QUEUE.get("max_concurrent", 1)))
MAX_QUEUED = max(1, int(QUEUE.get("max_queued", 12)))

_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue_lock = asyncio.Lock()
_queued = 0


def log(message: str):
    print(f"[NativeLab WhatsApp] {message}", flush=True)


def clip(text: str) -> str:
    text = str(text or "").strip() or "(empty)"
    return text[:MAX_REPLY_CHARS]


def require_capability(name: str):
    if not CAPS.get(name, True):
        raise RuntimeError(f"This WhatsApp profile does not allow '{name}'.")


async def run_queued(factory: Callable[[], Awaitable[str]]) -> str:
    global _queued
    if not QUEUE_ENABLED:
        return await factory()
    async with _queue_lock:
        if _queued >= MAX_QUEUED:
            log("Queue full; rejecting request")
            raise RuntimeError("NativeLab WhatsApp request queue is full. Try again in a moment.")
        _queued += 1
        log(f"Queued request; active/queued count is now {_queued}")
    try:
        async with _semaphore:
            return await factory()
    finally:
        async with _queue_lock:
            _queued = max(0, _queued - 1)
            log(f"Request finished; active/queued count is now {_queued}")


async def post_native(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    log(f"NativeLab POST {path}")
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{ENDPOINT_URL}{path}", json=payload) as response:
            data = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError(data.get("error", f"HTTP {response.status}"))
            return data


async def get_native(path: str) -> Dict[str, Any]:
    log(f"NativeLab GET {path}")
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{ENDPOINT_URL}{path}") as response:
            data = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError(data.get("error", f"HTTP {response.status}"))
            return data


async def send_whatsapp(to: str, text: str):
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        raise RuntimeError("WhatsApp access token and phone number ID are required")
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": clip(text)},
    }
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as response:
            raw = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"WhatsApp send HTTP {response.status}: {raw[:300]}")
            log(f"Sent WhatsApp reply to {to}")


async def help_text() -> str:
    lines = ["NativeLab WhatsApp commands:"]
    for row in command_catalog(CONFIG):
        lines.append(f"{row['command']} - {row['description']}")
    return "\n".join(lines)


async def answer_text(text: str) -> str:
    prompt = text.strip()
    if prompt.startswith("/ask "):
        require_capability("ask_model")
        prompt = prompt[5:].strip()
    elif prompt == "/help":
        return await help_text()
    elif prompt == "/status":
        require_capability("runtime")
        data = await get_native("/runtime")
        return f"NativeLab: {data.get('status', 'unknown')}\nModel: {data.get('model_name') or 'no model'}\nBackend: {data.get('backend', 'unknown')}"
    elif prompt == "/pipelines":
        require_capability("pipelines")
        data = await get_native("/pipelines")
        names = [p.get("name", "") for p in data.get("pipelines", []) if p.get("name")]
        return "\n".join(names[:40]) if names else "No saved pipelines found."
    elif prompt.startswith("/pipeline "):
        require_capability("pipelines")
        name = prompt.split(" ", 1)[1].strip()
        data = await get_native(f"/pipelines/{name}")
        if data.get("error"):
            return data["error"]
        definition = data.get("definition", {})
        return f"{data.get('name', name)}: {len(definition.get('blocks', []))} block(s), {len(definition.get('connections', []))} connection(s)"
    elif prompt == "/labs":
        require_capability("labs")
        data = await get_native("/labs")
        rows = [f"{lab.get('route')} - {lab.get('name')}" for lab in data.get("labs", [])]
        return "\n".join(rows) if rows else "No labs registered."
    elif prompt.startswith("/lab "):
        require_capability("labs")
        name = prompt.split(" ", 1)[1].strip()
        data = await get_native(f"/labs/{name}")
        return data.get("error") or f"{data.get('route')} - actions: {len(data.get('actions', []))}"
    elif prompt == "/models":
        require_capability("models")
        local = await get_native("/models")
        api = await get_native("/api_models")
        local_names = [m.get("name") or m.get("path", "") for m in local.get("models", [])]
        api_names = [m.get("name") or m.get("model_id", "") for m in api.get("api_models", [])]
        return "\n".join([f"Local models: {len(local_names)}", *local_names[:12], "", f"API models: {len(api_names)}", *api_names[:12]])
    else:
        if not DIRECT_MESSAGES:
            return await help_text()
        require_capability("ask_model")

    if not prompt:
        return await help_text()

    async def work():
        data = await post_native(
            "/call_llm",
            {
                "prompt": prompt,
                "system_prompt": SYSTEM_PROMPT,
                "n_predict": 700,
                "temperature": 0.4,
            },
        )
        return data.get("text", "")

    return await run_queued(work)


def extract_messages(payload: Dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                sender = msg.get("from", "")
                text = msg.get("text", {}).get("body", "")
                if sender and text:
                    out.append((sender, text))
    return out


async def verify(request: web.Request):
    mode = request.query.get("hub.mode")
    token = request.query.get("hub.verify_token")
    challenge = request.query.get("hub.challenge", "")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        log("Webhook verification succeeded")
        return web.Response(text=challenge)
    log("Webhook verification failed")
    return web.Response(status=403, text="verification failed")


async def webhook(request: web.Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    messages = extract_messages(payload)
    log(f"Webhook received {len(messages)} text message(s)")
    for sender, text in messages:
        log(f"Message from {sender}: {text[:120]}")
        try:
            reply = await answer_text(text)
            await send_whatsapp(sender, reply)
        except Exception as e:
            log(f"Message handling error: {e}")
            try:
                await send_whatsapp(sender, f"NativeLab WhatsApp bot error: {e}")
            except Exception as send_error:
                log(f"Could not send error response: {send_error}")
    return web.json_response({"ok": True})


def main():
    if not ACCESS_TOKEN:
        raise SystemExit("Set WhatsApp access token in the selected NativeLab profile or WHATSAPP_ACCESS_TOKEN.")
    if not PHONE_NUMBER_ID:
        raise SystemExit("Set WhatsApp phone number ID in the selected NativeLab profile or WHATSAPP_PHONE_NUMBER_ID.")
    profile = CONFIG.get("name") or PROFILE_NAME or "default"
    log(f"Starting profile '{profile}'")
    log(f"NativeLab endpoint: {ENDPOINT_URL}")
    log(f"Webhook: http://{WEBHOOK_HOST}:{WEBHOOK_PORT}{WEBHOOK_PATH}")
    log(f"Verify token: {VERIFY_TOKEN}")
    app = web.Application()
    app.router.add_get(WEBHOOK_PATH, verify)
    app.router.add_post(WEBHOOK_PATH, webhook)
    web.run_app(app, host=WEBHOOK_HOST, port=WEBHOOK_PORT, print=None)


if __name__ == "__main__":
    main()
