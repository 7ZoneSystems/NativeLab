"""
Discord bot runtime for saved NativeLab connector profiles.

Setup:
    python -m pip install discord.py aiohttp

Create a profile in NativeLab > Integrations > Discord Bot, then run:
    DISCORD_BOT_PROFILE=bot1 python nativelab/integrations/examples/discord_bot.py
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
    import discord
    from discord import app_commands
    from discord.ext import commands
except ModuleNotFoundError as e:
    missing = e.name or "discord.py/aiohttp"
    raise SystemExit(
        "Missing Discord bot dependency: "
        f"{missing}. Install NativeLab package dependencies or run "
        "`python -m pip install discord.py aiohttp`."
    ) from e

from nativelab.integrations.discord_connector import (
    DEFAULT_DISCORD_BOT,
    DEFAULT_DISCORD_SYSTEM_PROMPT,
    command_catalog,
    get_discord_bot,
)


PROFILE_NAME = os.getenv("DISCORD_BOT_PROFILE", "")
CONFIG = get_discord_bot(PROFILE_NAME) or dict(DEFAULT_DISCORD_BOT)
TOKEN = os.getenv("DISCORD_BOT_TOKEN", CONFIG.get("token", ""))
ENDPOINT_URL = os.getenv(
    "NATIVELAB_INTEGRATION_URL",
    CONFIG.get("endpoint_url", "http://127.0.0.1:8765"),
).rstrip("/")

REPLY = CONFIG.get("reply", {})
QUEUE = CONFIG.get("queue", {})
PRIV = CONFIG.get("privileges", {})
CAPS = CONFIG.get("capabilities", {})
SYSTEM_PROMPT = CONFIG.get("system_prompt") or DEFAULT_DISCORD_SYSTEM_PROMPT

MAX_REPLY_CHARS = int(REPLY.get("max_chars", 1900))
EPHEMERAL = bool(REPLY.get("ephemeral", False))
DIRECT_MENTIONS = bool(REPLY.get("direct_mentions", False))
QUEUE_ENABLED = bool(QUEUE.get("enabled", True))
MAX_CONCURRENT = max(1, int(QUEUE.get("max_concurrent", 1)))
MAX_QUEUED = max(1, int(QUEUE.get("max_queued", 12)))

_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue_lock = asyncio.Lock()
_queued = 0


intents = discord.Intents.default()
intents.message_content = bool(PRIV.get("message_content_intent", False) or DIRECT_MENTIONS)


class NativeLabDiscordBot(commands.Bot):
    async def setup_hook(self):
        guild_id = str(CONFIG.get("guild_id", "")).strip()
        if guild_id:
            guild = discord.Object(id=int(guild_id))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()


bot = NativeLabDiscordBot(command_prefix=commands.when_mentioned, intents=intents)


def log(message: str):
    print(f"[NativeLab Discord] {message}", flush=True)


async def post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    log(f"POST {path}")
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{ENDPOINT_URL}{path}", json=payload) as response:
            data = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError(data.get("error", f"HTTP {response.status}"))
            return data


async def get_json(path: str) -> Dict[str, Any]:
    log(f"GET {path}")
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{ENDPOINT_URL}{path}") as response:
            data = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError(data.get("error", f"HTTP {response.status}"))
            return data


async def run_queued(factory: Callable[[], Awaitable[str]]) -> str:
    global _queued
    if not QUEUE_ENABLED:
        return await factory()

    async with _queue_lock:
        if _queued >= MAX_QUEUED:
            log("Queue full; rejecting request")
            raise RuntimeError("NativeLab request queue is full. Try again in a moment.")
        _queued += 1
        log(f"Queued request; active/queued count is now {_queued}")
    try:
        async with _semaphore:
            return await factory()
    finally:
        async with _queue_lock:
            _queued = max(0, _queued - 1)
            log(f"Request finished; active/queued count is now {_queued}")


def clip(text: str) -> str:
    text = str(text or "").strip() or "(empty)"
    return text[:MAX_REPLY_CHARS]


async def respond(interaction: discord.Interaction, text: str):
    text = clip(text)
    if REPLY.get("mode") == "channel_message" and interaction.channel is not None:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=EPHEMERAL, thinking=True)
        await interaction.channel.send(text)
        return
    if interaction.response.is_done():
        await interaction.followup.send(text, ephemeral=EPHEMERAL)
    else:
        await interaction.response.send_message(text, ephemeral=EPHEMERAL)


async def reply_to_message(message: discord.Message, text: str):
    await message.reply(clip(text), mention_author=False)


def require_capability(name: str):
    if not CAPS.get(name, True):
        raise RuntimeError(f"This bot profile does not allow '{name}'.")


def clean_mention_prompt(content: str, bot_id: int) -> str:
    text = content or ""
    for token in (f"<@{bot_id}>", f"<@!{bot_id}>"):
        text = text.replace(token, " ")
    return " ".join(text.split())


@bot.event
async def on_ready():
    profile = CONFIG.get("name") or PROFILE_NAME or "default"
    log(f"Ready as {bot.user} using profile '{profile}'")
    log(f"Endpoint: {ENDPOINT_URL}")
    if DIRECT_MENTIONS:
        log("Direct @mention replies are enabled")


@bot.tree.command(name="help", description="Show NativeLab bot commands available in this profile.")
async def help_command(interaction: discord.Interaction):
    log(f"/help by {interaction.user}")
    lines = ["NativeLab commands:"]
    for row in command_catalog(CONFIG):
        lines.append(f"{row['command']} - {row['description']}")
    await respond(interaction, "\n".join(lines))


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if not DIRECT_MENTIONS:
        return
    if bot.user is None or bot.user not in message.mentions:
        return
    log(f"@mention by {message.author}")
    try:
        require_capability("ask_model")
        prompt = clean_mention_prompt(message.content, bot.user.id)
        if not prompt:
            lines = ["Mention me with a prompt, or use slash commands:"]
            for row in command_catalog(CONFIG):
                lines.append(f"{row['command']} - {row['description']}")
            await reply_to_message(message, "\n".join(lines))
            return

        async def work():
            data = await post_json(
                "/call_llm",
                {
                    "prompt": prompt,
                    "system_prompt": SYSTEM_PROMPT,
                    "n_predict": 700,
                    "temperature": 0.4,
                },
            )
            return data.get("text", "")

        if message.channel is not None:
            async with message.channel.typing():
                text = await run_queued(work)
        else:
            text = await run_queued(work)
        await reply_to_message(message, text)
    except Exception as e:
        log(f"@mention error: {e}")
        await reply_to_message(message, f"NativeLab bot error: {e}")


@bot.tree.command(name="ask", description="Ask the active NativeLab model.")
@app_commands.describe(prompt="Prompt to send to NativeLab")
async def ask(interaction: discord.Interaction, prompt: str):
    log(f"/ask by {interaction.user}")
    require_capability("ask_model")
    await interaction.response.defer(ephemeral=EPHEMERAL, thinking=True)

    async def work():
        data = await post_json(
            "/call_llm",
            {
                "prompt": prompt,
                "system_prompt": SYSTEM_PROMPT,
                "n_predict": 700,
                "temperature": 0.4,
            },
        )
        return data.get("text", "")

    await respond(interaction, await run_queued(work))


@bot.tree.command(name="status", description="Show NativeLab runtime status.")
async def status(interaction: discord.Interaction):
    log(f"/status by {interaction.user}")
    require_capability("runtime")
    data = await get_json("/runtime")
    model = data.get("model_name") or "no model"
    state = data.get("status") or "unknown"
    await respond(interaction, f"NativeLab: {state}\nModel: {model}\nBackend: {data.get('backend', 'unknown')}")


@bot.tree.command(name="pipelines", description="List saved NativeLab pipelines.")
async def pipelines(interaction: discord.Interaction):
    log(f"/pipelines by {interaction.user}")
    require_capability("pipelines")
    data = await get_json("/pipelines")
    names = [p.get("name", "") for p in data.get("pipelines", []) if p.get("name")]
    await respond(interaction, "\n".join(names[:40]) if names else "No saved pipelines found.")


@bot.tree.command(name="pipeline", description="Show one saved pipeline summary.")
@app_commands.describe(name="Saved pipeline name")
async def pipeline(interaction: discord.Interaction, name: str):
    log(f"/pipeline by {interaction.user}: {name}")
    require_capability("pipelines")
    data = await get_json(f"/pipelines/{name}")
    if data.get("error"):
        await respond(interaction, data["error"])
        return
    definition = data.get("definition", {})
    blocks = len(definition.get("blocks", []))
    connections = len(definition.get("connections", []))
    await respond(interaction, f"{data.get('name', name)}: {blocks} block(s), {connections} connection(s)")


@bot.tree.command(name="labs", description="List NativeLab labs routes.")
async def labs(interaction: discord.Interaction):
    log(f"/labs by {interaction.user}")
    require_capability("labs")
    data = await get_json("/labs")
    rows = [f"{lab.get('route')} - {lab.get('name')}" for lab in data.get("labs", [])]
    await respond(interaction, "\n".join(rows) if rows else "No labs registered.")


@bot.tree.command(name="lab", description="Show one NativeLab lab route.")
@app_commands.describe(name="Lab slug, for example py_to_doc")
async def lab(interaction: discord.Interaction, name: str):
    log(f"/lab by {interaction.user}: {name}")
    require_capability("labs")
    data = await get_json(f"/labs/{name}")
    await respond(interaction, data.get("error") or f"{data.get('route')} - actions: {len(data.get('actions', []))}")


@bot.tree.command(name="models", description="List models visible to NativeLab integrations.")
async def models(interaction: discord.Interaction):
    log(f"/models by {interaction.user}")
    require_capability("models")
    local = await get_json("/models")
    api = await get_json("/api_models")
    local_names = [m.get("name") or m.get("path", "") for m in local.get("models", [])]
    api_names = [m.get("name") or m.get("model_id", "") for m in api.get("api_models", [])]
    lines = [f"Local models: {len(local_names)}", *local_names[:12], "", f"API models: {len(api_names)}", *api_names[:12]]
    await respond(interaction, "\n".join(lines))


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    log(f"Command error: {error}")
    try:
        await respond(interaction, f"NativeLab bot error: {error}")
    except Exception as send_error:
        log(f"Could not send error response: {send_error}")


def main():
    if not TOKEN:
        raise SystemExit("Set DISCORD_BOT_TOKEN or save a token in the selected NativeLab Discord profile.")
    log("Starting Discord client")
    bot.run(TOKEN)


if __name__ == "__main__":
    main()
