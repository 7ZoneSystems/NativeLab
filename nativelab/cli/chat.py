"""Interactive chat REPL for the CLI.

Reuses `LabEndpoints` so the same call paths the GUI / Labs use are exercised
here too — local llama-server, llama-cli, or remote API.

Built-ins
---------
  /help            list commands
  /status          show backend, model, ctx, server port
  /load <path>     load a different GGUF model
  /unload          unload the current local model
  /ctx <n>         change context size and reload
  /system <text>   set / replace the system prompt
  /reset           clear conversation history
  /lint <file>     run a linter on a file (pyflakes/flake8/pylint)
  /save <file>     save the conversation as JSON
  /quit            exit

Inline references
-----------------
  @path/to/file.py reads the file and embeds its contents in the prompt
  (similar to Claude Code's `@file` syntax).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from nativelab.core.engine_global import LlamaEngine, ApiEngine
from nativelab.labs import LabEndpoints

from . import lint as _lint
from . import ui


_FILE_REF_RE = re.compile(r"@([^\s'\"()]+)")


# ─────────────────────────────────────────────────────────────────────────────
#  Engine bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _build_endpoints(model_path: str, ctx: int) -> LabEndpoints:
    """Spin up a local LlamaEngine, optionally load a model, return endpoints."""
    eng = LlamaEngine()
    api: Optional[ApiEngine] = None

    if model_path and Path(model_path).exists():
        ui.info(f"Loading model: {Path(model_path).name}")
        ok = eng.load(model_path, ctx=ctx, log_cb=lambda m: ui.info(m))
        if not ok:
            ui.warn("Engine reported load failure — chat will run in degraded mode.")

    endpoints = LabEndpoints()
    endpoints.bind_engines(
        llama_provider=lambda: eng,
        api_provider  =lambda: api,
    )

    def on_context(new_ctx: int) -> bool:
        nonlocal eng
        try:
            eng.shutdown()
        except Exception:
            pass
        eng = LlamaEngine()
        endpoints.bind_engines(
            llama_provider=lambda: eng,
            api_provider  =lambda: api,
        )
        if not model_path or not Path(model_path).exists():
            return False
        ok = eng.load(model_path, ctx=int(new_ctx),
                      log_cb=lambda m: ui.info(m))
        endpoints.notify_engine_changed()
        return bool(ok)

    def on_model(new_path: str) -> bool:
        nonlocal eng, model_path
        if not new_path or not Path(new_path).exists():
            return False
        try:
            eng.shutdown()
        except Exception:
            pass
        eng = LlamaEngine()
        endpoints.bind_engines(
            llama_provider=lambda: eng,
            api_provider  =lambda: api,
        )
        ok = eng.load(new_path, ctx=ctx, log_cb=lambda m: ui.info(m))
        if ok:
            model_path = new_path
        endpoints.notify_engine_changed()
        return bool(ok)

    def on_unload() -> None:
        try:
            eng.shutdown()
        except Exception:
            pass
        endpoints.notify_engine_changed()

    endpoints.bind_reverse_routes(
        on_context=on_context,
        on_model  =on_model,
        on_unload =on_unload,
    )
    return endpoints


# ─────────────────────────────────────────────────────────────────────────────
#  Reference expansion
# ─────────────────────────────────────────────────────────────────────────────

def _expand_file_refs(text: str, cwd: Path) -> str:
    """Replace `@path/to/file` tokens with embedded file contents."""
    def repl(m: re.Match) -> str:
        token = m.group(1)
        p = Path(token)
        if not p.is_absolute():
            p = (cwd / p).resolve()
        if not p.exists() or not p.is_file():
            return m.group(0)
        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"[could not read {p}: {e}]"
        if len(body) > 60_000:
            body = body[:60_000] + "\n…[truncated]"
        return f"\n\n```{p.suffix.lstrip('.') or 'text'} title={p.name}\n{body}\n```\n"
    return _FILE_REF_RE.sub(repl, text)


# ─────────────────────────────────────────────────────────────────────────────
#  REPL
# ─────────────────────────────────────────────────────────────────────────────

class ChatREPL:
    def __init__(self, endpoints: LabEndpoints,
                 system_prompt: str = "",
                 cwd: Optional[Path] = None):
        self.endpoints = endpoints
        self.system    = system_prompt
        self.history: List[dict] = []
        self.cwd = cwd or Path.cwd()

    # ── input loop ───────────────────────────────────────────────────────────
    def run(self) -> int:
        ui.banner("NativeLab CLI · chat",
                  f"backend: {self.endpoints.status_text}")
        ui.info("Type /help for commands. /quit to exit.")
        while True:
            try:
                line = input(ui.bold(ui.cyan("you ▸ ")))
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                if not self._handle_slash(line):
                    return 0
                continue

            self._send(line)

    # ── slash commands ───────────────────────────────────────────────────────
    def _handle_slash(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        cmd   = parts[0].lower()
        rest  = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            return False

        if cmd in ("/help", "/?"):
            self._cmd_help()
        elif cmd == "/status":
            self._cmd_status()
        elif cmd == "/load":
            self._cmd_load(rest)
        elif cmd == "/unload":
            self.endpoints.request_unload()
            ui.ok("Engine unloaded.")
        elif cmd == "/ctx":
            self._cmd_ctx(rest)
        elif cmd == "/system":
            self.system = rest
            ui.ok(f"System prompt set ({len(rest)} chars).")
        elif cmd == "/reset":
            self.history.clear()
            ui.ok("Conversation reset.")
        elif cmd == "/lint":
            self._cmd_lint(rest)
        elif cmd == "/save":
            self._cmd_save(rest)
        else:
            ui.warn(f"Unknown command: {cmd}. Try /help.")
        return True

    def _cmd_help(self) -> None:
        print(ui.bold("Commands"))
        for k, v in [
            ("/status",        "current backend, model, ctx, server port"),
            ("/load <path>",   "load a GGUF model"),
            ("/unload",        "unload the local model"),
            ("/ctx <n>",       "change context size and reload"),
            ("/system <text>", "set the system prompt"),
            ("/reset",         "clear conversation history"),
            ("/lint <file>",   "run a linter on a file"),
            ("/save <file>",   "save conversation as JSON"),
            ("/quit",          "exit"),
        ]:
            print(f"  {ui.cyan(k):<28} {v}")
        print()
        print(ui.bold("Inline refs in messages"))
        print("  @path/to/file.py     embed file contents in the prompt")

    def _cmd_status(self) -> None:
        s = self.endpoints.snapshot()
        print(ui.bold("Status"))
        for k, v in s.items():
            print(f"  {k:<12} {v}")

    def _cmd_load(self, rest: str) -> None:
        if not rest:
            ui.warn("Usage: /load <path-to-gguf>")
            return
        ok = self.endpoints.request_load_model(rest)
        ui.ok(f"Loaded {rest}") if ok else ui.err("Load failed.")

    def _cmd_ctx(self, rest: str) -> None:
        try:
            n = int(rest)
        except ValueError:
            ui.warn("Usage: /ctx <n>")
            return
        ok = self.endpoints.request_context(n)
        ui.ok(f"Context now {n}") if ok else ui.err("Context change failed.")

    def _cmd_lint(self, rest: str) -> None:
        if not rest:
            ui.warn("Usage: /lint <file.py> [<file.py> ...]")
            return
        _lint.lint_paths(rest.split())

    def _cmd_save(self, rest: str) -> None:
        path = Path(rest or f"chat_{datetime.now():%Y%m%d_%H%M%S}.json")
        if not path.is_absolute():
            path = self.cwd / path
        path.write_text(json.dumps({
            "system":  self.system,
            "history": self.history,
            "saved":   datetime.now().isoformat(timespec="seconds"),
        }, indent=2, ensure_ascii=False))
        ui.ok(f"Saved → {path}")

    # ── send ─────────────────────────────────────────────────────────────────
    def _send(self, text: str) -> None:
        if not self.endpoints.is_loaded:
            ui.err("No engine loaded. Use /load <path> or run "
                   "`nativelab --cli setup` first.")
            return

        expanded = _expand_file_refs(text, self.cwd)
        self.history.append({"role": "user", "content": expanded})

        try:
            sys.stdout.write(ui.bold(ui.green("bot ▸ ")))
            sys.stdout.flush()
            reply = self.endpoints.call_llm(
                messages      = self.history,
                system_prompt = self.system or None,
            )
            print(reply)
        except KeyboardInterrupt:
            print()
            ui.warn("(interrupted)")
            return
        except Exception as e:
            print()
            ui.err(f"LLM error: {e}")
            self.history.pop()
            return

        self.history.append({"role": "assistant", "content": reply})


# ─────────────────────────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run(model_path: str = "", ctx: int = 0,
        system_prompt: str = "") -> int:
    from nativelab.GlobalConfig.config_global import DEFAULT_CTX
    if ctx <= 0:
        ctx = DEFAULT_CTX()

    endpoints = _build_endpoints(model_path, ctx)
    repl      = ChatREPL(endpoints, system_prompt=system_prompt)
    return repl.run()
