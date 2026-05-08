"""CLI entry point for `nativelab --cli`.

Subcommands
-----------
  (none)            run-or-onboard: setup wizard if no prefs, then chat
  setup             force the onboarding wizard
  chat              jump straight into the REPL using saved prefs
  lint <files>      run a linter on the given files
  status            print backend / model / ctx and exit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import chat as _chat
from . import lint as _lint
from . import onboarding
from . import ui


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nativelab --cli",
        description="NativeLab terminal client — local LLM chat, linting, "
                    "and inline file embedding.",
    )
    sub = p.add_subparsers(dest="cmd")

    p_setup = sub.add_parser("setup", help="(re-)run the onboarding wizard")
    p_setup.add_argument("--reset", action="store_true",
                         help="discard saved prefs before starting")

    p_chat = sub.add_parser("chat", help="open the chat REPL")
    p_chat.add_argument("--model",  default="", help="path to .gguf to load")
    p_chat.add_argument("--ctx",    type=int, default=0, help="context size")
    p_chat.add_argument("--system", default="", help="system prompt")

    p_lint = sub.add_parser("lint", help="lint Python files")
    p_lint.add_argument("paths", nargs="+")

    sub.add_parser("status", help="print current model / backend and exit")

    return p


def _cmd_setup(args) -> int:
    if args.reset:
        try:
            onboarding.CLI_PREFS_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    return 0 if onboarding.run_wizard() else 1


def _cmd_chat(args) -> int:
    prefs = onboarding.load_prefs()
    model = args.model or prefs.get("model_path", "")
    ctx   = args.ctx   or prefs.get("ctx", 0)

    if not model:
        ui.warn("No model configured. Running setup first.")
        prefs = onboarding.run_wizard() or {}
        model = prefs.get("model_path", "")
        ctx   = prefs.get("ctx", 0)

    return _chat.run(model_path=model, ctx=int(ctx or 0),
                     system_prompt=args.system)


def _cmd_default() -> int:
    """No subcommand: setup if needed, else chat."""
    prefs = onboarding.load_prefs()
    if not prefs.get("model_path"):
        prefs = onboarding.run_wizard()
        if not prefs:
            return 1
    return _chat.run(
        model_path    = prefs.get("model_path", ""),
        ctx           = int(prefs.get("ctx") or 0),
    )


def _cmd_status() -> int:
    prefs = onboarding.load_prefs()
    if not prefs:
        ui.warn("No CLI prefs saved yet. Run `nativelab --cli setup`.")
        return 1
    print(ui.bold("Saved CLI preferences"))
    print(f"  model_path : {prefs.get('model_path') or '(none)'}")
    print(f"  ctx        : {prefs.get('ctx', '(unset)')}")
    p = prefs.get("model_path") or ""
    if p and not Path(p).exists():
        ui.warn(f"Model file is missing: {p}")
    return 0


def run(argv: Optional[List[str]] = None) -> int:
    """Main entry point — accepts the post-`--cli` argv list."""
    argv = list(argv) if argv is not None else sys.argv[1:]
    # Strip `--cli` itself so argparse doesn't trip on it when callers
    # forward the unfiltered argv.
    argv = [a for a in argv if a != "--cli"]

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.cmd == "setup":
            return _cmd_setup(args)
        if args.cmd == "chat":
            return _cmd_chat(args)
        if args.cmd == "lint":
            return _lint.lint_paths(args.paths)
        if args.cmd == "status":
            return _cmd_status()
        return _cmd_default()
    except KeyboardInterrupt:
        print()
        return 130
