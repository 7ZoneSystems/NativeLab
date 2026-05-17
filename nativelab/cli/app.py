"""CLI entry point for `nativelab --cli`.

Subcommands
-----------
  (none)            run-or-onboard: setup wizard if no prefs, then menu
  setup             force the onboarding wizard
  chat              jump straight into the REPL using saved prefs
  lint <files>      run a linter on the given files
  status            print backend / model / ctx and exit
  models            list/load local GGUF models
  api-models        list/load saved API profiles
  skills            manage shared model skills
  labs              run CLI Labs workflows
  pipeline          list/show/run saved visual pipelines
  integrations      inspect/manage integration profiles
  endpoint          inspect integration endpoint routes
  serve             run the integration HTTP endpoint
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import lint as _lint
from . import onboarding
from . import ui


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nativelab --cli",
        description="NativeLab terminal client - local LLM chat, linting, "
                    "inline file embedding, Labs, skills, pipelines, and integrations.",
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

    p_models = sub.add_parser("models", help="list/load local models")
    p_models.add_argument("action", nargs="?", default="list", choices=["list", "load", "default"])
    p_models.add_argument("target", nargs="?", default="")
    p_models.add_argument("--ctx", type=int, default=0)
    p_models.add_argument("--json", action="store_true")

    p_api = sub.add_parser("api-models", help="list/load saved API model profiles")
    p_api.add_argument("action", nargs="?", default="list", choices=["list", "load", "default"])
    p_api.add_argument("target", nargs="?", default="")
    p_api.add_argument("--json", action="store_true")

    p_skills = sub.add_parser("skills", help="manage shared model skills")
    p_skills.add_argument("action", nargs="?", default="list",
                          choices=["list", "create", "edit", "delete", "enable", "disable", "chat-on", "chat-off"])
    p_skills.add_argument("name", nargs="?", default="")
    p_skills.add_argument("--json", action="store_true")

    p_labs = sub.add_parser("labs", help="list/run Labs workflows")
    labs_sub = p_labs.add_subparsers(dest="lab_cmd")
    labs_sub.add_parser("list", help="list available labs").add_argument("--json", action="store_true")
    p_code = labs_sub.add_parser("code-edit", help="run structured code edit")
    p_code.add_argument("--file", default="")
    p_code.add_argument("--prompt", default="")
    p_code.add_argument("--save", action="store_true")
    p_code.add_argument("--save-as", default="")
    p_code.add_argument("--edit-response", action="store_true")
    p_code.add_argument("--no-diff", action="store_true")
    p_doc = labs_sub.add_parser("py-to-doc", help="generate Python docs")
    p_doc.add_argument("--mode", choices=["single", "queue", "project"], default="single")
    p_doc.add_argument("--file", action="append", default=[])
    p_doc.add_argument("--project", default="")
    p_doc.add_argument("--out-dir", default="./docs/generated")
    p_doc.add_argument("--out-name", default="README.md")
    p_doc.add_argument("--resume", action="store_true")
    p_doc.add_argument("--context-policy", choices=["none", "fixed", "auto"], default="none")
    p_doc.add_argument("--reset-per-function", action="store_true")
    p_doc.add_argument("--no-reset-per-function", action="store_false", dest="reset_per_function")
    p_doc.add_argument("--reset-per-class", action="store_true")
    p_doc.add_argument("--context-budget", type=int, default=4096)

    p_pipe = sub.add_parser("pipeline", help="list/show/run saved pipelines")
    p_pipe.add_argument("action", nargs="?", default="list", choices=["list", "show", "run"])
    p_pipe.add_argument("name", nargs="?", default="")
    p_pipe.add_argument("--text", default="")
    p_pipe.add_argument("--file", default="")
    p_pipe.add_argument("--json", action="store_true")

    p_int = sub.add_parser("integrations", help="inspect/manage integrations")
    p_int.add_argument("kind", nargs="?", default="list", choices=["list", "discord", "whatsapp"])
    p_int.add_argument("action", nargs="?", default="list", choices=["list", "show", "create", "edit", "delete", "run"])
    p_int.add_argument("name", nargs="?", default="")
    p_int.add_argument("--endpoint", default="")
    p_int.add_argument("--json", action="store_true")

    p_endpoint = sub.add_parser("endpoint", help="inspect an integration endpoint route")
    p_endpoint.add_argument("route", nargs="?", default="/snapshot")
    p_endpoint.add_argument("--json", action="store_true")

    p_serve = sub.add_parser("serve", help="run the local integration HTTP endpoint")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--model", default="")
    p_serve.add_argument("--ctx", type=int, default=0)

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
    from . import chat as _chat

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
    """No subcommand: setup if needed, else the terminal control center."""
    prefs = onboarding.load_prefs()
    if not prefs.get("model_path"):
        prefs = onboarding.run_wizard()
        if not prefs:
            return 1
    return _interactive_menu(prefs)


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
    print(f"  skills     : {'on' if prefs.get('skills_enabled') else 'off'}")
    return 0


def _runtime(model: str = "", ctx: int = 0, *, autoload: bool = True) -> CliRuntime:
    from .runtime import runtime_from_prefs

    return runtime_from_prefs(model, ctx, autoload=autoload)


def _cmd_models(args) -> int:
    from . import features

    if args.action == "list":
        features.list_models(as_json=args.json)
        return 0
    target = args.target or features.choose_api_target()
    if args.action == "default":
        features.set_default_model(target, args.ctx or None)
        return 0
    rt = _runtime("", args.ctx, autoload=False)
    return 0 if rt.load_model(target) else 1


def _cmd_api_models(args) -> int:
    from . import features

    if args.action == "list":
        features.list_api_models(as_json=args.json)
        return 0
    target = args.target or features.choose_model_target()
    if args.action == "default":
        features.set_default_model(target)
        return 0
    rt = _runtime("", 0, autoload=False)
    return 0 if rt.load_model(target) else 1


def _cmd_skills(args) -> int:
    from . import features

    if args.action == "list":
        features.list_skills(as_json=args.json)
    elif args.action in {"create", "edit"}:
        features.create_or_edit_skill(args.name)
    elif args.action == "delete":
        if not args.name:
            ui.warn("Usage: skills delete <name>"); return 2
        from nativelab.skill import delete_skill
        delete_skill(args.name); ui.ok(f"Deleted skill: {args.name}")
    elif args.action == "enable":
        if not args.name:
            ui.warn("Usage: skills enable <name>"); return 2
        features.set_skill_enabled(args.name, True)
    elif args.action == "disable":
        if not args.name:
            ui.warn("Usage: skills disable <name>"); return 2
        features.set_skill_enabled(args.name, False)
    elif args.action == "chat-on":
        features.set_chat_skills(None, True)
    elif args.action == "chat-off":
        features.set_chat_skills(None, False)
    return 0


def _cmd_labs(args) -> int:
    from . import features

    if args.lab_cmd in (None, "list"):
        features.endpoint_show("/labs", as_json=getattr(args, "json", False))
        return 0
    rt = _runtime()
    if args.lab_cmd == "code-edit":
        return features.code_edit(
            rt,
            file_path=args.file,
            prompt=args.prompt,
            save=args.save,
            save_as=args.save_as,
            edit_response=args.edit_response,
            diff=not args.no_diff,
        )
    if args.lab_cmd == "py-to-doc":
        return features.py_to_doc(
            rt,
            mode=args.mode,
            files=args.file,
            project=args.project,
            out_dir=args.out_dir,
            out_name=args.out_name,
            resume=args.resume,
            context_policy=args.context_policy,
            reset_per_function=args.reset_per_function,
            reset_per_class=args.reset_per_class,
            context_budget=args.context_budget,
        )
    return 2


def _cmd_pipeline(args) -> int:
    from . import features

    if args.action == "list":
        features.pipeline_list(as_json=args.json); return 0
    if not args.name:
        ui.warn("Usage: pipeline show|run <name>"); return 2
    if args.action == "show":
        features.pipeline_show(args.name, as_json=args.json); return 0
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    if not text:
        text = sys.stdin.read() if not sys.stdin.isatty() else ui.ask("Pipeline input")
    return features.pipeline_run(_runtime(), args.name, text)


def _cmd_integrations(args) -> int:
    from . import features

    if args.kind == "list":
        features.endpoint_show("/snapshot", as_json=args.json); return 0
    kind = args.kind
    if args.action == "list":
        features.bot_profiles(kind, as_json=args.json)
    elif args.action in {"show", "edit"}:
        if not args.name:
            ui.warn(f"Usage: integrations {kind} show <name>"); return 2
        if args.action == "show":
            features.bot_show(kind, args.name, as_json=args.json)
        else:
            features.bot_create(kind, args.name)
    elif args.action == "create":
        features.bot_create(kind, args.name)
    elif args.action == "delete":
        if not args.name:
            ui.warn(f"Usage: integrations {kind} delete <name>"); return 2
        features.bot_delete(kind, args.name)
    elif args.action == "run":
        if not args.name:
            ui.warn(f"Usage: integrations {kind} run <name>"); return 2
        return features.bot_run(kind, args.name, endpoint_url=args.endpoint)
    return 0


def _cmd_endpoint(args) -> int:
    from . import features

    features.endpoint_show(args.route, runtime=None, as_json=args.json)
    return 0


def _cmd_serve(args) -> int:
    from . import features

    rt = _runtime(args.model, args.ctx)
    features.serve_endpoint(rt, host=args.host, port=args.port)
    return 0


def _interactive_menu(prefs: dict) -> int:
    from . import chat as _chat
    from . import features

    runtime: CliRuntime | None = None

    def rt() -> CliRuntime:
        nonlocal runtime
        if runtime is None:
            runtime = _runtime()
        return runtime

    while True:
        ui.banner("NativeLab CLI", "terminal control center")
        choice = ui.choose("Choose workspace", [
            "Chat",
            "Models",
            "API Models",
            "Skills",
            "Labs",
            "Pipelines",
            "Integrations",
            "Status",
            "Setup",
            "Quit",
        ], default=0)
        if choice == 0:
            _chat.run_with_runtime(rt())
            continue
        if choice == 1:
            features.list_models()
            if ui.ask_yesno("Load or set default model now?", default=False):
                target = features.choose_api_target()
                if ui.ask_yesno("Load immediately?", default=True):
                    rt().load_model(target)
                else:
                    features.set_default_model(target)
        elif choice == 2:
            features.list_api_models()
            if ui.ask_yesno("Load or set default API model now?", default=False):
                target = features.choose_model_target()
                if ui.ask_yesno("Load immediately?", default=True):
                    rt().load_model(target)
                else:
                    features.set_default_model(target)
        elif choice == 3:
            _skills_menu(rt() if runtime else None)
        elif choice == 4:
            _labs_menu(rt)
        elif choice == 5:
            _pipeline_menu(rt)
        elif choice == 6:
            _integrations_menu(rt)
        elif choice == 7:
            _cmd_status()
            input(ui.dim("Press Enter..."))
        elif choice == 8:
            onboarding.run_wizard()
        else:
            return 0


def _skills_menu(runtime: CliRuntime | None) -> None:
    from . import features

    features.list_skills()
    choice = ui.choose("Skills action", ["Toggle chat skills", "Create/edit skill", "Enable skill", "Disable skill", "Back"], default=0)
    if choice == 0:
        prefs = onboarding.load_prefs()
        features.set_chat_skills(runtime, not bool(prefs.get("skills_enabled", False)))
    elif choice == 1:
        features.create_or_edit_skill()
    elif choice == 2:
        features.set_skill_enabled(ui.ask("Skill name"), True)
    elif choice == 3:
        features.set_skill_enabled(ui.ask("Skill name"), False)


def _labs_menu(rt_factory) -> None:
    from . import features

    features.endpoint_show("/labs")
    choice = ui.choose("Labs action", ["Code Edit", "py-to-doc", "Back"], default=0)
    if choice == 0:
        features.code_edit(rt_factory(), file_path=ui.ask("File path (blank for temp)"), prompt=ui.ask("Edit request"), save=ui.ask_yesno("Save to source?", False))
    elif choice == 1:
        mode = ["single", "queue", "project"][ui.choose("Mode", ["single", "queue", "project"], default=0)]
        policy = ["fixed", "auto", "none"][ui.choose("Context policy", ["fixed reset", "auto budget", "no reset"], default=0)]
        reset_fn = True
        reset_cls = False
        budget = 4096
        if policy == "fixed":
            reset_fn = ui.ask_yesno("Reset after each function?", True)
            reset_cls = ui.ask_yesno("Reset after each class?", False)
        elif policy == "auto":
            try:
                budget = int(ui.ask("Context budget tokens", "4096"))
            except Exception:
                budget = 4096
        if mode == "project":
            features.py_to_doc(
                rt_factory(), mode=mode, files=[], project=ui.ask("Project root"),
                out_dir=ui.ask("Output dir", "./docs/generated"),
                out_name=ui.ask("Output name", "README.md"),
                context_policy=policy,
                reset_per_function=reset_fn,
                reset_per_class=reset_cls,
                context_budget=budget,
            )
        else:
            files = ui.ask("Python file(s), comma-separated").split(",")
            features.py_to_doc(
                rt_factory(), mode=mode, files=[f.strip() for f in files if f.strip()],
                project="", out_dir=ui.ask("Output dir", "./docs/generated"),
                out_name=ui.ask("Output name", "README.md"),
                context_policy=policy,
                reset_per_function=reset_fn,
                reset_per_class=reset_cls,
                context_budget=budget,
            )


def _pipeline_menu(rt_factory) -> None:
    from . import features

    features.pipeline_list()
    choice = ui.choose("Pipeline action", ["Show", "Run", "Back"], default=0)
    if choice == 0:
        features.pipeline_show(ui.ask("Pipeline name"))
    elif choice == 1:
        features.pipeline_run(rt_factory(), ui.ask("Pipeline name"), ui.ask("Input text"))


def _integrations_menu(rt_factory) -> None:
    from . import features

    choice = ui.choose("Integrations action", ["Browse endpoint", "Discord profiles", "WhatsApp profiles", "Serve endpoint", "Back"], default=0)
    if choice == 0:
        features.endpoint_show(ui.ask("Route", "/snapshot"), runtime=rt_factory())
    elif choice == 1:
        features.bot_profiles("discord")
    elif choice == 2:
        features.bot_profiles("whatsapp")
    elif choice == 3:
        features.serve_endpoint(rt_factory())


def run(argv: Optional[List[str]] = None) -> int:
    """Main entry point - accepts the post-`--cli` argv list."""
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
        if args.cmd == "models":
            return _cmd_models(args)
        if args.cmd == "api-models":
            return _cmd_api_models(args)
        if args.cmd == "skills":
            return _cmd_skills(args)
        if args.cmd == "labs":
            return _cmd_labs(args)
        if args.cmd == "pipeline":
            return _cmd_pipeline(args)
        if args.cmd == "integrations":
            return _cmd_integrations(args)
        if args.cmd == "endpoint":
            return _cmd_endpoint(args)
        if args.cmd == "serve":
            return _cmd_serve(args)
        if args.cmd == "status":
            return _cmd_status()
        return _cmd_default()
    except KeyboardInterrupt:
        print()
        return 130
