"""CLI feature commands shared by argparse, menus, and chat slash commands."""
from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from urllib.parse import quote

from nativelab.integrations import IntegrationEndpoints
from nativelab.integrations.discord_connector import (
    DEFAULT_DISCORD_BOT,
    command_catalog as discord_command_catalog,
    delete_discord_bot,
    invite_url,
    load_discord_bots,
    upsert_discord_bot,
)
from nativelab.integrations.http_endpoint import IntegrationHttpEndpoint
from nativelab.integrations.whatsapp_connector import (
    DEFAULT_WHATSAPP_BOT,
    command_catalog as whatsapp_command_catalog,
    delete_whatsapp_bot,
    load_whatsapp_bots,
    upsert_whatsapp_bot,
)
from nativelab.GlobalConfig.timeouts import LONG_TIMEOUT_MS
from nativelab.skill import (
    active_skills,
    delete_skill,
    ensure_builtin_edit_skill,
    load_skills,
    upsert_skill,
)

from . import onboarding, ui

if TYPE_CHECKING:
    from .runtime import CliRuntime


SECRET_KEYS = {"api_key", "token", "access_token"}
LOCAL_LLM_DIR = Path("./localllm")
CUSTOM_MODELS_FILE = LOCAL_LLM_DIR / "custom_models.json"
MODEL_CONFIGS_FILE = LOCAL_LLM_DIR / "model_configs.json"
API_MODELS_FILE = LOCAL_LLM_DIR / "api_models.json"
EDIT_RESPONSE_FILE = LOCAL_LLM_DIR / "temp_code_edit_response.json"
PIPELINES_DIRS = [Path.home() / ".native_lab" / "pipelines", LOCAL_LLM_DIR / "pipelines"]


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _redact(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            k: ("***" if k in SECRET_KEYS and v else _redact(v))
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_redact(v) for v in data]
    return data


def print_rows(rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        ui.warn("No rows.")
        return
    widths = {
        col: max(len(col), *(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    print("  ".join(ui.bold(col.ljust(widths[col])) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def _safe_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def _api_ref(name: str) -> str:
    return "@api/" + quote(name or "", safe="")


def _load_model_rows() -> list[dict[str, Any]]:
    configs = _safe_json(MODEL_CONFIGS_FILE, {})
    custom = _safe_json(CUSTOM_MODELS_FILE, [])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(LOCAL_LLM_DIR.glob("*.gguf")):
        cfg = configs.get(str(path), {}) if isinstance(configs, dict) else {}
        seen.add(str(path))
        rows.append({
            "path": str(path),
            "name": path.name,
            "size_mb": round(path.stat().st_size / 1e6, 1) if path.exists() else 0,
            "source": "auto",
            "role": cfg.get("role", "general"),
            "ctx": cfg.get("ctx", ""),
        })
    for item in custom if isinstance(custom, list) else []:
        path = Path(str(item))
        if str(path) in seen or not path.exists():
            continue
        cfg = configs.get(str(path), {}) if isinstance(configs, dict) else {}
        rows.append({
            "path": str(path),
            "name": path.name,
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "source": "custom",
            "role": cfg.get("role", "general"),
            "ctx": cfg.get("ctx", ""),
        })
    return rows


def _load_api_rows() -> list[dict[str, Any]]:
    rows = _safe_json(API_MODELS_FILE, [])
    if not isinstance(rows, list):
        return []
    clean = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row["api_key"] = "***" if row.get("api_key") else ""
        row["ref"] = _api_ref(str(row.get("name", "")))
        clean.append(row)
    return clean


def _load_pipeline_rows() -> list[dict[str, Any]]:
    out = []
    seen: set[str] = set()
    for root in PIPELINES_DIRS:
        for path in sorted(root.glob("*.json")) if root.exists() else []:
            if path.stem in seen:
                continue
            seen.add(path.stem)
            data = _safe_json(path, {})
            out.append({
                "name": path.stem,
                "route": f"/pipelines/{path.stem}",
                "path": str(path),
                "exists": path.exists(),
                "blocks": len(data.get("blocks", [])) if isinstance(data, dict) else 0,
                "connections": len(data.get("connections", [])) if isinstance(data, dict) else 0,
            })
    return out


def _load_pipeline_definition(name: str) -> dict[str, Any]:
    safe = Path(name).stem
    for root in PIPELINES_DIRS:
        path = root / f"{safe}.json"
        if path.exists():
            return {
                "name": safe,
                "route": f"/pipelines/{safe}",
                "path": str(path),
                "definition": _safe_json(path, {}),
            }
    return {"error": "pipeline not found", "name": name}


# ── Models / API models ─────────────────────────────────────────────────────


def list_models(*, as_json: bool = False) -> list[dict[str, Any]]:
    try:
        from nativelab.Model.model_global import get_model_registry
        rows = [dict(m) for m in get_model_registry().all_models()]
    except Exception:
        rows = _load_model_rows()
    if as_json:
        print_json({"models": rows})
    else:
        table = [
            {
                "#": i + 1,
                "name": Path(m.get("path", "")).name,
                "role": m.get("role", ""),
                "ctx": m.get("ctx", ""),
                "path": m.get("path", ""),
            }
            for i, m in enumerate(rows)
        ]
        print_rows(table, ["#", "name", "role", "ctx", "path"])
    return rows


def list_api_models(*, as_json: bool = False) -> list[dict[str, Any]]:
    try:
        from nativelab.Model.model_global import api_model_ref, getapi_registry
        rows = []
        for cfg in getapi_registry().all():
            item = cfg.to_dict()
            item["api_key"] = "***" if item.get("api_key") else ""
            item["ref"] = api_model_ref(cfg.name)
            rows.append(item)
    except Exception:
        rows = _load_api_rows()
    if as_json:
        print_json({"api_models": rows})
    else:
        print_rows(
            [{"name": r.get("name"), "model_id": r.get("model_id"), "format": r.get("api_format"), "ref": r.get("ref")} for r in rows],
            ["name", "model_id", "format", "ref"],
        )
    return rows


def choose_model_target() -> str:
    try:
        from nativelab.Model.model_global import api_model_ref, get_model_registry, getapi_registry
        local = list(get_model_registry().all_models())
        api = list(getapi_registry().all())
        api_target = lambda cfg: api_model_ref(cfg.name)
        api_label = lambda cfg: f"API: {cfg.name} ({cfg.model_id})"
    except Exception:
        local = _load_model_rows()
        api = _load_api_rows()
        api_target = lambda cfg: cfg.get("ref", _api_ref(cfg.get("name", "")))
        api_label = lambda cfg: f"API: {cfg.get('name', '')} ({cfg.get('model_id', '')})"
    labels = [f"Local: {Path(m.get('path', '')).name}" for m in local]
    labels += [api_label(cfg) for cfg in api]
    labels.append("Enter path/ref manually")
    idx = ui.choose("Select model/API profile", labels, default=0)
    if idx < len(local):
        return str(local[idx].get("path", ""))
    api_idx = idx - len(local)
    if 0 <= api_idx < len(api):
        return api_target(api[api_idx])
    return ui.ask("Model path or API ref")


def choose_api_target() -> str:
    try:
        from nativelab.Model.model_global import api_model_ref, getapi_registry
        api = list(getapi_registry().all())
        api_target = lambda cfg: api_model_ref(cfg.name)
        labels = [f"{cfg.name} ({cfg.model_id})" for cfg in api]
    except Exception:
        api = _load_api_rows()
        api_target = lambda cfg: cfg.get("ref", _api_ref(cfg.get("name", "")))
        labels = [f"{cfg.get('name', '')} ({cfg.get('model_id', '')})" for cfg in api]
    labels.append("Enter API ref/name manually")
    idx = ui.choose("Select API profile", labels, default=0)
    if idx < len(api):
        return api_target(api[idx])
    return ui.ask("API ref or profile name")


def set_default_model(target: str, ctx: Optional[int] = None) -> None:
    prefs = onboarding.load_prefs()
    prefs["model_path"] = target
    if ctx:
        prefs["ctx"] = int(ctx)
    onboarding.save_prefs(prefs)
    ui.ok("CLI default model updated.")


# ── Skills ──────────────────────────────────────────────────────────────────


def list_skills(*, as_json: bool = False) -> list[dict[str, Any]]:
    ensure_builtin_edit_skill()
    rows = load_skills()
    if as_json:
        print_json({"skills": rows, "active": active_skills()})
    else:
        print_rows(
            [{"name": s.get("name"), "enabled": s.get("enabled", True), "description": s.get("description", "")} for s in rows],
            ["name", "enabled", "description"],
        )
    return rows


def create_or_edit_skill(name: str = "", *, enabled: Optional[bool] = None) -> None:
    current = {s.get("name"): s for s in load_skills()}
    name = name or ui.ask("Skill name")
    existing = current.get(name, {})
    description = ui.ask("Description", existing.get("description", ""))
    print(ui.dim("Enter instructions. Finish with a single line containing only EOF."))
    lines = []
    default = existing.get("instructions", "")
    if default:
        print(ui.dim("Current instructions:"))
        print(default)
    while True:
        line = input()
        if line.strip() == "EOF":
            break
        lines.append(line)
    instructions = "\n".join(lines).strip() or default
    upsert_skill({
        "name": name,
        "description": description,
        "instructions": instructions,
        "enabled": existing.get("enabled", True) if enabled is None else bool(enabled),
    })
    ui.ok(f"Saved skill: {name}")


def set_skill_enabled(name: str, enabled: bool) -> None:
    rows = load_skills()
    found = False
    for skill in rows:
        if skill.get("name") == name:
            skill["enabled"] = bool(enabled)
            upsert_skill(skill)
            found = True
            break
    if found:
        ui.ok(f"{name}: {'enabled' if enabled else 'disabled'}")
    else:
        ui.err(f"Skill not found: {name}")


def set_chat_skills(runtime: Optional[CliRuntime], enabled: bool) -> None:
    prefs = onboarding.load_prefs()
    prefs["skills_enabled"] = bool(enabled)
    onboarding.save_prefs(prefs)
    if runtime:
        runtime.set_skills_enabled(enabled, save=False)
    ui.ok(f"Chat skill injection {'enabled' if enabled else 'disabled'}.")


# ── Integration endpoint browser / server ───────────────────────────────────


def endpoint_show(path: str = "/snapshot", *, runtime: Optional[CliRuntime] = None, as_json: bool = False) -> dict:
    endpoints = runtime.integrations if runtime else IntegrationEndpoints()
    clean = "/" + (path or "").strip("/")
    data = endpoints.handle(path)
    if runtime is None:
        if clean == "/models":
            data = {"models": _load_model_rows()}
        elif clean == "/api_models":
            data = {"api_models": _load_api_rows()}
        elif clean == "/pipelines":
            data = {"pipelines": _load_pipeline_rows()}
        elif clean.startswith("/pipelines/"):
            data = _load_pipeline_definition(clean.split("/", 2)[2])
        elif clean in {"/", "/snapshot"}:
            data["models"] = data.get("models") or _load_model_rows()
            data["api_models"] = data.get("api_models") or _load_api_rows()
            data["pipelines"] = data.get("pipelines") or _load_pipeline_rows()
    if as_json:
        print_json(data)
    elif isinstance(data, dict):
        print_json(_redact(data))
    else:
        print(data)
    return data


def serve_endpoint(runtime: CliRuntime, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = IntegrationHttpEndpoint(runtime.integrations, host=host, port=port)
    url = server.start()
    ui.ok(f"NativeLab integration endpoint running at {url}")
    ui.info("Press Ctrl+C to stop.")
    try:
        import time
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print()
    finally:
        server.stop()
        ui.ok("Endpoint stopped.")


# ── Pipelines ────────────────────────────────────────────────────────────────


def pipeline_list(*, as_json: bool = False) -> list[dict[str, Any]]:
    rows = IntegrationEndpoints().pipelines()
    if not rows:
        rows = _load_pipeline_rows()
    if as_json:
        print_json({"pipelines": rows})
    else:
        print_rows(rows, ["name", "blocks", "connections", "path"])
    return rows


def pipeline_show(name: str, *, as_json: bool = False) -> dict:
    data = IntegrationEndpoints().pipeline_definition(name)
    if data.get("error"):
        fallback = _load_pipeline_definition(name)
        if not fallback.get("error"):
            data = fallback
    if as_json:
        print_json(data)
    else:
        if data.get("error"):
            ui.err(data["error"])
        else:
            d = data.get("definition", {})
            ui.info(f"Pipeline: {data.get('name')}  blocks={len(d.get('blocks', []))}  connections={len(d.get('connections', []))}")
            for block in d.get("blocks", []):
                print(f"  {block.get('bid')}: {block.get('btype')}  {block.get('label', '')}")
    return data


def pipeline_run(runtime: CliRuntime, name: str, text: str) -> int:
    from PyQt6.QtCore import QCoreApplication, QEventLoop
    from nativelab.pipelinebuilder.executionWorker import PipelineExecutionWorker
    from nativelab.pipelinebuilder.pipefunctions import load_pipeline

    if not runtime.endpoints.is_loaded:
        ui.err("No engine loaded for pipeline execution.")
        return 1
    app = QCoreApplication.instance() or QCoreApplication([])
    loop = QEventLoop()
    blocks, conns = load_pipeline(name)
    worker = PipelineExecutionWorker(blocks, conns, text, runtime.endpoints.active_engine())
    result = {"code": 0}

    worker.log_msg.connect(lambda msg: print(ui.dim(f"[pipeline] {msg}")))
    worker.step_started.connect(lambda bid, label: ui.info(f"step {bid}: {label}"))
    worker.step_token.connect(lambda bid, tok: (sys.stdout.write(tok), sys.stdout.flush()))
    worker.err.connect(lambda msg: (ui.err(msg), result.update(code=1), loop.quit()))

    def done(payload: str):
        try:
            data = json.loads(payload)
            final = data.get("text", payload)
        except Exception:
            final = payload
        print()
        print(ui.bold("Pipeline output"))
        print(final)
        loop.quit()

    worker.pipeline_done.connect(done)
    worker.start()
    loop.exec()
    worker.wait(LONG_TIMEOUT_MS)
    _ = app
    return int(result["code"])


# ── Labs: Code Edit and py-to-doc ───────────────────────────────────────────


def code_edit(
    runtime: CliRuntime,
    *,
    file_path: str = "",
    prompt: str = "",
    save: bool = False,
    save_as: str = "",
    edit_response: bool = False,
    diff: bool = True,
) -> int:
    from nativelab.labs.codeedit import (
        EDIT_SYSTEM_PROMPT,
        EDIT_TEMP_CODE_FILE,
        EDIT_TEMP_FILE,
        apply_operations,
        extract_json_object,
        render_diff_html,
        structure_for_code,
    )

    if not runtime.endpoints.is_loaded:
        ui.err("Load a model/API profile before running code-edit.")
        return 1
    source = Path(file_path).expanduser() if file_path else None
    code = ""
    if source and source.exists():
        code = source.read_text(encoding="utf-8", errors="replace")
    elif EDIT_TEMP_CODE_FILE.exists():
        code = EDIT_TEMP_CODE_FILE.read_text(encoding="utf-8", errors="replace")
    prompt = prompt or ui.ask("Structured edit request")
    structure = structure_for_code(code, str(source or ""))
    payload = {
        "path": str(source or ""),
        "language": structure.get("language", "code"),
        "structure": structure,
        "current_code": code,
        "request": prompt,
        "mode": "generate" if not code.strip() else "edit",
    }
    raw = runtime.endpoints.call_llm(
        system_prompt=EDIT_SYSTEM_PROMPT,
        prompt=json.dumps(payload, indent=2, ensure_ascii=False),
        n_predict=1800,
        temperature=0.15,
    )
    if edit_response:
        EDIT_RESPONSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        EDIT_RESPONSE_FILE.write_text(raw, encoding="utf-8")
        editor = os.environ.get("EDITOR")
        if editor:
            subprocess.call([editor, str(EDIT_RESPONSE_FILE)])
            raw = EDIT_RESPONSE_FILE.read_text(encoding="utf-8", errors="replace")
        else:
            ui.warn(f"Set EDITOR to edit the model JSON response. Response saved -> {EDIT_RESPONSE_FILE}")
    data = extract_json_object(raw)
    operations = data.get("operations", [])
    if not isinstance(operations, list) or not operations:
        ui.err("Model returned no edit operations.")
        return 1
    if not code.strip() and operations[0].get("op") != "full_replace":
        operations = [{"op": "full_replace", "code": data.get("code", "") or operations[0].get("code", "")}]
    updated = apply_operations(code, operations, structure)
    _, added, deleted = render_diff_html(code, updated)
    ui.ok(data.get("summary", "Structured edit applied."))
    print(f"Diff: {ui.green('+' + str(added))} / {ui.red('-' + str(deleted))}")
    if diff:
        for line in difflib.unified_diff(code.splitlines(), updated.splitlines(), fromfile="before", tofile="after", lineterm=""):
            if line.startswith("+") and not line.startswith("+++"):
                print(ui.green(line))
            elif line.startswith("-") and not line.startswith("---"):
                print(ui.red(line))
            else:
                print(line)
    EDIT_TEMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if EDIT_TEMP_FILE.exists():
        try:
            history = json.loads(EDIT_TEMP_FILE.read_text(encoding="utf-8")).get("history", [])
        except Exception:
            history = []
    history.append({
        "prompt": prompt,
        "summary": data.get("summary", ""),
        "raw_response": raw,
        "added": added,
        "deleted": deleted,
        "code_before": code,
        "code_after": updated,
    })
    EDIT_TEMP_CODE_FILE.write_text(updated, encoding="utf-8")
    EDIT_TEMP_FILE.write_text(json.dumps({
        "source_path": str(source or ""),
        "last_prompt": prompt,
        "code": updated,
        "temp_code_file": str(EDIT_TEMP_CODE_FILE),
        "history": history,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    target = Path(save_as).expanduser() if save_as else None
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(updated, encoding="utf-8")
        ui.ok(f"Saved -> {target}")
    elif save and source:
        source.write_text(updated, encoding="utf-8")
        ui.ok(f"Saved -> {source}")
    else:
        ui.info(f"Temp saved -> {EDIT_TEMP_CODE_FILE}")
    return 0


def py_to_doc(
    runtime: CliRuntime,
    *,
    mode: str,
    files: list[str],
    project: str,
    out_dir: str,
    out_name: str,
    resume: bool = False,
    context_policy: str = "none",
    reset_per_function: bool = False,
    reset_per_class: bool = False,
    context_budget: int = 4096,
) -> int:
    from PyQt6.QtCore import QCoreApplication, QEventLoop
    from nativelab.labs.pytodoc import (
        DEFAULT_CLASS_PROMPT,
        DEFAULT_FUNC_PROMPT,
        DEFAULT_OVERVIEW_PROMPT,
        PyToDocWorker,
        AUTO_CONTEXT_DEFAULT,
        AUTO_CONTEXT_MIN,
        CONTEXT_POLICY_AUTO,
        CONTEXT_POLICY_FIXED,
        CONTEXT_POLICY_NONE,
        discover_project_python_files,
        mirror_project_directories,
    )

    if not runtime.endpoints.is_loaded:
        ui.err("Load a model/API profile before running py-to-doc.")
        return 1
    if mode == "project":
        project_root = str(Path(project).expanduser())
        file_list = discover_project_python_files(project_root)
    else:
        file_list = [str(Path(f).expanduser()) for f in files]
        project_root = None
    if not file_list:
        ui.err("No Python files selected.")
        return 1
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if context_policy not in {CONTEXT_POLICY_NONE, CONTEXT_POLICY_FIXED, CONTEXT_POLICY_AUTO}:
        context_policy = CONTEXT_POLICY_FIXED
    context_budget = max(AUTO_CONTEXT_MIN, int(context_budget or AUTO_CONTEXT_DEFAULT))
    ui.info(f"Context policy: {context_policy}" + (f" (~{context_budget} tokens)" if context_policy == CONTEXT_POLICY_AUTO else ""))
    if (
        context_policy == CONTEXT_POLICY_AUTO
        and runtime.llama
        and runtime.llama.is_loaded
        and not (runtime.api and runtime.api.is_loaded)
        and int(getattr(runtime.llama, "ctx_value", 0) or 0) != context_budget
    ):
        ui.info(f"Reloading local model/server for py-to-doc context: {context_budget:,}")
        if not runtime.endpoints.request_context(context_budget):
            ui.err(f"Could not reload local model/server with context {context_budget:,}.")
            return 1
    if mode == "project":
        mirrored = mirror_project_directories(project_root or "", out_dir)
        ui.info(f"Project files selected: {len(file_list)}")
        ui.info(f"Output directory structure prepared: {mirrored} directories")
    app = QCoreApplication.instance() or QCoreApplication([])
    loop = QEventLoop()
    result = {"code": 0}
    worker = PyToDocWorker(
        file_path=file_list[0],
        out_path=out_dir,
        out_name=out_name,
        include_globals=True,
        context_policy=context_policy,
        fixed_reset_per_function=bool(reset_per_function),
        fixed_reset_per_class=bool(reset_per_class),
        auto_context_tokens=context_budget,
        prompt_overview=DEFAULT_OVERVIEW_PROMPT,
        prompt_class=DEFAULT_CLASS_PROMPT,
        prompt_function=DEFAULT_FUNC_PROMPT,
        endpoints=runtime.endpoints,
        file_list=file_list if mode in {"queue", "project"} else None,
        project_root=project_root,
        resume_required=bool(resume),
    )
    worker.log_msg.connect(lambda msg: print(ui.dim(msg)))
    worker.chunk.connect(lambda text: (sys.stdout.write(text), sys.stdout.flush()))
    worker.error.connect(lambda msg: (ui.err(msg), result.update(code=1), loop.quit()))
    worker.paused.connect(lambda path: (ui.warn(f"Paused: {path}"), loop.quit()))
    worker.done.connect(lambda: (ui.ok("py-to-doc complete."), loop.quit()))
    worker.start()
    loop.exec()
    worker.wait(LONG_TIMEOUT_MS)
    _ = app
    return int(result["code"])


# ── Bot profiles and foreground runners ─────────────────────────────────────


def bot_profiles(kind: str, *, as_json: bool = False) -> list[dict[str, Any]]:
    rows = load_discord_bots() if kind == "discord" else load_whatsapp_bots()
    if as_json:
        print_json({f"{kind}_bots": _redact(rows)})
    else:
        print_rows(
            [{"name": b.get("name"), "enabled": b.get("enabled", True), "endpoint_url": b.get("endpoint_url", "")} for b in rows],
            ["name", "enabled", "endpoint_url"],
        )
    return rows


def bot_show(kind: str, name: str, *, as_json: bool = False) -> Optional[dict]:
    rows = load_discord_bots() if kind == "discord" else load_whatsapp_bots()
    bot = next((b for b in rows if b.get("name") == name), None)
    if not bot:
        ui.err(f"{kind} profile not found: {name}")
        return None
    data = dict(bot)
    data["commands"] = discord_command_catalog(bot) if kind == "discord" else whatsapp_command_catalog(bot)
    print_json(_redact(data) if as_json else _redact(data))
    return bot


def bot_create(kind: str, name: str = "") -> None:
    existing_rows = load_discord_bots() if kind == "discord" else load_whatsapp_bots()
    profile_name = name or ui.ask("Profile name")
    existing = next((b for b in existing_rows if b.get("name") == profile_name), {})
    cfg = json.loads(json.dumps(DEFAULT_DISCORD_BOT if kind == "discord" else DEFAULT_WHATSAPP_BOT))
    for key, value in existing.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    cfg["name"] = profile_name
    cfg["endpoint_url"] = ui.ask("NativeLab endpoint", cfg.get("endpoint_url", "http://127.0.0.1:8765"))
    cfg["system_prompt"] = ui.ask("System prompt", cfg.get("system_prompt", ""))
    if kind == "discord":
        cfg["token"] = ui.ask("Discord bot token", cfg.get("token", ""))
        cfg["application_id"] = ui.ask("Application ID", cfg.get("application_id", ""))
        cfg["guild_id"] = ui.ask("Guild ID (optional)", cfg.get("guild_id", ""))
        cfg["reply"]["direct_mentions"] = ui.ask_yesno("Reply to @mentions without slash commands?", bool(cfg.get("reply", {}).get("direct_mentions", False)))
        cfg["reply"]["ephemeral"] = ui.ask_yesno("Use ephemeral slash replies?", bool(cfg.get("reply", {}).get("ephemeral", False)))
        cfg["queue"]["enabled"] = ui.ask_yesno("Enable request queueing?", bool(cfg.get("queue", {}).get("enabled", True)))
        cfg["queue"]["max_queued"] = ui.ask_int("Max queued requests", int(cfg.get("queue", {}).get("max_queued", 12)), lo=1)
        upsert_discord_bot(cfg)
        url = invite_url(cfg)
        if url:
            ui.info(f"Invite URL: {url}")
    else:
        cfg["access_token"] = ui.ask("WhatsApp access token", cfg.get("access_token", ""))
        cfg["phone_number_id"] = ui.ask("Phone number ID", cfg.get("phone_number_id", ""))
        cfg["business_account_id"] = ui.ask("Business account ID", cfg.get("business_account_id", ""))
        cfg["verify_token"] = ui.ask("Webhook verify token", cfg.get("verify_token", "nativelab-whatsapp"))
        cfg["webhook_host"] = ui.ask("Webhook host", cfg.get("webhook_host", "127.0.0.1"))
        cfg["webhook_port"] = ui.ask_int("Webhook port", int(cfg.get("webhook_port", 8770)), lo=1)
        cfg["reply"]["direct_messages"] = ui.ask_yesno("Reply to normal messages?", bool(cfg.get("reply", {}).get("direct_messages", True)))
        cfg["queue"]["enabled"] = ui.ask_yesno("Enable request queueing?", bool(cfg.get("queue", {}).get("enabled", True)))
        cfg["queue"]["max_queued"] = ui.ask_int("Max queued requests", int(cfg.get("queue", {}).get("max_queued", 12)), lo=1)
        upsert_whatsapp_bot(cfg)
    ui.ok(f"Saved {kind} profile: {cfg['name']}")


def bot_delete(kind: str, name: str) -> None:
    if kind == "discord":
        delete_discord_bot(name)
    else:
        delete_whatsapp_bot(name)
    ui.ok(f"Deleted {kind} profile: {name}")


def bot_run(kind: str, name: str, endpoint_url: str = "") -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if kind == "discord":
        env["DISCORD_BOT_PROFILE"] = name
        module = "nativelab.integrations.examples.discord_bot"
    else:
        env["WHATSAPP_BOT_PROFILE"] = name
        module = "nativelab.integrations.examples.whatsapp_bot"
    if endpoint_url:
        env["NATIVELAB_INTEGRATION_URL"] = endpoint_url
    cmd = [sys.executable, "-u", "-m", module]
    ui.info(f"Starting {kind} bot profile '{name}'. Press Ctrl+C to stop.")
    try:
        return subprocess.call(cmd, env=env, cwd=str(Path.cwd()))
    except KeyboardInterrupt:
        print()
        return 130
