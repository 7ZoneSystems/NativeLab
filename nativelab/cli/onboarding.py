"""First-run onboarding wizard for `nativelab --cli`.

Guides the user through:
  1. Verifying llama.cpp binaries (server + cli) are available.
  2. Picking or downloading a GGUF model.
  3. Choosing a context size.
  4. Saving the resulting selection so subsequent runs skip straight to chat.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import ui
from . import hf_download


# Persisted CLI preferences (separate from the GUI's APP_CONFIG).
CLI_PREFS_FILE = Path("./localllm/cli_prefs.json")


# ─────────────────────────────────────────────────────────────────────────────
#  Prefs (last-used model + ctx)
# ─────────────────────────────────────────────────────────────────────────────

def load_prefs() -> dict:
    if not CLI_PREFS_FILE.exists():
        return {}
    try:
        return json.loads(CLI_PREFS_FILE.read_text())
    except Exception:
        return {}


def save_prefs(d: dict) -> None:
    CLI_PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLI_PREFS_FILE.write_text(json.dumps(d, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
#  Binary check
# ─────────────────────────────────────────────────────────────────────────────

def check_binaries() -> bool:
    """Verify llama-server / llama-cli paths from SERVER_CONFIG, warn if missing.

    Returns True when at least one of (server, cli) is usable. Onboarding can
    proceed with API-only mode otherwise.
    """
    import nativelab.GlobalConfig.binaryResolve as _binres
    from nativelab.Server.server_global import SERVER_CONFIG

    server = SERVER_CONFIG.server_path or _binres.LLAMA_SERVER
    cli    = SERVER_CONFIG.cli_path    or _binres.LLAMA_CLI

    s_ok = bool(server) and Path(server).exists()
    c_ok = bool(cli)    and Path(cli).exists()

    print()
    print(ui.bold("Binary check"))
    print(f"  llama-server : "
          f"{ui.green(server) if s_ok else ui.red(server or '(not set)')}")
    print(f"  llama-cli    : "
          f"{ui.green(cli)    if c_ok else ui.red(cli    or '(not set)')}")

    if s_ok or c_ok:
        return True

    ui.warn("No llama.cpp binary found.")
    ui.info("Download a release zip into ./llama/bin/ from "
            "https://github.com/ggml-org/llama.cpp/releases, or open the GUI "
            "(`nativelab`) and use the Download tab - it has a one-click installer.")
    return ui.ask_yesno("Continue without local binaries (API-only mode)?", default=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Model picker
# ─────────────────────────────────────────────────────────────────────────────

def pick_or_download_model() -> Optional[str]:
    """Walk the user through choosing a model. Returns its absolute path or None."""
    from nativelab.Model.model_global import get_model_registry
    from nativelab.GlobalConfig.config_global import MODELS_DIR

    registry = get_model_registry()
    models   = registry.all_models()

    if models:
        print()
        print(ui.bold("Available models"))
        labels = [
            f"{m['name']}  ({m['size_mb']:>7.1f} MB)  [{m['family']} · {m['quant']}]"
            for m in models
        ]
        labels.append("Download a new model from HuggingFace")
        labels.append("Pick a GGUF file from disk")
        idx = ui.choose("Select a model", labels, default=0)
        if idx < len(models):
            return models[idx]["path"]
        if idx == len(models):
            return _download_flow(MODELS_DIR, registry)
        return _local_pick(registry)

    ui.info("No GGUF models registered yet.")
    if ui.ask_yesno("Download one from HuggingFace now?", default=True):
        return _download_flow(MODELS_DIR, registry)
    if ui.ask_yesno("Or point to an existing GGUF on disk?", default=True):
        return _local_pick(registry)
    return None


def _local_pick(registry) -> Optional[str]:
    while True:
        path = ui.ask("Path to .gguf file (blank to skip)")
        if not path:
            return None
        p = Path(path).expanduser().resolve()
        if not p.exists() or p.suffix.lower() != ".gguf":
            ui.warn(f"Not a .gguf file: {p}")
            continue
        registry.add(str(p))
        ui.ok(f"Registered: {p}")
        return str(p)


def _download_flow(models_dir: Path, registry) -> Optional[str]:
    print()
    print(ui.bold("HuggingFace download"))
    picks = hf_download.quick_picks()
    labels = [p["label"] for p in picks]
    labels.append("Enter a custom HuggingFace repo ID")
    labels.append("Cancel")

    idx = ui.choose("Pick a model to download", labels, default=0)
    if idx == len(labels) - 1:
        return None

    if idx < len(picks):
        repo_id, filename = picks[idx]["repo"], picks[idx]["file"]
    else:
        repo_id  = ui.ask("HuggingFace repo (e.g. TheBloke/Mistral-7B-Instruct-v0.2-GGUF)")
        if not repo_id:
            return None
        try:
            siblings = hf_download.list_repo_ggufs(repo_id)
        except Exception as e:
            ui.err(f"Couldn't list GGUFs for {repo_id}: {e}")
            return None
        if not siblings:
            ui.err(f"No GGUF files in {repo_id}")
            return None
        sib_labels = [
            f"{s['name']}  ({s['size'] / 1e6:.0f} MB)" if s["size"]
            else s["name"]
            for s in siblings
        ]
        sidx = ui.choose("Pick a quantization", sib_labels, default=0)
        filename = siblings[sidx]["name"]

    try:
        dest = hf_download.download_gguf(
            repo_id, filename, models_dir,
            expected_size=0,
        )
    except Exception as e:
        ui.err(f"Download failed: {e}")
        return None

    registry.add(str(dest))
    return str(dest)


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level wizard
# ─────────────────────────────────────────────────────────────────────────────

def run_wizard() -> Optional[dict]:
    """Run the full first-time setup. Returns a dict suitable for `save_prefs`."""
    ui.banner("NativeLab CLI", "Local-LLM chat in your terminal")

    if not check_binaries():
        ui.err("Aborted: cannot proceed without binaries or API mode.")
        return None

    model_path = pick_or_download_model()
    if not model_path:
        ui.warn("Skipping model selection - chat will require API setup.")
        ctx = 4096
    else:
        from nativelab.GlobalConfig.config_global import DEFAULT_CTX, MAX_CONTEXT_TOKENS
        ctx = ui.ask_int(
            "Context size (tokens)",
            default=DEFAULT_CTX(),
            lo=512, hi=MAX_CONTEXT_TOKENS,
        )

    prefs = {"model_path": model_path or "", "ctx": int(ctx)}
    save_prefs(prefs)
    ui.ok("Setup saved. You can re-run the wizard with `nativelab --cli setup`.")
    return prefs
