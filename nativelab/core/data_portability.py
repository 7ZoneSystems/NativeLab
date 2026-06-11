from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from nativelab.GlobalConfig.config_global import APP_CONFIG_FILE, PAUSED_JOBS_DIR, PIPELINES_DIR


FORMAT_NAME = "nativelab-portable-data"
FORMAT_VERSION = 1
ROOT = Path(".")
LOCAL_LLM_DIR = Path("./localllm")
SESSIONS_DIR = Path("./sessions")
CHAT_REFS_DIR = Path("./chat_refs")
REF_CACHE_DIR = Path("./ref_cache")
REF_INDEX_DIR = Path("./ref_index")
ALLOWED_RELATIVE_ROOTS = {
    "app_config.json",
    "chat_refs",
    "localllm",
    "paused_jobs",
    "ref_cache",
    "ref_index",
    "sessions",
}


@dataclass(frozen=True)
class ExportCategory:
    id: str
    label: str
    description: str
    paths: tuple[Path, ...] = ()
    globs: tuple[tuple[Path, str], ...] = ()


CATEGORIES: tuple[ExportCategory, ...] = (
    ExportCategory(
        "app_settings",
        "App settings",
        "Main app_config.json thresholds, defaults, HF/Ollama settings, and developer mode.",
        paths=(APP_CONFIG_FILE,),
    ),
    ExportCategory(
        "chat_history",
        "Chat history",
        "All saved chat sessions.",
        globs=((SESSIONS_DIR, "*.json"),),
    ),
    ExportCategory(
        "local_model_profiles",
        "Local model profiles",
        "Registered GGUF/Ollama/HF refs and their per-model parameters.",
        paths=(LOCAL_LLM_DIR / "custom_models.json", LOCAL_LLM_DIR / "model_configs.json"),
    ),
    ExportCategory(
        "api_models",
        "API model profiles",
        "Saved API model provider profiles.",
        paths=(LOCAL_LLM_DIR / "api_models.json",),
    ),
    ExportCategory(
        "pipelines",
        "Visual pipelines",
        "All saved Pipeline Builder workflows.",
        globs=((PIPELINES_DIR, "*.json"),),
    ),
    ExportCategory(
        "discord_bots",
        "Discord bot profiles",
        "Saved Discord connector profiles.",
        paths=(LOCAL_LLM_DIR / "integrations" / "discord_bots.json",),
    ),
    ExportCategory(
        "whatsapp_bots",
        "WhatsApp bot profiles",
        "Saved WhatsApp connector profiles.",
        paths=(LOCAL_LLM_DIR / "integrations" / "whatsapp_bots.json",),
    ),
    ExportCategory(
        "pytodoc",
        "Py-to-Doc data",
        "Py-to-Doc job checkpoints, pause/recovery files, and temp state.",
        paths=(LOCAL_LLM_DIR / "temp",),
        globs=((LOCAL_LLM_DIR / "pytodoc_jobs", "*.json"),),
    ),
    ExportCategory(
        "paused_jobs",
        "Paused summary jobs",
        "Paused PDF and multi-PDF jobs.",
        globs=((PAUSED_JOBS_DIR, "*.json"),),
    ),
    ExportCategory(
        "references",
        "Reference data",
        "Reference files, raw text cache, and reference indexes.",
        globs=(
            (CHAT_REFS_DIR, "*"),
            (REF_CACHE_DIR, "*"),
            (REF_INDEX_DIR, "*.json"),
        ),
    ),
    ExportCategory(
        "skills",
        "Skill library",
        "Saved NativeLab skills.",
        paths=(LOCAL_LLM_DIR / "skill" / "skills.json",),
    ),
    ExportCategory(
        "runtime_profiles",
        "Runtime profiles",
        "Server, parallel loading, CLI, MCP, and HF account credentials saved locally.",
        paths=(
            LOCAL_LLM_DIR / "server_config.json",
            LOCAL_LLM_DIR / "api_server_config.json",
            LOCAL_LLM_DIR / "parallel_prefs.json",
            LOCAL_LLM_DIR / "cli_prefs.json",
            LOCAL_LLM_DIR / "mcp_config.json",
            LOCAL_LLM_DIR / "cred" / "huggingface.json",
        ),
    ),
    ExportCategory(
        "structured_edit",
        "Structured edit temp data",
        "Code Edit temp workspace and saved model response.",
        paths=(
            LOCAL_LLM_DIR / "temp_code_edit.json",
            LOCAL_LLM_DIR / "temp_code_edit_file",
            LOCAL_LLM_DIR / "temp_code_edit_response.json",
        ),
    ),
)


def category_map() -> Dict[str, ExportCategory]:
    return {cat.id: cat for cat in CATEGORIES}


def category_options() -> list[dict[str, Any]]:
    return [
        {
            "id": cat.id,
            "label": cat.label,
            "description": cat.description,
            "count": len(_category_files(cat)),
        }
        for cat in CATEGORIES
    ]


def export_bundle(path: str | Path, category_ids: Optional[Iterable[str]] = None) -> dict[str, Any]:
    selected = {cat.id for cat in CATEGORIES} if category_ids is None else set(category_ids)
    bundle = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "categories": {},
    }
    for cat in CATEGORIES:
        if cat.id not in selected:
            continue
        files = [_read_entry(p) for p in _category_files(cat)]
        files = [entry for entry in files if entry is not None]
        bundle["categories"][cat.id] = {
            "label": cat.label,
            "description": cat.description,
            "files": files,
            "items": _category_items(cat.id),
        }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return summarize_bundle(bundle)


def load_bundle(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != FORMAT_NAME:
        raise ValueError("This is not a NativeLab data export file.")
    if int(data.get("version", 0) or 0) > FORMAT_VERSION:
        raise ValueError("This export was created by a newer NativeLab version.")
    return data


def summarize_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    categories = bundle.get("categories") or {}
    return {
        "format": bundle.get("format", ""),
        "version": bundle.get("version", 0),
        "exported_at": bundle.get("exported_at", ""),
        "categories": [
            {
                "id": cat_id,
                "label": data.get("label", cat_id),
                "description": data.get("description", ""),
                "file_count": len(data.get("files") or []),
                "items": data.get("items") or {},
            }
            for cat_id, data in categories.items()
        ],
    }


def import_bundle(
    bundle: dict[str, Any],
    category_ids: Optional[Iterable[str]] = None,
    *,
    selected_api_models: Optional[Iterable[str]] = None,
    selected_model_profiles: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    selected_categories = set((bundle.get("categories") or {}).keys()) if category_ids is None else set(category_ids)
    api_names = None if selected_api_models is None else set(selected_api_models)
    model_paths = None if selected_model_profiles is None else set(selected_model_profiles)
    imported: dict[str, int] = {}

    for cat_id, data in (bundle.get("categories") or {}).items():
        if cat_id not in selected_categories:
            continue
        if cat_id == "api_models":
            imported[cat_id] = _import_api_models(data, api_names)
            continue
        if cat_id == "local_model_profiles":
            imported[cat_id] = _import_model_profiles(data, model_paths)
            continue
        count = 0
        for entry in data.get("files") or []:
            if _write_entry(entry):
                count += 1
        imported[cat_id] = count

    return {"imported": imported}


def _category_files(cat: ExportCategory) -> list[Path]:
    files: list[Path] = []
    for p in cat.paths:
        if p.exists() and p.is_file():
            files.append(p)
    for folder, pattern in cat.globs:
        if not folder.exists():
            continue
        files.extend(p for p in sorted(folder.glob(pattern)) if p.is_file())
    seen = set()
    unique = []
    for p in files:
        key = _portable_path(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return f"~/{path.resolve().relative_to(Path.home()).as_posix()}" if _is_under(path, Path.home()) else str(path)


def _target_path(portable_path: str) -> Path:
    if portable_path.startswith("~/"):
        rel = Path(portable_path[2:])
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or len(rel.parts) != 3
            or rel.parts[0] != ".native_lab"
            or rel.parts[1] != "pipelines"
            or rel.suffix.lower() != ".json"
        ):
            raise ValueError(f"Unsupported export path: {portable_path}")
        return PIPELINES_DIR / rel.name
    rel = Path(portable_path)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] not in ALLOWED_RELATIVE_ROOTS:
        raise ValueError(f"Unsupported export path: {portable_path}")
    return ROOT / rel


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _read_entry(path: Path) -> Optional[dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    entry = {
        "path": _portable_path(path),
        "name": path.name,
        "encoding": "json" if path.suffix.lower() == ".json" else "text",
    }
    if entry["encoding"] == "json":
        try:
            entry["data"] = json.loads(raw.decode("utf-8"))
            return entry
        except Exception:
            entry["encoding"] = "base64"
    if entry["encoding"] == "text":
        try:
            entry["data"] = raw.decode("utf-8")
            return entry
        except UnicodeDecodeError:
            entry["encoding"] = "base64"
    entry["data"] = base64.b64encode(raw).decode("ascii")
    return entry


def _write_entry(entry: dict[str, Any]) -> bool:
    try:
        target = _target_path(str(entry.get("path", "")))
        target.parent.mkdir(parents=True, exist_ok=True)
        encoding = entry.get("encoding", "json")
        if encoding == "json":
            target.write_text(json.dumps(entry.get("data"), indent=2, ensure_ascii=False), encoding="utf-8")
        elif encoding == "base64":
            target.write_bytes(base64.b64decode(str(entry.get("data", ""))))
        else:
            target.write_text(str(entry.get("data", "")), encoding="utf-8")
        return True
    except Exception:
        return False


def _category_items(cat_id: str) -> dict[str, Any]:
    if cat_id == "api_models":
        rows = _load_json_file(LOCAL_LLM_DIR / "api_models.json", [])
        return {"api_models": [_api_model_name(row) for row in rows if isinstance(row, dict)]}
    if cat_id == "local_model_profiles":
        custom = _load_json_file(LOCAL_LLM_DIR / "custom_models.json", [])
        configs = _load_json_file(LOCAL_LLM_DIR / "model_configs.json", {})
        if not isinstance(custom, list):
            custom = []
        if not isinstance(configs, dict):
            configs = {}
        names = sorted({str(p) for p in list(custom or []) + list((configs or {}).keys()) if str(p)})
        return {"model_profiles": names}
    return {}


def _load_json_file(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _entry_data(data: dict[str, Any], filename: str, fallback):
    for entry in data.get("files") or []:
        if Path(str(entry.get("path", ""))).name == filename:
            return entry.get("data", fallback)
    return fallback


def _api_model_name(row: dict[str, Any]) -> str:
    return str(row.get("name") or row.get("model_id") or "").strip()


def _import_api_models(data: dict[str, Any], selected_names: Optional[set[str]]) -> int:
    imported_rows = _entry_data(data, "api_models.json", [])
    if not isinstance(imported_rows, list):
        return 0
    if selected_names is not None:
        imported_rows = [row for row in imported_rows if isinstance(row, dict) and _api_model_name(row) in selected_names]
    existing_path = LOCAL_LLM_DIR / "api_models.json"
    existing = _load_json_file(existing_path, [])
    by_name = {_api_model_name(row): row for row in existing if isinstance(row, dict) and _api_model_name(row)}
    for row in imported_rows:
        if isinstance(row, dict) and _api_model_name(row):
            by_name[_api_model_name(row)] = row
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_text(json.dumps(list(by_name.values()), indent=2, ensure_ascii=False), encoding="utf-8")
    return len(imported_rows)


def _import_model_profiles(data: dict[str, Any], selected_paths: Optional[set[str]]) -> int:
    custom = _entry_data(data, "custom_models.json", [])
    configs = _entry_data(data, "model_configs.json", {})
    if not isinstance(custom, list):
        custom = []
    if not isinstance(configs, dict):
        configs = {}
    all_paths = sorted({str(p) for p in list(custom) + list(configs.keys()) if str(p)})
    if selected_paths is not None:
        all_paths = [p for p in all_paths if p in selected_paths]

    custom_path = LOCAL_LLM_DIR / "custom_models.json"
    configs_path = LOCAL_LLM_DIR / "model_configs.json"
    existing_custom = _load_json_file(custom_path, [])
    existing_configs = _load_json_file(configs_path, {})
    if not isinstance(existing_custom, list):
        existing_custom = []
    if not isinstance(existing_configs, dict):
        existing_configs = {}

    for p in all_paths:
        if p in custom and p not in existing_custom:
            existing_custom.append(p)
        if p in configs:
            existing_configs[p] = configs[p]

    custom_path.parent.mkdir(parents=True, exist_ok=True)
    custom_path.write_text(json.dumps(existing_custom, indent=2, ensure_ascii=False), encoding="utf-8")
    configs_path.write_text(json.dumps(existing_configs, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(all_paths)
