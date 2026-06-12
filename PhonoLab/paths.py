from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent


def mobile_target() -> str:
    if os.environ.get("ANDROID_ARGUMENT") or "ANDROID_PRIVATE" in os.environ:
        return "android"
    if sys.platform == "ios" or os.environ.get("KIVY_IOS"):
        return "ios"
    return "desktop"


def _android_app_root() -> Path | None:
    try:
        from android.storage import app_storage_path  # type: ignore

        return Path(app_storage_path())
    except Exception:
        return None


def default_home() -> Path:
    configured = os.environ.get("PHONOLAB_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    target = mobile_target()
    if target == "android":
        root = _android_app_root()
        if root is not None:
            return root / "PhonoLab"
    if target == "ios":
        return Path.home() / "Documents" / "PhonoLab"
    return PACKAGE_DIR / "data"


PHONOLAB_HOME = default_home()
CONFIG_DIR = PHONOLAB_HOME / "config"
MODELS_DIR = PHONOLAB_HOME / "models"
RUNTIME_DIR = PHONOLAB_HOME / "runtime"
DOWNLOADS_DIR = PHONOLAB_HOME / "downloads"
STATE_DIR = PHONOLAB_HOME / "state"
LOG_DIR = PHONOLAB_HOME / "logs"

CONFIG_FILE = CONFIG_DIR / "phonolab_config.json"
REGISTRY_FILE = CONFIG_DIR / "model_registry.json"
SETUP_STATE_FILE = STATE_DIR / "llama_cpp_setup.json"
CHAT_HISTORY_FILE = STATE_DIR / "chat_history.json"


def ensure_dirs() -> None:
    for path in (PHONOLAB_HOME, CONFIG_DIR, MODELS_DIR, RUNTIME_DIR, DOWNLOADS_DIR, STATE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def safe_child_path(root: Path, relative_name: str) -> Path:
    original = str(relative_name or "").replace("\\", "/")
    if original.startswith("/"):
        raise ValueError(f"Unsafe file name: {relative_name}")
    raw = original.strip("/")
    if not raw:
        raise ValueError("Empty file name")
    rel = Path(raw)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"Unsafe file name: {relative_name}")
    base = Path(root).resolve()
    dest = (base / rel).resolve()
    try:
        dest.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Path escapes PhonoLab storage: {relative_name}") from exc
    return dest


ensure_dirs()
