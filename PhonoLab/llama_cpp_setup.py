from __future__ import annotations

import json
import shutil
import stat
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import MobileConfig, load_config, save_config
from .downloads import ProgressCallback, download_to_cache
from .paths import RUNTIME_DIR, SETUP_STATE_FILE, atomic_write_text, mobile_target, safe_child_path
from .safety import SafetyError


LLAMA_CPP_SOURCE_URL = "https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.zip"


@dataclass(frozen=True)
class LlamaCppPlan:
    target: str
    source_dir: str
    cli_path: str
    server_path: str
    ready: bool
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def runtime_bin_dir() -> Path:
    return RUNTIME_DIR / "bin"


def source_dir() -> Path:
    return RUNTIME_DIR / "src" / "llama.cpp"


def _candidate_bins(config: MobileConfig) -> list[Path]:
    ext = ".exe" if mobile_target() == "windows" else ""
    names = ["llama-cli", "main"]
    candidates: list[Path] = []
    if config.llama_cli_path:
        candidates.append(Path(config.llama_cli_path).expanduser())
    for name in names:
        candidates.append(runtime_bin_dir() / f"{name}{ext}")
    found = shutil.which("llama-cli")
    if found:
        candidates.append(Path(found))
    return candidates


def find_llama_cli(config: MobileConfig | None = None) -> Path | None:
    cfg = config or load_config()
    for path in _candidate_bins(cfg):
        try:
            if path.exists() and path.is_file():
                return path
        except Exception:
            continue
    return None


def current_plan(config: MobileConfig | None = None) -> LlamaCppPlan:
    cfg = config or load_config()
    target = mobile_target()
    cli = find_llama_cli(cfg)
    src = source_dir()
    if cli:
        return LlamaCppPlan(
            target=target,
            source_dir=str(src),
            cli_path=str(cli),
            server_path=str(cfg.llama_server_path or ""),
            ready=True,
            message="llama.cpp command runtime is available.",
        )
    if src.exists():
        return LlamaCppPlan(
            target=target,
            source_dir=str(src),
            cli_path="",
            server_path=str(cfg.llama_server_path or ""),
            ready=False,
            message=(
                "llama.cpp source is pulled. Bundle a platform-built llama-cli into "
                "PhonoLab/runtime/bin for mobile execution."
            ),
        )
    return LlamaCppPlan(
        target=target,
        source_dir=str(src),
        cli_path="",
        server_path=str(cfg.llama_server_path or ""),
        ready=False,
        message="llama.cpp has not been pulled yet.",
    )


def _write_state(status: str, **data) -> None:
    payload = {"version": 1, "status": status, **data}
    atomic_write_text(SETUP_STATE_FILE, json.dumps(payload, indent=2, sort_keys=True))


def _safe_extract_zip(archive: Path, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            for item in zf.infolist():
                if item.is_dir():
                    continue
                name = item.filename
                parts = Path(name).parts
                if not parts:
                    continue
                stripped = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
                if not str(stripped):
                    continue
                out = safe_child_path(tmp, str(stripped))
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(item) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        if dest.exists():
            shutil.rmtree(dest)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def pull_llama_cpp(progress: ProgressCallback | None = None) -> LlamaCppPlan:
    target = mobile_target()
    _write_state("running", stage="download_source", target=target)
    archive = download_to_cache(LLAMA_CPP_SOURCE_URL, "llama.cpp-master.zip", progress=progress)
    _write_state("running", stage="extract_source", target=target, archive=str(archive))
    src = source_dir()
    _safe_extract_zip(archive, src)
    _write_state("complete", stage="source_ready", target=target, source_dir=str(src))
    return current_plan()


def register_llama_cli(path: str | Path) -> LlamaCppPlan:
    candidate = Path(path).expanduser()
    if not candidate.exists() or not candidate.is_file():
        raise SafetyError(f"llama-cli binary does not exist: {candidate}")
    try:
        mode = candidate.stat().st_mode
        candidate.chmod(mode | stat.S_IXUSR)
    except Exception:
        pass
    cfg = load_config()
    cfg.llama_cli_path = str(candidate)
    save_config(cfg)
    _write_state("complete", stage="runtime_ready", target=mobile_target(), cli_path=str(candidate))
    return current_plan(cfg)


def setup_summary() -> dict:
    return current_plan().to_dict()
