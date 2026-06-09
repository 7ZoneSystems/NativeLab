from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from nativelab.imports.import_global import HAS_PSUTIL, QThread, pyqtSignal
from nativelab.GlobalConfig.config_global import (
    APP_CONFIG,
    MAX_CONTEXT_TOKENS,
    MODELS_DIR,
    refresh_binary_paths,
)
from nativelab.Model.model_global import (
    ModelConfig,
    detect_model_family,
    get_model_registry,
    is_model_ref_valid,
    make_hf_model_ref,
    model_ref_display_name,
)
from nativelab.Server.hfauth import get_hf_access_token, normalize_hf_exception
from nativelab.Server.hfdwld import (
    HfDownloadWorker,
    HfSnapshotDownloadWorker,
    LlamaCppDownloadWorker,
    fetch_hf_gguf_files,
    fetch_hf_snapshot_files,
    fetch_llama_cpp_releases,
)
from nativelab.Server.server_global import SERVER_CONFIG, detect_gpus


AUTO_SETUP_STATE_FILE = MODELS_DIR / "auto_setup_state.json"
AUTO_SETUP_STATE_VERSION = 2
BACKEND_LLAMA_CPP = "llama_cpp"
BACKEND_HF_TRANSFORMERS = "hf_transformers"
_ACTIVE_STATUSES = {"running", "paused", "error"}
_TERMINAL_STATUSES = {"complete", "declined"}


def normalize_setup_backend(value: str = "") -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"hf", "huggingface", "hugging_face", "transformers", BACKEND_HF_TRANSFORMERS}:
        return BACKEND_HF_TRANSFORMERS
    return BACKEND_LLAMA_CPP


def setup_backend_label(value: str = "") -> str:
    backend = normalize_setup_backend(value)
    if backend == BACKEND_HF_TRANSFORMERS:
        return "Hugging Face Transformers (hard)"
    return "llama.cpp GGUF (easy)"


def _now() -> float:
    return time.time()


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_path(text: str = "") -> Path:
    return Path(str(text or "")).expanduser()


@dataclass
class HardwareProfile:
    os_name: str
    arch: str
    cpu_threads: int
    ram_total_mb: int
    ram_available_mb: int
    ram_used_mb: int
    gpus: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accelerator(self) -> str:
        return str(self.gpus[0].get("type", "cpu")).strip().lower() if self.gpus else "cpu"

    @property
    def primary_vram_mb(self) -> int:
        if not self.gpus:
            return 0
        return int(self.gpus[0].get("vram_mb") or 0)

    @property
    def primary_vram_free_mb(self) -> int:
        if not self.gpus:
            return 0
        free = int(self.gpus[0].get("vram_free_mb") or 0)
        return free or self.primary_vram_mb


@dataclass
class AutoModelChoice:
    key: str
    label: str
    repo: str
    quant_preferences: list[str]
    min_ram_mb: int
    min_vram_mb: int = 0
    ctx: int = 4096
    n_predict: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    tier: str = "balanced"
    backend: str = BACKEND_LLAMA_CPP


AUTO_MODEL_CATALOG: list[AutoModelChoice] = [
    AutoModelChoice(
        key="tinyllama-1b",
        label="TinyLlama 1.1B Chat",
        repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        quant_preferences=["Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"],
        min_ram_mb=4096,
        ctx=2048,
        n_predict=384,
        tier="minimal",
    ),
    AutoModelChoice(
        key="llama32-3b",
        label="Llama 3.2 3B Instruct",
        repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        quant_preferences=["Q4_K_M", "Q5_K_M", "Q4_0", "Q3_K_M"],
        min_ram_mb=7168,
        min_vram_mb=3072,
        ctx=4096,
        tier="low",
    ),
    AutoModelChoice(
        key="qwen3-4b",
        label="Qwen3 4B Instruct",
        repo="unsloth/Qwen3-4B-Instruct-2507-GGUF",
        quant_preferences=["Q4_K_M", "Q5_K_M", "Q4_0", "Q6_K"],
        min_ram_mb=10240,
        min_vram_mb=4096,
        ctx=4096,
        tier="balanced",
    ),
    AutoModelChoice(
        key="llama31-8b",
        label="Llama 3.1 8B Instruct",
        repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        quant_preferences=["Q4_K_M", "Q5_K_M", "Q6_K", "Q4_0"],
        min_ram_mb=16384,
        min_vram_mb=8192,
        ctx=8192,
        n_predict=768,
        tier="high",
    ),
]


AUTO_HF_MODEL_CATALOG: list[AutoModelChoice] = [
    AutoModelChoice(
        key="qwen25-0_5b-hf",
        label="Qwen2.5 0.5B Instruct",
        repo="Qwen/Qwen2.5-0.5B-Instruct",
        quant_preferences=[],
        min_ram_mb=4096,
        ctx=2048,
        n_predict=384,
        tier="minimal",
        backend=BACKEND_HF_TRANSFORMERS,
    ),
    AutoModelChoice(
        key="qwen25-1_5b-hf",
        label="Qwen2.5 1.5B Instruct",
        repo="Qwen/Qwen2.5-1.5B-Instruct",
        quant_preferences=[],
        min_ram_mb=8192,
        ctx=4096,
        tier="low",
        backend=BACKEND_HF_TRANSFORMERS,
    ),
    AutoModelChoice(
        key="qwen3-4b-hf",
        label="Qwen3 4B Instruct",
        repo="Qwen/Qwen3-4B-Instruct-2507",
        quant_preferences=[],
        min_ram_mb=16384,
        min_vram_mb=6144,
        ctx=4096,
        tier="balanced",
        backend=BACKEND_HF_TRANSFORMERS,
    ),
    AutoModelChoice(
        key="qwen25-7b-hf",
        label="Qwen2.5 7B Instruct",
        repo="Qwen/Qwen2.5-7B-Instruct",
        quant_preferences=[],
        min_ram_mb=24576,
        min_vram_mb=10240,
        ctx=8192,
        n_predict=768,
        tier="high",
        backend=BACKEND_HF_TRANSFORMERS,
    ),
]


def read_auto_setup_state() -> dict[str, Any]:
    if not AUTO_SETUP_STATE_FILE.exists():
        return {}
    try:
        data = json.loads(AUTO_SETUP_STATE_FILE.read_text(encoding="utf-8"))
        return normalize_auto_setup_state(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_auto_setup_state(data: dict[str, Any]) -> None:
    AUTO_SETUP_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = normalize_auto_setup_state(data)
    tmp = AUTO_SETUP_STATE_FILE.with_name(AUTO_SETUP_STATE_FILE.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(AUTO_SETUP_STATE_FILE)


def normalize_auto_setup_state(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data or {})
    out["version"] = AUTO_SETUP_STATE_VERSION
    if out.get("backend"):
        out["backend"] = normalize_setup_backend(str(out.get("backend")))
    status = str(out.get("status", "") or "").strip().lower()
    if status not in _ACTIVE_STATUSES and status not in _TERMINAL_STATUSES:
        if out.get("stage"):
            status = "running"
        else:
            status = ""
    out["status"] = status
    out["stage"] = str(out.get("stage", "") or "").strip().lower()
    for key in ("model_path", "registered_ref", "runtime_path"):
        if key in out and out.get(key) is None:
            out[key] = ""
    return out


def clear_auto_setup_state() -> None:
    try:
        AUTO_SETUP_STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def auto_setup_resumable(state: Optional[dict[str, Any]] = None) -> bool:
    data = read_auto_setup_state() if state is None else state
    status = str(data.get("status", "")).lower()
    stage = str(data.get("stage", "")).lower()
    return bool(stage and status in _ACTIVE_STATUSES and stage not in {"complete", "declined"})


def auto_setup_declined(state: Optional[dict[str, Any]] = None) -> bool:
    data = read_auto_setup_state() if state is None else state
    return str(data.get("status", "")).lower() == "declined"


def decline_auto_setup() -> None:
    write_auto_setup_state({
        "status": "declined",
        "stage": "declined",
        "updated_at": _now(),
    })


def installed_model_count() -> int:
    try:
        return len(get_model_registry().all_models())
    except Exception:
        return 0


def api_model_count() -> int:
    try:
        from nativelab.Model.model_global import getapi_registry

        return len(getapi_registry().all())
    except Exception:
        return 0


def user_needs_auto_setup() -> bool:
    state = read_auto_setup_state()
    if auto_setup_resumable(state):
        return True
    if auto_setup_declined(state):
        return False
    if recover_completed_auto_setup(state):
        return False
    return installed_model_count() == 0 and api_model_count() == 0


def recover_completed_auto_setup(state: Optional[dict[str, Any]] = None) -> bool:
    data = read_auto_setup_state() if state is None else normalize_auto_setup_state(state)
    if str(data.get("status", "")).lower() != "complete":
        return False
    ref = str(data.get("registered_ref") or "")
    model_path = str(data.get("model_path") or "")
    backend = normalize_setup_backend(data.get("backend") or "")
    if model_path and not _as_path(model_path).exists():
        return False
    if not ref and model_path:
        ref = make_hf_model_ref(model_path) if backend == BACKEND_HF_TRANSFORMERS else model_path
    if not ref:
        return False
    if backend == BACKEND_HF_TRANSFORMERS and ref.startswith("hf:"):
        payload = ref[3:].strip()
        if payload and (_as_path(payload).is_absolute() or "/" in payload or "\\" in payload):
            if not _as_path(payload).exists():
                return False
    if is_model_ref_valid(ref):
        registry = get_model_registry()
        registry.add(ref)
        return True
    return False


def profile_hardware() -> HardwareProfile:
    total = 0
    available = 0
    if HAS_PSUTIL:
        try:
            import psutil

            vm = psutil.virtual_memory()
            total = int(vm.total / (1024 * 1024))
            available = int(vm.available / (1024 * 1024))
        except Exception:
            total = 0
            available = 0
    if not total:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total = int((pages * page_size) / (1024 * 1024))
        except Exception:
            total = 8192
    if not available:
        available = max(1024, int(total * 0.55))
    available = max(0, min(int(available), int(total)))
    cpus = os.cpu_count() or 4
    gpus = detect_gpus()
    return HardwareProfile(
        os_name=platform.system() or "Unknown",
        arch=platform.machine() or "",
        cpu_threads=max(1, int(cpus)),
        ram_total_mb=max(0, int(total)),
        ram_available_mb=max(0, int(available)),
        ram_used_mb=max(0, int(total - available)),
        gpus=gpus,
    )


def choose_model_for_hardware(hw: HardwareProfile, backend: str = BACKEND_LLAMA_CPP) -> AutoModelChoice:
    backend = normalize_setup_backend(backend)
    catalog = model_catalog_for_backend(backend)
    total = int(hw.ram_total_mb or 0)
    available = int(hw.ram_available_mb or 0)
    vram = int(hw.primary_vram_free_mb or hw.primary_vram_mb or 0)
    effective_gpu = hw.accelerator in {"cuda", "rocm", "vulkan", "metal"} and vram >= 2048

    if total >= 24576 and (vram >= 8192 or (not effective_gpu and available >= 14000)):
        return catalog[3]
    if total >= 12288 and (vram >= 4096 or available >= 7000):
        return catalog[2]
    if total >= 7168 and (vram >= 3072 or available >= 4500):
        return catalog[1]
    return catalog[0]


def model_catalog_for_backend(backend: str = BACKEND_LLAMA_CPP) -> list[AutoModelChoice]:
    normalized = normalize_setup_backend(backend)
    return AUTO_HF_MODEL_CATALOG if normalized == BACKEND_HF_TRANSFORMERS else AUTO_MODEL_CATALOG


def recommended_threads(hw: HardwareProfile, choice: AutoModelChoice) -> int:
    cpus = max(1, int(hw.cpu_threads or 1))
    if choice.tier == "minimal":
        return max(1, min(cpus, 4))
    if choice.tier in {"low", "balanced"}:
        return max(2, min(cpus - 1 if cpus > 2 else cpus, 8))
    return max(4, min(cpus - 2 if cpus > 6 else cpus, 12))


def choose_runtime_asset(releases: list, accelerator: str) -> Optional[dict[str, Any]]:
    accel = str(accelerator or "cpu").lower()
    avoid_gpu = {"cuda", "cublas", "cu12", "rocm", "hip", "vulkan", "kompute", "metal"}

    def score(name: str) -> int:
        n = name.lower()
        value = 0
        if accel == "cuda":
            value += 90 if any(k in n for k in ("cuda", "cublas", "cu12", "cu11")) else 0
            value += 10 if "vulkan" in n else 0
        elif accel == "rocm":
            value += 90 if any(k in n for k in ("rocm", "hip", "hipblas")) else 0
            value += 10 if "vulkan" in n else 0
        elif accel == "vulkan":
            value += 90 if "vulkan" in n or "kompute" in n else 0
        elif accel == "metal":
            value += 90 if "metal" in n or "macos" in n else 0
        else:
            value += 35 if not any(k in n for k in avoid_gpu) else -100
            value += 15 if "cpu" in n else 0
        if any(k in n for k in ("x64", "x86_64", "amd64", "aarch64", "arm64")):
            value += 5
        if "avx2" in n:
            value += 3
        if "noavx" in n:
            value -= 3
        return value

    best: Optional[dict[str, Any]] = None
    best_score = -1
    for rel in releases or []:
        tag = rel.get("tag", "")
        for asset in rel.get("assets", []) or []:
            name = str(asset.get("name", ""))
            if not name:
                continue
            asset_score = score(name)
            if best is None or asset_score > best_score:
                best = dict(asset)
                best["tag"] = tag
                best_score = asset_score
        if best is not None and best_score >= 90:
            break
    if accel == "cpu" and best_score < 0:
        return None
    return best


def choose_gguf_file(files: list, choice: AutoModelChoice) -> Optional[dict[str, Any]]:
    if not files:
        return None
    prefs = [q.upper() for q in choice.quant_preferences]

    def score(row: dict[str, Any]) -> tuple[int, int, str]:
        name = str(row.get("rfilename", ""))
        lower = name.lower()
        if not lower.endswith(".gguf"):
            return (-1000, 0, lower)
        if any(k in lower for k in ("mmproj", "projector", "clip")):
            return (-1000, 0, lower)
        if "-of-" in lower or "00001-of-" in lower:
            return (-900, 0, lower)
        value = 0
        upper = name.upper()
        for idx, quant in enumerate(prefs):
            if quant in upper:
                value += 120 - idx * 8
                break
        if "instruct" in lower or "chat" in lower:
            value += 8
        if "imatrix" in lower or "imat" in lower:
            value += 3
        size = int(row.get("size") or row.get("lfs", {}).get("size") or 0)
        return (value, -size, lower)

    ranked = sorted(files, key=score, reverse=True)
    if score(ranked[0])[0] <= 0:
        return None
    return ranked[0]


class _AutoSetupStopped(Exception):
    pass


class AutoSetupWorker(QThread):
    status = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(str)
    err = pyqtSignal(str)
    paused = pyqtSignal(bool)
    plan_ready = pyqtSignal(object)

    def __init__(self, *, resume: bool = True, backend: str = BACKEND_LLAMA_CPP, parent=None):
        super().__init__(parent)
        self._resume = bool(resume)
        self._pause_requested = False
        self._stop_requested = False
        self._current_worker: Any = None
        self._state: dict[str, Any] = read_auto_setup_state() if resume else {}
        resumable = auto_setup_resumable(self._state)
        if not resumable:
            self._state = {}
        saved_backend = self._state.get("backend") if resumable else ""
        self._backend = normalize_setup_backend(saved_backend or backend)
        self._state["backend"] = self._backend

    def pause(self):
        self._pause_requested = True
        self._save(status="paused")
        worker = self._current_worker
        if worker and hasattr(worker, "pause"):
            worker.pause()
        self.paused.emit(True)

    def resume(self):
        self._pause_requested = False
        self._save(status="running")
        worker = self._current_worker
        if worker and hasattr(worker, "resume"):
            worker.resume()
        self.paused.emit(False)

    def cancel(self):
        self._stop_requested = True
        self._pause_requested = True
        self._save(status="paused")
        worker = self._current_worker
        if worker and hasattr(worker, "abort"):
            try:
                worker.abort(delete_part=False)
            except TypeError:
                worker.abort()
        self.paused.emit(True)

    def run(self):
        try:
            self._run_pipeline()
        except _AutoSetupStopped:
            self.status.emit("Setup paused. Resume will continue from the saved checkpoint.")
            self.paused.emit(True)
        except Exception as exc:
            self._save(status="error", error=str(exc))
            self.err.emit(str(exc))

    def _run_pipeline(self):
        self._save(status="running", stage=self._state.get("stage") or "hardware")
        hw = self._hardware()
        backend = self._setup_backend()
        choice = self._choice(hw, backend)
        self.plan_ready.emit({
            "hardware": asdict(hw),
            "model": asdict(choice),
            "backend": backend,
            "backend_label": setup_backend_label(backend),
            "threads": recommended_threads(hw, choice),
            "accelerator": hw.accelerator,
        })
        self._pause_gate()

        runtime_path = ""
        if backend == BACKEND_LLAMA_CPP:
            runtime_path = self._install_runtime(hw)
            self._pause_gate()

            model_path = self._download_gguf_model(choice)
        else:
            model_path = self._download_hf_model(choice)
        self._pause_gate()

        registered_ref = self._register_model(model_path, choice, hw, runtime_path, backend)
        self._save(status="complete", stage="complete", model_path=model_path, registered_ref=registered_ref)
        self.done.emit(registered_ref)

    def _save(self, **updates):
        self._state.update(updates)
        self._state["updated_at"] = _now()
        write_auto_setup_state(self._state)

    def _run_child_worker(self, worker: Any, label: str) -> str:
        result: dict[str, str] = {}

        def _progress(*args):
            done = _coerce_int(args[0], 0) if len(args) >= 1 else 0
            total = _coerce_int(args[1], 0) if len(args) >= 2 else 0
            detail = str(args[2] or "") if len(args) >= 3 else ""
            progress_label = f"{label}: {detail}" if detail else label
            self.progress.emit(done, total, progress_label)

        worker.progress.connect(_progress)
        if hasattr(worker, "status"):
            worker.status.connect(self.status.emit)
        if hasattr(worker, "paused"):
            worker.paused.connect(self.paused.emit)
        worker.done.connect(lambda path: result.update(path=str(path or "")))
        worker.err.connect(lambda msg: result.update(error=str(msg or "")))
        self._current_worker = worker
        worker.run()
        self._current_worker = None
        if self._stop_requested:
            raise _AutoSetupStopped()
        if result.get("error"):
            raise RuntimeError(result["error"])
        path = result.get("path", "")
        if not path:
            raise _AutoSetupStopped()
        return path

    def _pause_gate(self):
        while self._pause_requested and not self._stop_requested:
            time.sleep(0.2)
        if self._stop_requested:
            raise _AutoSetupStopped()

    def _hardware(self) -> HardwareProfile:
        if isinstance(self._state.get("hardware"), dict):
            try:
                return HardwareProfile(**self._state["hardware"])
            except Exception:
                pass
        self.status.emit("Checking CPU, RAM, OS memory use, and GPU backend...")
        hw = profile_hardware()
        self._save(stage="hardware", hardware=asdict(hw))
        return hw

    def _setup_backend(self) -> str:
        backend = normalize_setup_backend(self._state.get("backend") or self._backend)
        self._backend = backend
        self._save(backend=backend)
        return backend

    def _choice(self, hw: HardwareProfile, backend: str) -> AutoModelChoice:
        if isinstance(self._state.get("model_choice"), dict):
            try:
                choice = AutoModelChoice(**self._state["model_choice"])
                if normalize_setup_backend(choice.backend) == normalize_setup_backend(backend):
                    return choice
            except Exception:
                pass
        choice = choose_model_for_hardware(hw, backend)
        self.status.emit(f"Selected {choice.label} for {setup_backend_label(backend)}.")
        self._save(stage="model_plan", model_choice=asdict(choice))
        return choice

    def _install_runtime(self, hw: HardwareProfile) -> str:
        if self._state.get("runtime_done") and self._state.get("runtime_path"):
            return str(self._state.get("runtime_path"))
        asset = self._state.get("runtime_asset")
        if not isinstance(asset, dict):
            self.status.emit("Fetching latest llama.cpp release list...")
            self._save(stage="runtime_fetch")
            releases = fetch_llama_cpp_releases()
            asset = choose_runtime_asset(releases, hw.accelerator)
            if not asset:
                raise RuntimeError("No compatible llama.cpp release asset found for this platform.")
            self._save(stage="runtime_install", runtime_asset=asset)
        self.status.emit(f"Installing llama.cpp {asset.get('tag', '')}: {asset.get('name', '')}")
        worker = LlamaCppDownloadWorker(
            url=str(asset.get("url") or ""),
            filename=str(asset.get("name") or "llama.cpp-runtime.zip"),
            dest_dir=Path("./llama"),
            expected_size=int(asset.get("size") or 0),
        )
        runtime_path = self._run_child_worker(worker, "llama.cpp runtime") or str(Path("./llama/bin"))
        self._save(stage="runtime_done", runtime_done=True, runtime_path=runtime_path)
        return runtime_path

    def _download_gguf_model(self, choice: AutoModelChoice) -> str:
        saved_path = str(self._state.get("model_path") or "")
        if saved_path and Path(saved_path).exists():
            return saved_path

        file_row = self._state.get("model_file")
        if str(self._state.get("model_repo") or "").strip("/") != choice.repo.strip("/"):
            file_row = {}
        if not isinstance(file_row, dict) or not file_row:
            self.status.emit(f"Searching Hugging Face repo {choice.repo}...")
            self._save(stage="model_search")
            token = get_hf_access_token()
            try:
                files = fetch_hf_gguf_files(choice.repo, token=token)
            except Exception as exc:
                raise RuntimeError(normalize_hf_exception(exc, repo_id=choice.repo))
            file_row = choose_gguf_file(files, choice)
            if not file_row:
                raise RuntimeError(f"No suitable single-file GGUF found in {choice.repo}.")
            self._save(stage="model_download", model_file=file_row, model_repo=choice.repo)

        filename = str(file_row.get("rfilename") or "")
        if not filename:
            raise RuntimeError("Selected Hugging Face file has no filename.")
        dest = MODELS_DIR
        final_path = dest / filename
        if final_path.exists():
            self._save(model_path=str(final_path))
            return str(final_path)

        self.status.emit(f"Downloading model: {filename}")
        worker = HfDownloadWorker(
            choice.repo,
            filename,
            dest,
            expected_size=int(file_row.get("size") or file_row.get("lfs", {}).get("size") or 0),
            token=get_hf_access_token(),
        )
        path = self._run_child_worker(worker, "model download")
        self._save(model_path=path)
        return path

    def _download_hf_model(self, choice: AutoModelChoice) -> str:
        saved_path = str(self._state.get("model_path") or "")
        if saved_path and Path(saved_path).exists():
            return saved_path

        snapshot = self._state.get("hf_snapshot")
        if isinstance(snapshot, dict) and str(snapshot.get("repo") or "").strip("/") != choice.repo.strip("/"):
            snapshot = {}
        if not isinstance(snapshot, dict) or not snapshot:
            self.status.emit(f"Preparing Hugging Face snapshot for {choice.repo}...")
            self._save(stage="hf_snapshot_search")
            try:
                snapshot = fetch_hf_snapshot_files(choice.repo, "main", get_hf_access_token())
            except Exception as exc:
                raise RuntimeError(normalize_hf_exception(exc, repo_id=choice.repo))
            if not snapshot.get("files"):
                raise RuntimeError(f"No Transformers runtime files found in {choice.repo}.")
            self._save(stage="hf_snapshot_download", hf_snapshot=snapshot)

        dest_root = Path(str(APP_CONFIG.get("hf_transformers_dir") or "localllm/hf_transformers")).expanduser()
        worker = HfSnapshotDownloadWorker(
            choice.repo,
            str(snapshot.get("revision") or "main"),
            list(snapshot.get("files") or []),
            dest_root,
            token=get_hf_access_token(),
        )
        path = self._run_child_worker(worker, "HF snapshot")
        self._save(model_path=path)
        return path

    def _register_model(
        self,
        model_path: str,
        choice: AutoModelChoice,
        hw: HardwareProfile,
        runtime_path: str,
        backend: str,
    ) -> str:
        self.status.emit("Registering model and applying machine-tuned defaults...")
        self._save(stage="register")
        backend = normalize_setup_backend(backend)
        registry_path = make_hf_model_ref(model_path) if backend == BACKEND_HF_TRANSFORMERS else model_path
        registry = get_model_registry()
        registry.add(registry_path)
        fam = detect_model_family(model_path)
        threads = recommended_threads(hw, choice)
        ctx = max(512, min(MAX_CONTEXT_TOKENS, int(choice.ctx)))
        cfg = ModelConfig(
            path=registry_path,
            role="general",
            backend=backend,
            vision=False,
            threads=threads,
            ctx=ctx,
            n_predict=int(choice.n_predict),
            temperature=float(choice.temperature),
            top_p=float(choice.top_p),
            repeat_penalty=float(choice.repeat_penalty),
            family=fam.family,
        )
        registry.set_config(registry_path, cfg)

        if backend == BACKEND_HF_TRANSFORMERS:
            self.status.emit(f"Ready: {model_ref_display_name(registry_path)}")
            return registry_path

        runtime = Path(runtime_path)
        server = runtime / ("llama-server.exe" if platform.system() == "Windows" else "llama-server")
        cli = runtime / ("llama-cli.exe" if platform.system() == "Windows" else "llama-cli")
        if server.exists():
            SERVER_CONFIG.server_path = str(server)
        if cli.exists():
            SERVER_CONFIG.cli_path = str(cli)

        gpus = hw.gpus or []
        SERVER_CONFIG.enable_gpu = bool(gpus and hw.accelerator in {"cuda", "rocm", "vulkan", "metal"})
        SERVER_CONFIG.ngl = -1 if SERVER_CONFIG.enable_gpu else 0
        SERVER_CONFIG.main_gpu = int(gpus[0].get("idx") or 0) if gpus else 0
        if len(gpus) > 1:
            total_vram = sum(max(0, int(g.get("vram_mb") or 0)) for g in gpus)
            if total_vram > 0:
                SERVER_CONFIG.tensor_split = ",".join(
                    f"{max(0, int(g.get('vram_mb') or 0)) / total_vram:.3f}".rstrip("0").rstrip(".")
                    for g in gpus
                )
        SERVER_CONFIG.save()
        refresh_binary_paths()
        self.status.emit(f"Ready: {model_ref_display_name(registry_path)}")
        return registry_path
