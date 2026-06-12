from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import MobileConfig, load_config
from .paths import DOWNLOADS_DIR, MODELS_DIR, safe_child_path
from .safety import SafetyError, validate_repo_id
from .small_models import MobileModelCandidate


ProgressCallback = Callable[[int, int, str], None]


class DownloadAborted(RuntimeError):
    pass


@dataclass(frozen=True)
class HfFile:
    name: str
    size: int = 0


def _headers(config: MobileConfig | None = None) -> dict[str, str]:
    cfg = config or load_config()
    headers = {"User-Agent": cfg.user_agent}
    if cfg.hf_token:
        headers["Authorization"] = f"Bearer {cfg.hf_token}"
    return headers


def fetch_hf_gguf_files(repo_id: str, config: MobileConfig | None = None) -> list[HfFile]:
    repo = validate_repo_id(repo_id)
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}", headers=_headers(config))
    with urllib.request.urlopen(req, timeout=45) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace"))
    files: list[HfFile] = []
    for row in data.get("siblings", []):
        name = str(row.get("rfilename") or "")
        if not name.lower().endswith(".gguf"):
            continue
        size = int(row.get("size") or row.get("lfs", {}).get("size") or 0)
        files.append(HfFile(name=name, size=size))
    return files


def choose_gguf_file(files: list[HfFile], candidate: MobileModelCandidate) -> HfFile:
    prefs = [item.upper() for item in candidate.quant_preferences]

    def score(item: HfFile) -> tuple[int, int, str]:
        lower = item.name.lower()
        if not lower.endswith(".gguf"):
            return (-1000, 0, lower)
        if any(part in lower for part in ("mmproj", "projector", "clip")):
            return (-1000, 0, lower)
        if "-of-" in lower or "00001-of-" in lower:
            return (-900, 0, lower)
        value = 0
        upper = item.name.upper()
        for idx, quant in enumerate(prefs):
            if quant in upper:
                value += 120 - idx * 8
                break
        if "instruct" in lower or "chat" in lower:
            value += 8
        if "imat" in lower or "imatrix" in lower:
            value += 3
        return (value, -int(item.size or 0), lower)

    ranked = sorted(files, key=score, reverse=True)
    if not ranked or score(ranked[0])[0] <= 0:
        raise SafetyError(f"No compatible GGUF file found for {candidate.label}.")
    return ranked[0]


class ResumableDownload:
    CHUNK = 256 * 1024
    MAX_RETRIES = 3

    def __init__(
        self,
        url: str,
        dest: Path,
        *,
        expected_size: int = 0,
        max_bytes: int | None = None,
        config: MobileConfig | None = None,
    ):
        self.url = str(url)
        self.dest = Path(dest)
        self.expected_size = int(expected_size or 0)
        self.max_bytes = int(max_bytes if max_bytes is not None else load_config().max_download_bytes)
        self.config = config or load_config()
        self._pause = threading.Event()
        self._pause.set()
        self._abort = threading.Event()

    @property
    def part_path(self) -> Path:
        return self.dest.with_name(self.dest.name + ".part")

    def pause(self) -> None:
        self._pause.clear()

    def resume(self) -> None:
        self._pause.set()

    def abort(self) -> None:
        self._abort.set()
        self._pause.set()

    def run(self, progress: ProgressCallback | None = None) -> Path:
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        if self.expected_size and self.expected_size > self.max_bytes:
            raise SafetyError("Download is larger than the configured mobile safety limit.")
        last_error: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._run_once(progress)
                self.part_path.replace(self.dest)
                return self.dest
            except DownloadAborted:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.MAX_RETRIES:
                    break
                time.sleep(2 * attempt)
        raise RuntimeError(str(last_error or "Download failed"))

    def _run_once(self, progress: ProgressCallback | None) -> None:
        resume_from = self.part_path.stat().st_size if self.part_path.exists() else 0
        headers = _headers(self.config)
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        req = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(req, timeout=45) as response:
            status = int(getattr(response, "status", 200) or 200)
            if resume_from and status != 206:
                resume_from = 0
            content_length = int(response.headers.get("Content-Length") or 0)
            total = self.expected_size or (resume_from + content_length if status == 206 else content_length)
            if total and total > self.max_bytes:
                raise SafetyError("Download is larger than the configured mobile safety limit.")
            mode = "ab" if resume_from else "wb"
            done = resume_from
            with open(self.part_path, mode) as handle:
                while True:
                    self._pause.wait()
                    if self._abort.is_set():
                        raise DownloadAborted()
                    chunk = response.read(self.CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if done > self.max_bytes:
                        raise SafetyError("Download exceeded the configured mobile safety limit.")
                    if progress:
                        progress(done, total, self.dest.name)
        if total and done < total:
            raise RuntimeError(f"Size mismatch: expected {total} bytes, got {done}.")


def hf_resolve_url(repo_id: str, filename: str, revision: str = "main") -> str:
    from urllib.parse import quote

    repo = validate_repo_id(repo_id)
    safe_name = "/".join(quote(part, safe="") for part in str(filename).split("/"))
    return f"https://huggingface.co/{repo}/resolve/{quote(revision, safe='')}/{safe_name}"


def download_candidate_model(
    candidate: MobileModelCandidate,
    *,
    progress: ProgressCallback | None = None,
    config: MobileConfig | None = None,
) -> tuple[Path, HfFile]:
    cfg = config or load_config()
    files = fetch_hf_gguf_files(candidate.repo, cfg)
    selected = choose_gguf_file(files, candidate)
    dest = safe_child_path(MODELS_DIR / candidate.key, Path(selected.name).name)
    job = ResumableDownload(
        hf_resolve_url(candidate.repo, selected.name),
        dest,
        expected_size=selected.size,
        config=cfg,
    )
    return job.run(progress), selected


def download_to_cache(url: str, filename: str, *, progress: ProgressCallback | None = None) -> Path:
    dest = safe_child_path(DOWNLOADS_DIR, filename)
    return ResumableDownload(url, dest).run(progress)
