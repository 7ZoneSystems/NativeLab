from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass

from .paths import mobile_target


@dataclass(frozen=True)
class HardwareProfile:
    target: str
    system: str
    arch: str
    cpu_threads: int
    ram_total_mb: int
    ram_available_mb: int
    accelerator: str

    def to_dict(self) -> dict:
        return asdict(self)


def _memory_with_psutil() -> tuple[int, int] | None:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.total / (1024 * 1024)), int(vm.available / (1024 * 1024))
    except Exception:
        return None


def _memory_with_sysconf() -> tuple[int, int] | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = int((pages * page_size) / (1024 * 1024))
        return total, max(512, int(total * 0.55))
    except Exception:
        return None


def detect_accelerator(target: str) -> str:
    if target == "ios":
        return "metal"
    if target == "android":
        return "vulkan"
    system = platform.system().lower()
    if system == "darwin":
        return "metal"
    return "cpu"


def profile_hardware() -> HardwareProfile:
    target = mobile_target()
    memory = _memory_with_psutil() or _memory_with_sysconf() or (4096, 2048)
    total, available = memory
    cpus = max(1, int(os.cpu_count() or 4))
    return HardwareProfile(
        target=target,
        system=platform.system() or target,
        arch=platform.machine() or "",
        cpu_threads=cpus,
        ram_total_mb=max(0, total),
        ram_available_mb=max(0, min(available, total)),
        accelerator=detect_accelerator(target),
    )


def recommended_threads(profile: HardwareProfile) -> int:
    cpus = max(1, profile.cpu_threads)
    if profile.target in {"android", "ios"}:
        return max(1, min(cpus, 4))
    return max(1, min(cpus, 8))
