from __future__ import annotations

from dataclasses import dataclass

from .hardware import HardwareProfile


@dataclass(frozen=True)
class MobileModelCandidate:
    key: str
    label: str
    repo: str
    quant_preferences: tuple[str, ...]
    min_ram_mb: int
    ctx: int
    tier: str


SMALL_MODEL_CATALOG: tuple[MobileModelCandidate, ...] = (
    MobileModelCandidate(
        key="smollm2-360m",
        label="SmolLM2 360M Instruct",
        repo="bartowski/SmolLM2-360M-Instruct-GGUF",
        quant_preferences=("Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"),
        min_ram_mb=2048,
        ctx=2048,
        tier="tiny",
    ),
    MobileModelCandidate(
        key="qwen25-05b",
        label="Qwen2.5 0.5B Instruct",
        repo="bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        quant_preferences=("Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"),
        min_ram_mb=3072,
        ctx=2048,
        tier="minimal",
    ),
    MobileModelCandidate(
        key="llama32-1b",
        label="Llama 3.2 1B Instruct",
        repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        quant_preferences=("Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"),
        min_ram_mb=4096,
        ctx=2048,
        tier="low",
    ),
    MobileModelCandidate(
        key="qwen25-15b",
        label="Qwen2.5 1.5B Instruct",
        repo="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        quant_preferences=("Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"),
        min_ram_mb=6144,
        ctx=4096,
        tier="balanced",
    ),
    MobileModelCandidate(
        key="tinyllama-11b",
        label="TinyLlama 1.1B Chat",
        repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        quant_preferences=("Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"),
        min_ram_mb=4096,
        ctx=2048,
        tier="fallback",
    ),
)


def choose_candidate(profile: HardwareProfile) -> MobileModelCandidate:
    total = int(profile.ram_total_mb or 0)
    eligible = [item for item in SMALL_MODEL_CATALOG if total >= item.min_ram_mb]
    if not eligible:
        return SMALL_MODEL_CATALOG[0]
    if profile.target in {"android", "ios"} and total < 6144:
        return eligible[min(len(eligible) - 1, 2)]
    return eligible[-1]


def by_key(key: str) -> MobileModelCandidate | None:
    wanted = str(key or "").strip()
    for item in SMALL_MODEL_CATALOG:
        if item.key == wanted:
            return item
    return None
