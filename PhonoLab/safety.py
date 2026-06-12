from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import MobileConfig
from .native import estimate_tokens, prompt_fits_context
from .paths import MODELS_DIR


REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


class SafetyError(RuntimeError):
    pass


class ContextLimitError(SafetyError):
    def __init__(self, prompt_tokens: int, context_tokens: int, reserved_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.context_tokens = context_tokens
        self.reserved_tokens = reserved_tokens
        super().__init__(
            "Prompt is too large for the loaded mobile context: "
            f"{prompt_tokens} prompt tokens + {reserved_tokens} reserved output tokens > "
            f"{context_tokens} context tokens."
        )


@dataclass(frozen=True)
class UserNotice:
    title: str
    summary: str
    action: str
    technical_detail: str = ""

    @property
    def user_message(self) -> str:
        return f"{self.summary}\n\nWhat to do:\n{self.action}"


def validate_repo_id(repo_id: str) -> str:
    repo = str(repo_id or "").strip().strip("/")
    if not REPO_ID_RE.match(repo):
        raise SafetyError("Use a Hugging Face repo id like owner/model-name.")
    return repo


def validate_model_path(path: str | Path, *, require_under_models: bool = True) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.exists() or not candidate.is_file():
        raise SafetyError(f"Model file does not exist: {candidate}")
    if candidate.suffix.lower() != ".gguf":
        raise SafetyError("PhonoLab mobile only supports local GGUF models through llama.cpp.")
    if require_under_models:
        base = MODELS_DIR.resolve()
        resolved = candidate.resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise SafetyError("Mobile models must be stored under the PhonoLab models directory.") from exc
    return candidate


def guard_prompt(prompt: str, config: MobileConfig) -> int:
    text = str(prompt or "")
    if len(text) > int(config.max_prompt_chars):
        raise SafetyError(
            f"Prompt has {len(text)} characters, which exceeds the mobile limit of "
            f"{config.max_prompt_chars} characters."
        )
    tokens = estimate_tokens(text)
    if not prompt_fits_context(text, config.ctx, config.n_predict):
        raise ContextLimitError(tokens, config.ctx, config.n_predict)
    return tokens


def sampler_args(config: MobileConfig) -> dict[str, Any]:
    return {
        "temperature": max(0.0, min(2.0, float(config.temperature))),
        "top_p": max(0.05, min(1.0, float(config.top_p))),
        "repeat_penalty": max(0.8, min(2.0, float(config.repeat_penalty))),
        "n_predict": max(32, min(4096, int(config.n_predict))),
        "ctx": max(512, min(32768, int(config.ctx))),
        "threads": max(1, min(8, int(config.threads))),
    }


def explain_error(raw: Any, *, source: str = "PhonoLab mobile engine") -> UserNotice:
    text = str(raw or "")
    try:
        from nativelab.core.llm_errors import explain_llm_error

        notice = explain_llm_error(text, source=source)
        return UserNotice(notice.title, notice.summary, notice.action, notice.technical_detail)
    except Exception:
        pass

    lower = text.lower()
    if "context" in lower and ("exceed" in lower or "token" in lower):
        return UserNotice(
            "Mobile Context Is Full",
            f"{source} could not run because the request is larger than the context window.",
            "- Send less text.\n- Lower output tokens.\n- Load the model with a larger context if the device can handle it.",
            text,
        )
    if "not found" in lower or "no such file" in lower:
        return UserNotice(
            "Mobile Runtime Missing",
            f"{source} could not find a required file.",
            "- Pull llama.cpp from setup.\n- Download or register a GGUF model.\n- Check that the binary/model path is still valid.",
            text,
        )
    return UserNotice(
        "PhonoLab Engine Error",
        f"{source} failed.",
        "- Try again with a smaller prompt.\n- Reload the model.\n- Re-run mobile setup if the runtime is missing.",
        text,
    )


def json_dumps_safe(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)
