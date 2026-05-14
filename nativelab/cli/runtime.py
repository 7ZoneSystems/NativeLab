"""Shared CLI runtime wiring for engines, Labs, skills, and integrations."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from nativelab.core.engine_global import ApiEngine, LlamaEngine
from nativelab.integrations import IntegrationEndpoints
from nativelab.labs.endpoints import LabEndpoints
from nativelab.Model.model_global import (
    api_model_ref,
    getapi_registry,
    is_api_model_ref,
)
from nativelab.skill import active_skill_context, ensure_builtin_edit_skill

from . import onboarding, ui


class CliRuntime:
    """One runtime shared by CLI menus, REPL commands, Labs, and integrations."""

    def __init__(self, model_path: str = "", ctx: int = 0, *, skills_enabled: bool = False):
        from nativelab.GlobalConfig.config_global import DEFAULT_CTX

        self.model_path = model_path or ""
        self.ctx = int(ctx or DEFAULT_CTX())
        self.skills_enabled = bool(skills_enabled)
        self.llama = LlamaEngine()
        self.api: Optional[ApiEngine] = None
        self.endpoints = LabEndpoints()
        self.integrations = IntegrationEndpoints(self.endpoints)
        ensure_builtin_edit_skill()
        self._bind()
        if self.model_path:
            self.load_model(self.model_path, save=False)

    def _bind(self) -> None:
        self.endpoints.bind_engines(
            llama_provider=lambda: self.llama,
            api_provider=lambda: self.api,
        )
        self.endpoints.bind_reverse_routes(
            on_context=self.set_context,
            on_model=self.load_model,
            on_unload=self.unload,
        )
        self.endpoints.set_skill_context_provider(
            lambda: active_skill_context() if self.skills_enabled else ""
        )
        self.integrations.bind_lab_endpoints(self.endpoints)

    def set_skills_enabled(self, enabled: bool, *, save: bool = True) -> None:
        self.skills_enabled = bool(enabled)
        if save:
            prefs = onboarding.load_prefs()
            prefs["skills_enabled"] = self.skills_enabled
            onboarding.save_prefs(prefs)

    def load_model(self, target: str, *, save: bool = True) -> bool:
        target = str(target or "").strip()
        if not target:
            return False
        if is_api_model_ref(target) or getapi_registry().get(target):
            return self._load_api(target, save=save)
        path = Path(target).expanduser()
        if not path.exists():
            return False
        if self.api and self.api.is_loaded:
            self.api.shutdown()
            self.api = None
        try:
            self.llama.shutdown()
        except Exception:
            pass
        self.llama = LlamaEngine()
        self._bind()
        ui.info(f"Loading model: {path.name}")
        ok = self.llama.load(str(path), ctx=self.ctx, log_cb=lambda m: ui.info(m))
        if ok:
            self.model_path = str(path)
            if save:
                self._save_model_prefs()
        self.endpoints.notify_engine_changed()
        return bool(ok)

    def _load_api(self, ref_or_name: str, *, save: bool = True) -> bool:
        ref = ref_or_name if is_api_model_ref(ref_or_name) else api_model_ref(ref_or_name)
        cfg = getapi_registry().get_by_ref(ref)
        if cfg is None:
            return False
        try:
            self.llama.shutdown()
        except Exception:
            pass
        if self.api and self.api.is_loaded:
            self.api.shutdown()
        ui.info(f"Loading API model: {cfg.name} ({cfg.model_id})")
        api = ApiEngine()
        ok = api.load(cfg, log_cb=lambda m: ui.info(m))
        if ok:
            self.api = api
            self.model_path = ref
            if save:
                self._save_model_prefs()
        self._bind()
        self.endpoints.notify_engine_changed()
        return bool(ok)

    def set_context(self, new_ctx: int, *, save: bool = True) -> bool:
        self.ctx = int(new_ctx)
        if self.api and self.api.is_loaded:
            ui.warn("Context changes apply to local GGUF models; API context is configured in the API profile.")
            if save:
                self._save_model_prefs()
            return True
        current = self.model_path
        if not current or not Path(current).exists():
            return False
        try:
            self.llama.shutdown()
        except Exception:
            pass
        self.llama = LlamaEngine()
        self._bind()
        ok = self.llama.load(current, ctx=self.ctx, log_cb=lambda m: ui.info(m))
        if ok and save:
            self._save_model_prefs()
        self.endpoints.notify_engine_changed()
        return bool(ok)

    def unload(self) -> None:
        try:
            self.llama.shutdown()
        except Exception:
            pass
        if self.api:
            self.api.shutdown()
            self.api = None
        self.endpoints.notify_engine_changed()

    def _save_model_prefs(self) -> None:
        prefs = onboarding.load_prefs()
        prefs.update({
            "model_path": self.model_path,
            "ctx": self.ctx,
            "skills_enabled": self.skills_enabled,
        })
        onboarding.save_prefs(prefs)


def runtime_from_prefs(model: str = "", ctx: int = 0, *, autoload: bool = True) -> CliRuntime:
    prefs = onboarding.load_prefs()
    target = model or prefs.get("model_path", "")
    size = int(ctx or prefs.get("ctx", 0) or 0)
    skills_enabled = bool(prefs.get("skills_enabled", False))
    return CliRuntime(target if autoload else "", size, skills_enabled=skills_enabled)
