from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class IntegrationEndpoints:
    """
    Read-only discovery endpoint for external integrations.

    This object intentionally returns plain dictionaries so chat bots, cloud
    functions, local scripts, and HTTP wrappers can serialize the response
    directly. When bound inside the GUI it can also include the live engine
    snapshot exposed by `LabEndpoints`.
    """

    def __init__(self, lab_endpoints=None):
        self._lab_endpoints = lab_endpoints

    def bind_lab_endpoints(self, lab_endpoints) -> None:
        self._lab_endpoints = lab_endpoints

    # Catalog / routing
    def routes(self) -> list[Dict[str, Any]]:
        return [
            {
                "path": "/snapshot",
                "method": "GET",
                "description": "Full integration catalog and live runtime snapshot.",
            },
            {
                "path": "/runtime",
                "method": "GET",
                "description": "Current loaded backend, model, context and server port.",
            },
            {
                "path": "/models",
                "method": "GET",
                "description": "Registered local GGUF models with roles and detected metadata.",
            },
            {
                "path": "/api_models",
                "method": "GET",
                "description": "Saved API model configs with secrets redacted.",
            },
            {
                "path": "/limits",
                "method": "GET",
                "description": "Runtime limits and tunables from app_config.json.",
            },
            {
                "path": "/pipelines",
                "method": "GET",
                "description": "Saved visual pipelines and block/connection counts.",
            },
            {
                "path": "/pipelines/{name}",
                "method": "GET",
                "description": "Raw saved pipeline JSON definition.",
            },
            {
                "path": "/labs",
                "method": "GET",
                "description": "Registered lab features and their integration paths.",
            },
            {
                "path": "/labs/py_to_doc",
                "method": "GET",
                "description": "py-to-doc lab metadata and expected integration inputs.",
            },
            {
                "path": "/call_llm",
                "method": "POST",
                "description": "Run prompt/messages through the active NativeLab engine.",
            },
            {
                "path": "/integrations/discord_bots",
                "method": "GET",
                "description": "Saved Discord connector profiles with tokens redacted.",
            },
            {
                "path": "/integrations/whatsapp_bots",
                "method": "GET",
                "description": "Saved WhatsApp connector profiles with tokens redacted.",
            },
        ]

    def catalog(self) -> Dict[str, Any]:
        return {
            "name": "NativeLab Integrations",
            "version": "v0.3.0",
            "routes": self.routes(),
            "http": {
                "default_host": "127.0.0.1",
                "default_port": 8765,
                "content_type": "application/json",
            },
            "actions": {
                "call_llm": "Route a prompt/messages through the active NativeLab engine.",
                "request_context": "Ask the host app to update context size.",
                "request_load_model": "Ask the host app to load a local model path.",
                "request_unload": "Ask the host app to unload the active model.",
            },
            "runtime": self.runtime(),
            "limits": self.limits(),
            "models": self.models(),
            "api_models": self.api_models(),
            "pipelines": self.pipelines(),
            "labs": self.labs(),
            "integrations": {
                "discord_bots": self.discord_bots(),
                "whatsapp_bots": self.whatsapp_bots(),
            },
        }

    def handle(self, path: str) -> Dict[str, Any]:
        clean = "/" + (path or "").strip("/")
        if clean in {"/", "/snapshot"}:
            return self.catalog()
        if clean == "/runtime":
            return self.runtime()
        if clean == "/models":
            return {"models": self.models()}
        if clean == "/api_models":
            return {"api_models": self.api_models()}
        if clean == "/limits":
            return self.limits()
        if clean == "/pipelines":
            return {"pipelines": self.pipelines()}
        if clean.startswith("/pipelines/"):
            return self.pipeline_definition(clean.split("/", 2)[2])
        if clean == "/labs":
            return {"labs": self.labs()}
        if clean.startswith("/labs/"):
            return self.lab_definition(clean.split("/", 2)[2])
        if clean == "/integrations/discord_bots":
            return {"discord_bots": self.discord_bots()}
        if clean == "/integrations/whatsapp_bots":
            return {"whatsapp_bots": self.whatsapp_bots()}
        return {
            "error": "unknown integration endpoint",
            "path": clean,
            "available_routes": [r["path"] for r in self.routes()],
        }

    def to_json(self, path: str = "/snapshot", *, indent: int = 2) -> str:
        return json.dumps(self.handle(path), indent=indent, ensure_ascii=False)

    def call_llm(self, *args, **kwargs) -> str:
        if self._lab_endpoints is None:
            raise RuntimeError("Integration endpoint is not bound to a running app")
        return self._lab_endpoints.call_llm(*args, **kwargs)

    def request_context(self, new_ctx: int) -> bool:
        if self._lab_endpoints is None:
            return False
        return bool(self._lab_endpoints.request_context(new_ctx))

    def request_load_model(self, model_path: str) -> bool:
        if self._lab_endpoints is None:
            return False
        return bool(self._lab_endpoints.request_load_model(model_path))

    def request_unload(self) -> None:
        if self._lab_endpoints is not None:
            self._lab_endpoints.request_unload()

    # Sections
    def runtime(self) -> Dict[str, Any]:
        if self._lab_endpoints is None:
            return {
                "bound": False,
                "status": "not bound to running app",
                "is_loaded": False,
                "mode": "external",
                "backend": "external",
            }
        snap = dict(self._lab_endpoints.snapshot())
        snap["bound"] = True
        return snap

    def limits(self) -> Dict[str, Any]:
        try:
            from nativelab.GlobalConfig.config_global import (
                APP_CONFIG,
                CONFIG_FIELD_META,
                DEFAULT_CTX,
                DEFAULT_N_PRED,
                DEFAULT_THREADS,
                MODEL_ROLES,
            )
        except Exception as e:
            return {
                "error": str(e),
                "defaults": {},
                "roles": [],
                "app_config": {},
                "fields": {},
            }
        return {
            "defaults": {
                "ctx": DEFAULT_CTX(),
                "threads": DEFAULT_THREADS(),
                "n_predict": DEFAULT_N_PRED,
            },
            "roles": list(MODEL_ROLES),
            "app_config": dict(APP_CONFIG),
            "fields": {
                key: {
                    "label": meta.get("label", key),
                    "description": meta.get("desc", ""),
                    "min": meta.get("min"),
                    "max": meta.get("max"),
                    "type": meta.get("type", ""),
                }
                for key, meta in CONFIG_FIELD_META.items()
            },
        }

    def models(self) -> list[Dict[str, Any]]:
        try:
            from nativelab.Model.model_global import get_model_registry
            return [dict(m) for m in get_model_registry().all_models()]
        except Exception:
            return []

    def api_models(self) -> list[Dict[str, Any]]:
        try:
            from nativelab.Model.model_global import api_model_ref, getapi_registry
        except Exception:
            return []
        out = []
        for cfg in getapi_registry().all():
            d = cfg.to_dict()
            d["api_key"] = "***" if d.get("api_key") else ""
            d["ref"] = api_model_ref(cfg.name)
            out.append(d)
        return out

    def pipelines(self) -> list[Dict[str, Any]]:
        try:
            from nativelab.GlobalConfig.config_global import PIPELINES_DIR
            from nativelab.pipelinebuilder.pipefunctions import list_saved_pipelines
        except Exception:
            return []
        rows = []
        for name in list_saved_pipelines():
            path = PIPELINES_DIR / f"{name}.json"
            data = self._read_pipeline_file(path)
            rows.append({
                "name": name,
                "route": f"/pipelines/{name}",
                "path": str(path),
                "exists": path.exists(),
                "blocks": len(data.get("blocks", [])),
                "connections": len(data.get("connections", [])),
                "block_types": sorted({
                    str(b.get("btype", "")) for b in data.get("blocks", [])
                    if b.get("btype")
                }),
            })
        return rows

    def pipeline_definition(self, name: str) -> Dict[str, Any]:
        try:
            from nativelab.GlobalConfig.config_global import PIPELINES_DIR
        except Exception as e:
            return {"error": str(e), "name": name}
        safe = Path(name).stem
        path = PIPELINES_DIR / f"{safe}.json"
        if not path.exists():
            return {"error": "pipeline not found", "name": name}
        return {
            "name": safe,
            "route": f"/pipelines/{safe}",
            "path": str(path),
            "definition": self._read_pipeline_file(path),
        }

    def labs(self) -> list[Dict[str, Any]]:
        rows = []
        try:
            from nativelab.labs.labs_tab import LAB_FEATURES
        except Exception:
            return [
                {
                    "name": "py-to-doc",
                    "slug": "py_to_doc",
                    "route": "/labs/py_to_doc",
                    "icon": "file-text",
                    "class": "nativelab.labs.pytodoc.PyToDocPanel",
                    "actions": self._lab_actions("py_to_doc"),
                },
            ]
        for panel_cls in LAB_FEATURES:
            name = getattr(panel_cls, "LAB_NAME", panel_cls.__name__)
            slug = self._lab_slug(name)
            rows.append({
                "name": name,
                "slug": slug,
                "route": f"/labs/{slug}",
                "icon": getattr(panel_cls, "LAB_ICON", ""),
                "class": f"{panel_cls.__module__}.{panel_cls.__name__}",
                "actions": self._lab_actions(slug),
            })
        return rows

    def lab_definition(self, slug: str) -> Dict[str, Any]:
        normalized = self._lab_slug(slug)
        for lab in self.labs():
            if lab["slug"] == normalized:
                return lab
        return {"error": "lab not found", "slug": slug}

    def discord_bots(self) -> list[Dict[str, Any]]:
        try:
            from nativelab.integrations.discord_connector import command_catalog, load_discord_bots
        except Exception:
            return []
        rows = []
        for bot in load_discord_bots():
            clean = dict(bot)
            clean["token"] = "***" if clean.get("token") else ""
            clean["commands"] = command_catalog(bot)
            rows.append(clean)
        return rows

    def whatsapp_bots(self) -> list[Dict[str, Any]]:
        try:
            from nativelab.integrations.whatsapp_connector import command_catalog, load_whatsapp_bots
        except Exception:
            return []
        rows = []
        for bot in load_whatsapp_bots():
            clean = dict(bot)
            clean["access_token"] = "***" if clean.get("access_token") else ""
            clean["commands"] = command_catalog(bot)
            rows.append(clean)
        return rows

    # Internals
    @staticmethod
    def _read_pipeline_file(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _lab_slug(name: str) -> str:
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _lab_actions(slug: str) -> list[Dict[str, Any]]:
        if slug == "py_to_doc":
            return [
                {
                    "path": "/labs/py_to_doc",
                    "description": "Inspect py-to-doc lab integration metadata.",
                },
                {
                    "name": "generate",
                    "description": "Build documentation from Python files using the active NativeLab endpoint.",
                    "inputs": {
                        "mode": "single | queue | project",
                        "files": "list[str]",
                        "output_dir": "str",
                        "options": {
                            "include_private": "bool",
                            "split_classes": "bool",
                            "split_functions": "bool",
                        },
                    },
                },
            ]
        return []
