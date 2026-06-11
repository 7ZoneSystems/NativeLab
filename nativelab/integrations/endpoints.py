from __future__ import annotations

import json
import time
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
                "path": "/v1/models",
                "method": "GET",
                "description": "OpenAI-compatible model list for saved pipeline models.",
            },
            {
                "path": "/v1/chat/completions",
                "method": "POST",
                "description": "OpenAI-compatible chat completion route. Use model='pipeline:<name>' to execute a saved pipeline.",
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
            {
                "path": "/skills",
                "method": "GET",
                "description": "NativeLab skill library with active skill descriptions and instructions.",
            },
        ]

    def catalog(self) -> Dict[str, Any]:
        return {
            "name": "NativeLab Integrations",
            "version": "v0.3.7",
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
            "pipeline_models": self.pipeline_models(),
            "labs": self.labs(),
            "integrations": {
                "discord_bots": self.discord_bots(),
                "whatsapp_bots": self.whatsapp_bots(),
            },
            "skills": self.skills(),
        }

    def handle(self, path: str) -> Dict[str, Any]:
        clean = "/" + (path or "").strip("/")
        if clean in {"/", "/snapshot"}:
            return self.catalog()
        if clean == "/runtime":
            return self.runtime()
        if clean == "/models":
            return {"models": self.models()}
        if clean == "/v1/models":
            return self.openai_models()
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
        if clean == "/skills":
            return {"skills": self.skills()}
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

    def openai_chat_completion(self, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        payload = dict(payload or {})
        if payload.get("stream"):
            return 400, self._openai_error(
                "Streaming pipeline model responses are not supported yet. Send stream=false.",
                code="streaming_not_supported",
            )
        model_id = str(payload.get("model", "") or "")
        pipeline_name = self.pipeline_name_from_model_id(model_id)
        if not pipeline_name:
            return 404, self._openai_error(
                f"Unknown pipeline model: {model_id or '(missing)'}",
                code="model_not_found",
            )
        prompt = self._prompt_from_openai_payload(payload)
        try:
            result = self.run_pipeline_model(pipeline_name, prompt)
        except FileNotFoundError:
            return 404, self._openai_error(
                f"Pipeline not found: {pipeline_name}",
                code="pipeline_not_found",
            )
        except Exception as exc:
            return 500, self._openai_error(str(exc), code="pipeline_execution_error")
        text = str(result.get("text", ""))
        return 200, {
            "id": f"chatcmpl-nativelab-pipeline-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.pipeline_model_id(pipeline_name),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": self._rough_token_count(prompt),
                "completion_tokens": self._rough_token_count(text),
                "total_tokens": self._rough_token_count(prompt) + self._rough_token_count(text),
            },
            "nativelab": {
                "kind": "pipeline",
                "pipeline": pipeline_name,
                "sender": result.get("sender", ""),
                "logs": result.get("logs", []),
                "steps": result.get("steps", []),
                "intermediates": result.get("intermediates", []),
            },
        }

    def run_pipeline_model(self, name: str, input_text: str) -> Dict[str, Any]:
        from nativelab.pipelinebuilder.executionWorker import run_pipeline_sync
        from nativelab.pipelinebuilder.pipefunctions import load_pipeline, safe_pipeline_name

        try:
            safe = safe_pipeline_name(name)
        except ValueError:
            raise FileNotFoundError("empty pipeline name")
        blocks, conns = load_pipeline(safe)
        return run_pipeline_sync(
            blocks,
            conns,
            str(input_text or ""),
            self._active_engine(),
            log_cb=lambda msg: self._endpoint_log("INFO", f"pipeline:{safe}: {msg}"),
        )

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
            from nativelab.pipelinebuilder.pipefunctions import list_saved_pipelines, pipeline_path
        except Exception:
            return []
        rows = []
        for name in list_saved_pipelines():
            path = pipeline_path(name)
            data = self._read_pipeline_file(path)
            rows.append({
                "name": name,
                "route": f"/pipelines/{name}",
                "model_id": self.pipeline_model_id(name),
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

    def pipeline_models(self) -> list[Dict[str, Any]]:
        rows = []
        for pipeline in self.pipelines():
            name = str(pipeline.get("name", "") or "")
            if not name:
                continue
            rows.append({
                "id": self.pipeline_model_id(name),
                "name": name,
                "object": "model",
                "created": 0,
                "owned_by": "nativelab-pipeline",
                "route": "/v1/chat/completions",
                "pipeline_route": pipeline.get("route", f"/pipelines/{name}"),
                "blocks": pipeline.get("blocks", 0),
                "connections": pipeline.get("connections", 0),
                "block_types": pipeline.get("block_types", []),
            })
        return rows

    def openai_models(self) -> Dict[str, Any]:
        return {"object": "list", "data": self.pipeline_models()}

    @staticmethod
    def pipeline_model_id(name: str) -> str:
        from nativelab.pipelinebuilder.pipefunctions import safe_pipeline_name
        return f"pipeline:{safe_pipeline_name(name)}"

    def pipeline_name_from_model_id(self, model_id: str) -> str:
        model_id = str(model_id or "").strip()
        if not model_id:
            return ""
        for prefix in ("pipeline:", "pipeline/"):
            if model_id.startswith(prefix):
                try:
                    from nativelab.pipelinebuilder.pipefunctions import safe_pipeline_name
                    return safe_pipeline_name(model_id[len(prefix):])
                except ValueError:
                    return ""
        saved = {str(row.get("name", "")) for row in self.pipelines()}
        if model_id in saved:
            return model_id
        return ""

    def pipeline_definition(self, name: str) -> Dict[str, Any]:
        try:
            from nativelab.pipelinebuilder.pipefunctions import pipeline_path, safe_pipeline_name
        except Exception as e:
            return {"error": str(e), "name": name}
        try:
            safe = safe_pipeline_name(name)
            path = pipeline_path(safe)
        except ValueError:
            return {"error": "invalid pipeline name", "name": name}
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

    def skills(self) -> list[Dict[str, Any]]:
        try:
            from nativelab.skill import ensure_builtin_edit_skill, load_skills
        except Exception:
            return []
        ensure_builtin_edit_skill()
        return load_skills()

    # Internals
    def _active_engine(self):
        if self._lab_endpoints is None:
            return None
        try:
            return self._lab_endpoints.active_engine()
        except Exception:
            return None

    def _endpoint_log(self, level: str, msg: str) -> None:
        if self._lab_endpoints is not None and hasattr(self._lab_endpoints, "_log"):
            try:
                self._lab_endpoints._log(level, msg)
                return
            except Exception:
                pass

    @staticmethod
    def _read_pipeline_file(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _openai_error(message: str, *, code: str = "error") -> Dict[str, Any]:
        return {
            "error": {
                "message": str(message),
                "type": "nativelab_pipeline_error",
                "code": str(code),
            }
        }

    @classmethod
    def _prompt_from_openai_payload(cls, payload: Dict[str, Any]) -> str:
        if payload.get("prompt") is not None:
            return cls._content_to_text(payload.get("prompt"))
        messages = payload.get("messages") or []
        if not isinstance(messages, list) or not messages:
            return ""
        if len(messages) == 1:
            return cls._content_to_text(messages[0].get("content", ""))
        lines = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user") or "user")
            content = cls._content_to_text(msg.get("content", ""))
            if content:
                lines.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(lines)

    @classmethod
    def _content_to_text(cls, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        parts.append(str(item.get("text", "")))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            if "text" in content:
                return str(content.get("text", ""))
            return json.dumps(content, ensure_ascii=False)
        return "" if content is None else str(content)

    @staticmethod
    def _rough_token_count(text: str) -> int:
        return len(str(text or "").split())

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
