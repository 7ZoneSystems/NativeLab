from __future__ import annotations

import re
from typing import Any

from nativelab.Model.model_global import get_model_registry, model_ref_display_name


def _model_id(name: str, used: set[str]) -> str:
    base = re.sub(r"\s+", "-", str(name or "model").strip())
    base = re.sub(r"[^A-Za-z0-9._:-]+", "-", base).strip("-") or "model"
    model_id = base
    i = 2
    while model_id in used:
        model_id = f"{base}-{i}"
        i += 1
    used.add(model_id)
    return model_id


def model_capabilities(row: dict[str, Any]) -> dict[str, Any]:
    backend = str(row.get("backend") or "llama_cpp")
    vision = bool(row.get("vision"))
    mmproj = str(row.get("mmproj") or "")
    image_ready = bool(
        vision
        and (
            backend in ("ollama", "hf_transformers")
            or (backend == "llama_cpp" and mmproj)
        )
    )
    return {
        "chat": True,
        "completion": True,
        "openai_chat_completions": True,
        "anthropic_messages": True,
        "text_input": True,
        "text_output": True,
        "image_input": image_ready,
        "vision_detected": vision,
        "image_status": "ready" if image_ready else ("needs_mmproj_or_backend" if vision else "text_only"),
        "backend": backend,
    }


def model_catalog() -> list[dict[str, Any]]:
    rows = get_model_registry().all_models()
    used: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        ref = str(row.get("path") or "")
        name = str(row.get("name") or model_ref_display_name(ref) or ref)
        model_id = _model_id(name, used)
        caps = model_capabilities(row)
        out.append({
            "id": model_id,
            "object": "model",
            "created": 0,
            "owned_by": "nativelab",
            "name": name,
            "native_ref": ref,
            "backend": row.get("backend", ""),
            "family": row.get("family", ""),
            "quant": row.get("quant", ""),
            "ctx": row.get("ctx", 0),
            "n_predict": row.get("n_predict", 0),
            "size_mb": row.get("size_mb", 0.0),
            "source": row.get("source", ""),
            "role": row.get("role", ""),
            "vision": bool(row.get("vision")),
            "vision_label": row.get("vision_label", ""),
            "mmproj": row.get("mmproj", ""),
            "capabilities": caps,
        })
    return out


def resolve_model_ref(model_id: str) -> str:
    needle = str(model_id or "").strip()
    if not needle:
        return ""
    for row in model_catalog():
        if needle in (
            str(row.get("id") or ""),
            str(row.get("name") or ""),
            str(row.get("native_ref") or ""),
        ):
            return str(row.get("native_ref") or "")
    return ""


def openai_model_list() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": row["id"],
                "object": "model",
                "created": 0,
                "owned_by": "nativelab",
                "native_ref": row["native_ref"],
                "capabilities": row["capabilities"],
                "metadata": {
                    "name": row["name"],
                    "backend": row["backend"],
                    "family": row["family"],
                    "quant": row["quant"],
                    "ctx": row["ctx"],
                    "vision": row["vision"],
                    "vision_label": row["vision_label"],
                    "mmproj": row["mmproj"],
                },
            }
            for row in model_catalog()
        ],
    }


def anthropic_model_list() -> dict[str, Any]:
    rows = model_catalog()
    return {
        "data": [
            {
                "id": row["id"],
                "type": "model",
                "display_name": row["name"],
                "created_at": "1970-01-01T00:00:00Z",
                "native_ref": row["native_ref"],
                "capabilities": row["capabilities"],
            }
            for row in rows
        ],
        "has_more": False,
        "first_id": rows[0]["id"] if rows else None,
        "last_id": rows[-1]["id"] if rows else None,
    }
