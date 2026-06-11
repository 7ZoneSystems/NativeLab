from __future__ import annotations

from typing import Any, Dict, List, Optional

from nativelab.Model.model_global import is_api_model_ref, is_model_ref_valid
from nativelab.native.pipeline_core import validate_records

from .blck_typ import PipelineBlockType


_LLM_BTYPES = {
    PipelineBlockType.LLM_IF,
    PipelineBlockType.LLM_SWITCH,
    PipelineBlockType.LLM_FILTER,
    PipelineBlockType.LLM_TRANSFORM,
    PipelineBlockType.LLM_SCORE,
}


def _valid_model_ref(model_ref: str) -> bool:
    return bool(model_ref) and (
        is_api_model_ref(model_ref) or is_model_ref_valid(model_ref)
    )


def pipeline_validation_records(blocks: list) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for block in blocks:
        metadata = dict(getattr(block, "metadata", {}) or {})
        model_path = str(getattr(block, "model_path", "") or "")
        llm_model_path = model_path or str(metadata.get("llm_model_path", "") or "")
        records.append({
            "bid": getattr(block, "bid", 0),
            "btype": getattr(block, "btype", ""),
            "label": getattr(block, "label", ""),
            "metadata": metadata,
            "has_ref_text": bool(metadata.get("ref_text")),
            "has_knowledge_text": bool(metadata.get("knowledge_text")),
            "has_pdf_path": bool(metadata.get("pdf_path")),
            "has_condition": bool(metadata.get("condition")),
            "has_switch_expr": bool(metadata.get("switch_expr")),
            "has_filter_cond": bool(metadata.get("filter_cond")),
            "has_transform_type": bool(metadata.get("transform_type")),
            "has_custom_code": bool(str(metadata.get("custom_code", "")).strip()),
            "has_llm_instruction": bool(str(metadata.get("llm_instruction", "")).strip()),
            "model_path": model_path,
            "model_valid": (
                getattr(block, "btype", "") != PipelineBlockType.MODEL
                or _valid_model_ref(model_path)
            ),
            "llm_model_path": llm_model_path,
            "llm_model_valid": (
                getattr(block, "btype", "") not in _LLM_BTYPES
                or _valid_model_ref(llm_model_path)
            ),
        })
    return records


def validate_pipeline(blocks: list, connections: list) -> Optional[str]:
    return validate_records(pipeline_validation_records(blocks), len(connections))
