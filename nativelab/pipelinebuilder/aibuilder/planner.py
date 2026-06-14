from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from nativelab.GlobalConfig.config_global import DEFAULT_CTX
from nativelab.Model.model_global import model_ref_display_name
from nativelab.core.context_meter import estimate_tokens, messages_text

from ..blck_typ import PipelineBlockType, PipelineConnection
from ..pipefunctions import save_pipeline
from ..pipblck import PipelineBlock
from ..validation import validate_pipeline

try:
    from nativelab.native import _native_core as _native
except Exception:  # pragma: no cover - optional native extension
    _native = None


AI_BUILDER_N_PREDICT = 1800
AI_BUILDER_RETRY_N_PREDICT = 2200
_MAX_TEXT_FIELD = 16000
_PORTS = {"N", "S", "E", "W"}
_MODEL_BACKED_TYPES = {
    PipelineBlockType.MODEL,
    PipelineBlockType.LLM_IF,
    PipelineBlockType.LLM_SWITCH,
    PipelineBlockType.LLM_FILTER,
    PipelineBlockType.LLM_TRANSFORM,
    PipelineBlockType.LLM_SCORE,
}
_ALLOWED_TYPES = {
    PipelineBlockType.INPUT,
    PipelineBlockType.OUTPUT,
    PipelineBlockType.MODEL,
    PipelineBlockType.INTERMEDIATE,
    PipelineBlockType.REFERENCE,
    PipelineBlockType.KNOWLEDGE,
    PipelineBlockType.PDF_SUMMARY,
    PipelineBlockType.IF_ELSE,
    PipelineBlockType.SWITCH,
    PipelineBlockType.FILTER,
    PipelineBlockType.TRANSFORM,
    PipelineBlockType.MERGE,
    PipelineBlockType.SPLIT,
    PipelineBlockType.CUSTOM_CODE,
    PipelineBlockType.LLM_IF,
    PipelineBlockType.LLM_SWITCH,
    PipelineBlockType.LLM_FILTER,
    PipelineBlockType.LLM_TRANSFORM,
    PipelineBlockType.LLM_SCORE,
    PipelineBlockType.MCP_SERVER,
}
_DEFAULT_LABELS = {
    PipelineBlockType.INPUT: "Input",
    PipelineBlockType.OUTPUT: "Output",
    PipelineBlockType.MODEL: "Your model",
    PipelineBlockType.INTERMEDIATE: "Intermediate",
    PipelineBlockType.REFERENCE: "Reference",
    PipelineBlockType.KNOWLEDGE: "Knowledge",
    PipelineBlockType.PDF_SUMMARY: "PDF Summary",
    PipelineBlockType.IF_ELSE: "IF / ELSE",
    PipelineBlockType.SWITCH: "Switch",
    PipelineBlockType.FILTER: "Filter",
    PipelineBlockType.TRANSFORM: "Transform",
    PipelineBlockType.MERGE: "Merge",
    PipelineBlockType.SPLIT: "Split",
    PipelineBlockType.CUSTOM_CODE: "Custom Code",
    PipelineBlockType.LLM_IF: "LLM IF / ELSE",
    PipelineBlockType.LLM_SWITCH: "LLM Switch",
    PipelineBlockType.LLM_FILTER: "LLM Filter",
    PipelineBlockType.LLM_TRANSFORM: "LLM Transform",
    PipelineBlockType.LLM_SCORE: "LLM Score",
    PipelineBlockType.MCP_SERVER: "MCP Server",
}


PIPELINE_AI_GUIDE = """You build NativeLab pipeline JSON.
Return exactly one JSON object. Do not return markdown, code fences, prose, or comments.

Schema:
{
  "version": 2,
  "title": "short name",
  "description": "short purpose",
  "blocks": [
    {
      "bid": 1,
      "btype": "input|output|model|intermediate|reference|knowledge|pdf_summary|if_else|switch|filter|transform|merge|split|custom_code|llm_if|llm_switch|llm_filter|llm_transform|llm_score|mcp_server",
      "x": 80, "y": 120, "w": 148, "h": 76,
      "model_path": "",
      "role": "general",
      "label": "Readable label",
      "metadata": {}
    }
  ],
  "connections": [
    {"from_block_id": 1, "from_port": "E", "to_block_id": 2, "to_port": "W", "is_loop": false, "loop_times": 1}
  ]
}

Rules:
- Include exactly one input block and at least one output block.
- Use unique positive integer bid values; every connection endpoint must refer to an existing block.
- Prefer left-to-right layout on a 20 px grid. Keep related branches vertically separated.
- Use ports E->W for normal flow, E/W for IF true/false arms, and N/S only when vertical layout is clearer.
- Model-backed blocks must leave model_path empty unless the user explicitly names a model; NativeLab will attach the active model.
- Do not connect model blocks directly to other model blocks. Put an intermediate, transform, filter, or merge block between them.
- Use reference only when static reference text is known. Put it in metadata.ref_text.
- Use knowledge only when reusable knowledge text is known. Put it in metadata.knowledge_text.
- Use pdf_summary only when the user gives a local PDF path. Put it in metadata.pdf_path.
- transform metadata: transform_type prefix|suffix|replace|uppercase|lowercase|strip|truncate|regex, transform_val, transform_find, transform_repl.
- if_else metadata: condition as a safe Python expression using text.
- switch metadata: switch_expr and port_labels mapping ports to values.
- filter metadata: filter_cond as a safe Python expression using text.
- merge metadata: merge_mode concat|prepend|json and merge_sep.
- custom_code metadata: custom_code only if essential; it must be deterministic, no imports, no filesystem, no network, no subprocess.
- llm_* metadata: llm_instruction, llm_max_tokens, llm_temp, llm_passthrough_on_err.
- mcp_server metadata: mcp_transport ("sse" or "stdio"), mcp_url (URL for SSE, shell command for stdio), mcp_name (display name), mcp_tool_name (exact tool name to call), mcp_arg_name (input argument name, default "input"). Use mcp_server when the user needs to call external tools, access files, databases, web APIs, or any MCP-compatible server. For npx packages use stdio transport with mcp_url like "npx -y @modelcontextprotocol/server-filesystem /path".
- Keep the graph simple enough to run, but include all blocks needed to satisfy the requested workflow.
"""


@dataclass(frozen=True)
class PipelinePromptBudget:
    input_tokens: int
    reserved_tokens: int
    projected_tokens: int
    limit_tokens: int
    overflow: bool
    messages: List[Dict[str, str]]


@dataclass(frozen=True)
class GeneratedPipeline:
    name: str
    raw_response: str
    data: Dict[str, Any]
    blocks: List[PipelineBlock]
    connections: List[PipelineConnection]


class PipelineJsonError(ValueError):
    def __init__(self, message: str, raw_response: str = ""):
        super().__init__(message)
        self.raw_response = str(raw_response or "")


def sanitize_pipeline_name(name: str) -> str:
    candidate = str(name or "").strip().replace("/", "-").replace("\\", "-")
    candidate = candidate.replace("..", ".")
    candidate = re.sub(r"[^A-Za-z0-9._ -]+", "-", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .-_")
    return (candidate[:80] or "ai-pipeline").strip() or "ai-pipeline"


def _native_estimate_tokens(text: str) -> int:
    value = str(text or "")
    if _native is not None:
        try:
            return int(_native.aibuilder_estimate_tokens(value))
        except Exception:
            pass
    return estimate_tokens(value)


def _engine_limit(engine: Any) -> int:
    try:
        value = int(getattr(engine, "ctx_value", 0) or 0)
    except Exception:
        value = 0
    return value if value > 0 else int(DEFAULT_CTX())


def build_ai_builder_messages(
    user_request: str,
    pipeline_name: str,
    *,
    active_model_label: str = "",
) -> List[Dict[str, str]]:
    safe_name = sanitize_pipeline_name(pipeline_name)
    request = str(user_request or "").strip()
    active_line = active_model_label.strip() or "NativeLab active model"
    return [
        {
            "role": "system",
            "content": PIPELINE_AI_GUIDE,
        },
        {
            "role": "user",
            "content": (
                f"Pipeline output JSON file name: {safe_name}\n"
                f"Active model available for empty model_path placeholders: {active_line}\n\n"
                "Build a NativeLab pipeline for this request:\n"
                f"{request}\n\n"
                "Return only the JSON object."
            ),
        },
    ]


def build_ai_builder_retry_messages(
    user_request: str,
    pipeline_name: str,
    *,
    active_model_label: str = "",
    previous_response: str = "",
) -> List[Dict[str, str]]:
    safe_name = sanitize_pipeline_name(pipeline_name)
    request = str(user_request or "").strip()
    active_line = active_model_label.strip() or "NativeLab active model"
    previous = str(previous_response or "").strip()
    if len(previous) > 1800:
        previous = previous[:1800] + "\n...[truncated]"
    return [
        {
            "role": "system",
            "content": (
                "You are generating NativeLab pipeline JSON. "
                "Your entire response must be exactly one valid JSON object. "
                "The first character must be { and the last character must be }. "
                "No markdown, no explanation, no code fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"File name: {safe_name}\n"
                f"Active model for empty model_path placeholders: {active_line}\n\n"
                "Use this exact schema:\n"
                '{"version":2,"title":"name","description":"purpose",'
                '"blocks":[{"bid":1,"btype":"input","x":80,"y":120,"w":148,"h":76,'
                '"model_path":"","role":"general","label":"Input","metadata":{}}],'
                '"connections":[{"from_block_id":1,"from_port":"E","to_block_id":2,'
                '"to_port":"W","is_loop":false,"loop_times":1}]}\n\n'
                "Allowed btype values: input, output, model, intermediate, reference, knowledge, "
                "pdf_summary, if_else, switch, filter, transform, merge, split, custom_code, "
                "llm_if, llm_switch, llm_filter, llm_transform, llm_score, mcp_server.\n"
                "Required: one input, at least one output, unique bid values, valid connection endpoints. "
                "Leave model_path empty for model and llm_* blocks unless the user explicitly named a model.\n\n"
                f"Pipeline request:\n{request}\n\n"
                f"Previous invalid response to avoid repeating:\n{previous}\n\n"
                "Return only the JSON object now."
            ),
        },
    ]


def estimate_ai_builder_budget(
    engine: Any,
    pipeline_name: str,
    user_request: str,
    *,
    active_model_label: str = "",
    n_predict: int = AI_BUILDER_N_PREDICT,
) -> PipelinePromptBudget:
    messages = build_ai_builder_messages(
        user_request,
        pipeline_name,
        active_model_label=active_model_label,
    )
    input_tokens = _native_estimate_tokens(messages_text(messages))
    reserved = max(0, int(n_predict or 0))
    limit = _engine_limit(engine)
    projected = input_tokens + reserved
    return PipelinePromptBudget(
        input_tokens=input_tokens,
        reserved_tokens=reserved,
        projected_tokens=projected,
        limit_tokens=limit,
        overflow=projected > limit,
        messages=messages,
    )


def estimate_ai_builder_retry_budget(
    engine: Any,
    pipeline_name: str,
    user_request: str,
    *,
    active_model_label: str = "",
    previous_response: str = "",
    n_predict: int = AI_BUILDER_RETRY_N_PREDICT,
) -> PipelinePromptBudget:
    messages = build_ai_builder_retry_messages(
        user_request,
        pipeline_name,
        active_model_label=active_model_label,
        previous_response=previous_response,
    )
    input_tokens = _native_estimate_tokens(messages_text(messages))
    reserved = max(0, int(n_predict or 0))
    limit = _engine_limit(engine)
    projected = input_tokens + reserved
    return PipelinePromptBudget(
        input_tokens=input_tokens,
        reserved_tokens=reserved,
        projected_tokens=projected,
        limit_tokens=limit,
        overflow=projected > limit,
        messages=messages,
    )


def _json_span(text: str) -> Optional[Tuple[int, int]]:
    value = str(text or "")
    if _native is not None:
        try:
            span = _native.aibuilder_json_span(value)
            if span is not None:
                return int(span[0]), int(span[1])
        except Exception:
            pass

    start = -1
    depth = 0
    in_string = False
    escaped = False
    for idx, ch in enumerate(value):
        if start < 0:
            if ch == "{":
                start = idx
                depth = 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, idx + 1
    return None


def extract_json_object(text: str) -> Dict[str, Any]:
    value = str(text or "")
    stripped = value.strip()
    if stripped:
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, dict):
                return decoded
            if isinstance(decoded, str):
                return extract_json_object(decoded)
        except Exception:
            pass

    span = _json_span(value)
    if span is None:
        raise PipelineJsonError("The model response did not contain a JSON object.", value)
    raw = value[span[0]:span[1]]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PipelineJsonError(f"The model returned malformed pipeline JSON: {exc}", value) from exc
    if not isinstance(data, dict):
        raise PipelineJsonError("The model response JSON must be an object.", value)
    return data


def _coerce_int(value: Any, default: int, *, lo: int = -100000, hi: int = 100000) -> int:
    try:
        out = int(value)
    except Exception:
        out = default
    return min(hi, max(lo, out))


def _clean_text(value: Any, default: str = "", *, limit: int = _MAX_TEXT_FIELD) -> str:
    text = str(value if value is not None else default)
    if len(text) > limit:
        return text[:limit]
    return text


def _json_safe_metadata(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    try:
        safe = json.loads(json.dumps(value, default=str))
    except Exception:
        safe = {str(k): str(v) for k, v in value.items()}
    return safe if isinstance(safe, dict) else {}


def _next_free_id(used: set[int], hint: int) -> int:
    candidate = max(1, int(hint or 1))
    while candidate in used:
        candidate += 1
    used.add(candidate)
    return candidate


def _default_label(btype: str) -> str:
    return _DEFAULT_LABELS.get(btype, btype.replace("_", " ").title())


def _normalize_metadata(btype: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    md = dict(metadata or {})
    if btype == PipelineBlockType.REFERENCE:
        md["ref_text"] = _clean_text(md.get("ref_text"), "Reference text placeholder.")
        md.setdefault("ref_name", "AI reference")
    elif btype == PipelineBlockType.KNOWLEDGE:
        md["knowledge_text"] = _clean_text(md.get("knowledge_text"), "Knowledge text placeholder.")
    elif btype == PipelineBlockType.IF_ELSE:
        md["condition"] = _clean_text(md.get("condition"), "True", limit=1000)
    elif btype == PipelineBlockType.SWITCH:
        md["switch_expr"] = _clean_text(md.get("switch_expr"), "''", limit=1000)
        if not isinstance(md.get("port_labels"), dict):
            md["port_labels"] = {"E": "default"}
    elif btype == PipelineBlockType.FILTER:
        md["filter_cond"] = _clean_text(md.get("filter_cond"), "True", limit=1000)
    elif btype == PipelineBlockType.TRANSFORM:
        md["transform_type"] = _clean_text(md.get("transform_type"), "strip", limit=40)
        md.setdefault("transform_val", "")
    elif btype == PipelineBlockType.MERGE:
        md["merge_mode"] = _clean_text(md.get("merge_mode"), "concat", limit=40)
        md.setdefault("merge_sep", "\n\n")
    elif btype == PipelineBlockType.CUSTOM_CODE:
        code = _clean_text(md.get("custom_code"), "result = text")
        if "import " in code or "__import__" in code or "open(" in code or "subprocess" in code:
            code = "result = text\nlog('AI generated custom code was replaced by a safe pass-through.')"
        try:
            compile(code, "<ai_pipeline_custom_code>", "exec")
        except SyntaxError:
            code = "result = text\nlog('AI generated custom code had a syntax error and was replaced.')"
        md["custom_code"] = code
    elif btype in _MODEL_BACKED_TYPES and btype != PipelineBlockType.MODEL:
        md["llm_instruction"] = _clean_text(
            md.get("llm_instruction"),
            "Use the block label and incoming text to complete this step.",
            limit=3000,
        )
        md["llm_max_tokens"] = _coerce_int(md.get("llm_max_tokens"), 512, lo=1, hi=4096)
        try:
            md["llm_temp"] = float(md.get("llm_temp", 0.3))
        except Exception:
            md["llm_temp"] = 0.3
        md["llm_passthrough_on_err"] = bool(md.get("llm_passthrough_on_err", True))
    elif btype == PipelineBlockType.MCP_SERVER:
        md["mcp_transport"] = _clean_text(md.get("mcp_transport"), "sse", limit=10)
        if md["mcp_transport"] not in ("sse", "stdio"):
            md["mcp_transport"] = "sse"
        md["mcp_url"] = _clean_text(md.get("mcp_url"), "", limit=2000)
        md["mcp_name"] = _clean_text(md.get("mcp_name"), "", limit=80)
        md["mcp_tool_name"] = _clean_text(md.get("mcp_tool_name"), "", limit=200)
        md["mcp_arg_name"] = _clean_text(md.get("mcp_arg_name"), "", limit=80)
        md.setdefault("mcp_connected", False)
        md.setdefault("mcp_tools", [])
    return md


def normalize_pipeline_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Pipeline data must be a JSON object.")

    raw_blocks = data.get("blocks") if isinstance(data.get("blocks"), list) else []
    blocks: List[Dict[str, Any]] = []
    used_ids: set[int] = set()
    old_to_new: Dict[Any, int] = {}

    for idx, raw in enumerate(raw_blocks):
        if not isinstance(raw, dict):
            continue
        btype = str(raw.get("btype") or "").strip()
        if btype not in _ALLOWED_TYPES:
            btype = PipelineBlockType.INTERMEDIATE
        old_id = raw.get("bid")
        bid = _next_free_id(used_ids, _coerce_int(old_id, idx + 1, lo=1, hi=999999))
        old_to_new[old_id] = bid
        old_to_new[str(old_id)] = bid
        label = _clean_text(raw.get("label"), _default_label(btype), limit=80).strip() or _default_label(btype)
        metadata = _normalize_metadata(btype, _json_safe_metadata(raw.get("metadata")))
        blocks.append({
            "bid": bid,
            "btype": btype,
            "x": _coerce_int(raw.get("x"), 80 + idx * 220),
            "y": _coerce_int(raw.get("y"), 120 + (idx % 4) * 120),
            "w": _coerce_int(raw.get("w"), 148, lo=96, hi=360),
            "h": _coerce_int(raw.get("h"), 76, lo=56, hi=220),
            "model_path": _clean_text(raw.get("model_path"), "", limit=2000),
            "role": _clean_text(raw.get("role"), "general", limit=80) or "general",
            "label": label,
            "metadata": metadata,
        })

    if not any(b["btype"] == PipelineBlockType.INPUT for b in blocks):
        bid = _next_free_id(used_ids, 1)
        blocks.insert(0, {
            "bid": bid, "btype": PipelineBlockType.INPUT, "x": 80, "y": 120,
            "w": 148, "h": 76, "model_path": "", "role": "general",
            "label": "Input", "metadata": {},
        })
    if not any(b["btype"] == PipelineBlockType.OUTPUT for b in blocks):
        bid = _next_free_id(used_ids, len(blocks) + 1)
        blocks.append({
            "bid": bid, "btype": PipelineBlockType.OUTPUT,
            "x": 80 + len(blocks) * 220, "y": 120, "w": 148, "h": 76,
            "model_path": "", "role": "general", "label": "Output", "metadata": {},
        })

    valid_ids = {b["bid"] for b in blocks}
    connections: List[Dict[str, Any]] = []
    seen_edges: set[Tuple[int, str, int, str]] = set()
    raw_connections = data.get("connections") if isinstance(data.get("connections"), list) else []
    for raw in raw_connections:
        if not isinstance(raw, dict):
            continue
        from_id = old_to_new.get(raw.get("from_block_id"), raw.get("from_block_id"))
        to_id = old_to_new.get(raw.get("to_block_id"), raw.get("to_block_id"))
        from_id = _coerce_int(from_id, 0, lo=1, hi=999999)
        to_id = _coerce_int(to_id, 0, lo=1, hi=999999)
        if from_id not in valid_ids or to_id not in valid_ids or from_id == to_id:
            continue
        from_port = str(raw.get("from_port") or "E").upper()
        to_port = str(raw.get("to_port") or "W").upper()
        if from_port not in _PORTS:
            from_port = "E"
        if to_port not in _PORTS:
            to_port = "W"
        edge = (from_id, from_port, to_id, to_port)
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        connections.append({
            "from_block_id": from_id,
            "from_port": from_port,
            "to_block_id": to_id,
            "to_port": to_port,
            "is_loop": bool(raw.get("is_loop", False)),
            "loop_times": _coerce_int(raw.get("loop_times"), 1, lo=1, hi=20),
        })

    if not connections and len(blocks) >= 2:
        ordered = sorted(blocks, key=lambda b: (b["x"], b["y"], b["bid"]))
        for left, right in zip(ordered, ordered[1:]):
            connections.append({
                "from_block_id": left["bid"], "from_port": "E",
                "to_block_id": right["bid"], "to_port": "W",
                "is_loop": False, "loop_times": 1,
            })

    return {
        "version": 2,
        "title": _clean_text(data.get("title"), "AI Pipeline", limit=120),
        "description": _clean_text(data.get("description"), "", limit=1000),
        "blocks": blocks,
        "connections": connections,
    }


def apply_active_model(
    data: Dict[str, Any],
    active_model_ref: str = "",
    active_model_role: str = "general",
) -> Dict[str, Any]:
    model_ref = str(active_model_ref or "")
    if not model_ref:
        return data
    label = model_ref_display_name(model_ref)[:18] or "Active model"
    for block in data.get("blocks", []):
        if not isinstance(block, dict):
            continue
        if block.get("btype") in _MODEL_BACKED_TYPES and not block.get("model_path"):
            block["model_path"] = model_ref
            block["role"] = active_model_role or block.get("role") or "general"
            if str(block.get("label") or "") in {"", "Model", "Your model", "LLM", "AI model"}:
                block["label"] = label
    return data


def pipeline_data_to_blocks(data: Dict[str, Any]) -> Tuple[List[PipelineBlock], List[PipelineConnection]]:
    blocks: List[PipelineBlock] = []
    max_bid = 0
    for raw in data.get("blocks", []):
        if not isinstance(raw, dict):
            continue
        block = PipelineBlock(
            str(raw.get("btype") or PipelineBlockType.INTERMEDIATE),
            _coerce_int(raw.get("x"), 100),
            _coerce_int(raw.get("y"), 100),
            _clean_text(raw.get("model_path"), ""),
            _clean_text(raw.get("role"), "general") or "general",
            _clean_text(raw.get("label"), ""),
        )
        block.bid = _coerce_int(raw.get("bid"), block.bid, lo=1, hi=999999)
        block.w = _coerce_int(raw.get("w"), getattr(block, "w", 148), lo=96, hi=360)
        block.h = _coerce_int(raw.get("h"), getattr(block, "h", 76), lo=56, hi=220)
        block.metadata = _json_safe_metadata(raw.get("metadata"))
        blocks.append(block)
        max_bid = max(max_bid, block.bid)

    connections: List[PipelineConnection] = []
    valid_ids = {b.bid for b in blocks}
    for raw in data.get("connections", []):
        if not isinstance(raw, dict):
            continue
        from_id = _coerce_int(raw.get("from_block_id"), 0, lo=1, hi=999999)
        to_id = _coerce_int(raw.get("to_block_id"), 0, lo=1, hi=999999)
        if from_id not in valid_ids or to_id not in valid_ids:
            continue
        from_port = str(raw.get("from_port") or "E").upper()
        to_port = str(raw.get("to_port") or "W").upper()
        connections.append(PipelineConnection(
            from_block_id=from_id,
            from_port=from_port if from_port in _PORTS else "E",
            to_block_id=to_id,
            to_port=to_port if to_port in _PORTS else "W",
            is_loop=bool(raw.get("is_loop", False)),
            loop_times=_coerce_int(raw.get("loop_times"), 1, lo=1, hi=20),
        ))

    PipelineBlock._id_counter = max(PipelineBlock._id_counter, max_bid)
    return blocks, connections


def save_generated_pipeline(
    name: str,
    raw_response: str,
    *,
    active_model_ref: str = "",
    active_model_role: str = "general",
) -> GeneratedPipeline:
    safe_name = sanitize_pipeline_name(name)
    data = normalize_pipeline_data(extract_json_object(raw_response))
    data = apply_active_model(data, active_model_ref, active_model_role)
    blocks, connections = pipeline_data_to_blocks(data)
    validation_error = validate_pipeline(blocks, connections)
    if validation_error:
        raise ValueError(f"Generated pipeline did not pass validation:\n\n{validation_error}")
    save_pipeline(safe_name, blocks, connections)
    return GeneratedPipeline(
        name=safe_name,
        raw_response=str(raw_response or ""),
        data=data,
        blocks=blocks,
        connections=connections,
    )
