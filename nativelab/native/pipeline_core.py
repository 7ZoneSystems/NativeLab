from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

try:
    from . import _native_core as _core
except Exception:
    _core = None


ConnectionEndpoint = Tuple[Any, Any]
ConnectionRecord = Tuple[int, int, str, int, bool, int]
RouteRecord = Tuple[int, int, str]


def native_pipeline_available() -> bool:
    return _core is not None and hasattr(_core, "pipeline_route_edges")


def normalize_ids(
    block_ids: Sequence[Any],
    connections: Sequence[ConnectionEndpoint],
    counter: int,
) -> Dict[str, Any]:
    if _core is not None and hasattr(_core, "pipeline_normalize_ids"):
        try:
            result = _core.pipeline_normalize_ids(list(block_ids), list(connections), int(counter))
            return {
                "ids": list(result.get("ids", [])),
                "connections": list(result.get("connections", [])),
                "counter": int(result.get("counter", counter)),
            }
        except Exception:
            pass
    return _normalize_ids_py(block_ids, connections, counter)


def _normalize_ids_py(
    block_ids: Sequence[Any],
    connections: Sequence[ConnectionEndpoint],
    counter: int,
) -> Dict[str, Any]:
    reserved = set()
    for raw_bid in block_ids:
        try:
            bid = int(raw_bid)
        except (TypeError, ValueError):
            continue
        if bid > 0:
            reserved.add(bid)

    counter = max([int(counter or 0), *reserved], default=int(counter or 0))
    used = set()
    remap: Dict[Any, int] = {}
    out_ids: List[int] = []

    def next_unused() -> int:
        nonlocal counter
        candidate = counter + 1
        while candidate in reserved:
            candidate += 1
        counter = candidate
        return candidate

    for original_bid in block_ids:
        try:
            bid = int(original_bid)
        except (TypeError, ValueError):
            bid = 0
        if bid <= 0 or bid in used:
            bid = next_unused()
            reserved.add(bid)
        out_ids.append(bid)
        used.add(bid)
        if original_bid not in remap:
            remap[original_bid] = bid
        if original_bid != bid and bid not in remap:
            remap[bid] = bid

    out_connections = [
        (remap.get(from_id, from_id), remap.get(to_id, to_id))
        for from_id, to_id in connections
    ]
    if used:
        counter = max(counter, max(used))
    return {"ids": out_ids, "connections": out_connections, "counter": counter}


def would_form_loop(
    connections: Sequence[ConnectionEndpoint],
    from_bid: int,
    to_bid: int,
) -> bool:
    pairs = [(int(a), int(b)) for a, b in connections]
    if _core is not None and hasattr(_core, "pipeline_would_form_loop"):
        try:
            return bool(_core.pipeline_would_form_loop(pairs, int(from_bid), int(to_bid)))
        except Exception:
            pass
    visited = set()
    queue = [int(to_bid)]
    while queue:
        cur = queue.pop()
        if cur == int(from_bid):
            return True
        if cur in visited:
            continue
        visited.add(cur)
        for c_from, c_to in pairs:
            if c_from == cur:
                queue.append(c_to)
    return False


def apply_transform(context: str, metadata: MutableMapping[str, Any]) -> str:
    if _core is not None and hasattr(_core, "pipeline_apply_transform"):
        try:
            return str(_core.pipeline_apply_transform(str(context), dict(metadata or {})))
        except Exception:
            pass
    ttype = (metadata or {}).get("transform_type", "prefix")
    val = (metadata or {}).get("transform_val", "")
    if ttype == "prefix":
        return f"{val}\n{context}"
    if ttype == "suffix":
        return f"{context}\n{val}"
    if ttype == "replace":
        find_s = (metadata or {}).get("transform_find", "")
        repl_s = (metadata or {}).get("transform_repl", "")
        return context.replace(find_s, repl_s) if find_s else context
    if ttype == "upper":
        return context.upper()
    if ttype == "lower":
        return context.lower()
    if ttype == "strip":
        return context.strip()
    if ttype == "truncate":
        n = int(val) if val else 500
        return context[:n]
    return context


def merge_texts(contexts: Iterable[str], metadata: MutableMapping[str, Any]) -> str:
    extras = [str(item) for item in contexts]
    if _core is not None and hasattr(_core, "pipeline_merge_texts"):
        try:
            return str(_core.pipeline_merge_texts(extras, dict(metadata or {})))
        except Exception:
            pass
    mode = (metadata or {}).get("merge_mode", "concat")
    sep = (metadata or {}).get("merge_sep", "\n\n---\n\n")
    if mode == "json":
        return json.dumps(extras, indent=2)
    if mode == "prepend":
        return str(sep).join(reversed(extras))
    return str(sep).join(extras)


def route_edges(
    records: Sequence[ConnectionRecord],
    from_bid: int,
    visit_counts: MutableMapping[str, int],
    mode: str = "all",
    branch_key: str = "",
    port_labels: Optional[MutableMapping[str, Any]] = None,
) -> List[RouteRecord]:
    record_list: List[ConnectionRecord] = [
        (int(idx), int(source), str(port), int(target), bool(is_loop), int(loop_times))
        for idx, source, port, target, is_loop, loop_times in records
    ]
    if _core is not None and hasattr(_core, "pipeline_route_edges"):
        try:
            native_visits = dict(visit_counts)
            result = _core.pipeline_route_edges(
                record_list,
                int(from_bid),
                native_visits,
                str(mode or "all"),
                str(branch_key or ""),
                dict(port_labels or {}),
            )
            visit_counts.clear()
            visit_counts.update({str(k): int(v) for k, v in native_visits.items()})
            return [(int(idx), int(target), str(port)) for idx, target, port in result]
        except Exception:
            pass

    out: List[RouteRecord] = []
    labels = port_labels or {}
    for idx, source, port, target, is_loop, loop_times in record_list:
        if source != int(from_bid):
            continue
        if not _route_take(str(mode or "all"), str(branch_key or ""), port, labels):
            continue
        key = f"{source}->{target}"
        visits = int(visit_counts.get(key, 0))
        limit = int(loop_times) if is_loop else 1
        if limit < 1:
            limit = 1
        if visits < limit:
            visit_counts[key] = visits + 1
            out.append((idx, target, port))
    return out


def _route_take(mode: str, branch_key: str, port: str, port_labels: MutableMapping[str, Any]) -> bool:
    if mode == "all":
        return True
    if mode == "if":
        return (
            port not in ("E", "W")
            or (port == "E" and branch_key == "TRUE")
            or (port == "W" and branch_key == "FALSE")
        )
    if mode in ("switch", "llm_switch"):
        label = str(port_labels.get(port, port))
        if mode == "llm_switch":
            return label.lower() == branch_key.lower() or label == "default" or not label
        return label == branch_key or label == "default" or not label
    if mode == "llm_score":
        return port == branch_key or port == "N" or port not in ("E", "S", "W", "N")
    return True


def validate_records(records: Sequence[Dict[str, Any]], connection_count: int) -> Optional[str]:
    record_list = [_prepare_validation_record(record) for record in records]
    # Run Python validation first (covers all block types including MCP)
    py_result = _validate_records_py(record_list, int(connection_count))
    if py_result is not None:
        return py_result
    # Also run native validation if available (for native-only checks)
    if _core is not None and hasattr(_core, "pipeline_validate_records"):
        try:
            result = _core.pipeline_validate_records(record_list, int(connection_count))
            return None if result is None else str(result)
        except Exception:
            pass
    return None


def _prepare_validation_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    meta = dict(out.get("metadata") or {})
    out["metadata"] = meta
    out.setdefault("has_ref_text", bool(meta.get("ref_text")))
    out.setdefault("has_knowledge_text", bool(meta.get("knowledge_text")))
    out.setdefault("has_pdf_path", bool(meta.get("pdf_path")))
    out.setdefault("has_condition", bool(meta.get("condition")))
    out.setdefault("has_switch_expr", bool(meta.get("switch_expr")))
    out.setdefault("has_filter_cond", bool(meta.get("filter_cond")))
    out.setdefault("has_transform_type", bool(meta.get("transform_type")))
    out.setdefault("has_custom_code", bool(str(meta.get("custom_code", "")).strip()))
    out.setdefault("has_llm_instruction", bool(str(meta.get("llm_instruction", "")).strip()))
    out.setdefault("has_mcp_url", bool(str(meta.get("mcp_url", "")).strip()))
    out.setdefault("has_mcp_tool", bool(str(meta.get("mcp_tool_name", "")).strip()))
    out.setdefault("has_ws_categories", bool(meta.get("ws_categories")))
    return out


def _validate_records_py(records: Sequence[Dict[str, Any]], connection_count: int) -> Optional[str]:
    if not records:
        return "Canvas is empty - add blocks first."
    if not any(record.get("btype") == "input" for record in records):
        return "Pipeline needs at least one INPUT block."
    if not any(record.get("btype") == "output" for record in records):
        return "Pipeline needs at least one OUTPUT block."
    if not connection_count:
        return "No connections drawn. Connect the blocks with arrows."

    llm_types = {"llm_if", "llm_switch", "llm_filter", "llm_transform", "llm_score"}
    for record in records:
        btype = record.get("btype")
        label = str(record.get("label", ""))
        meta = record.get("metadata") or {}
        if btype == "reference" and not meta.get("ref_text"):
            return f"Reference block '{label}' has no text.\nRight-click it \u2192 Configure block\u2026"
        if btype == "knowledge" and not meta.get("knowledge_text"):
            return f"Knowledge block '{label}' has no text.\nRight-click it \u2192 Configure block\u2026"
        if btype == "pdf_summary" and not meta.get("pdf_path"):
            return f"PDF block '{label}' has no PDF selected.\nRight-click it \u2192 Configure block\u2026"
        if btype == "if_else" and not meta.get("condition"):
            return f"IF/ELSE block '{label}' has no condition set.\nRight-click it \u2192 Configure block\u2026"
        if btype == "switch" and not meta.get("switch_expr"):
            return f"SWITCH block '{label}' has no expression set.\nRight-click it \u2192 Configure block\u2026"
        if btype == "filter" and not meta.get("filter_cond"):
            return f"FILTER block '{label}' has no condition set.\nRight-click it \u2192 Configure block\u2026"
        if btype == "transform" and not meta.get("transform_type"):
            return f"TRANSFORM block '{label}' has no transform type set.\nRight-click it \u2192 Configure block\u2026"
        if btype == "custom_code" and not meta.get("custom_code", "").strip():
            return f"Custom Code block '{label}' has no code.\nRight-click it \u2192 Configure block\u2026"
        if btype in llm_types:
            if not meta.get("llm_instruction", "").strip():
                return f"LLM logic block '{label}' has no instruction.\nRight-click it \u2192 Configure block\u2026"
            if not record.get("llm_model_valid"):
                return (
                    f"LLM logic block '{label}' has no valid model attached.\n"
                    f"Right-click it \u2192 Configure block\u2026 and select a model."
                )
        if btype == "model" and not record.get("model_valid"):
            return (
                f"Model block '{label}' has no valid model attached.\n"
                f"Double-click a model in the sidebar to add it."
            )
        if btype == "mcp_server":
            if not meta.get("mcp_url", "").strip():
                return f"MCP Server block '{label}' has no server URL.\nRight-click it → Configure block…"
            if not meta.get("mcp_tool_name", "").strip():
                return f"MCP Server block '{label}' has no tool selected.\nRight-click it → Test Connection, then select a tool."
        if btype == "web_search":
            if not meta.get("ws_categories"):
                return f"Web Search block '{label}' has no categories selected.\nRight-click it → Configure block…"
    return None
