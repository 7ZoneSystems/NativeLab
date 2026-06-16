from nativelab.core.engines.llamaengine import LlamaEngine
from nativelab.core.context_meter import context_meter
from nativelab.imports.import_global import QThread,Path, pyqtSignal, List, Dict, time, Optional, json, HAS_PDF
from nativelab.Model.model_global import detect_model_family, get_model_registry, is_api_model_ref, is_model_ref_valid, model_ref_display_name, model_ref_payload
from nativelab.GlobalConfig.config_global import LONG_TIMEOUT_SECONDS
from .blck_typ import PipelineConnection, PipelineBlockType
from .pipblck import PipelineBlock
from .execution_core import (
    collect_merge_inputs,
    knowledge_context,
    merge_contexts,
    output_payload,
    output_sender_label,
    reference_context,
    transform_text,
)
from .graph_ops import build_adjacency, route_connections


def _decode_pipeline_payload(payload: str) -> dict:
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            return {
                "text": str(data.get("text", payload)),
                "sender": str(data.get("sender", "")),
                "raw": payload,
            }
    except Exception:
        pass
    return {"text": str(payload), "sender": "", "raw": str(payload)}


def run_pipeline_sync(
    blocks: List[PipelineBlock],
    connections: List[PipelineConnection],
    input_text: str,
    primary_engine: "LlamaEngine",
    *,
    token_cb=None,
    log_cb=None,
    step_cb=None,
) -> dict:
    """
    Execute a pipeline synchronously using PipelineExecutionWorker itself.

    This is used by CLI and integration HTTP endpoints where starting a separate
    QThread/event loop would add another execution path. All block behavior,
    guardrails, model calls, and error handling remain inside the worker.
    """
    from .validation import validate_pipeline

    validation_error = validate_pipeline(blocks, connections)
    if validation_error:
        raise RuntimeError(validation_error)

    worker = PipelineExecutionWorker(blocks, connections, str(input_text or ""), primary_engine)
    result = {
        "text": "",
        "sender": "",
        "raw": "",
        "logs": [],
        "steps": [],
        "intermediates": [],
    }
    errors: list[str] = []

    def _log(msg: str):
        result["logs"].append(str(msg))
        if log_cb:
            log_cb(str(msg))

    def _step_started(bid: int, label: str):
        item = {"id": int(bid), "label": str(label)}
        result["steps"].append(item)
        if step_cb:
            step_cb(item)

    def _token(_bid: int, tok: str):
        if token_cb:
            token_cb(str(tok))

    def _intermediate(bid: int, label: str, text: str):
        result["intermediates"].append({
            "id": int(bid),
            "label": str(label),
            "text": str(text),
        })

    def _done(payload: str):
        result.update(_decode_pipeline_payload(str(payload)))

    worker.log_msg.connect(_log)
    worker.step_started.connect(_step_started)
    worker.step_token.connect(_token)
    worker.intermediate_live.connect(_intermediate)
    worker.pipeline_done.connect(_done)
    worker.err.connect(lambda msg: errors.append(str(msg)))
    worker.run()
    if errors:
        raise RuntimeError(errors[0])
    return result


class PipelineExecutionWorker(QThread):
    """
    Sequentially executes a validated pipeline graph.
    SERVER MODE ONLY - refuses to run if model cannot be started as a server.
    Retries server startup up to _SERVER_RETRIES times before aborting.
    """
    step_started      = pyqtSignal(int, str)     # (block_id, label)
    step_token        = pyqtSignal(int, str)     # (block_id, token)
    step_done         = pyqtSignal(int, str)     # (block_id, full_text)
    intermediate_live = pyqtSignal(int, str, str)# (block_id, label, text)
    pipeline_done     = pyqtSignal(str)
    err               = pyqtSignal(str)
    log_msg           = pyqtSignal(str)

    _SERVER_RETRIES   = 3
    _SERVER_RETRY_S   = 6

    def __init__(self, blocks: List[PipelineBlock],
                 connections: List[PipelineConnection],
                 input_text: str,
                 primary_engine: "LlamaEngine"):
        super().__init__()
        self.blocks         = {b.bid: b for b in blocks}
        self.connections    = connections
        self.input_text     = input_text
        self.primary_engine = primary_engine
        self._abort         = False
        self._last_model_error = ""

    def abort(self):
        self._abort = True

    # ── adjacency helpers ─────────────────────────────────────────────────────

    def _adjacency(self) -> Dict[int, List[PipelineConnection]]:
        return build_adjacency(self.connections)

    def _enqueue_outgoing(
        self,
        queue: list,
        adj: Dict[int, List[PipelineConnection]],
        bid: int,
        current_text: str,
        visit_counts: Dict[str, int],
        mode: str = "all",
        branch_key: str = "",
        port_labels: Optional[dict] = None,
        score_text: Optional[int] = None,
    ) -> int:
        selected = route_connections(
            adj.get(bid, []),
            bid,
            visit_counts,
            mode=mode,
            branch_key=branch_key,
            port_labels=port_labels or {},
        )
        for conn in selected:
            out_text = str(score_text) if mode == "llm_score" and conn.from_port == "N" else current_text
            queue.append((conn.to_block_id, out_text))
        return len(selected)

    def run(self):
        adj = self._adjacency()
        inp = next((b for b in self.blocks.values()
                    if b.btype == PipelineBlockType.INPUT), None)
        if not inp:
            self.err.emit("No INPUT block found."); return

        current_text = self.input_text
        visit_counts: Dict[str, int] = {}
        queue = [(inp.bid, self.input_text)]

        while queue and not self._abort:
            bid, context = queue.pop(0)
            b = self.blocks.get(bid)
            if not b:
                continue

            if b.btype == PipelineBlockType.INPUT:
                current_text = context

            elif b.btype == PipelineBlockType.INTERMEDIATE:
                self.step_started.emit(bid, b.label)
                inter_prompt   = b.metadata.get("inter_prompt", "").strip()
                inter_position = b.metadata.get("inter_position", "above")

                if inter_prompt:
                    if inter_position == "above":
                        current_text = f"{inter_prompt}\n\n{context}"
                    else:
                        current_text = f"{context}\n\n{inter_prompt}"
                    self.log_msg.emit(
                        f"'{b.label}' injected prompt "
                        f"({'above' if inter_position == 'above' else 'below'} output) "
                        f"- {len(current_text):,} chars total")
                else:
                    current_text = context

                self.intermediate_live.emit(bid, b.label, current_text)
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.OUTPUT:
                sender_label = output_sender_label(self.blocks, self.connections, bid)
                current_text = context
                # Emit step_done with the raw context (no intermediate noise)
                self.step_done.emit(bid, context)
                # pipeline_done carries (final_text, sender_label) encoded together
                # We encode as JSON so the tab can split them cleanly
                self.pipeline_done.emit(output_payload(context, sender_label))
                return

            elif b.btype == PipelineBlockType.MODEL:
                self.step_started.emit(bid, b.label)
                result = self._run_model(b, context)
                if result is None:
                    if not self._abort and not self._last_model_error:
                        self.err.emit(f"Model block '{b.label}' returned no output.")
                    self._last_model_error = ""
                    return
                current_text = result
                self.step_done.emit(bid, result)

            elif b.btype == PipelineBlockType.REFERENCE:
                self.step_started.emit(bid, b.label)
                current_text = self._run_reference(b, context)
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.KNOWLEDGE:
                self.step_started.emit(bid, b.label)
                current_text = self._run_knowledge(b, context)
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.PDF_SUMMARY:
                self.step_started.emit(bid, b.label)
                result = self._run_pdf(b, context)
                if result is None:
                    if not self._abort:
                        self.err.emit(f"PDF block '{b.label}' failed.")
                    return
                current_text = result
                self.step_done.emit(bid, result)

            # ── Logic block execution ─────────────────────────────────────────
            elif b.btype == PipelineBlockType.IF_ELSE:
                self.step_started.emit(bid, b.label)
                cond = b.metadata.get("condition", "True")
                _log_lines: list = []
                _safe_ns = {
                    "text": context,
                    "log": lambda m, _ll=_log_lines: _ll.append(str(m)),
                    "__builtins__": {
                        "len": len, "str": str, "int": int, "float": float,
                        "bool": bool, "list": list, "dict": dict, "any": any,
                        "all": all, "min": min, "max": max, "abs": abs,
                        "True": True, "False": False, "None": None,
                        "isinstance": isinstance,
                    },
                }
                try:
                    _result = eval(compile(cond, "<if_else>", "eval"), _safe_ns)
                    branch_taken = "TRUE" if _result else "FALSE"
                except Exception as _e:
                    self.log_msg.emit(
                        f"IF/ELSE '{b.label}' condition error: {_e} -> defaulting FALSE")
                    branch_taken = "FALSE"
                self.log_msg.emit(
                    f"⑂  IF/ELSE '{b.label}': condition='{cond[:40]}' → {branch_taken}")
                current_text = context
                self.step_done.emit(bid, current_text)
                # Only enqueue the arm matching branch_taken
                self._enqueue_outgoing(
                    queue, adj, bid, current_text, visit_counts,
                    mode="if", branch_key=branch_taken)
                continue  

            elif b.btype == PipelineBlockType.SWITCH:
                self.step_started.emit(bid, b.label)
                expr = b.metadata.get("switch_expr", "''")
                _safe_ns = {
                    "text": context,
                    "__builtins__": {
                        "len": len, "str": str, "int": int, "float": float,
                        "bool": bool, "True": True, "False": False, "None": None,
                    },
                }
                try:
                    switch_key = str(eval(compile(expr, "<switch>", "eval"), _safe_ns))
                except Exception as _e:
                    self.log_msg.emit(f"SWITCH '{b.label}' expr error: {_e}")
                    switch_key = "__error__"
                self.log_msg.emit(
                    f"⑃  SWITCH '{b.label}': key='{switch_key}'")
                current_text = context
                self.step_done.emit(bid, current_text)
                # port_labels maps from_port → arm key, e.g. {"E":"yes","W":"no"}
                _port_labels = b.metadata.get("port_labels", {})
                _matched = self._enqueue_outgoing(
                    queue, adj, bid, current_text, visit_counts,
                    mode="switch", branch_key=switch_key,
                    port_labels=_port_labels)
                if not _matched:
                    self.log_msg.emit(
                        f"SWITCH '{b.label}': no arm matched key '{switch_key}' - dropped.")
                continue

            elif b.btype == PipelineBlockType.FILTER:
                self.step_started.emit(bid, b.label)
                cond = b.metadata.get("filter_cond", "True")
                _safe_ns = {
                    "text": context,
                    "__builtins__": {
                        "len": len, "str": str, "int": int, "bool": bool,
                        "any": any, "all": all, "True": True, "False": False,
                        "None": None, "isinstance": isinstance,
                    },
                }
                try:
                    _pass = bool(eval(compile(cond, "<filter>", "eval"), _safe_ns))
                except Exception as _e:
                    self.log_msg.emit(f"FILTER error: {_e} -> dropping")
                    _pass = False
                if _pass:
                    self.log_msg.emit(f"⊘  FILTER '{b.label}': PASSED")
                    current_text = context
                    self.step_done.emit(bid, current_text)
                else:
                    self.log_msg.emit(f"⊘  FILTER '{b.label}': DROPPED - pipeline stopped here.")
                    self.pipeline_done.emit(
                        output_payload(
                            f"[FILTER DROPPED]\n\nCondition '{cond}' was False.\nOriginal text:\n{context}",
                            b.label,
                        ))
                    return

            elif b.btype == PipelineBlockType.TRANSFORM:
                self.step_started.emit(bid, b.label)
                ttype = b.metadata.get("transform_type", "prefix")
                try:
                    current_text = transform_text(context, b.metadata)
                except Exception as _e:
                    self.log_msg.emit(f"TRANSFORM error: {_e} - passing unchanged")
                    current_text = context
                self.log_msg.emit(
                    f"⟲  TRANSFORM '{b.label}': {ttype} → {len(current_text):,} chars")
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.MERGE:
                # Collect all contexts that arrived at this block this pass
                self.step_started.emit(bid, b.label)
                # The queue may have multiple entries for this bid; drain them
                _extras, queue = collect_merge_inputs(queue, bid, context)
                sep  = b.metadata.get("merge_sep", "\n\n---\n\n")
                try:
                    current_text = merge_contexts(_extras, b.metadata)
                except Exception as _e:
                    current_text = sep.join(_extras)
                self.log_msg.emit(
                    f"⊕  MERGE '{b.label}': {len(_extras)} inputs → {len(current_text):,} chars")
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.SPLIT:
                self.step_started.emit(bid, b.label)
                current_text = context
                self.log_msg.emit(
                    f"⑁  SPLIT '{b.label}': broadcasting to {len(adj.get(bid,[]))} outputs")
                self.step_done.emit(bid, current_text)
                # Enqueue ALL outgoing connections (fan-out)
                self._enqueue_outgoing(queue, adj, bid, current_text, visit_counts)
                continue

            # ── LLM Logic blocks ──────────────────────────────────────────────
            elif b.btype == PipelineBlockType.LLM_IF:
                self.step_started.emit(bid, b.label)
                instr = b.metadata.get("llm_instruction", "Does this text seem positive?")
                system = (
                    "You are a strict decision-making assistant. "
                    "Read the user's text and the condition carefully. "
                    "Respond with ONLY one word: YES or NO. "
                    "Do not add any explanation, punctuation, or extra words."
                )
                user_prompt = (
                    f"CONDITION: {instr}\n\n"
                    f"TEXT TO EVALUATE:\n{context[:3000]}\n\n"
                    f"Answer YES or NO:"
                )
                raw = self._llm_block_call(b, context, system, user_prompt)
                if raw is None:
                    return
                # Parse YES/NO robustly
                _answer_up = raw.strip().upper().split()[0] if raw.strip() else "NO"
                branch_taken = "TRUE" if _answer_up in ("YES", "Y", "TRUE", "1", "PASS", "POSITIVE") else "FALSE"
                self.log_msg.emit(
                    f"LLM-IF '{b.label}': raw='{raw[:60]}' -> {branch_taken}")
                current_text = context
                self.step_done.emit(bid, current_text)
                self._enqueue_outgoing(
                    queue, adj, bid, current_text, visit_counts,
                    mode="if", branch_key=branch_taken)
                continue

            elif b.btype == PipelineBlockType.LLM_SWITCH:
                self.step_started.emit(bid, b.label)
                instr = b.metadata.get("llm_instruction", "")
                _port_labels = b.metadata.get("port_labels", {})
                # Collect arm labels from outgoing connections
                arm_labels = list(dict.fromkeys(
                    str(_port_labels.get(c.from_port) or getattr(c, "branch_label", "") or "").strip()
                    for c in adj.get(bid, [])
                    if str(_port_labels.get(c.from_port) or getattr(c, "branch_label", "") or "").strip()
                ))
                arms_str = ", ".join(arm_labels) if arm_labels else "(no labels set)"
                system = (
                    "You are a strict classification assistant. "
                    "Read the user's text and task, then respond with ONLY the "
                    "exact category label from the provided list. "
                    "Do not add punctuation, explanation, or any other text. "
                    "If no category fits, respond with: other"
                )
                user_prompt = (
                    f"CLASSIFICATION TASK: {instr}\n"
                    f"VALID CATEGORIES (respond with exactly one): {arms_str}\n\n"
                    f"TEXT TO CLASSIFY:\n{context[:3000]}\n\n"
                    f"Your answer (one word/phrase only):"
                )
                raw = self._llm_block_call(b, context, system, user_prompt)
                if raw is None:
                    return
                switch_key = raw.strip().split("\n")[0].strip().lower()
                # Normalise to closest arm label (case-insensitive)
                _match = next(
                    (lbl for lbl in arm_labels if lbl.lower() == switch_key),
                    None)
                if _match is None:
                    # Fuzzy: substring match
                    _match = next(
                        (lbl for lbl in arm_labels if lbl.lower() in switch_key
                         or switch_key in lbl.lower()), "default")
                self.log_msg.emit(
                    f"LLM-SWITCH '{b.label}': classified as '{_match}' (raw: '{raw[:60]}')")
                current_text = context
                self.step_done.emit(bid, current_text)
                _matched_any = self._enqueue_outgoing(
                    queue, adj, bid, current_text, visit_counts,
                    mode="llm_switch", branch_key=_match,
                    port_labels=_port_labels)
                if not _matched_any:
                    self.log_msg.emit(
                        f"LLM-SWITCH '{b.label}': no arm matched '{_match}' - dropped")
                continue

            elif b.btype == PipelineBlockType.LLM_FILTER:
                self.step_started.emit(bid, b.label)
                instr = b.metadata.get("llm_instruction", "Pass all text")
                system = (
                    "You are a strict content filter. "
                    "Read the condition and the text. "
                    "Respond with ONLY: PASS or STOP. "
                    "If the condition is met and the text should continue, say PASS. "
                    "If the condition is NOT met and the text should be dropped, say STOP. "
                    "Do not add any explanation."
                )
                user_prompt = (
                    f"PASS CONDITION: {instr}\n\n"
                    f"TEXT:\n{context[:3000]}\n\n"
                    f"Should this text pass? Answer PASS or STOP:"
                )
                raw = self._llm_block_call(b, context, system, user_prompt)
                if raw is None:
                    return
                _ans = raw.strip().upper().split()[0] if raw.strip() else "STOP"
                _pass = _ans in ("PASS", "YES", "Y", "TRUE", "1", "ALLOW", "OK")
                if _pass:
                    self.log_msg.emit(f"LLM-FILTER '{b.label}': PASSED")
                    current_text = context
                    self.step_done.emit(bid, current_text)
                else:
                    self.log_msg.emit(
                        f"LLM-FILTER '{b.label}': STOPPED - model said: {raw[:80]}")
                    self.pipeline_done.emit(
                        output_payload(
                            (
                                f"[LLM FILTER STOPPED]\n\n"
                                f"Block: {b.label}\n"
                                f"Condition: {instr}\n"
                                f"Model decision: {raw[:200]}\n\n"
                                f"Original text:\n{context}"
                            ),
                            b.label,
                        ))
                    return

            elif b.btype == PipelineBlockType.LLM_TRANSFORM:
                self.step_started.emit(bid, b.label)
                instr = b.metadata.get("llm_instruction", "Pass the text through unchanged.")
                max_tok = int(b.metadata.get("llm_max_tokens", 512))
                system = (
                    "You are a precise text transformation assistant. "
                    "Follow the user's instruction exactly. "
                    "Output ONLY the transformed text - no preamble, no explanation, "
                    "no 'Here is the result:', just the transformed content itself."
                )
                user_prompt = (
                    f"INSTRUCTION: {instr}\n\n"
                    f"TEXT TO TRANSFORM:\n{context}"
                )
                raw = self._llm_block_call(b, context, system, user_prompt)
                if raw is None:
                    return
                # Strip common preamble patterns the model might add
                for _strip in ("Here is", "Here's", "Result:", "Output:", "Transformed:"):
                    if raw.lower().startswith(_strip.lower()):
                        raw = raw[len(_strip):].lstrip(": \n")
                        break
                current_text = raw
                self.log_msg.emit(
                    f"LLM-TRANSFORM '{b.label}': {len(context):,} -> {len(current_text):,} chars")
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.LLM_SCORE:
                self.step_started.emit(bid, b.label)
                instr = b.metadata.get("llm_instruction",
                                       "Rate the quality of this text (1=poor, 10=excellent)")
                system = (
                    "You are a precise scoring assistant. "
                    "Read the criterion and the text. "
                    "Respond with ONLY a single integer from 1 to 10. "
                    "No explanation, no punctuation, just the number."
                )
                user_prompt = (
                    f"SCORING CRITERION: {instr}\n\n"
                    f"TEXT TO SCORE:\n{context[:3000]}\n\n"
                    f"Score (1–10, integer only):"
                )
                raw = self._llm_block_call(b, context, system, user_prompt)
                if raw is None:
                    return
                # Parse score robustly
                try:
                    import re as _re2
                    _nums = _re2.findall(r'\b([1-9]|10)\b', raw)
                    score = int(_nums[0]) if _nums else 5
                except Exception:
                    score = 5
                score = max(1, min(10, score))
                if score <= 3:
                    band = "LOW"; arm_target = "E"
                elif score <= 7:
                    band = "MID"; arm_target = "S"
                else:
                    band = "HIGH"; arm_target = "W"
                self.log_msg.emit(
                    f"LLM-SCORE '{b.label}': score={score}/10 -> {band} (model: '{raw[:40]}')")
                current_text = context
                self.step_done.emit(bid, current_text)
                # Route by band or 'score' label (raw score string)
                self._enqueue_outgoing(
                    queue, adj, bid, current_text, visit_counts,
                    mode="llm_score", branch_key=arm_target,
                    score_text=score)
                continue

            elif b.btype == PipelineBlockType.CUSTOM_CODE:
                self.step_started.emit(bid, b.label)
                code = b.metadata.get("custom_code", "result = text")
                _log_lines: list = []
                _safe_builtins = {
                    "len": len, "str": str, "int": int, "float": float,
                    "bool": bool, "list": list, "dict": dict, "tuple": tuple,
                    "range": range, "enumerate": enumerate, "zip": zip,
                    "map": map, "filter": filter, "sorted": sorted,
                    "min": min, "max": max, "sum": sum, "abs": abs,
                    "round": round, "isinstance": isinstance, "hasattr": hasattr,
                    "getattr": getattr, "repr": repr, "type": type,
                    "print": lambda *a: _log_lines.append(" ".join(str(x) for x in a)),
                    "True": True, "False": False, "None": None,
                }
                _ns = {
                    "text":     context,
                    "result":   context,      # default: pass-through
                    "metadata": dict(b.metadata),
                    "log":      lambda m, _ll=_log_lines: _ll.append(str(m)),
                    "__builtins__": _safe_builtins,
                }
                try:
                    exec(compile(code, "<custom_code>", "exec"), _ns)
                    current_text = str(_ns.get("result", context))
                    for _line in _log_lines:
                        self.log_msg.emit(f"  ⌥  {_line}")
                    self.log_msg.emit(
                        f"⌥  CUSTOM_CODE '{b.label}': → {len(current_text):,} chars")
                except Exception as _e:
                    self.log_msg.emit(f"CUSTOM_CODE '{b.label}' runtime error: {_e}")
                    self.err.emit(
                        f"Custom Code block '{b.label}' raised an exception:\n\n{_e}")
                    return
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.MCP_SERVER:
                self.step_started.emit(bid, b.label)
                try:
                    result = self._run_mcp_block(b, context)
                except Exception as e:
                    self.log_msg.emit(f"MCP '{b.label}' unexpected error: {e}")
                    result = None
                if result is None:
                    if not self._abort:
                        self.err.emit(
                            f"MCP block '{b.label}' failed. "
                            f"Check the server URL/command and tool configuration.")
                    return
                current_text = result
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.WEB_SEARCH:
                self.step_started.emit(bid, b.label)
                try:
                    result = self._run_web_search_block(b, context)
                except Exception as e:
                    self.log_msg.emit(f"Web Search '{b.label}' unexpected error: {e}")
                    result = None
                if result is None:
                    if not self._abort:
                        self.err.emit(
                            f"Web Search block '{b.label}' failed. "
                            f"Check that SearXNG is installed in nativelab/web_search/searxng/")
                    return
                current_text = result
                self.step_done.emit(bid, current_text)

            # Enqueue outgoing edges
            self._enqueue_outgoing(queue, adj, bid, current_text, visit_counts)

        if not self._abort:
            self.pipeline_done.emit(current_text)

    # ── server-mode guarantee with retries ────────────────────────────────────

    def _ensure_server(self, eng: "LlamaEngine", target: str) -> bool:
        """
        Ensure eng is running in server mode for target.
        Loads the model if necessary, then retries server startup up to
        _SERVER_RETRIES times. Returns True only when confirmed server mode.
        """
        if not eng.is_loaded or eng.model_path != target:
            self.log_msg.emit(f"Loading model: {model_ref_display_name(target)}")
            if eng.is_loaded:
                eng.shutdown()
            ok = eng.load(target, log_cb=lambda m: self.log_msg.emit(m))
            if not ok:
                self.err.emit(f"Could not load model: {model_ref_display_name(target)}")
                return False
            self.log_msg.emit("Model loaded.")

        if eng.mode in ("ollama", "hf_transformers"):
            self.log_msg.emit(f"Backend ready: {eng.status_text}")
            return True

        if eng.mode == "server":
            self.log_msg.emit(f"Server already running on port {eng.server_port}")
            return True

        for attempt in range(1, self._SERVER_RETRIES + 1):
            self.log_msg.emit(
                f"Starting server (attempt {attempt}/{self._SERVER_RETRIES})...")
            ok = eng.ensure_server(log_cb=lambda m: self.log_msg.emit(m))
            if ok and eng.mode == "server":
                self.log_msg.emit(
                    f"Server mode confirmed - port {eng.server_port}")
                return True
            if attempt < self._SERVER_RETRIES:
                self.log_msg.emit(
                    f"Not ready - waiting {self._SERVER_RETRY_S}s before retry...")
                for _ in range(self._SERVER_RETRY_S * 10):
                    if self._abort:
                        return False
                    time.sleep(0.1)

        self.err.emit(
            f"'{model_ref_display_name(target)}' could not start in SERVER mode "
            f"after {self._SERVER_RETRIES} attempts.\n\n"
            f"Pipeline requires llama-server (not llama-cli).\n"
            f"Check that llama-server binary is present and the model path is valid.\n"
            f"See the Logs tab for details.")
        return False
    
    def _run_model(self, b: PipelineBlock, context: str) -> Optional[str]:
        self._last_model_error = ""
        target = b.model_path
        eng = self.primary_engine
        if eng and getattr(eng, "mode", "") == "api" and (
            not target or is_api_model_ref(target) or not is_model_ref_valid(target)
        ):
            return self._run_api_model_block(b, context)
        if eng and getattr(eng, "mode", "") == "api":
            self.err.emit(
                f"Model block '{b.label}' targets a local backend model, but the active pipeline engine is API. "
                "Load a local/Ollama/HF engine before running this block."
            )
            return None

        if not target or not is_model_ref_valid(target):
            self.err.emit(f"Model not found: {target}"); return None

        if not eng:
            self.err.emit("No engine available."); return None

        # ── enforce server mode (with retries) ──────────────────────────────
        if not self._ensure_server(eng, target):
            return None   # error already emitted by _ensure_server

        cfg = get_model_registry().get_config(target)

        ROLE_SYSTEM = {
            "general":       "You are a helpful assistant.",
            "reasoning":     "You are a careful analytical reasoning assistant. Think step by step.",
            "summarization": "You are an expert summarization assistant. Be clear and concise.",
            "coding":        "You are an expert software engineer. Write clean, well-commented code.",
            "secondary":     "You are a versatile general-purpose assistant.",
        }
        sys_msg = ROLE_SYSTEM.get(b.role, ROLE_SYSTEM["general"])

        try:
            return eng.generate_sync(
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": context},
                ],
                n_predict=cfg.n_predict,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                repeat_penalty=cfg.repeat_penalty,
                top_k=getattr(cfg, "top_k", 40),
                min_p=getattr(cfg, "min_p", 0.0),
                typical_p=getattr(cfg, "typical_p", 1.0),
                seed=getattr(cfg, "seed", -1),
                token_cb=lambda tok: self.step_token.emit(b.bid, tok),
                abort_cb=lambda: self._abort,
                context_source="Pipeline",
            ).strip()
        except Exception as e:
            self._last_model_error = f"Model block '{b.label}' error: {e}"
            self.err.emit(self._last_model_error); return None

    def _run_api_model_block(self, b: PipelineBlock, context: str) -> Optional[str]:
        eng = self.primary_engine
        cfg = getattr(eng, "_config", None)
        if cfg is None:
            self.err.emit("API engine is not configured."); return None
        ROLE_SYSTEM = {
            "general":       "You are a helpful assistant.",
            "reasoning":     "You are a careful analytical reasoning assistant. Think step by step.",
            "summarization": "You are an expert summarization assistant. Be clear and concise.",
            "coding":        "You are an expert software engineer. Write clean, well-commented code.",
            "secondary":     "You are a versatile general-purpose assistant.",
        }
        system = ROLE_SYSTEM.get(b.role, ROLE_SYSTEM["general"])
        return self._api_query_sync(
            system,
            context,
            max_tokens=0,
            temperature=float(getattr(cfg, "temperature", 0.7)),
            token_cb=lambda tok: self.step_token.emit(b.bid, tok),
        )

    # ── LLM query helper (synchronous, short answer) ──────────────────────────

    def _llm_query_sync(self, model_path: str, system_prompt: str,
                        user_prompt: str, max_tokens: int = 64,
                        temperature: float = 0.1) -> Optional[str]:
        """
        Call llama-server synchronously and return the full response text.
        Ensures the server is running for model_path before calling.
        Returns None on failure (error already logged to log_msg).
        """
        eng = self.primary_engine
        if not eng:
            self.log_msg.emit("No primary engine available for LLM logic block.")
            return None
        if getattr(eng, "mode", "") == "api":
            return self._api_query_sync(system_prompt, user_prompt, max_tokens, temperature)
        if not self._ensure_server(eng, model_path):
            return None
        try:
            cfg = get_model_registry().get_config(model_path)
            return eng.generate_sync(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                n_predict=max_tokens,
                temperature=temperature,
                top_p=getattr(cfg, "top_p", 0.9),
                repeat_penalty=getattr(cfg, "repeat_penalty", 1.1),
                top_k=getattr(cfg, "top_k", 40),
                min_p=getattr(cfg, "min_p", 0.0),
                typical_p=getattr(cfg, "typical_p", 1.0),
                seed=getattr(cfg, "seed", -1),
                context_source="Pipeline",
            ).strip()
        except Exception as _e:
            self.log_msg.emit(f"LLM query error: {_e}")
            return None

    def _llm_block_call(self, b: "PipelineBlock", context: str,
                        system: str, user_prompt: str) -> Optional[str]:
        """Wrapper around _llm_query_sync that reads block metadata for settings."""
        meta       = b.metadata
        model_path = b.model_path or meta.get("llm_model_path", "")
        max_tok    = int(meta.get("llm_max_tokens", 64))
        temp       = float(meta.get("llm_temp", 0.1))
        show_r     = bool(meta.get("llm_show_reasoning", True))
        passthru   = bool(meta.get("llm_passthrough_on_err", False))

        if getattr(self.primary_engine, "mode", "") == "api":
            model_path = model_path or getattr(self.primary_engine, "model_path", "")
        elif not model_path or not is_model_ref_valid(model_path):
            msg = f"LLM block '{b.label}': model not found - {model_path}"
            self.log_msg.emit(msg)
            if passthru:
                self.log_msg.emit(f"'{b.label}': passthrough-on-error -> continuing unchanged")
                return context
            self.err.emit(msg)
            return None

        result = self._llm_query_sync(model_path, system, user_prompt, max_tok, temp)

        if result is None:
            if passthru:
                self.log_msg.emit(
                    f"'{b.label}': LLM call failed, passthrough-on-error -> continuing unchanged")
                return context
            self.err.emit(
                f"LLM logic block '{b.label}' failed to get a response from the model.\n"
                f"Check that the model is valid and the server can start.")
            return None

        if show_r:
            self.log_msg.emit(f"  [{b.label}] model said: {result[:120]}")
        return result

    def _api_query_sync(self, system_prompt: str, user_prompt: str,
                        max_tokens: int = 512, temperature: float = 0.7,
                        token_cb=None) -> Optional[str]:
        eng = self.primary_engine
        cfg = getattr(eng, "_config", None)
        if cfg is None:
            self.log_msg.emit("API engine is not configured.")
            return None
        import urllib.error
        import urllib.request
        try:
            if cfg.api_format == "anthropic":
                context_meter.report_messages(
                    source="Pipeline",
                    engine=eng,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    n_predict=max_tokens,
                )
                payload = {
                    "model": cfg.model_id,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "max_tokens": int(max_tokens) if int(max_tokens or 0) > 0 else 8192,
                    "temperature": temperature,
                    "stream": False,
                }
                if system_prompt:
                    payload["system"] = system_prompt
                req = urllib.request.Request(
                    f"{cfg.base_url.rstrip('/')}/v1/messages",
                    data=json.dumps(payload).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": cfg.api_key,
                        "anthropic-version": "2023-06-01",
                    },
                )
                with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
                    d = json.loads(r.read().decode("utf-8", errors="replace"))
                content = d.get("content") or []
                text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
            else:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                context_meter.report_messages(
                    source="Pipeline",
                    engine=eng,
                    messages=messages,
                    n_predict=max_tokens,
                )
                payload = {
                    "model": cfg.model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False,
                }
                max_tokens = int(max_tokens or 0)
                if max_tokens > 0:
                    payload["max_tokens"] = max_tokens
                req = urllib.request.Request(
                    f"{cfg.base_url.rstrip('/')}/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {cfg.api_key}",
                    },
                )
                with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
                    d = json.loads(r.read().decode("utf-8", errors="replace"))
                choices = d.get("choices") or []
                text = choices[0].get("message", {}).get("content", "") if choices else ""
            if token_cb and text:
                token_cb(text)
            if text:
                context_meter.append_output(text)
            return text.strip()
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                raw = ""
            detail = f"API query error: HTTP {e.code}: {raw or getattr(e, 'reason', '')}"
            self._last_model_error = detail
            self.log_msg.emit(detail)
            self.err.emit(detail)
            return None
        except Exception as e:
            detail = f"API query error: {e}"
            self._last_model_error = detail
            self.log_msg.emit(detail)
            self.err.emit(detail)
            return None

    # ── context injection blocks ──────────────────────────────────────────────

    def _run_reference(self, b: PipelineBlock, context: str) -> str:
        text = b.metadata.get("ref_text", "")
        if not text:
            self.log_msg.emit(f"Reference '{b.label}' has no text - passing unchanged.")
            return context
        name = b.metadata.get("ref_name", b.label)
        self.log_msg.emit(f"Injecting reference '{name}' ({len(text):,} chars)")
        injected, _name = reference_context(context, b.metadata, b.label)
        return injected

    def _run_knowledge(self, b: PipelineBlock, context: str) -> str:
        text = b.metadata.get("knowledge_text", "")
        if not text:
            self.log_msg.emit(f"Knowledge '{b.label}' has no text - passing unchanged.")
            return context
        self.log_msg.emit(f"Injecting knowledge ({len(text):,} chars)")
        return knowledge_context(context, b.metadata)

    def _run_pdf(self, b: PipelineBlock, context: str) -> Optional[str]:
        pdf_path = b.metadata.get("pdf_path", "")
        if not pdf_path or not Path(pdf_path).exists():
            self.err.emit(f"PDF block '{b.label}': file not found → {pdf_path}")
            return None
        if not HAS_PDF:
            self.err.emit("pypdf not installed. Run: pip install pypdf")
            return None
        try:
            from pypdf import PdfReader
            reader   = PdfReader(pdf_path)
            pdf_text = "\n".join(pg.extract_text() or "" for pg in reader.pages)
        except Exception as e:
            self.err.emit(f"PDF read error: {e}"); return None
        if not pdf_text.strip():
            self.err.emit(f"PDF block '{b.label}': no text extracted."); return None

        fname    = Path(pdf_path).name
        pdf_role = b.metadata.get("pdf_role", "reference")   # "reference" | "main"
        self.log_msg.emit(
            f"PDF loaded: {fname} ({len(pdf_text):,} chars) "
            f"- role: {pdf_role.upper()}")

        # ── Summarise large PDFs ──────────────────────────────────────────────
        LIMIT = 4500
        if len(pdf_text) > LIMIT:
            pdf_text = self._summarise_pdf(b, fname, pdf_text)
            if pdf_text is None:
                return None

        # ── Compose output based on role ──────────────────────────────────────
        #
        #  pdf_role == "reference":
        #    prior context  → MAIN TEXT
        #    pdf_text       → REFERENCE (appended below)
        #
        #  pdf_role == "main":
        #    pdf_text       → MAIN TEXT
        #    prior context  → REFERENCE (appended below)
        #
        #  If either side is empty/None, it is simply omitted.

        prior = context.strip() if context else ""
        pdf   = pdf_text.strip()

        if pdf_role == "reference":
            main_text = prior
            ref_label = f"[PDF REFERENCE: {fname}]"
            ref_text  = pdf
        else:  # "main"
            main_text = pdf
            ref_label = "[PRIOR CONTEXT REFERENCE]"
            ref_text  = prior

        parts = []
        if main_text:
            parts.append(main_text)
        if ref_text:
            parts.append(f"{ref_label}\n{ref_text}\n[/REFERENCE]")

        result = "\n\n".join(parts)
        self.log_msg.emit(
            f"PDF block assembled: {len(result):,} chars "
            f"({'PDF=main, prior=ref' if pdf_role == 'main' else 'prior=main, PDF=ref'})")
        return result

    def _summarise_pdf(self, b: "PipelineBlock", fname: str,
                       pdf_text: str) -> Optional[str]:
        """Chunk-summarise a large PDF via the server. Returns summary string or None."""
        eng   = self.primary_engine
        CHUNK = 3000
        chunks, text = [], pdf_text
        while text:
            if len(text) <= CHUNK:
                chunks.append(text.strip()); break
            cut = text.rfind("\n\n", 0, CHUNK)
            if cut < 200: cut = CHUNK
            chunks.append(text[:cut].strip())
            text = text[cut:].lstrip()

        mp = getattr(eng, "model_path", "")
        fam = detect_model_family(model_ref_payload(mp) or mp)
        summaries = []
        for i, chunk in enumerate(chunks):
            if self._abort:
                return None
            self.log_msg.emit(f"  Summarising chunk {i+1}/{len(chunks)}...")
            plain_prompt = (
                f"Summarise this document section concisely. "
                f"File: '{fname}' - Section {i+1}/{len(chunks)}\n\n{chunk}"
            )
            if eng and eng.is_loaded and hasattr(eng, "generate_sync"):
                try:
                    result = eng.generate_sync(
                        prompt=plain_prompt,
                        n_predict=400,
                        temperature=0.3,
                        context_source="Pipeline",
                    )
                    summaries.append(f"[§{i+1}] {result.strip()}")
                    continue
                except Exception:
                    pass
            summaries.append(f"[§{i+1}] {chunk[:600]}…")

        summary = "\n\n".join(summaries)
        self.log_msg.emit(f"PDF summarised: {len(summary):,} chars")
        return summary

    # ── MCP server block ────────────────────────────────────────────────────

    def _run_mcp_block(self, b: "PipelineBlock", context: str) -> Optional[str]:
        """Execute an MCP tool call via the configured server."""
        transport = b.metadata.get("mcp_transport", "sse")
        url = b.metadata.get("mcp_url", "")
        tool_name = b.metadata.get("mcp_tool_name", "")
        arg_name = b.metadata.get("mcp_arg_name", "")
        auth_token = b.metadata.get("mcp_auth_token", "")
        auth_env_text = b.metadata.get("mcp_auth_env", "")

        if not url:
            self.log_msg.emit(f"MCP block '{b.label}': no server URL configured.")
            return None
        if not tool_name:
            self.log_msg.emit(f"MCP block '{b.label}': no tool selected.")
            return None

        # Build arguments — use configured arg name or default to "input"
        arguments = {}
        key = arg_name if arg_name else "input"
        arguments[key] = context

        self.log_msg.emit(
            f"MCP '{b.label}': calling tool '{tool_name}' "
            f"via {transport} → {url[:60]}")

        try:
            from nativelab.integrations.mcp_client import McpClient
            # Parse env vars from block metadata
            auth_env = {}
            for line in str(auth_env_text or "").splitlines():
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k:
                        auth_env[k] = v

            client = McpClient()
            ok, result = client.execute(
                transport, url, tool_name, arguments,
                auth_token=auth_token or None,
                auth_env=auth_env or None,
            )
            client.shutdown()

            if not ok:
                err = result if isinstance(result, str) else "Tool call failed"
                self.log_msg.emit(f"MCP '{b.label}' error: {err}")
                return None

            text = str(result) if result else ""
            if not text.strip():
                self.log_msg.emit(f"MCP '{b.label}': tool returned empty result")
                return context  # pass through unchanged

            self.log_msg.emit(
                f"MCP '{b.label}': tool returned {len(text):,} chars")
            return text
        except ImportError:
            self.log_msg.emit(
                f"MCP '{b.label}': mcp_client module not found. "
                f"Ensure nativelab.integrations.mcp_client is installed.")
            return None
        except Exception as e:
            self.log_msg.emit(f"MCP '{b.label}' exception: {e}")
            return None

    # ── web search block ────────────────────────────────────────────────────

    def _run_web_search_block(self, b: "PipelineBlock", context: str) -> Optional[str]:
        """Execute a web search using SearXNG in-process."""
        categories = b.metadata.get("ws_categories", ["general"])
        language = b.metadata.get("ws_language", "en")
        max_results = int(b.metadata.get("ws_max_results", 10))
        timeout = int(b.metadata.get("ws_timeout", 10))
        output_format = b.metadata.get("ws_output_format", "text")

        # Use incoming context as the search query
        query = context.strip()
        if not query:
            self.log_msg.emit(f"Web Search '{b.label}': empty query, passing through")
            return context

        self.log_msg.emit(
            f"Web Search '{b.label}': searching '{query[:50]}' "
            f"in {', '.join(categories)} ({language})")

        try:
            from nativelab.web_search import web_search, web_search_text

            if output_format == "json":
                import json
                results = web_search(
                    query,
                    categories=categories,
                    language=language,
                    max_results=max_results,
                    timeout=timeout,
                )
                text = json.dumps(results, ensure_ascii=False, indent=2)
            else:
                text = web_search_text(
                    query,
                    categories=categories,
                    language=language,
                    max_results=max_results,
                    timeout=timeout,
                )

            if not text or text.startswith("No results"):
                self.log_msg.emit(f"Web Search '{b.label}': no results found")
                # Pass through original context with a note
                return f"[Web search returned no results for: {query}]\n\n{context}"

            self.log_msg.emit(
                f"Web Search '{b.label}': got {len(text):,} chars of results")
            return text

        except ImportError:
            self.log_msg.emit(
                f"Web Search '{b.label}': nativelab.web_search not available. "
                f"Ensure SearXNG is installed in nativelab/web_search/searxng/")
            return None
        except Exception as e:
            self.log_msg.emit(f"Web Search '{b.label}' error: {e}")
            return None
    
