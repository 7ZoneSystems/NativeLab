from nativelab.core.engines.llamaengine import LlamaEngine
from nativelab.imports.import_global import QThread,Path, pyqtSignal, List, Dict, time, Optional, json, HAS_PDF
from nativelab.Model.model_global import detect_model_family, get_model_registry, is_api_model_ref, is_model_ref_valid, model_ref_display_name, model_ref_payload
from nativelab.GlobalConfig.config_global import LONG_TIMEOUT_SECONDS
from .blck_typ import PipelineConnection, PipelineBlockType
from .pipblck import PipelineBlock
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

    def abort(self):
        self._abort = True

    # ── adjacency helpers ─────────────────────────────────────────────────────

    def _adjacency(self) -> Dict[int, List[PipelineConnection]]:
        adj: Dict[int, List[PipelineConnection]] = {}
        for c in self.connections:
            adj.setdefault(c.from_block_id, []).append(c)
        return adj

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
                # Find which block sent us this context so we can label it
                sender_label = ""
                for conn in self.connections:
                    if conn.to_block_id == bid:
                        sender = self.blocks.get(conn.from_block_id)
                        if sender:
                            sender_label = sender.label
                        break

                current_text = context
                # Emit step_done with the raw context (no intermediate noise)
                self.step_done.emit(bid, context)
                # pipeline_done carries (final_text, sender_label) encoded together
                # We encode as JSON so the tab can split them cleanly
                import json as _json
                self.pipeline_done.emit(_json.dumps({
                    "text":   context,
                    "sender": sender_label,
                }))
                return

            elif b.btype == PipelineBlockType.MODEL:
                self.step_started.emit(bid, b.label)
                result = self._run_model(b, context)
                if result is None:
                    if not self._abort:
                        self.err.emit(f"Model block '{b.label}' returned no output.")
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
                for conn in adj.get(bid, []):
                    # from_port E → TRUE arm, W → FALSE arm; N/S = pass-through
                    port = conn.from_port
                    take = (port not in ("E", "W")) or (
                        port == "E" and branch_taken == "TRUE") or (
                        port == "W" and branch_taken == "FALSE")
                    if take:
                        key = f"{conn.from_block_id}->{conn.to_block_id}"
                        visits = visit_counts.get(key, 0)
                        limit  = conn.loop_times if conn.is_loop else 1
                        if visits < limit:
                            visit_counts[key] = visits + 1
                            queue.append((conn.to_block_id, current_text))
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
                _matched = False
                for conn in adj.get(bid, []):
                    # resolve arm key: metadata label → from_port letter → "default"
                    bl = _port_labels.get(conn.from_port, conn.from_port)
                    if bl == switch_key or bl == "default" or not bl:
                        _matched = True
                        key = f"{conn.from_block_id}->{conn.to_block_id}"
                        visits = visit_counts.get(key, 0)
                        limit  = conn.loop_times if conn.is_loop else 1
                        if visits < limit:
                            visit_counts[key] = visits + 1
                            queue.append((conn.to_block_id, current_text))
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
                        __import__("json").dumps(
                            {"text": f"[FILTER DROPPED]\n\nCondition '{cond}' was False.\nOriginal text:\n{context}",
                             "sender": b.label}))
                    return

            elif b.btype == PipelineBlockType.TRANSFORM:
                self.step_started.emit(bid, b.label)
                ttype = b.metadata.get("transform_type", "prefix")
                val   = b.metadata.get("transform_val", "")
                try:
                    if ttype == "prefix":
                        current_text = f"{val}\n{context}"
                    elif ttype == "suffix":
                        current_text = f"{context}\n{val}"
                    elif ttype == "replace":
                        find_s = b.metadata.get("transform_find", "")
                        repl_s = b.metadata.get("transform_repl", "")
                        current_text = context.replace(find_s, repl_s) if find_s else context
                    elif ttype == "upper":
                        current_text = context.upper()
                    elif ttype == "lower":
                        current_text = context.lower()
                    elif ttype == "strip":
                        current_text = context.strip()
                    elif ttype == "truncate":
                        n = int(val) if val else 500
                        current_text = context[:n]
                    else:
                        current_text = context
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
                _extras = [context]
                _new_queue = []
                for _qbid, _qctx in queue:
                    if _qbid == bid:
                        _extras.append(_qctx)
                    else:
                        _new_queue.append((_qbid, _qctx))
                queue = _new_queue
                mode = b.metadata.get("merge_mode", "concat")
                sep  = b.metadata.get("merge_sep", "\n\n---\n\n")
                try:
                    if mode == "concat":
                        current_text = sep.join(_extras)
                    elif mode == "prepend":
                        current_text = sep.join(reversed(_extras))
                    elif mode == "append":
                        current_text = sep.join(_extras)
                    elif mode == "json":
                        current_text = __import__("json").dumps(_extras, indent=2)
                    else:
                        current_text = sep.join(_extras)
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
                for conn in adj.get(bid, []):
                    key = f"{conn.from_block_id}->{conn.to_block_id}"
                    visits = visit_counts.get(key, 0)
                    limit  = conn.loop_times if conn.is_loop else 1
                    if visits < limit:
                        visit_counts[key] = visits + 1
                        queue.append((conn.to_block_id, current_text))
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
                for conn in adj.get(bid, []):
                    # from_port E → TRUE arm, W → FALSE arm; N/S = pass-through
                    port = conn.from_port
                    take = (port not in ("E", "W")) or (
                        port == "E" and branch_taken == "TRUE") or (
                        port == "W" and branch_taken == "FALSE")
                    if take:
                        key = f"{conn.from_block_id}->{conn.to_block_id}"
                        visits = visit_counts.get(key, 0)
                        limit  = conn.loop_times if conn.is_loop else 1
                        if visits < limit:
                            visit_counts[key] = visits + 1
                            queue.append((conn.to_block_id, current_text))
                continue

            elif b.btype == PipelineBlockType.LLM_SWITCH:
                self.step_started.emit(bid, b.label)
                instr = b.metadata.get("llm_instruction", "")
                # Collect arm labels from outgoing connections
                arm_labels = list({
                    getattr(c, "branch_label", "")
                    for c in adj.get(bid, [])
                    if getattr(c, "branch_label", "")
                })
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
                _port_labels = b.metadata.get("port_labels", {})
                _matched_any = False
                for conn in adj.get(bid, []):
                    bl = _port_labels.get(conn.from_port, conn.from_port)
                    if bl.lower() == _match.lower() or bl == "default" or not bl:
                        _matched_any = True
                        key = f"{conn.from_block_id}->{conn.to_block_id}"
                        visits = visit_counts.get(key, 0)
                        limit  = conn.loop_times if conn.is_loop else 1
                        if visits < limit:
                            visit_counts[key] = visits + 1
                            queue.append((conn.to_block_id, current_text))
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
                        __import__("json").dumps({
                            "text": (
                                f"[LLM FILTER STOPPED]\n\n"
                                f"Block: {b.label}\n"
                                f"Condition: {instr}\n"
                                f"Model decision: {raw[:200]}\n\n"
                                f"Original text:\n{context}"
                            ),
                            "sender": b.label,
                        }))
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
                for conn in adj.get(bid, []):
                    # from_port E → LOW, S → MID, W → HIGH; N = raw score pass-through
                    port = conn.from_port
                    take = (
                        port == arm_target          # E/S/W matches band
                        or port == "N"              # N port → raw score value
                        or port not in ("E","S","W","N")  # unknown port → always pass
                    )
                    if take:
                        key = f"{conn.from_block_id}->{conn.to_block_id}"
                        visits = visit_counts.get(key, 0)
                        limit  = conn.loop_times if conn.is_loop else 1
                        if visits < limit:
                            visit_counts[key] = visits + 1
                            out_txt = str(score) if port == "N" else current_text
                            queue.append((conn.to_block_id, out_txt))
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

            # Enqueue outgoing edges
            for conn in adj.get(bid, []):
                key    = f"{conn.from_block_id}->{conn.to_block_id}"
                visits = visit_counts.get(key, 0)
                limit  = conn.loop_times if conn.is_loop else 1
                if visits < limit:
                    visit_counts[key] = visits + 1
                    queue.append((conn.to_block_id, current_text))

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
                token_cb=lambda tok: self.step_token.emit(b.bid, tok),
                abort_cb=lambda: self._abort,
            ).strip()
        except Exception as e:
            self.err.emit(f"Model block error: {e}"); return None

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
            return eng.generate_sync(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                n_predict=max_tokens,
                temperature=temperature,
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
        import urllib.request
        try:
            if cfg.api_format == "anthropic":
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
            return text.strip()
        except Exception as e:
            self.log_msg.emit(f"API query error: {e}")
            return None

    # ── context injection blocks ──────────────────────────────────────────────

    def _run_reference(self, b: PipelineBlock, context: str) -> str:
        text = b.metadata.get("ref_text", "")
        if not text:
            self.log_msg.emit(f"Reference '{b.label}' has no text - passing unchanged.")
            return context
        name = b.metadata.get("ref_name", b.label)
        self.log_msg.emit(f"Injecting reference '{name}' ({len(text):,} chars)")
        return (
            f"[REFERENCE: {name}]\n"
            f"{text[:4000]}"
            + ("…\n[truncated]" if len(text) > 4000 else "")
            + f"\n[/REFERENCE]\n\n{context}"
        )

    def _run_knowledge(self, b: PipelineBlock, context: str) -> str:
        text = b.metadata.get("knowledge_text", "")
        if not text:
            self.log_msg.emit(f"Knowledge '{b.label}' has no text - passing unchanged.")
            return context
        self.log_msg.emit(f"Injecting knowledge ({len(text):,} chars)")
        return (
            f"Knowledge Base:\n"
            f"{text[:3000]}"
            + ("…\n[truncated]" if len(text) > 3000 else "")
            + f"\n\n---\n\n{context}"
        )

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
                    result = eng.generate_sync(prompt=plain_prompt, n_predict=400, temperature=0.3)
                    summaries.append(f"[§{i+1}] {result.strip()}")
                    continue
                except Exception:
                    pass
            summaries.append(f"[§{i+1}] {chunk[:600]}…")

        summary = "\n\n".join(summaries)
        self.log_msg.emit(f"PDF summarised: {len(summary):,} chars")
        return summary
    
