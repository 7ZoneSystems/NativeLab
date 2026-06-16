from nativelab.imports.import_global import QInputDialog,QDialog, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QPushButton, QFileDialog, QTextEdit, QFont, QFrame, QSpinBox, QMessageBox, Qt, Path, QLineEdit, QCheckBox, QWidget
from nativelab.Model.model_global import get_model_registry, detect_model_family, is_model_ref_valid
from nativelab.GlobalConfig.config_global import ROLE_ICONS
from .pipblck import PipelineBlock
from nativelab.UI.UI_const import C
from nativelab.UI.buildUI import prepare_adaptive_window
from nativelab.UI.icons import set_button_icon, set_label_icon, set_status_label
from nativelab.UI.toggle import ToggleSwitch
class LlmLogicEditorDialog(QDialog):
    """
    Configuration dialog for LLM-backed logic blocks.
    The user writes conditions and instructions in plain English.
    The attached model evaluates them at runtime against the incoming text.
    """

    # Per-type documentation shown in the dialog
    _TYPE_INFO = {
        "llm_if": {
            "icon":  "LLM IF / ELSE",
            "about": (
                "The model reads the incoming text and your condition, then answers "
                "YES or NO. YES routes to the TRUE (E) port, NO to the FALSE (W) port.\n\n"
                "Write your condition as a plain English question or statement.\n"
                "The model will always respond with a single word: YES or NO."
            ),
            "label":       "Condition (plain English):",
            "placeholder": "e.g.  Does this text contain a complaint or negative sentiment?\n"
                           "e.g.  Is the answer longer than a short paragraph?\n"
                           "e.g.  Does the user seem confused or ask a follow-up question?",
            "branch_hint": "TRUE (E port) → YES    FALSE (W port) → NO",
        },
        "llm_switch": {
            "icon":  "LLM SWITCH",
            "about": (
                "The model classifies the incoming text into one of your defined categories. "
                "Draw outgoing arrows and label each with the exact category name. "
                "The model picks the best match and only that arm is followed."
            ),
            "label":       "Classification task + categories:",
            "placeholder": "e.g.  Classify this text as one of: positive, negative, neutral\n"
                           "e.g.  What language is this in? english, french, spanish, other\n"
                           "e.g.  Is this a question, a complaint, a compliment, or other?",
            "branch_hint": "Connect outgoing arrows labelled with each category name.",
        },
        "llm_filter": {
            "icon":  "LLM FILTER",
            "about": (
                "The model decides whether the incoming text should continue through "
                "the pipeline (PASS) or be dropped (STOP). "
                "If dropped, the pipeline ends with a clear reason message."
            ),
            "label":       "Pass condition (plain English):",
            "placeholder": "e.g.  Only pass if this is a genuine technical question\n"
                           "e.g.  Pass only if the text contains a clear action item\n"
                           "e.g.  Allow through only if the topic is related to software",
            "branch_hint": "PASS → continues    STOP → pipeline ends with reason",
        },
        "llm_transform": {
            "icon":  "LLM TRANSFORM",
            "about": (
                "The model rewrites or transforms the incoming text according to your "
                "instruction. The result replaces the context for all downstream blocks. "
                "Be specific - the model will follow your instruction precisely."
            ),
            "label":       "Transformation instruction:",
            "placeholder": "e.g.  Summarise this in three bullet points\n"
                           "e.g.  Rewrite in a formal professional tone\n"
                           "e.g.  Extract only the action items as a numbered list\n"
                           "e.g.  Translate to Spanish",
            "branch_hint": "Single output - transformed text flows to all connected blocks.",
        },
        "llm_score": {
            "icon":  "LLM SCORE",
            "about": (
                "The model scores the incoming text from 1 to 10 on your criterion. "
                "Route the result by score band:\n"
                "  LOW  (1–3)   → E port\n"
                "  MID  (4–7)   → S port\n"
                "  HIGH (8–10)  → W port\n"
                "You can also connect a 'score' label to receive the raw score as text."
            ),
            "label":       "Scoring criterion:",
            "placeholder": "e.g.  Rate the clarity of this explanation (1=very unclear, 10=crystal clear)\n"
                           "e.g.  Score the sentiment positivity (1=very negative, 10=very positive)\n"
                           "e.g.  Rate the technical complexity (1=very simple, 10=expert-level)",
            "branch_hint": "LOW (E) 1–3    MID (S) 4–7    HIGH (W) 8–10    score label = raw number",
        },
    }

    def __init__(self, block: "PipelineBlock", parent=None):
        super().__init__(parent)
        self._block = block
        info = self._TYPE_INFO.get(block.btype, {})
        self.setWindowTitle(f"{info.get('icon','LLM Logic')} - {block.label}")
        prepare_adaptive_window(self, 700, 530, min_width=540, min_height=380)
        self._build(info)

    def _build(self, info: dict):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # ── Header ───────────────────────────────────────────────────────────
        hdr = QLabel(info.get("icon", "LLM Logic Block"))
        set_label_icon(hdr, "reasoning", hdr.text(), 18)
        hdr.setStyleSheet(
            f"color:{C['txt']};font-size:14px;font-weight:bold;")
        root.addWidget(hdr)

        about = QLabel(info.get("about", ""))
        about.setWordWrap(True)
        about.setStyleSheet(
            f"color:{C['txt2']};font-size:11px;"
            f"background:{C['bg2']};border-radius:6px;padding:10px 12px;"
            f"border-left:3px solid {C['acc']};")
        root.addWidget(about)

        # ── Branch routing hint ───────────────────────────────────────────────
        hint_lbl = QLabel(info.get('branch_hint',''))
        hint_lbl.setStyleSheet(
            f"color:{C['ok']};font-size:10px;font-weight:600;"
            f"padding:4px 8px;background:{C['bg2']};border-radius:4px;")
        hint_lbl.setWordWrap(True)
        root.addWidget(hint_lbl)

        # ── Model selector ────────────────────────────────────────────────────
        model_lbl = QLabel("MODEL  (required - evaluated at runtime):")
        model_lbl.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;font-weight:700;letter-spacing:1px;")
        root.addWidget(model_lbl)

        model_row = QHBoxLayout(); model_row.setSpacing(8)
        self.model_combo = QComboBox()
        self.model_combo.setFixedHeight(30)
        # Populate from registry
        _models = get_model_registry().all_models()
        _cur_path = self._block.model_path or self._block.metadata.get("llm_model_path", "")
        _sel_idx = 0
        for _i, _m in enumerate(_models):
            self.model_combo.addItem(
                f"{ROLE_ICONS.get(_m.get('role','general'),'General')}  {_m['name']}",
                _m["path"])
            if _m["path"] == _cur_path:
                _sel_idx = _i
        if _models:
            self.model_combo.setCurrentIndex(_sel_idx)
        else:
            self.model_combo.addItem("No models registered - add one in Models tab", "")

        btn_browse_model = QPushButton("Browse…")
        btn_browse_model.setFixedHeight(30); btn_browse_model.setFixedWidth(80)
        btn_browse_model.clicked.connect(self._browse_model)

        self.model_status = QLabel("")
        self.model_status.setFixedWidth(18)
        self.model_combo.currentIndexChanged.connect(self._update_model_status)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.model_status)
        model_row.addWidget(btn_browse_model)
        root.addLayout(model_row)
        self._update_model_status()

        # ── Condition / instruction editor ────────────────────────────────────
        instr_lbl = QLabel(info.get("label", "Instruction:"))
        instr_lbl.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;font-weight:700;letter-spacing:1px;")
        root.addWidget(instr_lbl)

        self.instr_edit = QTextEdit()
        self.instr_edit.setFont(QFont("Inter", 12))
        self.instr_edit.setMaximumHeight(130)
        self.instr_edit.setPlaceholderText(
            info.get("placeholder", "Enter your instruction in plain English…"))
        self.instr_edit.setPlainText(
            self._block.metadata.get("llm_instruction", ""))
        root.addWidget(self.instr_edit)

        # ── Advanced settings (collapsible) ───────────────────────────────────
        adv_hdr = QLabel("▸  ADVANCED SETTINGS  (click to expand)")
        adv_hdr.setStyleSheet(
            f"color:{C['txt2']};font-size:10px;font-weight:600;"
            f"padding:4px 2px;")
        adv_hdr.setCursor(Qt.CursorShape.PointingHandCursor)
        root.addWidget(adv_hdr)

        self._adv_frame = QFrame()
        self._adv_frame.setObjectName("tab_card")
        self._adv_frame.setVisible(False)
        adv_l = QVBoxLayout(self._adv_frame)
        adv_l.setContentsMargins(12, 8, 12, 10); adv_l.setSpacing(8)

        def _adv_row(lbl_text, widget):
            r = QHBoxLayout(); r.setSpacing(8)
            l = QLabel(lbl_text); l.setFixedWidth(180)
            l.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
            r.addWidget(l); r.addWidget(widget); r.addStretch()
            adv_l.addLayout(r)

        self.spin_max_tokens = QSpinBox()
        self.spin_max_tokens.setRange(8, 512)
        self.spin_max_tokens.setValue(int(self._block.metadata.get("llm_max_tokens", 64)))
        self.spin_max_tokens.setFixedHeight(26); self.spin_max_tokens.setFixedWidth(80)
        self.spin_max_tokens.setToolTip(
            "Max tokens the model generates for its answer.\n"
            "Keep small for routing blocks (16–64), larger for transform (128–512).")
        _adv_row("Max response tokens:", self.spin_max_tokens)

        self.spin_temp = QSpinBox()
        self.spin_temp.setRange(0, 100)
        self.spin_temp.setValue(int(float(self._block.metadata.get("llm_temp", 0.1)) * 100))
        self.spin_temp.setFixedHeight(26); self.spin_temp.setFixedWidth(80)
        self.spin_temp.setSuffix("  (÷100)")
        self.spin_temp.setToolTip(
            "Temperature for the LLM response (0–100 maps to 0.00–1.00).\n"
            "Use 0–15 for routing decisions, 30–60 for creative transforms.")
        _adv_row("Temperature (×100):", self.spin_temp)

        self.check_show_reasoning = ToggleSwitch("Show model reasoning in log")
        self.check_show_reasoning.setChecked(
            bool(self._block.metadata.get("llm_show_reasoning", True)))
        self.check_show_reasoning.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        adv_l.addWidget(self.check_show_reasoning)

        self.check_passthrough_on_err = ToggleSwitch(
            "Pass text through unchanged if model call fails (instead of stopping pipeline)")
        self.check_passthrough_on_err.setChecked(
            bool(self._block.metadata.get("llm_passthrough_on_err", False)))
        self.check_passthrough_on_err.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        adv_l.addWidget(self.check_passthrough_on_err)

        root.addWidget(self._adv_frame)
        adv_hdr.mousePressEvent = lambda _: self._toggle_adv(adv_hdr)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        btn_save = QPushButton("Save & Close")
        set_button_icon(btn_save, "save", "Save & Close")
        btn_save.setObjectName("btn_send"); btn_save.setFixedHeight(32)
        btn_save.clicked.connect(self._save_and_close)
        btn_cancel = QPushButton("Cancel")
        set_button_icon(btn_cancel, "x", "Cancel")
        btn_cancel.setFixedHeight(32)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        root.addLayout(btn_row)

    def _toggle_adv(self, hdr_lbl: QLabel):
        vis = not self._adv_frame.isVisible()
        self._adv_frame.setVisible(vis)
        hdr_lbl.setText(
            ("▾  ADVANCED SETTINGS  (click to collapse)"
             if vis else "▸  ADVANCED SETTINGS  (click to expand)"))

    def _update_model_status(self):
        path = self.model_combo.currentData() or ""
        ok = bool(path) and is_model_ref_valid(path)
        set_status_label(self.model_status, "", "ok" if ok else "err", 15)
        self.model_status.setToolTip(path if path else "No model selected")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", str(Path.home()),
            "GGUF Models (*.gguf);;All Files (*)")
        if path:
            # Add to registry if not already there
            get_model_registry().add(path)
            # Refresh combo
            already = self.model_combo.findData(path)
            if already == -1:
                fam = detect_model_family(path)
                self.model_combo.addItem(Path(path).name, path)
                already = self.model_combo.count() - 1
            self.model_combo.setCurrentIndex(already)

    def _save_and_close(self):
        instr = self.instr_edit.toPlainText().strip()
        if not instr:
            QMessageBox.warning(self, "Missing Instruction",
                                "Please enter a condition or instruction."); return
        model_path = self.model_combo.currentData() or ""
        if not model_path or not is_model_ref_valid(model_path):
            ans = QMessageBox.question(
                self, "No Model Selected",
                "No valid model is selected. The block will fail at runtime.\n\n"
                "Save anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return

        self._block.model_path = model_path
        self._block.metadata["llm_instruction"]      = instr
        self._block.metadata["llm_model_path"]       = model_path
        self._block.metadata["llm_max_tokens"]       = self.spin_max_tokens.value()
        self._block.metadata["llm_temp"]             = round(self.spin_temp.value() / 100, 2)
        self._block.metadata["llm_show_reasoning"]   = self.check_show_reasoning.isChecked()
        self._block.metadata["llm_passthrough_on_err"] = self.check_passthrough_on_err.isChecked()

        # Build a readable label from the instruction
        _preview = instr.replace("\n", " ").strip()[:22]
        _icon_map = {
            "llm_if": "LLM-IF", "llm_switch": "LLM-SW",
            "llm_filter": "LLM-FL", "llm_transform": "LLM-TX", "llm_score": "LLM-SC",
        }
        _icon = _icon_map.get(self._block.btype, "LLM")
        self._block.label = f"{_icon} {_preview}"
        self.accept()

class CodeEditorDialog(QDialog):
    """
    Full code editor for CUSTOM_CODE pipeline blocks.
    Shows available variables, validates syntax live, saves to block.metadata.
    """

    # Available variable documentation shown to user
    _VAR_DOCS = [
        ("text",    "str",  "The incoming context string from the previous block."),
        ("result",  "str",  "Write your output here - the pipeline continues with this."),
        ("metadata","dict", "Block metadata dict (read/write - persists across runs)."),
        ("log",     "fn",   "log('message') - writes to the pipeline execution log."),
    ]

    _TEMPLATE = """\
# CUSTOM_CODE block
# Variables available:
#   text     → incoming context string (read-only)
#   result   → set this to your output (default: text unchanged)
#   metadata → block metadata dict (persists across runs)
#   log(msg) → write to pipeline log
#
# Example: count words
result = text   # default: pass through unchanged

word_count = len(text.split())
log(f"Word count: {word_count}")

# You can also modify result:
# result = text.upper()
# result = f"[Processed]\\n{text}"
"""

    def __init__(self, block: "PipelineBlock", parent=None):             
        super().__init__(parent)
        self._block = block
        self.setWindowTitle(f"⌥ Code Editor - {block.label}")
        prepare_adaptive_window(self, 820, 620, min_width=620, min_height=440)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # ── Header ───────────────────────────────────────────────────────────
        hdr = QLabel("⌥  Custom Code Block Editor")
        hdr.setStyleSheet(
            f"color:{C['txt']};font-size:14px;font-weight:bold;")
        root.addWidget(hdr)
        desc = QLabel(
            "Write Python that runs inline during pipeline execution. "
            "Your code receives <b>text</b> (incoming context) and must set "
            "<b>result</b> (output sent to next block). "
            "Code runs in a sandboxed exec() with no network or disk access.")
        desc.setWordWrap(True)
        desc.setTextFormat(Qt.TextFormat.RichText)
        desc.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        root.addWidget(desc)

        # ── Available variables table ─────────────────────────────────────────
        vars_hdr = QLabel("AVAILABLE VARIABLES")
        vars_hdr.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;font-weight:700;letter-spacing:1px;")
        root.addWidget(vars_hdr)

        vars_frame = QFrame(); vars_frame.setObjectName("tab_card")
        vars_l = QHBoxLayout(vars_frame)
        vars_l.setContentsMargins(10, 8, 10, 8); vars_l.setSpacing(16)
        for name, typ, doc in self._VAR_DOCS:
            col = QVBoxLayout(); col.setSpacing(2)
            n_lbl = QLabel(f"<b>{name}</b>  <span style='color:{C['txt3']}'>{typ}</span>")
            n_lbl.setTextFormat(Qt.TextFormat.RichText)
            n_lbl.setStyleSheet(
                f"font-family:Consolas,monospace;font-size:12px;color:{C['acc2']};")
            d_lbl = QLabel(doc)
            d_lbl.setWordWrap(True)
            d_lbl.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
            col.addWidget(n_lbl); col.addWidget(d_lbl)
            vars_l.addLayout(col, 1)
        root.addWidget(vars_frame)

        # ── Code editor ───────────────────────────────────────────────────────
        code_hdr = QLabel("PYTHON CODE")
        code_hdr.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;font-weight:700;letter-spacing:1px;")
        root.addWidget(code_hdr)

        self.editor = QTextEdit()
        _code_font = QFont("Consolas", 11)
        _code_font.setPointSize(max(1, _code_font.pointSize()))
        self.editor.setFont(_code_font)
        self.editor.setObjectName("log_te")
        try:
            self.editor.setTabStopDistance(28.0)
        except Exception:
            pass
        self.editor.setPlaceholderText(
            "# Write Python here. Set 'result' to your output string.")
        saved_code = self._block.metadata.get("custom_code", "")
        self.editor.setPlainText(saved_code if saved_code else self._TEMPLATE)
        self.editor.textChanged.connect(self._on_edit)
        root.addWidget(self.editor, 1)

        # ── Syntax status ─────────────────────────────────────────────────────
        self.syntax_lbl = QLabel("Syntax OK")
        self.syntax_lbl.setStyleSheet(f"color:{C['ok']};font-size:11px;")
        root.addWidget(self.syntax_lbl)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        self.btn_test = QPushButton("Test with sample text...")
        set_button_icon(self.btn_test, "test-tube", "Test with sample text...")
        self.btn_test.setFixedHeight(30)
        self.btn_test.clicked.connect(self._run_test)
        btn_ok = QPushButton("Save & Close")
        set_button_icon(btn_ok, "save", "Save & Close")
        btn_ok.setObjectName("btn_send")
        btn_ok.setFixedHeight(30)
        btn_ok.clicked.connect(self._save_and_close)
        btn_cancel = QPushButton("Cancel")
        set_button_icon(btn_cancel, "x", "Cancel")
        btn_cancel.setFixedHeight(30)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_test)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        root.addLayout(btn_row)

        # Run initial syntax check
        self._on_edit()

    def _on_edit(self):
        code = self.editor.toPlainText()
        try:
            compile(code, "<custom_code>", "exec")
            self.syntax_lbl.setText("Syntax OK")
            self.syntax_lbl.setStyleSheet(f"color:{C['ok']};font-size:11px;")
        except SyntaxError as e:
            self.syntax_lbl.setText(
                f"Syntax error  line {e.lineno}: {e.msg}")
            self.syntax_lbl.setStyleSheet(f"color:{C['err']};font-size:11px;")

    def _run_test(self):
        code = self.editor.toPlainText()
        sample, ok = QInputDialog.getMultiLineText(
            self, "Test Code",
            "Enter sample text to pass as 'text':",
            "Hello from the pipeline test runner.")
        if not ok:
            return
        log_lines = []
        ns = {
            "text": sample,
            "result": sample,
            "metadata": dict(self._block.metadata),
            "log": lambda m: log_lines.append(str(m)),
            "__builtins__": {"len": len, "str": str, "int": int, "float": float,
                             "bool": bool, "list": list, "dict": dict, "tuple": tuple,
                             "range": range, "enumerate": enumerate, "zip": zip,
                             "map": map, "filter": filter, "sorted": sorted,
                             "min": min, "max": max, "sum": sum,
                             "abs": abs, "round": round, "print": lambda *a: log_lines.append(" ".join(str(x) for x in a)),
                             "isinstance": isinstance, "hasattr": hasattr,
                             "getattr": getattr, "setattr": setattr,
                             "repr": repr, "type": type,
                             "True": True, "False": False, "None": None},
        }
        try:
            exec(compile(code, "<custom_code>", "exec"), ns)
            out = str(ns.get("result", sample))
            log_s = "\n".join(log_lines) if log_lines else "(no log output)"
            QMessageBox.information(
                self, "Test Result",
                f"Log output:\n{log_s}\n\n"
                f"result  ({len(out)} chars):\n{out[:600]}"
                + ("…" if len(out) > 600 else ""))
        except Exception as e:
            QMessageBox.critical(self, "Runtime Error", str(e))

    def _save_and_close(self):
        code = self.editor.toPlainText()
        try:
            compile(code, "<custom_code>", "exec")
        except SyntaxError as e:
            QMessageBox.warning(
                self, "Syntax Error",
                f"Fix the syntax error before saving:\n\nLine {e.lineno}: {e.msg}")
            return
        self._block.metadata["custom_code"] = code
        preview = code.strip().splitlines()[0] if code.strip() else "code"
        # Strip comment lines for label
        for ln in code.strip().splitlines():
            stripped = ln.strip()
            if stripped and not stripped.startswith("#"):
                preview = stripped[:20]
                break
        self._block.label = f"⌥ {preview}"
        self.accept()


class McpServerEditorDialog(QDialog):
    """
    Configuration dialog for MCP_SERVER pipeline blocks.
    Lets the user enter an MCP server URL or command, test the connection,
    see available tools, and pick which tool the block will call.
    """

    def __init__(self, block: "PipelineBlock", parent=None):
        super().__init__(parent)
        self._block = block
        self._tester = None
        self._connected = False
        self._tools: list = []
        self.setWindowTitle(f"MCP Server - {block.label}")
        prepare_adaptive_window(self, 640, 520, min_width=500, min_height=400)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # ── Header ───────────────────────────────────────────────────────────
        hdr = QLabel("MCP Server Block")
        set_label_icon(hdr, "mcp", "MCP Server Block", 18)
        hdr.setStyleSheet(
            f"color:{C['txt']};font-size:14px;font-weight:bold;")
        root.addWidget(hdr)

        about = QLabel(
            "Connect to an MCP (Model Context Protocol) server to call external "
            "tools. Enter the server URL (SSE) or command (stdio), test the "
            "connection, then select which tool this block should invoke. "
            "The incoming pipeline text is passed as the tool's main argument.")
        about.setWordWrap(True)
        about.setStyleSheet(
            f"color:{C['txt2']};font-size:11px;"
            f"background:{C['bg2']};border-radius:6px;padding:10px 12px;"
            f"border-left:3px solid #22d3ee;")
        root.addWidget(about)

        # ── Server config ────────────────────────────────────────────────────
        srv_lbl = QLabel("SERVER CONFIGURATION")
        srv_lbl.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;font-weight:700;letter-spacing:1px;")
        root.addWidget(srv_lbl)

        srv_card = QFrame(); srv_card.setObjectName("tab_card")
        srv_l = QVBoxLayout(srv_card)
        srv_l.setContentsMargins(14, 12, 14, 12); srv_l.setSpacing(8)

        def _row(label: str, widget, width=100):
            r = QHBoxLayout(); r.setSpacing(8)
            l = QLabel(label); l.setFixedWidth(width)
            l.setObjectName("txt2"); l.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
            r.addWidget(l); r.addWidget(widget, 1); return r

        self.combo_transport = QComboBox()
        self.combo_transport.addItem("SSE   (HTTP endpoint)", "sse")
        self.combo_transport.addItem("stdio (local command)", "stdio")
        self.combo_transport.setFixedHeight(28)
        saved_transport = self._block.metadata.get("mcp_transport", "sse")
        self.combo_transport.setCurrentIndex(0 if saved_transport == "sse" else 1)
        srv_l.addLayout(_row("Transport:", self.combo_transport))

        self.edit_url = QLineEdit()
        self.edit_url.setPlaceholderText(
            "http://localhost:3000/sse   or   npx -y @modelcontextprotocol/server-filesystem /path")
        self.edit_url.setFixedHeight(28)
        self.edit_url.setText(self._block.metadata.get("mcp_url", ""))
        srv_l.addLayout(_row("URL / Command:", self.edit_url))

        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("Display name (e.g. 'Filesystem MCP')")
        self.edit_name.setFixedHeight(28)
        self.edit_name.setText(self._block.metadata.get("mcp_name", ""))
        srv_l.addLayout(_row("Name:", self.edit_name))

        root.addWidget(srv_card)

        # ── Authentication ───────────────────────────────────────────────────
        auth_lbl = QLabel("AUTHENTICATION  (optional)")
        auth_lbl.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;font-weight:700;letter-spacing:1px;")
        root.addWidget(auth_lbl)

        auth_card = QFrame(); auth_card.setObjectName("tab_card")
        auth_l = QVBoxLayout(auth_card)
        auth_l.setContentsMargins(14, 12, 14, 12); auth_l.setSpacing(8)

        auth_hint = QLabel(
            "If the server requires a token or API key, enter it below. "
            "For SSE servers it is sent as an Authorization header. "
            "For stdio servers it is passed as the MCP_AUTH_TOKEN env var.")
        auth_hint.setWordWrap(True)
        auth_hint.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
        auth_l.addWidget(auth_hint)

        self.edit_auth_token = QLineEdit()
        self.edit_auth_token.setPlaceholderText("API key or bearer token")
        self.edit_auth_token.setFixedHeight(28)
        self.edit_auth_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_auth_token.setText(self._block.metadata.get("mcp_auth_token", ""))
        auth_l.addLayout(_row("Token:", self.edit_auth_token))

        self.edit_auth_env = QTextEdit()
        self.edit_auth_env.setPlaceholderText(
            "KEY=value  (one per line)\n"
            "e.g. GITHUB_TOKEN=ghp_xxxx\n"
            "e.g. DATABASE_URL=postgres://...")
        self.edit_auth_env.setFixedHeight(68)
        self.edit_auth_env.setFont(QFont("Consolas", 10))
        self.edit_auth_env.setPlainText(self._block.metadata.get("mcp_auth_env", ""))
        auth_l.addWidget(self.edit_auth_env)

        root.addWidget(auth_card)

        # ── Test connection ──────────────────────────────────────────────────
        test_row = QHBoxLayout(); test_row.setSpacing(10)
        self.btn_test = QPushButton("Test Connection")
        set_button_icon(self.btn_test, "zap", "Test Connection")
        self.btn_test.setObjectName("btn_send")
        self.btn_test.setFixedHeight(30)
        self.btn_test.clicked.connect(self._test_connection)
        test_row.addWidget(self.btn_test)

        self.status_dot = QLabel("●")
        self.status_dot.setFixedWidth(20)
        self.status_dot.setStyleSheet(f"color:{C['txt3']};font-size:16px;")
        self.status_dot.setToolTip("Not tested")
        test_row.addWidget(self.status_dot)

        self.status_lbl = QLabel("Not tested")
        self.status_lbl.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        test_row.addWidget(self.status_lbl, 1)
        root.addLayout(test_row)

        # ── Tool selector ────────────────────────────────────────────────────
        tool_lbl = QLabel("TOOL SELECTION")
        tool_lbl.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;font-weight:700;letter-spacing:1px;")
        root.addWidget(tool_lbl)

        tool_card = QFrame(); tool_card.setObjectName("tab_card")
        tool_l = QVBoxLayout(tool_card)
        tool_l.setContentsMargins(14, 12, 14, 12); tool_l.setSpacing(8)

        self.combo_tool = QComboBox()
        self.combo_tool.setFixedHeight(28)
        self.combo_tool.addItem("Test connection first to see tools...", "")
        self.combo_tool.currentIndexChanged.connect(self._on_tool_selected)
        tool_l.addWidget(self.combo_tool)

        self.tool_desc = QLabel("")
        self.tool_desc.setWordWrap(True)
        self.tool_desc.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
        tool_l.addWidget(self.tool_desc)

        tool_input_row = QHBoxLayout(); tool_input_row.setSpacing(8)
        tool_input_lbl = QLabel("Input arg name:")
        tool_input_lbl.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        tool_input_lbl.setFixedWidth(110)
        self.edit_arg_name = QLineEdit()
        self.edit_arg_name.setPlaceholderText("auto-detect from schema")
        self.edit_arg_name.setFixedHeight(26)
        self.edit_arg_name.setText(self._block.metadata.get("mcp_arg_name", ""))
        tool_input_row.addWidget(tool_input_lbl)
        tool_input_row.addWidget(self.edit_arg_name, 1)
        tool_l.addLayout(tool_input_row)

        root.addWidget(tool_card)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        btn_save = QPushButton("Save & Close")
        set_button_icon(btn_save, "save", "Save & Close")
        btn_save.setObjectName("btn_send"); btn_save.setFixedHeight(32)
        btn_save.clicked.connect(self._save_and_close)
        btn_cancel = QPushButton("Cancel")
        set_button_icon(btn_cancel, "x", "Cancel")
        btn_cancel.setFixedHeight(32)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        root.addLayout(btn_row)

    def _set_status(self, ok: bool, msg: str):
        self._connected = ok
        color = C["ok"] if ok else C["err"]
        self.status_dot.setStyleSheet(f"color:{color};font-size:16px;")
        self.status_dot.setToolTip(msg)
        self.status_lbl.setText(msg[:80])

    def _test_connection(self):
        """Connect to the MCP server, list tools, populate the tool combo."""
        url = self.edit_url.text().strip()
        if not url:
            QMessageBox.warning(self, "Missing URL", "Enter a server URL or command.")
            return

        self.btn_test.setEnabled(False)
        self.btn_test.setText("Testing...")
        self.status_dot.setStyleSheet(f"color:{C['warn']};font-size:16px;")
        self.status_lbl.setText("Connecting...")

        transport = self.combo_transport.currentData()
        auth_token = self.edit_auth_token.text().strip()
        auth_env = self._parse_env_vars(self.edit_auth_env.toPlainText())

        try:
            from nativelab.integrations.mcp_client import McpClient
            client = McpClient()
            ok, tools = client.test_connection(
                transport, url,
                auth_token=auth_token or None,
                auth_env=auth_env or None,
            )
            client.shutdown()

            if ok:
                self._set_status(True, f"Connected — {len(tools)} tool(s) found")
                self._tools = tools
                self._populate_tools(tools)
            else:
                err = tools if isinstance(tools, str) else "Connection failed"
                self._set_status(False, err)
                self.combo_tool.clear()
                self.combo_tool.addItem("Connection failed", "")
        except ImportError as e:
            self._set_status(False, f"MCP client not available: {e}")
            self.combo_tool.clear()
            self.combo_tool.addItem("MCP client not installed", "")
        except Exception as e:
            self._set_status(False, str(e)[:100])
            self.combo_tool.clear()
            self.combo_tool.addItem("Error: " + str(e)[:60], "")
        finally:
            self.btn_test.setEnabled(True)
            self.btn_test.setText("Test Connection")

    def _populate_tools(self, tools: list):
        self.combo_tool.blockSignals(True)
        self.combo_tool.clear()
        saved_tool = self._block.metadata.get("mcp_tool_name", "")
        sel = 0
        for i, t in enumerate(tools):
            name = t.get("name", "unknown")
            desc = t.get("description", "")[:60]
            self.combo_tool.addItem(f"{name}  —  {desc}", name)
            if name == saved_tool:
                sel = i
        self.combo_tool.setCurrentIndex(sel)
        self.combo_tool.blockSignals(False)
        self._on_tool_selected()

    def _on_tool_selected(self):
        idx = self.combo_tool.currentIndex()
        if idx < 0 or idx >= len(self._tools):
            self.tool_desc.setText("")
            return
        t = self._tools[idx]
        desc = t.get("description", "No description")
        schema = t.get("inputSchema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])
        lines = [desc, ""]
        if props:
            lines.append("Parameters:")
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                lines.append(f"  • {pname} [{ptype}]{req}: {pdesc}")
        self.tool_desc.setText("\n".join(lines))

    @staticmethod
    def _parse_env_vars(text: str) -> dict:
        """Parse KEY=value lines into a dict, ignoring blanks and comments."""
        env = {}
        for line in str(text or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                if key:
                    env[key] = val
        return env

    def _save_and_close(self):
        url = self.edit_url.text().strip()
        if not url:
            QMessageBox.warning(self, "Missing URL", "Enter a server URL or command.")
            return

        transport = self.combo_transport.currentData()
        tool_name = self.combo_tool.currentData() or ""
        name = self.edit_name.text().strip()

        self._block.metadata["mcp_transport"] = transport
        self._block.metadata["mcp_url"] = url
        self._block.metadata["mcp_name"] = name
        self._block.metadata["mcp_tool_name"] = tool_name
        self._block.metadata["mcp_arg_name"] = self.edit_arg_name.text().strip()
        self._block.metadata["mcp_auth_token"] = self.edit_auth_token.text().strip()
        self._block.metadata["mcp_auth_env"] = self.edit_auth_env.toPlainText().strip()
        self._block.metadata["mcp_connected"] = self._connected
        self._block.metadata["mcp_tools"] = self._tools

        # Label
        display = name or tool_name or url[:24]
        self._block.label = f"MCP {display[:22]}"
        self.accept()


class WebSearchEditorDialog(QDialog):
    """
    Configuration dialog for WEB_SEARCH pipeline blocks.
    Uses SearXNG in-process to search the web.
    """

    AVAILABLE_CATEGORIES = [
        ("general", "General web search"),
        ("images", "Image search"),
        ("videos", "Video search"),
        ("news", "News search"),
        ("science", "Science"),
        ("it", "IT / Programming"),
        ("files", "File search"),
        ("music", "Music"),
        ("social media", "Social media"),
    ]

    def __init__(self, block: "PipelineBlock", parent=None):
        super().__init__(parent)
        self._block = block
        self.setWindowTitle(f"Web Search - {block.label}")
        prepare_adaptive_window(self, 680, 520, min_width=540, min_height=420)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # Header
        hdr = QLabel("Web Search Block")
        set_label_icon(hdr, "mcp", "Web Search Block", 18)
        hdr.setStyleSheet(f"color:{C['txt']};font-size:14px;font-weight:bold;")
        root.addWidget(hdr)

        about = QLabel(
            "Search the web using SearXNG. The incoming text is used as the "
            "search query. Results are returned as formatted text and passed "
            "to the next block.")
        about.setWordWrap(True)
        about.setStyleSheet(
            f"color:{C['txt2']};font-size:11px;"
            f"background:{C['bg2']};border-radius:6px;padding:10px 12px;"
            f"border-left:3px solid #f97316;")
        root.addWidget(about)

        # Settings card
        card = QFrame(); card.setObjectName("tab_card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(14, 12, 14, 12); cl.setSpacing(8)

        def _row(label, widget, width=100):
            r = QHBoxLayout(); r.setSpacing(8)
            l = QLabel(label); l.setFixedWidth(width)
            l.setObjectName("txt2"); l.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
            r.addWidget(l); r.addWidget(widget, 1); return r

        # Categories
        cat_lbl = QLabel("Categories (check to include):")
        cat_lbl.setStyleSheet(f"color:{C['txt3']};font-size:9px;font-weight:700;")
        cl.addWidget(cat_lbl)

        saved_cats = self._block.metadata.get("ws_categories", ["general"])
        self._cat_checks = {}
        cat_grid = QWidget()
        cat_gl = QHBoxLayout(cat_grid)
        cat_gl.setContentsMargins(0, 0, 0, 0); cat_gl.setSpacing(6)
        col = QVBoxLayout(); col.setSpacing(2)
        for i, (cat_id, cat_desc) in enumerate(self.AVAILABLE_CATEGORIES):
            cb = QCheckBox(cat_desc)
            cb.setChecked(cat_id in saved_cats)
            self._cat_checks[cat_id] = cb
            col.addWidget(cb)
            if (i + 1) % 5 == 0:
                cat_gl.addLayout(col)
                col = QVBoxLayout(); col.setSpacing(2)
        cat_gl.addLayout(col)
        cat_gl.addStretch()
        cl.addWidget(cat_grid)

        # Language
        self.edit_lang = QLineEdit()
        self.edit_lang.setPlaceholderText("en")
        self.edit_lang.setFixedHeight(26)
        self.edit_lang.setText(self._block.metadata.get("ws_language", "en"))
        cl.addLayout(_row("Language:", self.edit_lang))

        # Max results
        self.spin_max = QSpinBox()
        self.spin_max.setRange(1, 50)
        self.spin_max.setValue(int(self._block.metadata.get("ws_max_results", 10)))
        self.spin_max.setFixedHeight(26); self.spin_max.setFixedWidth(80)
        cl.addLayout(_row("Max results:", self.spin_max))

        # Timeout
        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(3, 30)
        self.spin_timeout.setValue(int(self._block.metadata.get("ws_timeout", 10)))
        self.spin_timeout.setFixedHeight(26); self.spin_timeout.setFixedWidth(80)
        self.spin_timeout.setSuffix(" sec")
        cl.addLayout(_row("Timeout:", self.spin_timeout))

        # Output format
        self.combo_format = QComboBox()
        self.combo_format.addItem("Formatted text (for context injection)", "text")
        self.combo_format.addItem("JSON (structured data)", "json")
        self.combo_format.setFixedHeight(26)
        saved_fmt = self._block.metadata.get("ws_output_format", "text")
        self.combo_format.setCurrentIndex(0 if saved_fmt == "text" else 1)
        cl.addLayout(_row("Output format:", self.combo_format))

        root.addWidget(card)

        # Test
        test_row = QHBoxLayout(); test_row.setSpacing(10)
        self.btn_test = QPushButton("Test Search")
        set_button_icon(self.btn_test, "zap", "Test Search")
        self.btn_test.setObjectName("btn_send")
        self.btn_test.setFixedHeight(30)
        self.btn_test.clicked.connect(self._test_search)
        test_row.addWidget(self.btn_test)
        self.test_status = QLabel("")
        self.test_status.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        test_row.addWidget(self.test_status, 1)
        root.addLayout(test_row)

        # Buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        btn_save = QPushButton("Save & Close")
        set_button_icon(btn_save, "save", "Save & Close")
        btn_save.setObjectName("btn_send"); btn_save.setFixedHeight(32)
        btn_save.clicked.connect(self._save_and_close)
        btn_cancel = QPushButton("Cancel")
        set_button_icon(btn_cancel, "x", "Cancel")
        btn_cancel.setFixedHeight(32)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        root.addLayout(btn_row)

    def _get_categories(self) -> list[str]:
        return [cat_id for cat_id, cb in self._cat_checks.items() if cb.isChecked()]

    def _test_search(self):
        categories = self._get_categories()
        if not categories:
            QMessageBox.warning(self, "No Categories", "Select at least one category.")
            return

        self.btn_test.setEnabled(False)
        self.btn_test.setText("Searching...")
        self.test_status.setText("Running test search...")

        try:
            from nativelab.web_search import web_search
            results = web_search(
                "python programming",
                categories=categories,
                language=self.edit_lang.text().strip() or "en",
                max_results=3,
                timeout=self.spin_timeout.value(),
            )
            if results:
                self.test_status.setText(f"✓ {len(results)} results found")
                self.test_status.setStyleSheet(f"color:{C['ok']};font-size:11px;")
            else:
                self.test_status.setText("No results (check network)")
                self.test_status.setStyleSheet(f"color:{C['warn']};font-size:11px;")
        except ImportError as e:
            self.test_status.setText(f"SearXNG not installed: {str(e)[:50]}")
            self.test_status.setStyleSheet(f"color:{C['err']};font-size:11px;")
        except Exception as e:
            self.test_status.setText(f"Error: {str(e)[:60]}")
            self.test_status.setStyleSheet(f"color:{C['err']};font-size:11px;")
        finally:
            self.btn_test.setEnabled(True)
            self.btn_test.setText("Test Search")

    def _save_and_close(self):
        categories = self._get_categories()
        if not categories:
            QMessageBox.warning(self, "No Categories", "Select at least one category.")
            return

        self._block.metadata["ws_categories"] = categories
        self._block.metadata["ws_language"] = self.edit_lang.text().strip() or "en"
        self._block.metadata["ws_max_results"] = self.spin_max.value()
        self._block.metadata["ws_timeout"] = self.spin_timeout.value()
        self._block.metadata["ws_output_format"] = self.combo_format.currentData()

        cats_str = ", ".join(categories[:3])
        self._block.label = f"Web: {cats_str[:18]}"
        self.accept()
