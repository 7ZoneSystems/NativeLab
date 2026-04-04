from nativelab.imports.import_global import QInputDialog,QDialog, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QPushButton, QFileDialog, QTextEdit, QFont, QFrame, QSpinBox, QCheckBox, QMessageBox, Qt, Path
from nativelab.Model.model_global import get_model_registry, detect_model_family
from nativelab.GlobalConfig.config_global import ROLE_ICONS
from .pipblck import PipelineBlock
from nativelab.UI.UI_const import C
class LlmLogicEditorDialog(QDialog if hasattr(__builtins__, '__import__') else object):
    """
    Configuration dialog for LLM-backed logic blocks.
    The user writes conditions and instructions in plain English.
    The attached model evaluates them at runtime against the incoming text.
    """

    # Per-type documentation shown in the dialog
    _TYPE_INFO = {
        "llm_if": {
            "icon":  "🧠 LLM IF / ELSE",
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
            "icon":  "🧠 LLM SWITCH",
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
            "icon":  "🧠 LLM FILTER",
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
            "icon":  "🧠 LLM TRANSFORM",
            "about": (
                "The model rewrites or transforms the incoming text according to your "
                "instruction. The result replaces the context for all downstream blocks. "
                "Be specific — the model will follow your instruction precisely."
            ),
            "label":       "Transformation instruction:",
            "placeholder": "e.g.  Summarise this in three bullet points\n"
                           "e.g.  Rewrite in a formal professional tone\n"
                           "e.g.  Extract only the action items as a numbered list\n"
                           "e.g.  Translate to Spanish",
            "branch_hint": "Single output — transformed text flows to all connected blocks.",
        },
        "llm_score": {
            "icon":  "🧠 LLM SCORE",
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
        try:
            from PyQt6.QtWidgets import QDialog
            super().__init__(parent)
        except Exception:
            return
        self._block = block
        info = self._TYPE_INFO.get(block.btype, {})
        self.setWindowTitle(f"{info.get('icon','🧠')} — {block.label}")
        self.setMinimumSize(640, 480)
        self.resize(700, 530)
        self._build(info)

    def _build(self, info: dict):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # ── Header ───────────────────────────────────────────────────────────
        hdr = QLabel(info.get("icon", "🧠  LLM Logic Block"))
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
        hint_lbl = QLabel(f"📌  {info.get('branch_hint','')}")
        hint_lbl.setStyleSheet(
            f"color:{C['ok']};font-size:10px;font-weight:600;"
            f"padding:4px 8px;background:{C['bg2']};border-radius:4px;")
        hint_lbl.setWordWrap(True)
        root.addWidget(hint_lbl)

        # ── Model selector ────────────────────────────────────────────────────
        model_lbl = QLabel("MODEL  (required — evaluated at runtime):")
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
                f"{ROLE_ICONS.get(_m.get('role','general'),'💬')}  {_m['name']}",
                _m["path"])
            if _m["path"] == _cur_path:
                _sel_idx = _i
        if _models:
            self.model_combo.setCurrentIndex(_sel_idx)
        else:
            self.model_combo.addItem("⚠️  No models registered — add one in Models tab", "")

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

        self.check_show_reasoning = QCheckBox("Show model reasoning in log")
        self.check_show_reasoning.setChecked(
            bool(self._block.metadata.get("llm_show_reasoning", True)))
        self.check_show_reasoning.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        adv_l.addWidget(self.check_show_reasoning)

        self.check_passthrough_on_err = QCheckBox(
            "Pass text through unchanged if model call fails (instead of stopping pipeline)")
        self.check_passthrough_on_err.setChecked(
            bool(self._block.metadata.get("llm_passthrough_on_err", False)))
        self.check_passthrough_on_err.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        adv_l.addWidget(self.check_passthrough_on_err)

        root.addWidget(self._adv_frame)
        adv_hdr.mousePressEvent = lambda _: self._toggle_adv(adv_hdr)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        btn_save = QPushButton("💾  Save & Close")
        btn_save.setObjectName("btn_send"); btn_save.setFixedHeight(32)
        btn_save.clicked.connect(self._save_and_close)
        btn_cancel = QPushButton("✕  Cancel")
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
        ok = bool(path) and Path(path).exists()
        self.model_status.setText("✅" if ok else "❌")
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
                self.model_combo.addItem(f"💬  {Path(path).name}", path)
                already = self.model_combo.count() - 1
            self.model_combo.setCurrentIndex(already)

    def _save_and_close(self):
        instr = self.instr_edit.toPlainText().strip()
        if not instr:
            QMessageBox.warning(self, "Missing Instruction",
                                "Please enter a condition or instruction."); return
        model_path = self.model_combo.currentData() or ""
        if not model_path or not Path(model_path).exists():
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
            "llm_if": "🧠⑂", "llm_switch": "🧠⑃",
            "llm_filter": "🧠⊘", "llm_transform": "🧠⟲", "llm_score": "🧠★",
        }
        _icon = _icon_map.get(self._block.btype, "🧠")
        self._block.label = f"{_icon} {_preview}"
        self.accept()

class CodeEditorDialog(QDialog if hasattr(__builtins__, '__import__') else object):
    """
    Full code editor for CUSTOM_CODE pipeline blocks.
    Shows available variables, validates syntax live, saves to block.metadata.
    """

    # Available variable documentation shown to user
    _VAR_DOCS = [
        ("text",    "str",  "The incoming context string from the previous block."),
        ("result",  "str",  "Write your output here — the pipeline continues with this."),
        ("metadata","dict", "Block metadata dict (read/write — persists across runs)."),
        ("log",     "fn",   "log('message') — writes to the pipeline execution log."),
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
        try:            
            super().__init__(parent)
        except Exception:
            return
        self._block = block
        self.setWindowTitle(f"⌥ Code Editor — {block.label}")
        self.setMinimumSize(760, 560)
        self.resize(820, 620)
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
        self.syntax_lbl = QLabel("✅  Syntax OK")
        self.syntax_lbl.setStyleSheet(f"color:{C['ok']};font-size:11px;")
        root.addWidget(self.syntax_lbl)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        self.btn_test = QPushButton("🧪  Test with sample text…")
        self.btn_test.setFixedHeight(30)
        self.btn_test.clicked.connect(self._run_test)
        btn_ok = QPushButton("💾  Save & Close")
        btn_ok.setObjectName("btn_send")
        btn_ok.setFixedHeight(30)
        btn_ok.clicked.connect(self._save_and_close)
        btn_cancel = QPushButton("✕  Cancel")
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
            self.syntax_lbl.setText("✅  Syntax OK")
            self.syntax_lbl.setStyleSheet(f"color:{C['ok']};font-size:11px;")
        except SyntaxError as e:
            self.syntax_lbl.setText(
                f"❌  Syntax error  line {e.lineno}: {e.msg}")
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
                self, "✅  Test Result",
                f"Log output:\n{log_s}\n\n"
                f"result  ({len(out)} chars):\n{out[:600]}"
                + ("…" if len(out) > 600 else ""))
        except Exception as e:
            QMessageBox.critical(self, "❌  Runtime Error", str(e))

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
