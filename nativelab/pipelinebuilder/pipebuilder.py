from nativelab.Model.ModelRegistry import get_model_registry
from nativelab.imports.import_global import QInputDialog,QMessageBox,datetime,QListWidgetItem,QApplication,QDialog, Path,Dict,Optional,QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QTextEdit, QFont, QFrame, QTabWidget, QScrollArea, QListWidget, QAbstractItemView, QWidget, QTimer, Qt
from .pipefunctions import list_saved_pipelines, load_pipeline, save_pipeline
from .blck_typ import PipelineBlockType 
from nativelab.core.engine_global import LlamaEngine
from .executionWorker import PipelineExecutionWorker
from nativelab.UI.UI_const import C
from .canvas import PipelineCanvas
from .outrender import PipelineOutputRenderer
from manual import make_manual_html, PIPELINE_MANUAL_HTML
from nativelab.GlobalConfig.config_global import ROLE_ICONS, PIPELINES_DIR
class PipelineBuilderTab(QWidget):
    """
    Full-featured pipeline builder tab.
    Left sidebar : block adders (flow + context) + model list.
    Centre       : interactive canvas.
    Right        : execution controls + per-intermediate live panes + final output.
    """

    def __init__(self, engine: "LlamaEngine", parent=None):
        super().__init__(parent)
        self._engine                = engine
        self._exec_worker: Optional[PipelineExecutionWorker] = None
        self._inter_tabs:  Dict[int, int] = {}   # block_id → tab index
        self._current_pipeline_name: str = ""
        self._build()

    def update_engine(self, engine: "LlamaEngine"):
        self._engine = engine

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ══════ LEFT SIDEBAR ══════════════════════════════════════════════════
        sb = QWidget()
        sb.setObjectName("session_sidebar")
        sb_l = QVBoxLayout(sb)
        sb_l.setContentsMargins(10, 14, 10, 14)
        sb_l.setSpacing(7)

        def _sec(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color:{C['txt3']};font-size:9px;"
                f"font-weight:700;letter-spacing:1.1px;")
            return lbl

        def _block_btn(icon_label: str, btype: str, color: str) -> QPushButton:
            btn = QPushButton(icon_label)
            btn.setFixedHeight(30)
            btn.setStyleSheet(
                f"QPushButton{{background:transparent;"
                f"color:{color};border:1px solid {color};"
                f"border-radius:6px;font-size:11px;font-weight:600;}}"
                f"QPushButton:hover{{background:{color};color:#fff;}}")
            btn.clicked.connect(lambda _, bt=btype: self._add_block(bt))
            return btn

        hdr = QLabel("🔗  Pipeline Builder")
        hdr.setStyleSheet(
            f"color:{C['txt']};font-size:12px;font-weight:700;")
        sb_l.addWidget(hdr)

        sep0 = QFrame(); sep0.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep0)

        sb_l.addWidget(_sec("FLOW BLOCKS"))
        sb_l.addWidget(_block_btn("▶  Input Block",          PipelineBlockType.INPUT,        C["ok"]))
        sb_l.addWidget(_block_btn("◈  Intermediate Block",   PipelineBlockType.INTERMEDIATE, C["warn"]))
        sb_l.addWidget(_block_btn("■  Output Block",         PipelineBlockType.OUTPUT,       C["err"]))

        sep_ctx = QFrame(); sep_ctx.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_ctx)
        sb_l.addWidget(_sec("CONTEXT BLOCKS"))
        sb_l.addWidget(_block_btn("📎  Reference",   PipelineBlockType.REFERENCE,   C["acc"]))
        sb_l.addWidget(_block_btn("💡  Knowledge",   PipelineBlockType.KNOWLEDGE,   C["acc2"]))
        sb_l.addWidget(_block_btn("📄  PDF Summary", PipelineBlockType.PDF_SUMMARY, C["pipeline"]))

        sep_logic = QFrame(); sep_logic.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_logic)
        sb_l.addWidget(_sec("LOGIC BLOCKS"))
        sb_l.addWidget(_block_btn("⑂  IF / ELSE",   PipelineBlockType.IF_ELSE,     "#f59e0b"))
        sb_l.addWidget(_block_btn("⑃  SWITCH",      PipelineBlockType.SWITCH,      "#f97316"))
        sb_l.addWidget(_block_btn("⊘  FILTER",      PipelineBlockType.FILTER,      "#84cc16"))
        sb_l.addWidget(_block_btn("⟲  TRANSFORM",   PipelineBlockType.TRANSFORM,   "#06b6d4"))
        sb_l.addWidget(_block_btn("⊕  MERGE",       PipelineBlockType.MERGE,       "#8b5cf6"))
        sb_l.addWidget(_block_btn("⑁  SPLIT",       PipelineBlockType.SPLIT,       "#ec4899"))

        sep_code = QFrame(); sep_code.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_code)
        sb_l.addWidget(_sec("CUSTOM CODE"))
        sb_l.addWidget(_block_btn("⌥  Custom Code", PipelineBlockType.CUSTOM_CODE, "#10b981"))

        sep_llm = QFrame(); sep_llm.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_llm)
        sb_l.addWidget(_sec("LLM LOGIC  (natural language)"))
        _llm_note = QLabel(
            "Conditions & instructions written\n"
            "in plain English — evaluated by\n"
            "the block's attached LLM model.")
        _llm_note.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;"
            f"padding:4px 2px;line-height:1.5;")
        sb_l.addWidget(_llm_note)
        sb_l.addWidget(_block_btn("🧠  LLM IF / ELSE",   PipelineBlockType.LLM_IF,        "#a855f7"))
        sb_l.addWidget(_block_btn("🧠  LLM SWITCH",      PipelineBlockType.LLM_SWITCH,     "#7c3aed"))
        sb_l.addWidget(_block_btn("🧠  LLM FILTER",      PipelineBlockType.LLM_FILTER,     "#6366f1"))
        sb_l.addWidget(_block_btn("🧠  LLM TRANSFORM",   PipelineBlockType.LLM_TRANSFORM,  "#0ea5e9"))
        sb_l.addWidget(_block_btn("🧠  LLM SCORE",       PipelineBlockType.LLM_SCORE,      "#d946ef"))

        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep1)

        sb_l.addWidget(_sec("MODELS  (dbl-click to add)"))
        self.model_list = QListWidget()
        self.model_list.setObjectName("model_list")
        self.model_list.itemDoubleClicked.connect(self._add_model_from_list)
        self.model_list.setDragEnabled(True)
        self.model_list.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.model_list.setToolTip(
            "Double-click OR drag onto the canvas to add a model block.")
        sb_l.addWidget(self.model_list, 1)

        self.btn_refresh = QPushButton("↻  Refresh")
        self.btn_refresh.setFixedHeight(26)
        self.btn_refresh.clicked.connect(self._refresh_models)
        sb_l.addWidget(self.btn_refresh)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep2)

        sb_l.addWidget(_sec("CANVAS CONTROLS"))
        self.btn_clear_canvas = QPushButton("🗑  Clear All")
        self.btn_clear_canvas.setFixedHeight(28)
        self.btn_clear_canvas.clicked.connect(self._clear_canvas)
        sb_l.addWidget(self.btn_clear_canvas)

        self.btn_save_pipeline = QPushButton("💾  Save Pipeline…")
        self.btn_save_pipeline.setFixedHeight(28)
        self.btn_save_pipeline.clicked.connect(self._save_pipeline)
        sb_l.addWidget(self.btn_save_pipeline)

        self.btn_load_pipeline = QPushButton("📂  Load Pipeline…")
        self.btn_load_pipeline.setFixedHeight(28)
        self.btn_load_pipeline.clicked.connect(self._load_pipeline)
        sb_l.addWidget(self.btn_load_pipeline)

        hint = QLabel(
            "Tip: draw a connection from a\n"
            "port dot (N·S·E·W) on one block\n"
            "to a port dot on another block.\n\n"
            "Model→Model requires an\n"
            "Intermediate block in between.")
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;"
            f"padding:6px 4px;line-height:1.6;")
        sb_l.addWidget(hint)

        sb_l.addStretch()

        sb_scroll = QScrollArea()
        sb_scroll.setWidget(sb)
        sb_scroll.setWidgetResizable(True)
        sb_scroll.setFixedWidth(214)
        sb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sb_scroll.setObjectName("session_sidebar")
        sb_scroll.setStyleSheet("QScrollArea#session_sidebar { border: none; }")
        root.addWidget(sb_scroll)

        # ══════ CENTRE: CANVAS ════════════════════════════════════════════════
        centre = QWidget()
        centre_l = QVBoxLayout(centre)
        centre_l.setContentsMargins(0, 0, 0, 0)
        centre_l.setSpacing(0)

        # toolbar
        toolbar = QWidget()
        toolbar.setObjectName("appearance_bar")
        toolbar.setFixedHeight(40)
        tb_l = QHBoxLayout(toolbar)
        tb_l.setContentsMargins(12, 4, 12, 4)
        tb_l.setSpacing(8)
        tb_title = QLabel("🔗  Pipeline Canvas")
        tb_title.setObjectName("appearance_hdr")
        tb_l.addWidget(tb_title)
        tb_l.addStretch()
        legend_items = [
            (C["ok"],       "▶ Input"),
            (C["warn"],     "◈ Intermediate"),
            (C["err"],      "■ Output"),
            (C["acc"],      "⚡ Model"),
            ("#f59e0b",     "⑂ IF/ELSE"),
            ("#06b6d4",     "⟲ Transform"),
            ("#10b981",     "⌥ Code"),
            ("#8b5cf6",     "⊕ Merge"),
            ("#ec4899",     "⑁ Split"),
            ("#a855f7",     "🧠 LLM-IF"),
            ("#7c3aed",     "🧠 LLM-SW"),
            ("#0ea5e9",     "🧠 LLM-TX"),
            ("#d946ef",     "🧠 LLM-SC"),
            (C["pipeline"], "⤵ Loop"),
            (C["acc2"],     "→ Forward"),
        ]
        for col, txt in legend_items:
            lbl = QLabel(txt)
            lbl.setStyleSheet(
                f"color:{col};font-size:10px;font-weight:600;"
                f"padding:1px 6px;"
                f"background:{C['bg2']};border-radius:4px;")
            tb_l.addWidget(lbl)
        centre_l.addWidget(toolbar)

        pill_outer = QWidget()
        pill_outer.setObjectName("appearance_bar")
        pill_outer.setFixedHeight(36)
        pill_outer_l = QHBoxLayout(pill_outer)
        pill_outer_l.setContentsMargins(8, 3, 8, 3)
        pill_outer_l.setSpacing(0)

        _pill_icon = QLabel("⚡")
        _pill_icon.setStyleSheet(f"color:{C['txt3']};font-size:10px;padding-right:4px;")
        pill_outer_l.addWidget(_pill_icon)

        self._pill_scroll = QScrollArea()
        self._pill_scroll.setWidgetResizable(True)
        self._pill_scroll.setFixedHeight(30)
        self._pill_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._pill_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._pill_scroll.setStyleSheet(
            "QScrollArea { border:none; background:transparent; }"
            "QScrollBar:horizontal { height:4px; }")

        self._pill_container = QWidget()
        self._pill_container.setStyleSheet("background:transparent;")
        self._pill_layout = QHBoxLayout(self._pill_container)
        self._pill_layout.setContentsMargins(0, 0, 0, 0)
        self._pill_layout.setSpacing(5)
        self._pill_layout.addStretch()
        self._pill_scroll.setWidget(self._pill_container)
        pill_outer_l.addWidget(self._pill_scroll, 1)
        centre_l.addWidget(pill_outer)

        canvas_scroll = QScrollArea()
        canvas_scroll.setWidgetResizable(False)
        canvas_scroll.setObjectName("chat_scroll")
        self.canvas = PipelineCanvas()
        self.canvas.blocks_changed.connect(self._on_blocks_changed)
        canvas_scroll.setWidget(self.canvas)
        centre_l.addWidget(canvas_scroll, 1)

        root.addWidget(centre, 1)

        # ══════ RIGHT: EXECUTION PANEL ════════════════════════════════════════
        rp = QWidget()
        rp.setFixedWidth(282)
        rp.setObjectName("ref_panel")
        rp_l = QVBoxLayout(rp)
        rp_l.setContentsMargins(12, 14, 12, 14)
        rp_l.setSpacing(7)

        exec_hdr_row = QHBoxLayout(); exec_hdr_row.setSpacing(6)
        exec_hdr = QLabel("▶  Execute Pipeline")
        exec_hdr.setStyleSheet(
            f"color:{C['txt']};font-size:12px;font-weight:700;")
        exec_hdr_row.addWidget(exec_hdr, 1)
        btn_manual = QPushButton("📖 Manual")
        btn_manual.setFixedHeight(24)
        btn_manual.setStyleSheet(
            f"QPushButton{{background:transparent;color:{C['acc']};"
            f"border:1px solid {C['acc']};border-radius:5px;"
            f"font-size:10px;font-weight:600;padding:0 8px;}}"
            f"QPushButton:hover{{background:{C['acc']};color:#fff;}}")
        btn_manual.clicked.connect(self._show_manual)
        exec_hdr_row.addWidget(btn_manual)
        rp_l.addLayout(exec_hdr_row)

        # Server status badge (auto-refreshed every 2.5 s)
        self.server_badge = QLabel("⚪  Engine status unknown")
        self.server_badge.setStyleSheet(
            f"color:{C['txt3']};font-size:10px;padding:2px 7px;"
            f"background:{C['bg2']};border-radius:4px;")
        rp_l.addWidget(self.server_badge)
        self._badge_timer = QTimer(self)
        self._badge_timer.timeout.connect(self._update_server_badge)
        self._badge_timer.start(2500)

        input_lbl = QLabel("Input text:")
        input_lbl.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        rp_l.addWidget(input_lbl)

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText(
            "Enter the prompt or text to feed\n"
            "into the first INPUT block…")
        self.input_edit.setMaximumHeight(100)
        rp_l.addWidget(self.input_edit)

        self.btn_run = QPushButton("▶  Run Pipeline")
        self.btn_run.setObjectName("btn_send")
        self.btn_run.setFixedHeight(34)
        self.btn_run.clicked.connect(self._run_pipeline)
        rp_l.addWidget(self.btn_run)

        self.btn_stop = QPushButton("⏹  Stop Execution")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(34)
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self._stop_pipeline)
        rp_l.addWidget(self.btn_stop)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        rp_l.addWidget(sep3)

        # Tabbed output: 📋 Log | ■ Output | one ◈ tab per intermediate block
        self.output_tabs = QTabWidget()
        self.output_tabs.setObjectName("ref_tabs")

        self.exec_log = QTextEdit()
        self.exec_log.setReadOnly(True)
        self.exec_log.setFont(QFont("Consolas", 9))
        self.exec_log.setObjectName("log_te")
        self.output_tabs.addTab(self.exec_log, "📋 Log")

        self.output_edit = PipelineOutputRenderer(
            placeholder="Final pipeline output appears here…")
        self.output_tabs.addTab(self.output_edit, "■ Output")
        rp_l.addWidget(self.output_tabs, 1)

        copy_btn = QPushButton("⧉  Copy Final Output")
        copy_btn.setFixedHeight(28)
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(
                self.output_edit.raw_text()))
        rp_l.addWidget(copy_btn)

        root.addWidget(rp)

        self._refresh_models()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _refresh_models(self):
        """Show the pipeline builder manual in a scrollable dialog."""
        dlg = QDialog(self)
        dlg.setWindowTitle("📖  Pipeline Builder Manual")
        dlg.setMinimumSize(680, 580)
        dlg.resize(740, 640)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)

        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("Inter", 11))
        te.setObjectName("chat_te")
        te.setHtml(PIPELINE_MANUAL_HTML)
        lay.addWidget(te, 1)

        btn_close = QPushButton("✕  Close")
        btn_close.setFixedHeight(32)
        btn_close.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(12, 8, 12, 12)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)
        dlg.exec()

    def _show_manual(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("📖  Pipeline Builder Manual")
        dlg.setMinimumSize(680, 580)
        dlg.resize(740, 640)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("Inter", 11))
        te.setObjectName("chat_te")
        te.setHtml(make_manual_html())
        lay.addWidget(te, 1)
        btn_close = QPushButton("✕  Close")
        btn_close.setFixedHeight(32)
        btn_close.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(12, 8, 12, 12)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)
        dlg.exec()

    def _refresh_models(self):
        self.model_list.clear()
        for m in get_model_registry().all_models():
            ri   = ROLE_ICONS.get(m.get("role", "general"), "💬")
            qt   = m.get("quant", "?")
            fam  = m.get("family", "?")
            item = QListWidgetItem(f"{ri}  {m['name']}\n    {fam} · {qt}")
            item.setData(Qt.ItemDataRole.UserRole,     m["path"])
            item.setData(Qt.ItemDataRole.UserRole + 1, m.get("role", "general"))
            self.model_list.addItem(item)

    # ── pipeline save / load ──────────────────────────────────────────────────

    def _save_pipeline(self):
        if not self.canvas.blocks:
            QMessageBox.warning(self, "Empty Pipeline",
                                "Add blocks before saving."); return
        saved = list_saved_pipelines()
        name, ok = QInputDialog.getText(
            self, "Save Pipeline",
            "Pipeline name:\n(Existing names will be overwritten)",
            text=self._current_pipeline_name or "")
        if not ok or not name.strip():
            return
        name = name.strip().replace("/", "-").replace("\\", "-")
        save_pipeline(name, self.canvas.blocks, self.canvas.connections)
        self._current_pipeline_name = name
        self._log(f"💾  Pipeline saved as '{name}'")
        QMessageBox.information(self, "Saved",
                                f"Pipeline '{name}' saved successfully.")

    def _load_pipeline(self):
        saved = list_saved_pipelines()
        if not saved:
            QMessageBox.information(self, "No Pipelines",
                                    "No saved pipelines found.\n"
                                    f"Pipelines are stored in:\n{PIPELINES_DIR}")
            return
        # Offer saved names + a delete option
        items = saved + ["─────────────", "🗑  Delete a pipeline…"]
        choice, ok = QInputDialog.getItem(
            self, "Load Pipeline", "Select a pipeline:", items, 0, False)
        if not ok:
            return
        if choice == "🗑  Delete a pipeline…":
            self._delete_pipeline_dialog(saved); return
        if choice.startswith("─"):
            return
        if self.canvas.blocks:
            ans = QMessageBox.question(
                self, "Replace Canvas",
                f"Load '{choice}' and replace the current canvas?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return
        try:
            blocks, conns = load_pipeline(choice)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e)); return
        self.canvas.clear_all()
        while self.output_tabs.count() > 2:
            self.output_tabs.removeTab(2)
        self._inter_tabs.clear()
        for b in blocks:
            self.canvas.blocks.append(b)
        self.canvas.connections = conns
        self.canvas.update()
        self.canvas.blocks_changed.emit()
        self._current_pipeline_name = choice
        self._log(f"📂  Pipeline '{choice}' loaded "
                  f"({len(blocks)} blocks, {len(conns)} connections)")

    def _delete_pipeline_dialog(self, saved: list):
        choice, ok = QInputDialog.getItem(
            self, "Delete Pipeline", "Pipeline to delete:", saved, 0, False)
        if not ok:
            return
        ans = QMessageBox.question(
            self, "Confirm Delete",
            f"Permanently delete '{choice}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ans == QMessageBox.StandardButton.Yes:
            (PIPELINES_DIR / f"{choice}.json").unlink(missing_ok=True)
            self._log(f"🗑  Pipeline '{choice}' deleted.")

    def _update_server_badge(self):
        eng = self._engine
        if not eng or not eng.is_loaded:
            self.server_badge.setText("⚪  No model loaded")
            self.server_badge.setStyleSheet(
                f"color:{C['txt3']};font-size:10px;padding:2px 7px;"
                f"background:{C['bg2']};border-radius:4px;")
        elif eng.mode == "server":
            self.server_badge.setText(f"🟢  Server · port {eng.server_port}")
            self.server_badge.setStyleSheet(
                f"color:{C['ok']};font-size:10px;padding:2px 7px;"
                f"background:{C['bg2']};border-radius:4px;")
        else:
            self.server_badge.setText("🟡  CLI mode — will switch on run")
            self.server_badge.setStyleSheet(
                f"color:{C['warn']};font-size:10px;padding:2px 7px;"
                f"background:{C['bg2']};border-radius:4px;")

    def _add_block(self, btype: str):
        count = sum(1 for b in self.canvas.blocks if b.btype == btype)
        x = 100 + count * 30
        y = 160 + (len(self.canvas.blocks) % 4) * 100
        b = self.canvas.add_block(btype, x=x, y=y)
        # Immediately prompt configuration for context blocks
        if btype in (PipelineBlockType.REFERENCE,
                     PipelineBlockType.KNOWLEDGE,
                     PipelineBlockType.PDF_SUMMARY):
            self.canvas._configure_context_block(b)
        # Immediately prompt configuration for logic blocks
        _LOGIC_BTYPES = {
            PipelineBlockType.IF_ELSE, PipelineBlockType.SWITCH,
            PipelineBlockType.FILTER,  PipelineBlockType.TRANSFORM,
            PipelineBlockType.MERGE,   PipelineBlockType.CUSTOM_CODE,
        }
        if btype in _LOGIC_BTYPES:
            self.canvas._configure_logic_block(b)
        # Immediately open LLM logic editor
        _LLM_BTYPES = {
            PipelineBlockType.LLM_IF, PipelineBlockType.LLM_SWITCH,
            PipelineBlockType.LLM_FILTER, PipelineBlockType.LLM_TRANSFORM,
            PipelineBlockType.LLM_SCORE,
        }
        if btype in _LLM_BTYPES:
            self.canvas._configure_llm_logic_block(b)

    def _add_model_from_list(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        role = item.data(Qt.ItemDataRole.UserRole + 1) or "general"
        if not path:
            return
        model_blocks = [b for b in self.canvas.blocks
                        if b.btype == PipelineBlockType.MODEL]
        x = 220 + len(model_blocks) * 180
        y = 260
        b = self.canvas.add_block(PipelineBlockType.MODEL, x=x, y=y,
                                   model_path=path, role=role)
        self._log(f"Added model: {b.label}  [{role}]")

    def _clear_canvas(self):
        if QMessageBox.question(
            self, "Clear Canvas",
            "Remove all blocks and connections from the pipeline?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            self.canvas.clear_all()
            # Remove intermediate pane tabs (keep Log [0] and Output [1])
            while self.output_tabs.count() > 2:
                self.output_tabs.removeTab(2)
            self._inter_tabs.clear()
            self.exec_log.clear()
            self.output_edit.clear()

    def _on_blocks_changed(self):
        """Sync intermediate live-output tabs and pill bar with current canvas blocks."""
        # ── Refresh pill bar ──────────────────────────────────────────────────
        _PILL_BTYPES = {
            PipelineBlockType.MODEL,
            PipelineBlockType.LLM_IF, PipelineBlockType.LLM_SWITCH,
            PipelineBlockType.LLM_FILTER, PipelineBlockType.LLM_TRANSFORM,
            PipelineBlockType.LLM_SCORE,
            PipelineBlockType.IF_ELSE, PipelineBlockType.SWITCH,
            PipelineBlockType.FILTER, PipelineBlockType.TRANSFORM,
            PipelineBlockType.CUSTOM_CODE,
        }
        _PILL_COLORS = {
            PipelineBlockType.MODEL:        C["acc"],
            PipelineBlockType.IF_ELSE:      "#f59e0b",
            PipelineBlockType.SWITCH:       "#f97316",
            PipelineBlockType.FILTER:       "#84cc16",
            PipelineBlockType.TRANSFORM:    "#06b6d4",
            PipelineBlockType.CUSTOM_CODE:  "#10b981",
            PipelineBlockType.LLM_IF:       "#a855f7",
            PipelineBlockType.LLM_SWITCH:   "#7c3aed",
            PipelineBlockType.LLM_FILTER:   "#6366f1",
            PipelineBlockType.LLM_TRANSFORM:"#0ea5e9",
            PipelineBlockType.LLM_SCORE:    "#d946ef",
        }
        # Clear old pills
        while self._pill_layout.count() > 1:  # keep final stretch
            item = self._pill_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Add one pill per relevant block
        _pill_blocks = [b for b in self.canvas.blocks if b.btype in _PILL_BTYPES]
        if not _pill_blocks:
            _empty = QLabel("No model or logic blocks yet — drag from sidebar or click buttons")
            _empty.setStyleSheet(f"color:{C['txt3']};font-size:9px;padding:2px 6px;")
            self._pill_layout.insertWidget(0, _empty)
        else:
            for _pb in _pill_blocks:
                _col = _PILL_COLORS.get(_pb.btype, C["acc"])
                _pill = QPushButton(_pb.label)
                _pill.setFixedHeight(22)
                _pill.setStyleSheet(
                    f"QPushButton{{background:transparent;color:{_col};"
                    f"border:1px solid {_col};border-radius:10px;"
                    f"font-size:10px;font-weight:600;padding:0 8px;}}"
                    f"QPushButton:hover{{background:{_col};color:#fff;}}")
                # Clicking the pill selects + centres the block on canvas
                def _make_jump(blk):
                    def _jump():
                        self.canvas._selected = blk
                        self.canvas.update()
                    return _jump
                _pill.clicked.connect(_make_jump(_pb))
                self._pill_layout.insertWidget(self._pill_layout.count() - 1, _pill)

        # ── Existing: sync intermediate live-output tabs ──────────────────────
        existing_inter_ids = {b.bid for b in self.canvas.blocks
                              if b.btype == PipelineBlockType.INTERMEDIATE}
        # Remove tabs for deleted intermediate blocks
        for bid in list(self._inter_tabs.keys()):
            if bid not in existing_inter_ids:
                idx = self._inter_tabs.pop(bid)
                if idx < self.output_tabs.count():
                    self.output_tabs.removeTab(idx)
                # Re-index remaining entries
                self._inter_tabs = {
                    b2: (i2 if i2 < idx else i2 - 1)
                    for b2, i2 in self._inter_tabs.items()
                }
        # Add tabs for new intermediate blocks
        for b in self.canvas.blocks:
            if b.btype == PipelineBlockType.INTERMEDIATE and b.bid not in self._inter_tabs:
                te = PipelineOutputRenderer(
                    placeholder=f"Live context arriving at ◈ '{b.label}' will appear here…")
                idx = self.output_tabs.addTab(te, f"◈ {b.label[:14]}")
                self._inter_tabs[b.bid] = idx

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.exec_log.append(
            f'<span style="color:{C["txt2"]}">[{ts}]</span> '
            f'<span style="color:{C["txt"]}">{msg}</span>')

    # ── execution ─────────────────────────────────────────────────────────────

    def _validate(self) -> Optional[str]:
        blocks = self.canvas.blocks
        if not blocks:
            return "Canvas is empty — add blocks first."
        if not any(b.btype == PipelineBlockType.INPUT for b in blocks):
            return "Pipeline needs at least one ▶ INPUT block."
        if not any(b.btype == PipelineBlockType.OUTPUT for b in blocks):
            return "Pipeline needs at least one ■ OUTPUT block."
        if not self.canvas.connections:
            return "No connections drawn. Connect the blocks with arrows."
        # Context blocks must be configured before running
        for b in blocks:
            meta = getattr(b, "metadata", {})
            if b.btype == PipelineBlockType.REFERENCE and not meta.get("ref_text"):
                return (f"Reference block '{b.label}' has no text.\n"
                        f"Right-click it → Configure block…")
            if b.btype == PipelineBlockType.KNOWLEDGE and not meta.get("knowledge_text"):
                return (f"Knowledge block '{b.label}' has no text.\n"
                        f"Right-click it → Configure block…")
            if b.btype == PipelineBlockType.PDF_SUMMARY and not meta.get("pdf_path"):
                return (f"PDF block '{b.label}' has no PDF selected.\n"
                        f"Right-click it → Configure block…")
            # Logic blocks
            if b.btype == PipelineBlockType.IF_ELSE and not meta.get("condition"):
                return (f"IF/ELSE block '{b.label}' has no condition set.\n"
                        f"Right-click it → Configure block…")
            if b.btype == PipelineBlockType.SWITCH and not meta.get("switch_expr"):
                return (f"SWITCH block '{b.label}' has no expression set.\n"
                        f"Right-click it → Configure block…")
            if b.btype == PipelineBlockType.FILTER and not meta.get("filter_cond"):
                return (f"FILTER block '{b.label}' has no condition set.\n"
                        f"Right-click it → Configure block…")
            if b.btype == PipelineBlockType.TRANSFORM and not meta.get("transform_type"):
                return (f"TRANSFORM block '{b.label}' has no transform type set.\n"
                        f"Right-click it → Configure block…")
            if b.btype == PipelineBlockType.CUSTOM_CODE and not meta.get("custom_code","").strip():
                return (f"Custom Code block '{b.label}' has no code.\n"
                        f"Right-click it → Configure block…")
            # LLM logic blocks
            _LLM_BTYPES_V = {
                PipelineBlockType.LLM_IF, PipelineBlockType.LLM_SWITCH,
                PipelineBlockType.LLM_FILTER, PipelineBlockType.LLM_TRANSFORM,
                PipelineBlockType.LLM_SCORE,
            }
            if b.btype in _LLM_BTYPES_V:
                if not meta.get("llm_instruction", "").strip():
                    return (f"LLM logic block '{b.label}' has no instruction.\n"
                            f"Right-click it → Configure block…")
                mp = b.model_path or meta.get("llm_model_path", "")
                if not mp or not Path(mp).exists():
                    return (f"LLM logic block '{b.label}' has no valid model attached.\n"
                            f"Right-click it → Configure block… and select a model.")
        # Model blocks must have a valid file
        for b in blocks:
            if b.btype == PipelineBlockType.MODEL:
                if not b.model_path or not Path(b.model_path).exists():
                    return (f"Model block '{b.label}' has no valid model file.\n"
                            f"Double-click a model in the sidebar to add it.")
        return None

    def _run_pipeline(self):
        err = self._validate()
        if err:
            QMessageBox.warning(self, "Invalid Pipeline", err); return

        text = self.input_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Input",
                                "Enter text in the Input field on the right."); return

        if not self._engine.is_loaded:
            QMessageBox.warning(self, "Engine Not Ready",
                                "Wait for the model to finish loading."); return

        self.exec_log.clear()
        self.output_edit.clear()
        # Reset intermediate live panes
        for bid, idx in self._inter_tabs.items():
            w = self.output_tabs.widget(idx)
            if isinstance(w, PipelineOutputRenderer):
                w.clear_content()
            b = self.canvas._block_by_id(bid)
            if b:
                self.output_tabs.setTabText(idx, f"◈ {b.label[:14]}")
        self.output_edit.clear_content()
        self.output_tabs.setCurrentIndex(0)
        self._log("▶  Pipeline execution started…")

        self.btn_run.setVisible(False)
        self.btn_stop.setVisible(True)

        self._exec_worker = PipelineExecutionWorker(
            list(self.canvas.blocks),
            list(self.canvas.connections),
            text,
            self._engine)
        self._exec_worker.step_started.connect(self._on_step_started)
        self._exec_worker.step_token.connect(self._on_step_token)
        self._exec_worker.step_done.connect(self._on_step_done)
        self._exec_worker.intermediate_live.connect(self._on_intermediate_live)
        self._exec_worker.pipeline_done.connect(self._on_pipeline_done)
        self._exec_worker.err.connect(self._on_exec_err)
        self._exec_worker.log_msg.connect(self._log)
        self._exec_worker.start()

    def _stop_pipeline(self):
        if self._exec_worker:
            self._exec_worker.abort()
            self._exec_worker.wait(2000)
            self._exec_worker = None
        self._log("⏹  Stopped by user.")
        self.btn_run.setVisible(True)
        self.btn_stop.setVisible(False)

    def _on_step_started(self, bid: int, label: str):
        self._log(f"⚡  Block: {label}")
        for b in self.canvas.blocks:
            b.selected = (b.bid == bid)
        self.canvas.update()

    def _on_step_token(self, _bid: int, token: str):
        self.output_edit.append_token(token)

    def _on_intermediate_live(self, bid: int, label: str, text: str):
        """Populate this intermediate block's dedicated output tab."""
        if bid in self._inter_tabs:
            idx    = self._inter_tabs[bid]
            widget = self.output_tabs.widget(idx)
            if isinstance(widget, PipelineOutputRenderer):
                widget.set_content(text)
                self.output_tabs.setTabText(idx, f"◈ {label[:12]} ✅")
                self.output_tabs.setCurrentIndex(idx)

    def _on_step_done(self, bid: int, text: str):
        b = self.canvas._block_by_id(bid)
        lbl = b.label if b else str(bid)
        self._log(f"✅  '{lbl}' → {len(text):,} chars")
        for blk in self.canvas.blocks:
            blk.selected = False
        self.canvas.update()

    def _on_pipeline_done(self, payload: str):
        import json as _json
        try:
            data   = _json.loads(payload)
            final  = data.get("text", payload)
            sender = data.get("sender", "")
        except Exception:
            final  = payload
            sender = ""

        header = (f"**Output from: {sender}**\n\n" if sender else "")
        self._log(f"🏁  Done! {len(final):,} chars from '{sender or 'pipeline'}'.")
        self.output_edit.set_content(header + final)
        self.output_tabs.setCurrentIndex(1)
        self.btn_run.setVisible(True)
        self.btn_stop.setVisible(False)
        self._exec_worker = None
        for b in self.canvas.blocks:
            b.selected = False
        self.canvas.update()

    def _on_exec_err(self, msg: str):
        self._log(f"❌  {msg}")
        self.btn_run.setVisible(True)
        self.btn_stop.setVisible(False)
        self._exec_worker = None
        for b in self.canvas.blocks:
            b.selected = False
        self.canvas.update()
