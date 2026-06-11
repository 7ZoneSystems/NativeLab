from nativelab.Model.ModelRegistry import get_model_registry
from nativelab.Model.model_global import api_model_ref, getapi_registry, is_api_model_ref, is_model_ref_valid, model_ref_display_name
from nativelab.imports.import_global import QInputDialog,QMessageBox,datetime,QListWidgetItem,QApplication,QComboBox,QDialog, Path,Dict,Optional,QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QTextEdit, QFont, QFrame, QTabWidget, QScrollArea, QListWidget, QAbstractItemView, QWidget, QTimer, QSplitter, QSizePolicy, Qt
from .pipefunctions import _pipeline_to_dict, list_example_pipelines, list_saved_pipelines, load_example_pipeline, load_pipeline, pipeline_path, save_pipeline
from .blck_typ import PipelineBlockType 
from nativelab.core.engine_global import LlamaEngine, engine_status
from .executionWorker import PipelineExecutionWorker
from .flowpreview import FlowPreviewController
from .validation import validate_pipeline
from .aibuilder.dialog import AiPipelineBuilderPanel
from nativelab.UI.UI_const import C
from nativelab.UI.buildUI import prepare_adaptive_window
from nativelab.UI.icons import icon, set_button_icon, set_label_icon, set_status_label
from nativelab.UI.llm_error_dialog import show_llm_error_dialog
from .canvas import PipelineCanvas
from .outrender import PipelineOutputRenderer
from manual import make_manual_html, PIPELINE_MANUAL_HTML
from nativelab.GlobalConfig.config_global import ROLE_ICONS, LONG_TIMEOUT_MS
class PipelineBuilderTab(QWidget):
    """
    Full-featured pipeline builder tab.
    Left sidebar : block adders (flow + context) + model list.
    Centre       : interactive canvas.
    Right        : execution controls + per-intermediate live panes + final output.
    """
    LEFT_DEFAULT_W = 214
    RIGHT_DEFAULT_W = 282
    LEFT_RETRACT_W = 76
    RIGHT_RETRACT_W = 88
    LEFT_MIN_W = 132
    RIGHT_MIN_W = 176

    def __init__(self, engine: "LlamaEngine", parent=None):
        super().__init__(parent)
        self._engine                = engine
        self._exec_worker: Optional[PipelineExecutionWorker] = None
        self._inter_tabs:  Dict[int, int] = {}   # block_id → tab index
        self._current_pipeline_name: str = ""
        self._left_scaled_buttons: list = []
        self._left_scaled_labels: list = []
        self._right_scaled_labels: list = []
        self._right_scaled_buttons: list = []
        self._left_sidebar_retracted = False
        self._right_sidebar_retracted = False
        self._last_left_width = self.LEFT_DEFAULT_W
        self._last_right_width = self.RIGHT_DEFAULT_W
        self._updating_splitter = False
        self._preview_active = False
        self._build()

    def update_engine(self, engine: "LlamaEngine"):
        self._engine = engine
        if hasattr(self, "ai_builder_panel"):
            self.ai_builder_panel.set_engine(engine)

    def refresh_theme(self):
        if hasattr(self, "canvas"):
            self.canvas.update()
        self._on_blocks_changed()
        self._update_server_badge()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        try:
            self.main_splitter.setChildrenCollapsible(True)
        except Exception:
            pass
        self.main_splitter.splitterMoved.connect(self._on_main_splitter_moved)
        root.addWidget(self.main_splitter, 1)

        # ══════ LEFT SIDEBAR ══════════════════════════════════════════════════
        sb = QWidget()
        sb.setObjectName("session_sidebar")
        sb_l = QVBoxLayout(sb)
        sb_l.setContentsMargins(10, 14, 10, 14)
        sb_l.setSpacing(7)

        def _sec(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setObjectName("txt3_tiny")
            self._left_scaled_labels.append(lbl)
            return lbl

        def _block_btn(label: str, btype: str, color: str, icon_name: str = "blocks") -> QPushButton:
            btn = QPushButton(label)
            set_button_icon(btn, icon_name, label)
            btn.setFixedHeight(30)
            self._left_scaled_buttons.append((btn, color, 30, True))
            btn.setStyleSheet(
                f"QPushButton{{background:transparent;"
                f"color:{color};border:1px solid {color};"
                f"border-radius:6px;font-size:11px;font-weight:600;}}"
                f"QPushButton:hover{{background:{color};color:#fff;}}")
            btn.clicked.connect(lambda _, bt=btype: self._add_block(bt))
            return btn

        # ── CHANGED: removed color:{C['txt']} - text color now driven by QSS
        hdr = QLabel("Pipeline Builder")
        set_label_icon(hdr, "pipeline", "Pipeline Builder", 16)
        hdr.setObjectName("pipeline_hdr")
        hdr.setStyleSheet("font-size:12px;font-weight:700;")
        self._left_scaled_labels.append(hdr)
        sb_l.addWidget(hdr)

        sep0 = QFrame(); sep0.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep0)

        sb_l.addWidget(_sec("EXAMPLE PRESETS"))
        self.pipeline_preset_combo = QComboBox()
        self.pipeline_preset_combo.setFixedHeight(28)
        self.pipeline_preset_combo.setToolTip(
            "Select a model below, then choose an example pipeline preset.")
        self.pipeline_preset_combo.currentIndexChanged.connect(self._on_pipeline_preset_selected)
        sb_l.addWidget(self.pipeline_preset_combo)
        preset_hint = QLabel("Select a model below first to auto-fill preset model blocks.")
        preset_hint.setWordWrap(True)
        preset_hint.setObjectName("txt3_block")
        self._left_scaled_labels.append(preset_hint)
        sb_l.addWidget(preset_hint)

        sep_examples = QFrame(); sep_examples.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_examples)

        sb_l.addWidget(_sec("FLOW BLOCKS"))
        sb_l.addWidget(_block_btn("Input Block",          PipelineBlockType.INPUT,        C["ok"], "input"))
        sb_l.addWidget(_block_btn("Intermediate Block",   PipelineBlockType.INTERMEDIATE, C["warn"], "blocks"))
        sb_l.addWidget(_block_btn("Output Block",         PipelineBlockType.OUTPUT,       C["err"], "output"))

        sep_ctx = QFrame(); sep_ctx.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_ctx)
        sb_l.addWidget(_sec("CONTEXT BLOCKS"))
        sb_l.addWidget(_block_btn("Reference",   PipelineBlockType.REFERENCE,   C["acc"], "reference"))
        sb_l.addWidget(_block_btn("Knowledge",   PipelineBlockType.KNOWLEDGE,   C["acc2"], "lightbulb"))
        sb_l.addWidget(_block_btn("PDF Summary", PipelineBlockType.PDF_SUMMARY, C["pipeline"], "pdf"))

        sep_logic = QFrame(); sep_logic.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_logic)
        sb_l.addWidget(_sec("LOGIC BLOCKS"))
        sb_l.addWidget(_block_btn("IF / ELSE",   PipelineBlockType.IF_ELSE,     "#f59e0b", "git-branch"))
        sb_l.addWidget(_block_btn("SWITCH",      PipelineBlockType.SWITCH,      "#f97316", "route"))
        sb_l.addWidget(_block_btn("FILTER",      PipelineBlockType.FILTER,      "#84cc16", "filter"))
        sb_l.addWidget(_block_btn("TRANSFORM",   PipelineBlockType.TRANSFORM,   "#06b6d4", "refresh-cw"))
        sb_l.addWidget(_block_btn("MERGE",       PipelineBlockType.MERGE,       "#8b5cf6", "combine"))
        sb_l.addWidget(_block_btn("SPLIT",       PipelineBlockType.SPLIT,       "#ec4899", "split"))

        sep_code = QFrame(); sep_code.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_code)
        sb_l.addWidget(_sec("CUSTOM CODE"))
        sb_l.addWidget(_block_btn("Custom Code", PipelineBlockType.CUSTOM_CODE, "#10b981", "code"))

        sep_llm = QFrame(); sep_llm.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_llm)
        sb_l.addWidget(_sec("LLM LOGIC  (natural language)"))
        _llm_note = QLabel(
            "Conditions & instructions written\n"
            "in plain English - evaluated by\n"
            "the block's attached LLM model.")
        _llm_note.setObjectName("txt3_block")
        self._left_scaled_labels.append(_llm_note)
        sb_l.addWidget(_llm_note)
        sb_l.addWidget(_block_btn("LLM IF / ELSE",   PipelineBlockType.LLM_IF,        "#a855f7", "brain"))
        sb_l.addWidget(_block_btn("LLM SWITCH",      PipelineBlockType.LLM_SWITCH,     "#7c3aed", "brain"))
        sb_l.addWidget(_block_btn("LLM FILTER",      PipelineBlockType.LLM_FILTER,     "#6366f1", "brain"))
        sb_l.addWidget(_block_btn("LLM TRANSFORM",   PipelineBlockType.LLM_TRANSFORM,  "#0ea5e9", "brain"))
        sb_l.addWidget(_block_btn("LLM SCORE",       PipelineBlockType.LLM_SCORE,      "#d946ef", "brain"))

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

        self.btn_refresh = QPushButton("Refresh")
        set_button_icon(self.btn_refresh, "refresh-cw", "Refresh")
        self.btn_refresh.setFixedHeight(26)
        self._left_scaled_buttons.append((self.btn_refresh, C["txt2"], 26, False))
        self.btn_refresh.clicked.connect(self._refresh_models)
        sb_l.addWidget(self.btn_refresh)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep2)

        sb_l.addWidget(_sec("CANVAS CONTROLS"))
        self.btn_clear_canvas = QPushButton("Clear All")
        set_button_icon(self.btn_clear_canvas, "delete", "Clear All")
        self.btn_clear_canvas.setFixedHeight(28)
        self._left_scaled_buttons.append((self.btn_clear_canvas, C["txt2"], 28, False))
        self.btn_clear_canvas.clicked.connect(self._clear_canvas)
        sb_l.addWidget(self.btn_clear_canvas)

        self.btn_save_pipeline = QPushButton("Save Pipeline...")
        set_button_icon(self.btn_save_pipeline, "save", "Save Pipeline...")
        self.btn_save_pipeline.setFixedHeight(28)
        self._left_scaled_buttons.append((self.btn_save_pipeline, C["txt2"], 28, False))
        self.btn_save_pipeline.clicked.connect(self._save_pipeline)
        sb_l.addWidget(self.btn_save_pipeline)

        self.btn_load_pipeline = QPushButton("Load Pipeline...")
        set_button_icon(self.btn_load_pipeline, "folder-open", "Load Pipeline...")
        self.btn_load_pipeline.setFixedHeight(28)
        self._left_scaled_buttons.append((self.btn_load_pipeline, C["txt2"], 28, False))
        self.btn_load_pipeline.clicked.connect(self._load_pipeline)
        sb_l.addWidget(self.btn_load_pipeline)

        self.btn_preview = QPushButton("Preview Flow")
        set_button_icon(self.btn_preview, "play", "Preview Flow")
        self.btn_preview.setFixedHeight(28)
        self._left_scaled_buttons.append((self.btn_preview, C["pipeline"], 28, True))
        self._set_preview_button_style(False)
        self.btn_preview.clicked.connect(self._toggle_flow_preview)
        sb_l.addWidget(self.btn_preview)
        self._preview_ctrl: Optional[FlowPreviewController] = None

        hint = QLabel(
            "Tip: draw a connection from a\n"
            "port dot (N·S·E·W) on one block\n"
            "to a port dot on another block.\n\n"
            "Model→Model requires an\n"
            "Intermediate block in between.")
        hint.setWordWrap(True)
        hint.setObjectName("txt3_block")
        self._left_scaled_labels.append(hint)
        sb_l.addWidget(hint)

        sb_l.addStretch()

        sb_scroll = QScrollArea()
        self.left_sidebar_scroll = sb_scroll
        sb_scroll.setWidget(sb)
        sb_scroll.setWidgetResizable(True)
        sb_scroll.setMinimumWidth(0)
        sb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sb_scroll.setObjectName("session_sidebar")
        sb_scroll.setStyleSheet("QScrollArea#session_sidebar { border: none; }")
        self.main_splitter.addWidget(sb_scroll)

        # ══════ CENTRE: CANVAS ════════════════════════════════════════════════
        centre = QWidget()
        centre.setMinimumWidth(360)
        try:
            centre.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        except Exception:
            pass
        self._canvas_centre = centre
        centre_l = QVBoxLayout(centre)
        centre_l.setContentsMargins(0, 0, 0, 0)
        centre_l.setSpacing(0)

        # toolbar
        toolbar = QWidget()
        toolbar.setObjectName("appearance_bar")
        toolbar.setFixedHeight(40)
        toolbar.setMinimumWidth(0)
        tb_l = QHBoxLayout(toolbar)
        tb_l.setContentsMargins(12, 4, 12, 4)
        tb_l.setSpacing(8)
        tb_title = QLabel("Pipeline Canvas")
        set_label_icon(tb_title, "pipeline", "Pipeline Canvas", 16)
        tb_title.setObjectName("appearance_hdr")
        tb_l.addWidget(tb_title)
        tb_l.addStretch()
        legend_items = [
            (C["ok"],       "Input"),
            (C["warn"],     "Intermediate"),
            (C["err"],      "Output"),
            (C["acc"],      "Model"),
            ("#f59e0b",     "IF/ELSE"),
            ("#06b6d4",     "Transform"),
            ("#10b981",     "Code"),
            ("#8b5cf6",     "Merge"),
            ("#ec4899",     "Split"),
            ("#a855f7",     "LLM-IF"),
            ("#7c3aed",     "LLM-SW"),
            ("#0ea5e9",     "LLM-TX"),
            ("#d946ef",     "LLM-SC"),
            (C["pipeline"], "Loop"),
            (C["acc2"],     "Forward"),
        ]
        for col, txt in legend_items:
            lbl = QLabel(txt)
            # ── CHANGED: removed background:{C['bg2']} - background now driven
            #    by QSS via objectName. Per-block `col` is semantic (not theme),
            #    so it is kept as an inline color for the text.
            lbl.setObjectName("legend_pill")
            try:
                lbl.setMinimumWidth(0)
                lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            except Exception:
                pass
            lbl.setStyleSheet(
                f"color:{col};font-size:10px;font-weight:600;"
                f"padding:1px 6px;"
                f"border-radius:4px;")
            tb_l.addWidget(lbl)
        centre_l.addWidget(toolbar)

        pill_outer = QWidget()
        pill_outer.setObjectName("appearance_bar")
        pill_outer.setFixedHeight(36)
        pill_outer_l = QHBoxLayout(pill_outer)
        pill_outer_l.setContentsMargins(8, 3, 8, 3)
        pill_outer_l.setSpacing(0)

        # ── CHANGED: removed inline color from _pill_icon; objectName drives it
        _pill_icon = QLabel("")
        set_label_icon(_pill_icon, "zap", "", 14)
        _pill_icon.setObjectName("txt3_xs")
        _pill_icon.setStyleSheet("padding-right:4px;")
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
        canvas_scroll.setMinimumWidth(360)
        try:
            canvas_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        except Exception:
            pass
        canvas_scroll.setWidgetResizable(False)
        self.canvas_scroll_area = canvas_scroll
        self.canvas = PipelineCanvas()
        self.canvas.blocks_changed.connect(self._on_blocks_changed)
        self.canvas.update()
        canvas_scroll.setWidget(self.canvas)
        centre_l.addWidget(canvas_scroll, 1)
        self.btn_open_left_sidebar = self._make_sidebar_reopen_button("left", centre)
        self.btn_open_right_sidebar = self._make_sidebar_reopen_button("right", centre)

        self.main_splitter.addWidget(centre)

        # ══════ RIGHT: EXECUTION / AI BUILDER PANEL ══════════════════════════
        rp = QWidget()
        self.right_sidebar_panel = rp
        rp.setMinimumWidth(0)
        try:
            rp.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        except Exception:
            pass
        rp.setObjectName("ref_panel")
        rp_outer_l = QVBoxLayout(rp)
        rp_outer_l.setContentsMargins(8, 8, 8, 8)
        rp_outer_l.setSpacing(0)

        self.right_tabs = QTabWidget()
        self.right_tabs.setObjectName("ref_tabs")
        self.right_tabs.setMinimumWidth(0)
        try:
            self.right_tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
            self.right_tabs.setUsesScrollButtons(True)
            self.right_tabs.setElideMode(Qt.TextElideMode.ElideRight)
        except Exception:
            pass
        rp_outer_l.addWidget(self.right_tabs, 1)

        exec_page = QWidget()
        exec_page.setMinimumWidth(0)
        try:
            exec_page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        except Exception:
            pass
        rp_l = QVBoxLayout(exec_page)
        rp_l.setContentsMargins(6, 8, 6, 8)
        rp_l.setSpacing(7)

        exec_hdr_row = QHBoxLayout(); exec_hdr_row.setSpacing(6)
        # ── CHANGED: removed color:{C['txt']} - text color now driven by QSS
        exec_hdr = QLabel("Execute Pipeline")
        set_label_icon(exec_hdr, "play", "Execute Pipeline", 16)
        exec_hdr.setObjectName("pipeline_hdr")
        exec_hdr.setStyleSheet("font-size:12px;font-weight:700;")
        self._right_scaled_labels.append(exec_hdr)
        exec_hdr_row.addWidget(exec_hdr, 1)
        btn_manual = QPushButton("Manual")
        set_button_icon(btn_manual, "book-open", "Manual")
        btn_manual.setFixedHeight(24)
        self._right_scaled_buttons.append((btn_manual, 24))
        btn_manual.setStyleSheet(
            f"QPushButton{{background:transparent;color:{C['acc']};"
            f"border:1px solid {C['acc']};border-radius:5px;"
            f"font-size:10px;font-weight:600;padding:0 8px;}}"
            f"QPushButton:hover{{background:{C['acc']};color:#fff;}}")
        btn_manual.clicked.connect(self._show_manual)
        exec_hdr_row.addWidget(btn_manual)
        rp_l.addLayout(exec_hdr_row)

        # Server status badge - state-driven via QSS property, not inline color
        self.server_badge = QLabel("Engine status unknown")
        self.server_badge.setObjectName("status_badge")
        self.server_badge.setProperty("state", "idle")
        self._right_scaled_labels.append(self.server_badge)
        rp_l.addWidget(self.server_badge)
        self._badge_timer = QTimer(self)
        self._badge_timer.timeout.connect(self._update_server_badge)
        self._badge_timer.start(2500)

        # ── CHANGED: removed color:{C['txt2']} - objectName drives the color
        input_lbl = QLabel("Input text:")
        input_lbl.setObjectName("txt2")
        input_lbl.setStyleSheet("font-size:11px;")
        self._right_scaled_labels.append(input_lbl)
        rp_l.addWidget(input_lbl)

        self.input_edit = QTextEdit()
        self.input_edit.setMinimumWidth(0)
        try:
            self.input_edit.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        except Exception:
            pass
        self.input_edit.setPlaceholderText(
            "Enter the prompt or text to feed\n"
            "into the first INPUT block…")
        self.input_edit.setMaximumHeight(100)
        rp_l.addWidget(self.input_edit)

        self.btn_run = QPushButton("Run Pipeline")
        set_button_icon(self.btn_run, "play", "Run Pipeline")
        self.btn_run.setObjectName("btn_send")
        self.btn_run.setFixedHeight(34)
        self._right_scaled_buttons.append((self.btn_run, 34))
        self.btn_run.clicked.connect(self._run_pipeline)
        rp_l.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop Execution")
        set_button_icon(self.btn_stop, "stop-circle", "Stop Execution")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(34)
        self.btn_stop.setVisible(False)
        self._right_scaled_buttons.append((self.btn_stop, 34))
        self.btn_stop.clicked.connect(self._stop_pipeline)
        rp_l.addWidget(self.btn_stop)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        rp_l.addWidget(sep3)

        # Tabbed output: Log | Output | one tab per intermediate block
        self.output_tabs = QTabWidget()
        self.output_tabs.setObjectName("ref_tabs")
        self.output_tabs.setMinimumWidth(0)
        try:
            self.output_tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
            self.output_tabs.setUsesScrollButtons(True)
            self.output_tabs.setElideMode(Qt.TextElideMode.ElideRight)
        except Exception:
            pass

        self.exec_log = QTextEdit()
        self.exec_log.setMinimumWidth(0)
        self.exec_log.setReadOnly(True)
        self.exec_log.setFont(QFont("Consolas", 9))
        self.exec_log.setObjectName("log_te")
        self.output_tabs.addTab(self.exec_log, icon("list"), "Log")

        self.output_edit = PipelineOutputRenderer(
            placeholder="Final pipeline output appears here…")
        self.output_tabs.addTab(self.output_edit, icon("output"), "Output")
        rp_l.addWidget(self.output_tabs, 1)

        copy_btn = QPushButton("Copy Final Output")
        set_button_icon(copy_btn, "copy", "Copy Final Output")
        copy_btn.setFixedHeight(28)
        self._right_scaled_buttons.append((copy_btn, 28))
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(
                self.output_edit.raw_text()))
        rp_l.addWidget(copy_btn)

        self.right_tabs.addTab(exec_page, icon("play"), "Execution")

        ai_page = QWidget()
        ai_page.setMinimumWidth(0)
        try:
            ai_page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        except Exception:
            pass
        ai_l = QVBoxLayout(ai_page)
        ai_l.setContentsMargins(0, 0, 0, 0)
        ai_l.setSpacing(0)
        self.ai_builder_panel = AiPipelineBuilderPanel(
            self._engine,
            parent=ai_page,
            load_callback=self._load_ai_generated_pipeline,
            active_model_provider=self._preset_model_ref,
            canvas_state_provider=self._ai_canvas_state,
            show_close=False,
            compact=True,
        )
        ai_scroll = QScrollArea()
        ai_scroll.setMinimumWidth(0)
        try:
            ai_scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        except Exception:
            pass
        ai_scroll.setWidgetResizable(True)
        ai_scroll.setWidget(self.ai_builder_panel)
        ai_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ai_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        ai_scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        ai_l.addWidget(ai_scroll, 1)
        self.right_tabs.addTab(ai_page, icon("brain"), "AI Builder")
        self.right_tabs.setCurrentIndex(0)

        self.main_splitter.addWidget(rp)
        self._install_initial_sidebar_sizes()

        self._refresh_pipeline_presets()
        self._refresh_models()
        QTimer.singleShot(0, self._install_initial_sidebar_sizes)

    # ── resizable sidebars ───────────────────────────────────────────────────

    @staticmethod
    def _scaled_px(base: int, width: int, default_width: int,
                   min_scale: float = 0.82, max_scale: float = 1.18) -> int:
        try:
            scale = float(width or default_width) / float(default_width or width or 1)
        except Exception:
            scale = 1.0
        scale = max(min_scale, min(max_scale, scale))
        return max(8, int(round(float(base) * scale)))

    def _safe_splitter_sizes(self) -> list:
        try:
            sizes = list(self.main_splitter.sizes())
        except Exception:
            sizes = []
        if len(sizes) != 3:
            return [
                0 if self._left_sidebar_retracted else self._last_left_width,
                900,
                0 if self._right_sidebar_retracted else self._last_right_width,
            ]
        return [max(0, int(v or 0)) for v in sizes]

    def _set_font_px(self, widget, px: int):
        try:
            font = widget.font()
            font.setPointSize(max(1, int(px)))
            widget.setFont(font)
        except Exception:
            pass

    def _set_preview_button_style(self, active: bool, px: int = 11):
        if not hasattr(self, "btn_preview"):
            return
        if active:
            self.btn_preview.setStyleSheet(
                f"QPushButton{{background:{C['pipeline']};color:{C['bg1']};"
                f"border:1px solid {C['pipeline']};border-radius:6px;"
                f"font-size:{px}px;font-weight:600;}}"
                f"QPushButton:hover{{background:{C['acc2']};color:{C['bg1']};}}")
        else:
            self.btn_preview.setStyleSheet(
                f"QPushButton{{background:transparent;color:{C['pipeline']};"
                f"border:1px solid {C['pipeline']};border-radius:6px;"
                f"font-size:{px}px;font-weight:600;}}"
                f"QPushButton:hover{{background:{C['pipeline']};color:{C['bg1']};}}")

    def _make_sidebar_reopen_button(self, side: str, parent: QWidget) -> QPushButton:
        btn = QPushButton(">" if side == "left" else "<", parent)
        btn.setToolTip(f"Open {side} sidebar")
        btn.setFixedSize(28, 28)
        btn.setStyleSheet(
            f"QPushButton{{background:{C['bg1']};color:{C['acc']};"
            f"border:1px solid {C['acc']};border-radius:14px;"
            f"font-size:15px;font-weight:800;}}"
            f"QPushButton:hover{{background:{C['acc']};color:#fff;}}")
        btn.clicked.connect(lambda _, s=side: self._restore_sidebar(s))
        btn.setVisible(False)
        return btn

    def _position_sidebar_reopen_buttons(self):
        centre = getattr(self, "_canvas_centre", None)
        canvas_scroll = getattr(self, "canvas_scroll_area", None)
        if centre is None:
            return
        try:
            base_y = canvas_scroll.y() if canvas_scroll is not None else 0
            area_h = canvas_scroll.height() if canvas_scroll is not None else centre.height()
            btn_h = 28
            y = max(8, base_y + max(0, (area_h - btn_h) // 2))
            positions = (
                ("btn_open_left_sidebar", 8),
                ("btn_open_right_sidebar", max(8, centre.width() - 36)),
            )
            for name, x in positions:
                btn = getattr(self, name, None)
                if btn is None:
                    continue
                btn.move(int(x), int(y))
                btn.raise_()
        except RuntimeError:
            pass

    def _install_initial_sidebar_sizes(self):
        for idx, stretch in enumerate((0, 1, 0)):
            try:
                self.main_splitter.setStretchFactor(idx, stretch)
            except Exception:
                pass
        self._left_sidebar_retracted = False
        self._right_sidebar_retracted = False
        self._apply_splitter_state(total_hint=1400)

    def _apply_splitter_state(self, total_hint: int = 0):
        sizes = self._safe_splitter_sizes()
        total = max(int(total_hint or 0), sum(sizes), 900)
        left = 0 if self._left_sidebar_retracted else max(self.LEFT_MIN_W, self._last_left_width)
        right = 0 if self._right_sidebar_retracted else max(self.RIGHT_MIN_W, self._last_right_width)
        centre = max(360, total - left - right)
        self._set_right_tab_labels(right)

        self._updating_splitter = True
        try:
            self.left_sidebar_scroll.setVisible(not self._left_sidebar_retracted)
            self.right_sidebar_panel.setVisible(not self._right_sidebar_retracted)
            self.main_splitter.setSizes([left, centre, right])
        finally:
            self._updating_splitter = False
        self._update_sidebar_reopen_buttons()
        self._position_sidebar_reopen_buttons()
        self._apply_sidebar_scaling()

    def _retract_sidebar(self, side: str):
        sizes = self._safe_splitter_sizes()
        if side == "left" and not self._left_sidebar_retracted:
            if sizes[0] > self.LEFT_RETRACT_W:
                self._last_left_width = max(self.LEFT_MIN_W, sizes[0])
            self._left_sidebar_retracted = True
        elif side == "right" and not self._right_sidebar_retracted:
            if sizes[2] > self.RIGHT_RETRACT_W:
                self._last_right_width = max(self.RIGHT_MIN_W, sizes[2])
            self._right_sidebar_retracted = True
        self._apply_splitter_state(total_hint=sum(sizes))

    def _restore_sidebar(self, side: str):
        sizes = self._safe_splitter_sizes()
        if side == "left":
            self._left_sidebar_retracted = False
            self._last_left_width = max(self.LEFT_MIN_W, self._last_left_width)
        elif side == "right":
            self._right_sidebar_retracted = False
            self._last_right_width = max(self.RIGHT_MIN_W, self._last_right_width)
        self._apply_splitter_state(total_hint=sum(sizes))

    def _update_sidebar_reopen_buttons(self):
        for name, visible in (
            ("btn_open_left_sidebar", self._left_sidebar_retracted),
            ("btn_open_right_sidebar", self._right_sidebar_retracted),
        ):
            btn = getattr(self, name, None)
            if btn is None:
                continue
            try:
                btn.setVisible(bool(visible))
                if visible:
                    btn.raise_()
            except RuntimeError:
                pass
        self._position_sidebar_reopen_buttons()

    def _on_main_splitter_moved(self, _pos: int, _index: int):
        if self._updating_splitter:
            return
        sizes = self._safe_splitter_sizes()
        if not self._left_sidebar_retracted:
            if sizes[0] <= self.LEFT_RETRACT_W:
                self._retract_sidebar("left")
                return
            self._last_left_width = max(self.LEFT_MIN_W, sizes[0])
        if not self._right_sidebar_retracted:
            if sizes[2] <= self.RIGHT_RETRACT_W:
                self._retract_sidebar("right")
                return
            self._last_right_width = max(self.RIGHT_MIN_W, sizes[2])
        self._position_sidebar_reopen_buttons()
        self._apply_sidebar_scaling()

    def _apply_sidebar_scaling(self):
        if not self._left_sidebar_retracted:
            self._apply_left_sidebar_scale(self._last_left_width)
        if not self._right_sidebar_retracted:
            self._apply_right_sidebar_scale(self._last_right_width)

    def _apply_left_sidebar_scale(self, width: int):
        body_px = self._scaled_px(11, width, self.LEFT_DEFAULT_W)
        tiny_px = self._scaled_px(9, width, self.LEFT_DEFAULT_W)
        header_px = self._scaled_px(12, width, self.LEFT_DEFAULT_W)
        try:
            self.left_sidebar_scroll.widget().setStyleSheet(
                f"QWidget#session_sidebar{{background:{C['bg1']};"
                f"border-right:1px solid {C['bdr']};font-size:{body_px}px;}}"
                f"QLabel{{font-size:{body_px}px;}}"
                f"QComboBox,QListWidget{{font-size:{body_px}px;}}")
        except Exception:
            pass
        for label in self._left_scaled_labels:
            try:
                name = label.objectName()
            except Exception:
                name = ""
            px = header_px if name == "pipeline_hdr" else tiny_px if name == "txt3_tiny" else body_px
            self._set_font_px(label, px)
        for btn, color, base_h, custom_style in self._left_scaled_buttons:
            px = self._scaled_px(11, width, self.LEFT_DEFAULT_W)
            h = max(22, self._scaled_px(base_h, width, self.LEFT_DEFAULT_W, 0.88, 1.12))
            try:
                btn.setFixedHeight(h)
            except Exception:
                pass
            self._set_font_px(btn, px)
            if hasattr(self, "btn_preview") and btn is self.btn_preview:
                self._set_preview_button_style(self._preview_active, px)
                continue
            if custom_style:
                btn.setStyleSheet(
                    f"QPushButton{{background:transparent;color:{color};"
                    f"border:1px solid {color};border-radius:6px;"
                    f"font-size:{px}px;font-weight:600;}}"
                    f"QPushButton:hover{{background:{color};color:#fff;}}")

    def _apply_right_sidebar_scale(self, width: int):
        body_px = self._scaled_px(11, width, self.RIGHT_DEFAULT_W, 0.86, 1.16)
        header_px = self._scaled_px(12, width, self.RIGHT_DEFAULT_W, 0.86, 1.16)
        try:
            self.right_sidebar_panel.setStyleSheet(
                f"QWidget#ref_panel{{background:{C['bg1']};"
                f"border-left:1px solid {C['bdr']};font-size:{body_px}px;}}"
                f"QLabel{{font-size:{body_px}px;}}"
                f"QPushButton,QTextEdit,QLineEdit,QComboBox,QListWidget,QTabWidget{{font-size:{body_px}px;}}"
                f"QTabBar::tab{{font-size:{body_px}px;}}")
        except Exception:
            pass
        for label in self._right_scaled_labels:
            try:
                name = label.objectName()
            except Exception:
                name = ""
            self._set_font_px(label, header_px if name == "pipeline_hdr" else body_px)
        for btn, base_h in self._right_scaled_buttons:
            self._set_font_px(btn, body_px)
            try:
                btn.setFixedHeight(max(22, self._scaled_px(base_h, width, self.RIGHT_DEFAULT_W, 0.88, 1.1)))
            except Exception:
                pass
        if hasattr(self, "ai_builder_panel"):
            try:
                self.ai_builder_panel.set_sidebar_width(width)
            except Exception:
                pass
        self._set_right_tab_labels(width)

    def _set_right_tab_labels(self, width: int):
        if not hasattr(self, "right_tabs"):
            return
        tight = int(width or 0) < 240
        labels = ("Exec", "AI") if tight else ("Execution", "AI Builder")
        try:
            if self.right_tabs.tabText(0) != labels[0]:
                self.right_tabs.setTabText(0, labels[0])
            if self.right_tabs.tabText(1) != labels[1]:
                self.right_tabs.setTabText(1, labels[1])
        except Exception:
            pass

    def resizeEvent(self, event):
        try:
            super().resizeEvent(event)
        except Exception:
            pass
        if hasattr(self, "main_splitter"):
            QTimer.singleShot(0, self._sync_sidebar_sizes_from_splitter)
            QTimer.singleShot(0, self._position_sidebar_reopen_buttons)

    def _sync_sidebar_sizes_from_splitter(self):
        if self._updating_splitter or not hasattr(self, "main_splitter"):
            return
        sizes = self._safe_splitter_sizes()
        changed = False
        if not self._left_sidebar_retracted:
            if sizes[0] <= self.LEFT_RETRACT_W:
                self._left_sidebar_retracted = True
                changed = True
            else:
                self._last_left_width = max(self.LEFT_MIN_W, sizes[0])
        if not self._right_sidebar_retracted:
            if sizes[2] <= self.RIGHT_RETRACT_W:
                self._right_sidebar_retracted = True
                changed = True
            else:
                self._last_right_width = max(self.RIGHT_MIN_W, sizes[2])
        if changed:
            self._apply_splitter_state(total_hint=sum(sizes))
        else:
            self._update_sidebar_reopen_buttons()
            self._position_sidebar_reopen_buttons()
            self._apply_sidebar_scaling()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _refresh_models(self):
        """Show the pipeline builder manual in a scrollable dialog."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Pipeline Builder Manual")
        prepare_adaptive_window(dlg, 740, 640, min_width=560, min_height=420)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)

        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("Inter", 11))
        te.setObjectName("chat_te")
        te.setHtml(PIPELINE_MANUAL_HTML)
        lay.addWidget(te, 1)

        btn_close = QPushButton("Close")
        set_button_icon(btn_close, "x", "Close")
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
        dlg.setWindowTitle("Pipeline Builder Manual")
        prepare_adaptive_window(dlg, 740, 640, min_width=560, min_height=420)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("Inter", 11))
        te.setObjectName("chat_te")
        te.setHtml(make_manual_html())
        lay.addWidget(te, 1)
        btn_close = QPushButton("Close")
        set_button_icon(btn_close, "x", "Close")
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
            ri   = ROLE_ICONS.get(m.get("role", "general"), "General")
            qt   = m.get("quant", "?")
            fam  = m.get("family", "?")
            vision = f" · VLM:{m.get('vision_label') or 'vision'}" if m.get("vision") else ""
            item = QListWidgetItem(f"{ri}  {m['name']}\n    {fam} · {qt}{vision}")
            item.setIcon(icon("vision" if m.get("vision") else "models"))
            item.setData(Qt.ItemDataRole.UserRole,     m["path"])
            item.setData(Qt.ItemDataRole.UserRole + 1, m.get("role", "general"))
            self.model_list.addItem(item)
        for cfg in getapi_registry().all():
            item = QListWidgetItem(f"{cfg.name}\n    API · {cfg.provider} · {cfg.model_id}")
            item.setIcon(icon("api"))
            item.setData(Qt.ItemDataRole.UserRole, api_model_ref(cfg.name))
            item.setData(Qt.ItemDataRole.UserRole + 1, "general")
            self.model_list.addItem(item)

    def _refresh_pipeline_presets(self):
        combo = getattr(self, "pipeline_preset_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Choose example...", "")
        for item in list_example_pipelines():
            title = item.get("title") or item.get("name", "")
            combo.addItem(str(title), item.get("name", ""))
            idx = combo.count() - 1
            combo.setItemData(idx, item.get("description", ""), Qt.ItemDataRole.ToolTipRole)
        combo.blockSignals(False)

    def _on_pipeline_preset_selected(self, index: int):
        combo = getattr(self, "pipeline_preset_combo", None)
        if combo is None or index <= 0:
            return
        name = str(combo.itemData(index) or "")
        if not name:
            return
        try:
            self._load_pipeline_example(name)
        finally:
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)

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
        self._log(f"Pipeline saved as '{name}'")
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
        items = saved + ["─────────────", "Delete a pipeline..."]
        choice, ok = QInputDialog.getItem(
            self, "Load Pipeline", "Select a pipeline:", items, 0, False)
        if not ok:
            return
        if choice == "Delete a pipeline...":
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
        self._replace_canvas_with_pipeline(blocks, conns)
        self._current_pipeline_name = choice
        self._log(f"Pipeline '{choice}' loaded "
                  f"({len(blocks)} blocks, {len(conns)} connections)")

    def _load_pipeline_example(self, name: str):
        if self.canvas.blocks:
            ans = QMessageBox.question(
                self, "Replace Canvas",
                "Load this example preset and replace the current canvas?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return
        try:
            blocks, conns = load_example_pipeline(name)
        except Exception as e:
            QMessageBox.critical(self, "Preset Load Error", str(e)); return
        model_ref, role = self._preset_model_ref()
        placeholder_count = self._fill_example_model_placeholders(blocks, model_ref, role)
        self._replace_canvas_with_pipeline(blocks, conns)
        self._current_pipeline_name = ""
        if placeholder_count and model_ref:
            self._log(
                f"Example preset '{name}' loaded with "
                f"{placeholder_count} model block(s) set to {model_ref_display_name(model_ref)}")
        else:
            self._log(f"Example preset '{name}' loaded.")
            if placeholder_count:
                QMessageBox.information(
                    self,
                    "Preset Loaded",
                    "The preset contains placeholder model blocks.\n\n"
                    "Select a model in the sidebar before choosing a preset to auto-fill them, "
                    "or delete the placeholder blocks and add your model blocks manually.")

    def _replace_canvas_with_pipeline(self, blocks: list, conns: list):
        self.canvas.clear_all()
        while self.output_tabs.count() > 2:
            self.output_tabs.removeTab(2)
        self._inter_tabs.clear()
        for b in blocks:
            self.canvas.blocks.append(b)
        self.canvas.connections = conns
        self.canvas.normalize_block_ids()
        self.canvas.ensure_canvas_fits_blocks()
        self.canvas.update()
        self.canvas.blocks_changed.emit()

    def _load_ai_generated_pipeline(self, name: str):
        if self.canvas.blocks:
            ans = QMessageBox.question(
                self,
                "Replace Canvas",
                f"Load generated pipeline '{name}' and replace the current canvas?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        try:
            blocks, conns = load_pipeline(name)
        except Exception as e:
            QMessageBox.critical(self, "AI Pipeline Load Error", str(e))
            return
        self._replace_canvas_with_pipeline(blocks, conns)
        self._current_pipeline_name = name
        if hasattr(self, "right_tabs"):
            self.right_tabs.setCurrentIndex(0)
        self._log(f"AI-generated pipeline '{name}' loaded for testing.")

    def _ai_canvas_state(self) -> dict:
        data = _pipeline_to_dict(self.canvas.blocks, self.canvas.connections)
        data["current_pipeline_name"] = self._current_pipeline_name
        data["canvas_empty"] = not bool(self.canvas.blocks)
        return data

    def _preset_model_ref(self) -> tuple[str, str]:
        item = self.model_list.currentItem() if hasattr(self, "model_list") else None
        if item is None and hasattr(self, "model_list") and self.model_list.count() == 1:
            item = self.model_list.item(0)
        if item is not None:
            path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            role = str(item.data(Qt.ItemDataRole.UserRole + 1) or "general")
            if path:
                return path, role
        active_path = str(getattr(self._engine, "model_path", "") or "")
        if active_path and (is_api_model_ref(active_path) or is_model_ref_valid(active_path)):
            return active_path, "general"
        return "", "general"

    def _fill_example_model_placeholders(self, blocks: list, model_ref: str, role: str) -> int:
        model_backed_types = {
            PipelineBlockType.MODEL,
            PipelineBlockType.LLM_IF,
            PipelineBlockType.LLM_SWITCH,
            PipelineBlockType.LLM_FILTER,
            PipelineBlockType.LLM_TRANSFORM,
            PipelineBlockType.LLM_SCORE,
        }
        placeholders = [
            b for b in blocks
            if b.btype in model_backed_types and not getattr(b, "model_path", "")
        ]
        if not model_ref:
            if getattr(self._engine, "mode", "") == "api":
                for b in placeholders:
                    b.label = b.label if b.label not in {"Your model", "Model"} else "Active API model"
            return len(placeholders)
        for b in placeholders:
            b.model_path = model_ref
            if not getattr(b, "role", ""):
                b.role = role or "general"
            if b.label in {"Your model", "Model"}:
                b.label = model_ref_display_name(model_ref)[:18]
        return len(placeholders)

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
            pipeline_path(choice).unlink(missing_ok=True)
            self._log(f"Pipeline '{choice}' deleted.")

    def _update_server_badge(self):
        status = engine_status(self._engine, none_text="No model loaded")
        text = f"Server · port {status.server_port}" if status.mode == "server" and status.server_port else status.status_text
        set_status_label(self.server_badge, text, status.state)
        self.server_badge.setProperty("state", status.state)
        self.server_badge.style().unpolish(self.server_badge)
        self.server_badge.style().polish(self.server_badge)

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
            _empty = QLabel("No model or logic blocks yet - drag from sidebar or click buttons")
            _empty.setObjectName("txt3_block")
            _empty.setStyleSheet("padding:2px 6px;")
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
                    placeholder=f"Live context arriving at '{b.label}' will appear here...")
                idx = self.output_tabs.addTab(te, b.label[:14])
                self._inter_tabs[b.bid] = idx

    # ── flow preview ──────────────────────────────────────────────────────────

    def _toggle_flow_preview(self):
        if self._preview_ctrl is not None:
            self._stop_flow_preview(); return
        if not self.canvas.blocks or not self.canvas.connections:
            QMessageBox.information(
                self, "Preview Flow",
                "Add blocks and connect them before previewing."); return
        if not any(b.btype == PipelineBlockType.INPUT for b in self.canvas.blocks):
            QMessageBox.information(
                self, "Preview Flow",
                "Pipeline needs an INPUT block to start the preview."); return
        self._preview_ctrl = FlowPreviewController(
            self.canvas,
            list(self.canvas.blocks),
            list(self.canvas.connections),
            parent=self)
        self._preview_ctrl.finished.connect(self._stop_flow_preview)
        set_button_icon(self.btn_preview, "stop-circle", "Stop Preview")
        self._preview_active = True
        self._set_preview_button_style(True, self._scaled_px(11, self._last_left_width, self.LEFT_DEFAULT_W))
        self._preview_ctrl.start()

    def _stop_flow_preview(self):
        if self._preview_ctrl is not None:
            self._preview_ctrl.stop()
            self._preview_ctrl = None
        set_button_icon(self.btn_preview, "play", "Preview Flow")
        self._preview_active = False
        self._set_preview_button_style(False, self._scaled_px(11, self._last_left_width, self.LEFT_DEFAULT_W))

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.exec_log.append(
            f'<span class="log_ts">[{ts}]</span> '
            f'<span class="log_msg">{msg}</span>')

    # ── execution ─────────────────────────────────────────────────────────────

    def _validate(self) -> Optional[str]:
        return validate_pipeline(self.canvas.blocks, self.canvas.connections)

    def _run_pipeline(self):
        err = self._validate()
        if err:
            QMessageBox.warning(self, "Invalid Pipeline", err); return
        if hasattr(self, "right_tabs"):
            self.right_tabs.setCurrentIndex(0)

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
                self.output_tabs.setTabText(idx, b.label[:14])
        self.output_edit.clear_content()
        self.output_tabs.setCurrentIndex(0)
        self._log("Pipeline execution started...")

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
            self._exec_worker.wait(LONG_TIMEOUT_MS)
            self._exec_worker = None
        self._log("Stopped by user.")
        self.btn_run.setVisible(True)
        self.btn_stop.setVisible(False)

    def _on_step_started(self, bid: int, label: str):
        self._log(f"Block: {label}")
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
                self.output_tabs.setTabText(idx, f"{label[:12]} OK")
                self.output_tabs.setCurrentIndex(idx)

    def _on_step_done(self, bid: int, text: str):
        b = self.canvas._block_by_id(bid)
        lbl = b.label if b else str(bid)
        self._log(f"'{lbl}' -> {len(text):,} chars")
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
        self._log(f"Done! {len(final):,} chars from '{sender or 'pipeline'}'.")
        self.output_edit.set_content(header + final)
        self.output_tabs.setCurrentIndex(1)
        self.btn_run.setVisible(True)
        self.btn_stop.setVisible(False)
        self._exec_worker = None
        for b in self.canvas.blocks:
            b.selected = False
        self.canvas.update()

    def _on_exec_err(self, msg: str):
        self._log(str(msg))
        notice = show_llm_error_dialog(self, msg, source="Pipeline builder")
        try:
            self.output_edit.set_content(f"{notice.title}\n\n{notice.user_message}")
            self.output_tabs.setCurrentIndex(1)
        except Exception:
            pass
        self.btn_run.setVisible(True)
        self.btn_stop.setVisible(False)
        self._exec_worker = None
        for b in self.canvas.blocks:
            b.selected = False
        self.canvas.update()
