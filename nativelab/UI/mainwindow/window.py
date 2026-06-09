"""Main application window assembled from focused feature mixins."""
from .shared import *
from .auto_setup import AutoSetupMixin
from .context_controls import ContextControlsMixin
from .ui_build import UiBuildMixin
from .models import ModelManagementMixin
from .sessions import SessionsMixin
from .engine_runtime import EngineRuntimeMixin
from .labs import LabsIntegrationMixin
from .chat_pipeline import ChatPipelineMixin
from .documents import DocumentsMixin
from .status_view import StatusViewMixin


class MainWindow(
    AutoSetupMixin,
    ContextControlsMixin,
    UiBuildMixin,
    ModelManagementMixin,
    SessionsMixin,
    EngineRuntimeMixin,
    LabsIntegrationMixin,
    ChatPipelineMixin,
    DocumentsMixin,
    StatusViewMixin,
    QMainWindow,
):
    _lab_context_request = pyqtSignal(int, object)
    _lab_model_request = pyqtSignal(str, object)
    _lab_reload_request = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Native Lab Pro")
        prepare_adaptive_window(
            self,
            1300,
            840,
            min_width=900,
            min_height=600,
            max_width_ratio=0.96,
            max_height_ratio=0.94,
            center=False,
        )

        self.engine   = LlamaEngine()
        self._lab_endpoints = LabEndpoints(self)
        self._integration_endpoints = IntegrationEndpoints(self._lab_endpoints)
        self._lab_context_request.connect(self._handle_labs_context_request)
        self._lab_model_request.connect(self._handle_labs_model_request)
        self._lab_reload_request.connect(self._handle_labs_reload_request)
        self.sessions: Dict[str, Session] = {}
        self.active:   Optional[Session]  = None

        self._worker:          Optional[QThread] = None
        self._stream_w:        Any = None
        self._summary_worker:  Optional[QThread] = None
        self._summary_bubble:  Any = None
        self._pipeline_worker: Any = None
        self.reasoning_engine:     Any = None
        self.summarization_engine: Any = None
        self.coding_engine:        Any = None
        self.secondary_engine:     Any = None
        self._thinking_block:  Any = None
        self._pipeline_reason_w: Any = None
        self._pipeline_code_w:   Any = None
        self._pipeline_insight_widgets: list = []
        self._chat_pipeline_worker: Optional[QThread] = None
        self._api_engine:   Any = None
        self._auto_setup_worker: Optional[AutoSetupWorker] = None

        self._force_coding_mode:  bool = False
        self._pending_ref_ctx:    str  = ""
        self._pending_ref_images: list = []
        self._deferred_send: Optional[tuple[str, str, list]] = None
        # ── busy / session-tracking ───────────────────────────────────────────
        self._busy_session_id:    str  = ""   # sid of the session currently generating
        self._stream_session_id:  str  = ""   # sid that owns the active stream worker
        self._stream_buffer:      str  = ""   # shadow text buffer; survives widget deletion
        self._summary_session_id: str  = ""   # sid that owns the active summary worker
        self._multi_pdf_worker:  Optional[QThread] = None
        self._pause_banner: Optional[QWidget] = None
        self._summarizing_active: bool = False
        self._last_context_snapshot: dict = {}
        self.current_ctx = DEFAULT_CTX()
        self._ctx_reload_timer = QTimer(self)
        self._ctx_reload_timer.setSingleShot(True)
        self._ctx_reload_timer.timeout.connect(self._apply_new_context)

        self._load_sessions()
        ensure_builtin_edit_skill()
        self._build_ui()
        self._build_menu()
        self._build_status_bar()
        context_meter.updated.connect(self._on_context_meter_update)
        # Load saved custom palettes before applying stylesheet
        global C_LIGHT, C_DARK, C, QSS
        set_theme(
            CURRENT_THEME,
            APP_CONFIG.get("custom_light_palette"),
            APP_CONFIG.get("custom_dark_palette"),
        )
        apply_theme_palette(self, C)
        self.setStyleSheet(build_qss(C))
        apply_theme_palette(self, C)
        self.appearance_tab.load_palette(C_LIGHT if CURRENT_THEME == "light" else C_DARK)

        if self.sessions:
            last = max(self.sessions.values(), key=lambda s: s.id)
            self._switch_session(last.id)
        else:
            self._new_session()

        # ── Restore saved theme preference ────────────────────────────────────
        _saved_theme = APP_CONFIG.get("theme", CURRENT_THEME)
        if _saved_theme != CURRENT_THEME:
            self._toggle_theme()   # silently apply saved theme
        self._update_theme_action_label()

        self._set_engine_status("Model not loaded", "idle")

        # Auto-load parallel engines if prefs say so
        if PARALLEL_PREFS.enabled and PARALLEL_PREFS.auto_load_roles:
            QTimer.singleShot(1000, self._auto_load_parallel_engines)
        QTimer.singleShot(350, self._refresh_auto_setup_prompt)
