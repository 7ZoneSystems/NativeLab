from __future__ import annotations

from typing import Any, Callable, Optional

from nativelab.imports.import_global import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    Qt,
    QTextEdit,
    QThread,
    QTimer,
    QVBoxLayout,
    QWidget,
    pyqtSignal,
)
from nativelab.Model.model_global import model_ref_display_name
from nativelab.UI.UI_const import C
from nativelab.UI.buildUI import prepare_adaptive_window
from nativelab.UI.icons import set_button_icon, set_label_icon, set_status_label
from nativelab.UI.llm_error_dialog import show_llm_error_dialog

from .engine_call import generate_pipeline_response
from .context import (
    AiBuilderHistoryStore,
    build_smart_context_request,
    deterministic_compact,
    pipeline_digest,
)
from .planner import (
    AI_BUILDER_N_PREDICT,
    AI_BUILDER_RETRY_N_PREDICT,
    GeneratedPipeline,
    PipelineJsonError,
    build_ai_builder_messages,
    estimate_ai_builder_budget,
    estimate_ai_builder_retry_budget,
    normalize_pipeline_data,
    extract_json_object,
    sanitize_pipeline_name,
    save_generated_pipeline,
)
from .mcp_verify import (
    McpVerificationReport,
    extract_mcp_blocks,
    verify_all_mcp_blocks,
    fix_mcp_blocks_after_verification,
)


class AiPipelineContextCompactWorker(QThread):
    done = pyqtSignal(str)
    err = pyqtSignal(str)

    def __init__(self, engine: Any, *, history_text: str, summary: str, canvas_state: Any, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._history_text = history_text
        self._summary = summary
        self._canvas_state = canvas_state

    def run(self):
        try:
            if self._engine is None or not getattr(self._engine, "is_loaded", False):
                self.done.emit(deterministic_compact(
                    _InlineHistory(self._summary, self._history_text),  # type: ignore[arg-type]
                    self._canvas_state,
                ))
                return
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Compact NativeLab AI Pipeline Builder history. "
                        "Preserve user goals, chosen pipeline structure, unresolved constraints, "
                        "model/canvas assumptions, and pending edits. Return concise plain text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Existing compact summary:\n{self._summary or '(none)'}\n\n"
                        f"Recent history:\n{self._history_text or '(none)'}\n\n"
                        f"Current canvas state:\n{self._canvas_state}"
                    ),
                },
            ]
            text = generate_pipeline_response(
                self._engine,
                messages,
                n_predict=700,
                temperature=0.1,
            )
            self.done.emit(str(text or "").strip())
        except Exception as exc:
            self.err.emit(str(exc))


class _InlineHistory:
    def __init__(self, summary: str, recent: str):
        self._summary = summary
        self._recent = recent

    def recent_text(self, limit: int = 6) -> str:
        _ = limit
        return self._recent


class AiPipelineBuildWorker(QThread):
    done = pyqtSignal(object)
    err = pyqtSignal(str)
    status = pyqtSignal(str)
    # Emitted when MCP servers need auth: list of (block_bid, block_label, url, error)
    mcp_auth_needed = pyqtSignal(list)

    def __init__(
        self,
        engine: Any,
        *,
        pipeline_name: str,
        user_request: str,
        active_model_ref: str = "",
        active_model_role: str = "general",
        active_model_label: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._engine = engine
        self._pipeline_name = pipeline_name
        self._user_request = user_request
        self._active_model_ref = active_model_ref
        self._active_model_role = active_model_role
        self._active_model_label = active_model_label
        self._abort = False
        self._auth_event = None  # threading.Event set when user provides auth
        self._auth_updates: dict = {}  # bid → {mcp_auth_token, mcp_auth_env}

    def abort(self):
        self._abort = True

    def provide_auth(self, auth_updates: dict):
        """
        Called from UI thread when user has configured auth for MCP servers.
        auth_updates: {block_bid: {"mcp_auth_token": str, "mcp_auth_env": str}}
        """
        self._auth_updates = auth_updates
        if self._auth_event:
            self._auth_event.set()

    def _raw_preview(self, raw: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return "(empty response)"
        if len(text) > 1200:
            return text[:1200] + "\n...[truncated]"
        return text

    def _save_or_retry(self, raw: str) -> GeneratedPipeline:
        try:
            return save_generated_pipeline(
                self._pipeline_name,
                raw,
                active_model_ref=self._active_model_ref,
                active_model_role=self._active_model_role,
            )
        except PipelineJsonError as first_error:
            self.status.emit(
                "The first model response was not valid pipeline JSON. "
                "Retrying once with a stricter JSON-only prompt..."
            )
            self.status.emit("First raw response preview:\n" + self._raw_preview(first_error.raw_response))
            retry_budget = estimate_ai_builder_retry_budget(
                self._engine,
                self._pipeline_name,
                self._user_request,
                active_model_label=self._active_model_label,
                previous_response=first_error.raw_response,
                n_predict=AI_BUILDER_RETRY_N_PREDICT,
            )
            if retry_budget.overflow:
                raise PipelineJsonError(
                    "The model did not return JSON, and the stricter retry prompt would exceed "
                    f"the current context limit ({retry_budget.projected_tokens} / "
                    f"{retry_budget.limit_tokens} tokens). Increase context and reload the model, "
                    "or shorten the request.",
                    first_error.raw_response,
                ) from first_error
            retry_raw = generate_pipeline_response(
                self._engine,
                retry_budget.messages,
                n_predict=AI_BUILDER_RETRY_N_PREDICT,
                temperature=0.0,
                abort_cb=lambda: self._abort,
            )
            if self._abort:
                raise PipelineJsonError("AI pipeline build was cancelled.", retry_raw) from first_error
            try:
                return save_generated_pipeline(
                    self._pipeline_name,
                    retry_raw,
                    active_model_ref=self._active_model_ref,
                    active_model_role=self._active_model_role,
                )
            except PipelineJsonError as retry_error:
                self.status.emit("Retry raw response preview:\n" + self._raw_preview(retry_error.raw_response))
                raise PipelineJsonError(
                    "The model still did not return valid pipeline JSON after a strict retry. "
                    "Try a more direct request, for example: 'Make a 3 block input -> model -> output pipeline'.",
                    retry_error.raw_response,
                ) from retry_error

    def _verify_and_fix_mcp(self, raw: str) -> GeneratedPipeline:
        """
        Parse generated JSON, verify MCP servers, fix/remove failed blocks,
        then save. If MCP blocks were removed, asks the AI to regenerate
        with the failure info so it can adapt.
        """
        # Parse the raw JSON
        data = extract_json_object(raw)
        data = normalize_pipeline_data(data)

        blocks = data.get("blocks", [])
        mcp_blocks = extract_mcp_blocks(blocks)

        if not mcp_blocks:
            # No MCP blocks - proceed normally
            return self._save_or_retry(raw)

        self.status.emit(
            f"Found {len(mcp_blocks)} MCP server block(s). "
            f"Verifying connections before saving..."
        )

        # Verify all MCP servers
        report = verify_all_mcp_blocks(
            blocks,
            max_retries=2,
            auto_install=True,
            log_cb=lambda m: self.status.emit(m),
            abort_cb=lambda: self._abort,
        )

        if self._abort:
            raise PipelineJsonError("AI pipeline build was cancelled during MCP verification.", raw)

        # Handle servers that need authentication
        if report.auth_needed and not self._abort:
            auth_list = [
                (r.block_bid, r.block_label, r.url, r.auth_error_detail)
                for r in report.auth_needed
            ]
            self.status.emit(
                f"🔒 {len(auth_list)} MCP server(s) require authentication. "
                f"Waiting for user to configure credentials..."
            )
            self.mcp_auth_needed.emit(auth_list)

            # Wait for user to provide auth (or abort)
            import threading
            self._auth_event = threading.Event()
            self._auth_event.wait()  # Blocks until provide_auth() or abort
            self._auth_event = None

            if self._abort:
                raise PipelineJsonError("AI pipeline build was cancelled during auth setup.", raw)

            # Apply auth updates to blocks
            if self._auth_updates:
                for block in blocks:
                    bid = block.get("bid", 0)
                    if bid in self._auth_updates:
                        block["metadata"].update(self._auth_updates[bid])
                self.status.emit("Auth credentials received. Re-verifying MCP servers...")
                # Re-run verification with updated auth
                report = verify_all_mcp_blocks(
                    blocks,
                    max_retries=1,
                    auto_install=False,
                    log_cb=lambda m: self.status.emit(m),
                    abort_cb=lambda: self._abort,
                )
                if self._abort:
                    raise PipelineJsonError("AI pipeline build was cancelled.", raw)
            self._auth_updates = {}

        if report.all_ok:
            # All MCP servers verified - update blocks with verified tools
            for block in blocks:
                bid = block.get("bid", 0)
                for r in report.results:
                    if r.block_bid == bid and r.success:
                        block["metadata"]["mcp_connected"] = True
                        block["metadata"]["mcp_tools"] = r.tools_found
            # Save with verified data
            from .planner import apply_active_model, pipeline_data_to_blocks, save_pipeline
            from ..validation import validate_pipeline
            data = apply_active_model(data, self._active_model_ref, self._active_model_role)
            blocks_final, connections = pipeline_data_to_blocks(data)
            err = validate_pipeline(blocks_final, connections)
            if err:
                raise ValueError(f"Generated pipeline did not pass validation:\n\n{err}")
            from ..pipefunctions import save_pipeline as _save
            _save(self._pipeline_name, blocks_final, connections)
            return GeneratedPipeline(
                name=self._pipeline_name,
                raw_response=raw,
                data=data,
                blocks=blocks_final,
                connections=connections,
            )

        # Some MCP servers failed - fix blocks and try to regenerate
        self.status.emit("Some MCP servers failed verification. Fixing pipeline...")

        fixed_blocks, fix_messages = fix_mcp_blocks_after_verification(
            blocks, report, log_cb=lambda m: self.status.emit(m),
        )

        for msg in fix_messages:
            self.status.emit(f"  → {msg}")

        if report.removed_blocks:
            # Ask AI to regenerate with info about which MCP servers failed
            failed_info = "; ".join(
                f"'{r.block_label}' ({r.url[:40]}): {r.error[:60]}"
                for r in report.failed
            )
            self.status.emit(
                f"Removed {len(report.removed_blocks)} MCP block(s). "
                f"Asking AI to adapt the pipeline..."
            )

            # Build a retry request that tells the AI what failed
            retry_request = (
                f"{self._user_request}\n\n"
                f"IMPORTANT: The following MCP servers were unreachable and have been removed: "
                f"{failed_info}. "
                f"Rebuild the pipeline WITHOUT those MCP blocks. "
                f"Use alternative approaches (local model, custom code, etc.) "
                f"to accomplish the same goal."
            )

            retry_messages = build_ai_builder_messages(
                retry_request,
                self._pipeline_name,
                active_model_label=self._active_model_label,
            )

            try:
                retry_raw = generate_pipeline_response(
                    self._engine,
                    retry_messages,
                    n_predict=AI_BUILDER_N_PREDICT,
                    temperature=0.15,
                    abort_cb=lambda: self._abort,
                )
                if self._abort:
                    raise PipelineJsonError("AI pipeline build was cancelled.", retry_raw)

                # Save the regenerated pipeline (no more MCP verification needed
                # since we told the AI not to use MCP)
                return self._save_or_retry(retry_raw)

            except PipelineJsonError:
                # If regeneration fails, save with fixed blocks
                self.status.emit("AI regeneration failed. Saving pipeline with fixed blocks...")
                data["blocks"] = fixed_blocks
                from .planner import apply_active_model, pipeline_data_to_blocks
                from ..validation import validate_pipeline
                from ..pipefunctions import save_pipeline as _save
                data = apply_active_model(data, self._active_model_ref, self._active_model_role)
                blocks_final, connections = pipeline_data_to_blocks(data)
                err = validate_pipeline(blocks_final, connections)
                if err:
                    raise ValueError(f"Fixed pipeline did not pass validation:\n\n{err}")
                _save(self._pipeline_name, blocks_final, connections)
                return GeneratedPipeline(
                    name=self._pipeline_name,
                    raw_response=raw,
                    data=data,
                    blocks=blocks_final,
                    connections=connections,
                )

        # All MCP blocks were fixed (not removed) - save with fixed data
        data["blocks"] = fixed_blocks
        from .planner import apply_active_model, pipeline_data_to_blocks
        from ..validation import validate_pipeline
        from ..pipefunctions import save_pipeline as _save
        data = apply_active_model(data, self._active_model_ref, self._active_model_role)
        blocks_final, connections = pipeline_data_to_blocks(data)
        err = validate_pipeline(blocks_final, connections)
        if err:
            raise ValueError(f"Fixed pipeline did not pass validation:\n\n{err}")
        _save(self._pipeline_name, blocks_final, connections)
        return GeneratedPipeline(
            name=self._pipeline_name,
            raw_response=raw,
            data=data,
            blocks=blocks_final,
            connections=connections,
        )

    def run(self):
        try:
            budget = estimate_ai_builder_budget(
                self._engine,
                self._pipeline_name,
                self._user_request,
                active_model_label=self._active_model_label,
                n_predict=AI_BUILDER_N_PREDICT,
            )
            self.status.emit(
                f"Sending builder prompt: {budget.input_tokens} input tokens, "
                f"{budget.reserved_tokens} reserved output tokens."
            )
            raw = generate_pipeline_response(
                self._engine,
                budget.messages,
                n_predict=AI_BUILDER_N_PREDICT,
                temperature=0.15,
                abort_cb=lambda: self._abort,
            )
            if self._abort:
                self.err.emit("AI pipeline build was cancelled.")
                return
            self.status.emit("Validating and saving generated pipeline...")
            # Use MCP-aware verification: tests servers, installs packages,
            # retries on failure, removes blocks that can't connect
            try:
                result = self._verify_and_fix_mcp(raw)
            except Exception as mcp_err:
                self.status.emit(
                    f"MCP/Web Search verification failed: {mcp_err}. "
                    f"Saving pipeline without verification.")
                result = self._save_or_retry(raw)
            self.done.emit(result)
        except Exception as exc:
            self.err.emit(str(exc))


class AiPipelineBuilderPanel(QWidget):
    def __init__(
        self,
        engine: Any,
        *,
        parent=None,
        load_callback: Optional[Callable[[str], None]] = None,
        close_callback: Optional[Callable[[], None]] = None,
        active_model_provider: Optional[Callable[[], tuple[str, str]]] = None,
        canvas_state_provider: Optional[Callable[[], dict]] = None,
        history_store: Optional[AiBuilderHistoryStore] = None,
        active_model_ref: str = "",
        active_model_role: str = "general",
        show_close: bool = False,
        compact: bool = False,
    ):
        super().__init__(parent)
        self._engine = engine
        self._load_callback = load_callback
        self._close_callback = close_callback
        self._active_model_provider = active_model_provider
        self._canvas_state_provider = canvas_state_provider
        self._active_model_ref = active_model_ref
        self._active_model_role = active_model_role or "general"
        self._show_close = bool(show_close)
        self._compact = bool(compact)
        self._worker: Optional[AiPipelineBuildWorker] = None
        self._context_worker: Optional[AiPipelineContextCompactWorker] = None
        self._history = history_store or AiBuilderHistoryStore()
        self._generated_name = ""
        self._pending_user_request = ""
        self._adaptive_labels: list[QLabel] = []
        self._adaptive_buttons: list[QPushButton] = []
        self._adaptive_fields: list[QWidget] = []
        self._sidebar_width = 0
        self._root_layout = None
        self._build()
        self._apply_sidebar_width()
        self._refresh_budget()
        self._update_history_label()

    def set_engine(self, engine: Any):
        self._engine = engine
        self._refresh_budget()

    def _current_active_model(self) -> tuple[str, str]:
        if callable(self._active_model_provider):
            try:
                ref, role = self._active_model_provider()
                return str(ref or ""), str(role or "general")
            except Exception:
                pass
        return self._active_model_ref, self._active_model_role

    def _active_model_label(self) -> str:
        model_ref, _role = self._current_active_model()
        if model_ref:
            return model_ref_display_name(model_ref)
        model_path = str(getattr(self._engine, "model_path", "") or "")
        return model_ref_display_name(model_path) if model_path else ""

    def _canvas_state(self) -> dict:
        if callable(self._canvas_state_provider):
            try:
                state = self._canvas_state_provider()
                return state if isinstance(state, dict) else {}
            except Exception:
                return {}
        return {}

    def _build(self):
        self.setMinimumWidth(0)
        try:
            self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        except Exception:
            pass

        root = QVBoxLayout(self)
        self._root_layout = root
        if self._compact:
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(7)
        else:
            root.setContentsMargins(16, 14, 16, 14)
            root.setSpacing(10)

        hdr = QLabel("AI Pipeline Builder")
        set_label_icon(hdr, "brain", "AI Pipeline Builder", 16 if self._compact else 18)
        hdr.setObjectName("pipeline_hdr")
        hdr.setStyleSheet(f"font-size:{12 if self._compact else 14}px;font-weight:700;")
        self._prepare_label(hdr)
        root.addWidget(hdr)

        hint = QLabel(
            "Describe the pipeline you want. The loaded model will return NativeLab pipeline JSON, "
            "then NativeLab validates and saves it through the normal pipeline subsystem."
        )
        hint.setWordWrap(True)
        hint.setObjectName("txt2")
        self._prepare_label(hint)
        root.addWidget(hint)

        self.history_lbl = QLabel("")
        self.history_lbl.setObjectName("txt3_block")
        self.history_lbl.setWordWrap(True)
        self._prepare_label(self.history_lbl)
        self.btn_context = QPushButton("Context")
        set_button_icon(self.btn_context, "summary", "Context")
        self.btn_context.setFixedHeight(24)
        self._prepare_button(self.btn_context)
        self.btn_context.clicked.connect(self._start_context_compact)
        self.btn_clear_history = QPushButton("Clear")
        set_button_icon(self.btn_clear_history, "clear", "Clear")
        self.btn_clear_history.setFixedHeight(24)
        self._prepare_button(self.btn_clear_history)
        self.btn_clear_history.clicked.connect(self._clear_history)
        if self._compact:
            hist_col = QVBoxLayout()
            hist_col.setSpacing(4)
            hist_col.addWidget(self.history_lbl)
            hist_btn_row = QHBoxLayout()
            hist_btn_row.setSpacing(6)
            hist_btn_row.addWidget(self.btn_context)
            hist_btn_row.addWidget(self.btn_clear_history)
            hist_col.addLayout(hist_btn_row)
            root.addLayout(hist_col)
        else:
            hist_row = QHBoxLayout()
            hist_row.setSpacing(6)
            hist_row.addWidget(self.history_lbl, 1)
            hist_row.addWidget(self.btn_context)
            hist_row.addWidget(self.btn_clear_history)
            root.addLayout(hist_row)

        name_lbl = QLabel("Output JSON name")
        name_lbl.setObjectName("txt3_tiny")
        self._prepare_label(name_lbl)
        root.addWidget(name_lbl)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("example: research-triage-pipeline")
        self.name_edit.setText("ai-pipeline")
        self._prepare_field(self.name_edit)
        self.name_edit.textChanged.connect(self._refresh_budget)
        root.addWidget(self.name_edit)

        req_lbl = QLabel("Pipeline request")
        req_lbl.setObjectName("txt3_tiny")
        self._prepare_label(req_lbl)
        root.addWidget(req_lbl)
        self.request_edit = QTextEdit()
        self.request_edit.setPlaceholderText(
            "Example: Build a pipeline that takes a user question, adds reference context, "
            "classifies urgency, summarizes sources, and returns a final answer."
        )
        self._prepare_field(self.request_edit)
        if self._compact:
            self.request_edit.setMaximumHeight(138)
        self.request_edit.textChanged.connect(self._refresh_budget)
        root.addWidget(self.request_edit, 1)

        # ── Device Capabilities Panel ──
        self._device_panel = self._build_device_panel()
        root.addWidget(self._device_panel)

        self.budget_lbl = QLabel("")
        self.budget_lbl.setObjectName("status_badge")
        self.budget_lbl.setProperty("state", "idle")
        self.budget_lbl.setWordWrap(True)
        self._prepare_label(self.budget_lbl)
        root.addWidget(self.budget_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        self.status_edit = QTextEdit()
        self.status_edit.setReadOnly(True)
        self.status_edit.setMaximumHeight(150 if self._compact else 108)
        self.status_edit.setObjectName("log_te")
        self._prepare_field(self.status_edit)
        root.addWidget(self.status_edit)

        btn_row = QVBoxLayout() if self._compact else QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_build = QPushButton("Build & Save")
        set_button_icon(self.btn_build, "brain", "Build & Save")
        self.btn_build.setObjectName("btn_send")
        self.btn_build.setFixedHeight(32)
        self._prepare_button(self.btn_build)
        self.btn_build.clicked.connect(self._start_build)
        btn_row.addWidget(self.btn_build)

        self.btn_load = QPushButton("Load / Test")
        set_button_icon(self.btn_load, "play", "Load / Test")
        self.btn_load.setFixedHeight(32)
        self.btn_load.setEnabled(False)
        self._prepare_button(self.btn_load)
        self.btn_load.clicked.connect(self._load_generated_pipeline)
        btn_row.addWidget(self.btn_load)

        if not self._compact:
            btn_row.addStretch()
        self.btn_close = None
        if self._show_close:
            self.btn_close = QPushButton("Close")
            set_button_icon(self.btn_close, "x", "Close")
            self.btn_close.setFixedHeight(32)
            self._prepare_button(self.btn_close)
            self.btn_close.clicked.connect(self._request_close)
            btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

    def _prepare_label(self, label: QLabel):
        try:
            label.setMinimumWidth(0)
        except Exception:
            pass
        try:
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        except Exception:
            pass
        self._adaptive_labels.append(label)

    def _prepare_button(self, button: QPushButton):
        try:
            button.setMinimumWidth(0)
        except Exception:
            pass
        try:
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        except Exception:
            pass
        self._adaptive_buttons.append(button)

    def _prepare_field(self, field: QWidget):
        try:
            field.setMinimumWidth(0)
        except Exception:
            pass
        try:
            field.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        except Exception:
            pass
        self._adaptive_fields.append(field)

    def _build_device_panel(self) -> QWidget:
        """Build collapsible device capabilities panel."""
        from nativelab.Model.APImodels import getapi_registry, is_phonolab_device
        from nativelab.UI.icons import icon

        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Header row (clickable to expand/collapse)
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(6)
        hdr_icon = QLabel()
        hdr_icon.setPixmap(icon("globe").pixmap(14, 14))
        hdr_row.addWidget(hdr_icon)
        hdr_label = QLabel("Device Capabilities")
        hdr_label.setStyleSheet(f"font-size:11px;font-weight:bold;color:{C.get('txt2', '#7a7a9a')};")
        hdr_row.addWidget(hdr_label, 1)
        self._device_expand_btn = QPushButton("+")
        self._device_expand_btn.setFixedSize(20, 20)
        self._device_expand_btn.setStyleSheet(f"QPushButton {{ background:transparent;color:{C.get('txt2', '#7a7a9a')};border:none;font-weight:bold; }}")
        hdr_row.addWidget(self._device_expand_btn)
        outer.addLayout(hdr_row)

        # Content (hidden by default)
        self._device_content = QWidget()
        content_layout = QVBoxLayout(self._device_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        self._device_content.setVisible(False)

        # Device list
        registry = getapi_registry()
        devices = [cfg for cfg in registry.all() if is_phonolab_device(cfg)]

        if not devices:
            no_dev = QLabel("No PhonoLab devices registered. Use Dev > Devices to scan and register.")
            no_dev.setWordWrap(True)
            no_dev.setStyleSheet(f"color:{C.get('txt3', '#48485e')};font-size:11px;")
            content_layout.addWidget(no_dev)
        else:
            self._device_sliders = {}
            for cfg in devices:
                dev_frame = QFrame()
                dev_frame.setStyleSheet(f"QFrame {{ background:{C.get('bg', '#09090d')};border:1px solid {C.get('bdr', '#252538')};border-radius:6px; padding:6px; }}")
                dev_layout = QVBoxLayout(dev_frame)
                dev_layout.setContentsMargins(8, 6, 8, 6)
                dev_layout.setSpacing(4)

                # Device name + live status
                name_row = QHBoxLayout()
                name_row.setSpacing(6)
                name_lbl = QLabel(cfg.name)
                name_lbl.setStyleSheet(f"font-size:12px;font-weight:bold;color:{C.get('txt', '#ededf5')};")
                name_row.addWidget(name_lbl, 1)

                status_lbl = QLabel(getattr(cfg, "device_status", "unknown").upper())
                status_lbl.setStyleSheet(f"font-size:10px;color:{C.get('txt2', '#7a7a9a')};")
                name_row.addWidget(status_lbl)
                dev_layout.addLayout(name_row)

                # Live model info
                info_lbl = QLabel(f"Model: {cfg.model_id} | {getattr(cfg, 'cpu_cores', 0)} cores | {getattr(cfg, 'ram_mb', 0)}MB")
                info_lbl.setStyleSheet(f"font-size:10px;color:{C.get('txt3', '#48485e')};")
                dev_layout.addWidget(info_lbl)

                sliders = {}

                # Temperature
                temp_row, temp_slider, temp_val = self._make_slider(
                    "Temperature", 0, 200, int(cfg.temperature * 100), f"{cfg.temperature:.2f}")
                dev_layout.addLayout(temp_row)
                sliders["temperature"] = (temp_slider, 100)

                # Top-P
                topp_row, topp_slider, topp_val = self._make_slider(
                    "Top-P", 0, 100, 90, "0.90")
                dev_layout.addLayout(topp_row)
                sliders["top_p"] = (topp_slider, 100)

                # Top-K
                topk_row, topk_slider, topk_val = self._make_slider(
                    "Top-K", 0, 200, 40, "40")
                dev_layout.addLayout(topk_row)
                sliders["top_k"] = (topk_slider, 1)

                # Repeat Penalty
                rep_row, rep_slider, rep_val = self._make_slider(
                    "Repeat Penalty", 100, 200, 110, "1.10")
                dev_layout.addLayout(rep_row)
                sliders["repeat_penalty"] = (rep_slider, 100)

                # Max Tokens
                tok_row, tok_slider, tok_val = self._make_slider(
                    "Max Tokens", 64, 4096, 512, "512")
                dev_layout.addLayout(tok_row)
                sliders["max_tokens"] = (tok_slider, 1)

                # Apply button
                apply_btn = QPushButton(f"Apply to {cfg.name}")
                apply_btn.setFixedHeight(26)
                apply_btn.setStyleSheet(f"QPushButton {{ background:{C.get('acc', '#55C2A4')};color:#fff;border:none;border-radius:4px;font-weight:bold;font-size:11px; }} QPushButton:hover {{ background:{C.get('acc2', '#3da88a')}; }}")
                apply_btn.clicked.connect(lambda checked, c=cfg, s=sliders: self._apply_device_config(c, s))
                dev_layout.addWidget(apply_btn)

                content_layout.addWidget(dev_frame)
                self._device_sliders[cfg.name] = sliders

        outer.addWidget(self._device_content)

        # Toggle expand/collapse
        self._device_expand_btn.clicked.connect(self._toggle_device_panel)

        return panel

    def _make_slider(self, label: str, min_val: int, max_val: int, default: int, fmt: str):
        """Helper to create a labeled slider row."""
        row = QHBoxLayout()
        row.setSpacing(6)

        lbl = QLabel(f"{label}: {fmt}")
        lbl.setFixedWidth(110)
        lbl.setStyleSheet(f"font-size:11px;color:{C.get('txt', '#ededf5')};")
        row.addWidget(lbl)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setStyleSheet(
            f"QSlider::groove:horizontal {{ background:{C.get('bg', '#09090d')};height:4px;border-radius:2px; }}"
            f"QSlider::handle:horizontal {{ background:{C.get('acc', '#55C2A4')};width:12px;height:12px;margin:-4px 0;border-radius:6px; }}"
        )
        row.addWidget(slider, 1)

        val_lbl = QLabel(fmt)
        val_lbl.setFixedWidth(40)
        val_lbl.setStyleSheet(f"font-size:11px;color:{C.get('txt2', '#7a7a9a')};")
        row.addWidget(val_lbl)

        def _update(v):
            if label == "Temperature" or label == "Top-P" or label == "Repeat Penalty":
                val_lbl.setText(f"{v/100:.2f}")
            else:
                val_lbl.setText(str(v))

        slider.valueChanged.connect(_update)

        return row, slider, val_lbl

    def _toggle_device_panel(self):
        visible = not self._device_content.isVisible()
        self._device_content.setVisible(visible)
        self._device_expand_btn.setText("-" if visible else "+")

    def _apply_device_config(self, cfg, sliders: dict):
        """Apply slider values to a device via the backend."""
        from nativelab.core.backend import get_backend
        from nativelab.api_server.device_discovery import DiscoveredDevice

        config = {}
        for key, (slider, divisor) in sliders.items():
            val = slider.value()
            if divisor > 1:
                config[key] = val / divisor
            else:
                config[key] = val

        device = DiscoveredDevice(
            ip=cfg.base_url.split("//")[1].split(":")[0],
            port=int(cfg.base_url.split(":")[-1].split("/")[0]) if ":" in cfg.base_url else 8787,
            api_key=cfg.api_key,
        )

        backend = get_backend()
        result = backend.update_device_config(device, config)
        if result.ok:
            self._append_status(f"Config applied to {cfg.name}")
        else:
            self._append_status(f"Failed to apply config to {cfg.name}: {result.error}")

    def _append_status(self, msg: str):
        """Append a message to the status text edit."""
        self.status_edit.append(msg)

    def set_sidebar_width(self, width: int):
        self._sidebar_width = max(0, int(width or 0))
        self._apply_sidebar_width()

    def resizeEvent(self, event):  # type: ignore[override]
        try:
            super().resizeEvent(event)
        except Exception:
            pass
        QTimer.singleShot(0, self._apply_sidebar_width)

    def _apply_sidebar_width(self):
        if not self._compact:
            return
        width = self._sidebar_width or max(0, int(self.width() or 0))
        if not width:
            return
        tight = width < 225
        very_tight = width < 190
        body_px = 9 if very_tight else 10 if tight else 11
        header_px = 10 if very_tight else 11 if tight else 12
        margin = 4 if very_tight else 6 if tight else 8
        spacing = 4 if tight else 7
        if self._root_layout is not None:
            try:
                self._root_layout.setContentsMargins(margin, margin, margin, margin)
                self._root_layout.setSpacing(spacing)
            except Exception:
                pass
        for label in self._adaptive_labels:
            try:
                font = label.font()
                font.setPointSize(header_px if label.objectName() == "pipeline_hdr" else body_px)
                label.setFont(font)
            except Exception:
                pass
        for button in self._adaptive_buttons:
            try:
                font = button.font()
                font.setPointSize(body_px)
                button.setFont(font)
                button.setFixedHeight(28 if tight else 32)
            except Exception:
                pass
        try:
            self.btn_context.setFixedHeight(24 if not very_tight else 22)
            self.btn_clear_history.setFixedHeight(24 if not very_tight else 22)
        except Exception:
            pass
        try:
            self.request_edit.setMaximumHeight(104 if very_tight else 122 if tight else 138)
            self.status_edit.setMaximumHeight(118 if very_tight else 134 if tight else 150)
        except Exception:
            pass
        try:
            self.setStyleSheet(
                f"QWidget{{font-size:{body_px}px;}}"
                f"QLabel{{font-size:{body_px}px;}}"
                f"QLineEdit,QTextEdit,QPushButton{{font-size:{body_px}px;}}"
                f"QLabel#pipeline_hdr{{font-size:{header_px}px;font-weight:700;}}"
            )
        except Exception:
            pass

    def _request_close(self):
        if self._is_running():
            QMessageBox.information(
                self,
                "AI Pipeline Builder",
                "Wait for the current build to finish before closing this window."
            )
            return
        if callable(self._close_callback):
            self._close_callback()

    def _log(self, text: str):
        self.status_edit.append(str(text))

    def _update_history_label(self):
        count = len(self._history.messages)
        compacted = " · compacted" if self._history.summary.strip() else ""
        self.history_lbl.setText(f"History: {count} turn(s){compacted}")

    def _clear_history(self):
        if self._is_running() or self._is_context_running():
            return
        self._history.clear()
        self._update_history_label()
        self._log("AI Builder history cleared.")

    def _is_context_running(self) -> bool:
        return bool(self._context_worker and self._context_worker.isRunning())

    def _smart_context(self, request: str):
        return build_smart_context_request(
            request,
            history=self._history,
            canvas_state=self._canvas_state(),
        )

    def _set_budget_state(self, state: str):
        self.budget_lbl.setProperty("state", state)
        s = self.budget_lbl.style()
        if s:
            s.unpolish(self.budget_lbl)
            s.polish(self.budget_lbl)

    def _refresh_budget(self):
        name = sanitize_pipeline_name(self.name_edit.text() if hasattr(self, "name_edit") else "")
        request = self.request_edit.toPlainText() if hasattr(self, "request_edit") else ""
        smart = self._smart_context(request)
        request_for_model = smart.model_request or request
        budget = estimate_ai_builder_budget(
            self._engine,
            name,
            request_for_model,
            active_model_label=self._active_model_label(),
            n_predict=AI_BUILDER_N_PREDICT,
        )
        self.budget_lbl.setText(
            f"Context preflight: {budget.input_tokens} input + "
            f"{budget.reserved_tokens} output reserve = {budget.projected_tokens} / "
            f"{budget.limit_tokens} tokens"
        )
        self._set_budget_state("warn" if budget.overflow else "ok")

    def _is_running(self) -> bool:
        return bool(self._worker and self._worker.isRunning())

    def _set_running(self, running: bool):
        self.btn_build.setEnabled(not running)
        self.btn_context.setEnabled(not running)
        self.btn_clear_history.setEnabled(not running)
        if self.btn_close is not None:
            self.btn_close.setEnabled(not running)
        self.name_edit.setEnabled(not running)
        self.request_edit.setEnabled(not running)
        if running:
            self.btn_load.setEnabled(False)

    def _start_build(self):
        if self._is_running():
            return
        if self._engine is None or not getattr(self._engine, "is_loaded", False):
            QMessageBox.warning(self, "AI Pipeline Builder", "Load a model before using AI Pipeline Builder.")
            return
        request = self.request_edit.toPlainText().strip()
        if not request:
            QMessageBox.warning(self, "AI Pipeline Builder", "Describe the pipeline you want to create.")
            return
        smart = self._smart_context(request)
        if smart.command == "get_data":
            self._log(smart.notice)
            return
        if smart.command == "context":
            self._start_context_compact()
            return
        name = sanitize_pipeline_name(self.name_edit.text())
        if self.name_edit.text().strip() != name:
            self.name_edit.setText(name)
        budget = estimate_ai_builder_budget(
            self._engine,
            name,
            smart.model_request,
            active_model_label=self._active_model_label(),
            n_predict=AI_BUILDER_N_PREDICT,
        )
        if budget.overflow:
            QMessageBox.warning(
                self,
                "Context Limit Too Small",
                "The AI pipeline builder prompt is too large for the loaded model context.\n\n"
                f"Estimated: {budget.input_tokens} input tokens + "
                f"{budget.reserved_tokens} output tokens = {budget.projected_tokens}.\n"
                f"Current context limit: {budget.limit_tokens}.\n\n"
                "Increase the model context limit and reload the model, or shorten the pipeline request."
            )
            self._log("Context preflight blocked the request before sending it to the model.")
            return

        active_model_ref, active_model_role = self._current_active_model()
        self._generated_name = ""
        self._pending_user_request = smart.user_request
        self.status_edit.clear()
        self._set_running(True)
        self._log("Building pipeline with loaded model...")
        if smart.notice:
            self._log(smart.notice)
        self._history.append_user(smart.user_request)
        self._update_history_label()
        self._worker = AiPipelineBuildWorker(
            self._engine,
            pipeline_name=name,
            user_request=smart.model_request,
            active_model_ref=active_model_ref,
            active_model_role=active_model_role,
            active_model_label=self._active_model_label(),
            parent=self,
        )
        self._worker.status.connect(self._log)
        self._worker.done.connect(self._on_done)
        self._worker.err.connect(self._on_error)
        self._worker.mcp_auth_needed.connect(self._on_mcp_auth_needed)
        self._worker.finished.connect(lambda: self._set_running(False))
        self._worker.start()

    def _on_done(self, result: GeneratedPipeline):
        self._generated_name = result.name
        digest = pipeline_digest(result.data)
        self._history.append_assistant(digest, pipeline=result.name)
        self._pending_user_request = ""
        self._update_history_label()
        self._log(
            f"Saved '{result.name}' with {len(result.blocks)} blocks and "
            f"{len(result.connections)} connections."
        )
        self.btn_load.setEnabled(True)
        QMessageBox.information(
            self,
            "Pipeline Saved",
            f"Pipeline '{result.name}' was saved successfully.\n\n"
            "Use Load / Test to place it on the canvas."
        )

    def _on_error(self, message: str):
        if self._pending_user_request:
            self._history.append_assistant(f"Error: {message}", error=True)
            self._pending_user_request = ""
            self._update_history_label()
        self._log(message)
        show_llm_error_dialog(self, message, source="AI Pipeline Builder")

    def _on_mcp_auth_needed(self, auth_list: list):
        """
        Show auth configuration dialog for MCP servers that need credentials.
        auth_list: [(block_bid, block_label, url, error_detail), ...]
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("MCP Server Authentication Required")
        prepare_adaptive_window(dlg, 560, 420, min_width=440, min_height=340)
        root = QVBoxLayout(dlg)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        hdr = QLabel("MCP servers need authentication")
        hdr.setStyleSheet(f"color:{C['txt']};font-size:14px;font-weight:bold;")
        root.addWidget(hdr)

        info = QLabel(
            "The following MCP servers require credentials (API key, token, or "
            "environment variables) before they can connect. Configure them below "
            "or click Skip to remove these servers from the pipeline."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        root.addWidget(info)

        # Per-server auth entries
        entries = {}  # bid → (token_edit, env_edit)
        for bid, label, url, error in auth_list:
            card = QFrame(); card.setObjectName("tab_card")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(12, 10, 12, 10); cl.setSpacing(6)

            title = QLabel(f"🔒 {label}")
            title.setStyleSheet(
                f"color:{C['txt']};font-size:12px;font-weight:600;")
            cl.addWidget(title)

            url_lbl = QLabel(f"URL: {url[:60]}")
            url_lbl.setStyleSheet(f"color:{C['txt3']};font-size:10px;")
            cl.addWidget(url_lbl)

            if error:
                err_lbl = QLabel(f"Error: {error[:120]}")
                err_lbl.setWordWrap(True)
                err_lbl.setStyleSheet(f"color:{C['err']};font-size:10px;")
                cl.addWidget(err_lbl)

            token_edit = QLineEdit()
            token_edit.setPlaceholderText("API key or bearer token")
            token_edit.setEchoMode(QLineEdit.EchoMode.Password)
            token_edit.setFixedHeight(26)
            trow = QHBoxLayout(); trow.setSpacing(6)
            tlbl = QLabel("Token:"); tlbl.setFixedWidth(60)
            tlbl.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
            trow.addWidget(tlbl); trow.addWidget(token_edit, 1)
            cl.addLayout(trow)

            env_edit = QLineEdit()
            env_edit.setPlaceholderText("KEY=value (comma-separated, e.g. GITHUB_TOKEN=ghp_xxx,DATABASE_URL=...)")
            env_edit.setFixedHeight(26)
            erow = QHBoxLayout(); erow.setSpacing(6)
            elbl = QLabel("Env vars:"); elbl.setFixedWidth(60)
            elbl.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
            erow.addWidget(elbl); erow.addWidget(env_edit, 1)
            cl.addLayout(erow)

            entries[bid] = (token_edit, env_edit)
            root.addWidget(card)

        # Buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        btn_skip = QPushButton("Skip (remove these servers)")
        btn_skip.setFixedHeight(30)
        btn_skip.clicked.connect(dlg.reject)
        btn_apply = QPushButton("Apply & Retry")
        set_button_icon(btn_apply, "zap", "Apply & Retry")
        btn_apply.setObjectName("btn_send")
        btn_apply.setFixedHeight(30)
        btn_apply.clicked.connect(dlg.accept)
        btn_row.addStretch()
        btn_row.addWidget(btn_skip)
        btn_row.addWidget(btn_apply)
        root.addLayout(btn_row)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Collect auth updates
            updates = {}
            for bid, (token_edit, env_edit) in entries.items():
                token = token_edit.text().strip()
                env_text = env_edit.text().strip()
                if token or env_text:
                    updates[bid] = {
                        "mcp_auth_token": token,
                        "mcp_auth_env": env_text.replace(",", "\n") if "," in env_text else env_text,
                    }
            self._log(f"Auth configured for {len(updates)} server(s). Retrying...")
            if self._worker:
                self._worker.provide_auth(updates)
        else:
            # User skipped - pass empty auth (blocks will be removed/kept as-is)
            self._log("Auth skipped - servers without credentials will be handled.")
            if self._worker:
                self._worker.provide_auth({})

    def _start_context_compact(self):
        if self._is_running() or self._is_context_running():
            return
        if self._history.is_empty():
            self._log("No AI Builder history to compact.")
            return
        self._log("Compacting AI Builder history...")
        self._set_running(True)
        self._context_worker = AiPipelineContextCompactWorker(
            self._engine,
            history_text=self._history.recent_text(limit=20),
            summary=self._history.summary,
            canvas_state=self._canvas_state(),
            parent=self,
        )
        self._context_worker.done.connect(self._on_context_compacted)
        self._context_worker.err.connect(self._on_context_error)
        self._context_worker.finished.connect(lambda: self._set_running(False))
        self._context_worker.start()

    def _on_context_compacted(self, summary: str):
        self._history.compact(summary or deterministic_compact(self._history, self._canvas_state()))
        self._update_history_label()
        self._log("Context compacted and saved.")

    def _on_context_error(self, message: str):
        fallback = deterministic_compact(self._history, self._canvas_state())
        self._history.compact(fallback)
        self._update_history_label()
        self._log(f"AI context compact failed; saved local compact instead: {message}")

    def _load_generated_pipeline(self):
        if not self._generated_name:
            return
        if callable(self._load_callback):
            self._load_callback(self._generated_name)
        if callable(self._close_callback):
            self._close_callback()

    def is_running(self) -> bool:
        return self._is_running()


class AiPipelineBuilderDialog(QDialog):
    def __init__(
        self,
        engine: Any,
        *,
        parent=None,
        load_callback: Optional[Callable[[str], None]] = None,
        active_model_ref: str = "",
        active_model_role: str = "general",
    ):
        super().__init__(parent)
        self.setWindowTitle("AI Pipeline Builder")
        prepare_adaptive_window(self, 760, 620, min_width=560, min_height=420)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.panel = AiPipelineBuilderPanel(
            engine,
            parent=self,
            load_callback=load_callback,
            close_callback=self.accept,
            active_model_ref=active_model_ref,
            active_model_role=active_model_role,
            show_close=True,
            compact=False,
        )
        root.addWidget(self.panel)

    def reject(self):
        if self.panel.is_running():
            QMessageBox.information(
                self,
                "AI Pipeline Builder",
                "Wait for the current build to finish before closing this window."
            )
            return
        super().reject()

    def closeEvent(self, event):  # type: ignore[override]
        if self.panel.is_running():
            event.ignore()
            QMessageBox.information(
                self,
                "AI Pipeline Builder",
                "Wait for the current build to finish before closing this window."
            )
            return
        super().closeEvent(event)
