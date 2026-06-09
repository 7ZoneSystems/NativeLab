"""MainWindow feature mixin extracted from nativelab.main."""
from .shared import *


class SessionsMixin:
    def _load_sessions(self):
        self.sessions = {}
        for p in SESSIONS_DIR.glob("*.json"):
            try:
                s = Session.load(p)
                self.sessions[s.id] = s
            except Exception as e:
                self._log("WARN", f"Skipping corrupt session {p.name}: {e}")

    def _refresh_sidebar(self):
        self.sidebar.refresh(
            self.sessions,
            self.active.id if self.active else "",
            busy_id=self._busy_session_id,
        )

    def _new_session(self):
        s = Session.new()
        self.sessions[s.id] = s
        s.save()
        self._switch_session(s.id)

    def _switch_session(self, sid: str):
        # Null the widget refs NOW so callbacks that fire after the switch
        # won't try to write to widgets that are about to be destroyed by
        # clear_messages().  Workers keep running; _stream_buffer / session
        # ids ensure output is still saved to the correct session on completion.
        self._stream_w         = None
        self._summary_bubble   = None
        self._pipeline_reason_w = None
        self._pipeline_code_w   = None

        # Only reset the generating UI indicator if we're not the busy session
        if sid != self._busy_session_id:
            self.input_bar.set_generating(False)

        s = self.sessions.get(sid)
        if not s: return
        self.active = s
        self.chat_area.clear_messages()
        for m in s.messages:
            self.chat_area.add_message(m.role, m.content, m.timestamp)
        self._refresh_sidebar()
        self.sidebar.set_active(sid)
        self._last_context_snapshot = {}
        self._update_ctx_bar()
        # Sync ChatModule reference panel to this session
        if hasattr(self, "chat_module"):
            self.chat_module.set_session(sid)

    def _delete_session(self, sid: str):
        name = self.sessions[sid].title
        if QMessageBox.question(
            self, "Delete Session", f'Delete "{name}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        p = SESSIONS_DIR / f"{sid}.json"
        if p.exists():
            p.unlink()
        del self.sessions[sid]
        if self.active and self.active.id == sid:
            self.active = None
            self.chat_area.clear_messages()
        if self.sessions:
            last = max(self.sessions.values(), key=lambda s: s.id)
            self._switch_session(last.id)
        else:
            self._new_session()

    def _rename_session(self, sid: str, title: str):
        if sid in self.sessions:
            self.sessions[sid].title = title
            self.sessions[sid].save()
            self._refresh_sidebar()

    def _clear_chat(self):
        if not self.active: return
        if QMessageBox.question(
            self, "Clear Chat", "Clear all messages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        self.active.messages.clear()
        self.active.save()
        self.chat_area.clear_messages()
        self._last_context_snapshot = {}
        self._update_ctx_bar()

    def _on_chat_model_selected(self):
        selected = self.input_bar.selected_model
        if is_api_model_ref(selected):
            if self._api_engine and self._api_engine.is_loaded and self._api_engine.model_path == selected:
                return
            cfg = getapi_registry().get_by_ref(selected)
            label = f"Selected API: {cfg.provider} · {cfg.model_id}" if cfg else "Selected API model"
            self._set_engine_status(f"{label} (not loaded)", "idle")
            self.lbl_family.setText("API")
            return
        if is_external_model_ref(selected):
            if selected and selected == getattr(self.engine, "model_path", "") and self.engine.is_loaded:
                self._set_engine_status_from_engine(self.engine)
                return
            self._set_engine_status(f"Selected {model_ref_backend(selected)} model not loaded", "idle")
            return
        loaded_path = getattr(self.engine, "model_path", "")
        if selected and selected == loaded_path and self.engine.is_loaded:
            self._set_engine_status_from_engine(self.engine)
            return
        self._set_engine_status("Selected model not loaded", "idle")
