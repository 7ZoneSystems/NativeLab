"""MainWindow feature mixin extracted from nativelab.main."""
from .shared import *


class LabsIntegrationMixin:
    def _wire_lab_endpoints(self):
        """Bind the labs tab to live engine state + reverse-route actions."""
        ep = self._lab_endpoints
        ep.bind_engines(
            llama_provider=lambda: self.engine,
            api_provider  =lambda: self._api_engine,
        )
        ep.bind_reverse_routes(
            on_context=self._labs_request_context,
            on_model  =self._labs_request_load_model,
            on_unload =self._labs_request_unload,
            on_reload =self._labs_request_reload_active_model,
            on_wait_loaded=self._labs_wait_until_loaded,
            on_is_loading=self._labs_is_model_loading,
        )
        ep.set_skill_context_provider(self._active_skill_context)
        ep.log_msg.connect(self.log_console.log)
        self.labs_tab.set_endpoints(ep)
        self._integration_endpoints.bind_lab_endpoints(ep)
        if hasattr(self, "integrations_tab"):
            self.integrations_tab.set_endpoints(self._integration_endpoints)
            try:
                ep.engine_changed.connect(self.integrations_tab.refresh)
            except Exception:
                pass
        if hasattr(self, "api_server_tab"):
            self.api_server_tab.set_endpoints(ep)

    def _skills_enabled(self) -> bool:
        return bool(
            hasattr(self, "input_bar")
            and getattr(self.input_bar, "skills_enabled", False)
        )

    def _active_skill_context(self) -> str:
        if not self._skills_enabled():
            return ""
        return active_skill_context()

    def _notify_labs(self):
        """Emit engine_changed/status_changed so lab panels can refresh."""
        if hasattr(self, "_lab_endpoints"):
            self._lab_endpoints.notify_engine_changed()

    def _labs_request_context(self, new_ctx: int) -> bool:
        if QThread.currentThread() != self.thread():
            payload = {"event": threading.Event(), "ok": False, "error": None}
            self._lab_context_request.emit(int(new_ctx), payload)
            if not payload["event"].wait(LONG_TIMEOUT_MS / 1000.0):
                return False
            if payload["error"] is not None:
                raise payload["error"]
            return bool(payload["ok"])
        return self._labs_request_context_impl(new_ctx)

    def _handle_labs_context_request(self, new_ctx: int, payload: dict):
        try:
            payload["ok"] = bool(self._labs_request_context_impl(new_ctx))
        except Exception as exc:
            payload["error"] = exc
        finally:
            payload["event"].set()

    def _labs_request_context_impl(self, new_ctx: int) -> bool:
        """Reverse route: a lab feature asks the host to change ctx."""
        if not hasattr(self, "ctx_slider"):
            return False
        try:
            requested_ctx = max(512, min(MAX_CONTEXT_TOKENS, int(new_ctx)))
            force_same_ctx = requested_ctx == int(getattr(self.engine, "ctx_value", 0) or 0)
            self.ctx_slider.setValue(requested_ctx)
        except Exception:
            return False
        self._suppress_ctx_confirm_once = True
        self._force_ctx_reload_once = force_same_ctx
        self._apply_new_context(allow_busy=True)
        loader = getattr(self, "_loader", None)
        if loader and loader.isRunning():
            try:
                from nativelab.imports.qt_compat import QEventLoop
                loop = QEventLoop()
                loader.finished.connect(lambda *_: loop.quit())
                loop.exec()
            except Exception:
                loader.wait(LONG_TIMEOUT_MS)
        return bool(
            self.engine.is_loaded
            and int(getattr(self.engine, "ctx_value", 0)) == requested_ctx
        )

    def _labs_request_load_model(self, model_path: str) -> bool:
        """Reverse route: a lab feature asks the host to load a model."""
        if QThread.currentThread() != self.thread():
            payload = {"event": threading.Event(), "ok": False, "error": None}
            self._lab_model_request.emit(str(model_path), payload)
            if not payload["event"].wait(LONG_TIMEOUT_MS / 1000.0):
                return False
            if payload["error"] is not None:
                raise payload["error"]
            return bool(payload["ok"])
        return self._labs_request_load_model_impl(model_path)

    def _handle_labs_model_request(self, model_path: str, payload: dict):
        try:
            payload["ok"] = bool(self._labs_request_load_model_impl(model_path))
        except Exception as exc:
            payload["error"] = exc
        finally:
            payload["event"].set()

    def _labs_request_load_model_impl(self, model_path: str) -> bool:
        """Main-thread implementation for loading a model requested by endpoints."""
        if is_api_model_ref(model_path):
            idx = self.input_bar.model_combo.findData(model_path)
            if idx == -1:
                cfg = getapi_registry().get_by_ref(model_path)
                self.input_bar.model_combo.addItem(
                    api_model_label(cfg) if cfg else model_path, model_path)
                idx = self.input_bar.model_combo.findData(model_path)
            self.input_bar.model_combo.setCurrentIndex(idx)
            self._start_api_model_load(model_path)
            return True
        if not model_path or not is_model_ref_valid(model_path):
            return False
        idx = self.input_bar.model_combo.findData(model_path)
        if idx == -1:
            self.input_bar.model_combo.addItem(model_ref_display_name(model_path), model_path)
            idx = self.input_bar.model_combo.findData(model_path)
        self.input_bar.model_combo.setCurrentIndex(idx)
        self.engine.shutdown()
        QTimer.singleShot(200, self._start_model_load)
        return True

    def _labs_wait_until_loaded(self, timeout_ms: int = LONG_TIMEOUT_MS) -> bool:
        """Reverse route: let labs patiently wait for a host-side load."""
        if self._lab_endpoints.is_loaded:
            return True
        loaders = [
            getattr(self, "_loader", None),
            getattr(self, "_api_loader", None),
        ]
        running = [loader for loader in loaders if loader and loader.isRunning()]
        if not running:
            return False
        deadline = time.monotonic() + (max(0, int(timeout_ms or 0)) / 1000.0)
        for loader in running:
            if not loader or not loader.isRunning():
                continue
            remaining_ms = int(max(1, (deadline - time.monotonic()) * 1000))
            if remaining_ms <= 0:
                break
            loader.wait(remaining_ms)
        return bool(self._lab_endpoints.is_loaded)

    def _labs_is_model_loading(self) -> bool:
        loaders = [
            getattr(self, "_loader", None),
            getattr(self, "_api_loader", None),
        ]
        return any(loader and loader.isRunning() for loader in loaders)

    def _labs_request_reload_active_model(self) -> bool:
        if QThread.currentThread() != self.thread():
            payload = {"event": threading.Event(), "ok": False, "error": None}
            self._lab_reload_request.emit(payload)
            if not payload["event"].wait(LONG_TIMEOUT_MS / 1000.0):
                return False
            if payload["error"] is not None:
                raise payload["error"]
            return bool(payload["ok"])
        return self._labs_request_reload_active_model_impl()

    def _handle_labs_reload_request(self, payload: dict):
        try:
            payload["ok"] = bool(self._labs_request_reload_active_model_impl())
        except Exception as exc:
            payload["error"] = exc
        finally:
            payload["event"].set()

    def _labs_request_reload_active_model_impl(self) -> bool:
        """Reverse route: a lab feature asks for a fresh local model instance."""
        if self._api_engine and self._api_engine.is_loaded:
            self._log("INFO", "py-to-doc model reload skipped for active API backend")
            return True
        if not self._labs_wait_until_loaded(LONG_TIMEOUT_MS) and not self.engine.is_loaded:
            return False
        if not self.engine or not self.engine.is_loaded:
            return False
        model = getattr(self.engine, "model_path", "")
        ctx = int(
            getattr(self.engine, "ctx_value", self.ctx_slider.value())
            or self.ctx_slider.value()
        )
        if not model or not is_model_ref_valid(model):
            return False
        self._log("INFO", f"py-to-doc reloading active model: {model_ref_display_name(model)}")
        if self._worker:
            if not stop_worker(self._worker, LONG_TIMEOUT_MS):
                self._log("WARN", "py-to-doc reload delayed; active worker did not stop.")
                return False
            self._worker = None
        try:
            self.engine.shutdown()
        except Exception:
            pass
        self.engine = LlamaEngine()
        self.ctx_slider.setValue(max(512, min(MAX_CONTEXT_TOKENS, ctx)))
        self.current_ctx = self.ctx_slider.value()
        self._set_engine_status("Reloading model...", "loading")
        self._notify_labs()
        cfg = get_model_registry().get_config(model)
        self._loader = ModelLoaderThread(self.engine, model, self.current_ctx, cfg.threads)
        self._loader.log.connect(self._log)
        self._loader.finished.connect(self._on_model_loaded)
        self._loader.start()
        return self._labs_wait_until_loaded(LONG_TIMEOUT_MS)

    def _labs_request_unload(self) -> None:
        """Reverse route: a lab feature asks the host to unload the primary engine."""
        self.engine.shutdown()
        self.engine = LlamaEngine()
        self._notify_labs()
