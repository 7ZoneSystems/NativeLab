"""MainWindow feature mixin extracted from nativelab.main."""
from .shared import *


class ContextControlsMixin:
    def _apply_new_context(self, *, allow_busy: bool = False):
        if not self.engine.is_loaded:
            return
        new_ctx = self.ctx_slider.value()
        force_reload = bool(getattr(self, "_force_ctx_reload_once", False))
        self._force_ctx_reload_once = False
        if new_ctx == getattr(self.engine, "ctx_value", DEFAULT_CTX()) and not force_reload:
            return
        skip_confirm = bool(getattr(self, "_suppress_ctx_confirm_once", False))
        self._suppress_ctx_confirm_once = False
        if new_ctx > 8192 and not skip_confirm:
            ram_estimate = (new_ctx / 1024) * 0.5
            result = QMessageBox.question(
                self, "Confirm Context Reload",
                f"Changing context to {new_ctx:,} tokens requires restarting the model\n"
                f"and may use an additional ~{ram_estimate:.0f} MB of RAM.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if result != QMessageBox.StandardButton.Yes:
                loaded_ctx = getattr(self.engine, "ctx_value", DEFAULT_CTX())
                self.ctx_slider.blockSignals(True)
                self.ctx_slider.setValue(loaded_ctx)
                self.ctx_slider.blockSignals(False)
                self.ctx_input.setText(str(loaded_ctx))
                self.current_ctx = loaded_ctx
                return
        if not allow_busy and not self._can_restart_primary_engine("changing context"):
            loaded_ctx = getattr(self.engine, "ctx_value", DEFAULT_CTX())
            self.ctx_slider.blockSignals(True)
            self.ctx_slider.setValue(loaded_ctx)
            self.ctx_slider.blockSignals(False)
            self.ctx_input.setText(str(loaded_ctx))
            self.current_ctx = loaded_ctx
            return
        self._log("INFO", f"Reloading model with context {new_ctx}")
        if self._worker:
            if not stop_worker(self._worker, LONG_TIMEOUT_MS):
                self._log("WARN", "Context reload delayed; active generation did not stop.")
                return
            self._worker = None
        self.engine.shutdown()
        self.current_ctx = new_ctx
        self.ctx_bar.setRange(0, new_ctx)
        self._start_model_load()

    def _on_ctx_changed(self, value: int):
        self.ctx_input.setText(str(value))
        self.current_ctx = value
        if hasattr(self, "model_list"):
            try:
                item = self.model_list.currentItem()
                if item:
                    path = item.data(Qt.ItemDataRole.UserRole)
                    if path and path == getattr(self.engine, "model_path", ""):
                        self.cfg_ctx.blockSignals(True)
                        self.cfg_ctx.setText(str(value))
                        self.cfg_ctx.blockSignals(False)
            except RuntimeError:
                pass
        color = C["ok"]
        warn_text = ""
        if value > 24576:
            color = C["err"]; warn_text = "!"
            self.ctx_warn.setToolTip("Very high context.\nExpect heavy RAM usage.")
        elif value > 16384:
            color = C["warn"]; warn_text = "!"
            self.ctx_warn.setToolTip("High context.\nPerformance may degrade.")
        else:
            self.ctx_warn.setToolTip("")
        self._ctx_reload_timer.start(2000)
        self._apply_ctx_slider_style(color)
        self.ctx_warn.setText(warn_text)
        self._update_ctx_bar(limit_override=value)

    def _apply_ctx_slider_style(self, color: str):
        if not hasattr(self, "ctx_slider"):
            return
        self.ctx_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height:5px; background:{C['bg2']}; border-radius:3px;
            }}
            QSlider::handle:horizontal {{
                background:{color}; width:13px; margin:-4px 0; border-radius:7px;
            }}
            QSlider::sub-page:horizontal {{
                background:{color}; border-radius:3px;
            }}
        """)

    def _on_ctx_input_changed(self):
        try:
            value = max(512, min(MAX_CONTEXT_TOKENS, int(self.ctx_input.text())))
            self.ctx_slider.setValue(value)
        except ValueError:
            self.ctx_input.setText(str(self.ctx_slider.value()))
