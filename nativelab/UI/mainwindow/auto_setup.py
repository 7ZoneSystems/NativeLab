"""MainWindow feature mixin extracted from nativelab.main."""
from .shared import *


class AutoSetupMixin:
    def _refresh_auto_setup_prompt(self):
        if not hasattr(self, "chat_area"):
            return
        worker = getattr(self, "_auto_setup_worker", None)
        try:
            if worker is not None and worker.isRunning():
                self.chat_area.set_auto_setup_running(True)
                return
        except Exception:
            pass
        state = read_auto_setup_state()
        resumable = auto_setup_resumable(state)
        if user_needs_auto_setup():
            self.chat_area.show_auto_setup_prompt(
                self._start_auto_setup,
                self._decline_auto_setup,
                self._pause_auto_setup,
                self._resume_auto_setup,
                self._cancel_auto_setup,
                resume=resumable,
                default_backend=normalize_setup_backend(state.get("backend") or "llama_cpp"),
                backend_locked=resumable,
            )
        else:
            self.chat_area.hide_auto_setup_prompt()

    def _decline_auto_setup(self):
        decline_auto_setup()
        if hasattr(self, "chat_area"):
            self.chat_area.hide_auto_setup_prompt()
        self._log("INFO", "First-run auto setup declined.")

    def _start_auto_setup(self):
        worker = getattr(self, "_auto_setup_worker", None)
        try:
            if worker is not None and worker.isRunning():
                worker.resume()
                self.chat_area.set_auto_setup_running(True, paused=False)
                return
        except Exception:
            pass

        backend = "llama_cpp"
        if hasattr(self, "chat_area") and hasattr(self.chat_area, "selected_auto_setup_backend"):
            backend = self.chat_area.selected_auto_setup_backend()
        self._auto_setup_worker = AutoSetupWorker(resume=True, backend=backend)
        self._auto_setup_worker.status.connect(self._on_auto_setup_status)
        self._auto_setup_worker.progress.connect(self._on_auto_setup_progress)
        self._auto_setup_worker.paused.connect(self._on_auto_setup_paused)
        self._auto_setup_worker.plan_ready.connect(self._on_auto_setup_plan)
        self._auto_setup_worker.done.connect(self._on_auto_setup_done)
        self._auto_setup_worker.err.connect(self._on_auto_setup_error)
        if hasattr(self, "chat_area"):
            self.chat_area.set_auto_setup_running(True, paused=False)
            self.chat_area.update_auto_setup_status("Starting automatic setup...")
        self._log("INFO", "Starting first-run auto setup.")
        self._auto_setup_worker.start()

    def _pause_auto_setup(self):
        worker = getattr(self, "_auto_setup_worker", None)
        if worker is not None:
            worker.pause()
        if hasattr(self, "chat_area"):
            self.chat_area.set_auto_setup_running(True, paused=True)
            self.chat_area.update_auto_setup_status("Setup paused. Resume keeps existing partial downloads.")

    def _resume_auto_setup(self):
        worker = getattr(self, "_auto_setup_worker", None)
        try:
            if worker is not None and worker.isRunning():
                worker.resume()
                if hasattr(self, "chat_area"):
                    self.chat_area.set_auto_setup_running(True, paused=False)
                    self.chat_area.update_auto_setup_status("Resuming setup...")
                return
        except Exception:
            pass
        self._start_auto_setup()

    def _cancel_auto_setup(self):
        worker = getattr(self, "_auto_setup_worker", None)
        if worker is not None:
            worker.cancel()
        if hasattr(self, "chat_area"):
            self.chat_area.set_auto_setup_running(True, paused=True)
            self.chat_area.update_auto_setup_status("Setup stopped. Partial downloads are kept for resume.")

    def _on_auto_setup_plan(self, plan: object):
        try:
            model = (plan or {}).get("model", {})
            hardware = (plan or {}).get("hardware", {})
            accelerator = (plan or {}).get("accelerator", "cpu")
            threads = (plan or {}).get("threads", 0)
            backend = (plan or {}).get("backend_label", "llama.cpp GGUF (easy)")
            ram_gb = int(hardware.get("ram_total_mb") or 0) / 1024
            used_gb = int(hardware.get("ram_used_mb") or 0) / 1024
            msg = (
                f"Plan: {backend}  ·  {model.get('label', 'model')}  "
                f"RAM {ram_gb:.1f} GB, OS using {used_gb:.1f} GB, "
                f"{accelerator}, {threads} threads."
            )
        except Exception:
            msg = "Hardware plan ready."
        self._log("INFO", msg)
        if hasattr(self, "chat_area"):
            self.chat_area.update_auto_setup_status(msg)

    def _on_auto_setup_status(self, message: str):
        self._log("INFO", f"Auto setup: {message}")
        if hasattr(self, "chat_area"):
            self.chat_area.update_auto_setup_status(message)

    def _on_auto_setup_progress(self, done: int, total: int, label: str):
        if hasattr(self, "chat_area"):
            text = str(label or "Downloading")
            if total:
                pct = int(done * 100 / total)
                text = f"{text}: {done / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct}%)"
            self.chat_area.update_auto_setup_status(text, done, total)

    def _on_auto_setup_paused(self, is_paused: bool):
        if hasattr(self, "chat_area"):
            self.chat_area.set_auto_setup_running(True, paused=bool(is_paused))

    def _on_auto_setup_error(self, message: str):
        self._auto_setup_worker = None
        self._log("ERROR", f"Auto setup failed: {message}")
        self._refresh_auto_setup_prompt()
        if hasattr(self, "chat_area"):
            self.chat_area.update_auto_setup_status(f"Setup needs attention: {message}")

    def _on_auto_setup_done(self, model_path: str):
        self._auto_setup_worker = None
        self._log("INFO", f"Auto setup complete: {model_ref_display_name(model_path)}")
        if hasattr(self, "model_list"):
            self._refresh_model_list()
        if hasattr(self, "input_bar"):
            self._sync_input_bar_combo()
            idx = self.input_bar.model_combo.findData(model_path)
            if idx >= 0:
                self.input_bar.model_combo.setCurrentIndex(idx)
        if hasattr(self, "server_tab") and hasattr(self.server_tab, "refresh_values"):
            self.server_tab.refresh_values()
        if hasattr(self, "chat_area"):
            self.chat_area.update_auto_setup_status("Setup complete. Loading the model now.")
            self.chat_area.hide_auto_setup_prompt()
        if model_path and is_model_ref_valid(model_path):
            QTimer.singleShot(200, self._start_model_load)
