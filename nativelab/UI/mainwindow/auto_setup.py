"""MainWindow feature mixin extracted from nativelab.main."""
from .shared import *


class AutoSetupMixin:
    def _auto_setup_chat_call(self, method: str, *args, **kwargs):
        chat_area = getattr(self, "chat_area", None)
        fn = getattr(chat_area, method, None)
        if not callable(fn):
            return None
        try:
            return fn(*args, **kwargs)
        except RuntimeError as exc:
            self._log("WARN", f"Auto setup UI update skipped after widget teardown: {exc}")
            return None
        except Exception as exc:
            self._log("WARN", f"Auto setup UI update failed: {exc}")
            return None

    def _refresh_auto_setup_prompt(self):
        if not hasattr(self, "chat_area"):
            return
        worker = getattr(self, "_auto_setup_worker", None)
        try:
            if worker is not None and worker.isRunning():
                self._auto_setup_chat_call("set_auto_setup_running", True)
                return
        except Exception:
            pass
        state = read_auto_setup_state()
        resumable = auto_setup_resumable(state)
        if user_needs_auto_setup():
            self._auto_setup_chat_call(
                "show_auto_setup_prompt",
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
            self._auto_setup_chat_call("hide_auto_setup_prompt")

    def _decline_auto_setup(self):
        decline_auto_setup()
        self._auto_setup_chat_call("hide_auto_setup_prompt")
        self._log("INFO", "First-run auto setup declined.")

    def _set_help_auto_setup_status(self, text: str):
        label = getattr(self, "help_auto_setup_status", None)
        if label is not None:
            try:
                label.setText(str(text or ""))
            except RuntimeError:
                self.help_auto_setup_status = None
            except Exception as exc:
                self._log("WARN", f"Auto setup help status update failed: {exc}")

    def _start_auto_setup(self, backend: str | None = None):
        worker = getattr(self, "_auto_setup_worker", None)
        try:
            if worker is not None and worker.isRunning():
                worker.resume()
                self._auto_setup_chat_call("set_auto_setup_running", True, paused=False)
                self._set_help_auto_setup_status("Auto setup resumed.")
                return
        except Exception:
            pass

        selected_backend = ""
        if isinstance(backend, str) and backend.strip():
            selected_backend = backend
        else:
            selected_backend = self._auto_setup_chat_call("selected_auto_setup_backend") or "llama_cpp"
        selected_backend = normalize_setup_backend(selected_backend)
        self._auto_setup_worker = AutoSetupWorker(resume=True, backend=selected_backend)
        self._auto_setup_worker.status.connect(self._on_auto_setup_status)
        self._auto_setup_worker.progress.connect(self._on_auto_setup_progress)
        self._auto_setup_worker.paused.connect(self._on_auto_setup_paused)
        self._auto_setup_worker.plan_ready.connect(self._on_auto_setup_plan)
        self._auto_setup_worker.done.connect(self._on_auto_setup_done)
        self._auto_setup_worker.err.connect(self._on_auto_setup_error)
        self._auto_setup_chat_call("set_auto_setup_running", True, paused=False)
        self._auto_setup_chat_call("update_auto_setup_status", "Starting automatic setup...")
        self._set_help_auto_setup_status("Starting automatic setup...")
        self._log("INFO", "Starting first-run auto setup.")
        self._auto_setup_worker.start()

    def _pause_auto_setup(self):
        worker = getattr(self, "_auto_setup_worker", None)
        if worker is not None:
            worker.pause()
        self._auto_setup_chat_call("set_auto_setup_running", True, paused=True)
        self._auto_setup_chat_call("update_auto_setup_status", "Setup paused. Resume keeps existing partial downloads.")
        self._set_help_auto_setup_status("Setup paused. Resume keeps existing partial downloads.")

    def _resume_auto_setup(self):
        worker = getattr(self, "_auto_setup_worker", None)
        try:
            if worker is not None and worker.isRunning():
                worker.resume()
                self._auto_setup_chat_call("set_auto_setup_running", True, paused=False)
                self._auto_setup_chat_call("update_auto_setup_status", "Resuming setup...")
                self._set_help_auto_setup_status("Resuming setup...")
                return
        except Exception:
            pass
        self._start_auto_setup()

    def _cancel_auto_setup(self):
        worker = getattr(self, "_auto_setup_worker", None)
        if worker is not None:
            worker.cancel()
        self._auto_setup_chat_call("set_auto_setup_running", True, paused=True)
        self._auto_setup_chat_call("update_auto_setup_status", "Setup stopped. Partial downloads are kept for resume.")
        self._set_help_auto_setup_status("Setup stopped. Partial downloads are kept for resume.")

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
        self._auto_setup_chat_call("update_auto_setup_status", msg)
        self._set_help_auto_setup_status(msg)

    def _on_auto_setup_status(self, message: str):
        self._log("INFO", f"Auto setup: {message}")
        self._auto_setup_chat_call("update_auto_setup_status", message)
        self._set_help_auto_setup_status(message)

    def _on_auto_setup_progress(self, done: int, total: int, label: str):
        text = str(label or "Downloading")
        if total:
            pct = int(done * 100 / total)
            text = f"{text}: {done / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct}%)"
        self._auto_setup_chat_call("update_auto_setup_status", text, done, total)
        self._set_help_auto_setup_status(text)

    def _on_auto_setup_paused(self, is_paused: bool):
        self._auto_setup_chat_call("set_auto_setup_running", True, paused=bool(is_paused))

    def _on_auto_setup_error(self, message: str):
        self._auto_setup_worker = None
        self._log("ERROR", f"Auto setup failed: {message}")
        self._refresh_auto_setup_prompt()
        self._auto_setup_chat_call("update_auto_setup_status", f"Setup needs attention: {message}")
        self._set_help_auto_setup_status(f"Setup needs attention: {message}")

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
        self._auto_setup_chat_call("update_auto_setup_status", "Setup complete. Loading the model now.")
        self._auto_setup_chat_call("hide_auto_setup_prompt")
        self._set_help_auto_setup_status("Setup complete. Loading the model now.")
        if model_path and is_model_ref_valid(model_path):
            QTimer.singleShot(200, self._start_model_load)
