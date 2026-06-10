import os
import unittest
from unittest.mock import patch

os.environ.setdefault("NATIVELAB_NO_GUI", "1")

from nativelab.UI.Qt6widgets.chatarea import ChatArea, _AUTO_SETUP_WIDGET_ATTRS
from nativelab.UI.mainwindow import auto_setup as auto_setup_ui


class _DeletedWidget:
    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise RuntimeError("wrapped C/C++ object of type QPushButton has been deleted")

        return _raise


class _LiveWidget:
    def __init__(self, data="llama_cpp"):
        self.data = data
        self.calls = []

    def setEnabled(self, value):
        self.calls.append(("setEnabled", value))

    def setVisible(self, value):
        self.calls.append(("setVisible", value))

    def setRange(self, start, end):
        self.calls.append(("setRange", start, end))

    def setValue(self, value):
        self.calls.append(("setValue", value))

    def setText(self, text):
        self.calls.append(("setText", text))

    def currentData(self):
        return self.data

    def deleteLater(self):
        self.calls.append(("deleteLater",))


class _Layout:
    def __init__(self):
        self.removed = []

    def removeWidget(self, widget):
        self.removed.append(widget)


class _Signal:
    def connect(self, callback):
        self.callback = callback


class _Worker:
    last_backend = None

    def __init__(self, *, resume=True, backend="llama_cpp", parent=None):
        _ = resume, parent
        _Worker.last_backend = backend
        self.status = _Signal()
        self.progress = _Signal()
        self.paused = _Signal()
        self.plan_ready = _Signal()
        self.done = _Signal()
        self.err = _Signal()
        self.started = False

    def isRunning(self):
        return False

    def start(self):
        self.started = True


class _Chat:
    def __init__(self):
        self.calls = []

    def selected_auto_setup_backend(self):
        return "hf_transformers"

    def set_auto_setup_running(self, *args, **kwargs):
        self.calls.append(("set_auto_setup_running", args, kwargs))

    def update_auto_setup_status(self, *args, **kwargs):
        self.calls.append(("update_auto_setup_status", args, kwargs))


class _Owner(auto_setup_ui.AutoSetupMixin):
    def __init__(self):
        self.chat_area = _Chat()
        self.help_auto_setup_status = None
        self._auto_setup_worker = None
        self.logs = []

    def _log(self, level, message):
        self.logs.append((level, message))


class AutoSetupUiEdgeTests(unittest.TestCase):
    def _area(self):
        area = object.__new__(ChatArea)
        area._vbox = _Layout()
        for name in _AUTO_SETUP_WIDGET_ATTRS:
            setattr(area, name, None)
        return area

    def test_hide_auto_setup_prompt_clears_all_stale_refs(self):
        area = self._area()
        banner = _LiveWidget()
        area._auto_setup_banner = banner
        area._auto_setup_yes_btn = _DeletedWidget()
        area._auto_setup_cancel_btn = _DeletedWidget()

        area.hide_auto_setup_prompt()

        self.assertEqual(area._vbox.removed, [banner])
        for name in _AUTO_SETUP_WIDGET_ATTRS:
            self.assertIsNone(getattr(area, name))

    def test_running_update_drops_deleted_controls_without_crashing(self):
        area = self._area()
        area._auto_setup_banner = _LiveWidget()
        area._auto_setup_backend_combo = _LiveWidget()
        area._auto_setup_yes_btn = _DeletedWidget()
        area._auto_setup_no_btn = _DeletedWidget()
        area._auto_setup_pause_btn = _LiveWidget()
        area._auto_setup_resume_btn = _LiveWidget()
        area._auto_setup_cancel_btn = _LiveWidget()
        area._auto_setup_progress = _LiveWidget()

        area.set_auto_setup_running(True, paused=True)
        area.update_auto_setup_status("Downloading", 5, 10)

        self.assertIsNone(area._auto_setup_yes_btn)
        self.assertIsNone(area._auto_setup_no_btn)
        self.assertIn(("setVisible", True), area._auto_setup_resume_btn.calls)

    def test_qt_clicked_bool_does_not_override_selected_backend(self):
        owner = _Owner()
        with patch.object(auto_setup_ui, "AutoSetupWorker", _Worker):
            owner._start_auto_setup(False)

        self.assertEqual(_Worker.last_backend, "hf_transformers")
        self.assertIsNotNone(owner._auto_setup_worker)


if __name__ == "__main__":
    unittest.main()
