import unittest

from nativelab.UI.mainwindow.status_view import StatusViewMixin
from nativelab.UI.mainwindow.window import MainWindow
from nativelab.UI.mainwindow.shared import QMainWindow


class MainWindowSplitTests(unittest.TestCase):
    def test_mixin_super_chain_reaches_qmainwindow(self):
        mro = MainWindow.mro()
        self.assertLess(mro.index(StatusViewMixin), mro.index(QMainWindow))

    def test_event_filter_has_safe_fallback(self):
        self.assertFalse(StatusViewMixin().eventFilter(None, None))

    def test_extracted_class_attributes_are_available(self):
        self.assertIn("write code", MainWindow._CODING_KEYWORDS)

    def test_vline_remains_static_helper(self):
        self.assertIsNotNone(MainWindow._vline())

    def test_help_page_methods_are_exposed(self):
        for name in ("_build_help_tab", "_start_auto_setup_from_help", "_show_help_doc"):
            self.assertTrue(callable(getattr(MainWindow, name, None)))


if __name__ == "__main__":
    unittest.main()
