"""NativeLab GUI entrypoint.

The large application window is assembled in nativelab.UI.mainwindow; this module
keeps CLI routing and QApplication startup only.
"""
import os as _entry_os
import signal
import sys as _entry_sys


if "--cli" in _entry_sys.argv[1:]:
    _entry_os.environ["NATIVELAB_CLI"] = "1"
    _entry_os.environ["NATIVELAB_NO_GUI"] = "1"
    from nativelab.cli import run_cli as _entry_run_cli

    _entry_sys.exit(_entry_run_cli(_entry_sys.argv[1:]))

from nativelab.imports.import_global import QApplication, QFont, QIcon, Qt, _platform
from nativelab.UI.mainwindow import MainWindow


def _configure_qt_platform():
    if _platform.system() != "Darwin":
        return
    set_attribute = getattr(QApplication, "setAttribute", None)
    app_attrs = getattr(Qt, "ApplicationAttribute", None)
    dont_use_native_menu = getattr(app_attrs, "AA_DontUseNativeMenuBar", None)
    if callable(set_attribute) and dont_use_native_menu is not None:
        try:
            set_attribute(dont_use_native_menu, True)
        except Exception:
            pass


def main():
    if "--cli" in _entry_sys.argv[1:]:
        _entry_os.environ["NATIVELAB_CLI"] = "1"
        _entry_os.environ["NATIVELAB_NO_GUI"] = "1"
        from nativelab.cli import run_cli

        _entry_sys.exit(run_cli(_entry_sys.argv[1:]))

    _configure_qt_platform()
    app = QApplication(_entry_sys.argv)
    app.setApplicationName("NativeLab")
    base_dir = _entry_os.path.dirname(__file__)
    icon_path = _entry_os.path.join(base_dir, "icon.png")

    app.setWindowIcon(QIcon(icon_path))
    font = QFont("Inter")
    if not font.exactMatch():
        font = QFont("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    win = MainWindow()
    win.setWindowTitle("NativeLab")
    win.show()
    _entry_sys.exit(app.exec())


if __name__ == "__main__":
    main()
