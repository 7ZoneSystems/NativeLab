"""Qt compatibility boundary for GUI and headless CLI runs.

The GUI imports the real PyQt objects. CLI mode intentionally does not import
PyQt, but some shared backend modules still define Qt-style workers/signals.
This module gives those shared modules a small headless surface so importing
them from `nativelab --cli` stays lightweight.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Callable


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


HEADLESS_QT = (
    _env_truthy("NATIVELAB_CLI")
    or _env_truthy("NATIVELAB_NO_GUI")
    or "--cli" in sys.argv[1:]
)


if not HEADLESS_QT:
    from PyQt6.QtCore import (  # type: ignore
        QCoreApplication,
        QDataStream,
        QEasingCurve,
        QEvent,
        QEventLoop,
        QIODevice,
        QObject,
        QPointF,
        QPropertyAnimation,
        QRect,
        QRectF,
        QSize,
        Qt,
        QThread,
        QTimer,
        QVariant,
        pyqtProperty,
        pyqtSignal,
    )
    from PyQt6.QtGui import (  # type: ignore
        QAction,
        QBrush,
        QColor,
        QFont,
        QIcon,
        QKeySequence,
        QLinearGradient,
        QMouseEvent,
        QPainter,
        QPainterPath,
        QPalette,
        QPen,
        QPolygonF,
        QTextCharFormat,
        QTextCursor,
        QTextFormat,
    )
    from PyQt6.QtWidgets import (  # type: ignore
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QColorDialog,
        QComboBox,
        QDialog,
        QFileDialog,
        QFrame,
        QGraphicsOpacityEffect,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QTabWidget,
        QTextBrowser,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
else:
    class _Signal:
        def __init__(self):
            self._callbacks: list[Callable[..., Any]] = []

        def connect(self, callback: Callable[..., Any], *args, **kwargs) -> None:
            _ = args, kwargs
            if callable(callback) and callback not in self._callbacks:
                self._callbacks.append(callback)

        def disconnect(self, callback: Callable[..., Any] | None = None) -> None:
            if callback is None:
                self._callbacks.clear()
                return
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

        def emit(self, *args, **kwargs) -> None:
            for callback in list(self._callbacks):
                callback(*args, **kwargs)


    class _SignalDescriptor:
        def __init__(self):
            self._name = ""

        def __set_name__(self, owner, name):
            self._name = f"__qt_signal_{name}"

        def __get__(self, instance, owner=None):
            if instance is None:
                return self
            signal = instance.__dict__.get(self._name)
            if signal is None:
                signal = _Signal()
                instance.__dict__[self._name] = signal
            return signal


    def pyqtSignal(*args, **kwargs):  # type: ignore[no-redef]
        _ = args, kwargs
        return _SignalDescriptor()


    def pyqtProperty(type_=None, fget=None, fset=None, freset=None, doc=None, **kwargs):  # type: ignore[no-redef]
        _ = type_, freset, kwargs
        if fget is None:
            return lambda getter: property(getter, doc=doc)
        return property(fget, fset, doc=doc)


    class _QtValue:
        def __init__(self, name: str = ""):
            self._name = name

        def __getattr__(self, name: str):
            return _QtValue(f"{self._name}.{name}" if self._name else name)

        def __call__(self, *args, **kwargs):
            _ = args, kwargs
            return _QtStub()

        def __or__(self, other):
            _ = other
            return self

        def __and__(self, other):
            _ = other
            return self

        def __invert__(self):
            return self

        def __bool__(self) -> bool:
            return False

        def __int__(self) -> int:
            return 0

        def __repr__(self) -> str:
            return self._name or "QtValue"


    class _QtStubMeta(type):
        def __getattr__(cls, name: str):
            return _QtValue(f"{cls.__name__}.{name}")


    class _QtStub(metaclass=_QtStubMeta):
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        def __getattr__(self, name: str):
            return _QtValue(f"{self.__class__.__name__}.{name}")

        def __call__(self, *args, **kwargs):
            _ = args, kwargs
            return _QtStub()

        def __bool__(self) -> bool:
            return False


    class QObject(_QtStub):  # type: ignore[no-redef]
        def deleteLater(self) -> None:
            pass

        def moveToThread(self, thread) -> None:
            _ = thread


    class QThread(QObject):  # type: ignore[no-redef]
        finished = pyqtSignal()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._thread: threading.Thread | None = None

        def start(self) -> None:
            if self.isRunning():
                return
            self._thread = threading.Thread(target=self._bootstrap, daemon=True)
            self._thread.start()

        def _bootstrap(self) -> None:
            try:
                self.run()
            finally:
                try:
                    self.finished.emit()
                except Exception:
                    pass

        def run(self) -> None:
            pass

        def wait(self, timeout: int | None = None) -> bool:
            if self._thread is None:
                return True
            seconds = None if timeout is None or timeout < 0 else timeout / 1000.0
            self._thread.join(seconds)
            return not self._thread.is_alive()

        def isRunning(self) -> bool:
            return bool(self._thread and self._thread.is_alive())

        def quit(self) -> None:
            pass

        def terminate(self) -> None:
            pass

        @staticmethod
        def currentThread():
            return threading.current_thread()


    class QEventLoop(QObject):  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._event = threading.Event()

        def exec(self) -> int:
            self._event.wait()
            return 0

        def quit(self) -> None:
            self._event.set()


    class QCoreApplication(QObject):  # type: ignore[no-redef]
        _instance = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            QCoreApplication._instance = self

        @classmethod
        def instance(cls):
            return cls._instance

        @staticmethod
        def processEvents(*args, **kwargs) -> None:
            _ = args, kwargs

        def exec(self) -> int:
            return 0

        def quit(self) -> None:
            pass


    class QApplication(QCoreApplication):  # type: ignore[no-redef]
        @staticmethod
        def allWidgets() -> list:
            return []

        @staticmethod
        def clipboard():
            class _Clipboard:
                def setText(self, text: str) -> None:
                    self.text = text

            return _Clipboard()


    class QTimer(QObject):  # type: ignore[no-redef]
        timeout = pyqtSignal()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._active = False
            self._interval = 0

        def setInterval(self, interval: int) -> None:
            self._interval = int(interval or 0)

        def start(self, interval: int | None = None) -> None:
            if interval is not None:
                self.setInterval(interval)
            self._active = True

        def stop(self) -> None:
            self._active = False

        def isActive(self) -> bool:
            return self._active

        @staticmethod
        def singleShot(interval: int, callback: Callable[[], Any]) -> None:
            _ = interval
            if callable(callback):
                callback()


    Qt = _QtValue("Qt")  # type: ignore[assignment]

    QAction = QAbstractItemView = QBrush = QCheckBox = QColor = QColorDialog = QComboBox = QDataStream = _QtStub
    QDialog = QEasingCurve = QEvent = QFileDialog = QFont = QFrame = QGraphicsOpacityEffect = QGroupBox = _QtStub
    QHBoxLayout = QIcon = QIODevice = QInputDialog = QKeySequence = QLabel = QLineEdit = _QtStub
    QLinearGradient = QListWidget = QListWidgetItem = QMainWindow = QMenu = QMessageBox = _QtStub
    QMouseEvent = QPainter = QPainterPath = QPalette = QPen = QPointF = QPolygonF = _QtStub
    QProgressBar = QPropertyAnimation = QPushButton = QRect = QRectF = QSize = QSizePolicy = _QtStub
    QScrollArea = QSlider = QSpinBox = QSplitter = QStackedWidget = QTabWidget = QTextBrowser = _QtStub
    QTextCharFormat = QTextCursor = QTextEdit = QTextFormat = QVariant = QVBoxLayout = QWidget = _QtStub


__all__ = [
    "HEADLESS_QT",
    "QAbstractItemView",
    "QAction",
    "QApplication",
    "QBrush",
    "QCheckBox",
    "QColor",
    "QColorDialog",
    "QComboBox",
    "QCoreApplication",
    "QDataStream",
    "QDialog",
    "QEasingCurve",
    "QEvent",
    "QEventLoop",
    "QFileDialog",
    "QFont",
    "QFrame",
    "QGraphicsOpacityEffect",
    "QGroupBox",
    "QHBoxLayout",
    "QIODevice",
    "QIcon",
    "QInputDialog",
    "QKeySequence",
    "QLabel",
    "QLineEdit",
    "QLinearGradient",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QMenu",
    "QMessageBox",
    "QMouseEvent",
    "QObject",
    "QPainter",
    "QPainterPath",
    "QPalette",
    "QPen",
    "QPointF",
    "QPolygonF",
    "QProgressBar",
    "QPropertyAnimation",
    "QPushButton",
    "QRect",
    "QRectF",
    "QScrollArea",
    "QSize",
    "QSizePolicy",
    "QSlider",
    "QSpinBox",
    "QSplitter",
    "QStackedWidget",
    "QTabWidget",
    "QTextBrowser",
    "QTextCharFormat",
    "QTextCursor",
    "QTextEdit",
    "QTextFormat",
    "QThread",
    "QTimer",
    "QVBoxLayout",
    "QVariant",
    "QWidget",
    "Qt",
    "pyqtProperty",
    "pyqtSignal",
]
