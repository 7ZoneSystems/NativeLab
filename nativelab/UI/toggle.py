from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QRect, QRectF, QSize, Qt, QPropertyAnimation, pyqtProperty
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen
from PyQt6.QtWidgets import QCheckBox


class ToggleSwitch(QCheckBox):
    """Checkbox-compatible boolean control painted as a sliding switch."""

    TRACK_W = 42
    TRACK_H = 22
    KNOB = 18
    GAP = 10

    def __init__(self, text: str = "", parent=None):
        super().__init__("", parent)
        self._label = str(text or "")
        super().setText(self._label)
        self._offset = 1.0 if self.isChecked() else 0.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(28)
        self.toggled.connect(self._animate_to_state)

    @pyqtProperty(float)
    def offset(self) -> float:
        return self._offset

    @offset.setter
    def offset(self, value: float) -> None:
        self._offset = max(0.0, min(1.0, float(value)))
        self.update()

    def setText(self, text: str) -> None:
        self._label = str(text or "")
        super().setText(self._label)
        self.updateGeometry()
        self.update()

    def text(self) -> str:
        return self._label

    def setChecked(self, checked: bool) -> None:
        super().setChecked(bool(checked))
        self._offset = 1.0 if checked else 0.0
        self.update()

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        icon_w = 20 if not self.icon().isNull() else 0
        text_w = fm.horizontalAdvance(self._label) if self._label else 0
        label_w = icon_w + text_w + (4 if icon_w and text_w else 0)
        return QSize(self.TRACK_W + self.GAP + label_w + 4, max(28, fm.height() + 8))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _animate_to_state(self, checked: bool) -> None:
        if not self.isVisible():
            self._offset = 1.0 if checked else 0.0
            return
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def _theme_colors(self) -> dict:
        try:
            from nativelab.UI import UI_const
            return dict(getattr(UI_const, "C", {}) or {})
        except Exception:
            return {}

    def hitButton(self, pos) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, _event) -> None:
        pal = self.palette()
        enabled = self.isEnabled()
        c = self._theme_colors()
        text_color = QColor(c.get("txt" if enabled else "txt3", pal.windowText().color().name()))
        base = QColor(c.get("surface", pal.base().color().name()))
        border = QColor(c.get("bdr2", pal.mid().color().name()))
        accent = QColor(c.get("acc", pal.highlight().color().name()))
        knob = QColor(c.get("txt", pal.windowText().color().name()))
        if not enabled:
            base = QColor(c.get("bg3", pal.button().color().name()))
            accent = QColor(c.get("bdr2", pal.mid().color().name()))
            knob = QColor(c.get("txt3", pal.mid().color().name()))

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        track_x = 0
        track_y = (self.height() - self.TRACK_H) / 2
        track = QRectF(track_x, track_y, self.TRACK_W, self.TRACK_H)
        fill = QColor(accent if self.isChecked() else base)
        if self.isChecked():
            fill.setAlpha(235 if enabled else 125)
        painter.setPen(QPen(accent if self.isChecked() else border, 1.2))
        painter.setBrush(fill)
        painter.drawRoundedRect(track, self.TRACK_H / 2, self.TRACK_H / 2)

        knob_x = track_x + 2 + self._offset * (self.TRACK_W - self.KNOB - 4)
        knob_rect = QRectF(knob_x, track_y + 2, self.KNOB, self.KNOB)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(knob)
        painter.drawEllipse(knob_rect)

        label_x = self.TRACK_W + self.GAP
        if not self.icon().isNull():
            icon_rect = QRect(label_x, int((self.height() - 16) / 2), 16, 16)
            mode = QIcon.Mode.Normal if enabled else QIcon.Mode.Disabled
            self.icon().paint(painter, icon_rect, Qt.AlignmentFlag.AlignCenter, mode)
            label_x += 20

        if self._label:
            painter.setPen(text_color)
            text_rect = QRect(label_x, 0, max(0, self.width() - label_x), self.height())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._label)
