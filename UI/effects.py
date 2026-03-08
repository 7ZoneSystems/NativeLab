from imports.import_global import Qt, pyqtProperty, QWidget, QGraphicsOpacityEffect, QPropertyAnimation, QEasingCurve, QPainter, QColor
def fade_in(widget: QWidget, duration: int = 180):
    """Fade a widget in. Skipped if widget has its own paintEvent (avoids QPainter conflicts)."""
    # PipelineCanvas and other custom-painted widgets must not get opacity effects
    if type(widget).__name__ in ("PipelineCanvas", "ThinkingBlock"):
        return
    try:
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    except Exception:
        pass

class FadeOverlay(QWidget):
    """Full-size overlay for tab-switch fade.
    Uses its own paintEvent — never touches QGraphicsOpacityEffect,
    so no QPainter conflicts with PipelineCanvas or other custom painters."""
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._alpha = 0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.hide()

    def _get_alpha(self): return self._alpha
    def _set_alpha(self, v: int):
        self._alpha = max(0, min(255, v))
        self.update()
    alpha = pyqtProperty(int, _get_alpha, _set_alpha)

    def paintEvent(self, _):
        from UI.buildUI import C
        p = QPainter(self)
        col = QColor(C["bg0"])
        col.setAlpha(self._alpha)
        p.fillRect(self.rect(), col)
        p.end()
