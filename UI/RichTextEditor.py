from imports.import_global import Dict, QTextBrowser, QTimer, QApplication, Qt 
from .UI_const import C
class RichTextEdit(QTextBrowser):
    """
    QTextEdit subclass.
    • Intercepts mouse clicks on  copy://BLOCK_ID  anchors and
      copies the stored code block to the clipboard.
    • Stores {block_id: raw_code} internally.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._code_blocks: Dict[str, str] = {}   # bid → raw code text
        self.setReadOnly(True)
        self.setOpenLinks(False)

    def store_block(self, bid: str, code: str):
        self._code_blocks[bid] = code

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            anchor = self.anchorAt(event.pos())
            if anchor.startswith("copy://"):
                bid  = anchor[7:]
                code = self._code_blocks.get(bid, "")
                if code:
                    QApplication.clipboard().setText(code)
                    # Flash visual feedback via cursor position trick
                    self.flash_copy(bid)
                event.accept()
                return
        super().mousePressEvent(event)

    def flash_copy(self, bid: str):
        """Walk up to the MessageWidget and briefly swap its copy button label."""
        try:
            w = self.parent()
            while w and not hasattr(w, "_copy_btn"):
                w = w.parent()
            if w and hasattr(w, "_copy_btn"):
                btn = w._copy_btn
                btn.setText("✓")
                btn.setStyleSheet(
                    btn.styleSheet().replace(C["txt2"], C["ok"]))
                def _restore():
                    btn.setText("⧉")
                    btn.setStyleSheet(
                        btn.styleSheet().replace(C["ok"], C["txt2"]))
                QTimer.singleShot(1400, _restore)
        except Exception:
            pass
