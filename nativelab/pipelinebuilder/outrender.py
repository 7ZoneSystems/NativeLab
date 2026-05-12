from nativelab.imports.import_global import QTextEdit, QTimer
from nativelab.UI.UI_const import C
class PipelineOutputRenderer(QTextEdit):
    """
    Read-only rich output pane used for pipeline intermediate and final output.
    Parses the same tokens as the main chat window:
      - ```lang ... ```  → syntax-highlighted code block (Consolas, bg panel)
      - **bold**         → bold
      - `inline code`    → monospace highlighted span
      - Plain text lines → normal paragraph
    Accepts either streaming token-by-token calls (append_token) or
    a full set_content(text) replacement.
    """

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setObjectName("chat_te")
        self._raw = ""          # accumulated raw text
        self._placeholder_text = placeholder
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(120)   # debounce re-renders during streaming
        self._render_timer.timeout.connect(self._render)
        if placeholder:
            self.setPlaceholderText(placeholder)

    # ── public API ────────────────────────────────────────────────────────────

    def append_token(self, token: str):
        self._raw += token
        self._render_timer.start()   # debounce

    def set_content(self, text: str):
        self._raw = text
        self._render()

    def clear_content(self):
        self._raw = ""
        self.clear()

    def raw_text(self) -> str:
        return self._raw

    # ── renderer ──────────────────────────────────────────────────────────────

    def _render(self):
        html  = self._to_html(self._raw)
        cur   = self.verticalScrollBar().value()
        at_bt = cur >= self.verticalScrollBar().maximum() - 40
        self.setHtml(html)
        if at_bt:
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().maximum())
        else:
            self.verticalScrollBar().setValue(cur)

    def _to_html(self, text: str) -> str:
        """Convert raw pipeline text to HTML using the shared chat renderer."""
        from nativelab.UI.md_to_html import md_to_html
        TXT = C.get("txt", "#cdd6f4")
        body_css = (
            f"font-family:'Inter',sans-serif;font-size:12px;"
            f"color:{TXT};background:transparent;"
            f"margin:0;padding:4px 8px;line-height:1.65;"
        )
        return (
            f'<html><body style="{body_css}">'
            + md_to_html(text, colors=C)
            + "</body></html>"
        )
