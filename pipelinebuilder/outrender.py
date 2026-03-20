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
        """Convert raw pipeline text to HTML using the same rules as the chat window."""
        CODE_BG   = C.get("bg1", "#1e1e2e")
        CODE_FG   = C.get("acc2", "#a6e3a1")
        INLINE_BG = C.get("bg2", "#313244")
        INLINE_FG = C.get("acc",  "#cba6f7")
        TXT       = C.get("txt",  "#cdd6f4")
        TXT2      = C.get("txt2", "#a6adc8")

        body_css = (
            f"font-family:'Inter',sans-serif;font-size:12px;"
            f"color:{TXT};background:transparent;"
            f"margin:0;padding:4px 8px;line-height:1.65;"
        )
        code_css = (
            f"font-family:'Consolas','Fira Code',monospace;font-size:11px;"
            f"color:{CODE_FG};background:{CODE_BG};"
            f"display:block;padding:10px 14px;margin:6px 0;"
            f"border-radius:6px;border-left:3px solid {CODE_FG};"
            f"white-space:pre;"
        )
        inline_css = (
            f"font-family:'Consolas','Fira Code',monospace;font-size:11px;"
            f"color:{INLINE_FG};background:{INLINE_BG};"
            f"padding:1px 4px;border-radius:3px;"
        )

        import re, html as _html

        # Split into code-block segments and normal text segments
        segments   = re.split(r'(```(?:[a-zA-Z0-9+\-]*)\n[\s\S]*?```)', text)
        html_parts = []

        for seg in segments:
            cb = re.match(r'```([a-zA-Z0-9+\-]*)\n([\s\S]*?)```', seg)
            if cb:
                lang  = cb.group(1) or "text"
                code  = _html.escape(cb.group(2).rstrip())
                html_parts.append(
                    f'<div style="{code_css}">'
                    f'<span style="font-size:9px;color:{TXT2};'
                    f'font-family:Inter,sans-serif;">{lang}</span><br>'
                    f'{code}</div>')
            else:
                # Process inline elements line-by-line
                lines = seg.split("\n")
                para_lines = []
                for line in lines:
                    # Escape HTML first
                    safe = _html.escape(line)
                    # **bold**
                    safe = re.sub(
                        r'\*\*(.+?)\*\*',
                        r'<b>\1</b>', safe)
                    # `inline code`
                    safe = re.sub(
                        r'`([^`]+)`',
                        f'<span style="{inline_css}">\\1</span>', safe)
                    # Headers: ### ## #
                    hm = re.match(r'^(#{1,3})\s+(.*)', safe)
                    if hm:
                        lvl  = len(hm.group(1))
                        size = {1: "16px", 2: "14px", 3: "13px"}.get(lvl, "13px")
                        safe = (f'<span style="font-size:{size};font-weight:700;'
                                f'color:{TXT};">{hm.group(2)}</span>')
                    # Bullet lists: - or *
                    elif re.match(r'^[\-\*]\s+', safe):
                        safe = "• " + safe[2:].lstrip()
                    para_lines.append(safe)

                # Group into paragraphs separated by blank lines
                combined = "<br>".join(para_lines)
                if combined.strip():
                    html_parts.append(
                        f'<p style="margin:2px 0;color:{TXT};">{combined}</p>')

        return (
            f'<html><body style="{body_css}">'
            + "".join(html_parts)
            + "</body></html>"
        )
