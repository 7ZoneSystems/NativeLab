from __future__ import annotations

from typing import Any

from nativelab.imports.import_global import QMessageBox
from nativelab.core.llm_errors import LlmErrorNotice, explain_llm_error


def show_llm_error_dialog(parent: Any, raw_error: Any, *, source: str = "LLM engine") -> LlmErrorNotice:
    notice = explain_llm_error(raw_error, source=source)
    try:
        box = QMessageBox(parent)
        try:
            box.setIcon(QMessageBox.Icon.Warning)
        except Exception:
            pass
        box.setWindowTitle(notice.title)
        box.setText(notice.summary)
        box.setInformativeText(f"What to do:\n{notice.action}")
        if notice.technical_detail:
            box.setDetailedText(notice.technical_detail)
        box.exec()
    except Exception:
        try:
            QMessageBox.warning(parent, notice.title, notice.user_message)
        except Exception:
            pass
    return notice
