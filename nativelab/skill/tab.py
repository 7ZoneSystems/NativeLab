from __future__ import annotations

from nativelab.imports.import_global import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QColor,
    Qt,
    pyqtSignal,
)
from nativelab.UI.UI_const import C
from nativelab.UI.icons import icon, icon_size, set_button_icon, set_label_icon

from .manager import SKILLS_FILE, delete_skill, ensure_builtin_edit_skill, load_skills, upsert_skill


class SkillsTab(QWidget):
    skills_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._skills: list[dict] = []
        ensure_builtin_edit_skill()
        self._build()
        self._reload()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        left = QFrame()
        left.setObjectName("tab_card")
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(12, 12, 12, 12)
        left_l.setSpacing(8)

        self.hdr = QLabel("Skills")
        set_label_icon(self.hdr, "lightbulb", "Skills", 18)
        self.hdr.setStyleSheet("font-size:14px;font-weight:bold;")
        left_l.addWidget(self.hdr)

        self.skill_list = QListWidget()
        self.skill_list.setObjectName("model_list")
        self.skill_list.setIconSize(icon_size(18))
        self.skill_list.currentItemChanged.connect(self._load_selected)
        left_l.addWidget(self.skill_list, 1)

        row = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_delete = QPushButton("Delete")
        set_button_icon(self.btn_new, "plus", "New")
        set_button_icon(self.btn_delete, "delete", "Delete")
        self.btn_new.clicked.connect(self._new_skill)
        self.btn_delete.clicked.connect(self._delete_selected)
        row.addWidget(self.btn_new)
        row.addWidget(self.btn_delete)
        left_l.addLayout(row)
        root.addWidget(left, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setObjectName("chat_scroll")
        body = QWidget()
        body.setObjectName("chat_container")
        form = QVBoxLayout(body)
        form.setContentsMargins(22, 18, 22, 22)
        form.setSpacing(10)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        self.title = QLabel("Model Skill Library")
        set_label_icon(self.title, "lightbulb", "Model Skill Library", 18)
        self.title.setStyleSheet("font-size:16px;font-weight:bold;")
        form.addWidget(self.title)

        sub = QLabel(
            "Create reusable skills that can be injected into any model call when "
            "the chat Skills toggle is enabled. Active skills are shared through "
            "the NativeLab skill endpoint and integration surface."
        )
        sub.setWordWrap(True)
        sub.setObjectName("txt2_small")
        form.addWidget(sub)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Skill name")
        self.name_edit.setFixedHeight(30)
        form.addLayout(self._field("Name:", self.name_edit))

        self.enabled_chk = QCheckBox("Active")
        self.enabled_chk.setChecked(True)
        form.addWidget(self.enabled_chk)

        self.desc_edit = QTextEdit()
        self.desc_edit.setPlaceholderText("Short description shown to the model")
        self.desc_edit.setFixedHeight(70)
        form.addLayout(self._field("Description:", self.desc_edit))

        self.instructions_edit = QTextEdit()
        self.instructions_edit.setPlaceholderText("Skill instructions / behavior / examples")
        self.instructions_edit.setMinimumHeight(240)
        form.addLayout(self._field("Instructions:", self.instructions_edit))

        save_row = QHBoxLayout()
        self.btn_save = QPushButton("Save Skill")
        set_button_icon(self.btn_save, "save", "Save Skill")
        self.btn_save.setFixedHeight(32)
        self.btn_save.clicked.connect(self._save_skill)
        self.path_lbl = QLabel(f"Saved in {SKILLS_FILE}")
        self.path_lbl.setObjectName("txt2_small")
        save_row.addWidget(self.btn_save)
        save_row.addWidget(self.path_lbl, 1)
        form.addLayout(save_row)
        form.addStretch()

    @staticmethod
    def _field(label: str, widget) -> QVBoxLayout:
        box = QVBoxLayout()
        lbl = QLabel(label)
        lbl.setObjectName("txt2")
        box.addWidget(lbl)
        box.addWidget(widget)
        return box

    def _reload(self):
        current = self.skill_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.skill_list.currentItem() else ""
        self._skills = load_skills()
        self.skill_list.blockSignals(True)
        self.skill_list.clear()
        for skill in self._skills:
            label = skill.get("name", "Unnamed")
            if not skill.get("enabled", True):
                label = f"{label} (off)"
            item = QListWidgetItem(icon("lightbulb"), label)
            item.setData(Qt.ItemDataRole.UserRole, skill.get("name", ""))
            self.skill_list.addItem(item)
        self.skill_list.blockSignals(False)
        if self.skill_list.count():
            row = 0
            for i in range(self.skill_list.count()):
                if self.skill_list.item(i).data(Qt.ItemDataRole.UserRole) == current:
                    row = i
                    break
            self.skill_list.setCurrentRow(row)
            self._load_selected()
        else:
            self._new_skill()
        self.refresh_theme()

    def refresh_theme(self):
        set_label_icon(self.hdr, "lightbulb", "Skills", 18)
        set_label_icon(self.title, "lightbulb", "Model Skill Library", 18)
        current = self.skill_list.currentRow()
        for i in range(self.skill_list.count()):
            item = self.skill_list.item(i)
            if item:
                item.setForeground(QColor(C["acc"] if i == current else C["txt2"]))

    def _new_skill(self):
        self.skill_list.clearSelection()
        self.name_edit.setText("")
        self.enabled_chk.setChecked(True)
        self.desc_edit.clear()
        self.instructions_edit.clear()

    def _load_selected(self):
        self.refresh_theme()
        item = self.skill_list.currentItem()
        if not item:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        for skill in self._skills:
            if skill.get("name") == name:
                self.name_edit.setText(skill.get("name", ""))
                self.enabled_chk.setChecked(bool(skill.get("enabled", True)))
                self.desc_edit.setPlainText(skill.get("description", ""))
                self.instructions_edit.setPlainText(skill.get("instructions", ""))
                return

    def _save_skill(self):
        try:
            saved = upsert_skill({
                "name": self.name_edit.text().strip(),
                "enabled": self.enabled_chk.isChecked(),
                "description": self.desc_edit.toPlainText().strip(),
                "instructions": self.instructions_edit.toPlainText().strip(),
            })
        except Exception as exc:
            QMessageBox.warning(self, "Skill Error", str(exc))
            return
        self._reload()
        for i in range(self.skill_list.count()):
            if self.skill_list.item(i).data(Qt.ItemDataRole.UserRole) == saved["name"]:
                self.skill_list.setCurrentRow(i)
                break
        self.skills_changed.emit()

    def _delete_selected(self):
        name = self.name_edit.text().strip()
        if not name:
            return
        if QMessageBox.question(
            self,
            "Delete Skill",
            f"Delete skill '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        delete_skill(name)
        self._reload()
        self.skills_changed.emit()
