from nativelab.UI.UI_const import C
def build_qss(c: dict) -> str:
    """Generate a complete Qt stylesheet from a colour palette dict."""
    _FUI = "'Inter','Segoe UI','SF Pro Display',system-ui,-apple-system,sans-serif"
    _is_light = c["bg0"].startswith("#f") or c["bg0"].startswith("#e")
    _acc_r, _acc_g, _acc_b = (
        (74, 118, 82) if _is_light else (105, 92, 235)
    )
    def rgba(a): return f"rgba({_acc_r},{_acc_g},{_acc_b},{a})"

    return f"""
/* ═══════════════════════════════════════════════════
   Native Lab Pro — {"Cream & Sage Light" if _is_light else "Studio Dark"}
   ═══════════════════════════════════════════════════ */

QMainWindow, QDialog {{
    background:{c['bg0']};
    color:{c['txt']};
}}
* {{
    font-family:{_FUI};
    font-size:13px;
    color:{c['txt']};
}}
QWidget {{
    background:{c['bg0']};
    color:{c['txt']};
    selection-background-color:{c['acc_dim']};
    selection-color:{c['txt']};
}}
/* widgets that must stay transparent */
QFrame        {{ background:transparent; }}
QScrollArea   {{ background:transparent; border:none; }}
QStackedWidget{{ background:{c['bg0']}; }}
QTabWidget > QStackedWidget {{ background:{c['bg0']}; }}
QTabWidget > QStackedWidget > QWidget {{ background:{c['bg0']}; }}
QAbstractScrollArea#bubble_te > QWidget {{ background:transparent; }}
QSplitter     {{ background:{c['bg0']}; }}
/* log console text area */
QTextEdit#log_te {{
    background:{c['bg2']};
    color:{c['acc2']};
    border:none;
    padding:6px;
    font-family:Consolas,monospace;
    font-size:11px;
}}
/* ref sidebar panels */
QWidget#ref_panel_v2 {{ background:{c['bg1']}; border-left:1px solid {c['bdr']}; }}
QWidget#ref_panel_v1 {{ background:{c['bg1']}; border-left:1px solid {c['bdr']}; }}
/* thinking block */
QFrame#thinking_frame {{
    background:{c['bg2']};
    border:1px solid {c['bdr2']};
    border-top:none;
    border-radius:0 0 8px 8px;
}}
QTextEdit#thinking_te {{
    background:transparent;
    color:{c['txt2']};
    border:none;
    padding:10px;
    font-size:11px;
}}
QPushButton#thinking_toggle {{
    background:{c['bg2']};
    color:{c['txt2']};
    border:1px solid {c['bdr']};
    border-radius:8px;
    padding:8px 16px;
    text-align:left;
    font-size:12px;
    font-weight:500;
}}
QPushButton#thinking_toggle:hover {{
    background:{c['bg3']};
    color:{c['acc']};
    border-color:{c['bdr2']};
}}

/* ── Scrollbars ──────────────────────────────────── */
QScrollBar:vertical {{
    background:transparent; width:6px; margin:3px 0; border:none;
}}
QScrollBar::handle:vertical {{
    background:{rgba(0.30)}; border-radius:3px; min-height:32px;
}}
QScrollBar::handle:vertical:hover {{ background:{rgba(0.58)}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,  QScrollBar::sub-page:vertical {{
    height:0; width:0; background:none;
}}
QScrollBar:horizontal {{
    background:transparent; height:6px; border:none;
}}
QScrollBar::handle:horizontal {{
    background:{rgba(0.30)}; border-radius:3px; min-width:32px;
}}
QScrollBar::handle:horizontal:hover {{ background:{rgba(0.58)}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    height:0; width:0; background:none;
}}

/* ── Text inputs ─────────────────────────────────── */
QTextEdit, QPlainTextEdit {{
    background:{c['surface']};
    color:{c['txt']};
    border:1px solid {c['bdr']};
    border-radius:9px;
    padding:9px 13px;
    font-size:13px;
    line-height:1.6;
    selection-background-color:{rgba(0.28)};
}}
QTextEdit:focus, QPlainTextEdit:focus {{
    border:1px solid {rgba(0.55)};
    background:{c['surface2']};
}}
QLineEdit {{
    background:{c['surface']};
    color:{c['txt']};
    border:1px solid {c['bdr']};
    border-radius:7px;
    padding:6px 12px;
    font-size:13px;
    selection-background-color:{rgba(0.28)};
}}
QLineEdit:focus {{
    border:1px solid {rgba(0.55)};
    background:{c['surface2']};
}}

/* ── Buttons ─────────────────────────────────────── */
QPushButton {{
    background:{c['surface']};
    color:{c['txt2']};
    border:1px solid {c['bdr']};
    border-radius:7px;
    padding:6px 16px;
    min-height:30px;
    font-size:13px;
    font-weight:500;
    letter-spacing:0.1px;
}}
QPushButton:hover {{
    background:{c['surface2']};
    color:{c['txt']};
    border-color:{c['bdr2']};
}}
QPushButton:pressed {{
    background:{rgba(0.18)};
    border-color:{rgba(0.45)};
    color:{c['txt']};
}}
QPushButton:disabled {{
    color:{c['txt3']};
    background:{c['bg3']};
    border-color:{c['bdr']};
}}

/* ── Accent buttons ──────────────────────────────── */
QPushButton#btn_send {{
    background:{c['acc']};
    color:#ffffff;
    border:none;
    font-weight:600;
    border-radius:8px;
    font-size:13px;
    letter-spacing:0.2px;
}}
QPushButton#btn_send:hover  {{ background:{c['acc2']}; color:#ffffff; }}
QPushButton#btn_send:pressed {{ background:{c['glow']}; }}

QPushButton#btn_stop {{
    background:{c['err']};
    color:#ffffff;
    border:none;
    font-weight:600;
    border-radius:8px;
}}
QPushButton#btn_stop:hover {{ background:#d9413a; }}

QPushButton#btn_new {{
    background:{c['ok']};
    color:#ffffff;
    border:none;
    font-weight:700;
    border-radius:8px;
    font-size:13px;
}}
QPushButton#btn_new:hover {{ background:{c['acc']}; color:#ffffff; }}

/* ── List widget ─────────────────────────────────── */
QListWidget {{
    background:transparent;
    border:none;
    outline:none;
    font-size:13px;
}}
QListWidget::item {{
    padding:8px 12px;
    border-radius:7px;
    margin:1px 4px;
    color:{c['txt2']};
    font-size:13px;
}}
QListWidget::item:selected {{
    background:{rgba(0.16)};
    color:{c['txt']};
    border:1px solid {rgba(0.22)};
}}
QListWidget::item:hover:!selected {{
    background:{rgba(0.07)};
    color:{c['txt']};
}}

/* ── Tabs ────────────────────────────────────────── */
QTabWidget::pane {{ border:none; background:{c['bg0']}; }}
QTabBar {{ background:transparent; }}
QTabBar::tab {{
    background:transparent;
    color:{c['txt']};
    padding:10px 22px;
    border:none;
    border-bottom:2px solid transparent;
    font-size:13px;
    font-weight:500;
    margin-right:1px;
}}
QTabBar::tab:selected {{
    color:{c['txt']};
    border-bottom:2px solid {c['acc']};
    font-weight:600;
}}
QTabBar::tab:hover:!selected {{
    color:{c['acc']};
    background:{rgba(0.07)};
}}

/* ── ComboBox ────────────────────────────────────── */
QComboBox {{
    background:{c['surface']};
    color:{c['txt']};
    border:1px solid {c['bdr']};
    border-radius:7px;
    padding:6px 12px;
    min-height:30px;
    font-size:13px;
}}
QComboBox:hover  {{ border-color:{c['bdr2']}; background:{c['surface2']}; }}
QComboBox:focus  {{ border-color:{rgba(0.55)}; }}
QComboBox QAbstractItemView {{
    background:{c['surface']};
    color:{c['txt']};
    border:1px solid {c['bdr2']};
    border-radius:8px;
    padding:4px;
    margin:1px;
    selection-background-color:{rgba(0.20)};
    outline:none;
    font-size:13px;
}}
QComboBox::drop-down {{ border:none; width:22px; }}
QComboBox QAbstractItemView::item {{ border-radius:6px; padding:6px 10px; }}
QAbstractItemView {{ background:{c['surface']}; border:1px solid {c['bdr2']}; border-radius:8px; }}

/* ── Sliders ─────────────────────────────────────── */
QSlider::groove:horizontal {{
    height:5px; background:{c['bdr2']}; border-radius:3px;
}}
QSlider::handle:horizontal {{
    background:{c['acc']}; width:14px; height:14px;
    margin:-5px 0; border-radius:7px;
    border:2px solid {rgba(0.40)};
}}
QSlider::handle:horizontal:hover {{ background:{c['acc2']}; }}
QSlider::sub-page:horizontal {{
    background:{c['acc']}; border-radius:3px;
}}

/* ── Checkboxes ──────────────────────────────────── */
QCheckBox {{ color:{c['txt']}; spacing:8px; font-size:13px; }}
QCheckBox::indicator {{
    width:16px; height:16px;
    background:{c['surface']};
    border:1.5px solid {c['bdr2']};
    border-radius:4px;
}}
QCheckBox::indicator:checked {{
    background:{c['acc']}; border-color:{c['acc']};
}}
QCheckBox::indicator:hover {{ border-color:{rgba(0.60)}; }}

/* ── Group boxes ─────────────────────────────────── */
QGroupBox {{
    border:1px solid {c['bdr']};
    border-radius:9px;
    margin-top:12px; padding-top:12px;
    color:{c['txt2']};
    font-size:11px; font-weight:600; letter-spacing:0.5px;
}}
QGroupBox::title {{
    subcontrol-origin:margin; left:14px; padding:0 6px;
    color:{c['acc']};
}}

/* ── Progress bars ───────────────────────────────── */
QProgressBar {{
    background:{c['surface']};
    border:none; border-radius:4px;
    height:5px; color:transparent;
}}
QProgressBar::chunk {{ background:{c['acc']}; border-radius:3px; }}

/* ── Status bar ──────────────────────────────────── */
QStatusBar {{
    background:{c['bg0']};
    color:{c['txt3']};
    border-top:1px solid {c['bdr']};
    font-size:11px; padding:0 10px;
}}

/* ── Menu bar ────────────────────────────────────── */
QMenuBar {{
    background:{c['bg1']};
    color:{c['txt2']};
    border-bottom:1px solid {c['bdr']};
    padding:2px 4px; font-size:13px;
}}
QMenuBar::item {{ padding:5px 14px; border-radius:5px; }}
QMenuBar::item:selected {{ background:{rgba(0.14)}; color:{c['txt']}; }}
QMenu {{
    background:{c['surface']};
    color:{c['txt']};
    border:1px solid {c['bdr2']};
    padding:6px; border-radius:10px;
}}
QMenu::item {{ padding:7px 22px; border-radius:6px; font-size:13px; }}
QMenu::item:selected {{ background:{rgba(0.16)}; color:{c['txt']}; }}
QMenu::separator {{ height:1px; background:{c['bdr']}; margin:5px 10px; }}

/* ── Splitter ────────────────────────────────────── */
QSplitter::handle {{ background:{c['bdr']}; }}
QSplitter::handle:horizontal {{ width:1px; }}
QSplitter::handle:vertical   {{ height:1px; }}

/* ── Labels ──────────────────────────────────────── */
QLabel {{ background:transparent; color:{c['txt']}; font-size:13px; }}

/* ── ComboBox popup container fix ───────────────── */
QComboBox + QFrame, QComboBox + QWidget {{
    background:transparent;
    border:none;
}}

/* ── Tooltips ────────────────────────────────────── */
QToolTip {{
    background:{c['surface']};
    color:{c['txt']};
    border:1px solid {rgba(0.30)};
    padding:6px 12px; border-radius:7px; font-size:12px;
}}

/* ── Frames / separators ─────────────────────────── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color:{c['bdr']}; }}

/* ── SpinBox ─────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background:{c['surface']};
    color:{c['txt']};
    border:1px solid {c['bdr']};
    border-radius:7px;
    padding:5px 10px; font-size:13px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color:{rgba(0.55)}; }}
QSpinBox::up-button,   QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background:{c['bdr']}; border:none; width:18px; border-radius:3px;
}}
QSpinBox::up-button:hover,   QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background:{c['bdr2']};
}}

/* ── Cards in Models / Config tabs ─────────────────────── */
QFrame#tab_card {{
    background:{c['surface']};
    border:1px solid {c['bdr']};
    border-radius:12px;
}}
QFrame#tab_card QLabel {{
    background:transparent;
    color:{c['txt']};
}}

/* ── Code / mode toggle buttons in InputBar ────────────── */
QPushButton#code_btn {{
    background:{c['bg2']};
    color:{c['txt2']};
    border:1px solid {c['bdr']};
    border-radius:8px;
    padding:4px 8px;
    font-size:12px;
}}
QPushButton#code_btn:checked {{
    background:{c['acc_dim']};
    color:{c['acc']};
    border-color:{c['acc']};
    font-weight:600;
}}
QPushButton#code_btn:hover {{
    background:{c['bg3']};
    color:{c['txt']};
    border-color:{c['bdr2']};
}}
QComboBox#summary_combo {{
    background:{c['bg2']};
    color:{c['acc']};
    border:1px solid {c['bdr']};
    border-radius:8px;
    padding:3px 8px;
    font-size:10px;
}}
QLabel#pipeline_badge {{
    color:{c['pipeline']};
    font-size:10px;
    padding:2px 7px;
    background:{c['bg2']};
    border-radius:4px;
    border:1px solid {c['bdr']};
}}
QLabel#family_badge {{
    color:{c['acc2']};
    font-size:10px;
    background:{c['bg2']};
    border-radius:4px;
    padding:2px 7px;
}}

/* ── Engine / model list widgets ────────────────────────── */
QListWidget#model_list, QListWidget#engine_list {{
    background:{c['bg1']};
    border:none;
    font-size:11px;
    outline:none;
}}
QListWidget#model_list::item, QListWidget#engine_list::item {{
    padding:8px 12px;
    border-bottom:1px solid {c['bdr']};
    color:{c['txt2']};
}}
QListWidget#model_list::item:selected,
QListWidget#engine_list::item:selected {{
    background:{c['acc_dim']};
    color:{c['acc']};
}}
QListWidget#paused_list {{
    background:{c['bg1']};
    border:none;
    font-size:10px;
    outline:none;
}}
QListWidget#paused_list::item {{
    padding:5px 10px;
    border-bottom:1px solid {c['bdr']};
    color:{c['warn']};
}}

/* ── Ref panel tab widget ───────────────────────────────── */
QTabWidget#ref_tabs::pane {{
    background:{c['bg1']};
    border:1px solid {c['bdr']};
    border-radius:6px;
}}
QTabWidget#ref_tabs QTabBar::tab {{
    background:{c['bg2']};
    color:{c['txt']};
    padding:5px 12px;
    border-radius:4px 4px 0 0;
    font-size:10px;
    border:1px solid {c['bdr']};
    border-bottom:none;
}}
QTabWidget#ref_tabs QTabBar::tab:selected {{
    color:{c['txt']};
    background:{c['bg1']};
    border-bottom:2px solid {c['acc']};
    font-weight:600;
}}
QTabWidget#ref_tabs QTabBar::tab:hover:!selected {{
    background:{c['bg3']};
    color:{c['acc']};
}}

/* ── Ref panel list widgets ─────────────────────────────── */
QListWidget#ref_doc_list, QListWidget#ref_script_list {{
    background:transparent;
    border:none;
    font-size:10px;
    outline:none;
}}
QListWidget#ref_doc_list::item, QListWidget#ref_script_list::item {{
    padding:5px 8px;
    border-radius:5px;
    margin:2px 0;
    color:{c['txt2']};
    min-height:18px;
}}
QListWidget#ref_doc_list::item:hover,
QListWidget#ref_script_list::item:hover {{
    background:{c['acc_dim']};
}}
QListWidget#ref_doc_list::item:selected,
QListWidget#ref_script_list::item:selected {{
    background:{c['acc_dim']};
    color:{c['acc']};
    border:1px solid {c['bdr2']};
}}

/* ── Ref panel info / detail labels ─────────────────────── */
QLabel#ref_info_banner {{
    color:{c['txt2']};
    font-size:9px;
    padding:4px 6px;
    background:{c['bg2']};
    border-radius:4px;
}}
QLabel#ref_script_detail {{
    color:{c['txt2']};
    font-size:9px;
    padding:6px 8px;
    background:{c['bg2']};
    border:1px solid {c['bdr']};
    border-radius:6px;
    min-height:60px;
}}
QLabel#ref_ram_info {{
    color:{c['txt2']};
    font-size:9px;
    padding:2px;
}}
QLabel#ref_hdr {{
    color:{c['txt']};
    font-weight:700;
    font-size:12px;
}}
QLabel#ram_badge_lbl {{
    color:{c['warn']};
    font-size:9px;
    padding:1px 5px;
    background:{c['bg2']};
    border-radius:3px;
}}

/* ── Fix white bleed on QPushButton borders in dark mode ── */
QPushButton {{
    outline:none;
}}
QTabBar {{ background:{c['bg1']}; }}
QTabBar::scroller {{ background:{c['bg1']}; }}
QTabBar QToolButton {{ background:{c['bg1']}; border:none; color:{c['txt2']}; }}

/* ── Appearance tab ─────────────────────────────────────── */
QWidget#appearance_bar {{
    background:{c['bg1']};
    border-bottom:1px solid {c['bdr']};
}}
QLabel#appearance_hdr {{
    color:{c['txt']};
    font-size:13px;
    font-weight:700;
}}
QLabel#appearance_group_hdr {{
    color:{c['acc']};
    font-size:10px;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:0.8px;
    margin-bottom:4px;
}}
QLabel#appearance_row_lbl {{
    color:{c['txt2']};
    font-size:11px;
}}
QLabel#appearance_sl_lbl {{
    color:{c['txt3']};
    font-size:9px;
    font-weight:700;
}}
QLineEdit#appearance_hex {{
    background:{c['bg2']};
    color:{c['txt']};
    border:1px solid {c['bdr']};
    border-radius:5px;
    padding:2px 6px;
    font-size:11px;
    font-family:Consolas,monospace;
}}
QLineEdit#appearance_hex:focus {{
    border-color:{c['acc']};
}}
QSlider#appearance_slider::groove:horizontal {{
    height:4px;
    background:{c['bg3']};
    border-radius:2px;
}}
QSlider#appearance_slider::handle:horizontal {{
    background:{c['acc']};
    width:12px;
    height:12px;
    margin:-4px 0;
    border-radius:6px;
}}
QSlider#appearance_slider::sub-page:horizontal {{
    background:{c['acc_dim']};
    border-radius:2px;
}}
QPushButton#appearance_btn {{
    background:{c['bg2']};
    color:{c['txt2']};
    border:1px solid {c['bdr']};
    border-radius:7px;
    padding:5px 14px;
    font-size:11px;
    font-weight:500;
}}
QPushButton#appearance_btn:hover {{
    background:{c['bg3']};
    color:{c['txt']};
}}
QPushButton#appearance_btn_acc {{
    background:{c['acc']};
    color:#ffffff;
    border:none;
    border-radius:7px;
    padding:5px 14px;
    font-size:11px;
    font-weight:600;
}}
QPushButton#appearance_btn_acc:hover {{
    background:{c['acc2']};
}}

/* ── Named layout containers (override inline styles) ────── */
QScrollArea#chat_scroll         {{ background:{c['bg0']}; border:none; }}
QWidget#chat_container          {{ background:{c['bg0']}; }}
QWidget#input_bar               {{ background:{c['bg1']}; border-top:1px solid {c['bdr']}; }}
QWidget#session_sidebar         {{ background:{c['bg1']}; border-right:1px solid {c['bdr']}; }}
QWidget#ref_panel               {{ background:{c['bg1']}; border-left:1px solid {c['bdr']}; }}
QWidget#chat_module_root        {{ background:{c['bg0']}; }}

/* ── Message bubbles via object name ────────────────────── */
QFrame#bubble_user {{
    background:{c['usr']};
    border:1px solid {rgba(0.30)};
    border-radius:14px 14px 4px 14px;
}}
QFrame#bubble_ast {{
    background:{c['ast']};
    border:1px solid {rgba(0.14)};
    border-radius:4px 14px 14px 14px;
}}
QFrame#bubble_rsn {{
    background:{c['rsn']};
    border:1px solid rgba(24,176,202,0.22);
    border-radius:4px 14px 14px 14px;
}}
QFrame#bubble_cod {{
    background:{c['cod']};
    border:1px solid rgba(28,184,138,0.20);
    border-radius:4px 14px 14px 14px;
}}

/* ── Chat text body inside bubbles ──────────────────────── */
QTextBrowser#bubble_te {{
    background:transparent;
    color:{c['txt']};
    font-size:14px;
    border:none;
    padding:0;
    line-height:1.7;
}}
QLabel#txt2 {{
    color:{c['txt2']};
    font-size:12px;
}}
QLabel#txt2_XXL {{
    color:{c['txt2']};
    font-size:20px;
}}
QLabel#txt2_small {{
    color:{c['txt2']};
    font-size:11px;
}}
QLabel#txt2_xs {{
    color:{c['txt2']};
    font-size:10px;
}}
QLabel#txt3_xs {{
    color:{c['txt3']};
    font-size:10px;
}}
/* ===== GPU badge ===== */

QLabel#gpu_badge {{
    font-size:11px;
    padding:5px 10px;
    border-radius:5px;
    background:{c['bg2']};
}}

/* States */
QLabel#gpu_badge[state="cuda"],
QLabel#gpu_badge[state="metal"] {{
    color:{c['ok']};
}}
QLabel#gpu_badge[state="vulkan"] {{
    color:{c['warn']};
}}
QLabel#gpu_badge[state="none"] {{
    color:{c['txt2']};
}}
/* ===== Resolved paths box ===== */

QLabel#resolved_box {{
    color:{c['txt2']};
    font-size:10px;
    padding:6px 8px;
    background:{c['bg2']};
    border-radius:5px;
}}
/* ===== Pipeline / sidebar text system ===== */
QLabel#txt3_tiny {{
    color:{c['txt3']};
    font-size:9px;
    font-weight:700;
    letter-spacing:1.1px;
}}

QLabel#txt3_block {{
    color:{c['txt3']};
    font-size:9px;
    padding:4px 2px;
}}

/* Badge system (like resolved box) */
QLabel#status_badge {{
    font-size:10px;
    padding:2px 7px;
    border-radius:4px;
    background:{c['bg2']};
}}
/* Log panel */
QTextEdit#log_te span.log_ts {{
    color: {c['txt2']};
}}

QTextEdit#log_te span.log_msg {{
    color: {c['txt']};
}}

QLabel#status_badge[state="idle"]  {{ color:{c['txt3']}; }}
QLabel#status_badge[state="ok"]    {{ color:{c['ok']}; }}
QLabel#status_badge[state="warn"]  {{ color:{c['warn']}; }}

/* ══════════════════════════════════════════════════════════════════
   Labs Tab
   ══════════════════════════════════════════════════════════════════ */

QWidget#labs_sidebar {{
    background:{c['bg1']};
    border-right:1px solid {c['bdr']};
}}
QLabel#labs_sidebar_hdr {{
    color:{c['txt3']};
    font-size:9px;
    font-weight:700;
    letter-spacing:1.2px;
    padding:14px 16px 6px 16px;
    background:transparent;
}}
QListWidget#labs_nav {{
    background:transparent;
    border:none;
    outline:none;
    padding:0 6px;
}}
QListWidget#labs_nav::item {{
    padding:9px 10px;
    border-radius:7px;
    margin:1px 0;
    color:{c['txt2']};
    font-size:13px;
    font-weight:500;
}}
QListWidget#labs_nav::item:selected {{
    background:{c['acc_dim']};
    color:{c['acc']};
    font-weight:600;
}}
QListWidget#labs_nav::item:hover:!selected {{
    background:{c['highlight']};
    color:{c['txt']};
}}
QStackedWidget#labs_stack {{
    background:{c['bg0']};
}}
QLabel#labs_panel_header {{
    font-size:16px;
    font-weight:bold;
    margin-bottom:4px;
    color:{c['txt']};
    background:transparent;
}}
QPushButton#labs_generate_btn {{
    background:{c['acc']};
    color:#ffffff;
    border:none;
    font-weight:600;
    border-radius:8px;
    font-size:13px;
    letter-spacing:0.15px;
    min-height:38px;
    padding:0 24px;
}}
QPushButton#labs_generate_btn:hover  {{ background:{c['acc2']}; color:#ffffff; }}
QPushButton#labs_generate_btn:pressed {{ background:{c['glow']}; }}
QPushButton#labs_generate_btn:disabled {{
    background:{c['bg3']};
    color:{c['txt3']};
    border:1px solid {c['bdr']};
}}
QTextEdit#labs_preview_te {{
    background:{c['surface']};
    color:{c['txt']};
    border:1px solid {c['bdr']};
    border-radius:9px;
    padding:10px 14px;
    font-size:12px;
    selection-background-color:{c['acc_dim']};
}}
QTextEdit#labs_preview_te:focus {{
    border-color:{c['bdr2']};
    background:{c['surface2']};
}}
 """



API_MODELS_QSS_BLOCK = """
/* ══════════════════════════════════════════════════════
   ApiModelsTab — objectName-based styles
   ══════════════════════════════════════════════════════ */

/* Section header labels  (e.g. "NEW CONNECTION", "SAVED CONFIGS") */
QLabel#sec_lbl {{
    color:{c['txt3']};
    font-size:9px;
    font-weight:700;
    letter-spacing:1.2px;
    background:transparent;
}}

/* Secondary/muted small text  (e.g. subtitle, card sub-lines) */
QLabel#txt2_small {{
    color:{c['txt2']};
    font-size:11px;
    background:transparent;
}}

/* All text inputs in this tab */
QLineEdit#input {{
    background:{c['bg1']};
    border:1px solid {c['bdr']};
    border-radius:6px;
    color:{c['txt']};
    padding:0 10px;
    font-size:12px;
}}
QLineEdit#input:focus {{
    border-color:{c['acc']};
    background:{c['surface2']};
}}
QLineEdit#input:disabled {{
    color:{c['txt3']};
    background:{c['bg3']};
}}

/* Multi-line text edit (system-prompt area) — inherits global QTextEdit,
   but this tightens the border-radius to match the compact card style */
QWidget#card_inner QTextEdit {{
    background:{c['bg1']};
    border:1px solid {c['bdr']};
    border-radius:6px;
    color:{c['txt']};
    padding:6px 10px;
    font-size:11px;
}}
QWidget#card_inner QTextEdit:focus {{
    border-color:{c['acc']};
}}

/* Combo boxes */
QComboBox#combo {{
    background:{c['bg1']};
    border:1px solid {c['bdr']};
    border-radius:6px;
    color:{c['txt']};
    padding:0 10px;
    font-size:12px;
}}
QComboBox#combo:hover {{
    border-color:{c['bdr2']};
    background:{c['surface2']};
}}
QComboBox#combo:focus {{
    border-color:{c['acc']};
}}
QComboBox#combo::drop-down {{
    border:none;
    width:22px;
}}
QComboBox#combo QAbstractItemView {{
    background:{c['bg2']};
    color:{c['txt']};
    selection-background-color:{c['acc_dim']};
    border:1px solid {c['bdr2']};
    border-radius:6px;
    padding:3px;
    margin:1px;
    outline:none;
    font-size:12px;
}}

/* Outline action buttons  (color role stored in btn_color property) */
QPushButton#outline_btn {{
    background:transparent;
    border:1px solid {c['bdr2']};
    border-radius:7px;
    color:{c['txt2']};
    font-size:11px;
    font-weight:600;
    padding:0 16px;
}}
QPushButton#outline_btn:hover {{
    background:{c['bdr2']};
    color:{c['txt']};
}}
QPushButton#outline_btn:disabled {{
    color:{c['txt3']};
    border-color:{c['bdr']};
}}

/* Semantic color variants via btn_color property */
QPushButton#outline_btn[btn_color="ok"] {{
    color:{c['ok']};
    border-color:{c['ok']};
}}
QPushButton#outline_btn[btn_color="ok"]:hover {{
    background:{c['ok']};
    color:#ffffff;
}}
QPushButton#outline_btn[btn_color="acc"] {{
    color:{c['acc']};
    border-color:{c['acc']};
}}
QPushButton#outline_btn[btn_color="acc"]:hover {{
    background:{c['acc']};
    color:#ffffff;
}}
QPushButton#outline_btn[btn_color="err"] {{
    color:{c['err']};
    border-color:{c['err']};
}}
QPushButton#outline_btn[btn_color="err"]:hover {{
    background:{c['err']};
    color:#ffffff;
}}

/* Saved-config card frames */
QFrame#card {{
    background:{c['bg2']};
    border:1px solid {c['bdr']};
    border-radius:8px;
}}
QFrame#card QLabel {{
    background:transparent;
}}

/* Connection card scroll area */
QScrollArea#card_scroll {{
    background:{c['bg2']};
    border:1px solid {c['bdr']};
    border-radius:10px;
}}
QScrollArea#card_scroll > QWidget > QWidget {{
    background:{c['bg2']};
}}
QScrollArea#card_scroll QScrollBar:vertical {{
    background:{c['bg1']};
    width:6px;
    border-radius:3px;
}}
QScrollArea#card_scroll QScrollBar::handle:vertical {{
    background:{c['bdr2']};
    border-radius:3px;
}}
QScrollArea#card_scroll QScrollBar::add-line:vertical,
QScrollArea#card_scroll QScrollBar::sub-line:vertical {{
    height:0px;
}}

/* Status label  — state property drives color */
QLabel#api_status {{
    font-size:11px;
    background:transparent;
}}
QLabel#api_status[state=""] {{
    color:{c['txt3']};
}}
QLabel#api_status[state="warn"] {{
    color:{c['warn']};
}}
QLabel#api_status[state="ok"] {{
    color:{c['ok']};
}}
QLabel#api_status[state="err"] {{
    color:{c['err']};
}}
"""
LABS_QSS_BLOCK = """
QWidget#labs_sidebar {{
    background:{c['bg1']};
    border-right:1px solid {c['bdr']};
}}
 
/* Section header inside the sidebar */
QLabel#labs_sidebar_hdr {{
    color:{c['txt3']};
    font-size:9px;
    font-weight:700;
    letter-spacing:1.2px;
    padding:14px 16px 6px 16px;
    background:transparent;
}}
 
/* Nav list — inherits global QListWidget but we tune spacing */
QListWidget#labs_nav {{
    background:transparent;
    border:none;
    outline:none;
    padding:0 6px;
}}
QListWidget#labs_nav::item {{
    padding:9px 10px;
    border-radius:7px;
    margin:1px 0;
    color:{c['txt2']};
    font-size:13px;
    font-weight:500;
}}
QListWidget#labs_nav::item:selected {{
    background:{c['acc_dim']};
    color:{c['acc']};
    border:1px solid {c['acc_dim']};
    font-weight:600;
}}
QListWidget#labs_nav::item:hover:!selected {{
    background:{c['highlight']};
    color:{c['txt']};
}}
 
/* Stacked content area */
QStackedWidget#labs_stack {{
    background:{c['bg0']};
}}
 
/* Panel header label */
QLabel#labs_panel_header {{
    font-size:16px;
    font-weight:bold;
    margin-bottom:4px;
    color:{c['txt']};
    background:transparent;
}}
 
/* Primary generate button — accent filled */
QPushButton#labs_generate_btn {{
    background:{c['acc']};
    color:#ffffff;
    border:none;
    font-weight:600;
    border-radius:8px;
    font-size:13px;
    letter-spacing:0.15px;
    min-height:38px;
    padding:0 24px;
}}
QPushButton#labs_generate_btn:hover {{
    background:{c['acc2']};
    color:#ffffff;
}}
QPushButton#labs_generate_btn:pressed {{
    background:{c['glow']};
}}
QPushButton#labs_generate_btn:disabled {{
    background:{c['bg3']};
    color:{c['txt3']};
    border:1px solid {c['bdr']};
}}
 
/* Live-preview text area — monospaced, subtle surface */
QTextEdit#labs_preview_te {{
    background:{c['surface']};
    color:{c['txt']};
    border:1px solid {c['bdr']};
    border-radius:9px;
    padding:10px 14px;
    font-size:12px;
    line-height:1.6;
    selection-background-color:{c['acc_dim']};
}}
QTextEdit#labs_preview_te:focus {{
    border-color:{c['bdr2']};
    background:{c['surface2']};
}}
"""
QSS = build_qss(C)
