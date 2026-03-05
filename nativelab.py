#!/usr/bin/env python3
"""
Native Lab Pro v2 — Local LLM Desktop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v2 New Features:
  · Full GGUF quant format detection (Q2_K → F32, IQ* imatrix quants)
  · Auto model-family detection from filename → correct prompt template
    (DeepSeek, Mistral, LLaMA-2/3, Phi, Qwen/ChatML, Gemma, CodeLlama,
     Falcon, Vicuna, OpenChat, Neural-Chat, Starling, Yi, Command-R…)
  · Parallel model loading toggle with CPU/RAM warnings
  · Pipeline mode: Reasoning → Coding chain
    (reasoning model summarises intent → coding model generates code)
  · Python/code snippet copy buttons inside chat bubbles
  · All existing v1 features preserved
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import sys, os, re, json, time, socket, signal, subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import re, ast, json, hashlib, pickle, threading, time

# ── PyQt6 ────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QFileDialog, QLabel, QListWidget,
    QListWidgetItem, QSplitter, QTabWidget, QScrollArea, QFrame,
    QComboBox, QProgressBar, QMenu, QMessageBox, QInputDialog,
    QSizePolicy, QLineEdit, QSlider, QCheckBox, QGroupBox, QTextBrowser
)
from PyQt6.QtGui import (
    QFont, QColor, QTextCursor, QAction, QKeySequence, QIcon
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import psutil; HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from PyPDF2 import PdfReader; HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import hashlib, pickle, struct
    HAS_HASH = True
except ImportError:
    HAS_HASH = False

# ── Reference system constants ────────────────────────────────────────────────
REFS_DIR          = Path("chat_refs")
REF_CACHE_DIR     = Path("ref_cache")
REF_INDEX_DIR     = Path("ref_index")
PAUSED_JOBS_DIR   = Path("paused_jobs")
APP_CONFIG_FILE   = Path("app_config.json")
REFS_DIR.mkdir(parents=True, exist_ok=True)
REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
REF_INDEX_DIR.mkdir(parents=True, exist_ok=True)
PAUSED_JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ── App-wide user config (all thresholds editable in Config tab) ──────────────
APP_CONFIG_DEFAULTS = {
    "ram_watchdog_mb":        800,
    "chunk_index_size":       400,
    "max_ram_chunks":         120,
    "summary_chunk_chars":    3000,
    "summary_ctx_carry":      600,
    "summary_n_pred_sect":    380,
    "summary_n_pred_final":   700,
    "multipdf_n_pred_sect":   380,
    "multipdf_n_pred_final":  900,
    "ref_top_k":              6,
    "ref_max_context_chars":  3000,
    "pause_after_chunks":     2,   # auto-pause suggestion threshold
    "default_threads":        12,
    "default_ctx":            4096,
    "default_n_predict":      512,
    "tps_display":            True,
    "auto_spill_on_start":    False,
}

def _load_app_config() -> Dict:
    cfg = dict(APP_CONFIG_DEFAULTS)
    if APP_CONFIG_FILE.exists():
        try:
            saved = json.loads(APP_CONFIG_FILE.read_text())
            cfg.update({k: v for k, v in saved.items() if k in cfg})
        except Exception:
            pass
    return cfg

def _save_app_config(cfg: Dict):
    APP_CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

APP_CONFIG = _load_app_config()

# Dynamic accessors (use these everywhere instead of bare constants)
def RAM_WATCHDOG_MB()     -> float: return float(APP_CONFIG["ram_watchdog_mb"])
def CHUNK_INDEX_SIZE()    -> int:   return int(APP_CONFIG["chunk_index_size"])
def MAX_RAM_CHUNKS()      -> int:   return int(APP_CONFIG["max_ram_chunks"])
def DEFAULT_THREADS()     -> int:   return int(APP_CONFIG["default_threads"])
def DEFAULT_CTX()         -> int:   return int(APP_CONFIG["default_ctx"])
def DEFAULT_N_PRED()      -> int:   return int(APP_CONFIG["default_n_predict"])

# ═════════════════════════════ CONFIGURATION ════════════════════════════════

import sys as _sys
_BASE = Path(_sys._MEIPASS) if getattr(_sys, "frozen", False) else Path(".")
_EXT  = ".exe" if __import__("platform").system() == "Windows" else ""

LLAMA_CLI    = str(_BASE / f"llama-bin/llama-cli{_EXT}")
LLAMA_SERVER = str(_BASE / f"llama-bin/llama-server{_EXT}")

# Fallback to local llama/bin for dev mode
if not Path(LLAMA_CLI).exists():
    LLAMA_CLI    = f"./llama/bin/llama-cli{_EXT}"
    LLAMA_SERVER = f"./llama/bin/llama-server{_EXT}"
MODELS_DIR   = Path("./localllm")
SESSIONS_DIR = Path("./sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL      = "./localllm/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
DEFAULT_THREADS    = 12
DEFAULT_CTX        = 4096
DEFAULT_N_PRED     = 512
CUSTOM_MODELS_FILE  = Path("./localllm/custom_models.json")
MODEL_CONFIGS_FILE  = Path("./localllm/model_configs.json")
PARALLEL_PREFS_FILE = Path("./localllm/parallel_prefs.json")

# ── model roles ───────────────────────────────────────────────────────────────
MODEL_ROLES = ["general", "reasoning", "summarization", "coding", "secondary"]
ROLE_ICONS  = {
    "general":       "💬",
    "reasoning":     "🧠",
    "summarization": "📄",
    "coding":        "💻",
    "secondary":     "🔀",
}

# ═════════════════════════════ MODEL FAMILY DETECTION ═══════════════════════

# All GGUF quantization formats supported by llama.cpp
GGUF_QUANT_PATTERNS = [
    # imatrix importance quants
    r'IQ1_S', r'IQ1_M', r'IQ2_XXS', r'IQ2_XS', r'IQ2_S', r'IQ2_M',
    r'IQ3_XXS', r'IQ3_XS', r'IQ3_S', r'IQ3_M', r'IQ4_XS', r'IQ4_NL',
    # K-quants
    r'Q2_K(?:_S)?', r'Q3_K_(?:S|M|L)', r'Q3_K',
    r'Q4_K_(?:S|M)', r'Q4_K',
    r'Q5_K_(?:S|M)', r'Q5_K',
    r'Q6_K',
    # legacy quants
    r'Q4_0', r'Q4_1', r'Q5_0', r'Q5_1', r'Q8_0',
    # float
    r'F16', r'F32', r'BF16',
    # GGML legacy
    r'Q4_0_4_4', r'Q4_0_4_8', r'Q4_0_8_8',
]

QUANT_REGEX = re.compile(
    r'\.(' + '|'.join(GGUF_QUANT_PATTERNS) + r')\.gguf$',
    re.IGNORECASE
)

@dataclass
class ModelFamily:
    """Detected model family with prompt template configuration."""
    name:           str
    family:         str      # internal key
    template:       str      # prompt template style
    system_prefix:  str = ""
    system_suffix:  str = ""
    user_prefix:    str = ""
    user_suffix:    str = ""
    assistant_prefix: str = ""
    assistant_suffix: str = ""
    stop_tokens:    List[str] = field(default_factory=list)
    bos:            str = "<s>"
    eos:            str = "</s>"

C = {
    "bg0":  "#07070f", "bg1":  "#0d0d1c", "bg2":  "#13132a",
    "bg3":  "#1a1a35",
    "acc":  "#a78bfa", "acc2": "#c4b5fd",
    "usr":  "#12103a", "ast":  "#0b1a12",
    "txt":  "#ede8ff", "txt2": "#7e7a9a", "bdr":  "#252340",
    "ok":   "#34d399", "warn": "#fbbf24", "err":  "#f87171",
    "glow": "#7c3aed",
    "pipeline": "#22d3ee",
}

# ── Prompt templates for each family ─────────────────────────────────────────
FAMILY_TEMPLATES: Dict[str, ModelFamily] = {
    "deepseek": ModelFamily(
        name="DeepSeek", family="deepseek", template="deepseek",
        user_prefix="User: ", user_suffix="\n\nAssistant:",
        assistant_prefix="",
        assistant_suffix="\n\n",
        stop_tokens=["<|EOT|>", "\nUser:", "\n\nUser:"],
        bos="<｜begin▁of▁sentence｜>", eos="<｜end▁of▁sentence｜>",
    ),
    "deepseek-coder": ModelFamily(
        name="DeepSeek-Coder", family="deepseek-coder", template="deepseek",
        user_prefix="### Instruction:\n", user_suffix="\n### Response:\n",
        assistant_prefix="",
        assistant_suffix="\n",
        stop_tokens=["<|EOT|>", "\n### Instruction:"],
        bos="<｜begin▁of▁sentence｜>", eos="<｜end▁of▁sentence｜>",
    ),
    "deepseek-r1": ModelFamily(
        name="DeepSeek-R1", family="deepseek-r1", template="deepseek-r1",
        user_prefix="<|User|>", user_suffix="<|Assistant|><think>\n",
        assistant_prefix="",
        assistant_suffix="</think>\n",
        stop_tokens=["<|end▁of▁sentence|>", "<｜end▁of▁sentence｜>",
                     "<|EOT|>", "\n<|User|>"],
        bos="<｜begin▁of▁sentence｜>", eos="<｜end▁of▁sentence｜>",
    ),
    "mistral": ModelFamily(
        name="Mistral Instruct", family="mistral", template="mistral",
        user_prefix="[INST] ", user_suffix=" [/INST]",
        assistant_prefix="",          # ← THIS must exist, even if empty string
        assistant_suffix="</s>",
        stop_tokens=["</s>", "[INST]", "[/INST]", "### Human:", "### Assistant:",
                     "[ORG_NAME]", "[NAME]", "[USER]", "\n[", "```\n\n["],
        bos="<s>", eos="</s>",
    ),
    "mixtral": ModelFamily(
        name="Mixtral MoE", family="mixtral", template="mistral",
        user_prefix="[INST] ", user_suffix=" [/INST]",
        assistant_suffix="</s>",
        stop_tokens=["</s>", "[INST]"],
        bos="<s>", eos="</s>",
    ),
    "llama2": ModelFamily(
        name="LLaMA-2 Chat", family="llama2", template="llama2",
        system_prefix="[INST] <<SYS>>\n", system_suffix="\n<</SYS>>\n\n",
        user_prefix="", user_suffix=" [/INST]",
        assistant_suffix="</s><s>[INST] ",
        stop_tokens=["</s>", "[INST]"],
        bos="<s>", eos="</s>",
    ),
    "llama3": ModelFamily(
        name="LLaMA-3 Instruct", family="llama3", template="llama3",
        system_prefix="<|start_header_id|>system<|end_header_id|>\n\n",
        system_suffix="<|eot_id|>",
        user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        user_suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        assistant_suffix="<|eot_id|>",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
        bos="<|begin_of_text|>", eos="<|end_of_text|>",
    ),
    "phi": ModelFamily(
        name="Phi Instruct", family="phi", template="phi",
        system_prefix="<|system|>\n", system_suffix="<|end|>\n",
        user_prefix="<|user|>\n", user_suffix="<|end|>\n<|assistant|>\n",
        assistant_suffix="<|end|>\n",
        stop_tokens=["<|end|>", "<|user|>"],
        bos="", eos="<|endoftext|>",
    ),
    "phi3": ModelFamily(
        name="Phi-3 Instruct", family="phi3", template="phi",
        system_prefix="<|system|>\n", system_suffix="<|end|>\n",
        user_prefix="<|user|>\n", user_suffix="<|end|>\n<|assistant|>\n",
        assistant_suffix="<|end|>\n",
        stop_tokens=["<|end|>", "<|user|>", "<|endoftext|>"],
        bos="", eos="<|endoftext|>",
    ),
    "qwen": ModelFamily(
        name="Qwen ChatML", family="qwen", template="chatml",
        system_prefix="<|im_start|>system\n", system_suffix="<|im_end|>\n",
        user_prefix="<|im_start|>user\n", user_suffix="<|im_end|>\n<|im_start|>assistant\n",
        assistant_suffix="<|im_end|>\n",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        bos="", eos="<|endoftext|>",
    ),
    "gemma": ModelFamily(
        name="Gemma Instruct", family="gemma", template="gemma",
        user_prefix="<start_of_turn>user\n", user_suffix="<end_of_turn>\n<start_of_turn>model\n",
        assistant_suffix="<end_of_turn>\n",
        stop_tokens=["<end_of_turn>", "<eos>"],
        bos="<bos>", eos="<eos>",
    ),
    "codellama": ModelFamily(
        name="CodeLlama Instruct", family="codellama", template="llama2",
        system_prefix="[INST] <<SYS>>\n", system_suffix="\n<</SYS>>\n\n",
        user_prefix="", user_suffix=" [/INST]",
        assistant_suffix="</s><s>",
        stop_tokens=["</s>", "[INST]", "Source:"],
        bos="<s>", eos="</s>",
    ),
    "falcon": ModelFamily(
        name="Falcon", family="falcon", template="falcon",
        user_prefix="User: ", user_suffix="\nAssistant:",
        assistant_suffix="\n",
        stop_tokens=["User:", "\nUser:", "<|endoftext|>"],
        bos="", eos="<|endoftext|>",
    ),
    "vicuna": ModelFamily(
        name="Vicuna", family="vicuna", template="vicuna",
        system_prefix="", system_suffix="\n\n",
        user_prefix="USER: ", user_suffix="\nASSISTANT:",
        assistant_suffix="</s>",
        stop_tokens=["</s>", "USER:"],
        bos="<s>", eos="</s>",
    ),
    "openchat": ModelFamily(
        name="OpenChat", family="openchat", template="openchat",
        user_prefix="GPT4 Correct User: ", user_suffix="<|end_of_turn|>GPT4 Correct Assistant:",
        assistant_suffix="<|end_of_turn|>",
        stop_tokens=["<|end_of_turn|>"],
        bos="<s>", eos="</s>",
    ),
    "neural-chat": ModelFamily(
        name="Neural Chat", family="neural-chat", template="neural-chat",
        system_prefix="### System:\n", system_suffix="\n",
        user_prefix="### User:\n", user_suffix="\n### Assistant:\n",
        assistant_suffix="\n",
        stop_tokens=["### User:", "<|endoftext|>"],
        bos="", eos="<|endoftext|>",
    ),
    "starling": ModelFamily(
        name="Starling", family="starling", template="openchat",
        user_prefix="GPT4 Correct User: ", user_suffix="<|end_of_turn|>GPT4 Correct Assistant:",
        assistant_suffix="<|end_of_turn|>",
        stop_tokens=["<|end_of_turn|>"],
        bos="<s>", eos="</s>",
    ),
    "yi": ModelFamily(
        name="Yi Chat", family="yi", template="chatml",
        system_prefix="<|im_start|>system\n", system_suffix="<|im_end|>\n",
        user_prefix="<|im_start|>user\n", user_suffix="<|im_end|>\n<|im_start|>assistant\n",
        assistant_suffix="<|im_end|>\n",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        bos="", eos="<|endoftext|>",
    ),
    "command-r": ModelFamily(
        name="Command-R", family="command-r", template="command-r",
        system_prefix="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
        system_suffix="<|END_OF_TURN_TOKEN|>",
        user_prefix="<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
        user_suffix="<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        assistant_suffix="<|END_OF_TURN_TOKEN|>",
        stop_tokens=["<|END_OF_TURN_TOKEN|>"],
        bos="", eos="<|END_OF_TURN_TOKEN|>",
    ),
    "orca": ModelFamily(
        name="Orca / ChatML", family="orca", template="chatml",
        system_prefix="<|im_start|>system\n", system_suffix="<|im_end|>\n",
        user_prefix="<|im_start|>user\n", user_suffix="<|im_end|>\n<|im_start|>assistant\n",
        assistant_suffix="<|im_end|>\n",
        stop_tokens=["<|im_end|>"],
        bos="", eos="<|endoftext|>",
    ),
    "zephyr": ModelFamily(
        name="Zephyr", family="zephyr", template="zephyr",
        system_prefix="<|system|>\n", system_suffix="</s>\n",
        user_prefix="<|user|>\n", user_suffix="</s>\n<|assistant|>\n",
        assistant_suffix="</s>\n",
        stop_tokens=["</s>", "<|user|>"],
        bos="<s>", eos="</s>",
    ),
    "solar": ModelFamily(
        name="Solar Instruct", family="solar", template="mistral",
        user_prefix="### User:\n", user_suffix="\n\n### Assistant:\n",
        assistant_suffix="\n",
        stop_tokens=["### User:", "</s>"],
        bos="<s>", eos="</s>",
    ),
    "default": ModelFamily(
        name="Generic Instruct", family="default", template="default",
        user_prefix="### Instruction:\n", user_suffix="\n### Response:\n",
        assistant_suffix="\n",
        stop_tokens=["### Instruction:", "### Human:", "</s>"],
        bos="", eos="",
    ),
}


def detect_model_family(filename: str) -> ModelFamily:
    """
    Detect model family from GGUF filename and return the matching template.
    Uses prioritised pattern matching so more specific names win.
    """
    name = Path(filename).stem.lower()

    # Order matters — more specific patterns first
    patterns: List[Tuple[List[str], str]] = [
        # DeepSeek variants
        (["deepseek-r1"],                         "deepseek-r1"),
        (["deepseek-coder", "deepseek_coder"],     "deepseek-coder"),
        (["deepseek"],                             "deepseek"),
        # Mistral / Mixtral
        (["mixtral"],                              "mixtral"),
        (["mistral"],                              "mistral"),
        # LLaMA variants
        (["llama-3", "llama3", "llama_3", "meta-llama-3"], "llama3"),
        (["codellama", "code-llama", "code_llama"],"codellama"),
        (["llama-2", "llama2", "llama_2"],         "llama2"),
        (["llama"],                                "llama2"),      # fallback
        # Phi
        (["phi-3", "phi3"],                        "phi3"),
        (["phi"],                                  "phi"),
        # Qwen
        (["qwen"],                                 "qwen"),
        # Gemma
        (["gemma"],                                "gemma"),
        # Yi
        (["yi-"],                                  "yi"),
        # Command-R
        (["command-r", "command_r"],               "command-r"),
        # Orca / ChatML
        (["orca", "openorca"],                     "orca"),
        # Falcon
        (["falcon"],                               "falcon"),
        # Vicuna
        (["vicuna"],                               "vicuna"),
        # OpenChat
        (["openchat"],                             "openchat"),
        # Neural Chat
        (["neural-chat", "neural_chat"],           "neural-chat"),
        # Starling
        (["starling"],                             "starling"),
        # Zephyr
        (["zephyr"],                               "zephyr"),
        # Solar
        (["solar"],                                "solar"),
    ]

    for keywords, family_key in patterns:
        for kw in keywords:
            if kw in name:
                return FAMILY_TEMPLATES[family_key]

    return FAMILY_TEMPLATES["default"]


def detect_quant_type(filename: str) -> str:
    """Extract the quantization type from a GGUF filename."""
    m = QUANT_REGEX.search(filename)
    if m:
        return m.group(1).upper()
    # Try without leading dot
    stem = Path(filename).stem.upper()
    for pat in GGUF_QUANT_PATTERNS:
        if re.search(pat, stem, re.IGNORECASE):
            return re.search(pat, stem, re.IGNORECASE).group(0).upper()
    return "UNKNOWN"


def quant_info(quant: str) -> Tuple[str, str]:
    """Return (quality_label, color_hint) for a quant type."""
    q = quant.upper()
    if any(x in q for x in ["F32", "F16", "BF16"]):
        return "Full precision", "#34d399"
    if any(x in q for x in ["Q8", "Q6"]):
        return "Near-lossless", "#34d399"
    if any(x in q for x in ["Q5", "IQ4"]):
        return "High quality", "#a78bfa"
    if any(x in q for x in ["Q4", "IQ3"]):
        return "Balanced", "#fbbf24"
    if any(x in q for x in ["Q3", "IQ2"]):
        return "Compressed", "#f87171"
    if any(x in q for x in ["Q2", "IQ1"]):
        return "Very compressed", "#ef4444"
    return "Unknown", "#7e7a9a"


# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "bg0":  "#07070f", "bg1":  "#0d0d1c", "bg2":  "#13132a",
    "bg3":  "#1a1a35",
    "acc":  "#a78bfa", "acc2": "#c4b5fd",
    "usr":  "#12103a", "ast":  "#0b1a12",
    "txt":  "#ede8ff", "txt2": "#7e7a9a", "bdr":  "#252340",
    "ok":   "#34d399", "warn": "#fbbf24", "err":  "#f87171",
    "glow": "#7c3aed",
    "pipeline": "#22d3ee",
}

QSS = f"""
QMainWindow, QDialog {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 {C['bg0']}, stop:0.5 #0b0b18, stop:1 #0d0a1a);
    color:{C['txt']};
}}
QWidget {{
    background:transparent; color:{C['txt']};
    font-family:'Segoe UI','Inter',sans-serif; font-size:13px;
}}
QScrollArea {{ background:transparent; border:none; }}
QScrollBar:vertical {{
    background:rgba(20,18,40,0.4); width:6px; margin:0; border:none; border-radius:3px;
}}
QScrollBar::handle:vertical {{
    background:rgba(167,139,250,0.35); border-radius:3px; min-height:18px;
}}
QScrollBar::handle:vertical:hover {{ background:rgba(167,139,250,0.6); }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QScrollBar:horizontal {{
    background:rgba(20,18,40,0.4); height:6px; border:none; border-radius:3px;
}}
QScrollBar::handle:horizontal {{
    background:rgba(167,139,250,0.35); border-radius:3px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}
QTextEdit, QLineEdit {{
    background:rgba(19,19,42,0.85);
    color:{C['txt']};
    border:1px solid {C['bdr']};
    border-radius:10px;
    padding:6px 10px;
    selection-background-color:rgba(124,58,237,0.5);
}}
QTextEdit:focus, QLineEdit:focus {{
    border-color:{C['acc']};
    background:rgba(26,26,53,0.95);
}}
QLineEdit {{ padding:4px 10px; }}
QPushButton {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(37,35,64,0.9), stop:1 rgba(26,26,50,0.9));
    color:{C['txt']};
    border:1px solid {C['bdr']};
    border-radius:8px;
    padding:5px 14px;
    min-height:26px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(124,58,237,0.55), stop:1 rgba(109,40,217,0.45));
    color:#fff;
    border-color:rgba(167,139,250,0.7);
}}
QPushButton:pressed {{
    background:rgba(109,40,217,0.7);
    border-color:{C['acc2']};
}}
QPushButton:disabled {{ color:{C['txt2']}; background:rgba(19,19,42,0.5); border-color:{C['bdr']}; }}
QPushButton#btn_send {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(124,58,237,0.9), stop:1 rgba(109,40,217,0.8));
    color:#fff; border:1px solid rgba(167,139,250,0.5); font-weight:600; border-radius:8px;
}}
QPushButton#btn_send:hover {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(167,139,250,0.95), stop:1 rgba(124,58,237,0.9));
}}
QPushButton#btn_stop {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(248,113,113,0.85), stop:1 rgba(220,38,38,0.75));
    color:#fff; border:1px solid rgba(248,113,113,0.4); font-weight:600; border-radius:8px;
}}
QPushButton#btn_stop:hover {{ background:rgba(248,113,113,0.95); }}
QPushButton#btn_new  {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(52,211,153,0.8), stop:1 rgba(16,185,129,0.7));
    color:#07070f; border:none; font-weight:600; border-radius:8px;
}}
QPushButton#btn_new:hover {{ background:rgba(52,211,153,0.95); }}
QListWidget {{
    background:rgba(13,13,28,0.6); border:none; outline:none; border-radius:8px;
}}
QListWidget::item {{ padding:6px 10px; border-radius:6px; margin:1px 3px; }}
QListWidget::item:selected {{
    background:rgba(124,58,237,0.3); color:{C['acc2']};
    border:1px solid rgba(167,139,250,0.25);
}}
QListWidget::item:hover:!selected {{ background:rgba(37,35,64,0.6); }}
QTabWidget::pane {{
    border:1px solid {C['bdr']}; border-top:none;
    background:rgba(7,7,15,0.95); border-radius:0 0 10px 10px;
}}
QTabBar::tab {{
    background:rgba(13,13,28,0.7); color:{C['txt2']};
    padding:9px 24px; border:none;
    border-bottom:2px solid transparent;
    border-radius:8px 8px 0 0;
    margin-right:2px;
}}
QTabBar::tab:selected {{
    color:{C['txt']}; border-bottom:2px solid {C['acc']};
    background:rgba(19,19,42,0.95);
}}
QTabBar::tab:hover:!selected {{ color:{C['acc2']}; background:rgba(26,26,53,0.8); }}
QComboBox {{
    background:rgba(19,19,42,0.85); color:{C['txt']};
    border:1px solid {C['bdr']}; border-radius:8px; padding:4px 10px; min-height:26px;
}}
QComboBox:focus, QComboBox:hover {{ border-color:{C['acc']}; }}
QComboBox QAbstractItemView {{
    background:rgba(13,13,28,0.97); color:{C['txt']};
    selection-background-color:rgba(124,58,237,0.4);
    border:1px solid {C['bdr']}; border-radius:8px; padding:2px; outline:none;
}}
QComboBox::drop-down {{ border:none; width:20px; }}
QProgressBar {{
    background:rgba(37,35,64,0.5); border:1px solid {C['bdr']};
    border-radius:4px; height:8px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C['acc']}, stop:1 {C['acc2']});
    border-radius:4px;
}}
QStatusBar {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 rgba(13,13,28,0.98), stop:1 rgba(10,10,20,0.98));
    color:{C['txt2']}; border-top:1px solid {C['bdr']}; font-size:11px; padding:2px 0;
}}
QMenuBar {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 rgba(13,13,28,0.98), stop:1 rgba(10,10,20,0.98));
    color:{C['txt']}; border-bottom:1px solid {C['bdr']}; padding:2px 0;
}}
QMenuBar::item {{ padding:4px 12px; border-radius:6px; }}
QMenuBar::item:selected {{ background:rgba(124,58,237,0.3); color:{C['acc2']}; }}
QMenu {{
    background:rgba(13,13,28,0.97); color:{C['txt']};
    border:1px solid rgba(167,139,250,0.2);
    padding:6px; border-radius:10px;
}}
QMenu::item {{ padding:6px 20px; border-radius:6px; }}
QMenu::item:selected {{ background:rgba(124,58,237,0.35); color:{C['acc2']}; }}
QMenu::separator {{ height:1px; background:{C['bdr']}; margin:4px 8px; }}
QSplitter::handle {{ background:{C['bdr']}; }}
QSplitter::handle:horizontal {{ width:1px; }}
QLabel {{ background:transparent; color:{C['txt2']}; }}
QToolTip {{
    background:rgba(19,19,42,0.97); color:{C['txt']};
    border:1px solid rgba(167,139,250,0.3);
    padding:5px 10px; border-radius:8px; font-size:11px;
}}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color:{C['bdr']}; }}
QCheckBox {{
    color:{C['txt']}; spacing:6px;
}}
QCheckBox::indicator {{
    width:16px; height:16px;
    background:rgba(19,19,42,0.85);
    border:1px solid {C['bdr']}; border-radius:4px;
}}
QCheckBox::indicator:checked {{
    background:rgba(124,58,237,0.7);
    border-color:{C['acc']};
}}
QGroupBox {{
    border:1px solid {C['bdr']}; border-radius:8px;
    margin-top:8px; padding-top:8px;
    color:{C['txt2']}; font-size:11px;
}}
QGroupBox::title {{
    subcontrol-origin:margin; left:12px; padding:0 4px;
    color:{C['acc2']};
}}
"""
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
                    self._flash_copy(bid)
                event.accept()
                return
        super().mousePressEvent(event)

    def _flash_copy(self, bid: str):
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

# ═════════════════════════════ MODEL REGISTRY ═══════════════════════════════

@dataclass
class ModelConfig:
    path:           str
    role:           str   = "general"
    threads:        int   = DEFAULT_THREADS
    ctx:            int   = DEFAULT_CTX
    temperature:    float = 0.7
    top_p:          float = 0.9
    repeat_penalty: float = 1.1
    n_predict:      int   = DEFAULT_N_PRED
    # auto-detected, stored for display
    family:         str   = "default"

    def to_dict(self) -> Dict:
        return {
            "path": self.path, "role": self.role,
            "threads": self.threads, "ctx": self.ctx,
            "temperature": self.temperature, "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty, "n_predict": self.n_predict,
            "family": self.family,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def size_mb(self) -> float:
        p = Path(self.path)
        return round(p.stat().st_size / 1e6, 1) if p.exists() else 0.0

    @property
    def detected_family(self) -> ModelFamily:
        return detect_model_family(self.path)

    @property
    def quant_type(self) -> str:
        return detect_quant_type(self.path)


class ModelRegistry:
    def __init__(self):
        self._custom:  List[str]              = []
        self._configs: Dict[str, ModelConfig] = {}
        self._load()

    def _load(self):
        if CUSTOM_MODELS_FILE.exists():
            try:
                self._custom = json.loads(CUSTOM_MODELS_FILE.read_text())
            except Exception:
                self._custom = []
        if MODEL_CONFIGS_FILE.exists():
            try:
                raw = json.loads(MODEL_CONFIGS_FILE.read_text())
                self._configs = {p: ModelConfig.from_dict(d) for p, d in raw.items()}
            except Exception:
                self._configs = {}

    def save(self):
        CUSTOM_MODELS_FILE.write_text(json.dumps(self._custom, indent=2))
        MODEL_CONFIGS_FILE.write_text(
            json.dumps({p: c.to_dict() for p, c in self._configs.items()}, indent=2))

    def add(self, path: str):
        if path not in self._custom:
            self._custom.append(path)
        if path not in self._configs:
            fam = detect_model_family(path)
            self._configs[path] = ModelConfig(path=path, family=fam.family)
        self.save()

    def remove(self, path: str):
        self._custom  = [p for p in self._custom if p != path]
        self._configs.pop(path, None)
        self.save()

    def get_config(self, path: str) -> ModelConfig:
        if path not in self._configs:
            fam = detect_model_family(path)
            self._configs[path] = ModelConfig(path=path, family=fam.family)
        return self._configs[path]

    def set_config(self, path: str, cfg: ModelConfig):
        self._configs[path] = cfg
        self.save()

    def all_models(self) -> List[Dict]:
        seen:   set        = set()
        models: List[Dict] = []
        if MODELS_DIR.exists():
            for f in sorted(MODELS_DIR.glob("*.gguf")):
                seen.add(str(f))
                cfg = self.get_config(str(f))
                fam = detect_model_family(str(f))
                qt  = detect_quant_type(str(f))
                models.append({
                    "path": str(f), "name": f.name,
                    "size_mb": round(f.stat().st_size / 1e6, 1),
                    "source": "auto", "role": cfg.role,
                    "family": fam.name, "quant": qt,
                })
        for p in self._custom:
            fp = Path(p)
            if fp.exists() and p not in seen:
                seen.add(p)
                cfg = self.get_config(p)
                fam = detect_model_family(p)
                qt  = detect_quant_type(p)
                models.append({
                    "path": p, "name": fp.name,
                    "size_mb": round(fp.stat().st_size / 1e6, 1),
                    "source": "custom", "role": cfg.role,
                    "family": fam.name, "quant": qt,
                })
        return models


MODEL_REGISTRY = ModelRegistry()

# ═════════════════════════════ PARALLEL LOADING PREFS ═══════════════════════

@dataclass
class ParallelPrefs:
    enabled:           bool = False
    auto_load_roles:   List[str] = field(default_factory=list)
    pipeline_mode:     bool = False   # reasoning → coding chain
    warned:            bool = False

    def save(self):
        PARALLEL_PREFS_FILE.write_text(json.dumps({
            "enabled": self.enabled,
            "auto_load_roles": self.auto_load_roles,
            "pipeline_mode": self.pipeline_mode,
            "warned": self.warned,
        }, indent=2))

    @classmethod
    def load(cls) -> "ParallelPrefs":
        if PARALLEL_PREFS_FILE.exists():
            try:
                d = json.loads(PARALLEL_PREFS_FILE.read_text())
                return cls(**{k: v for k, v in d.items()
                               if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls()


PARALLEL_PREFS = ParallelPrefs.load()

# ═════════════════════════════ DATA MODELS ══════════════════════════════════

@dataclass
class Message:
    role:      str
    content:   str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))


@dataclass
class Session:
    id:       str
    title:    str
    created:  str
    messages: List[Message] = field(default_factory=list)

    @classmethod
    def new(cls, title: str = "New Chat") -> "Session":
        now = datetime.now()
        return cls(id=now.strftime("%Y-%m-%d_%H%M%S"),
                   title=title,
                   created=now.strftime("%Y-%m-%d"))

    @classmethod
    def load(cls, path: Path) -> "Session":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            id=d["id"], title=d["title"], created=d["created"],
            messages=[Message(**m) for m in d.get("messages", [])]
        )

    def save(self):
        path = SESSIONS_DIR / f"{self.id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "id": self.id, "title": self.title, "created": self.created,
                "messages": [{"role": m.role, "content": m.content,
                               "timestamp": m.timestamp} for m in self.messages]
            }, f, indent=2, ensure_ascii=False)

    def add_message(self, role: str, content: str) -> Message:
        m = Message(role=role, content=content)
        self.messages.append(m)
        return m

    def build_prompt(self, model_path: str = "", max_chars: int = 8000) -> str:
        """Build prompt using the correct template for the detected model family."""
        fam = detect_model_family(model_path) if model_path else FAMILY_TEMPLATES["mistral"]
        return self._build_with_template(fam, max_chars)

    def _build_with_template(self, fam: ModelFamily, max_chars: int) -> str:
        recent: List[Message] = []
        used = 0
        for m in reversed(self.messages):
            used += len(m.content)
            if used > max_chars:
                break
            recent.insert(0, m)

        parts: List[str] = []
        i = 0
        first_turn = True
        while i < len(recent):
            m = recent[i]
            if m.role == "user":
                reply = (recent[i + 1].content
                         if i + 1 < len(recent) and recent[i + 1].role == "assistant"
                         else "")
                # BOS only on the very first turn to avoid token duplication
                bos = fam.bos if first_turn else ""
                first_turn = False
                user_part = fam.user_prefix + m.content + fam.user_suffix
                if reply:
                    parts.append(bos + user_part +
                                 fam.assistant_prefix + reply + fam.assistant_suffix)
                    i += 2
                else:
                    parts.append(bos + user_part + fam.assistant_prefix)
                    i += 1
            else:
                i += 1

        return "".join(parts)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}\n\n*{self.created}*\n\n---\n\n"]
        for m in self.messages:
            icon = "**You**" if m.role == "user" else "**Assistant**"
            lines.append(f"{icon} · {m.timestamp}\n\n{m.content}\n\n---\n\n")
        return "".join(lines)

    def to_txt(self) -> str:
        lines = [f"{self.title}\n{'='*40}\n{self.created}\n\n"]
        for m in self.messages:
            lines.append(f"[{m.role.upper()}] {m.timestamp}\n{m.content}\n\n")
        return "".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "id": self.id, "title": self.title, "created": self.created,
            "messages": [{"role": m.role, "content": m.content,
                           "timestamp": m.timestamp} for m in self.messages]
        }, indent=2, ensure_ascii=False)

    @property
    def approx_tokens(self) -> int:
        return sum(len(m.content) for m in self.messages) // 4


# ═════════════════════════════ ENGINE LAYER ═════════════════════════════════

def _free_port(lo: int = 8600, hi: int = 8700) -> int:
    for p in range(lo, hi):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return lo


class ServerStreamWorker(QThread):
    token  = pyqtSignal(str)
    done   = pyqtSignal(float)
    err    = pyqtSignal(str)

    def __init__(self, port: int, prompt: str, n_predict: int = DEFAULT_N_PRED,
                 stop_tokens: List[str] = None, temperature: float = 0.7,
                 top_p: float = 0.9, repeat_penalty: float = 1.1):
        super().__init__()
        self.port          = port
        self.prompt        = prompt
        self.n_predict     = n_predict
        self.stop_tokens   = stop_tokens or ["</s>", "[INST]", "### Human:"]
        self.temperature   = temperature
        self.top_p         = top_p
        self.repeat_penalty = repeat_penalty
        self._abort        = False

    def run(self):
        import http.client
        t0 = time.time(); n = 0
        try:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=900)
            body = json.dumps({
                "prompt":         self.prompt,
                "n_predict":      self.n_predict,
                "stream":         True,
                "temperature":    self.temperature,
                "top_p":          self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "stop":           self.stop_tokens,
            })
            conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
            r = conn.getresponse()
            if r.status != 200:
                self.err.emit(f"HTTP {r.status}"); return

            buf = b""
            while not self._abort:
                b = r.read(1)
                if not b: break
                buf += b
                if b == b"\n":
                    line = buf.decode("utf-8", errors="replace").strip()
                    buf  = b""
                    if line.startswith("data: "):
                        try:
                            d = json.loads(line[6:])
                            if d.get("stop"): break
                            c = d.get("content", "")
                            if c:
                                n += 1; self.token.emit(c)
                        except json.JSONDecodeError:
                            pass

            elapsed = time.time() - t0
            self.done.emit(n / elapsed if elapsed > 0 else 0.0)
        except Exception as e:
            self.err.emit(str(e))

    def abort(self):
        self._abort = True


class CliStreamWorker(QThread):
    token  = pyqtSignal(str)
    done   = pyqtSignal(float)
    err    = pyqtSignal(str)

    def __init__(self, cmd: list):
        super().__init__()
        self.cmd    = cmd
        self._abort = False
        self.proc:  Optional[subprocess.Popen] = None

    def run(self):
        t0 = time.time(); n = 0
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                bufsize=0
            )
            while not self._abort:
                b = self.proc.stdout.read(1)
                if not b: break
                c = b.decode("utf-8", errors="replace")
                n += 1; self.token.emit(c)
            self.proc.wait(timeout=5)
            elapsed = time.time() - t0
            self.done.emit(n / elapsed if elapsed > 0 else 0.0)
        except Exception as e:
            self.err.emit(str(e))

    def abort(self):
        self._abort = True
        if self.proc:
            try: self.proc.terminate()
            except OSError: pass


# ── Pipeline Worker: Reasoning → Coding chain ─────────────────────────────────

class PipelineWorker(QThread):
    """
    Multi-engine structural insight → coding pipeline.

    Stage 1 (parallel):  Every active non-coding engine is asked to give a
                         *structural-level insight* about the user's coding
                         request — no code, just intent, architecture and design.
                         Each engine runs sequentially (llama.cpp single-process
                         constraint) but their outputs are collected and labelled.

    Stage 2 (coding):    The coding engine receives ALL structural insights from
                         stage 1 as rich context and generates the final code.
    """

    # per-engine insight signals
    insight_started  = pyqtSignal(int, str)   # (engine_idx, engine_label)
    insight_token    = pyqtSignal(int, str)   # (engine_idx, token)
    insight_done     = pyqtSignal(int, str)   # (engine_idx, full_text)

    # coding stage signals
    coding_token     = pyqtSignal(str)
    coding_done      = pyqtSignal(float)      # tok/s

    err              = pyqtSignal(str)
    stage_changed    = pyqtSignal(str)        # "insights" | "coding"

    # keep old signal names alive so existing connections don't break
    reasoning_token  = pyqtSignal(str)
    reasoning_done   = pyqtSignal(str)

    INSIGHT_PROMPT_TEMPLATE = (
        "You are a senior software architect reviewing a coding request.\n"
        "Your task: provide STRUCTURAL INSIGHT ONLY — no code, no pseudocode.\n\n"
        "Describe:\n"
        "1. High-level purpose and user intent\n"
        "2. Recommended architecture / design pattern (e.g. class hierarchy, "
        "pipeline, event-driven, functional)\n"
        "3. Key components / modules and their responsibilities\n"
        "4. Data flow between components\n"
        "5. Important algorithms or data structures to use\n"
        "6. Critical edge cases and error-handling considerations\n"
        "7. Suggested external libraries / imports\n\n"
        "Be thorough but concise. Structure your answer with short headers. "
        "Do NOT write any actual code.\n\n"
        "Coding request: {prompt}"
    )

    def __init__(self,
                 insight_engines,
                 coding_eng,
                 prompt: str,
                 n_predict_insight: int = 512,
                 n_predict_code: int = 1024):
        """
        insight_engines : list of (label, engine) for structural-insight stage
        coding_eng      : engine that will generate the final code
        """
        super().__init__()
        self.insight_engines    = insight_engines   # [(label, eng), ...]
        self.coding_eng         = coding_eng
        self.prompt             = prompt
        self.n_predict_insight  = n_predict_insight
        self.n_predict_code     = n_predict_code
        self._abort             = False
        self._insights          = []  # [(label, text), ...]

    def abort(self):
        self._abort = True

    def run(self):
        # Stage 1: structural insights from all non-coding engines
        self.stage_changed.emit("insights")

        for idx, (label, eng) in enumerate(self.insight_engines):
            if self._abort:
                self.err.emit("Aborted before all insights completed")
                return

            self.insight_started.emit(idx, label)

            fam = detect_model_family(getattr(eng, "model_path", ""))
            insight_prompt = (
                fam.bos +
                fam.user_prefix +
                self.INSIGHT_PROMPT_TEMPLATE.format(prompt=self.prompt) +
                fam.user_suffix +
                fam.assistant_prefix
            )

            text = self._infer_blocking(
                eng, insight_prompt,
                self.n_predict_insight,
                token_cb=lambda t, i=idx: self.insight_token.emit(i, t)
            )

            if self._abort:
                return
            if text is None:
                self.err.emit(f"Insight stage failed for engine: {label}")
                return

            cleaned = text.strip()
            self._insights.append((label, cleaned))
            self.insight_done.emit(idx, cleaned)

            # bridge old reasoning_done signal for single-engine compat
            if idx == 0:
                self.reasoning_done.emit(cleaned)

        if not self._insights:
            self.err.emit("No structural insights were produced")
            return

        # Stage 2: coding engine receives all insights
        self.stage_changed.emit("coding")

        insights_block = ""
        for label, text in self._insights:
            bar = "=" * max(4, len(label) + 4)
            insights_block += (
                f"[ {label} ]\n"
                f"{text}\n"
                f"{bar}\n\n"
            )

        code_fam = detect_model_family(getattr(self.coding_eng, "model_path", ""))
        sep = "=" * 60
        code_prompt = (
            code_fam.bos +
            code_fam.user_prefix +
            f"You are an expert code generation assistant.\n\n"
            f"{sep}\n"
            f"STRUCTURAL INSIGHTS FROM ANALYSIS MODELS\n"
            f"{sep}\n"
            f"{insights_block}"
            f"{sep}\n"
            f"ORIGINAL REQUEST\n"
            f"{sep}\n"
            f"{self.prompt}\n\n"
            f"Using the structural insights above as your blueprint, generate "
            f"complete, working, well-commented code. Follow the architecture "
            f"and design patterns recommended in the insights. Include all "
            f"necessary imports. Add docstrings and inline comments where useful. "
            f"Handle the edge cases mentioned in the insights." +
            code_fam.user_suffix +
            code_fam.assistant_prefix
        )

        t0 = time.time()
        n_tokens = [0]
        self._infer_blocking(
            self.coding_eng, code_prompt,
            self.n_predict_code,
            token_cb=lambda t: (
                self.coding_token.emit(t),
                n_tokens.__setitem__(0, n_tokens[0] + 1)
            )
        )
        elapsed = time.time() - t0
        tps = n_tokens[0] / elapsed if elapsed > 0 else 0.0
        self.coding_done.emit(tps)

    def _infer_blocking(self, eng, prompt: str,
                        n_predict: int, token_cb=None):
        """Blocking inference that calls token_cb for each token."""
        if eng.mode == "server":
            return self._infer_server(eng, prompt, n_predict, token_cb)
        return self._infer_cli(eng, prompt, n_predict, token_cb)

    def _infer_server(self, eng, prompt: str,
                      n_predict: int, token_cb=None):
        import http.client
        fam = detect_model_family(getattr(eng, "model_path", ""))
        try:
            conn = http.client.HTTPConnection("127.0.0.1", eng.server_port, timeout=600)
            body = json.dumps({
                "prompt":         prompt,
                "n_predict":      n_predict,
                "stream":         True,
                "temperature":    0.7,
                "top_p":          0.9,
                "repeat_penalty": 1.1,
                "stop":           fam.stop_tokens,
            })
            conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
            r = conn.getresponse()
            if r.status != 200:
                return None
            result = []
            buf = b""
            while not self._abort:
                b = r.read(1)
                if not b:
                    break
                buf += b
                if b == b"\n":
                    line = buf.decode("utf-8", errors="replace").strip()
                    buf  = b""
                    if line.startswith("data: "):
                        try:
                            d = json.loads(line[6:])
                            if d.get("stop"):
                                break
                            c = d.get("content", "")
                            if c:
                                result.append(c)
                                if token_cb:
                                    token_cb(c)
                        except json.JSONDecodeError:
                            pass
            return "".join(result)
        except Exception:
            return None

    def _infer_cli(self, eng, prompt: str,
                   n_predict: int, token_cb=None):
        try:
            proc = subprocess.Popen(
                [LLAMA_CLI, "-m", eng.model_path,
                 "-t", str(DEFAULT_THREADS),
                 "--ctx-size", str(getattr(eng, "ctx_value", DEFAULT_CTX)),
                 "-n", str(n_predict),
                 "--no-display-prompt", "--no-escape",
                 "-p", prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                bufsize=0
            )
            result = []
            while not self._abort:
                b = proc.stdout.read(1)
                if not b:
                    break
                c = b.decode("utf-8", errors="replace")
                result.append(c)
                if token_cb:
                    token_cb(c)
            proc.terminate()
            return "".join(result)
        except Exception:
            return None


# ── Paused job persistence ────────────────────────────────────────────────────

def _save_paused_job(job_id: str, state: dict):
    p = PAUSED_JOBS_DIR / f"{job_id}.json"
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

def _load_paused_job(job_id: str) -> Optional[dict]:
    p = PAUSED_JOBS_DIR / f"{job_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None

def _delete_paused_job(job_id: str):
    p = PAUSED_JOBS_DIR / f"{job_id}.json"
    if p.exists():
        try: p.unlink()
        except Exception: pass

def list_paused_jobs() -> List[dict]:
    jobs = []
    for p in PAUSED_JOBS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            jobs.append(d)
        except Exception:
            pass
    return jobs


class ChunkedSummaryWorker(QThread):
    section_done  = pyqtSignal(int, int, str, str)
    final_done    = pyqtSignal(str)
    progress      = pyqtSignal(str)
    err           = pyqtSignal(str)
    pause_suggest = pyqtSignal(str)   # job_id — UI can offer pause

    def __init__(self, engine: "LlamaEngine", text: str, filename: str = "",
                 engine2: "LlamaEngine" = None,
                 resume_job_id: str = "",
                 session_id: str = ""):
        super().__init__()
        self.engine        = engine
        self.engine2       = engine2
        self.text          = text
        self.filename      = filename
        self.resume_job_id = resume_job_id
        self.session_id    = session_id
        self._abort        = False
        self._pause        = False
        self.job_id        = resume_job_id or f"sum_{_simple_hash(filename + text[:32])}_{int(time.time())}"

    def abort(self):
        self._abort = True

    def request_pause(self):
        self._pause = True

    def run(self):
        cfg = APP_CONFIG
        CHUNK_CHARS  = int(cfg["summary_chunk_chars"])
        CTX_CARRY    = int(cfg["summary_ctx_carry"])
        N_PRED_SECT  = int(cfg["summary_n_pred_sect"])
        N_PRED_FINAL = int(cfg["summary_n_pred_final"])
        PAUSE_THRESH = int(cfg["pause_after_chunks"])

        chunks = self._split(self.text, CHUNK_CHARS)
        total  = len(chunks)

        # Resume state
        start_idx = 0
        section_summaries: List[str] = []
        running_ctx = ""

        if self.resume_job_id:
            state = _load_paused_job(self.resume_job_id)
            if state:
                start_idx         = state.get("next_chunk", 0)
                section_summaries = state.get("summaries", [])
                running_ctx       = state.get("running_ctx", "")
                self.progress.emit(
                    f"Resuming from chunk {start_idx + 1} / {total}…")

        fam = detect_model_family(getattr(self.engine, "model_path", ""))

        for i in range(start_idx, total):
            if self._abort:
                self.err.emit("Aborted by user."); return

            if self._pause:
                # Save state to disk and signal UI
                state = {
                    "job_id":      self.job_id,
                    "filename":    self.filename,
                    "session_id":  self.session_id,
                    "total":       total,
                    "next_chunk":  i,
                    "summaries":   section_summaries,
                    "running_ctx": running_ctx,
                    "raw_text":    self.text,
                    "model_path":  getattr(self.engine, "model_path", ""),
                    "paused_at":   datetime.now().isoformat(),
                }
                _save_paused_job(self.job_id, state)
                self.progress.emit(
                    f"⏸  Paused after chunk {i} / {total}. State saved to disk.")
                self.err.emit(f"__PAUSED__:{self.job_id}")
                return

            # Auto-suggest pause after PAUSE_THRESH chunks if many remain
            if i > 0 and i == PAUSE_THRESH and (total - i) > PAUSE_THRESH:
                self.pause_suggest.emit(self.job_id)

            chunk = chunks[i]
            self.progress.emit(f"Summarising section {i+1} / {total}…")
            ctx_block = (f"Running context from previous sections:\n{running_ctx}\n\n"
                         if running_ctx else "")
            prompt = (
                fam.bos + fam.user_prefix +
                f"You are summarising a long document section by section.\n"
                f"File: '{self.filename}'  |  Section {i+1} of {total}\n\n"
                f"{ctx_block}"
                f"Current section:\n{chunk}\n\n"
                f"Summarise this section clearly. Retain all key facts, named entities, "
                f"numbers, arguments, and logical connections." +
                fam.user_suffix + fam.assistant_prefix
            )
            summary = self._infer(prompt, N_PRED_SECT)
            if summary is None:
                self.err.emit(f"Inference failed on section {i+1}"); return
            summary = summary.strip()
            section_summaries.append(f"[Section {i+1} of {total}]\n{summary}")
            running_ctx = summary[-CTX_CARRY:]
            self.section_done.emit(i + 1, total, chunk, summary)

            # Incremental disk save every 3 chunks
            if (i + 1) % 3 == 0:
                _save_paused_job(self.job_id + "_autosave", {
                    "job_id":      self.job_id,
                    "filename":    self.filename,
                    "session_id":  self.session_id,
                    "total":       total,
                    "next_chunk":  i + 1,
                    "summaries":   section_summaries,
                    "running_ctx": running_ctx,
                    "raw_text":    self.text,
                    "model_path":  getattr(self.engine, "model_path", ""),
                    "paused_at":   datetime.now().isoformat(),
                })

        if self._abort: return

        fin_eng   = self.engine2 if (self.engine2 and self.engine2.is_loaded) else self.engine
        fin_label = "reasoning model" if fin_eng is self.engine2 else "primary model"
        self.progress.emit(f"Running final consolidation pass ({fin_label})…")

        fin_fam   = detect_model_family(getattr(fin_eng, "model_path", ""))
        all_sects = "\n\n".join(section_summaries)
        final_prompt = (
            fin_fam.bos + fin_fam.user_prefix +
            f"You have finished reading all {total} sections of '{self.filename}'.\n\n"
            f"Section-by-section summaries:\n{all_sects}\n\n"
            f"Now write a single, well-structured, coherent final summary." +
            fin_fam.user_suffix + fin_fam.assistant_prefix
        )
        final = self._infer_with(fin_eng, final_prompt, N_PRED_FINAL)
        if final is None:
            self.err.emit("Final consolidation pass failed"); return

        # Cleanup autosave
        _delete_paused_job(self.job_id + "_autosave")
        _delete_paused_job(self.job_id)
        self.final_done.emit(final.strip())

    def _split(self, text: str, chunk_chars: int) -> List[str]:
        chunks: List[str] = []
        while text:
            if len(text) <= chunk_chars:
                chunks.append(text.strip()); break
            cut = text.rfind("\n\n", 0, chunk_chars)
            if cut < 200:
                cut = text.rfind("\n",  0, chunk_chars)
            if cut < 200:
                cut = chunk_chars
            chunks.append(text[:cut].strip())
            text = text[cut:].lstrip()
        return [c for c in chunks if c]

    def _infer(self, prompt: str, n_predict: int) -> Optional[str]:
        return self._infer_with(self.engine, prompt, n_predict)

    def _infer_with(self, eng: "LlamaEngine", prompt: str,
                    n_predict: int) -> Optional[str]:
        if eng.mode == "server":
            return self._infer_server(prompt, n_predict, eng.server_port)
        return self._infer_cli(prompt, n_predict, eng.model_path,
                               getattr(eng, "ctx_value", DEFAULT_CTX()))

    def _infer_server(self, prompt: str, n_predict: int, port: int = 0) -> Optional[str]:
        import http.client
        fam = detect_model_family(getattr(self.engine, "model_path", ""))
        if not port: port = self.engine.server_port
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
            body = json.dumps({
                "prompt": prompt, "n_predict": n_predict,
                "stream": False, "temperature": 0.3, "top_p": 0.9,
                "repeat_penalty": 1.15, "stop": fam.stop_tokens,
            })
            conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
            r = conn.getresponse()
            if r.status != 200: return None
            d = json.loads(r.read().decode("utf-8", errors="replace"))
            return d.get("content", "")
        except Exception:
            return None

    def _infer_cli(self, prompt: str, n_predict: int,
                   model_path: str = "", ctx: int = 0) -> Optional[str]:
        if not ctx: ctx = DEFAULT_CTX()
        if not model_path: model_path = self.engine.model_path
        try:
            result = subprocess.run(
                [LLAMA_CLI, "-m", model_path, "-t", str(DEFAULT_THREADS()),
                 "--ctx-size", str(ctx), "-n", str(n_predict),
                 "--no-display-prompt", "--no-escape",
                 "--temp", "0.3", "--repeat-penalty", "1.15", "-p", prompt],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, timeout=300,
            )
            return result.stdout.decode("utf-8", errors="replace")
        except Exception:
            return None

class ModelLoaderThread(QThread):
    finished = pyqtSignal(bool, str)
    log      = pyqtSignal(str, str)

    def __init__(self, engine: "LlamaEngine", model_path: str, ctx: int):
        super().__init__()
        self.engine     = engine
        self.model_path = model_path
        self.ctx        = ctx

    def run(self):
        ok = self.engine.load(self.model_path, ctx=self.ctx)
        self.finished.emit(ok, self.engine.status_text)


class LlamaEngine:
    def __init__(self):
        self.server_proc: Optional[subprocess.Popen] = None
        self.server_port: int = 0
        self.model_path:  str = ""
        self.mode = "unloaded"
        self._log = lambda m: None

    def load(self, model_path: str,
             threads: int = DEFAULT_THREADS,
             ctx:     int = DEFAULT_CTX,
             log_cb=None) -> bool:
        self.model_path = model_path
        self._log = log_cb or (lambda m: None)
        self.ctx_value = ctx
        if not Path(model_path).exists():
            self._log(f"[ERROR] Model not found: {model_path}")
            return False

        if Path(LLAMA_SERVER).exists():
            ok = self._start_server(model_path, threads, ctx)
            if ok: return True
            self._log("[WARN] Falling back to llama-cli mode")

        if Path(LLAMA_CLI).exists():
            self._log("[INFO] Using llama-cli (per-prompt) mode")
            self.mode = "cli"
            return True

        self._log("[ERROR] Neither llama-server nor llama-cli found")
        return False

    def _check_existing_server(self, port: int) -> bool:
        import http.client
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            conn.request("GET", "/")
            res = conn.getresponse()
            return res.status in (200, 404)
        except Exception:
            return False

    def create_worker(self, prompt: str, n_predict: int = DEFAULT_N_PRED,
                      model_path: str = "") -> QThread:
        fam  = detect_model_family(model_path or self.model_path)
        cfg  = MODEL_REGISTRY.get_config(self.model_path)
        if self.mode == "server":
            return ServerStreamWorker(
                self.server_port, prompt, n_predict,
                stop_tokens=fam.stop_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                repeat_penalty=cfg.repeat_penalty,
            )
        cmd = [
            LLAMA_CLI, "-m", self.model_path,
            "-t", str(DEFAULT_THREADS), "--ctx-size", str(self.ctx_value),
            "-n", str(n_predict), "--no-display-prompt", "--no-escape",
            "-p", prompt,
        ]
        return CliStreamWorker(cmd)

    def ensure_server(self, log_cb=None) -> bool:
        """
        If already in server mode, return True immediately.
        If in CLI mode, try to start a fresh server on a new port.
        If server start fails, return False — caller should abort pipeline.
        """
        if self.mode == "server":
            return True
        if not self.model_path or not Path(self.model_path).exists():
            return False
        if log_cb:
            self._log = log_cb
        self._log(f"[INFO] ensure_server: attempting server start for {Path(self.model_path).name}")
        ok = self._start_server(self.model_path, DEFAULT_THREADS, self.ctx_value)
        if ok:
            self._log(f"[INFO] ensure_server: server started on port {self.server_port}")
        else:
            self._log(f"[WARN] ensure_server: server start failed for {Path(self.model_path).name}")
        return ok

    def ensure_server_or_reload(self, log_cb=None) -> bool:
        """
        Like ensure_server but if server start fails, kills any existing
        server proc and retries once on a fresh port.
        """
        if self.mode == "server":
            return True
        # Kill existing proc if any
        if self.server_proc:
            try:
                self.server_proc.terminate()
                self.server_proc.wait(timeout=5)
            except Exception:
                pass
            self.server_proc = None
        self.server_port = _free_port()
        return self.ensure_server(log_cb=log_cb)

    def shutdown(self):
        if self.server_proc:
            try:
                self.server_proc.terminate()
                self.server_proc.wait(timeout=5)
            except Exception:
                pass
        self.server_proc = None
        self.mode = "unloaded"

    @property
    def is_loaded(self) -> bool:
        return self.mode != "unloaded"

    @property
    def status_text(self) -> str:
        if self.mode == "server": return f"🟢 Server  :{self.server_port}"
        if self.mode == "cli":    return "🟡 CLI Mode"
        return "⚪ Not Loaded"

    def _start_server(self, model_path: str, threads: int, ctx: int) -> bool:
        import http.client
        # Only reuse an existing server if it's for the exact same model
        for test_port in range(8600, 8700):
            if self._check_existing_server(test_port):
                # Verify it's serving our model by checking /props endpoint
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", test_port, timeout=2)
                    conn.request("GET", "/props")
                    res = conn.getresponse()
                    props = json.loads(res.read().decode("utf-8", errors="replace"))
                    running_model = props.get("model_path", props.get("default_generation_settings", {}).get("model", ""))
                    if Path(running_model).resolve() == Path(model_path).resolve():
                        self.server_port = test_port
                        self.mode = "server"
                        self._log(f"[INFO] Reusing existing server for same model on port {test_port}")
                        return True
                    else:
                        self._log(f"[INFO] Port {test_port} has different model — skipping reuse")
                except Exception:
                    pass

        self.server_port = _free_port()
        cmd = [LLAMA_SERVER, "-m", model_path, "-t", str(threads),
               "--ctx-size", str(ctx), "--port", str(self.server_port),
               "--host", "127.0.0.1"]
        self._log(f"[INFO] Starting llama-server on port {self.server_port}…")
        try:
            self.server_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL)
        except Exception as e:
            self._log(f"[ERROR] Could not start server: {e}")
            self.server_proc = None
            return False

        if self.server_proc is None:
            return False

        for _ in range(90):
            time.sleep(0.5)
            if self.server_proc is None: return False
            if self.server_proc.poll() is not None:
                self._log(f"[ERROR] llama-server exited unexpectedly")
                self.server_proc = None
                return False
            if self._check_existing_server(self.server_port):
                self._log(f"[INFO] llama-server ready on port {self.server_port}")
                self.mode = "server"
                return True

        self._log("[ERROR] Server did not respond within timeout")
        try: self.server_proc.terminate()
        except Exception: pass
        self.server_proc = None
        return False

# ═════════════════════════════ REFERENCE ENGINE ══════════════════════════════

import threading

def _ram_free_mb() -> float:
    if HAS_PSUTIL:
        return psutil.virtual_memory().available / (1024 * 1024)
    return 9999.0

def _cpu_count() -> int:
    try:
        import multiprocessing
        n = multiprocessing.cpu_count()
        return n if n and n > 0 else 4
    except Exception:
        return 4


def _simple_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:12]


@dataclass
class RefChunk:
    idx:     int
    text:    str
    score:   float = 0.0


class SmartReference:
    """
    A single uploaded reference file with intelligent lazy chunk indexing.
    Supports RAM-limited streaming: cold chunks live on disk, hot chunks in RAM.
    """

    def __init__(self, ref_id: str, name: str, ftype: str, raw_text: str):
        self.ref_id   = ref_id
        self.name     = name
        self.ftype    = ftype           # "pdf" | "python" | "text"
        self._raw     = raw_text
        self._chunks: List[RefChunk]    = []
        self._hot:    Dict[int, str]    = {}   # idx → text (RAM cache)
        self._disk_path = REF_CACHE_DIR / f"{ref_id}.pkl"
        self._indexed   = False
        self._lock      = threading.Lock()
        self._build_index()

    def _build_index(self):
        """Split raw text into overlapping chunks and save cold index to disk."""
        text   = self._raw
        step   = CHUNK_INDEX_SIZE()
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i: i + step + 80]
            chunks.append(RefChunk(idx=len(chunks), text=chunk))
            i += step
        self._chunks = chunks

        if _ram_free_mb() > RAM_WATCHDOG_MB():
            for c in chunks[:MAX_RAM_CHUNKS()]:
                self._hot[c.idx] = c.text
        else:
            self._spill_to_disk()

        self._indexed = True

    def _spill_to_disk(self):
        try:
            with open(self._disk_path, "wb") as f:
                pickle.dump([c.text for c in self._chunks], f)
            self._hot.clear()
        except Exception:
            pass

    def _load_chunk_from_disk(self, idx: int) -> str:
        try:
            if self._disk_path.exists():
                with open(self._disk_path, "rb") as f:
                    all_texts = pickle.load(f)
                if idx < len(all_texts):
                    return all_texts[idx]
        except Exception:
            pass
        return ""

    def _get_chunk_text(self, idx: int) -> str:
        if idx in self._hot:
            return self._hot[idx]
        text = self._load_chunk_from_disk(idx)
        if _ram_free_mb() > RAM_WATCHDOG_MB() + 200:
            self._hot[idx] = text
        return text

    def query(self, query_text: str, top_k: int = 6) -> str:
        """Return the top_k most relevant chunks for the query (keyword scoring)."""
        if not self._chunks:
            return ""
        words = set(re.findall(r'\w+', query_text.lower()))
        if not words:
            return self._get_chunk_text(0)[:1200]

        scored: List[Tuple[float, int]] = []
        for c in self._chunks:
            text  = self._get_chunk_text(c.idx).lower()
            score = sum(1 for w in words if w in text)
            # Boost for exact phrase
            if query_text.lower()[:30] in text:
                score += 5
            scored.append((score, c.idx))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, idx in scored[:top_k]:
            if score == 0 and results:
                break
            results.append(self._get_chunk_text(idx))

        return "\n\n---\n\n".join(results)[:3000]

    def full_text_preview(self) -> str:
        return self._raw[:400] + ("…" if len(self._raw) > 400 else "")

    def cleanup(self):
        self._hot.clear()
        if self._disk_path.exists():
            try: self._disk_path.unlink()
            except Exception: pass

# ═══════════════════════════════════════════════════════════════════════════
# ScriptSmartReference — RAM/disk-aware with parsed structure
# ═══════════════════════════════════════════════════════════════════════════

class ScriptSmartReference:
    """
    Extends the reference concept for source code files.
    Stores:
      • ParsedScript  — structured AST/regex parse result (always in RAM)
      • Raw text chunks on disk (same model as SmartReference)
    """

    def __init__(self, ref_id: str, name: str, lang_key: str,
                 raw: str, cache_dir: Path):
        self.ref_id   = ref_id
        self.name     = name
        self.ftype    = "script"
        self.lang_key = lang_key
        self._raw     = raw
        self._cache_dir = cache_dir

        # Parse immediately (fast — no LLM involved)
        self.parsed: ParsedScript = ScriptParser.parse(name, raw)

        # Disk cache path
        self._disk_path = cache_dir / f"{ref_id}_script.pkl"
        self._save_to_disk()

    def _save_to_disk(self):
        try:
            with open(self._disk_path, "wb") as f:
                pickle.dump(self.parsed, f, protocol=4)
        except Exception:
            pass

    def query(self, query: str, top_k: int = 6) -> str:
        return self.parsed.build_context(query=query, top_k_items=top_k)

    def full_text_preview(self) -> str:
        hdr = self.parsed.summary_header()
        fns = ", ".join(f.name for f in self.parsed.functions[:8])
        cls = ", ".join(c.name for c in self.parsed.classes[:5])
        return (
            f"{hdr}\n"
            + (f"Functions: {fns}\n" if fns else "")
            + (f"Classes:   {cls}\n" if cls else "")
            + f"\n{self._raw[:300]}…"
        )

    @property
    def _hot(self):  # compat with ReferencePanel RAM badge code
        return {}

    @property
    def _chunks(self):  # compat
        return [1]  # one "chunk" for display purposes

    def cleanup(self):
        if self._disk_path.exists():
            try:
                self._disk_path.unlink()
            except Exception:
                pass

class SessionReferenceStore:
    """Manages all SmartReference objects for one chat session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._refs: Dict[str, SmartReference] = {}
        self._index_path = REF_INDEX_DIR / f"{session_id}_refs.json"
        self._load_meta()

    def _load_meta(self):
        if self._index_path.exists():
            try:
                meta = json.loads(self._index_path.read_text())
                for r in meta.get("refs", []):
                    raw_path = REF_CACHE_DIR / f"{r['ref_id']}_raw.txt"
                    if not raw_path.exists():
                        continue
                    raw = raw_path.read_text(encoding="utf-8", errors="replace")
                    if r.get("ftype") == "script":
                        lang_key = r.get("lang_key", "text")
                        self._refs[r["ref_id"]] = ScriptSmartReference(
                            r["ref_id"], r["name"], lang_key, raw, REF_CACHE_DIR)
                    else:
                        self._refs[r["ref_id"]] = SmartReference(
                            r["ref_id"], r["name"], r["ftype"], raw)
            except Exception:
                pass

    def _save_meta(self):
        entries = []
        for r in self._refs.values():
            entry = {"ref_id": r.ref_id, "name": r.name, "ftype": r.ftype}
            if isinstance(r, ScriptSmartReference):
                entry["lang_key"] = r.lang_key
            entries.append(entry)
        meta = {"refs": entries}
        self._index_path.write_text(json.dumps(meta), encoding="utf-8")

    def add_reference(self, name: str, ftype: str, raw_text: str) -> SmartReference:
        ref_id = _simple_hash(name + raw_text[:64])
        raw_path = REF_CACHE_DIR / f"{ref_id}_raw.txt"
        raw_path.write_text(raw_text, encoding="utf-8", errors="replace")
        ref = SmartReference(ref_id, name, ftype, raw_text)
        self._refs[ref_id] = ref
        self._save_meta()
        return ref

    def remove(self, ref_id: str):
        r = self._refs.pop(ref_id, None)
        if r:
            r.cleanup()
        raw_path = REF_CACHE_DIR / f"{ref_id}_raw.txt"
        if raw_path.exists():
            try: raw_path.unlink()
            except Exception: pass
        self._save_meta()

    def build_context_block(self, query: str) -> str:
        if not self._refs:
            return ""
        parts = []
        for ref in self._refs.values():
            snippet = ref.query(query, top_k=5)
            if snippet.strip():
                ftype_label = {"pdf": "📄 PDF", "python": "🐍 Python", "text": "📝 Text"}.get(ref.ftype, "📎")
                parts.append(
                    f"[REFERENCE: {ftype_label} · {ref.name}]\n{snippet}\n[/REFERENCE]")
        return "\n\n".join(parts)

    @property
    def refs(self) -> Dict[str, "SmartReference"]:
        return self._refs

    def flush_ram(self):
        """Emergency RAM flush — spill all chunks to disk."""
        for ref in self._refs.values():
            ref._spill_to_disk()

    def reactive_reload(self, query: str):
        """Selectively reload relevant chunks before final summarization."""
        if _ram_free_mb() < RAM_WATCHDOG_MB():
            return
        for ref in self._refs.values():
            words = set(re.findall(r'\w+', query.lower()))
            for c in ref._chunks[:MAX_RAM_CHUNKS()]:
                if c.idx not in ref._hot:
                    text = ref._load_chunk_from_disk(c.idx)
                    if any(w in text.lower() for w in words):
                        ref._hot[c.idx] = text
                        if _ram_free_mb() < RAM_WATCHDOG_MB():
                            return
  
    def _add_script_reference(self, store, name: str, raw: str) -> ScriptSmartReference:
        """Add a parsed script reference to the store."""
        ref_id = _simple_hash(name + raw[:64])
        # Save raw to disk for resume/reload
        raw_path = REF_CACHE_DIR / f"{ref_id}_raw.txt"
        raw_path.write_text(raw, encoding="utf-8", errors="replace")

        ext = Path(name).suffix.lower()
        _, lang_key = SCRIPT_LANGUAGES.get(ext, ("Text", "text"))
        cache_dir   = REF_CACHE_DIR

        ref = ScriptSmartReference(ref_id, name, lang_key, raw, cache_dir)
        store._refs[ref_id] = ref
        store._save_meta()
        return ref


    def _build_context_block_extended(self, query: str) -> str:
        """
        Extended build_context_block that handles ScriptSmartReference.
        """
        store = self
        if not store._refs:
            return ""
        parts = []
        for ref in store._refs.values():
            if isinstance(ref, ScriptSmartReference):
                snippet = ref.query(query, top_k=6)
                label   = f"💻 Script ({ref.lang_key.upper()}) · {ref.name}"
            else:
                snippet = ref.query(query, top_k=5)
                ftype_label = {
                    "pdf": "📄 PDF", "python": "🐍 Python", "text": "📝 Text"
                }.get(getattr(ref, "ftype", ""), "📎")
                label = f"{ftype_label} · {ref.name}"

            if snippet.strip():
                parts.append(
                    f"[REFERENCE: {label}]\n{snippet}\n[/REFERENCE]")
        return "\n\n".join(parts)

def _add_script_reference(store: "SessionReferenceStore",
                          name: str, raw: str) -> "ScriptSmartReference":
    """Module-level wrapper — called by ReferencePanelV2."""
    return store._add_script_reference(store, name, raw)


def _build_context_block_extended(store: "SessionReferenceStore",
                                  query: str) -> str:
    """Module-level wrapper — called by ReferencePanelV2.get_context_for."""
    return store._build_context_block_extended(query)


# ── Global per-session reference store cache ──────────────────────────────────
_SESSION_REF_STORES: Dict[str, SessionReferenceStore] = {}

def get_ref_store(session_id: str) -> SessionReferenceStore:
    if session_id not in _SESSION_REF_STORES:
        _SESSION_REF_STORES[session_id] = SessionReferenceStore(session_id)
    return _SESSION_REF_STORES[session_id]


# ── RAM Watchdog ──────────────────────────────────────────────────────────────

class RamWatchdog:
    """Monitors RAM and triggers disk-spill or processing-pause signals."""
    triggered = False

    @staticmethod
    def check_and_spill(session_id: str) -> bool:
        """Returns True if spill was triggered."""
        if _ram_free_mb() < RAM_WATCHDOG_MB():
            store = get_ref_store(session_id)
            store.flush_ram()
            RamWatchdog.triggered = True
            return True
        return False

# ─── Multi-PDF Summarization Worker ──────────────────────────────────────────

class MultiPdfSummaryWorker(QThread):
    """
    Summarizes multiple large PDFs with:
    - Adaptive RAM watchdog (disk-spill when RAM low)
    - Per-PDF chunked summarization with pause/resume support
    - Reactive reload before final cross-document consolidation
    """
    file_started    = pyqtSignal(int, str, int)
    file_progress   = pyqtSignal(int, str)
    file_done       = pyqtSignal(int, str)
    section_done    = pyqtSignal(int, int, str, str)
    ram_warning     = pyqtSignal(str)
    final_done      = pyqtSignal(str)
    progress        = pyqtSignal(str)
    err             = pyqtSignal(str)
    pause_suggest   = pyqtSignal(str)   # job_id

    def __init__(self, engine: "LlamaEngine", pdf_texts: List[Tuple[str, str]],
                 session_id: str, engine2: "LlamaEngine" = None,
                 resume_job_id: str = ""):
        super().__init__()
        self.engine        = engine
        self.engine2       = engine2
        self.pdf_texts     = pdf_texts
        self.session_id    = session_id
        self.resume_job_id = resume_job_id
        self._abort        = False
        self._pause        = False
        self._disk_summaries: Dict[str, Path] = {}

        # Stable job id derived from filenames so resume matches original
        names_key = "_".join(Path(fn).stem for fn, _ in pdf_texts)[:40]
        self.job_id = resume_job_id or f"mpdf_{_simple_hash(names_key)}_{int(time.time())}"

    def abort(self):
        self._abort = True

    def request_pause(self):
        self._pause = True

    def _save_state(self, next_fi: int, next_ci: int,
                    file_summaries_so_far: List[str],
                    running_ctx: str,
                    completed_file_summaries: List[Tuple[str, str]]):
        """Persist full job state to disk for resume."""
        # Serialise disk_summaries as {filename: str_path}
        disk_paths = {fn: str(p) for fn, p in self._disk_summaries.items()}
        state = {
            "job_id":             self.job_id,
            "session_id":         self.session_id,
            "filename":           f"{len(self.pdf_texts)} PDFs",
            "total":              len(self.pdf_texts),
            "next_fi":            next_fi,
            "next_ci":            next_ci,
            "file_summaries_so_far": file_summaries_so_far,
            "running_ctx":        running_ctx,
            "completed_files":    completed_file_summaries,
            "disk_summaries":     disk_paths,
            "pdf_texts":          [[fn, txt] for fn, txt in self.pdf_texts],
            "model_path":         getattr(self.engine, "model_path", ""),
            "paused_at":          datetime.now().isoformat(),
            # For display in Config tab
            "next_chunk":         next_ci,
        }
        _save_paused_job(self.job_id, state)

    def run(self):
        cfg          = APP_CONFIG
        CHUNK_CHARS  = int(cfg["multipdf_n_pred_sect"])   # reuse naming
        CHUNK_CHARS  = int(cfg["summary_chunk_chars"])
        CTX_CARRY    = int(cfg["summary_ctx_carry"])
        N_PRED_SECT  = int(cfg["multipdf_n_pred_sect"])
        N_PRED_FINAL = int(cfg["multipdf_n_pred_final"])
        PAUSE_THRESH = int(cfg["pause_after_chunks"])

        # ── Resume state ──────────────────────────────────────────────────────
        start_fi   = 0
        start_ci   = 0
        completed_file_summaries: List[Tuple[str, str]] = []
        file_summaries_so_far: List[str] = []
        running_ctx = ""

        if self.resume_job_id:
            state = _load_paused_job(self.resume_job_id)
            if state:
                start_fi  = state.get("next_fi", 0)
                start_ci  = state.get("next_ci", 0)
                completed_file_summaries = [
                    tuple(x) for x in state.get("completed_files", [])]
                file_summaries_so_far = state.get("file_summaries_so_far", [])
                running_ctx = state.get("running_ctx", "")
                # Restore disk summary paths
                for fn, sp in state.get("disk_summaries", {}).items():
                    p = Path(sp)
                    if p.exists():
                        self._disk_summaries[fn] = p
                self.progress.emit(
                    f"Resuming multi-PDF job from file {start_fi+1}, "
                    f"chunk {start_ci+1}…")

        total_chunks_done = 0

        for fi, (filename, text) in enumerate(self.pdf_texts):
            if fi < start_fi:
                continue   # already completed in previous run

            if self._abort:
                self.err.emit("Aborted"); return

            chunks = self._split(text, CHUNK_CHARS)
            n_chunks = len(chunks)
            self.file_started.emit(fi, filename, n_chunks)
            self.progress.emit(f"Processing '{filename}' — {n_chunks} chunks…")

            spilled = RamWatchdog.check_and_spill(self.session_id)
            if spilled:
                self.ram_warning.emit(
                    f"⚠️ Low RAM before '{filename}' — spilled cache to disk.")

            ci_start = start_ci if fi == start_fi else 0
            if fi > start_fi:
                file_summaries_so_far = []   # reset for new file
            fam = detect_model_family(getattr(self.engine, "model_path", ""))

            for i in range(ci_start, n_chunks):
                if self._abort:
                    return

                if self._pause:
                    self._save_state(fi, i, file_summaries_so_far,
                                     running_ctx, list(completed_file_summaries))
                    self.progress.emit(
                        f"⏸  Paused at file {fi+1}, chunk {i+1}. State saved.")
                    self.err.emit(f"__PAUSED__:{self.job_id}")
                    return

                # Auto-pause suggestion
                total_chunks_done += 1
                if total_chunks_done == PAUSE_THRESH and PAUSE_THRESH > 0:
                    remaining = sum(
                        len(self._split(t, CHUNK_CHARS)) - (i+1 if fidx == fi else 0)
                        for fidx, (_, t) in enumerate(self.pdf_texts)
                        if fidx >= fi
                    )
                    if remaining > PAUSE_THRESH:
                        self.pause_suggest.emit(self.job_id)

                if i % 5 == 0:
                    spilled = RamWatchdog.check_and_spill(self.session_id)
                    if spilled:
                        self.ram_warning.emit(
                            f"⚠️ RAM low at chunk {i+1} of '{filename}'.")

                self.file_progress.emit(fi, f"Chunk {i+1}/{n_chunks}")
                ctx_block = (f"Context from previous chunks:\n{running_ctx}\n\n"
                             if running_ctx else "")
                prompt = (
                    fam.bos + fam.user_prefix +
                    f"Summarise this document chunk. File: '{filename}' | "
                    f"Chunk {i+1}/{n_chunks}\n\n{ctx_block}"
                    f"Chunk:\n{chunks[i]}\n\n"
                    f"Summarise clearly, keep all key facts and data." +
                    fam.user_suffix + fam.assistant_prefix
                )
                summary = self._infer(prompt, N_PRED_SECT)
                if summary is None:
                    self.err.emit(f"Inference failed at chunk {i+1} of '{filename}'")
                    return
                summary = summary.strip()
                file_summaries_so_far.append(f"[Chunk {i+1}/{n_chunks}]\n{summary}")
                running_ctx = summary[-CTX_CARRY:]
                self.section_done.emit(i + 1, n_chunks, chunks[i], summary)

                # Incremental autosave every 3 chunks
                if (i + 1) % 3 == 0:
                    self._save_state(fi, i + 1, file_summaries_so_far,
                                     running_ctx, list(completed_file_summaries))

            # Per-file consolidation
            all_chunks_text = "\n\n".join(file_summaries_so_far)
            consolidate_prompt = (
                fam.bos + fam.user_prefix +
                f"You have summarised all {n_chunks} chunks of '{filename}'.\n\n"
                f"Chunk summaries:\n{all_chunks_text}\n\n"
                f"Write a single coherent summary of this entire document." +
                fam.user_suffix + fam.assistant_prefix
            )
            file_summary = self._infer(consolidate_prompt, N_PRED_FINAL)
            if file_summary is None:
                file_summary = "\n".join(file_summaries_so_far)
            file_summary = file_summary.strip()

            disk_path = REF_CACHE_DIR / f"mpdf_{_simple_hash(filename)}_{fi}.txt"
            disk_path.write_text(file_summary, encoding="utf-8")
            self._disk_summaries[filename] = disk_path
            completed_file_summaries.append((filename, file_summary))
            self.file_done.emit(fi, file_summary)

            # Reset for next file
            file_summaries_so_far = []
            running_ctx = ""
            start_ci = 0   # only first file can have a mid-file resume offset

        if self._abort:
            return

        self.progress.emit("♻️  Reactive reload before final consolidation…")
        query_hint = " ".join(fn for fn, _ in self.pdf_texts)
        get_ref_store(self.session_id).reactive_reload(query_hint)

        self.progress.emit("📝  Final cross-document consolidation…")
        fin_eng = self.engine2 if (self.engine2 and self.engine2.is_loaded) else self.engine
        fin_fam = detect_model_family(getattr(fin_eng, "model_path", ""))

        all_summaries_text = ""
        for filename, disk_path in self._disk_summaries.items():
            try:
                txt = disk_path.read_text(encoding="utf-8")
            except Exception:
                txt = dict(completed_file_summaries).get(filename, "")
            all_summaries_text += f"\n\n=== {filename} ===\n{txt}"

        final_prompt = (
            fin_fam.bos + fin_fam.user_prefix +
            f"You have received summaries of {len(self.pdf_texts)} documents.\n\n"
            f"Per-document summaries:\n{all_summaries_text}\n\n"
            f"Write a final consolidated summary covering all documents, "
            f"including key themes, differences, and connections between them." +
            fin_fam.user_suffix + fin_fam.assistant_prefix
        )
        final = self._infer_with(fin_eng, final_prompt, N_PRED_FINAL + 400)
        if final is None:
            final = all_summaries_text
        self.final_done.emit(final.strip())

        # Cleanup
        _delete_paused_job(self.job_id)
        for dp in self._disk_summaries.values():
            try: dp.unlink()
            except Exception: pass

    def _split(self, text: str, chunk_chars: int) -> List[str]:
        chunks = []
        while text:
            if len(text) <= chunk_chars:
                chunks.append(text.strip()); break
            cut = text.rfind("\n\n", 0, chunk_chars)
            if cut < 200: cut = text.rfind("\n", 0, chunk_chars)
            if cut < 200: cut = chunk_chars
            chunks.append(text[:cut].strip())
            text = text[cut:].lstrip()
        return [c for c in chunks if c]

    def _infer(self, prompt: str, n_predict: int) -> Optional[str]:
        return self._infer_with(self.engine, prompt, n_predict)

    def _infer_with(self, eng, prompt: str, n_predict: int) -> Optional[str]:
        if eng.mode == "server":
            return self._infer_server(eng, prompt, n_predict)
        return self._infer_cli(eng, prompt, n_predict)

    def _infer_server(self, eng, prompt: str, n_predict: int) -> Optional[str]:
        import http.client
        fam = detect_model_family(getattr(eng, "model_path", ""))
        try:
            conn = http.client.HTTPConnection("127.0.0.1", eng.server_port, timeout=400)
            body = json.dumps({
                "prompt": prompt, "n_predict": n_predict, "stream": False,
                "temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.15,
                "stop": fam.stop_tokens,
            })
            conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
            r = conn.getresponse()
            if r.status != 200: return None
            d = json.loads(r.read().decode("utf-8", errors="replace"))
            return d.get("content", "")
        except Exception:
            return None

    def _infer_cli(self, eng, prompt: str, n_predict: int) -> Optional[str]:
        try:
            result = subprocess.run(
                [LLAMA_CLI, "-m", eng.model_path, "-t", str(DEFAULT_THREADS()),
                 "--ctx-size", str(getattr(eng, "ctx_value", DEFAULT_CTX())),
                 "-n", str(n_predict), "--no-display-prompt", "--no-escape",
                 "--temp", "0.3", "--repeat-penalty", "1.15", "-p", prompt],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, timeout=400,
            )
            return result.stdout.decode("utf-8", errors="replace")
        except Exception:
            return None
        
# ═════════════════════════════ MARKDOWN RENDERER ════════════════════════════

def _md_to_html(text: str,
                code_store: Optional[Dict[str, str]] = None) -> str:
    """
    Markdown → HTML for Qt's limited renderer.

    Parameters
    ----------
    text       : raw markdown / plain text from the model
    code_store : dict that will be filled with {block_id: raw_code}.
                 Pass the RichTextEdit's ._code_blocks dict so copy works.

    Qt renderer constraints
    ───────────────────────
    ✓  <table>, <tr>, <td align="…">
    ✓  Inline styles: color, background-color, font-family, font-size,
       font-weight, padding, margin, border (simple 1px solid …),
       white-space:pre, width
    ✗  float, position, display:flex/grid
    ✗  JavaScript / onclick
    ✗  CSS pseudo-elements, :hover
    """
    if code_store is None:
        code_store = {}

    # Escape HTML entities BEFORE any substitutions
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    _counter = [0]

    # ── Fenced code blocks ──────────────────────────────────────────────────
    def _fenced(m: re.Match) -> str:
        lang  = m.group(1).strip().lower()
        body  = m.group(2)          # already HTML-escaped by outer replace
        bid   = f"cb{abs(hash(body[:40]))}_{_counter[0]}"
        _counter[0] += 1

        # Raw code for clipboard (un-escape entities)
        raw_code = (body
                    .replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">"))
        code_store[bid] = raw_code

        n_lines = body.count("\n") + 1
        lang_up = lang.upper() if lang else "CODE"

        LANG_COL = {
            "python": "#f59e0b", "py": "#f59e0b",
            "javascript": "#f7df1e", "js": "#f7df1e",
            "typescript": "#3178c6", "ts": "#3178c6",
            "sql":  "#00b4d8", "rust": "#f74c00",
            "bash": "#4ade80", "sh": "#4ade80",
            "go":   "#00acd7", "c":   "#a9b8c3",
            "cpp":  "#f34b7d", "java": "#b07219",
            "json": "#cbcb41", "yaml": "#cc3e44",
        }.get(lang, "#a78bfa")

        # Syntax highlighting (Python only — safe regex on escaped text)
        display = body
        if lang in ("python", "py", ""):
            KW   = "#c792ea"; STR = "#c3e88d"; NUM = "#f78c6c"; CMT = "#546e7a"
            # Comments first (they take priority)
            display = re.sub(
                r'(#[^\n]*)',
                f'<span style="color:{CMT};">\\1</span>',
                display)
            # Strings (double)
            display = re.sub(
                r'(&quot;(?:[^&]|&(?!quot;))*?&quot;|"(?:[^"\\]|\\.)*?")',
                f'<span style="color:{STR};">\\1</span>',
                display)
            # Numbers
            display = re.sub(
                r'\b(\d+\.?\d*)\b',
                f'<span style="color:{NUM};">\\1</span>',
                display)
            # Keywords
            for kw in ("def","class","import","from","return","yield",
                       "if","elif","else","for","while","try","except",
                       "finally","with","as","pass","break","continue",
                       "lambda","and","or","not","in","is","None",
                       "True","False","async","await","raise","del",
                       "global","nonlocal","assert"):
                display = re.sub(
                    rf'\b({re.escape(kw)})\b',
                    f'<span style="color:{KW};font-weight:600;">\\1</span>',
                    display)

        # Table-based layout (Qt supports tables reliably)
        html = (
            f'<table width="100%" cellpadding="0" cellspacing="0" '
            f'style="background:#0a0a18;'
            f'border:1px solid #252340;'
            f'border-radius:6px;margin:8px 0;">'

            # ── toolbar row ──
            f'<tr>'
            f'<td style="padding:5px 12px;'
            f'border-bottom:1px solid #252340;'
            f'background:#0d0d1e;">'
            f'<span style="color:{LANG_COL};font-size:9px;font-weight:700;'
            f'font-family:Consolas,monospace;background:rgba(255,255,255,0.06);'
            f'border:1px solid rgba(255,255,255,0.09);border-radius:4px;'
            f'padding:1px 6px;">{lang_up}</span>'
            f'<span style="color:#4a4860;font-size:10px;"> &nbsp;{n_lines} lines</span>'
            f'</td>'
            f'<td align="right" style="padding:5px 12px;'
            f'border-bottom:1px solid #252340;'
            f'background:#0d0d1e;">'
            f'<a href="copy://{bid}" '
            f'style="color:#a78bfa;font-size:10px;'
            f'text-decoration:none;font-family:Segoe UI,sans-serif;">'
            f'⧉ Copy</a>'
            f'</td>'
            f'</tr>'

            # ── code body row ──
            f'<tr>'
            f'<td colspan="2" style="padding:10px 14px;">'
            f'<pre style="margin:0;'
            f'font-family:Consolas,&quot;Courier New&quot;,monospace;'
            f'font-size:12px;color:#d4c6ff;'
            f'white-space:pre-wrap;line-height:1.6;">'
            f'{display}</pre>'
            f'</td>'
            f'</tr>'
            f'</table>'
        )
        return html

    text = re.sub(r'```(\w*)\n?(.*?)```', _fenced, text, flags=re.DOTALL)

    # ── Inline code ──────────────────────────────────────────────────────────
    text = re.sub(
        r'`([^`\n]+)`',
        r'<code style="background:#1c1c3a;border-radius:3px;'
        r'padding:1px 5px;font-family:Consolas,monospace;'
        r'font-size:12px;color:#c4b5fd;">\1</code>',
        text)

    # ── Headers ──────────────────────────────────────────────────────────────
    text = re.sub(r'^### (.+)$',
        r'<p style="color:#c4b5fd;font-size:13px;margin:10px 0 2px;'
        r'font-weight:600;">\1</p>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',
        r'<p style="color:#c4b5fd;font-size:14px;margin:12px 0 3px;'
        r'font-weight:700;">\1</p>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$',
        r'<p style="color:#c4b5fd;font-size:15px;margin:12px 0 4px;'
        r'font-weight:700;">\1</p>', text, flags=re.MULTILINE)

    # ── Bold / italic ─────────────────────────────────────────────────────────
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'\*\*(.+?)\*\*',     r'<b>\1</b>',         text)
    text = re.sub(r'\*(.+?)\*',         r'<i>\1</i>',          text)

    # ── Horizontal rules ─────────────────────────────────────────────────────
    text = re.sub(
        r'^---+$',
        r'<table width="100%" cellpadding="0" cellspacing="0" style="margin:8px 0;">'
        r'<tr><td style="border-top:1px solid #252340;"></td></tr></table>',
        text, flags=re.MULTILINE)

    # ── Bullet lists ──────────────────────────────────────────────────────────
    text = re.sub(
        r'^[ \t]*[-*•] (.+)$',
        r'<p style="margin:2px 0;padding-left:16px;">'
        r'<span style="color:#a78bfa;">•</span>&nbsp;\1</p>',
        text, flags=re.MULTILINE)

    # ── Numbered lists ────────────────────────────────────────────────────────
    text = re.sub(
        r'^[ \t]*(\d+)\. (.+)$',
        r'<p style="margin:2px 0;padding-left:16px;">'
        r'<span style="color:#a78bfa;">\1.</span>&nbsp;\2</p>',
        text, flags=re.MULTILINE)

    # ── Newlines → <br> (only outside block tags) ────────────────────────────
    parts = re.split(r'(<table[\s\S]*?</table>|<p[\s\S]*?</p>)', text)
    out = []
    for part in parts:
        if part.startswith('<'):
            out.append(part)
        else:
            out.append(part.replace('\n', '<br>'))
    return ''.join(out)

# ═══════════════════════════════════════════════════════════════════════════
# 4. SCRIPT PARSER — advanced multi-language structured parser
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_LANGUAGES = {
    # extension → (display_name, parser_key)
    ".py":    ("Python",      "python"),
    ".js":    ("JavaScript",  "js"),
    ".jsx":   ("React/JSX",   "js"),
    ".ts":    ("TypeScript",  "ts"),
    ".tsx":   ("React/TSX",   "ts"),
    ".sql":   ("SQL",         "sql"),
    ".rs":    ("Rust",        "rust"),
    ".go":    ("Go",          "go"),
    ".c":     ("C",           "c"),
    ".h":     ("C Header",    "c"),
    ".cpp":   ("C++",         "cpp"),
    ".hpp":   ("C++ Header",  "cpp"),
    ".java":  ("Java",        "java"),
    ".kt":    ("Kotlin",      "kotlin"),
    ".rb":    ("Ruby",        "ruby"),
    ".sh":    ("Bash",        "bash"),
    ".bash":  ("Bash",        "bash"),
    ".yaml":  ("YAML",        "yaml"),
    ".yml":   ("YAML",        "yaml"),
    ".json":  ("JSON",        "json"),
    ".toml":  ("TOML",        "toml"),
    ".md":    ("Markdown",    "text"),
    ".txt":   ("Text",        "text"),
    ".lua":   ("Lua",         "lua"),
    ".swift": ("Swift",       "swift"),
    ".cs":    ("C#",          "csharp"),
    ".php":   ("PHP",         "php"),
    ".r":     ("R",           "r"),
    ".jl":    ("Julia",       "julia"),
}

@dataclass
class ParsedItem:
    """A single parsed code element (function, class, import, etc.)."""
    kind:       str          # "import" | "function" | "class" | "constant" |
                             # "type" | "query" | "table" | "route" | "schema"
    name:       str
    signature:  str = ""     # full declaration line / signature
    body:       str = ""     # raw body text
    docstring:  str = ""     # extracted docstring / doc comment
    decorators: List[str] = field(default_factory=list)
    children:   List["ParsedItem"] = field(default_factory=list)  # methods in class
    line_start: int = 0
    line_end:   int = 0
    language:   str = ""

    def keyword_score(self, words: set) -> float:
        target = (self.name + " " + self.signature + " " +
                  self.docstring).lower()
        return sum(1.0 for w in words if w in target)

    def to_context_snippet(self, include_body: bool = True,
                           max_body: int = 600) -> str:
        parts = []
        if self.decorators:
            parts.append("\n".join(self.decorators))
        parts.append(self.signature if self.signature else self.name)
        if self.docstring:
            parts.append(f"    \"\"\"{self.docstring.strip()[:300]}\"\"\"")
        if include_body and self.body:
            body = self.body[:max_body]
            if len(self.body) > max_body:
                body += "\n    # … (truncated)"
            parts.append(body)
        # children (methods)
        for child in self.children[:6]:
            parts.append(f"    {child.signature}")
            if child.docstring:
                parts.append(f'        """{child.docstring.strip()[:120]}"""')
        return "\n".join(parts)


@dataclass
class ParsedScript:
    """
    Structured representation of a source-code file,
    ready to be injected as rich LLM context.
    """
    language:   str
    lang_key:   str          # e.g. "python", "js", "sql"
    filename:   str
    raw:        str          # full raw text (kept for fallback)
    imports:    List[ParsedItem] = field(default_factory=list)
    functions:  List[ParsedItem] = field(default_factory=list)
    classes:    List[ParsedItem] = field(default_factory=list)
    constants:  List[ParsedItem] = field(default_factory=list)
    types:      List[ParsedItem] = field(default_factory=list)
    misc:       List[ParsedItem] = field(default_factory=list)
    errors:     List[str]        = field(default_factory=list)

    # ── meta ──────────────────────────────────────────────────────────────────
    @property
    def total_items(self) -> int:
        return (len(self.imports) + len(self.functions) +
                len(self.classes) + len(self.constants) +
                len(self.types) + len(self.misc))

    @property
    def all_items(self) -> List[ParsedItem]:
        return (self.imports + self.functions + self.classes +
                self.constants + self.types + self.misc)

    def summary_header(self) -> str:
        """One-line structural overview."""
        parts = []
        if self.imports:    parts.append(f"{len(self.imports)} imports")
        if self.classes:    parts.append(f"{len(self.classes)} classes")
        if self.functions:  parts.append(f"{len(self.functions)} functions/methods")
        if self.constants:  parts.append(f"{len(self.constants)} constants")
        if self.types:      parts.append(f"{len(self.types)} type defs")
        return (f"[{self.language} · {self.filename}]  "
                + (", ".join(parts) if parts else "no parsed items"))

    # ── context block builder ─────────────────────────────────────────────────
    def build_context(self, query: str = "", top_k_items: int = 8,
                      max_chars: int = 3500) -> str:
        """
        Build a structured context block for LLM injection.
        When a query is provided, ranks items by keyword relevance.
        """
        header = (
            f"╔══ SCRIPT REFERENCE ══════════════════════════════════════\n"
            f"║  File    : {self.filename}\n"
            f"║  Language: {self.language}\n"
            f"║  Structure: {self.summary_header()}\n"
            f"╚══════════════════════════════════════════════════════════\n\n"
        )

        sections: List[str] = []
        total_chars = len(header)

        # Imports always included (compact)
        if self.imports:
            imp_block = "── Imports ──\n"
            imp_block += "\n".join(i.signature for i in self.imports[:30])
            sections.append(imp_block)
            total_chars += len(imp_block)

        # Constants / type aliases (compact)
        if self.constants or self.types:
            ct_block = "\n── Constants & Types ──\n"
            for item in (self.constants + self.types)[:20]:
                ct_block += item.signature + "\n"
            sections.append(ct_block)
            total_chars += len(ct_block)

        # Rank remaining items
        words = set(re.findall(r'\w+', query.lower())) if query else set()
        candidates: List[Tuple[float, ParsedItem]] = []
        for item in self.functions + self.classes + self.misc:
            score = item.keyword_score(words) if words else 0.0
            candidates.append((score, item))
        candidates.sort(key=lambda x: -x[0])

        for score, item in candidates[:top_k_items]:
            if total_chars >= max_chars:
                break
            budget = min(600, max_chars - total_chars)
            snippet = item.to_context_snippet(
                include_body=(score > 0 or len(candidates) <= 4),
                max_body=budget)
            section = f"\n── {item.kind.capitalize()}: {item.name} ──\n{snippet}\n"
            sections.append(section)
            total_chars += len(section)

        return header + "\n".join(sections)

    def query(self, query: str, top_k: int = 6) -> str:
        """Compatibility with SmartReference.query() interface."""
        return self.build_context(query=query, top_k_items=top_k)

class ScriptParser:
    """
    Multi-language script parser.
    Uses Python's AST for .py files and regex-based parsers for others.
    """

    @classmethod
    def parse(cls, filename: str, raw: str) -> "ParsedScript":
        ext = Path(filename).suffix.lower()
        lang, key = SCRIPT_LANGUAGES.get(ext, ("Text", "text"))
        ps = ParsedScript(language=lang, lang_key=key,
                          filename=filename, raw=raw)
        try:
            parser = getattr(cls, f"_parse_{key}", cls._parse_generic)
            parser(ps, raw)
        except Exception as e:
            ps.errors.append(f"Parser error: {e}")
            cls._parse_generic(ps, raw)
        return ps

    # ── Python parser (AST) ───────────────────────────────────────────────────
    @classmethod
    def _parse_python(cls, ps: ParsedScript, raw: str):
        try:
            tree = ast.parse(raw)
        except SyntaxError as e:
            ps.errors.append(f"SyntaxError: {e}")
            cls._parse_generic(ps, raw)
            return

        lines = raw.splitlines()

        def _src(node) -> str:
            try:
                return ast.get_source_segment(raw, node) or ""
            except Exception:
                return ""

        def _docstr(node) -> str:
            try:
                v = ast.get_docstring(node)
                return v or ""
            except Exception:
                return ""

        def _sig(node, lines) -> str:
            """Reconstruct signature line(s)."""
            try:
                l = node.lineno - 1
                sig_lines = []
                while l < len(lines):
                    sig_lines.append(lines[l])
                    if ":" in lines[l]:
                        break
                    l += 1
                return "\n".join(sig_lines)
            except Exception:
                return getattr(node, "name", "")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    n = alias.asname or alias.name
                    ps.imports.append(ParsedItem(
                        kind="import", name=n,
                        signature=f"import {alias.name}"
                        + (f" as {alias.asname}" if alias.asname else ""),
                        language="python"))

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names  = ", ".join(
                    (a.asname or a.name) for a in node.names)
                full   = f"from {module} import {names}"
                ps.imports.append(ParsedItem(
                    kind="import", name=module,
                    signature=full, language="python"))

            elif isinstance(node, ast.FunctionDef) and \
                 not isinstance(node, ast.AsyncFunctionDef):
                if not any(
                    isinstance(p, ast.FunctionDef)
                    for p in ast.walk(tree)
                    if isinstance(getattr(p, "body", None), list)
                    and any(child is node for child in p.body)
                ):
                    decs = [ast.unparse(d) for d in node.decorator_list]
                    ps.functions.append(ParsedItem(
                        kind="function", name=node.name,
                        signature=_sig(node, lines),
                        body=_src(node),
                        docstring=_docstr(node),
                        decorators=[f"@{d}" for d in decs],
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        language="python"))

            elif isinstance(node, ast.AsyncFunctionDef):
                decs = [ast.unparse(d) for d in node.decorator_list]
                ps.functions.append(ParsedItem(
                    kind="function", name=node.name,
                    signature="async " + _sig(node, lines),
                    body=_src(node), docstring=_docstr(node),
                    decorators=[f"@{d}" for d in decs],
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    language="python"))

            elif isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases]
                methods = []
                for child in ast.walk(node):
                    if isinstance(child, (ast.FunctionDef,
                                          ast.AsyncFunctionDef)) and \
                       child is not node:
                        decs = [ast.unparse(d) for d in child.decorator_list]
                        methods.append(ParsedItem(
                            kind="method", name=child.name,
                            signature=_sig(child, lines),
                            body=_src(child),
                            docstring=_docstr(child),
                            decorators=[f"@{d}" for d in decs],
                            language="python"))

                base_str = f"({', '.join(bases)})" if bases else ""
                ps.classes.append(ParsedItem(
                    kind="class", name=node.name,
                    signature=f"class {node.name}{base_str}:",
                    docstring=_docstr(node),
                    children=methods,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    language="python"))

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name.isupper() or name.startswith("_"):
                            try:
                                val = ast.unparse(node.value)[:80]
                            except Exception:
                                val = "..."
                            ps.constants.append(ParsedItem(
                                kind="constant", name=name,
                                signature=f"{name} = {val}",
                                language="python"))

        # Remove top-level functions that are actually methods
        class_method_names: set = set()
        for cls_item in ps.classes:
            for m in cls_item.children:
                class_method_names.add(m.name)
        ps.functions = [f for f in ps.functions
                        if f.name not in class_method_names
                        or f.line_start not in
                        {c.line_start for c in ps.classes}]

    # ── JavaScript / TypeScript parser (regex) ────────────────────────────────
    @classmethod
    def _parse_js(cls, ps: ParsedScript, raw: str):
        cls._parse_js_ts(ps, raw, "javascript")

    @classmethod
    def _parse_ts(cls, ps: ParsedScript, raw: str):
        cls._parse_js_ts(ps, raw, "typescript")

    @classmethod
    def _parse_js_ts(cls, ps: ParsedScript, raw: str, lang: str):
        lines = raw.splitlines()

        # Imports
        for m in re.finditer(
                r'^(?:import|export\s+(?:default\s+)?)?(?:const|let|var)?\s*'
                r'\{?[^}]*\}?\s*(?:from\s+)?["\'][^"\']+["\']',
                raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(0)[:60],
                signature=m.group(0).split("\n")[0][:120],
                language=lang))

        for m in re.finditer(
                r'^import\s+.*?(?:from\s+)?["\'][^"\']+["\'];?$',
                raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(0)[:60],
                signature=m.group(0)[:120], language=lang))

        # Functions
        for m in re.finditer(
                r'^(?:export\s+)?(?:default\s+)?(?:async\s+)?'
                r'function\s+(\w+)\s*\([^)]*\)',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), line_start=ln, language=lang))

        # Arrow functions assigned to const
        for m in re.finditer(
                r'^(?:export\s+)?const\s+(\w+)\s*=\s*'
                r'(?:async\s+)?\([^)]*\)\s*=>',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), line_start=ln, language=lang))

        # Classes
        for m in re.finditer(
                r'^(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.classes.append(ParsedItem(
                kind="class", name=m.group(1),
                signature=m.group(0),
                line_start=ln, language=lang))

        # TypeScript types / interfaces
        if lang == "typescript":
            for m in re.finditer(
                    r'^(?:export\s+)?(?:type|interface)\s+(\w+)',
                    raw, re.MULTILINE):
                ln = raw[:m.start()].count("\n")
                ps.types.append(ParsedItem(
                    kind="type", name=m.group(1),
                    signature=m.group(0), line_start=ln, language=lang))

        # Constants
        for m in re.finditer(
                r'^(?:export\s+)?const\s+([A-Z_][A-Z0-9_]{2,})\s*=\s*(.{1,80})',
                raw, re.MULTILINE):
            ps.constants.append(ParsedItem(
                kind="constant", name=m.group(1),
                signature=f"const {m.group(1)} = {m.group(2)[:60]}",
                language=lang))

    # ── SQL parser ────────────────────────────────────────────────────────────
    @classmethod
    def _parse_sql(cls, ps: ParsedScript, raw: str):
        # Tables
        for m in re.finditer(
                r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)',
                raw, re.IGNORECASE):
            ln = raw[:m.start()].count("\n")
            ps.misc.append(ParsedItem(
                kind="table", name=m.group(1),
                signature=m.group(0), line_start=ln, language="sql"))

        # Views
        for m in re.finditer(
                r'CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(\w+)',
                raw, re.IGNORECASE):
            ln = raw[:m.start()].count("\n")
            ps.misc.append(ParsedItem(
                kind="view", name=m.group(1),
                signature=m.group(0), line_start=ln, language="sql"))

        # Procedures / Functions
        for m in re.finditer(
                r'CREATE\s+(?:OR\s+REPLACE\s+)?'
                r'(?:FUNCTION|PROCEDURE)\s+(\w+)',
                raw, re.IGNORECASE):
            ln = raw[:m.start()].count("\n")
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), line_start=ln, language="sql"))

        # Indexes
        for m in re.finditer(
                r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+(\w+)\s+ON\s+(\w+)',
                raw, re.IGNORECASE):
            ps.misc.append(ParsedItem(
                kind="index", name=m.group(1),
                signature=m.group(0), language="sql"))

        # Named queries (CTEs)
        for m in re.finditer(r'\bWITH\s+(\w+)\s+AS\s*\(',
                             raw, re.IGNORECASE):
            ps.misc.append(ParsedItem(
                kind="query", name=m.group(1),
                signature=m.group(0), language="sql"))

    # ── Rust parser ───────────────────────────────────────────────────────────
    @classmethod
    def _parse_rust(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^use\s+([^;]+);', raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(1).split("::")[-1],
                signature=m.group(0)[:100], language="rust"))
        for m in re.finditer(
                r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), line_start=ln, language="rust"))
        for m in re.finditer(
                r'^(?:pub\s+)?struct\s+(\w+)', raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.classes.append(ParsedItem(
                kind="struct", name=m.group(1),
                signature=m.group(0), line_start=ln, language="rust"))
        for m in re.finditer(
                r'^(?:pub\s+)?(?:trait|impl(?:\s+\w+\s+for)?)\s+(\w+)',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.types.append(ParsedItem(
                kind="trait/impl", name=m.group(1),
                signature=m.group(0), line_start=ln, language="rust"))
        for m in re.finditer(
                r'^(?:pub\s+)?const\s+([A-Z_]+)\s*:\s*[^=]+=\s*(.{1,60})',
                raw, re.MULTILINE):
            ps.constants.append(ParsedItem(
                kind="constant", name=m.group(1),
                signature=m.group(0)[:100], language="rust"))

    # ── Go parser ────────────────────────────────────────────────────────────
    @classmethod
    def _parse_go(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(
                r'^import\s*\(([^)]+)\)', raw, re.MULTILINE | re.DOTALL):
            for pkg in m.group(1).splitlines():
                pkg = pkg.strip().strip('"')
                if pkg:
                    ps.imports.append(ParsedItem(
                        kind="import", name=pkg,
                        signature=f'import "{pkg}"', language="go"))
        for m in re.finditer(
                r'^func\s+(?:\(\s*\w+\s+\*?\w+\s*\)\s+)?(\w+)\s*\([^)]*\)',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), line_start=ln, language="go"))
        for m in re.finditer(r'^type\s+(\w+)\s+struct', raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.classes.append(ParsedItem(
                kind="struct", name=m.group(1),
                signature=m.group(0), line_start=ln, language="go"))
        for m in re.finditer(r'^type\s+(\w+)\s+interface', raw, re.MULTILINE):
            ps.types.append(ParsedItem(
                kind="interface", name=m.group(1),
                signature=m.group(0), language="go"))

    # ── C / C++ parser ────────────────────────────────────────────────────────
    @classmethod
    def _parse_c(cls, ps: ParsedScript, raw: str):
        cls._parse_cpp(ps, raw)

    @classmethod
    def _parse_cpp(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^#include\s*[<"]([^>"]+)[>"]',
                             raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(1),
                signature=m.group(0), language="cpp"))
        for m in re.finditer(
                r'^(?:class|struct)\s+(\w+)',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.classes.append(ParsedItem(
                kind=m.group(0).split()[0], name=m.group(1),
                signature=m.group(0), line_start=ln, language="cpp"))
        for m in re.finditer(
                r'^(?:[\w:*&<>\s]+)\s+(\w+)\s*\([^)]*\)\s*(?:const)?\s*[{;]',
                raw, re.MULTILINE):
            name = m.group(1)
            if name not in ("if", "while", "for", "switch", "return"):
                ln = raw[:m.start()].count("\n")
                ps.functions.append(ParsedItem(
                    kind="function", name=name,
                    signature=m.group(0).rstrip("{;").strip(),
                    line_start=ln, language="cpp"))
        for m in re.finditer(
                r'^#define\s+([A-Z_][A-Z0-9_]+)\s+(.{1,60})',
                raw, re.MULTILINE):
            ps.constants.append(ParsedItem(
                kind="constant", name=m.group(1),
                signature=m.group(0)[:100], language="cpp"))

    # ── Java / Kotlin parser ──────────────────────────────────────────────────
    @classmethod
    def _parse_java(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^import\s+([\w.]+);', raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(1).split(".")[-1],
                signature=m.group(0), language="java"))
        for m in re.finditer(
                r'^(?:public|private|protected|abstract|final|static)*\s*'
                r'class\s+(\w+)',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.classes.append(ParsedItem(
                kind="class", name=m.group(1),
                signature=m.group(0), line_start=ln, language="java"))
        for m in re.finditer(
                r'(?:public|private|protected|static|final|void|synchronized)*\s+'
                r'(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\([^)]*\)',
                raw, re.MULTILINE):
            if m.group(1) not in ("if","while","for","switch","new","return"):
                ln = raw[:m.start()].count("\n")
                ps.functions.append(ParsedItem(
                    kind="method", name=m.group(1),
                    signature=m.group(0), line_start=ln, language="java"))

    @classmethod
    def _parse_kotlin(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^import\s+([\w.]+)', raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(1).split(".")[-1],
                signature=m.group(0), language="kotlin"))
        for m in re.finditer(
                r'^(?:data\s+)?class\s+(\w+)', raw, re.MULTILINE):
            ps.classes.append(ParsedItem(
                kind="class", name=m.group(1),
                signature=m.group(0), language="kotlin"))
        for m in re.finditer(
                r'^(?:fun|suspend fun|private fun|public fun|internal fun)\s+(\w+)',
                raw, re.MULTILINE):
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), language="kotlin"))

    # ── Bash parser ───────────────────────────────────────────────────────────
    @classmethod
    def _parse_bash(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^source\s+(\S+)', raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(1),
                signature=m.group(0), language="bash"))
        for m in re.finditer(
                r'^(?:function\s+)?(\w+)\s*\(\s*\)\s*\{',
                raw, re.MULTILINE):
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), language="bash"))
        for m in re.finditer(
                r'^([A-Z_][A-Z0-9_]*)=(.{1,80})', raw, re.MULTILINE):
            ps.constants.append(ParsedItem(
                kind="constant", name=m.group(1),
                signature=m.group(0)[:100], language="bash"))

    # ── Ruby parser ───────────────────────────────────────────────────────────
    @classmethod
    def _parse_ruby(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^require(?:_relative)?\s+["\']([^"\']+)["\']',
                             raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(1),
                signature=m.group(0), language="ruby"))
        for m in re.finditer(r'^class\s+(\w+)', raw, re.MULTILINE):
            ps.classes.append(ParsedItem(
                kind="class", name=m.group(1),
                signature=m.group(0), language="ruby"))
        for m in re.finditer(r'^\s*def\s+(\w+)', raw, re.MULTILINE):
            ps.functions.append(ParsedItem(
                kind="method", name=m.group(1),
                signature=m.group(0).strip(), language="ruby"))

    # ── JSON / YAML / TOML (structural only) ─────────────────────────────────
    @classmethod
    def _parse_json(cls, ps: ParsedScript, raw: str):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for k, v in list(data.items())[:40]:
                    ps.misc.append(ParsedItem(
                        kind="key", name=k,
                        signature=f'"{k}": {json.dumps(v)[:80]}',
                        language="json"))
        except Exception as e:
            ps.errors.append(str(e))
            cls._parse_generic(ps, raw)

    @classmethod
    def _parse_yaml(cls, ps: ParsedScript, raw: str):
        # Basic key extraction without pyyaml dependency
        for m in re.finditer(r'^(\w[\w_-]*):\s*(.{0,80})',
                             raw, re.MULTILINE):
            ps.misc.append(ParsedItem(
                kind="key", name=m.group(1),
                signature=f"{m.group(1)}: {m.group(2)[:60]}",
                language="yaml"))

    @classmethod
    def _parse_toml(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^\[([^\]]+)\]', raw, re.MULTILINE):
            ps.misc.append(ParsedItem(
                kind="section", name=m.group(1),
                signature=f"[{m.group(1)}]", language="toml"))
        for m in re.finditer(r'^(\w[\w_-]*)\s*=\s*(.{0,80})',
                             raw, re.MULTILINE):
            ps.constants.append(ParsedItem(
                kind="constant", name=m.group(1),
                signature=f"{m.group(1)} = {m.group(2)[:60]}",
                language="toml"))

    # ── Generic fallback ──────────────────────────────────────────────────────
    @classmethod
    def _parse_generic(cls, ps: ParsedScript, raw: str):
        """Best-effort parser for unknown/unsupported languages."""
        lines = raw.splitlines()
        for i, line in enumerate(lines[:200]):
            stripped = line.strip()
            # Anything that looks like a function/method definition
            if re.match(r'\w.*\(.*\)\s*[:{]?\s*$', stripped) and \
               not stripped.startswith(("#", "//", "*", "/*")):
                ps.misc.append(ParsedItem(
                    kind="item", name=stripped[:50],
                    signature=stripped[:120],
                    line_start=i + 1, language=ps.lang_key))


# ═════════════════════════════ UI COMPONENTS ════════════════════════════════

class MessageWidget(QWidget):
    _COLLAPSE_PX = 260

    def __init__(self, role: str, content: str, timestamp: str, tag: str = ""):
        super().__init__()
        self.role         = role
        self._text        = content
        self._collapsed   = False
        self._collapsible = False
        self._pending_txt = ""
        self._tag         = tag

        outer = QHBoxLayout()
        outer.setContentsMargins(12, 4, 12, 4)
        outer.setSpacing(0)

        self.bubble = QFrame()
        if role == "user":
            bb = ("qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                  "stop:0 rgba(30,24,70,0.88),stop:1 rgba(18,14,58,0.92))")
            bc = "rgba(124,58,237,0.35)"
        elif tag == "🧠 Reasoning":
            bb = ("qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                  "stop:0 rgba(14,22,40,0.88),stop:1 rgba(8,16,32,0.92))")
            bc = "rgba(34,211,238,0.28)"
        else:
            bb = ("qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                  "stop:0 rgba(11,26,18,0.88),stop:1 rgba(8,20,14,0.92))")
            bc = "rgba(52,211,153,0.22)"

        self.bubble.setStyleSheet(
            f"QFrame{{background:{bb};border:1px solid {bc};border-radius:14px;}}")
        self.bubble.setMaximumWidth(860)
        self.bubble.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        bl = QVBoxLayout()
        bl.setContentsMargins(14, 10, 14, 10)
        bl.setSpacing(6)

        # ── header row ───────────────────────────────────────────────────────
        hdr = QHBoxLayout(); hdr.setSpacing(8)
        name_text  = "You" if role == "user" else (tag or "Assistant")
        name_color = (C["acc"] if role == "user" else
                      (C["pipeline"] if tag == "🧠 Reasoning" else C["ok"]))
        name_lbl   = QLabel(name_text)
        name_lbl.setStyleSheet(
            f"color:{name_color};font-weight:700;font-size:11px;letter-spacing:0.3px;")
        ts_lbl = QLabel(timestamp)
        ts_lbl.setStyleSheet(f"color:{C['txt2']};font-size:10px;")

        self._copy_btn = QPushButton("⧉")
        self._copy_btn.setFixedSize(22, 22)
        self._copy_btn.setToolTip("Copy whole message")
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.setStyleSheet(
            f"QPushButton{{background:rgba(167,139,250,0.08);color:{C['txt2']};"
            f"border:1px solid rgba(167,139,250,0.15);border-radius:5px;"
            f"font-size:11px;padding:0;}}"
            f"QPushButton:hover{{background:rgba(167,139,250,0.22);color:{C['acc2']};}}")
        self._copy_btn.clicked.connect(self._copy_all)

        hdr.addWidget(name_lbl)
        hdr.addStretch()
        hdr.addWidget(ts_lbl)
        hdr.addWidget(self._copy_btn)

        # ── body (RichTextEdit) ───────────────────────────────────────────────
        self.te = RichTextEdit()
        self.te.setFrameShape(QFrame.Shape.NoFrame)
        self.te.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.te.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.te.setStyleSheet(
            f"QTextEdit{{background:transparent;color:{C['txt']};"
            f"font-size:13px;padding:0;border:none;line-height:1.6;}}")
        self.te.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.te.document().contentsChanged.connect(self._fit)

        bl.addLayout(hdr)
        bl.addWidget(self.te)

        self._expand_btn: Optional[QPushButton] = None
        self.bubble.setLayout(bl)

        if role == "user":
            outer.addStretch(); outer.addWidget(self.bubble)
        else:
            outer.addWidget(self.bubble); outer.addStretch()

        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush_pending)

        self.setLayout(outer)

        if content:
            self._render_html(content)
        else:
            self._fit()

    # ── rendering ─────────────────────────────────────────────────────────────
    def _render_html(self, text: str):
        html = _md_to_html(text, code_store=self.te._code_blocks)
        self.te.setHtml(
            f'<div style="color:{C["txt"]};font-size:13px;'
            f'line-height:1.65;font-family:Segoe UI,Inter,sans-serif;">'
            f'{html}</div>')
        self._text = text
        self._fit()
        QTimer.singleShot(60, self._maybe_add_expander)

    def _maybe_add_expander(self):
        natural_h = int(self.te.document().size().height()) + 6
        if natural_h > self._COLLAPSE_PX + 60 and not self._collapsible:
            self._collapsible = True
            self._collapsed   = True
            self.te.setFixedHeight(self._COLLAPSE_PX)
            btn = QPushButton("▼  Show more")
            btn.setStyleSheet(
                f"QPushButton{{background:rgba(124,58,237,0.12);color:{C['acc']};"
                f"border:1px solid rgba(167,139,250,0.2);border-radius:6px;"
                f"padding:3px 12px;font-size:11px;}}"
                f"QPushButton:hover{{background:rgba(124,58,237,0.25);color:{C['acc2']};}}")
            btn.clicked.connect(self._toggle_expand)
            self.bubble.layout().addWidget(btn)
            self._expand_btn = btn

    def _toggle_expand(self):
        if self._collapsed:
            self._collapsed = False
            h = int(self.te.document().size().height()) + 6
            self.te.setFixedHeight(h)
            self._expand_btn.setText("▲  Show less")
        else:
            self._collapsed = True
            self.te.setFixedHeight(self._COLLAPSE_PX)
            self._expand_btn.setText("▼  Show more")

    def finalize(self):
        raw = self._text or self.te.toPlainText()
        if raw.strip():
            self._render_html(raw)

    def append_text(self, text: str):
        self._text += text
        self._pending_txt += text
        if not self._flush_timer.isActive():
            self._flush_timer.start(40)

    def _flush_pending(self):
        if not self._pending_txt:
            return
        cur = self.te.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.insertText(self._pending_txt)
        self._pending_txt = ""
        self.te.setTextCursor(cur)
        self._fit()

    def _fit(self):
        h = int(self.te.document().size().height()) + 6
        self.te.setFixedHeight(max(h, 22))

    def _copy_all(self):
        QApplication.clipboard().setText(self._text)

    @property
    def full_text(self) -> str:
        return self._text if self._text else self.te.toPlainText()

class ThinkingBlock(QWidget):
    def __init__(self, total_chunks: int):
        super().__init__()
        self._total   = total_chunks
        self._done    = 0
        self._entries: List[str] = []

        root = QVBoxLayout()
        root.setContentsMargins(16, 4, 16, 4)
        root.setSpacing(0)

        self._toggle_btn = QPushButton(f"▶  🧠 Thinking…  (0 / {total_chunks} sections)")
        self._toggle_btn.setStyleSheet(
            f"QPushButton{{background:rgba(124,58,237,0.12);color:{C['acc2']};"
            f"border:1px solid rgba(167,139,250,0.25);border-radius:8px;"
            f"padding:6px 14px;text-align:left;font-size:12px;font-weight:500;}}"
            f"QPushButton:hover{{background:rgba(124,58,237,0.22);"
            f"border-color:rgba(167,139,250,0.45);}}"
        )
        self._toggle_btn.clicked.connect(self._toggle)
        root.addWidget(self._toggle_btn)

        self._content_frame = QFrame()
        self._content_frame.setStyleSheet(
            f"QFrame{{background:rgba(10,10,24,0.92);"
            f"border:1px solid rgba(167,139,250,0.18);"
            f"border-top:none;border-radius:0 0 8px 8px;padding:4px 0;}}"
        )
        cf_layout = QVBoxLayout()
        cf_layout.setContentsMargins(0, 0, 0, 0)
        cf_layout.setSpacing(0)

        self._te = QTextEdit()
        self._te.setReadOnly(True)
        self._te.setFont(QFont("Consolas", 10))
        self._te.setFixedHeight(180)
        self._te.setStyleSheet(
            f"QTextEdit{{background:transparent;color:{C['txt2']};"
            f"border:none;padding:10px;font-size:11px;line-height:1.5;}}"
        )
        cf_layout.addWidget(self._te)
        self._content_frame.setLayout(cf_layout)
        self._content_frame.setVisible(False)
        root.addWidget(self._content_frame)
        self.setLayout(root)

    def add_section(self, num: int, total: int, chunk_text: str, summary: str):
        self._done = num
        preview  = chunk_text[:300].replace("\n", " ")
        if len(chunk_text) > 300: preview += "…"
        entry = (
            f"─── Section {num} / {total} ───\n"
            f"  📥 Input preview:  {preview}\n"
            f"  📝 Summary:        {summary[:400].strip()}"
            + ("…" if len(summary) > 400 else "") + "\n"
        )
        self._entries.append(entry)
        self._te.append(entry)
        self._te.moveCursor(QTextCursor.MoveOperation.End)
        self._toggle_btn.setText(f"▶  🧠 Thinking…  ({num} / {total} sections complete)")

    def add_phase(self, label: str):
        msg = f"\n═══ {label} ═══\n"
        self._entries.append(msg)
        self._te.append(f'<span style="color:{C["acc"]}">{msg}</span>')

    def mark_done(self):
        self._toggle_btn.setText(f"▼  ✅ Thinking complete  ({self._done} / {self._total} sections)")
        self._toggle_btn.setStyleSheet(
            f"QPushButton{{background:rgba(52,211,153,0.1);color:{C['ok']};"
            f"border:1px solid rgba(52,211,153,0.25);border-radius:8px;"
            f"padding:6px 14px;text-align:left;font-size:12px;font-weight:500;}}"
            f"QPushButton:hover{{background:rgba(52,211,153,0.18);}}"
        )
        self._content_frame.setVisible(True)

    def _toggle(self):
        vis = not self._content_frame.isVisible()
        self._content_frame.setVisible(vis)
        label = self._toggle_btn.text()
        if label.startswith("▶"):
            self._toggle_btn.setText(label.replace("▶", "▼", 1))
        else:
            self._toggle_btn.setText(label.replace("▼", "▶", 1))

_SCRIPT_EXTENSIONS_FILTER = (
    "Source files ("
    + " ".join(f"*{ext}" for ext in SCRIPT_LANGUAGES.keys())
    + ");;All Files (*)"
)


class ReferencePanelV2(QWidget):
    """
    Collapsible reference sidebar v2:
      • 📄 Documents tab  — PDF, text, .py plain (unchanged)
      • 💻 Scripts tab    — any source file with full AST parsing
      • Multi-PDF summarize button
    """
    refs_changed = pyqtSignal()

    def __init__(self, session_id: str,
                 # these are injected from the main app context:
                 get_ref_store_fn,
                 ram_watchdog_fn,
                 ram_mb_fn,
                 has_pdf: bool = False,
                 pdf_reader_cls=None,
                 parent=None):
        super().__init__(parent)
        self.session_id       = session_id
        self._get_store       = get_ref_store_fn
        self._ram_watchdog    = ram_watchdog_fn
        self._ram_free_mb     = ram_mb_fn
        self._has_pdf         = has_pdf
        self._PdfReader       = pdf_reader_cls
        self._building        = False

        self.setMaximumWidth(300)
        self.setMinimumWidth(240)
        self.setStyleSheet(
            f"background:rgba(10,10,22,0.97);"
            f"border-left:1px solid {C['bdr']};")

        root = QVBoxLayout()
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(7)

        # ── Header ───────────────────────────────────────────────────────────
        hdr_row = QHBoxLayout(); hdr_row.setSpacing(6)
        hdr = QLabel("📎  References")
        hdr.setStyleSheet(
            f"color:{C['txt']};font-weight:700;font-size:12px;")
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()
        self.ram_badge = QLabel("")
        self.ram_badge.setStyleSheet(
            f"color:{C['warn']};font-size:9px;padding:1px 5px;"
            f"background:rgba(251,191,36,0.1);border-radius:3px;")
        hdr_row.addWidget(self.ram_badge)
        root.addLayout(hdr_row)

        # ── Tab widget (Docs / Scripts) ───────────────────────────────────────
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            f"QTabWidget::pane{{background:{C['bg1']};border:none;border-radius:6px;}}"
            f"QTabBar::tab{{background:{C['bg2']};color:{C['txt2']};"
            f"padding:5px 12px;border-radius:4px 4px 0 0;font-size:10px;}}"
            f"QTabBar::tab:selected{{color:{C['txt']};background:{C['bg1']};"
            f"border-bottom:2px solid {C['acc']};font-weight:600;}}"
        )
        root.addWidget(self.tab_widget, 1)

        # ── Docs tab ─────────────────────────────────────────────────────────
        docs_widget = QWidget()
        docs_l = QVBoxLayout()
        docs_l.setContentsMargins(6, 6, 6, 6); docs_l.setSpacing(5)

        doc_btn_row = QHBoxLayout(); doc_btn_row.setSpacing(5)
        self.add_pdf_btn = QPushButton("＋ PDF")
        self.add_py_btn  = QPushButton("＋ .py")
        self.add_txt_btn = QPushButton("＋ Text")
        for b in (self.add_pdf_btn, self.add_py_btn, self.add_txt_btn):
            b.setFixedHeight(24)
            b.setStyleSheet(self._mini_btn_style(C["acc"]))
        self.add_pdf_btn.clicked.connect(lambda: self._add_doc("pdf"))
        self.add_py_btn.clicked.connect(lambda: self._add_doc("python"))
        self.add_txt_btn.clicked.connect(lambda: self._add_doc("text"))
        doc_btn_row.addWidget(self.add_pdf_btn)
        doc_btn_row.addWidget(self.add_py_btn)
        doc_btn_row.addWidget(self.add_txt_btn)
        docs_l.addLayout(doc_btn_row)

        self.multi_pdf_btn = QPushButton("📚  Summarize Multiple PDFs")
        self.multi_pdf_btn.setFixedHeight(26)
        self.multi_pdf_btn.setStyleSheet(self._mini_btn_style(C["ok"]))
        self.multi_pdf_btn.clicked.connect(self._multi_pdf_requested)
        docs_l.addWidget(self.multi_pdf_btn)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};")
        docs_l.addWidget(sep1)

        self.doc_list = QListWidget()
        self.doc_list.setStyleSheet(self._list_style())
        self.doc_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.doc_list.customContextMenuRequested.connect(
            lambda pos: self._ctx_menu(pos, self.doc_list))
        docs_l.addWidget(self.doc_list, 1)
        docs_widget.setLayout(docs_l)
        self.tab_widget.addTab(docs_widget, "📄 Docs")

        # ── Scripts tab ───────────────────────────────────────────────────────
        scripts_widget = QWidget()
        scripts_l = QVBoxLayout()
        scripts_l.setContentsMargins(6, 6, 6, 6); scripts_l.setSpacing(5)

        scr_btn_row = QHBoxLayout(); scr_btn_row.setSpacing(5)
        self.add_script_btn = QPushButton("＋ Add Script")
        self.add_script_btn.setFixedHeight(24)
        self.add_script_btn.setStyleSheet(self._mini_btn_style(C["pipeline"]))
        self.add_script_btn.clicked.connect(self._add_script)
        scr_btn_row.addWidget(self.add_script_btn)
        scr_btn_row.addStretch()
        scripts_l.addLayout(scr_btn_row)

        # Script info banner
        info = QLabel(
            "Scripts are parsed into structured indexes.\n"
            "Functions, classes, imports & types are extracted\n"
            "and injected as rich context for coding prompts.")
        info.setWordWrap(True)
        info.setStyleSheet(
            f"color:{C['txt2']};font-size:9px;padding:4px 6px;"
            f"background:{C['bg2']};border-radius:4px;")
        scripts_l.addWidget(info)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};")
        scripts_l.addWidget(sep2)

        self.script_list = QListWidget()
        self.script_list.setStyleSheet(self._list_style())
        self.script_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.script_list.customContextMenuRequested.connect(
            lambda pos: self._ctx_menu(pos, self.script_list))
        self.script_list.currentItemChanged.connect(
            self._on_script_selected)
        scripts_l.addWidget(self.script_list, 1)

        # Script detail pane
        self.script_detail = QLabel("")
        self.script_detail.setWordWrap(True)
        self.script_detail.setTextFormat(Qt.TextFormat.RichText)
        self.script_detail.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.script_detail.setStyleSheet(
            f"color:{C['txt2']};font-size:9px;padding:6px 8px;"
            f"background:{C['bg2']};border:1px solid {C['bdr']};"
            f"border-radius:6px;min-height:60px;")
        scripts_l.addWidget(self.script_detail)

        scripts_widget.setLayout(scripts_l)
        self.tab_widget.addTab(scripts_widget, "💻 Scripts")

        # ── RAM info ─────────────────────────────────────────────────────────
        self.ram_info = QLabel("")
        self.ram_info.setWordWrap(True)
        self.ram_info.setStyleSheet(
            f"color:{C['txt2']};font-size:9px;padding:2px;")
        root.addWidget(self.ram_info)

        self.setLayout(root)

        self._ram_timer = QTimer(self)
        self._ram_timer.timeout.connect(self._update_ram_badge)
        self._ram_timer.start(3000)
        self._refresh()

    # ── Style helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _mini_btn_style(color: str) -> str:
        # Derive a soft tint from whatever accent color is passed in
        tint  = color.lstrip("#")
        r,g,b = int(tint[0:2],16), int(tint[2:4],16), int(tint[4:6],16)
        return (
            f"QPushButton{{background:rgba({r},{g},{b},0.12);color:{color};"
            f"border:1px solid rgba({r},{g},{b},0.30);border-radius:5px;"
            f"font-size:10px;padding:0 6px;}}"
            f"QPushButton:hover{{background:rgba({r},{g},{b},0.28);}}")

    @staticmethod
    def _list_style() -> str:
        return (
            f"QListWidget{{background:transparent;border:none;"
            f"font-size:10px;outline:none;}}"
            f"QListWidget::item{{padding:5px 8px;border-radius:5px;"
            f"margin:2px 0;min-height:18px;}}"
            f"QListWidget::item:hover{{background:rgba(167,139,250,0.08);}}"
            f"QListWidget::item:selected{{background:rgba(124,58,237,0.22);"
            f"color:{C['acc2']};border:1px solid rgba(167,139,250,0.2);}}")

    # ── Store access ──────────────────────────────────────────────────────────
    @property
    def _store(self):
        return self._get_store(self.session_id)

    def update_session(self, session_id: str):
        self.session_id = session_id
        self._refresh()

    # ── Refresh both lists ────────────────────────────────────────────────────
    def _refresh(self):
        self.doc_list.clear()
        self.script_list.clear()
        store = self._store
        for ref in store._refs.values():
            if isinstance(ref, ScriptSmartReference):
                self._add_script_list_item(ref)
            else:
                self._add_doc_list_item(ref)

    def _add_doc_list_item(self, ref):
        icon = {"pdf": "📄", "python": "🐍", "text": "📝"}.get(
            getattr(ref, "ftype", ""), "📎")
        item = QListWidgetItem(f"{icon}  {ref.name}")
        item.setData(Qt.ItemDataRole.UserRole, ref.ref_id)
        item.setToolTip(ref.full_text_preview())
        item.setForeground(QColor(C["txt"]))
        self.doc_list.addItem(item)

    def _add_script_list_item(self, ref: ScriptSmartReference):
        ps  = ref.parsed
        fn  = len(ps.functions)
        cls = len(ps.classes)
        tag = f"  [{ps.language} · {fn}fn · {cls}cl]"
        item = QListWidgetItem(f"💻  {ref.name}{tag}")
        item.setData(Qt.ItemDataRole.UserRole, ref.ref_id)
        item.setToolTip(ref.full_text_preview())
        item.setForeground(QColor(C["pipeline"]))
        self.script_list.addItem(item)

    # ── Context menu (remove) ─────────────────────────────────────────────────
    def _ctx_menu(self, pos, list_widget: QListWidget):
        from PyQt6.QtWidgets import QMenu
        item = list_widget.itemAt(pos)
        if not item:
            return
        ref_id = item.data(Qt.ItemDataRole.UserRole)
        menu   = QMenu(self)
        act_rm = menu.addAction("🗑  Remove Reference")
        chosen = menu.exec(list_widget.mapToGlobal(pos))
        if chosen == act_rm:
            self._store.remove(ref_id)
            self._refresh()
            self.refs_changed.emit()

    # ── Script selection → detail pane ────────────────────────────────────────
    def _on_script_selected(self, item, _=None):
        if not item:
            return
        ref_id = item.data(Qt.ItemDataRole.UserRole)
        ref    = self._store._refs.get(ref_id)
        if not isinstance(ref, ScriptSmartReference):
            return
        ps   = ref.parsed
        detail = (
            f"<b>{ref.name}</b>  [{ps.language}]<br>"
            f"Functions: {', '.join(f.name for f in ps.functions[:10])}"
            + ("…" if len(ps.functions) > 10 else "")
            + f"<br>Classes: {', '.join(c.name for c in ps.classes[:8])}"
            + ("…" if len(ps.classes) > 8 else "")
            + f"<br>Imports: {len(ps.imports)}"
            + (f"<br>Types: {len(ps.types)}" if ps.types else "")
            + (f"<br><span style='color:{C['err']};'>"
               f"Errors: {'; '.join(ps.errors[:2])}</span>"
               if ps.errors else "")
        )
        self.script_detail.setText(detail)

    # ── Add document ──────────────────────────────────────────────────────────
    def _add_doc(self, ftype: str):
        if ftype == "pdf" and not self._has_pdf:
            QMessageBox.warning(
                self, "Missing Dep", "Install PyPDF2: pip install PyPDF2")
            return
        filters = {
            "pdf":    "PDF Files (*.pdf)",
            "python": "Python Files (*.py)",
            "text":   "Text/Markdown (*.txt *.md *.rst)",
        }
        path, _ = QFileDialog.getOpenFileName(
            self, f"Add {ftype.upper()} Reference",
            str(Path.home()), filters.get(ftype, "All Files (*)"))
        if not path:
            return
        name = Path(path).name
        if ftype == "pdf":
            reader = self._PdfReader(path)
            raw = "\n".join(
                page.extract_text() or "" for page in reader.pages)
        else:
            try:
                raw = Path(path).read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                QMessageBox.warning(self, "Read Error", str(e)); return
        if not raw.strip():
            QMessageBox.warning(self, "Empty", "No text extracted."); return
        self._store.add_reference(name, ftype, raw)
        self._refresh()
        self.refs_changed.emit()

    # ── Add script (with parser) ──────────────────────────────────────────────
    def _add_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Add Source Code Script",
            str(Path.home()), _SCRIPT_EXTENSIONS_FILTER)
        if not path:
            return
        name = Path(path).name
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            QMessageBox.warning(self, "Read Error", str(e)); return
        if not raw.strip():
            QMessageBox.warning(self, "Empty File", "File has no content."); return

        # RAM check before parse
        self._ram_watchdog(self.session_id)

        ref = _add_script_reference(self._store, name, raw)
        self._refresh()
        self.refs_changed.emit()

        ps = ref.parsed
        # Flash parse result
        QMessageBox.information(
            self, "Script Parsed ✅",
            f"Parsed {name}  [{ps.language}]\n\n"
            + ps.summary_header()
            + ("\n\nErrors:\n" + "\n".join(ps.errors) if ps.errors else ""))

    # ── Multi-PDF ─────────────────────────────────────────────────────────────
    def _multi_pdf_requested(self):
        if not self._has_pdf:
            QMessageBox.warning(
                self, "Missing Dep", "Install PyPDF2: pip install PyPDF2")
            return
        from PyQt6.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Multiple PDFs",
            str(Path.home()), "PDF Files (*.pdf)")
        if not paths:
            return
        self._pending_multi_pdfs = paths
        self.refs_changed.emit()

    # ── RAM badge ─────────────────────────────────────────────────────────────
    def _update_ram_badge(self):
        free_mb = self._ram_free_mb()
        # reuse ram_watchdog_mb constant from main
        try:
            threshold = RAM_WATCHDOG_MB()
        except Exception:
            threshold = 800
        if free_mb < threshold:
            self.ram_badge.setText(f"⚠️ {free_mb:.0f}MB free")
            self.ram_badge.setVisible(True)
        else:
            self.ram_badge.setVisible(False)

        store   = self._store
        n_refs  = len(store._refs)
        n_hot   = sum(len(getattr(r, "_hot", {})) for r in store._refs.values())
        n_total = sum(len(getattr(r, "_chunks", [1])) for r in store._refs.values())
        if n_refs > 0:
            self.ram_info.setText(
                f"{n_refs} ref(s) · {n_hot}/{n_total} chunks hot · "
                f"{free_mb:.0f} MB free RAM")
        else:
            self.ram_info.setText(f"{free_mb:.0f} MB free RAM")

    # ── Context for prompt ────────────────────────────────────────────────────
    def get_context_for(self, query: str) -> str:
        return _build_context_block_extended(self._store, query)

class ReferencePanel(QWidget):
    """
    Compact collapsible reference sidebar for a chat session.
    Shows attached files with remove buttons and a RAM status badge.
    """
    refs_changed = pyqtSignal()

    def __init__(self, session_id: str, parent=None):
        super().__init__(parent)
        self.session_id = session_id
        self._store     = get_ref_store(session_id)
        self._building  = False
        self.setMaximumWidth(280)
        self.setMinimumWidth(220)
        self.setStyleSheet(
            f"background:rgba(10,10,22,0.97);"
            f"border-left:1px solid {C['bdr']};"
        )
        root = QVBoxLayout()
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(7)

        # Header
        hdr_row = QHBoxLayout(); hdr_row.setSpacing(6)
        hdr = QLabel("📎  References")
        hdr.setStyleSheet(f"color:{C['txt']};font-weight:700;font-size:12px;")
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()

        self.ram_badge = QLabel("")
        self.ram_badge.setStyleSheet(
            f"color:{C['warn']};font-size:9px;padding:1px 5px;"
            f"background:rgba(251,191,36,0.1);border-radius:3px;"
        )
        hdr_row.addWidget(self.ram_badge)
        root.addLayout(hdr_row)

        # Add buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(6)
        self.add_pdf_btn = QPushButton("＋ PDF")
        self.add_py_btn  = QPushButton("＋ .py")
        self.add_txt_btn = QPushButton("＋ Text")
        for b in (self.add_pdf_btn, self.add_py_btn, self.add_txt_btn):
            b.setFixedHeight(26)
            b.setStyleSheet(
                f"QPushButton{{background:rgba(124,58,237,0.15);color:{C['acc']};"
                f"border:1px solid rgba(167,139,250,0.25);border-radius:6px;"
                f"font-size:10px;padding:0 6px;}}"
                f"QPushButton:hover{{background:rgba(124,58,237,0.3);color:{C['acc2']};}}"
            )
        self.add_pdf_btn.clicked.connect(lambda: self._add_file("pdf"))
        self.add_py_btn.clicked.connect(lambda: self._add_file("python"))
        self.add_txt_btn.clicked.connect(lambda: self._add_file("text"))
        btn_row.addWidget(self.add_pdf_btn)
        btn_row.addWidget(self.add_py_btn)
        btn_row.addWidget(self.add_txt_btn)
        root.addLayout(btn_row)

        # Multi-PDF summarize button
        self.multi_pdf_btn = QPushButton("📚  Summarize Multiple PDFs")
        self.multi_pdf_btn.setFixedHeight(28)
        self.multi_pdf_btn.setStyleSheet(
            f"QPushButton{{background:rgba(52,211,153,0.1);color:{C['ok']};"
            f"border:1px solid rgba(52,211,153,0.25);border-radius:6px;"
            f"font-size:10px;padding:0 8px;}}"
            f"QPushButton:hover{{background:rgba(52,211,153,0.22);}}"
        )
        self.multi_pdf_btn.clicked.connect(self._multi_pdf_requested)
        root.addWidget(self.multi_pdf_btn)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};")
        root.addWidget(sep)

        # Reference list
        self.ref_list = QListWidget()
        self.ref_list.setStyleSheet(
            f"QListWidget{{background:transparent;border:none;font-size:10px;outline:none;}}"
            f"QListWidget::item{{padding:4px 6px;border-radius:4px;margin:1px 0;}}"
            f"QListWidget::item:selected{{background:rgba(124,58,237,0.25);color:{C['acc2']};}}"
        )
        self.ref_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.ref_list.customContextMenuRequested.connect(self._ref_ctx_menu)
        root.addWidget(self.ref_list, 1)

        # RAM info
        self.ram_info = QLabel("")
        self.ram_info.setWordWrap(True)
        self.ram_info.setStyleSheet(f"color:{C['txt2']};font-size:9px;padding:2px;")
        root.addWidget(self.ram_info)

        self.setLayout(root)
        self._ram_timer = QTimer(self)
        self._ram_timer.timeout.connect(self._update_ram_badge)
        self._ram_timer.start(3000)
        self._refresh()

    def update_session(self, session_id: str):
        self.session_id = session_id
        self._store     = get_ref_store(session_id)
        self._refresh()

    def _refresh(self):
        self.ref_list.clear()
        for ref in self._store.refs.values():
            icon  = {"pdf": "📄", "python": "🐍", "text": "📝"}.get(ref.ftype, "📎")
            n_hot = len(ref._hot)
            n_tot = len(ref._chunks)
            ram_tag = f"  [{n_hot}/{n_tot} chunks in RAM]"
            item  = QListWidgetItem(f"{icon}  {ref.name}{ram_tag}")
            item.setData(Qt.ItemDataRole.UserRole, ref.ref_id)
            item.setToolTip(ref.full_text_preview())
            item.setForeground(QColor(C["txt"]))
            self.ref_list.addItem(item)

    def _ref_ctx_menu(self, pos):
        item = self.ref_list.itemAt(pos)
        if not item: return
        ref_id = item.data(Qt.ItemDataRole.UserRole)
        menu   = QMenu(self)
        act_rm = menu.addAction("🗑  Remove Reference")
        chosen = menu.exec(self.ref_list.mapToGlobal(pos))
        if chosen == act_rm:
            self._store.remove(ref_id)
            self._refresh()
            self.refs_changed.emit()

    def _add_file(self, ftype: str):
        if not HAS_PDF and ftype == "pdf":
            QMessageBox.warning(self, "Missing Dep", "Install PyPDF2: pip install PyPDF2")
            return
        filters = {"pdf": "PDF Files (*.pdf)", "python": "Python Files (*.py)",
                   "text": "Text/Markdown (*.txt *.md *.rst)"}
        path, _ = QFileDialog.getOpenFileName(
            self, f"Add {ftype.upper()} Reference",
            str(Path.home()), filters.get(ftype, "All Files (*)"))
        if not path: return

        filename = Path(path).name
        if ftype == "pdf":
            if not HAS_PDF: return
            reader = PdfReader(path)
            raw = "\n".join(
                page.extract_text() or "" for page in reader.pages)
        else:
            try:
                raw = Path(path).read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                QMessageBox.warning(self, "Read Error", str(e)); return

        if not raw.strip():
            QMessageBox.warning(self, "Empty File", "No text extracted."); return

        # RAM watchdog before adding
        if _ram_free_mb() < RAM_WATCHDOG_MB():
            self._store.flush_ram()

        ref = self._store.add_reference(filename, ftype, raw)
        self._refresh()
        self.refs_changed.emit()

    def _multi_pdf_requested(self):
        """Open multi-file picker and emit signal with paths."""
        if not HAS_PDF:
            QMessageBox.warning(self, "Missing Dep", "Install PyPDF2: pip install PyPDF2")
            return
        # Use a custom dialog to pick multiple PDFs
        from PyQt6.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Multiple PDFs for Bulk Summarization",
            str(Path.home()), "PDF Files (*.pdf)")
        if not paths: return
        # Store picked paths temporarily for MainWindow to pick up
        self._pending_multi_pdfs = paths
        self.refs_changed.emit()   # MainWindow listens and handles

    def _update_ram_badge(self):
        free_mb = _ram_free_mb()
        if free_mb < RAM_WATCHDOG_MB():
            self.ram_badge.setText(f"⚠️ RAM: {free_mb:.0f}MB free")
            self.ram_badge.setVisible(True)
        else:
            self.ram_badge.setVisible(False)
        n_refs  = len(self._store.refs)
        n_hot   = sum(len(r._hot) for r in self._store.refs.values())
        n_total = sum(len(r._chunks) for r in self._store.refs.values())
        if n_refs > 0:
            self.ram_info.setText(
                f"{n_refs} ref(s) · {n_hot}/{n_total} chunks hot · "
                f"{free_mb:.0f} MB free RAM")
        else:
            self.ram_info.setText(f"{free_mb:.0f} MB free RAM")

    def get_context_for(self, query: str) -> str:
        return self._store.build_context_block(query)

class ChatArea(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(
            f"QScrollArea{{background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {C['bg0']},stop:1 #090914);border:none;}}"
        )
        self._container = QWidget()
        self._container.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {C['bg0']},stop:1 #090914);"
        )
        self._vbox = QVBoxLayout()
        self._vbox.setContentsMargins(0, 12, 0, 12)
        self._vbox.setSpacing(2)
        self._vbox.addStretch()
        self._container.setLayout(self._vbox)
        self.setWidget(self._container)
        self._widgets: List[QWidget] = []

    def add_message(self, role: str, content: str, timestamp: str,
                    tag: str = "") -> MessageWidget:
        w = MessageWidget(role, content, timestamp, tag=tag)
        self._vbox.insertWidget(self._vbox.count() - 1, w)
        self._widgets.append(w)
        QTimer.singleShot(30, self._scroll_bottom)
        return w

    def add_thinking_block(self, total_chunks: int) -> ThinkingBlock:
        tb = ThinkingBlock(total_chunks)
        self._vbox.insertWidget(self._vbox.count() - 1, tb)
        self._widgets.append(tb)
        QTimer.singleShot(30, self._scroll_bottom)
        return tb

    def add_pipeline_divider(self, label: str):
        """Insert a visual pipeline stage divider."""
        lbl = QLabel(f"  ⬇  {label}  ⬇")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color:{C['pipeline']};font-size:11px;padding:4px 0;"
            f"background:rgba(34,211,238,0.06);border-radius:6px;margin:2px 60px;"
        )
        self._vbox.insertWidget(self._vbox.count() - 1, lbl)
        self._widgets.append(lbl)

    def clear_messages(self):
        for w in self._widgets:
            self._vbox.removeWidget(w)
            w.deleteLater()
        self._widgets.clear()

    def _scroll_bottom(self):
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class ChatModule(QWidget):
    """
    Self-contained chat module: ChatArea + ReferencePanel + InputBar.
    Modular — can be embedded anywhere or swapped out.
    """

    send_requested  = pyqtSignal(str, str)   # (text, ref_context)
    stop_requested  = pyqtSignal()
    pdf_requested   = pyqtSignal()
    clear_requested = pyqtSignal()
    multi_pdf_requested = pyqtSignal(list)   # list of paths

    def __init__(self, session_id: str = "", parent=None):
        super().__init__(parent)
        self._session_id  = session_id
        self._refs_visible = False

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar with ref toggle
        topbar = QHBoxLayout()
        topbar.setContentsMargins(8, 4, 8, 0)
        topbar.setSpacing(6)
        topbar.addStretch()
        self.ref_toggle_btn = QPushButton("📎  References  (0)")
        self.ref_toggle_btn.setFixedHeight(26)
        self.ref_toggle_btn.setCheckable(True)
        self.ref_toggle_btn.setStyleSheet(
            f"QPushButton{{background:rgba(124,58,237,0.1);color:{C['txt2']};"
            f"border:1px solid rgba(167,139,250,0.2);border-radius:6px;"
            f"font-size:10px;padding:0 10px;}}"
            f"QPushButton:checked{{background:rgba(124,58,237,0.25);color:{C['acc']};"
            f"border-color:rgba(167,139,250,0.5);font-weight:600;}}"
            f"QPushButton:hover{{color:{C['acc2']};}}"
        )
        self.ref_toggle_btn.clicked.connect(self._toggle_refs)
        topbar.addWidget(self.ref_toggle_btn)
        root.addLayout(topbar)

        # Main splitter: chat | ref panel
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(2)

        self.chat_area = ChatArea()
        self.main_splitter.addWidget(self.chat_area)

        self.ref_panel = ReferencePanelV2(
            session_id or "default",
            get_ref_store_fn = get_ref_store,
            ram_watchdog_fn  = lambda sid: RamWatchdog.check_and_spill(sid),
            ram_mb_fn   = _ram_free_mb,
            has_pdf          = HAS_PDF,
            pdf_reader_cls   = PdfReader if HAS_PDF else None,
        )
        self.ref_panel.refs_changed.connect(self._on_refs_changed)
        self.ref_panel.setVisible(False)
        self.main_splitter.addWidget(self.ref_panel)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)

        root.addWidget(self.main_splitter, 1)

        self.input_bar = InputBar()
        self.input_bar.send_requested.connect(self._on_send)
        self.input_bar.stop_requested.connect(self.stop_requested)
        self.input_bar.pdf_requested.connect(self.pdf_requested)
        self.input_bar.clear_requested.connect(self.clear_requested)
        root.addWidget(self.input_bar, 0)

        self.setLayout(root)

    def set_session(self, session_id: str):
        self._session_id = session_id
        self.ref_panel.update_session(session_id)
        self._update_ref_badge()

    def _toggle_refs(self, checked: bool):
        self._refs_visible = checked
        self.ref_panel.setVisible(checked)
        if checked:
            self.main_splitter.setSizes([800, 260])
        else:
            self.main_splitter.setSizes([1, 0])

    def _on_send(self, text: str):
        ref_ctx = self.ref_panel.get_context_for(text)
        self.send_requested.emit(text, ref_ctx)

    def _on_refs_changed(self):
        self._update_ref_badge()
        # Check if multi-PDF was triggered
        if hasattr(self.ref_panel, "_pending_multi_pdfs"):
            paths = self.ref_panel._pending_multi_pdfs
            del self.ref_panel._pending_multi_pdfs
            self.multi_pdf_requested.emit(paths)

    def _update_ref_badge(self):
        n = len(self.ref_panel._store.refs)
        self.ref_toggle_btn.setText(f"📎  References  ({n})")

    # Proxy properties
    @property
    def selected_model(self) -> str:
        return self.input_bar.selected_model

    def set_generating(self, active: bool):
        self.input_bar.set_generating(active)

    def set_pipeline_mode(self, active: bool):
        self.input_bar.set_pipeline_mode(active)

class InputBar(QWidget):
    send_requested  = pyqtSignal(str)
    stop_requested  = pyqtSignal()
    pdf_requested   = pyqtSignal()
    clear_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 rgba(13,13,28,0.98),stop:1 rgba(10,10,22,0.99));"
            f"border-top:1px solid {C['bdr']};"
        )
        root = QVBoxLayout()
        root.setContentsMargins(14, 8, 14, 10)
        root.setSpacing(7)

        toolbar = QHBoxLayout(); toolbar.setSpacing(8)
        model_lbl = QLabel("Model:")
        model_lbl.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        toolbar.addWidget(model_lbl)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(240)
        self._populate_models()
        toolbar.addWidget(self.model_combo)

        # Family info badge
        self.family_badge = QLabel("")
        self.family_badge.setStyleSheet(
            f"color:{C['acc2']};font-size:10px;"
            f"background:{C['bg2']};border-radius:4px;padding:2px 7px;"
        )
        toolbar.addWidget(self.family_badge)
        self.model_combo.currentIndexChanged.connect(self._update_family_badge)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        toolbar.addStretch()

        self.pdf_btn   = QPushButton("📄  PDF")
        self.clear_btn = QPushButton("🗑  Clear")
        for b in (self.pdf_btn, self.clear_btn):
            b.setFixedWidth(86); b.setFixedHeight(30)
        self.pdf_btn.clicked.connect(self.pdf_requested)
        self.clear_btn.clicked.connect(self.clear_requested)
        toolbar.addWidget(self.pdf_btn)
        toolbar.addWidget(self.clear_btn)

        self.code_btn = QPushButton("💻  Code")
        self.code_btn.setFixedSize(80, 30)
        self.code_btn.setCheckable(True)
        self.code_btn.setToolTip(
            "Force coding engine for this message\n"
            "(auto-detects code keywords even when off)")
        self.code_btn.setStyleSheet(
            f"QPushButton{{background:rgba(19,19,42,0.8);color:{C['txt2']};"
            f"border:1px solid {C['bdr']};border-radius:8px;padding:4px 8px;}}"
            f"QPushButton:checked{{background:rgba(124,58,237,0.25);color:{C['acc2']};"
            f"border-color:rgba(167,139,250,0.5);font-weight:600;}}"
            f"QPushButton:hover{{background:rgba(124,58,237,0.18);color:{C['acc']};"
            f"border-color:rgba(167,139,250,0.3);}}"
        )
        toolbar.addWidget(self.code_btn)

        # Pipeline mode indicator
        self.pipeline_badge = QLabel("")
        self.pipeline_badge.setStyleSheet(
            f"color:{C['pipeline']};font-size:10px;padding:2px 7px;"
            f"background:rgba(34,211,238,0.1);border-radius:4px;"
            f"border:1px solid rgba(34,211,238,0.2);"
        )
        self.pipeline_badge.setVisible(False)
        toolbar.addWidget(self.pipeline_badge)

        root.addLayout(toolbar)

        row = QHBoxLayout(); row.setSpacing(10)
        self.input = QTextEdit()
        self.input.setPlaceholderText("Type a message…  (Enter = send · Shift+Enter = newline)")
        self.input.setMaximumHeight(108)
        self.input.setMinimumHeight(52)
        self.input.setStyleSheet(
            f"QTextEdit{{background:rgba(19,19,42,0.85);color:{C['txt']};"
            f"border:1px solid {C['bdr']};border-radius:12px;"
            f"padding:10px 12px;font-size:13px;line-height:1.5;}}"
            f"QTextEdit:focus{{border-color:rgba(167,139,250,0.55);"
            f"background:rgba(26,26,53,0.95);}}"
        )
        self.input.installEventFilter(self)

        self.send_btn = QPushButton("Send ➤")
        self.send_btn.setObjectName("btn_send")
        self.send_btn.setFixedSize(90, 52)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.clicked.connect(self._emit_send)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setObjectName("btn_stop")
        self.stop_btn.setFixedSize(90, 52)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self.stop_requested)

        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        row.addWidget(self.stop_btn)
        root.addLayout(row)
        self.setLayout(root)
        self._update_family_badge()

    def _populate_models(self):
        self.model_combo.clear()
        for m in MODEL_REGISTRY.all_models():
            self.model_combo.addItem(m["name"], m["path"])
        if self.model_combo.count() == 0:
            self.model_combo.addItem(DEFAULT_MODEL, str(MODELS_DIR / DEFAULT_MODEL))

    def _update_family_badge(self):
        path = self.model_combo.currentData() or ""
        if path:
            fam   = detect_model_family(path)
            quant = detect_quant_type(path)
            ql, _ = quant_info(quant)
            self.family_badge.setText(f"{fam.name}  ·  {quant}  ·  {ql}")
        else:
            self.family_badge.setText("")

    def _on_model_combo_changed(self, _index: int):
        # Walk up to MainWindow and trigger primary model reload
        w = self.parent()
        while w and not hasattr(w, "_start_model_load"):
            w = w.parent()
        if w and hasattr(w, "_start_model_load"):
            # Small delay so UI settles before reload
            QTimer.singleShot(300, w._start_model_load)

    def set_pipeline_mode(self, active: bool):
        active = bool(active)
        if not hasattr(self, "pipeline_badge") or self.pipeline_badge is None:
            return
        self.pipeline_badge.setVisible(active)
        if active:
            self.pipeline_badge.setText("🔗 Pipeline Mode")

    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == QEvent.Type.KeyPress:
            if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                    and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)):
                self._emit_send()
                return True
        return super().eventFilter(obj, event)

    def _emit_send(self):
        text = self.input.toPlainText().strip()
        if text:
            self.input.clear()
            self.send_requested.emit(text)

    def set_generating(self, active: bool):
        self.send_btn.setVisible(not active)
        self.stop_btn.setVisible(active)
        self.input.setEnabled(not active)

    @property
    def selected_model(self) -> str:
        return self.model_combo.currentData() or ""


class SessionSidebar(QWidget):
    session_selected = pyqtSignal(str)
    new_session      = pyqtSignal()
    session_deleted  = pyqtSignal(str)
    session_renamed  = pyqtSignal(str, str)
    session_exported = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(180)
        self.setMaximumWidth(300)
        self.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 rgba(10,10,20,0.99),stop:1 rgba(13,13,26,0.97));"
            f"border-right:1px solid {C['bdr']};"
        )
        root = QVBoxLayout()
        root.setContentsMargins(10, 12, 10, 10)
        root.setSpacing(8)

        hdr = QLabel("💬  Chats")
        hdr.setStyleSheet(f"color:{C['txt']};font-size:14px;font-weight:700;"
                          f"letter-spacing:0.3px;padding:2px 4px;")
        root.addWidget(hdr)

        self.new_btn = QPushButton("＋  New Chat")
        self.new_btn.setObjectName("btn_new")
        self.new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.new_btn.clicked.connect(self.new_session)
        root.addWidget(self.new_btn)

        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍  Search…")
        self.search.textChanged.connect(self._redraw)
        root.addWidget(self.search)

        self.lst = QListWidget()
        self.lst.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.lst.customContextMenuRequested.connect(self._ctx_menu)
        self.lst.itemClicked.connect(self._on_click)
        root.addWidget(self.lst, 1)
        self.setLayout(root)
        self._sessions: Dict[str, Session] = {}
        self._active:   str = ""

    def refresh(self, sessions: Dict[str, Session], active_id: str = ""):
        self._sessions = sessions
        self._active   = active_id
        self._redraw()

    def set_active(self, sid: str):
        self._active = sid
        self._redraw()

    def _redraw(self, _=None):
        q = self.search.text().lower()
        self.lst.clear()
        grouped: Dict[str, List[Session]] = {}
        for s in sorted(self._sessions.values(), key=lambda x: x.id, reverse=True):
            if q and q not in s.title.lower(): continue
            grouped.setdefault(s.created, []).append(s)

        for date in sorted(grouped.keys(), reverse=True):
            di = QListWidgetItem(f"  📅  {date}")
            di.setFlags(Qt.ItemFlag.NoItemFlags)
            di.setForeground(QColor(C["txt2"]))
            f = di.font(); f.setPointSize(10); di.setFont(f)
            self.lst.addItem(di)
            for s in grouped[date]:
                title = (s.title[:34] + "…") if len(s.title) > 34 else s.title
                item  = QListWidgetItem(f"    {title}")
                item.setData(Qt.ItemDataRole.UserRole, s.id)
                if s.id == self._active:
                    item.setForeground(QColor(C["acc"]))
                    fo = item.font(); fo.setBold(True); item.setFont(fo)
                self.lst.addItem(item)

    def _on_click(self, item: QListWidgetItem):
        sid = item.data(Qt.ItemDataRole.UserRole)
        if sid: self.session_selected.emit(sid)

    def _ctx_menu(self, pos):
        item = self.lst.itemAt(pos)
        if not item: return
        sid = item.data(Qt.ItemDataRole.UserRole)
        if not sid: return
        menu = QMenu(self)
        act_rename = menu.addAction("✏️  Rename")
        act_export = menu.addAction("📤  Export Markdown")
        menu.addSeparator()
        act_del    = menu.addAction("🗑  Delete")
        chosen = menu.exec(self.lst.mapToGlobal(pos))
        if chosen == act_del:
            self.session_deleted.emit(sid)
        elif chosen == act_rename:
            text, ok = QInputDialog.getText(
                self, "Rename Session", "New title:",
                text=self._sessions[sid].title)
            if ok and text.strip():
                self.session_renamed.emit(sid, text.strip())
        elif chosen == act_export:
            self.session_exported.emit(sid)


class LogConsole(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(14, 8, 14, 6)
        lbl = QLabel("🐞  Debug Console")
        lbl.setStyleSheet(f"color:{C['txt']};font-weight:700;font-size:13px;")
        clr = QPushButton("Clear")
        clr.setFixedSize(70, 28)
        clr.clicked.connect(lambda: self.te.clear())
        toolbar.addWidget(lbl); toolbar.addStretch(); toolbar.addWidget(clr)
        self.te = QTextEdit()
        self.te.setReadOnly(True)
        self.te.setFont(QFont("Consolas", 10))
        self.te.setStyleSheet(
            f"background:rgba(7,7,15,0.98);color:#b8a8ff;border:none;padding:6px;")
        root.addLayout(toolbar)
        root.addWidget(self.te)
        self.setLayout(root)

    def log(self, level: str, msg: str):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        lc = {"INFO": C["acc"], "WARN": C["warn"], "ERROR": C["err"]}.get(level, C["txt2"])
        self.te.append(
            f'<span style="color:{C["txt2"]}">[{ts}]</span> '
            f'<span style="color:{lc}">[{level}]</span> '
            f'<span style="color:{C["txt"]}">{msg}</span>'
        )


# ═════════════════════════════ PARALLEL LOADING DIALOG ══════════════════════

# ─── Config descriptions ──────────────────────────────────────────────────────
CONFIG_FIELD_META = {
    "ram_watchdog_mb": {
        "label": "RAM Watchdog (MB)",
        "desc":  (
            "Free RAM threshold in megabytes. When available RAM drops below "
            "this value, reference chunk caches are automatically spilled to disk "
            "to prevent system memory exhaustion. Lower = more aggressive spilling. "
            "Recommended: 400–1200 MB depending on your total RAM."
        ),
        "min": 100, "max": 8000, "type": "int",
    },
    "chunk_index_size": {
        "label": "Ref Chunk Size (chars)",
        "desc":  (
            "Character size of each indexed chunk when a reference file is loaded. "
            "Smaller chunks = finer retrieval granularity but more memory entries. "
            "Larger chunks = broader context per hit but less precise. "
            "Recommended: 300–600."
        ),
        "min": 100, "max": 2000, "type": "int",
    },
    "max_ram_chunks": {
        "label": "Max RAM Chunks (per ref)",
        "desc":  (
            "Maximum number of reference chunks kept in RAM per loaded file. "
            "Chunks beyond this limit are evicted to disk. On systems with "
            "8 GB RAM, 80–120 is safe. On 4 GB, consider 40–60."
        ),
        "min": 10, "max": 500, "type": "int",
    },
    "summary_chunk_chars": {
        "label": "Summary Chunk Size (chars)",
        "desc":  (
            "Size of each text chunk sent to the model during PDF summarization. "
            "Smaller values = more chunks, less RAM per step, but more inference calls. "
            "Larger values = fewer calls but each prompt uses more context window. "
            "Recommended: 2000–4000."
        ),
        "min": 500, "max": 8000, "type": "int",
    },
    "summary_ctx_carry": {
        "label": "Summary Context Carry (chars)",
        "desc":  (
            "Number of characters from the previous chunk's summary to carry forward "
            "as context for the next chunk. Helps maintain narrative continuity "
            "across long documents. Recommended: 400–800."
        ),
        "min": 100, "max": 2000, "type": "int",
    },
    "summary_n_pred_sect": {
        "label": "Summary Tokens / Section",
        "desc":  (
            "Maximum tokens the model generates for each section summary. "
            "Higher = more detailed section summaries but slower processing. "
            "Recommended: 250–500."
        ),
        "min": 64, "max": 1024, "type": "int",
    },
    "summary_n_pred_final": {
        "label": "Summary Tokens / Final Pass",
        "desc":  (
            "Maximum tokens for the final consolidation pass that synthesizes "
            "all section summaries into one cohesive document summary. "
            "Recommended: 500–900."
        ),
        "min": 128, "max": 2048, "type": "int",
    },
    "multipdf_n_pred_sect": {
        "label": "Multi-PDF Tokens / Section",
        "desc":  (
            "Tokens per section for multi-PDF batch summarization. "
            "Same as Summary Tokens / Section but applied to each document "
            "in a multi-document batch job. Recommended: 300–500."
        ),
        "min": 64, "max": 1024, "type": "int",
    },
    "multipdf_n_pred_final": {
        "label": "Multi-PDF Tokens / Final",
        "desc":  (
            "Tokens for the final cross-document consolidation in multi-PDF mode. "
            "This pass synthesizes summaries across all loaded documents, "
            "so a higher value gives richer cross-document analysis. "
            "Recommended: 700–1200."
        ),
        "min": 128, "max": 2048, "type": "int",
    },
    "ref_top_k": {
        "label": "Reference Top-K Chunks",
        "desc":  (
            "Number of most relevant chunks retrieved from each reference file "
            "per query. Higher = more context injected but larger prompts. "
            "Lower = faster, smaller prompts. Recommended: 4–8."
        ),
        "min": 1, "max": 20, "type": "int",
    },
    "ref_max_context_chars": {
        "label": "Ref Max Context (chars)",
        "desc":  (
            "Maximum total characters injected from all references combined "
            "into each prompt. Guards against overflowing the model's context window. "
            "Recommended: 1500–4000."
        ),
        "min": 200, "max": 12000, "type": "int",
    },
    "pause_after_chunks": {
        "label": "Pause-Suggest Threshold (chunks)",
        "desc":  (
            "After this many chunks are processed, the app will show a "
            "pause/save banner if many chunks remain. Set to 0 to disable "
            "auto-pause suggestions. Recommended: 2–5."
        ),
        "min": 0, "max": 50, "type": "int",
    },
    "default_threads": {
        "label": "Default CPU Threads",
        "desc":  (
            "Number of CPU threads used by llama.cpp for inference. "
            "Setting this higher than your physical core count may reduce performance. "
            f"Your system has {_cpu_count()} logical CPUs. "
            "Recommended: physical core count (not hyperthreaded)."
        ),
        "min": 1, "max": 64, "type": "int",
    },
    "default_ctx": {
        "label": "Default Context Window (tokens)",
        "desc":  (
            "Default token context window loaded with each model. "
            "Larger context = more conversation history, more RAM. "
            "Each 4096 additional tokens uses ~0.5 GB extra RAM. "
            "Recommended: 2048–8192 for most systems."
        ),
        "min": 512, "max": 32768, "type": "int",
    },
    "default_n_predict": {
        "label": "Default Max New Tokens",
        "desc":  (
            "Default maximum number of tokens the model generates per response. "
            "Does not affect RAM usage, only generation length. "
            "Recommended: 256–1024."
        ),
        "min": 32, "max": 4096, "type": "int",
    },
    "auto_spill_on_start": {
        "label": "Auto-Spill Refs on Startup",
        "desc":  (
            "When enabled, all reference chunk caches are immediately spilled to disk "
            "on app startup regardless of available RAM. Useful on systems with "
            "very low RAM where you need the model to have maximum headroom."
        ),
        "min": 0, "max": 1, "type": "bool",
    },
}


class ConfigTab(QWidget):
    """Full configuration tab — all thresholds with descriptions."""

    config_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fields: Dict[str, QWidget] = {}
        self._build()

    def _build(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        self.setLayout(outer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea{{background:{C['bg0']};border:none;}}")

        inner = QWidget()
        inner.setStyleSheet(f"background:{C['bg0']};")
        root = QVBoxLayout()
        root.setContentsMargins(22, 18, 22, 22); root.setSpacing(0)
        inner.setLayout(root); scroll.setWidget(inner); outer.addWidget(scroll)

        # Header
        hdr = QLabel("⚙️  App Configuration")
        hdr.setStyleSheet(f"color:{C['txt']};font-size:16px;font-weight:bold;margin-bottom:4px;")
        root.addWidget(hdr)
        sub = QLabel(
            "All thresholds and defaults are persisted to app_config.json. "
            "Hover over any field for a full description. Changes take effect immediately.")
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{C['txt2']};font-size:11px;margin-bottom:16px;")
        root.addWidget(sub)

        # Group fields by category
        categories = [
            ("🧠  Memory & RAM",   ["ram_watchdog_mb", "max_ram_chunks", "auto_spill_on_start"]),
            ("📄  Reference Engine", ["chunk_index_size", "ref_top_k", "ref_max_context_chars"]),
            ("📝  Summarization",   ["summary_chunk_chars", "summary_ctx_carry",
                                     "summary_n_pred_sect", "summary_n_pred_final",
                                     "pause_after_chunks"]),
            ("📚  Multi-PDF",       ["multipdf_n_pred_sect", "multipdf_n_pred_final"]),
            ("⚡  Model Defaults",  ["default_threads", "default_ctx", "default_n_predict"]),
        ]

        for cat_title, keys in categories:
            root.addSpacing(10)
            cat_lbl = QLabel(cat_title)
            cat_lbl.setStyleSheet(
                f"color:{C['txt']};font-size:12px;font-weight:bold;"
                f"letter-spacing:0.4px;padding:2px 0;")
            root.addWidget(cat_lbl)

            card = QFrame()
            card.setStyleSheet(
                f"QFrame{{background:rgba(19,19,42,0.85);"
                f"border:1px solid {C['bdr']};border-radius:12px;}}")
            card_l = QVBoxLayout()
            card_l.setContentsMargins(18, 12, 18, 14); card_l.setSpacing(10)
            card.setLayout(card_l)

            for key in keys:
                meta = CONFIG_FIELD_META.get(key, {})
                val  = APP_CONFIG.get(key, APP_CONFIG_DEFAULTS.get(key, 0))
                row  = self._make_field(key, meta, val)
                card_l.addWidget(row)

            root.addWidget(card)

        # Paused jobs section
        root.addSpacing(14)
        pj_lbl = QLabel("⏸  Paused Summarization Jobs")
        pj_lbl.setStyleSheet(
            f"color:{C['txt']};font-size:12px;font-weight:bold;padding:2px 0;")
        root.addWidget(pj_lbl)

        pj_card = QFrame()
        pj_card.setStyleSheet(
            f"QFrame{{background:rgba(19,19,42,0.85);"
            f"border:1px solid {C['bdr']};border-radius:12px;}}")
        pj_l = QVBoxLayout()
        pj_l.setContentsMargins(14, 10, 14, 12); pj_l.setSpacing(8)
        pj_card.setLayout(pj_l)

        pj_desc = QLabel(
            "Summarization jobs paused mid-way are saved here. "
            "Select a job and click Resume to continue from where it left off.")
        pj_desc.setWordWrap(True)
        pj_desc.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        pj_l.addWidget(pj_desc)

        self.paused_jobs_list = QListWidget()
        self.paused_jobs_list.setFixedHeight(110)
        self.paused_jobs_list.setStyleSheet(
            f"QListWidget{{background:{C['bg1']};border:none;font-size:10px;outline:none;}}"
            f"QListWidget::item{{padding:5px 10px;border-bottom:1px solid {C['bdr']};}}"
            f"QListWidget::item:selected{{background:rgba(124,58,237,0.25);color:{C['acc2']};}}")
        pj_l.addWidget(self.paused_jobs_list)

        pj_btn_row = QHBoxLayout(); pj_btn_row.setSpacing(8)
        self.btn_resume_job   = QPushButton("▶  Resume Job")
        self.btn_delete_job   = QPushButton("🗑  Delete Job")
        self.btn_refresh_jobs = QPushButton("↻  Refresh")
        for b in (self.btn_resume_job, self.btn_delete_job, self.btn_refresh_jobs):
            b.setFixedHeight(28); pj_btn_row.addWidget(b)
        pj_btn_row.addStretch()
        self.btn_refresh_jobs.clicked.connect(self.refresh_paused_jobs)
        self.btn_delete_job.clicked.connect(self._delete_paused_job)
        pj_l.addLayout(pj_btn_row)
        root.addWidget(pj_card)

        # Save / Reset buttons
        root.addSpacing(14)
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        save_btn = QPushButton("💾  Save All Settings")
        save_btn.setObjectName("btn_send")
        save_btn.setFixedHeight(34)
        save_btn.clicked.connect(self._save_all)
        reset_btn = QPushButton("↺  Reset to Defaults")
        reset_btn.setFixedHeight(34)
        reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(save_btn); btn_row.addWidget(reset_btn); btn_row.addStretch()
        root.addLayout(btn_row)

        root.addStretch()
        self.refresh_paused_jobs()

    def _make_field(self, key: str, meta: dict, current_val) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("QFrame{background:transparent;border:none;}")
        fl = QVBoxLayout()
        fl.setContentsMargins(0, 0, 0, 2); fl.setSpacing(3)

        top_row = QHBoxLayout(); top_row.setSpacing(10)
        lbl = QLabel(meta.get("label", key))
        lbl.setStyleSheet(f"color:{C['txt']};font-size:12px;font-weight:600;")
        lbl.setFixedWidth(230)
        top_row.addWidget(lbl)

        ftype = meta.get("type", "int")
        if ftype == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(current_val))
        else:
            widget = QLineEdit(str(current_val))
            widget.setFixedWidth(90)
            widget.setFixedHeight(26)
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            mn = meta.get("min", 0); mx = meta.get("max", 99999)
            widget.setToolTip(f"Range: {mn} – {mx}")

        self._fields[key] = widget
        top_row.addWidget(widget)

        # Range hint
        if ftype != "bool":
            rng_lbl = QLabel(f"({meta.get('min', 0)} – {meta.get('max', '∞')})")
            rng_lbl.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
            top_row.addWidget(rng_lbl)
        top_row.addStretch()
        fl.addLayout(top_row)

        # Description
        desc = QLabel(meta.get("desc", ""))
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color:{C['txt2']};font-size:10px;padding-left:2px;"
            f"line-height:1.5;")
        fl.addWidget(desc)

        frame.setLayout(fl)
        return frame

    def _save_all(self):
        for key, widget in self._fields.items():
            meta  = CONFIG_FIELD_META.get(key, {})
            ftype = meta.get("type", "int")
            mn    = meta.get("min", 0)
            mx    = meta.get("max", 999999)
            if ftype == "bool":
                APP_CONFIG[key] = widget.isChecked()
            else:
                try:
                    v = int(widget.text())
                    v = max(mn, min(mx, v))
                    APP_CONFIG[key] = v
                    widget.setText(str(v))
                except ValueError:
                    widget.setText(str(APP_CONFIG.get(key, APP_CONFIG_DEFAULTS.get(key, 0))))
        _save_app_config(APP_CONFIG)
        self.config_changed.emit()
        QMessageBox.information(self, "Saved", "Configuration saved successfully.")

    def _reset_defaults(self):
        if QMessageBox.question(
            self, "Reset Defaults", "Reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        APP_CONFIG.update(APP_CONFIG_DEFAULTS)
        _save_app_config(APP_CONFIG)
        for key, widget in self._fields.items():
            val = APP_CONFIG_DEFAULTS.get(key, 0)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(val))
            else:
                widget.setText(str(val))
        self.config_changed.emit()

    def refresh_paused_jobs(self):
        self.paused_jobs_list.clear()
        for job in list_paused_jobs():
            if job.get("job_id", "").endswith("_autosave"):
                continue
            jid   = job.get("job_id", "?")
            fname = job.get("filename", "?")
            nc    = job.get("next_chunk", 0)
            tot   = job.get("total", "?")
            ts    = job.get("paused_at", "")[:16]
            item  = QListWidgetItem(
                f"⏸  {fname}  —  chunk {nc}/{tot}  —  paused {ts}")
            item.setData(Qt.ItemDataRole.UserRole, jid)
            item.setForeground(QColor(C["warn"]))
            self.paused_jobs_list.addItem(item)

    def _delete_paused_job(self):
        item = self.paused_jobs_list.currentItem()
        if not item: return
        jid = item.data(Qt.ItemDataRole.UserRole)
        _delete_paused_job(jid)
        self.refresh_paused_jobs()

    def get_selected_job_id(self) -> str:
        item = self.paused_jobs_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def get_selected_job_state(self) -> Optional[dict]:
        jid = self.get_selected_job_id()
        return _load_paused_job(jid) if jid else None

class ParallelLoadingDialog(QWidget):
    """Embedded panel (not a popup) in the Models tab for parallel loading config."""

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._prefs = PARALLEL_PREFS
        self._build()

    def _build(self):
        root = QVBoxLayout()
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # ── enable toggle ──────────────────────────────────────────────────────
        top_row = QHBoxLayout(); top_row.setSpacing(10)
        self.chk_enable = QCheckBox("Enable Parallel Model Loading")
        self.chk_enable.setStyleSheet(f"color:{C['txt']};font-weight:600;font-size:12px;")
        self.chk_enable.setChecked(self._prefs.enabled)
        self.chk_enable.toggled.connect(self._on_toggle)
        top_row.addWidget(self.chk_enable)
        top_row.addStretch()
        root.addLayout(top_row)

        # ── warning banner ─────────────────────────────────────────────────────
        self.warn_banner = QLabel(
            "⚠️  Parallel loading runs multiple model engines simultaneously.\n"
            "Each model consumes its own RAM slice (typically 4–16 GB per model).\n"
            "High CPU/RAM usage is expected. Ensure your system has sufficient memory\n"
            "before enabling. Swap usage may cause severe performance degradation."
        )
        self.warn_banner.setWordWrap(True)
        self.warn_banner.setStyleSheet(
            f"color:{C['warn']};font-size:11px;padding:10px 12px;"
            f"background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.25);"
            f"border-radius:8px;"
        )
        root.addWidget(self.warn_banner)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        root.addWidget(sep)

        # ── auto-load role checkboxes ──────────────────────────────────────────
        auto_lbl = QLabel("Auto-load these roles on startup:")
        auto_lbl.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        root.addWidget(auto_lbl)

        self._role_checks: Dict[str, QCheckBox] = {}
        role_grid = QHBoxLayout(); role_grid.setSpacing(12)
        for role in MODEL_ROLES:
            if role == "general": continue  # always loaded
            chk = QCheckBox(f"{ROLE_ICONS[role]} {role.capitalize()}")
            chk.setChecked(role in self._prefs.auto_load_roles)
            chk.setEnabled(self._prefs.enabled)
            chk.toggled.connect(self._save)
            self._role_checks[role] = chk
            role_grid.addWidget(chk)
        role_grid.addStretch()
        root.addLayout(role_grid)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        root.addWidget(sep2)

        # ── pipeline mode ──────────────────────────────────────────────────────
        pipeline_row = QHBoxLayout(); pipeline_row.setSpacing(10)
        self.chk_pipeline = QCheckBox("🔗  Enable Reasoning → Coding Pipeline Mode")
        self.chk_pipeline.setStyleSheet(f"color:{C['txt']};font-size:12px;")
        self.chk_pipeline.setChecked(self._prefs.pipeline_mode)
        self.chk_pipeline.setEnabled(self._prefs.enabled)
        self.chk_pipeline.toggled.connect(self._save)
        pipeline_row.addWidget(self.chk_pipeline)
        pipeline_row.addStretch()
        root.addLayout(pipeline_row)

        pipeline_desc = QLabel(
            "When enabled and both Reasoning + Coding engines are loaded:\n"
            "  1. 🧠 Reasoning model analyses your coding request → produces a detailed plan\n"
            "  2. 💻 Coding model receives the plan as context → generates the code\n"
            "Only activates when the prompt looks like a coding request."
        )
        pipeline_desc.setWordWrap(True)
        pipeline_desc.setStyleSheet(f"color:{C['txt2']};font-size:11px;padding-left:4px;")
        root.addWidget(pipeline_desc)

        # ── RAM estimate helper ────────────────────────────────────────────────
        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        root.addWidget(sep3)

        self.ram_estimate_lbl = QLabel("")
        self.ram_estimate_lbl.setWordWrap(True)
        self.ram_estimate_lbl.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
        root.addWidget(self.ram_estimate_lbl)
        self._update_ram_estimate()

        root.addStretch()
        self.setLayout(root)
        self._update_enabled_state()

    def _on_toggle(self, checked: bool):
        if checked and not self._prefs.warned:
            result = QMessageBox.warning(
                self, "⚠️ Parallel Loading Warning",
                "Parallel model loading runs multiple GGUF engines simultaneously.\n\n"
                "Each model occupies its own RAM allocation:\n"
                "  • Q4 7B model  ≈  4–5 GB RAM\n"
                "  • Q5 13B model ≈ 9–10 GB RAM\n"
                "  • Q4 70B model ≈ 38–40 GB RAM\n\n"
                "Running 2–3 models simultaneously requires 12–30+ GB of RAM.\n"
                "Insufficient RAM will cause heavy swap usage or system freezes.\n\n"
                "Only proceed if you have sufficient memory.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if result != QMessageBox.StandardButton.Yes:
                self.chk_enable.blockSignals(True)
                self.chk_enable.setChecked(False)
                self.chk_enable.blockSignals(False)
                return
            self._prefs.warned = True

        self._prefs.enabled = checked
        self._update_enabled_state()
        self._save()

    def _update_enabled_state(self):
        en = self._prefs.enabled
        for chk in self._role_checks.values():
            chk.setEnabled(en)
        self.chk_pipeline.setEnabled(en)
        self.warn_banner.setVisible(en)

    def _save(self):
        self._prefs.auto_load_roles = [
            r for r, chk in self._role_checks.items() if chk.isChecked()]
        self._prefs.pipeline_mode = self.chk_pipeline.isChecked()
        self._prefs.save()
        self._update_ram_estimate()
        self.settings_changed.emit()

    def _update_ram_estimate(self):
        if not self._prefs.enabled:
            self.ram_estimate_lbl.setText("")
            return
        n_models = 1 + len(self._prefs.auto_load_roles)
        estimate  = n_models * 6   # rough 6 GB average
        avail     = ""
        if HAS_PSUTIL:
            mem   = psutil.virtual_memory()
            avail = f"  |  System RAM: {mem.total // (1024**3)} GB available"
        self.ram_estimate_lbl.setText(
            f"Estimated RAM for {n_models} model(s): ~{estimate} GB{avail}\n"
            f"(Rough average; actual usage depends on model size and quant type)"
        )

    @property
    def prefs(self) -> ParallelPrefs:
        return self._prefs


# ═════════════════════════════ MAIN WINDOW ══════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Native Lab Pro")
        self.setMinimumSize(1100, 700)
        self.resize(1300, 840)

        self.engine   = LlamaEngine()
        self.sessions: Dict[str, Session] = {}
        self.active:   Optional[Session]  = None

        self._worker:          Optional[QThread]       = None
        self._stream_w:        Optional[MessageWidget] = None
        self._summary_worker:  Optional[QThread]       = None
        self._summary_bubble:  Optional[MessageWidget] = None 
        self._pipeline_worker: Optional[PipelineWorker] = None
        self.reasoning_engine:     Optional[LlamaEngine] = None
        self.summarization_engine: Optional[LlamaEngine] = None
        self.coding_engine:        Optional[LlamaEngine] = None
        self.secondary_engine:     Optional[LlamaEngine] = None
        self._thinking_block:  Optional[ThinkingBlock]  = None
        self._pipeline_reason_w: Optional[MessageWidget] = None
        self._pipeline_code_w:   Optional[MessageWidget] = None
        self._pipeline_insight_widgets: list = []

        self._force_coding_mode: bool = False
        self._pending_ref_ctx:   str  = ""
        self._multi_pdf_worker:  Optional[QThread] = None
        self.current_ctx = DEFAULT_CTX
        self._ctx_reload_timer = QTimer(self)
        self._ctx_reload_timer.setSingleShot(True)
        self._ctx_reload_timer.timeout.connect(self._apply_new_context)

        self._load_sessions()
        self._build_ui()
        self._build_menu()
        self._build_status_bar()
        QApplication.instance().setStyleSheet(QSS)

        if self.sessions:
            last = max(self.sessions.values(), key=lambda s: s.id)
            self._switch_session(last.id)
        else:
            self._new_session()

        QTimer.singleShot(300, self._start_model_load)

        # Auto-load parallel engines if prefs say so
        if PARALLEL_PREFS.enabled and PARALLEL_PREFS.auto_load_roles:
            QTimer.singleShot(1000, self._auto_load_parallel_engines)

    def _auto_load_parallel_engines(self):
        """Load all engines whose roles are in the auto_load list."""
        for role in PARALLEL_PREFS.auto_load_roles:
            models = MODEL_REGISTRY.all_models()
            for m in models:
                if m.get("role") == role and Path(m["path"]).exists():
                    self._start_role_engine_load(role, m["path"])
                    break

    def _start_role_engine_load(self, role: str, path: str):
        attr = f"{role}_engine"
        if not getattr(self, attr, None):
            setattr(self, attr, LlamaEngine())
        eng = getattr(self, attr)
        cfg = MODEL_REGISTRY.get_config(path)
        loader = ModelLoaderThread(eng, path, cfg.ctx)
        loader.log.connect(self._log)
        loader.finished.connect(
            lambda ok, st, r=role, n=Path(path).name:
            self._on_role_engine_loaded(ok, st, r, n, None))
        loader.start()
        setattr(self, f"_loader_{role}", loader)
        self._log("INFO", f"Auto-loading {role} engine: {Path(path).name}")

    # ── context management ────────────────────────────────────────────────────

    def _apply_new_context(self):
        if not self.engine.is_loaded:
            return
        new_ctx = self.ctx_slider.value()
        if new_ctx == getattr(self.engine, "ctx_value", DEFAULT_CTX):
            return
        if new_ctx > 8192:
            ram_estimate = (new_ctx / 1024) * 0.5
            result = QMessageBox.question(
                self, "Confirm Context Reload",
                f"Changing context to {new_ctx:,} tokens requires restarting the model\n"
                f"and may use an additional ~{ram_estimate:.0f} MB of RAM.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if result != QMessageBox.StandardButton.Yes:
                loaded_ctx = getattr(self.engine, "ctx_value", DEFAULT_CTX)
                self.ctx_slider.blockSignals(True)
                self.ctx_slider.setValue(loaded_ctx)
                self.ctx_slider.blockSignals(False)
                self.ctx_input.setText(str(loaded_ctx))
                self.current_ctx = loaded_ctx
                return
        self._log("INFO", f"Reloading model with context {new_ctx}")
        if self._worker:
            if hasattr(self._worker, "abort"):
                self._worker.abort()
            self._worker.wait(1000)
            self._worker = None
        self.engine.shutdown()
        self.current_ctx = new_ctx
        self.ctx_bar.setRange(0, new_ctx)
        self._start_model_load()

    def _on_ctx_changed(self, value: int):
        self.ctx_input.setText(str(value))
        self.current_ctx = value
        if hasattr(self, "model_list"):
            try:
                item = self.model_list.currentItem()
                if item:
                    path = item.data(Qt.ItemDataRole.UserRole)
                    if path and path == getattr(self.engine, "model_path", ""):
                        self.cfg_ctx.blockSignals(True)
                        self.cfg_ctx.setText(str(value))
                        self.cfg_ctx.blockSignals(False)
            except RuntimeError:
                pass
        color = C["ok"]
        warn_text = ""
        if value > 24576:
            color = C["err"]; warn_text = "⚠"
            self.ctx_warn.setToolTip("Very high context.\nExpect heavy RAM usage.")
        elif value > 16384:
            color = C["warn"]; warn_text = "⚠"
            self.ctx_warn.setToolTip("High context.\nPerformance may degrade.")
        else:
            self.ctx_warn.setToolTip("")
        self._ctx_reload_timer.start(2000)
        self.ctx_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height:6px; background:{C['bg2']}; border-radius:3px;
            }}
            QSlider::handle:horizontal {{
                background:{color}; width:14px; margin:-4px 0; border-radius:7px;
            }}
            QSlider::sub-page:horizontal {{
                background:{color}; border-radius:3px;
            }}
        """)
        self.ctx_warn.setText(warn_text)

    def _on_ctx_input_changed(self):
        try:
            value = max(512, min(32768, int(self.ctx_input.text())))
            self.ctx_slider.setValue(value)
        except ValueError:
            self.ctx_input.setText(str(self.ctx_slider.value()))

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        ml = QHBoxLayout()
        ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(2)

        self.sidebar = SessionSidebar()
        self.sidebar.session_selected.connect(self._switch_session)
        self.sidebar.new_session.connect(self._new_session)
        self.sidebar.session_deleted.connect(self._delete_session)
        self.sidebar.session_renamed.connect(self._rename_session)
        self.sidebar.session_exported.connect(
            lambda sid: self._export_session(sid, "md"))
        self.splitter.addWidget(self.sidebar)

        self.tabs = QTabWidget()

        # ── Chat tab (ChatModule) ──
        self.chat_module = ChatModule(
            session_id=self.active.id if self.active else "default")
        self.chat_area = self.chat_module.chat_area
        self.input_bar = self.chat_module.input_bar
        self.chat_module.send_requested.connect(self._on_send_with_refs)
        self.chat_module.stop_requested.connect(self._on_stop)
        self.chat_module.pdf_requested.connect(self._load_pdf)
        self.chat_module.clear_requested.connect(self._clear_chat)
        self.chat_module.multi_pdf_requested.connect(self._start_multi_pdf)
        self.input_bar.code_btn.toggled.connect(
            lambda chk: setattr(self, "_force_coding_mode", chk))
        self.tabs.addTab(self.chat_module, "💬  Chat")

        # ── Models tab ──
        self.models_tab = self._build_models_tab()
        self.tabs.addTab(self.models_tab, "🗂  Models")

        # ── Config tab ──
        self.config_tab = ConfigTab()
        self.config_tab.config_changed.connect(self._on_config_changed)
        self.config_tab.btn_resume_job.clicked.connect(self._resume_paused_job)
        self.tabs.addTab(self.config_tab, "⚙️  Config")

        # ── Logs tab ──
        self.log_console = LogConsole()
        self.tabs.addTab(self.log_console, "🐞  Logs")

        self.splitter.addWidget(self.tabs)
        self.splitter.setSizes([220, 1080])
        self.splitter.setStretchFactor(1, 1)
        ml.addWidget(self.splitter)
        central.setLayout(ml)
        self.setCentralWidget(central)

    # ── models tab ───────────────────────────────────────────────────────────

    def _build_models_tab(self) -> QWidget:
        outer = QWidget()
        outer_l = QVBoxLayout()
        outer_l.setContentsMargins(0, 0, 0, 0); outer_l.setSpacing(0)
        outer.setLayout(outer_l)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea{{background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {C['bg0']},stop:1 #090914);border:none;}}"
        )

        w = QWidget()
        w.setStyleSheet(f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                        f"stop:0 {C['bg0']},stop:1 #090914);")
        root = QVBoxLayout()
        root.setContentsMargins(20, 18, 20, 18); root.setSpacing(0)
        w.setLayout(root); scroll.setWidget(w); outer_l.addWidget(scroll)

        def _section_label(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{C['txt']};font-size:12px;font-weight:bold;"
                              f"letter-spacing:0.5px;padding:0;margin-bottom:2px;")
            return lbl

        def _card(layout) -> QFrame:
            card = QFrame()
            card.setStyleSheet(
                f"QFrame{{background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                f"stop:0 rgba(19,19,42,0.85),stop:1 rgba(13,13,28,0.9));"
                f"border:1px solid {C['bdr']};border-radius:12px;}}"
            )
            card.setLayout(layout); return card

        # ── header ───────────────────────────────────────────────────────────
        hdr = QLabel("🗂  GGUF Model Manager")
        hdr.setStyleSheet(f"color:{C['txt']};font-size:16px;font-weight:bold;margin-bottom:4px;")
        root.addWidget(hdr)
        note = QLabel("Add models, assign roles, and configure the reasoning→coding pipeline.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{C['txt2']};font-size:11px;margin-bottom:14px;")
        root.addWidget(note)

        # ── MODEL LIBRARY ─────────────────────────────────────────────────────
        root.addWidget(_section_label("MODEL LIBRARY"))
        list_card_l = QVBoxLayout()
        list_card_l.setContentsMargins(0, 0, 0, 0); list_card_l.setSpacing(0)

        # legend
        legend_row = QHBoxLayout()
        legend_row.setContentsMargins(10, 8, 10, 6); legend_row.setSpacing(14)
        for role, icon in ROLE_ICONS.items():
            pill = QLabel(f"{icon} {role.capitalize()}")
            pill.setStyleSheet(f"color:{C['txt2']};font-size:10px;"
                               f"background:{C['bg2']};border-radius:4px;padding:2px 6px;")
            legend_row.addWidget(pill)
        legend_row.addStretch()
        list_card_l.addLayout(legend_row)

        self.model_list = QListWidget()
        self.model_list.setStyleSheet(
            f"QListWidget{{background:{C['bg1']};border:none;"
            f"border-top:1px solid {C['bdr']};font-size:11px;outline:none;}}"
            f"QListWidget::item{{padding:8px 12px;border-bottom:1px solid {C['bdr']};}}"
            f"QListWidget::item:selected{{background:{C['bg2']};color:{C['acc']};}}"
            f"QListWidget::item:hover:!selected{{background:{C['bg2']};}}"
        )
        self.model_list.setMinimumHeight(150)
        self.model_list.setMaximumHeight(240)
        self.model_list.currentItemChanged.connect(self._on_model_list_select)
        list_card_l.addWidget(self.model_list)

        btn_strip = QHBoxLayout()
        btn_strip.setContentsMargins(10, 8, 10, 8); btn_strip.setSpacing(8)
        self.btn_browse_model = QPushButton("📂  Browse GGUF…")
        self.btn_load_primary = QPushButton("⚡  Load Selected")
        self.btn_load_primary.setObjectName("btn_send")
        self.btn_remove_model = QPushButton("🗑  Remove")
        self.btn_remove_model.setObjectName("btn_stop")
        for b in (self.btn_browse_model, self.btn_load_primary, self.btn_remove_model):
            b.setFixedHeight(30); btn_strip.addWidget(b)
        btn_strip.addStretch()
        list_card_l.addLayout(btn_strip)
        root.addWidget(_card(list_card_l))
        root.addSpacing(14)
        self.btn_browse_model.clicked.connect(self._browse_add_model)
        self.btn_load_primary.clicked.connect(self._load_selected_as_primary)
        self.btn_remove_model.clicked.connect(self._remove_selected_model)

        # ── PER-MODEL PARAMETERS ──────────────────────────────────────────────
        root.addWidget(_section_label("PER-MODEL PARAMETERS"))
        hint = QLabel("Select a model above to edit its parameters.")
        hint.setStyleSheet(f"color:{C['txt2']};font-size:10px;margin-bottom:6px;")
        root.addWidget(hint)

        cfg_card_l = QVBoxLayout()
        cfg_card_l.setContentsMargins(16, 14, 16, 14); cfg_card_l.setSpacing(10)
        LW = 110

        def _field_row(label_text: str, *widgets, stretch=True) -> QHBoxLayout:
            row = QHBoxLayout(); row.setSpacing(8)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(LW)
            lbl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
            row.addWidget(lbl)
            for ww in widgets: row.addWidget(ww)
            if stretch: row.addStretch()
            return row

        # Detected family banner (read-only, auto-set)
        self.cfg_family_lbl = QLabel("—")
        self.cfg_family_lbl.setStyleSheet(
            f"color:{C['acc2']};font-size:11px;"
            f"background:{C['bg2']};border-radius:4px;padding:3px 8px;"
        )
        cfg_card_l.addLayout(_field_row("Detected Family:", self.cfg_family_lbl))

        # Quant type banner
        self.cfg_quant_lbl = QLabel("—")
        self.cfg_quant_lbl.setStyleSheet(
            f"color:{C['ok']};font-size:11px;"
            f"background:{C['bg2']};border-radius:4px;padding:3px 8px;"
        )
        cfg_card_l.addLayout(_field_row("Quant Type:", self.cfg_quant_lbl))

        dv0 = QFrame(); dv0.setFrameShape(QFrame.Shape.HLine)
        dv0.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        cfg_card_l.addWidget(dv0)

        self.cfg_role = QComboBox()
        self.cfg_role.setMinimumWidth(200); self.cfg_role.setFixedHeight(28)
        for r in MODEL_ROLES:
            self.cfg_role.addItem(f"{ROLE_ICONS[r]}  {r.capitalize()}", r)
        cfg_card_l.addLayout(_field_row("Role:", self.cfg_role))

        dv1 = QFrame(); dv1.setFrameShape(QFrame.Shape.HLine)
        dv1.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        cfg_card_l.addWidget(dv1)

        self.cfg_threads = QLineEdit(str(DEFAULT_THREADS))
        self.cfg_threads.setFixedWidth(64); self.cfg_threads.setFixedHeight(28)
        self.cfg_threads.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Threads:", self.cfg_threads))

        self.cfg_ctx = QLineEdit(str(DEFAULT_CTX))
        self.cfg_ctx.setFixedWidth(80); self.cfg_ctx.setFixedHeight(28)
        self.cfg_ctx.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Context (tokens):", self.cfg_ctx))

        self.cfg_temp = QLineEdit("0.7")
        self.cfg_temp.setFixedWidth(64); self.cfg_temp.setFixedHeight(28)
        self.cfg_temp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Temperature:", self.cfg_temp))

        self.cfg_topp = QLineEdit("0.9")
        self.cfg_topp.setFixedWidth(64); self.cfg_topp.setFixedHeight(28)
        self.cfg_topp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Top-P:", self.cfg_topp))

        self.cfg_rep = QLineEdit("1.1")
        self.cfg_rep.setFixedWidth(64); self.cfg_rep.setFixedHeight(28)
        self.cfg_rep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Repeat Penalty:", self.cfg_rep))

        self.cfg_npred = QLineEdit(str(DEFAULT_N_PRED))
        self.cfg_npred.setFixedWidth(80); self.cfg_npred.setFixedHeight(28)
        self.cfg_npred.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Max Tokens:", self.cfg_npred))

        self.cfg_param_warn = QLabel("")
        self.cfg_param_warn.setWordWrap(True)
        self.cfg_param_warn.setStyleSheet(
            f"color:{C['warn']};font-size:11px;padding:4px 8px;"
            f"background:#2a2000;border-radius:5px;border:1px solid #5a4500;"
        )
        self.cfg_param_warn.setVisible(False)
        cfg_card_l.addWidget(self.cfg_param_warn)

        dv2 = QFrame(); dv2.setFrameShape(QFrame.Shape.HLine)
        dv2.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        cfg_card_l.addWidget(dv2)

        save_row = QHBoxLayout(); save_row.setSpacing(8)
        self.btn_save_cfg = QPushButton("💾  Save Parameters")
        self.btn_save_cfg.setFixedHeight(30)
        self.btn_save_cfg.clicked.connect(self._save_model_config)
        save_row.addWidget(self.btn_save_cfg)
        save_row.addStretch()
        cfg_card_l.addLayout(save_row)
        for wf in (self.cfg_ctx, self.cfg_threads, self.cfg_temp):
            wf.textChanged.connect(self._check_param_warnings)
        root.addWidget(_card(cfg_card_l))
        root.addSpacing(14)

        # ── ACTIVE ENGINES ────────────────────────────────────────────────────
        root.addWidget(_section_label("ACTIVE ENGINES"))
        eng_card_l = QVBoxLayout()
        eng_card_l.setContentsMargins(0, 0, 0, 0); eng_card_l.setSpacing(0)
        self.engine_status_list = QListWidget()
        self.engine_status_list.setFixedHeight(130)
        self.engine_status_list.setStyleSheet(
            f"QListWidget{{background:{C['bg1']};border:none;"
            f"font-size:11px;font-family:'Consolas','Courier New',monospace;outline:none;}}"
            f"QListWidget::item{{padding:6px 14px;border-bottom:1px solid {C['bdr']};}}"
            f"QListWidget::item:hover{{background:{C['bg2']};}}"
        )
        eng_card_l.addWidget(self.engine_status_list)
        eng_btn_strip = QHBoxLayout()
        eng_btn_strip.setContentsMargins(10, 8, 10, 8); eng_btn_strip.setSpacing(8)
        self.btn_load_role_engine = QPushButton("⚡  Load Engine for Role")
        self.btn_load_role_engine.setFixedHeight(30)
        self.btn_load_role_engine.clicked.connect(self._load_engine_for_selected)
        self.btn_unload_all = QPushButton("⏏  Unload All")
        self.btn_unload_all.setFixedHeight(30)
        self.btn_unload_all.clicked.connect(self._unload_all_engines)
        eng_btn_strip.addWidget(self.btn_load_role_engine)
        eng_btn_strip.addWidget(self.btn_unload_all)
        eng_btn_strip.addStretch()
        eng_card_l.addLayout(eng_btn_strip)
        root.addWidget(_card(eng_card_l))
        root.addSpacing(14)

        # ── PARALLEL LOADING ──────────────────────────────────────────────────
        root.addWidget(_section_label("PARALLEL LOADING & PIPELINE"))
        par_card_l = QVBoxLayout()
        par_card_l.setContentsMargins(0, 0, 0, 0)
        self.parallel_panel = ParallelLoadingDialog()
        self.parallel_panel.settings_changed.connect(self._on_parallel_settings_changed)
        par_card_l.addWidget(self.parallel_panel)
        root.addWidget(_card(par_card_l))
        root.addSpacing(14)

        # hidden compat stubs
        self.reasoning_status      = QLabel()
        self.summary_engine_status = QLabel()
        self.coding_engine_status  = QLabel()
        for lbl in (self.reasoning_status, self.summary_engine_status, self.coding_engine_status):
            lbl.setVisible(False); root.addWidget(lbl)

        root.addStretch()
        self._refresh_model_list()
        return outer

    def _on_parallel_settings_changed(self):
        pipeline_on = bool(
            PARALLEL_PREFS.enabled and
            PARALLEL_PREFS.pipeline_mode and
            self.reasoning_engine is not None and self.reasoning_engine.is_loaded and
            self.coding_engine is not None and self.coding_engine.is_loaded
        )
        self.input_bar.set_pipeline_mode(pipeline_on)
        self._log("INFO",
            f"Parallel settings: enabled={PARALLEL_PREFS.enabled}, "
            f"pipeline={PARALLEL_PREFS.pipeline_mode}, "
            f"auto_roles={PARALLEL_PREFS.auto_load_roles}")

    # ── model config helpers ──────────────────────────────────────────────────

    def _check_param_warnings(self):
        warnings = []
        try:
            ctx = int(self.cfg_ctx.text())
            if ctx > 24576:
                warnings.append(f"⚠  Context {ctx:,} tokens is very high")
            elif ctx > 16384:
                warnings.append(f"⚠  Context {ctx:,} tokens is high")
        except ValueError:
            pass
        try:
            threads = int(self.cfg_threads.text())
            import multiprocessing
            ncpu = multiprocessing.cpu_count()
            if threads > ncpu:
                warnings.append(f"⚠  {threads} threads exceeds {ncpu} logical CPUs")
        except (ValueError, NotImplementedError):
            pass
        try:
            temp = float(self.cfg_temp.text())
            if temp > 1.5:
                warnings.append("⚠  Temperature > 1.5")
            elif temp < 0.05:
                warnings.append("⚠  Temperature near 0")
        except ValueError:
            pass
        if warnings:
            self.cfg_param_warn.setText("\n".join(warnings))
            self.cfg_param_warn.setVisible(True)
        else:
            self.cfg_param_warn.setVisible(False)

    def _on_model_list_select(self, item: Optional[QListWidgetItem], _=None):
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path: return

        # Update family / quant labels
        fam   = detect_model_family(path)
        quant = detect_quant_type(path)
        ql, qcolor = quant_info(quant)
        self.cfg_family_lbl.setText(f"{fam.name}  (template: {fam.template})")
        self.cfg_quant_lbl.setText(f"{quant}  ·  {ql}")
        self.cfg_quant_lbl.setStyleSheet(
            f"color:{qcolor};font-size:11px;"
            f"background:{C['bg2']};border-radius:4px;padding:3px 8px;")

        cfg = MODEL_REGISTRY.get_config(path)
        idx = self.cfg_role.findData(cfg.role)
        self.cfg_role.setCurrentIndex(max(idx, 0))
        self.cfg_threads.setText(str(cfg.threads))
        if path == getattr(self.engine, "model_path", "") and hasattr(self, "ctx_slider"):
            self.cfg_ctx.setText(str(self.ctx_slider.value()))
        else:
            self.cfg_ctx.setText(str(cfg.ctx))
        self.cfg_temp.setText(str(cfg.temperature))
        self.cfg_topp.setText(str(cfg.top_p))
        self.cfg_rep.setText(str(cfg.repeat_penalty))
        self.cfg_npred.setText(str(cfg.n_predict))
        self._check_param_warnings()

    def _save_model_config(self):
        item = self.model_list.currentItem()
        if not item:
            self._log("WARN", "No model selected"); return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path: return
        try:
            ctx = int(self.cfg_ctx.text()); threads = int(self.cfg_threads.text())
            temp = float(self.cfg_temp.text()); topp = float(self.cfg_topp.text())
            rep = float(self.cfg_rep.text()); npred = int(self.cfg_npred.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Parameter", "One or more fields are invalid."); return

        dangers = []
        if ctx > 24576:  dangers.append(f"Context = {ctx:,} tokens (very high)")
        if threads > 32: dangers.append(f"Threads = {threads}")
        if temp > 2.0:   dangers.append(f"Temperature = {temp}")
        if dangers:
            msg = "High-compute parameters:\n\n" + "\n".join(f"  • {d}" for d in dangers) + "\n\nSave?"
            if QMessageBox.warning(self, "⚠ Confirm", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                return

        fam = detect_model_family(path)
        cfg = ModelConfig(
            path=path, role=self.cfg_role.currentData() or "general",
            threads=threads, ctx=ctx, temperature=temp, top_p=topp,
            repeat_penalty=rep, n_predict=npred, family=fam.family,
        )
        MODEL_REGISTRY.set_config(path, cfg)
        self._refresh_model_list()
        self._log("INFO", f"Saved config for {Path(path).name}: family={fam.name}, "
                          f"role={cfg.role}, ctx={cfg.ctx}")

        if path == getattr(self.engine, "model_path", ""):
            if ctx != self.ctx_slider.value():
                self.ctx_slider.blockSignals(True)
                self.ctx_slider.setValue(ctx)
                self.ctx_input.setText(str(ctx))
                self.ctx_slider.blockSignals(False)
                self.current_ctx = ctx
                self._ctx_reload_timer.start(500)

    def _load_engine_for_selected(self):
        item = self.model_list.currentItem()
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "File Not Found", f"Cannot find:\n{path}"); return
        cfg = MODEL_REGISTRY.get_config(path)
        role = cfg.role

        if PARALLEL_PREFS.enabled and role != "general":
            # Warn if loading a second/third model
            n_loaded = sum(1 for r in ("reasoning","summarization","coding","secondary")
                           if getattr(self, f"{r}_engine", None) and
                           getattr(self, f"{r}_engine").is_loaded)
            if n_loaded >= 1:
                size_mb = ModelConfig(path=path).size_mb
                ram_est = max(size_mb * 1.1 / 1000, 1)
                QMessageBox.information(
                    self, "⚠️ Parallel RAM Usage",
                    f"Loading an additional engine (~{ram_est:.1f} GB).\n"
                    f"Total parallel engines after this: {n_loaded + 2}\n\n"
                    f"Ensure you have sufficient free RAM."
                )

        if role == "general":
            idx = self.input_bar.model_combo.findData(path)
            if idx == -1:
                self.input_bar.model_combo.addItem(Path(path).name, path)
                idx = self.input_bar.model_combo.findData(path)
            self.input_bar.model_combo.setCurrentIndex(idx)
            self.engine.shutdown()
            QTimer.singleShot(200, self._start_model_load)
        elif role in ("reasoning", "summarization", "coding", "secondary"):
            self._start_role_engine_load(role, path)
        self._refresh_engine_status()

    def _on_role_engine_loaded(self, ok: bool, status: str, role: str,
                                name: str, lbl):
        color = C["ok"] if ok else C["err"]
        icon  = ROLE_ICONS.get(role, "🔌")
        text  = f"{icon} {role.capitalize()}:  {'✅  ' if ok else '❌  '}{name}"
        if lbl:
            try:
                lbl.setText(text)
                lbl.setStyleSheet(f"color:{color};font-size:11px;")
            except RuntimeError:
                pass
        self._log("INFO" if ok else "ERROR", f"{role} engine: {status}")
        self._refresh_engine_status()
        # Update pipeline badge in input bar
        self._on_parallel_settings_changed()

    def _refresh_engine_status(self):
        self.engine_status_list.clear()
        engines = {"General (primary)": self.engine}
        for role in ("reasoning", "summarization", "coding", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng:
                engines[role.capitalize()] = eng
        for role_name, eng in engines.items():
            icon       = "🟢" if eng.is_loaded else "⚪"
            model_name = Path(eng.model_path).name if eng.model_path else "not loaded"
            fam_tag    = ""
            if eng.model_path:
                fam = detect_model_family(eng.model_path)
                qt  = detect_quant_type(eng.model_path)
                fam_tag = f"  [{fam.name} · {qt}]"
            mode_tag = f"  [{eng.mode}]" if eng.is_loaded else ""
            item = QListWidgetItem(
                f"  {icon}  {role_name:<22}  {model_name}{mode_tag}{fam_tag}")
            item.setForeground(QColor(C["ok"] if eng.is_loaded else C["txt2"]))
            self.engine_status_list.addItem(item)

    def _unload_all_engines(self):
        if QMessageBox.question(
            self, "Unload All Engines",
            "Unload all model engines?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        for role in ("reasoning", "summarization", "coding", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng:
                eng.shutdown()
                setattr(self, f"{role}_engine", None)
        self.engine.shutdown()
        self._refresh_engine_status()
        self._on_parallel_settings_changed()
        self._log("INFO", "All engines unloaded.")

    def _refresh_model_list(self):
        self.model_list.clear()
        active = getattr(self.engine, "model_path", "")
        for m in MODEL_REGISTRY.all_models():
            tag       = "📌" if m["source"] == "custom" else "📦"
            role_icon = ROLE_ICONS.get(m.get("role", "general"), "💬")
            ql, qc    = quant_info(m.get("quant", ""))
            label = (f"{tag}  {role_icon} [{m.get('role','general'):<14}]  "
                     f"{m['name']}   ({m['size_mb']} MB)  "
                     f"[{m.get('family','?')}·{m.get('quant','?')}·{ql}]")
            if m["path"] == active: label += "  ✅"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, m["path"])
            if m["path"] == active:
                item.setForeground(QColor(C["ok"]))
            self.model_list.addItem(item)
        if hasattr(self, "engine_status_list"):
            self._refresh_engine_status()

    def _sync_input_bar_combo(self):
        cur = self.input_bar.model_combo.currentData()
        self.input_bar.model_combo.blockSignals(True)
        self.input_bar.model_combo.clear()
        for m in MODEL_REGISTRY.all_models():
            self.input_bar.model_combo.addItem(m["name"], m["path"])
        idx = self.input_bar.model_combo.findData(cur)
        self.input_bar.model_combo.setCurrentIndex(max(idx, 0))
        self.input_bar.model_combo.blockSignals(False)
        self.input_bar._update_family_badge()
        if hasattr(self, "model_list"):
            self._refresh_model_list()

    def _browse_add_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", str(Path.home()),
            "GGUF Models (*.gguf);;All Files (*)")
        if not path: return
        MODEL_REGISTRY.add(path)
        fam   = detect_model_family(path)
        quant = detect_quant_type(path)
        ql, _ = quant_info(quant)
        self._refresh_model_list()
        self._sync_input_bar_combo()
        self._log("INFO",
            f"Added model: {Path(path).name}  →  {fam.name}  ·  {quant}  ·  {ql}")

    def _load_selected_as_primary(self):
        item = self.model_list.currentItem()
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "File Not Found", f"Cannot find:\n{path}"); return
        idx = self.input_bar.model_combo.findData(path)
        if idx == -1:
            self.input_bar.model_combo.addItem(Path(path).name, path)
            idx = self.input_bar.model_combo.findData(path)
        self.input_bar.model_combo.setCurrentIndex(idx)
        self.engine.shutdown()
        QTimer.singleShot(200, self._start_model_load)
        self._log("INFO", f"Loading primary model: {Path(path).name}")

    def _remove_selected_model(self):
        item = self.model_list.currentItem()
        if not item: return
        MODEL_REGISTRY.remove(item.data(Qt.ItemDataRole.UserRole))
        self._refresh_model_list()
        self._sync_input_bar_combo()

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        fm.addAction(QAction("New Session\tCtrl+N", self, triggered=self._new_session))
        fm.addSeparator()
        xm = fm.addMenu("Export Current Session")
        xm.addAction(QAction("JSON",     self, triggered=lambda: self._export_active("json")))
        xm.addAction(QAction("Markdown", self, triggered=lambda: self._export_active("md")))
        xm.addAction(QAction("TXT",      self, triggered=lambda: self._export_active("txt")))
        fm.addSeparator()
        fm.addAction(QAction("Quit\tCtrl+Q", self, triggered=self.close))
        vm = mb.addMenu("View")
        vm.addAction(QAction("Toggle Sidebar\tCtrl+B", self, triggered=self._toggle_sidebar))
        vm.addAction(QAction("Go to Logs\tCtrl+L",    self, triggered=self._goto_logs))
        vm.addAction(QAction("Go to Models\tCtrl+M",  self, triggered=self._goto_models_tab))
        mm = mb.addMenu("Model")
        mm.addAction(QAction("Reload Model", self, triggered=self._reload_model))

    def _build_status_bar(self):
        sb = self.statusBar()
        self.lbl_engine = QLabel("⚪  Loading…")
        self.lbl_engine.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
        sb.addWidget(self.lbl_engine)
        sb.addWidget(self._vline())

        # Family badge in status bar
        self.lbl_family = QLabel("")
        self.lbl_family.setStyleSheet(f"color:{C['acc2']};padding:0 6px;font-size:10px;")
        sb.addWidget(self.lbl_family)
        sb.addWidget(self._vline())

        sb.addWidget(QLabel("  Context:"))
        self.ctx_slider = QSlider(Qt.Orientation.Horizontal)
        self.ctx_slider.setRange(512, 32768)
        self.ctx_slider.setFixedWidth(140)
        self.ctx_slider.blockSignals(True)
        self.ctx_slider.setValue(DEFAULT_CTX)
        self.ctx_slider.blockSignals(False)
        self.ctx_slider.valueChanged.connect(self._on_ctx_changed)
        sb.addWidget(self.ctx_slider)

        self.ctx_input = QLineEdit(str(DEFAULT_CTX))
        self.ctx_input.setFixedWidth(60)
        self.ctx_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ctx_input.editingFinished.connect(self._on_ctx_input_changed)
        sb.addWidget(self.ctx_input)

        self.ctx_warn = QLabel("")
        self.ctx_warn.setFixedWidth(24)
        self.ctx_warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ctx_warn.setStyleSheet(f"color:{C['warn']};font-weight:bold;")
        sb.addWidget(self.ctx_warn)

        self.ctx_bar = QProgressBar()
        self.ctx_bar.setRange(0, DEFAULT_CTX)
        self.ctx_bar.setValue(0)
        self.ctx_bar.setFixedWidth(100)
        self.ctx_bar.setFixedHeight(8)
        self.ctx_bar.setTextVisible(False)
        sb.addWidget(self.ctx_bar)

        self.ctx_lbl = QLabel(f"0 / {DEFAULT_CTX}")
        self.ctx_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
        sb.addWidget(self.ctx_lbl)

        sb.addPermanentWidget(self._vline())
        self.tps_lbl = QLabel("— tok/s")
        self.tps_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
        sb.addPermanentWidget(self.tps_lbl)

        if HAS_PSUTIL:
            sb.addPermanentWidget(self._vline())
            self.ram_lbl = QLabel("RAM: —")
            self.ram_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
            sb.addPermanentWidget(self.ram_lbl)
            self._ram_timer = QTimer(self)
            self._ram_timer.timeout.connect(self._update_ram)
            self._ram_timer.start(2500)

    @staticmethod
    def _vline() -> QFrame:
        f = QFrame(); f.setFrameShape(QFrame.Shape.VLine)
        f.setStyleSheet(f"color:{C['bdr']};"); return f

    # ── session management ────────────────────────────────────────────────────

    def _load_sessions(self):
        self.sessions = {}
        for p in SESSIONS_DIR.glob("*.json"):
            try:
                s = Session.load(p)
                self.sessions[s.id] = s
            except Exception as e:
                self._log("WARN", f"Skipping corrupt session {p.name}: {e}")

    def _refresh_sidebar(self):
        self.sidebar.refresh(self.sessions, self.active.id if self.active else "")

    def _new_session(self):
        s = Session.new()
        self.sessions[s.id] = s
        s.save()
        self._switch_session(s.id)

    def _switch_session(self, sid: str):
        if self._worker:
            try:
                if hasattr(self._worker, "abort"): self._worker.abort()
                self._worker.wait(2000)
            except Exception:
                pass
            self._worker = None
            self._stream_w = None
            self.input_bar.set_generating(False)

        s = self.sessions.get(sid)
        if not s: return
        self.active = s
        self.chat_area.clear_messages()
        for m in s.messages:
            self.chat_area.add_message(m.role, m.content, m.timestamp)
        self._refresh_sidebar()
        self.sidebar.set_active(sid)
        self._update_ctx_bar()
        # Sync ChatModule reference panel to this session
        if hasattr(self, "chat_module"):
            self.chat_module.set_session(sid)

    def _delete_session(self, sid: str):
        name = self.sessions[sid].title
        if QMessageBox.question(
            self, "Delete Session", f'Delete "{name}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        p = SESSIONS_DIR / f"{sid}.json"
        if p.exists():
            p.unlink()
        del self.sessions[sid]
        if self.active and self.active.id == sid:
            self.active = None
            self.chat_area.clear_messages()
        if self.sessions:
            last = max(self.sessions.values(), key=lambda s: s.id)
            self._switch_session(last.id)
        else:
            self._new_session()

    def _rename_session(self, sid: str, title: str):
        if sid in self.sessions:
            self.sessions[sid].title = title
            self.sessions[sid].save()
            self._refresh_sidebar()

    def _clear_chat(self):
        if not self.active: return
        if QMessageBox.question(
            self, "Clear Chat", "Clear all messages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        self.active.messages.clear()
        self.active.save()
        self.chat_area.clear_messages()
        self._update_ctx_bar()

    # ── model loading ─────────────────────────────────────────────────────────

    def _start_model_load(self):
        model = self.input_bar.selected_model
        if not model or not Path(model).exists():
            model = str(MODELS_DIR / DEFAULT_MODEL)
        self.lbl_engine.setText("🔄  Loading model…")
        fam   = detect_model_family(model)
        quant = detect_quant_type(model)
        ql, _ = quant_info(quant)
        self.lbl_family.setText(f"{fam.name}  ·  {quant}  ·  {ql}")
        self._log("INFO", f"Loading model: {Path(model).name}  [{fam.name} / {quant}]")
        self._loader = ModelLoaderThread(self.engine, model, self.ctx_slider.value())
        self._loader.log.connect(self._log)
        self._loader.finished.connect(self._on_model_loaded)
        self._loader.start()

    def _on_model_loaded(self, ok: bool, status: str):
        self.ctx_slider.setEnabled(True)
        self.lbl_engine.setText(status)
        color = C["ok"] if ok else C["err"]
        self.lbl_engine.setStyleSheet(f"color:{color};padding:0 8px;")
        self._log("INFO" if ok else "ERROR", f"Model load: {status}")
        if hasattr(self, "model_list"):
            self._refresh_model_list()

    def _reload_model(self):
        self.engine.shutdown()
        QTimer.singleShot(500, self._start_model_load)

    # ── coding detection ──────────────────────────────────────────────────────

    _CODING_KEYWORDS = (
        "def ", "class ", "import ", "function ", "```", "debug ",
        "fix bug", "write code", "python ", "javascript", "typescript",
        "rust ", "c++", "c#", "golang", " sql", "regex", "script ",
        "implement ", "algorithm", "refactor", "syntax error",
        "code to ", "write a ", "create a ", "build a ", "generate code",
    )

    def _is_coding_prompt(self, text: str) -> bool:
        tl = text.lower()
        return self._force_coding_mode or any(k in tl for k in self._CODING_KEYWORDS)

    def _can_use_pipeline(self, text: str) -> bool:
        """Return True if pipeline mode should be used for this prompt."""
        return (
            PARALLEL_PREFS.enabled and
            PARALLEL_PREFS.pipeline_mode and
            self._is_coding_prompt(text) and
            self.reasoning_engine is not None and self.reasoning_engine.is_loaded and
            self.coding_engine is not None and self.coding_engine.is_loaded
        )

    def _active_engine_for(self, text: str) -> "LlamaEngine":
        if self._is_coding_prompt(text) and \
                self.coding_engine and self.coding_engine.is_loaded:
            return self.coding_engine
        return self.engine

    # ── chat send ─────────────────────────────────────────────────────────────

    def _on_send_with_refs(self, text: str, ref_ctx: str):
        """Called by ChatModule — injects reference context into prompt."""
        self._pending_ref_ctx = ref_ctx
        self._on_send(text)

    def _on_send(self, text: str):
        if not self.active:    self._new_session()
        if not self.engine.is_loaded:
            self._log("WARN", "Model not yet loaded — please wait."); return

        ts = datetime.now().strftime("%H:%M")
        if not self.active.messages:
            self.active.title = text[:40].replace("\n", " ")
            self.active.save()

        self.active.add_message("user", text)
        self.active.save()
        self.chat_area.add_message("user", text, ts)
        self._update_ctx_bar()

        # ── Pipeline mode ─────────────────────────────────────────────────────
        if self._can_use_pipeline(text):
            self._start_pipeline(text, ts)
            return

        # ── Normal / single-engine mode ───────────────────────────────────────
        active_eng = self._active_engine_for(text)
        is_coding  = active_eng is self.coding_engine
        eng_label  = "💻 Coding engine" if is_coding else active_eng.status_text

        ctx_chars = getattr(active_eng, "ctx_value", DEFAULT_CTX) * 4
        prompt    = self.active.build_prompt(
            model_path=active_eng.model_path,
            max_chars=ctx_chars
        )
        # Inject reference context if available
        ref_ctx = getattr(self, "_pending_ref_ctx", "")
        if ref_ctx:
            fam = detect_model_family(active_eng.model_path)
            ref_block = (
                f"{fam.bos}{fam.user_prefix}"
                f"The following reference material is provided for context:\n\n"
                f"{ref_ctx}\n\n"
                f"Use this reference when answering the user's question."
                f"{fam.user_suffix}{fam.assistant_prefix}"
            )
            prompt = ref_block + "\n" + prompt
            self._pending_ref_ctx = ""

        cfg_pred = DEFAULT_N_PRED
        if active_eng.model_path:
            cfg_pred = MODEL_REGISTRY.get_config(active_eng.model_path).n_predict

        self._stream_w = self.chat_area.add_message(
            "assistant", "", ts, tag="💻 Coding" if is_coding else "")
        self._log("INFO", f"Prompt ≈ {len(prompt)} chars · engine: {eng_label}")

        self._worker = active_eng.create_worker(
            prompt, n_predict=cfg_pred, model_path=active_eng.model_path)
        self._worker.token.connect(self._on_token)
        self._worker.done.connect(self._on_done)
        self._worker.err.connect(self._on_err)
        self._worker.start()

        self.input_bar.set_generating(True)
        lbl_txt = "💻 Coding…" if is_coding else "⚡  Generating…"
        self.lbl_engine.setText(lbl_txt)
        self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")

    # ── Pipeline mode orchestration ───────────────────────────────────────────

    def _collect_insight_engines(self):
        """Return list of (label, engine) for all active non-coding loaded engines."""
        candidates = [
            ("🧠 Reasoning",     self.reasoning_engine),
            ("📝 Summarization", self.summarization_engine),
            ("🔮 Secondary",     self.secondary_engine),
        ]
        # Also include primary engine if it's not the coding engine
        if self.engine and self.engine.is_loaded and self.engine is not self.coding_engine:
            candidates.append(("⚡ Primary", self.engine))

        return [
            (label, eng)
            for label, eng in candidates
            if eng is not None and eng.is_loaded and eng is not self.coding_engine
        ]

    def _start_pipeline(self, text: str, ts: str):
        # Ensure all engines are in server mode before starting pipeline
        # to avoid CLI prompt echo glitch
        engines_to_check = []
        if self.coding_engine and self.coding_engine.is_loaded:
            engines_to_check.append(("coding", self.coding_engine))
        for role in ("reasoning", "summarization", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng and eng.is_loaded:
                engines_to_check.append((role, eng))

        for role, eng in engines_to_check:
            if eng.mode != "server":
                self._log("WARN", f"{role} engine in CLI mode — attempting server upgrade…")
                ok = eng.ensure_server_or_reload(log_cb=self._log)
                if not ok:
                    self._log("ERROR",
                        f"{role} engine could not start server mode — aborting pipeline")
                    self.chat_area.add_message(
                        "assistant",
                        f"⚠️ Pipeline aborted: **{role}** engine could not start in server mode.\n"
                        f"Try reloading the model from the Models tab.",
                        ts)
                    self.input_bar.set_generating(False)
                    return

        insight_engines = self._collect_insight_engines()

        if not insight_engines:
            # Fallback: no insight engines, just run coding engine directly
            self._log("WARN", "No insight engines available — running coding engine directly")
            active_eng = self.coding_engine
            ctx_chars  = getattr(active_eng, "ctx_value", DEFAULT_CTX) * 4
            prompt     = self.active.build_prompt(model_path=active_eng.model_path, max_chars=ctx_chars)
            cfg_pred   = MODEL_REGISTRY.get_config(active_eng.model_path).n_predict
            self._stream_w = self.chat_area.add_message("assistant", "", ts, tag="💻 Coding")
            self._worker = active_eng.create_worker(prompt, n_predict=cfg_pred, model_path=active_eng.model_path)
            self._worker.token.connect(self._on_token)
            self._worker.done.connect(self._on_done)
            self._worker.err.connect(self._on_err)
            self._worker.start()
            self.input_bar.set_generating(True)
            return

        n_engines = len(insight_engines)
        self._log("INFO", f"🔗 Pipeline: {n_engines} insight engine(s) → coding")
        self.lbl_engine.setText("🧠 Structural Insights…")
        self.lbl_engine.setStyleSheet(f"color:{C['pipeline']};padding:0 8px;")
        self.input_bar.set_generating(True)

        # Create one bubble per insight engine + divider + coding bubble
        self._pipeline_insight_widgets = []
        for label, eng in insight_engines:
            w = self.chat_area.add_message("assistant", "", ts, tag=label)
            self._pipeline_insight_widgets.append(w)

        self.chat_area.add_pipeline_divider(
            f"{n_engines} model(s) analysed → Coding model generating"
        )
        self._pipeline_reason_w = self._pipeline_insight_widgets[0] if self._pipeline_insight_widgets else None
        self._pipeline_code_w   = self.chat_area.add_message("assistant", "", ts, tag="💻 Coding")

        insight_np = max(
            (MODEL_REGISTRY.get_config(eng.model_path).n_predict for _, eng in insight_engines),
            default=512
        )
        code_np = MODEL_REGISTRY.get_config(self.coding_engine.model_path).n_predict

        self._pipeline_worker = PipelineWorker(
            insight_engines, self.coding_engine, text,
            n_predict_insight=insight_np, n_predict_code=max(code_np, 1024)
        )
        self._pipeline_worker.insight_started.connect(self._on_pipeline_insight_started)
        self._pipeline_worker.insight_token.connect(self._on_pipeline_insight_token)
        self._pipeline_worker.insight_done.connect(self._on_pipeline_insight_done)
        self._pipeline_worker.coding_token.connect(self._on_pipeline_code_token)
        self._pipeline_worker.coding_done.connect(self._on_pipeline_done)
        self._pipeline_worker.stage_changed.connect(self._on_pipeline_stage)
        self._pipeline_worker.err.connect(self._on_pipeline_err)
        self._pipeline_worker.start()

    def _on_pipeline_insight_started(self, idx: int, label: str):
        self.lbl_engine.setText(f"{label} analysing…")
        self.lbl_engine.setStyleSheet(f"color:{C['pipeline']};padding:0 8px;")

    def _on_pipeline_insight_token(self, idx: int, token: str):
        widgets = getattr(self, "_pipeline_insight_widgets", [])
        if idx < len(widgets) and widgets[idx]:
            try:
                widgets[idx].append_text(token)
            except RuntimeError:
                pass
        self.chat_area._scroll_bottom()

    def _on_pipeline_insight_done(self, idx: int, full_text: str):
        widgets = getattr(self, "_pipeline_insight_widgets", [])
        if idx < len(widgets) and widgets[idx]:
            try:
                widgets[idx].finalize()
            except RuntimeError:
                pass

    # keep old name for compat
    def _on_pipeline_reason_token(self, text: str):
        self._on_pipeline_insight_token(0, text)

    def _on_pipeline_reason_done(self, full_text: str):
        self._on_pipeline_insight_done(0, full_text)

    def _on_pipeline_stage(self, stage: str):
        if stage == "coding":
            self.lbl_engine.setText("💻 Coding…")
            self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        elif stage == "insights":
            self.lbl_engine.setText("🧠 Structural Insights…")
            self.lbl_engine.setStyleSheet(f"color:{C['pipeline']};padding:0 8px;")

    def _on_pipeline_code_token(self, text: str):
        if self._pipeline_code_w:
            try:
                self._pipeline_code_w.append_text(text)
            except RuntimeError:
                pass
        self.chat_area._scroll_bottom()

    def _on_pipeline_done(self, tps: float):
        self.tps_lbl.setText(f"{tps:.1f} tok/s")
        self.lbl_engine.setText(self.engine.status_text)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self.input_bar.set_generating(False)

        # Collect all insight texts for session save
        insight_parts = []
        for w in getattr(self, "_pipeline_insight_widgets", []):
            if w:
                try:
                    w.finalize()
                    t = w.full_text.strip()
                    if t:
                        insight_parts.append(t)
                except RuntimeError:
                    pass

        code_text = ""
        if self._pipeline_code_w:
            try:
                self._pipeline_code_w.finalize()
                code_text = self._pipeline_code_w.full_text.strip()
            except RuntimeError:
                pass

        if self.active and (insight_parts or code_text):
            insights_joined = "\n\n---\n\n".join(
                f"**[Structural Insight {i+1}]**\n\n{t}"
                for i, t in enumerate(insight_parts)
            )
            self.active.add_message("assistant",
                f"{insights_joined}\n\n---\n\n**[Code Output]**\n\n{code_text}")
            self.active.save()

        self._pipeline_insight_widgets = []
        self._pipeline_reason_w = None
        self._pipeline_code_w   = None
        self._pipeline_worker   = None
        self._update_ctx_bar()

    def _on_pipeline_err(self, msg: str):
        self._log("ERROR", f"Pipeline error: {msg}")
        self.lbl_engine.setText("❌  Pipeline Error")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")
        self.input_bar.set_generating(False)
        if self._pipeline_code_w:
            try:
                self._pipeline_code_w.append_text(f"\n\n⚠️ Pipeline error: {msg}")
            except RuntimeError:
                pass
        self._pipeline_insight_widgets = []
        self._pipeline_reason_w = None
        self._pipeline_code_w   = None
        self._pipeline_worker   = None

    # ── normal streaming handlers ──────────────────────────────────────────────

    def _on_token(self, text: str):
        if not self._stream_w: return
        try:
            self._stream_w.append_text(text)
        except RuntimeError:
            return
        self.chat_area._scroll_bottom()

    def _on_done(self, tps: float):
        self.tps_lbl.setText(f"{tps:.1f} tok/s")
        self.lbl_engine.setText(self.engine.status_text)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self.input_bar.set_generating(False)
        if self._stream_w:
            try: self._stream_w.finalize()
            except RuntimeError: pass
        self._save_streamed()

    def _on_err(self, msg: str):
        self._log("ERROR", msg)
        self.lbl_engine.setText("❌  Error")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")
        self.input_bar.set_generating(False)
        if self._stream_w:
            self._stream_w.append_text(f"\n\n⚠️ Error: {msg}")
        if self._stream_w:
            try: self._stream_w.finalize()
            except RuntimeError: pass
        self._save_streamed()

    def _on_stop(self):
        if self._pipeline_worker:
            if hasattr(self._pipeline_worker, "abort"):
                self._pipeline_worker.abort()
            self._pipeline_worker.wait(2000)
            self._pipeline_worker = None
            for w in (self._pipeline_reason_w, self._pipeline_code_w):
                if w:
                    try: w.finalize()
                    except RuntimeError: pass
            self._pipeline_reason_w = None
            self._pipeline_code_w   = None

        if self._worker:
            if hasattr(self._worker, "abort"): self._worker.abort()
            self._worker.wait(2000)
            self._worker = None

        if self._summary_worker:
            if hasattr(self._summary_worker, "request_pause"):
                self._summary_worker.request_pause()
                self._summary_worker.wait(4000)
            else:
                if hasattr(self._summary_worker, "abort"):
                    self._summary_worker.abort()
                self._summary_worker.wait(2000)
            self._summary_worker = None
            if hasattr(self, "_summary_bubble") and self._summary_bubble:
                self._summary_bubble.append_text(
                    "\n\n⏸ Paused & saved to disk. Resume from the Config tab.")
                self._summary_bubble = None
            if hasattr(self, "config_tab"):
                self.config_tab.refresh_paused_jobs()

        if self._multi_pdf_worker:
            if hasattr(self._multi_pdf_worker, "request_pause"):
                self._multi_pdf_worker.request_pause()
                self._multi_pdf_worker.wait(4000)
            else:
                if hasattr(self._multi_pdf_worker, "abort"):
                    self._multi_pdf_worker.abort()
                self._multi_pdf_worker.wait(2000)
            self._multi_pdf_worker = None
            if hasattr(self, "_summary_bubble") and self._summary_bubble:
                self._summary_bubble.append_text(
                    "\n\n⏸ Multi-PDF paused & saved. Resume from the Config tab.")
                self._summary_bubble = None
            if hasattr(self, "config_tab"):
                self.config_tab.refresh_paused_jobs()

        if self._stream_w:
            try: self._stream_w.finalize()
            except RuntimeError: pass

        self.input_bar.set_generating(False)
        self.lbl_engine.setText(self.engine.status_text)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self._log("INFO", "Generation stopped by user.")
        self._save_streamed(suffix=" ✋")

    def _save_streamed(self, suffix: str = ""):
        content = ""
        if self._stream_w:
            try:
                self._stream_w._flush_timer.stop()
                self._stream_w._flush_pending()
                content = self._stream_w.full_text.strip()
            except RuntimeError:
                content = ""
        if content and self.active:
            self.active.add_message("assistant", content + suffix)
            self.active.save()
        self._stream_w = None
        self._worker   = None
        self._update_ctx_bar()

    # ── PDF loading ────────────────────────────────────────────────────────────

    def _load_pdf(self):
        if not HAS_PDF:
            QMessageBox.warning(self, "Missing Dependency",
                                "Install PyPDF2:  pip install PyPDF2"); return
        if not self.engine.is_loaded:
            QMessageBox.warning(self, "Model Not Ready",
                                "Wait for the model to finish loading."); return

        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if not path: return

        reader   = PdfReader(path)
        n_pages  = len(reader.pages)
        filename = Path(path).name
        self._log("INFO", f"Reading PDF: {filename}  ({n_pages} pages)")

        text = ""
        for i, page in enumerate(reader.pages):
            text += page.extract_text() or ""
            if i % 10 == 0:
                self._log("INFO", f"  …extracted page {i+1}/{n_pages}")

        if not text.strip():
            QMessageBox.warning(self, "Empty PDF", "No text could be extracted."); return

        ctx_chars   = getattr(self.engine, "ctx_value", DEFAULT_CTX) * 3
        DIRECT_LIMIT = min(ctx_chars, 6000)

        if len(text) <= DIRECT_LIMIT:
            self.input_bar.input.setPlainText(
                f"The following is extracted from '{filename}':\n\n{text}"
                "\n\nPlease summarise the key points.")
            self._log("INFO", "Short document — loaded directly into prompt.")
            return

        if not self.active: self._new_session()
        self.active.title = f"Summary: {filename}"
        self.active.save()
        self._refresh_sidebar()

        ts = datetime.now().strftime("%H:%M")
        self._summary_bubble = self.chat_area.add_message(
            "assistant",
            f"📄  Starting chunked summarisation of **{filename}** "
            f"({n_pages} pages, {len(text):,} chars)…\n", ts)

        self.input_bar.set_generating(True)
        self.lbl_engine.setText("📄  Summarising…")
        self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")

        if self._summary_worker:
            if hasattr(self._summary_worker, "abort"): self._summary_worker.abort()
            self._summary_worker.wait(1000)

        from math import ceil
        estimated_chunks = ceil(len(text) / int(APP_CONFIG["summary_chunk_chars"]))
        self._thinking_block = self.chat_area.add_thinking_block(estimated_chunks)

        self._summary_worker = ChunkedSummaryWorker(
            self.engine, text, filename,
            engine2=getattr(self, "summarization_engine", None) or
                    getattr(self, "reasoning_engine", None),
            session_id=self.active.id if self.active else "")
        self._summary_worker.progress.connect(self._on_summary_progress)
        self._summary_worker.section_done.connect(self._on_section_done)
        self._summary_worker.final_done.connect(self._on_summary_final)
        self._summary_worker.err.connect(self._on_summary_err_or_pause)
        self._summary_worker.pause_suggest.connect(self._on_pause_suggest)
        self._summary_worker.start()

    def _on_summary_progress(self, msg: str):
        self._log("INFO", msg)
        self.lbl_engine.setText(f"📄  {msg}")
        if "final consolidation" in msg.lower():
            try:
                tb = getattr(self, "_thinking_block", None)
                if tb is not None: tb.add_phase("Final consolidation pass")
            except RuntimeError:
                pass

    def _on_section_done(self, num: int, total: int, chunk_text: str, summary: str):
        self._log("INFO", f"Section {num}/{total} done ({len(summary)} chars)")
        try:
            tb = getattr(self, "_thinking_block", None)
            if tb is not None: tb.add_section(num, total, chunk_text, summary)
        except RuntimeError:
            pass
        try:
            bubble = getattr(self, "_summary_bubble", None)
            if bubble is not None:
                bubble.append_text(f"\n✅ Section {num}/{total} summarised.\n")
        except RuntimeError:
            pass
        self.chat_area._scroll_bottom()

    def _on_summary_final(self, final: str):
        self._log("INFO", f"Final summary ready ({len(final)} chars)")
        try:
            bubble = getattr(self, "_summary_bubble", None)
            if bubble is not None:
                bubble._flush_timer.stop()
                bubble._flush_pending()
                bubble.append_text("\n\n---\n**Final Summary:**\n\n" + final)
                QTimer.singleShot(80, bubble._flush_pending)
        except RuntimeError:
            pass
        finally:
            self._summary_bubble = None
        try:
            tb = getattr(self, "_thinking_block", None)
            if tb is not None: tb.mark_done()
        except RuntimeError:
            pass
        finally:
            self._thinking_block = None

        final_text = final
        def _persist():
            try:
                if self.active:
                    self.active.add_message("assistant", f"**Document Summary**\n\n{final_text}")
                    self.active.save()
            except Exception as e:
                self._log("WARN", f"Could not save final summary: {e}")
            self._summary_worker = None
            self.input_bar.set_generating(False)
            self.lbl_engine.setText(self.engine.status_text)
            self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
            self._update_ctx_bar()
        QTimer.singleShot(120, _persist)

    def _on_multi_pdf_ram_warning(self, msg: str):
        self._log("WARN", msg)
        bubble = getattr(self, "_summary_bubble", None)
        if bubble:
            try:
                bubble.append_text(f"\n{msg}\n")
            except RuntimeError:
                pass

    def _start_multi_pdf(self, paths: List[str]):
        """Start multi-PDF summarization with adaptive RAM watchdog."""
        if not HAS_PDF:
            QMessageBox.warning(self, "Missing Dep", "Install PyPDF2"); return
        if not self.engine.is_loaded:
            QMessageBox.warning(self, "Not Ready", "Wait for model to load."); return

        pdf_texts: List[Tuple[str, str]] = []
        for path in paths:
            try:
                reader = PdfReader(path)
                text   = "\n".join(p.extract_text() or "" for p in reader.pages)
                if text.strip():
                    pdf_texts.append((Path(path).name, text))
                    self._log("INFO", f"Loaded: {Path(path).name} ({len(text):,} chars)")
            except Exception as e:
                self._log("WARN", f"Could not read {path}: {e}")

        if not pdf_texts:
            QMessageBox.warning(self, "No Content", "No readable PDFs found."); return

        if not self.active: self._new_session()
        self.active.title = f"Multi-PDF: {len(pdf_texts)} docs"
        self.active.save()
        self._refresh_sidebar()

        ts = datetime.now().strftime("%H:%M")
        self._summary_bubble = self.chat_area.add_message(
            "assistant",
            f"📚  Starting multi-PDF summarization:\n"
            + "\n".join(f"  • {fn} ({len(t):,} chars)" for fn, t in pdf_texts)
            + f"\n\n⏳ Processing…\n", ts)

        from math import ceil
        total_chunks = sum(
            ceil(len(t) / int(APP_CONFIG["summary_chunk_chars"])) for _, t in pdf_texts)
        self._thinking_block = self.chat_area.add_thinking_block(total_chunks)

        self.input_bar.set_generating(True)
        self.lbl_engine.setText("📚  Multi-PDF…")
        self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")

        sid = self.active.id if self.active else "default"
        self._multi_pdf_worker = MultiPdfSummaryWorker(
            self.engine, pdf_texts, sid,
            engine2=getattr(self, "summarization_engine", None) or
                    getattr(self, "reasoning_engine", None))
        self._multi_pdf_worker.file_started.connect(
            lambda fi, fn, nc: self._log("INFO", f"PDF {fi+1}: {fn} ({nc} chunks)"))
        self._multi_pdf_worker.file_progress.connect(
            lambda fi, msg: self.lbl_engine.setText(f"📄 PDF {fi+1}: {msg}"))
        self._multi_pdf_worker.ram_warning.connect(self._on_multi_pdf_ram_warning)
        self._multi_pdf_worker.section_done.connect(self._on_section_done)
        self._multi_pdf_worker.file_done.connect(
            lambda fi, s: (
                self._log("INFO", f"File {fi+1} summary done ({len(s)} chars)"),
                self._summary_bubble and self._summary_bubble.append_text(
                    f"\n✅ Document {fi+1} summarised.\n")))
        self._multi_pdf_worker.progress.connect(self._on_summary_progress)
        self._multi_pdf_worker.final_done.connect(self._on_multi_pdf_final)
        self._multi_pdf_worker.err.connect(self._on_summary_err_or_pause)
        self._multi_pdf_worker.pause_suggest.connect(self._on_pause_suggest)
        self._multi_pdf_worker.start()

    def _on_multi_pdf_final(self, final: str):
        self._log("INFO", f"Multi-PDF final summary ready ({len(final)} chars)")
        try:
            bubble = getattr(self, "_summary_bubble", None)
            if bubble:
                bubble.append_text("\n\n---\n**Final Multi-Document Summary:**\n\n" + final)
                QTimer.singleShot(80, bubble._flush_pending)
        except RuntimeError:
            pass
        self._summary_bubble = None
        try:
            tb = getattr(self, "_thinking_block", None)
            if tb: tb.mark_done()
        except RuntimeError:
            pass
        self._thinking_block = None

        if self.active:
            self.active.add_message("assistant",
                f"**Multi-Document Summary**\n\n{final}")
            self.active.save()

        self._multi_pdf_worker = None
        self.input_bar.set_generating(False)
        self.lbl_engine.setText(self.engine.status_text)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self._update_ctx_bar()

    def _on_config_changed(self):
        self._log("INFO", "App config updated and saved.")

    def _resume_paused_job(self):
        state = self.config_tab.get_selected_job_state()
        if not state:
            QMessageBox.warning(self, "No Job Selected", "Select a paused job first.")
            return
        if not self.engine.is_loaded:
            QMessageBox.warning(self, "Not Ready", "Wait for model to load."); return

        raw_text = state.get("raw_text", "")        
        filename   = state.get("filename", "unknown")
        job_id     = state.get("job_id", "")
        session_id = state.get("session_id", "")
        pdf_texts  = state.get("pdf_texts", None)   # present for multi-PDF jobs

        # ── Multi-PDF resume ──────────────────────────────────────────────────
        if pdf_texts is not None:
            if not self.engine.is_loaded:
                QMessageBox.warning(self, "Not Ready", "Wait for model to load.")
                return
            if not self.active: self._new_session()
            self.active.title = f"Resume Multi-PDF: {filename}"
            self.active.save(); self._refresh_sidebar()
            ts = datetime.now().strftime("%H:%M")
            next_fi = state.get("next_fi", 0)
            next_ci = state.get("next_ci", 0)
            total   = state.get("total", len(pdf_texts))
            self._summary_bubble = self.chat_area.add_message(
                "assistant",
                f"▶  Resuming multi-PDF job from file {next_fi+1}/{total}, "
                f"chunk {next_ci+1}…\n", ts)
            from math import ceil
            est = sum(
                ceil(len(t) / int(APP_CONFIG["summary_chunk_chars"]))
                for _, t in pdf_texts[next_fi:])
            self._thinking_block = self.chat_area.add_thinking_block(max(est, 1))
            self.input_bar.set_generating(True)
            self.lbl_engine.setText("▶  Resuming Multi-PDF…")
            self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")
            sid = self.active.id if self.active else "default"
            self._multi_pdf_worker = MultiPdfSummaryWorker(
                self.engine,
                [(fn, txt) for fn, txt in pdf_texts],
                sid,
                engine2=getattr(self, "summarization_engine", None) or
                        getattr(self, "reasoning_engine", None),
                resume_job_id=job_id)
            self._multi_pdf_worker.file_started.connect(
                lambda fi, fn, nc: self._log("INFO", f"PDF {fi+1}: {fn} ({nc} chunks)"))
            self._multi_pdf_worker.file_progress.connect(
                lambda fi, msg: self.lbl_engine.setText(f"📄 PDF {fi+1}: {msg}"))
            self._multi_pdf_worker.ram_warning.connect(
                lambda msg: self._log("WARN", msg))
            self._multi_pdf_worker.section_done.connect(self._on_section_done)
            self._multi_pdf_worker.file_done.connect(
                lambda fi, s: self._log("INFO", f"File {fi+1} done ({len(s)} chars)"))
            self._multi_pdf_worker.progress.connect(self._on_summary_progress)
            self._multi_pdf_worker.final_done.connect(self._on_multi_pdf_final)
            self._multi_pdf_worker.err.connect(self._on_summary_err_or_pause)
            self._multi_pdf_worker.pause_suggest.connect(self._on_pause_suggest)
            self._multi_pdf_worker.start()
            self.config_tab.refresh_paused_jobs()
            self._log("INFO", f"Resumed multi-PDF job: {job_id}")
            return

        # ── Single-PDF resume (original path) ─────────────────────────────────
        if not raw_text:
            QMessageBox.warning(self, "No Data", "Paused job has no raw text saved.")
            return

        if not self.active: self._new_session()
        self.active.title = f"Resume: {filename}"
        self.active.save()
        self._refresh_sidebar()

        ts = datetime.now().strftime("%H:%M")
        next_chunk = state.get("next_chunk", 0)
        total      = state.get("total", "?")
        self._summary_bubble = self.chat_area.add_message(
            "assistant",
            f"▶  Resuming summarization of **{filename}** "
            f"from chunk {next_chunk}/{total}…\n", ts)

        from math import ceil
        est_remaining = (total - next_chunk) if isinstance(total, int) else 5
        self._thinking_block = self.chat_area.add_thinking_block(
            max(est_remaining, 1))

        self.input_bar.set_generating(True)
        self.lbl_engine.setText(f"▶  Resuming '{filename}'…")
        self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")

        self._summary_worker = ChunkedSummaryWorker(
            self.engine, raw_text, filename,
            engine2=getattr(self, "summarization_engine", None) or
                    getattr(self, "reasoning_engine", None),
            resume_job_id=job_id,
            session_id=session_id or (self.active.id if self.active else ""))
        self._summary_worker.progress.connect(self._on_summary_progress)
        self._summary_worker.section_done.connect(self._on_section_done)
        self._summary_worker.final_done.connect(self._on_summary_final)
        self._summary_worker.err.connect(self._on_summary_err_or_pause)
        self._summary_worker.pause_suggest.connect(self._on_pause_suggest)
        self._summary_worker.start()
        self.config_tab.refresh_paused_jobs()
        self._log("INFO", f"Resumed paused job: {job_id}")

    def _on_pause_suggest(self, job_id: str):
        """Show pause banner in chat when auto-pause threshold is hit."""
        ts = datetime.now().strftime("%H:%M")
        banner = self.chat_area.add_message(
            "assistant",
            f"💡 **Pause available** — The summarization has processed "
            f"{APP_CONFIG['pause_after_chunks']} chunks. You can pause & save "
            f"state now to resume later, or continue processing.\n\n"
            f"Click ⏸ **Pause** in the stop button area to pause, "
            f"or ignore this to keep going.", ts)

    def _on_summary_err_or_pause(self, msg: str):
        if msg.startswith("__PAUSED__:"):
            job_id = msg.split(":", 1)[1]
            self._log("INFO", f"Job paused: {job_id}")
            self.input_bar.set_generating(False)
            self.lbl_engine.setText("⏸  Paused — state saved")
            self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")
            self._summary_worker = None
            if hasattr(self, "config_tab"):
                self.config_tab.refresh_paused_jobs()
            return
        self._on_summary_err(msg)

    def _on_summary_err(self, msg: str):
        self._log("ERROR", f"Summary pipeline error: {msg}")
        if hasattr(self, "_summary_bubble") and self._summary_bubble:
            self._summary_bubble.append_text(f"\n\n⚠️ Error: {msg}")
        self._summary_worker = None
        self._summary_bubble = None
        self.input_bar.set_generating(False)
        self.lbl_engine.setText("❌  Summary failed")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")

    # ── export ────────────────────────────────────────────────────────────────

    def _export_active(self, fmt: str):
        if self.active: self._export_session(self.active.id, fmt)

    def _export_session(self, sid: str, fmt: str):
        s = self.sessions.get(sid)
        if not s: return
        ext_map = {"json": "JSON (*.json)", "md": "Markdown (*.md)", "txt": "Plain Text (*.txt)"}
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Session", f"{s.id}.{fmt}", ext_map.get(fmt, "*"))
        if not path: return
        content = {"json": s.to_json, "md": s.to_markdown, "txt": s.to_txt}.get(fmt, s.to_json)()
        Path(path).write_text(content, encoding="utf-8")
        self._log("INFO", f"Exported to {path}")

    # ── status bar helpers ────────────────────────────────────────────────────

    def _update_ctx_bar(self):
        if not self.active: return
        est     = self.active.approx_tokens
        max_ctx = getattr(self.engine, "ctx_value", self.ctx_slider.value())
        self.ctx_bar.setRange(0, max_ctx)
        self.ctx_bar.setValue(min(est, max_ctx))
        self.ctx_lbl.setText(f"{est:,} / {max_ctx:,}")
        pct   = est / max_ctx if max_ctx > 0 else 0
        color = C["ok"] if pct < 0.6 else C["warn"] if pct < 0.85 else C["err"]
        self.ctx_bar.setStyleSheet(
            f"QProgressBar{{background:{C['bg2']};border:1px solid {C['bdr']};"
            f"border-radius:3px;height:8px;}}"
            f"QProgressBar::chunk{{background:{color};border-radius:3px;}}"
        )

    def _update_ram(self):
        mem  = psutil.virtual_memory()
        used = mem.used  // (1024 ** 3)
        tot  = mem.total // (1024 ** 3)
        self.ram_lbl.setText(f"RAM: {used}/{tot} GB")

    def _log(self, level: str, msg: str):
        if hasattr(self, "log_console"):
            self.log_console.log(level, msg)
        else:
            print(f"[{level}] {msg}", file=sys.stderr)

    def _toggle_sidebar(self):
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def _goto_logs(self):
        self.tabs.setCurrentWidget(self.log_console)

    def _goto_models_tab(self):
        self.tabs.setCurrentWidget(self.models_tab)

    def closeEvent(self, event):
        self._log("INFO", "Shutdown — stopping engine…")
        if self._worker:
            try:
                if hasattr(self._worker, "abort"): self._worker.abort()
                self._worker.wait(1000)
            except Exception:
                pass
        if self._pipeline_worker:
            try:
                self._pipeline_worker.abort()
                self._pipeline_worker.wait(1000)
            except Exception:
                pass
        self.engine.shutdown()
        for role in ("reasoning", "summarization", "coding", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng: eng.shutdown()
        super().closeEvent(event)


# ═════════════════════════════ ENTRY POINT ══════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Native Lab Pro")
    app.setWindowIcon(QIcon('icon.png'))
    app.setFont(QFont("Segoe UI", 11))
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    win = MainWindow()
    win.setWindowTitle("✦  Native Lab Pro  v2")
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()