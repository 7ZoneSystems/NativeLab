from nativelab.imports.import_global import List, field, dataclass, Dict, Path, Tuple, re
from nativelab.GlobalConfig.config import QUANT_REGEX, GGUF_QUANT_PATTERNS 
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


@dataclass
class VisionModelInfo:
    is_vision: bool = False
    label: str = ""
    family_hint: str = ""
    needs_mmproj: bool = True


_VLM_PATTERNS: List[Tuple[List[str], str, str, bool]] = [
    (["llava", "bakllava", "moondream"], "LLaVA VLM", "llava", True),
    (["minicpm-v", "minicpmv", "minicpm-o", "minicpmo"], "MiniCPM-V", "minicpm-v", True),
    (["qwen2-vl", "qwen2.5-vl", "qwen-vl", "qwen2vl", "qwen25vl", "qwenvl"], "Qwen-VL", "qwen-vl", True),
    (["internvl", "intern-vl"], "InternVL", "internvl", True),
    (["pixtral"], "Pixtral", "pixtral", True),
    (["paligemma"], "PaliGemma", "paligemma", True),
    (["gemma-3", "gemma3"], "Gemma 3 Vision", "gemma3", True),
    (["llama-vision", "llama3.2-vision", "llama-3.2-vision", "vision-instruct"], "Llama Vision", "llama-vision", True),
    (["mllama", "multi-modal", "multimodal", "vlm"], "Vision-Language Model", "vlm", True),
]


def detect_vision_model(filename: str) -> VisionModelInfo:
    """Detect whether a model filename looks like a vision-language GGUF."""
    name = Path(filename).stem.lower()
    for keywords, label, hint, needs_mmproj in _VLM_PATTERNS:
        if any(kw in name for kw in keywords):
            return VisionModelInfo(True, label, hint, needs_mmproj)
    return VisionModelInfo()


def detect_mmproj_for_model(model_path: str) -> str:
    """Find a likely llama.cpp multimodal projector next to a VLM model."""
    p = Path(model_path)
    if not p.exists():
        return ""
    folder = p.parent
    stem = p.stem.lower()
    candidates = []
    for f in folder.glob("*.gguf"):
        n = f.name.lower()
        if f == p:
            continue
        if any(k in n for k in ("mmproj", "projector", "vision", "clip")):
            score = 10
            for token in re.split(r'[-_. ]+', stem):
                if len(token) > 2 and token in n:
                    score += 1
            candidates.append((score, f))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: (-x[0], x[1].name.lower()))
    return str(candidates[0][1])

def detect_model_family(filename: str) -> ModelFamily:
    """
    Detect model family from GGUF filename and return the matching template.
    Uses prioritised pattern matching so more specific names win.
    """
    from .templates import FAMILY_TEMPLATES   # lazy import

    name = Path(filename).stem.lower()
    # Order matters - more specific patterns first
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
        match = re.search(pat, stem, re.IGNORECASE)
        if match:
            return match.group(0).upper()
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
