from imports.import_global import Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model_family import ModelFamily

from .model_family import ModelFamily
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

API_PROVIDERS: Dict[str, Dict] = {
    "OpenAI":     {"base_url": "https://api.openai.com/v1",          "format": "openai",
                   "models": ["gpt-4o","gpt-4o-mini","o1","o1-mini","o3-mini","gpt-4-turbo","gpt-3.5-turbo"]},
    "Anthropic":  {"base_url": "https://api.anthropic.com",           "format": "anthropic",
                   "models": ["claude-opus-4-5","claude-sonnet-4-5","claude-haiku-4-5",
                               "claude-3-5-sonnet-20241022","claude-3-haiku-20240307"]},
    "Groq":       {"base_url": "https://api.groq.com/openai/v1",      "format": "openai",
                   "models": ["llama-3.3-70b-versatile","llama-3.1-70b-versatile",
                               "llama-3.1-8b-instant","mixtral-8x7b-32768","gemma2-9b-it"]},
    "Mistral":    {"base_url": "https://api.mistral.ai/v1",           "format": "openai",
                   "models": ["mistral-large-latest","mistral-medium-latest",
                               "mistral-small-latest","codestral-latest"]},
    "Together AI":{"base_url": "https://api.together.xyz/v1",         "format": "openai",
                   "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo",
                               "meta-llama/Llama-3.1-8B-Instruct-Turbo",
                               "mistralai/Mixtral-8x7B-Instruct-v0.1",
                               "Qwen/Qwen2.5-72B-Instruct-Turbo"]},
    "OpenRouter": {"base_url": "https://openrouter.ai/api/v1",        "format": "openai",
                   "models": ["openai/gpt-4o","anthropic/claude-3.5-sonnet",
                               "meta-llama/llama-3.3-70b-instruct",
                               "google/gemini-pro-1.5","deepseek/deepseek-r1"]},
    "Ollama":     {"base_url": "http://localhost:11434/v1",            "format": "openai",
                   "models": ["llama3.2","llama3.1","qwen2.5","phi4","mistral","deepseek-r1"]},
    "Custom":     {"base_url": "",                                     "format": "openai",
                   "models": []},
}


PROMPT_TEMPLATES: Dict[str, Dict] = {
    "default":   {"label": "Default (provider handles it)",
                  "system": "", "user_prefix": "", "user_suffix": "",
                  "assistant_prefix": ""},
    "chatml":    {"label": "ChatML  (OpenHermes, Qwen, etc.)",
                  "system": "You are a helpful assistant.",
                  "user_prefix": "<|im_start|>user\n", "user_suffix": "<|im_end|>\n",
                  "assistant_prefix": "<|im_start|>assistant\n"},
    "llama2":    {"label": "Llama-2 Chat",
                  "system": "You are a helpful assistant.",
                  "user_prefix": "[INST] ", "user_suffix": " [/INST]",
                  "assistant_prefix": ""},
    "alpaca":    {"label": "Alpaca / Vicuna",
                  "system": "Below is an instruction that describes a task. Write a response.",
                  "user_prefix": "### Instruction:\n", "user_suffix": "\n### Response:\n",
                  "assistant_prefix": ""},
    "gemma":     {"label": "Gemma",
                  "system": "",
                  "user_prefix": "<start_of_turn>user\n", "user_suffix": "<end_of_turn>\n",
                  "assistant_prefix": "<start_of_turn>model\n"},
    "phi3":      {"label": "Phi-3",
                  "system": "",
                  "user_prefix": "<|user|>\n", "user_suffix": "<|end|>\n",
                  "assistant_prefix": "<|assistant|>\n"},
    "custom":    {"label": "Custom (fill in below)",
                  "system": "", "user_prefix": "", "user_suffix": "",
                  "assistant_prefix": ""},
}

# ----- for pdf summarization component -----
MODE_SECTION_INSTRUCTIONS = {
    "summary": (
        "Summarise this section clearly. Retain all key facts, named entities, "
        "numbers, arguments, and logical connections. Use bullet points for key facts."
    ),
    "logical": (
        "Explain the logic, mechanism, or methodology described in this section. "
        "Break it down step by step. Use numbered points and sub-bullets. "
        "Preserve technical accuracy. Focus on HOW and WHY things work."
    ),
    "advice": (
        "Extract actionable advice, recommendations, or implications from this section. "
        "Frame findings as practical advice. Use bullet points. "
        "Focus on what the reader should DO or KNOW based on this content."
    ),
}

MODE_FINAL_INSTRUCTIONS = {
    "summary": (
        "Write a single, well-structured, coherent final summary.\n"
        "Use clear headings (##) and bullet points for key facts.\n"
        "Do NOT omit any core findings, data points, or named entities.\n"
        "Structure: Overview → Key Points → Findings → Conclusion."
    ),
    "logical": (
        "Write a structured explanation of the complete logic, mechanism, and methodology "
        "of this document.\n"
        "Use ## headings for each major concept or process.\n"
        "Use numbered steps and sub-bullets for mechanisms.\n"
        "Explain the WHY and HOW at each stage.\n"
        "Structure: Core Premise → Mechanisms → Process Flow → Key Findings → Implications."
    ),
    "advice": (
        "Write a structured advisory brief based on the full document.\n"
        "Use ## headings for each advice category.\n"
        "Use bullet points for every recommendation.\n"
        "Make all advice actionable and specific.\n"
        "Structure: Key Takeaways → Recommendations → What To Do → What To Avoid → Next Steps."
    ),
}

#--- templates for script parser -----
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