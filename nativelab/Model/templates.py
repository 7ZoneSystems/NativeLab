from nativelab.imports.import_global import Dict
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
                   "models": ["gpt-5.5","gpt-5","gpt-5-mini","o3","o4-mini","gpt-4.1","gpt-4.1-mini","gpt-4o","gpt-4o-mini"]},
    "Anthropic":  {"base_url": "https://api.anthropic.com",           "format": "anthropic",
                   "models": ["claude-opus-4-7","claude-opus-4-6","claude-sonnet-4-6",
                               "claude-haiku-4-5-20251001"]},
    "Groq":       {"base_url": "https://api.groq.com/openai/v1",      "format": "openai",
                   "models": ["llama-3.3-70b-versatile","llama-3.1-8b-instant",
                               "meta-llama/llama-4-scout-17b-16e-instruct",
                               "meta-llama/llama-4-maverick-17b-128e-instruct",
                               "openai/gpt-oss-120b"]},
    "Mistral":    {"base_url": "https://api.mistral.ai/v1",           "format": "openai",
                   "models": ["mistral-large-latest","mistral-medium-latest",
                               "mistral-small-latest","codestral-latest",
                               "magistral-medium-latest","magistral-small-latest"]},
    "Together AI":{"base_url": "https://api.together.xyz/v1",         "format": "openai",
                   "models": ["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                               "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                               "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                               "Qwen/Qwen3.5-397B-A17B",
                               "openai/gpt-oss-120b"]},
    "OpenRouter": {"base_url": "https://openrouter.ai/api/v1",        "format": "openai",
                   "models": ["openai/gpt-4.1","openai/o3",
                               "anthropic/claude-sonnet-4-6",
                               "meta-llama/llama-4-maverick",
                               "google/gemini-2.5-pro","deepseek/deepseek-r1"]},
    "Ollama":     {"base_url": "http://localhost:11434/v1",            "format": "openai",
                   "models": ["llama4","llama3.3","qwen3.5","qwen3.6","gemma4","phi4","mistral","deepseek-r1"]},
    "Doubleword": {"base_url": "https://api.doubleword.ai/v1",         "format": "openai",
                   "models": ["deepseek-ai/DeepSeek-V4-Pro",
                               "deepseek-ai/DeepSeek-V4-Flash",
                               "moonshotai/Kimi-K2.6",
                               "mistralai/Devstral-2-123B-Instruct-2512",
                               "Qwen/Qwen3.6-35B-A3B-FP8",
                               "Qwen/Qwen3.5-397B-A17B",
                               "Qwen/Qwen3.5-35B-A3B-FP8",
                               "Qwen/Qwen3.5-9B",
                               "Qwen/Qwen3.5-4B",
                               "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
                               "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
                               "Qwen/Qwen3-14B-FP8",
                               "google/gemma-4-31B-it",
                               "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
                               "zai-org/GLM-5.1-FP8",
                               "openai/gpt-oss-20b"]},
    "xAI (Grok)": {"base_url": "https://api.x.ai/v1",                 "format": "openai",
                   "models": ["grok-4.3","grok-4.20",
                               "grok-4-1-fast-reasoning","grok-4-1-fast-non-reasoning",
                               "grok-3-mini-beta","grok-3-mini-fast-beta"]},
    "Custom":     {"base_url": "",                                     "format": "openai",
                   "models": []},
}

POPULAR_MODEL_PRESETS: Dict[str, list] = {
    "gguf": [
        {"label": "Qwen3 4B Instruct 2507 - balanced small", "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "family": "qwen", "size": "4B", "task": "general"},
        {"label": "Qwen3 0.6B - tiny/fast", "repo": "Qwen/Qwen3-0.6B-GGUF", "family": "qwen", "size": "0.6B", "task": "tiny"},
        {"label": "Qwen3 4B - compact general", "repo": "Qwen/Qwen3-4B-GGUF", "family": "qwen", "size": "4B", "task": "general"},
        {"label": "Qwen3 14B - strong general", "repo": "Qwen/Qwen3-14B-GGUF", "family": "qwen", "size": "14B", "task": "general"},
        {"label": "Qwen3 30B A3B - MoE general", "repo": "Qwen/Qwen3-30B-A3B-GGUF", "family": "qwen", "size": "30B-A3B", "task": "general"},
        {"label": "Qwen3 Coder 30B A3B - coding", "repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "family": "qwen", "size": "30B-A3B", "task": "coding"},
        {"label": "Qwen2.5 Coder 7B - coding", "repo": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "family": "qwen", "size": "7B", "task": "coding"},
        {"label": "Qwen2.5 Coder 32B - coding", "repo": "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF", "family": "qwen", "size": "32B", "task": "coding"},
        {"label": "Qwen2.5 14B Instruct - multilingual", "repo": "bartowski/Qwen2.5-14B-Instruct-GGUF", "family": "qwen", "size": "14B", "task": "general"},
        {"label": "Llama 3.2 1B Instruct - tiny", "repo": "unsloth/Llama-3.2-1B-Instruct-GGUF", "family": "llama3", "size": "1B", "task": "tiny"},
        {"label": "Llama 3.2 3B Instruct - laptop friendly", "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF", "family": "llama3", "size": "3B", "task": "general"},
        {"label": "Llama 3.1 8B Instruct - popular baseline", "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "family": "llama3", "size": "8B", "task": "general"},
        {"label": "Llama 3.1 70B Instruct - high quality", "repo": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF", "family": "llama3", "size": "70B", "task": "general"},
        {"label": "DeepSeek R1 Distill Llama 8B - reasoning", "repo": "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "family": "deepseek-r1", "size": "8B", "task": "reasoning"},
        {"label": "DeepSeek R1 Distill Qwen 7B - reasoning", "repo": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF", "family": "deepseek-r1", "size": "7B", "task": "reasoning"},
        {"label": "DeepSeek R1 Distill Qwen 32B - reasoning", "repo": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF", "family": "deepseek-r1", "size": "32B", "task": "reasoning"},
        {"label": "DeepSeek R1 - full reasoning MoE", "repo": "unsloth/DeepSeek-R1-GGUF", "family": "deepseek-r1", "size": "671B", "task": "reasoning"},
        {"label": "Mistral 7B Instruct v0.3 - classic general", "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF", "family": "mistral", "size": "7B", "task": "general"},
        {"label": "Mistral Nemo 12B - long context", "repo": "bartowski/Mistral-Nemo-Instruct-2407-GGUF", "family": "mistral", "size": "12B", "task": "general"},
        {"label": "Devstral Small 2507 - coding agent", "repo": "mistralai/Devstral-Small-2507_gguf", "family": "mistral", "size": "24B", "task": "coding"},
        {"label": "Phi 3 Mini 4K - very small", "repo": "microsoft/Phi-3-mini-4k-instruct-gguf", "family": "phi3", "size": "3.8B", "task": "general"},
        {"label": "Phi 3.5 Mini - strong small model", "repo": "bartowski/Phi-3.5-mini-instruct-GGUF", "family": "phi3", "size": "3.8B", "task": "general"},
        {"label": "Phi 4 Mini Instruct - small general", "repo": "unsloth/Phi-4-mini-instruct-GGUF", "family": "phi", "size": "3.8B", "task": "general"},
        {"label": "Gemma 2 9B IT - efficient general", "repo": "bartowski/gemma-2-9b-it-GGUF", "family": "gemma", "size": "9B", "task": "general"},
        {"label": "Gemma 4 E2B IT - new small vision/text", "repo": "bartowski/google_gemma-4-E2B-it-GGUF", "family": "gemma", "size": "E2B", "task": "general"},
        {"label": "Gemma 4 E4B IT - new balanced", "repo": "bartowski/google_gemma-4-E4B-it-GGUF", "family": "gemma", "size": "E4B", "task": "general"},
        {"label": "Gemma 4 26B A4B IT - high quality", "repo": "bartowski/google_gemma-4-26B-A4B-it-GGUF", "family": "gemma", "size": "26B-A4B", "task": "general"},
        {"label": "GPT-OSS 20B - open-weight reasoning", "repo": "unsloth/gpt-oss-20b-GGUF", "family": "default", "size": "20B", "task": "reasoning"},
        {"label": "SQLCoder 7B 2 - SQL", "repo": "QuantFactory/sqlcoder-7b-2-GGUF", "family": "default", "size": "7B", "task": "coding"},
        {"label": "TinyLlama 1.1B - minimal hardware", "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "family": "llama2", "size": "1.1B", "task": "tiny"},
    ],
    "hf_transformers": [
        {"label": "Qwen3 4B Instruct 2507 - balanced", "repo": "Qwen/Qwen3-4B-Instruct-2507", "revision": "main", "family": "qwen", "vision": False},
        {"label": "Qwen3 4B Thinking 2507 - reasoning", "repo": "Qwen/Qwen3-4B-Thinking-2507", "revision": "main", "family": "qwen", "vision": False},
        {"label": "Qwen3 Coder 30B A3B - coding", "repo": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "revision": "main", "family": "qwen", "vision": False},
        {"label": "Qwen2.5 Coder 7B - coding", "repo": "Qwen/Qwen2.5-Coder-7B-Instruct", "revision": "main", "family": "qwen", "vision": False},
        {"label": "Qwen2.5 Coder 32B - coding", "repo": "Qwen/Qwen2.5-Coder-32B-Instruct", "revision": "main", "family": "qwen", "vision": False},
        {"label": "Qwen2.5 VL 3B - vision", "repo": "Qwen/Qwen2.5-VL-3B-Instruct", "revision": "main", "family": "qwen", "vision": True},
        {"label": "Qwen2.5 VL 7B - vision", "repo": "Qwen/Qwen2.5-VL-7B-Instruct", "revision": "main", "family": "qwen", "vision": True},
        {"label": "Llama 3.2 1B Instruct - tiny", "repo": "meta-llama/Llama-3.2-1B-Instruct", "revision": "main", "family": "llama3", "vision": False},
        {"label": "Llama 3.2 3B Instruct - laptop friendly", "repo": "meta-llama/Llama-3.2-3B-Instruct", "revision": "main", "family": "llama3", "vision": False},
        {"label": "Llama 3.1 8B Instruct - popular baseline", "repo": "meta-llama/Llama-3.1-8B-Instruct", "revision": "main", "family": "llama3", "vision": False},
        {"label": "Llama 3.2 11B Vision - vision", "repo": "meta-llama/Llama-3.2-11B-Vision-Instruct", "revision": "main", "family": "llama3", "vision": True},
        {"label": "DeepSeek R1 Distill Qwen 1.5B - reasoning", "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "revision": "main", "family": "deepseek-r1", "vision": False},
        {"label": "DeepSeek R1 Distill Qwen 7B - reasoning", "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "revision": "main", "family": "deepseek-r1", "vision": False},
        {"label": "DeepSeek R1 Distill Qwen 14B - reasoning", "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "revision": "main", "family": "deepseek-r1", "vision": False},
        {"label": "DeepSeek R1 Distill Llama 8B - reasoning", "repo": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "revision": "main", "family": "deepseek-r1", "vision": False},
        {"label": "Mistral 7B Instruct v0.3 - classic", "repo": "mistralai/Mistral-7B-Instruct-v0.3", "revision": "main", "family": "mistral", "vision": False},
        {"label": "Mistral Nemo Instruct 2407 - 12B", "repo": "mistralai/Mistral-Nemo-Instruct-2407", "revision": "main", "family": "mistral", "vision": False},
        {"label": "Devstral Small 2507 - coding agent", "repo": "mistralai/Devstral-Small-2507", "revision": "main", "family": "mistral", "vision": False},
        {"label": "Phi 4 Mini Instruct - small", "repo": "microsoft/Phi-4-mini-instruct", "revision": "main", "family": "phi", "vision": False},
        {"label": "Phi 3.5 Mini Instruct - small", "repo": "microsoft/Phi-3.5-mini-instruct", "revision": "main", "family": "phi3", "vision": False},
        {"label": "Gemma 2 2B IT - tiny", "repo": "google/gemma-2-2b-it", "revision": "main", "family": "gemma", "vision": False},
        {"label": "Gemma 2 9B IT - efficient", "repo": "google/gemma-2-9b-it", "revision": "main", "family": "gemma", "vision": False},
        {"label": "Gemma 3 4B IT - multimodal", "repo": "google/gemma-3-4b-it", "revision": "main", "family": "gemma", "vision": True},
        {"label": "SmolLM2 1.7B Instruct - tiny", "repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "revision": "main", "family": "default", "vision": False},
        {"label": "Granite 3.3 8B Instruct - enterprise", "repo": "ibm-granite/granite-3.3-8b-instruct", "revision": "main", "family": "default", "vision": False},
    ],
    "ollama": [
        {"label": "Llama 3.1 - popular general", "model": "llama3.1", "family": "llama3", "size": "8B/70B/405B", "vision": False},
        {"label": "Llama 3.2 - tiny general", "model": "llama3.2", "family": "llama3", "size": "1B/3B", "vision": False},
        {"label": "Llama 3.2 Vision - image input", "model": "llama3.2-vision", "family": "llama3", "size": "11B/90B", "vision": True},
        {"label": "Llama 3.3 - 70B quality", "model": "llama3.3", "family": "llama3", "size": "70B", "vision": False},
        {"label": "DeepSeek R1 - reasoning", "model": "deepseek-r1", "family": "deepseek-r1", "size": "1.5B-671B", "vision": False},
        {"label": "Qwen3 - tools/thinking", "model": "qwen3", "family": "qwen", "size": "0.6B-235B", "vision": False},
        {"label": "Qwen3.5 - multimodal/tools", "model": "qwen3.5", "family": "qwen", "size": "0.8B-122B", "vision": True},
        {"label": "Qwen3 VL - vision", "model": "qwen3-vl", "family": "qwen", "size": "2B-235B", "vision": True},
        {"label": "Qwen2.5 - multilingual", "model": "qwen2.5", "family": "qwen", "size": "0.5B-72B", "vision": False},
        {"label": "Qwen2.5 Coder - coding", "model": "qwen2.5-coder", "family": "qwen", "size": "0.5B-32B", "vision": False},
        {"label": "Qwen2.5 VL - vision", "model": "qwen2.5vl", "family": "qwen", "size": "3B-72B", "vision": True},
        {"label": "Gemma 3 - vision/general", "model": "gemma3", "family": "gemma", "size": "270M-27B", "vision": True},
        {"label": "Gemma 4 - latest Gemma", "model": "gemma4", "family": "gemma", "size": "E2B-31B", "vision": True},
        {"label": "Gemma 3n - efficient device model", "model": "gemma3n", "family": "gemma", "size": "E2B/E4B", "vision": False},
        {"label": "Gemma 2 - efficient general", "model": "gemma2", "family": "gemma", "size": "2B/9B/27B", "vision": False},
        {"label": "Mistral 7B - classic general", "model": "mistral", "family": "mistral", "size": "7B", "vision": False},
        {"label": "Mistral Nemo - 12B long context", "model": "mistral-nemo", "family": "mistral", "size": "12B", "vision": False},
        {"label": "Mistral Small 3.2 - vision/tools", "model": "mistral-small3.2", "family": "mistral", "size": "24B", "vision": True},
        {"label": "Phi 4 - compact quality", "model": "phi4", "family": "phi", "size": "14B", "vision": False},
        {"label": "Phi 4 Mini - small", "model": "phi4-mini", "family": "phi", "size": "3.8B", "vision": False},
        {"label": "Phi 4 Reasoning - math/reasoning", "model": "phi4-reasoning", "family": "phi", "size": "14B", "vision": False},
        {"label": "Phi 3 - lightweight", "model": "phi3", "family": "phi3", "size": "3.8B/14B", "vision": False},
        {"label": "GPT-OSS - open-weight reasoning", "model": "gpt-oss", "family": "default", "size": "20B/120B", "vision": False},
        {"label": "CodeLlama - code", "model": "codellama", "family": "codellama", "size": "7B-70B", "vision": False},
        {"label": "DeepSeek Coder - code", "model": "deepseek-coder", "family": "deepseek-coder", "size": "1.3B-33B", "vision": False},
        {"label": "QwQ - reasoning", "model": "qwq", "family": "qwen", "size": "32B", "vision": False},
        {"label": "TinyLlama - minimal hardware", "model": "tinyllama", "family": "llama2", "size": "1.1B", "vision": False},
        {"label": "LLaVA - vision", "model": "llava", "family": "vicuna", "size": "7B/13B/34B", "vision": True},
        {"label": "MiniCPM-V - vision", "model": "minicpm-v", "family": "default", "size": "8B", "vision": True},
    ],
}


def popular_model_presets(kind: str = "") -> list:
    """Return curated model download presets for the Download tab."""
    if not kind:
        return POPULAR_MODEL_PRESETS
    return list(POPULAR_MODEL_PRESETS.get(kind, []))


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
