# Models

Everything about the model side of NativeLab - registries, prompt templates, quantizations, roles, and API providers.

---

## Model registry

`ModelRegistry` (in `nativelab/Model/ModelRegistry.py`) maintains two sources of truth and merges them:

- Models auto-discovered in `MODELS_DIR` (`./localllm/*.gguf`).
- Models added manually via the GUI's Models tab or the CLI wizard.

Both sets are persisted to `custom_models.json` (paths) and `model_configs.json` (per-model parameters).

```python
from nativelab.Model.model_global import get_model_registry

reg = get_model_registry()
reg.add("/path/to/some.gguf")
reg.all_models()        # → [{"path", "name", "size_mb", "family", "quant", "role", …}]
reg.get_config(path)    # → ModelConfig dataclass
```

---

## Roles

Each model gets one role. Role drives which engine slot it lands in and influences prompt routing.

| Role            | Icon | Used for                                                     |
| --------------- | ---- | ------------------------------------------------------------ |
| `general`       | 💬   | Primary chat engine.                                         |
| `reasoning`     | 🧠   | Architectural insights in pipeline mode; final summarisation pass. |
| `summarization` | 📄   | Dedicated PDF summariser.                                    |
| `coding`        | 💻   | Receives prompts auto-detected as coding requests.           |
| `secondary`     | 🔀   | Auxiliary insight provider in multi-engine pipeline runs.    |

---

## Per-model parameters

Stored in `model_configs.json` per absolute path:

```python
threads        # CPU threads passed to llama.cpp
ctx            # context window size in tokens
temperature    # 0.0–2.0
top_p          # nucleus sampling
repeat_penalty # 1.0 = off
n_predict      # max tokens generated per call
family         # auto-detected family key (mistral, llama3, qwen, …)
```

Editable per-file from the Models tab in the GUI. Defaults come from `app_config.json`.

---

## Auto family detection

`detect_model_family()` matches filename substrings in priority order to pick the right `ModelFamily` template. `FAMILY_TEMPLATES` (in `nativelab/Model/templates.py`) ships with 20+ families - each holds the BOS/EOS, system slot, user/assistant prefixes/suffixes, and stop tokens for that family's chat format.

A few examples:

| Family            | User / Assistant template                                    | BOS / EOS                                                |
| ----------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| **DeepSeek**      | `User: {u}\n\nAssistant:`                                    | `<｜begin▁of▁sentence｜>` / `<｜end▁of▁sentence｜>`     |
| **DeepSeek-R1**   | adds `<think>\n` after assistant prefix                      | same as DeepSeek                                         |
| **Mistral / Mixtral** | `[INST] {u} [/INST]`                                     | `<s>` / `</s>`                                           |
| **LLaMA-2**       | `[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{u} [/INST]`          | `<s>` / `</s>`                                           |
| **LLaMA-3**       | `<\|start_header_id\|>user<\|end_header_id\|>\n\n{u}<\|eot_id\|>` | `<\|begin_of_text\|>`                              |
| **Phi / Phi-3**   | `<\|user\|>\n{u}<\|end\|>\n<\|assistant\|>\n`                | -                                                        |
| **Qwen / ChatML / Yi / Orca** | `<\|im_start\|>user\n{u}<\|im_end\|>\n<\|im_start\|>assistant\n` | -                                              |
| **Gemma**         | `<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n` | -                                                     |
| **Command-R**     | `<\|START_OF_TURN_TOKEN\|><\|USER_TOKEN\|>{u}<\|END_OF_TURN_TOKEN\|>…` | -                                                |

Full list - DeepSeek (+R1), Mistral, Mixtral, LLaMA-2, LLaMA-3, Phi, Phi-3, Phi-3.5, Qwen, ChatML, Gemma, CodeLlama, Falcon, Vicuna, OpenChat, Neural-Chat, Starling, Yi, Zephyr, Solar, Orca, Command-R.

The detected family is shown next to every model in the Models tab and in the CLI's `/status` output.

---

## Quantization detection

`detect_quant_type()` recognises every quant in current llama.cpp builds.

**imatrix importance quants** - `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ2_M`, `IQ3_XXS`, `IQ3_XS`, `IQ3_S`, `IQ3_M`, `IQ4_XS`, `IQ4_NL`.

**K-quants** - `Q2_K`, `Q3_K_S/M/L`, `Q4_K_S/M`, `Q5_K_S/M`, `Q6_K`.

**Legacy** - `Q4_0/1`, `Q5_0/1`, `Q8_0`.

**Float** - `F16`, `F32`, `BF16`.

Each quant maps to a quality tier with a color label:

| Tier             | Quants                | Label                          |
| ---------------- | --------------------- | ------------------------------ |
| 🟢 Full          | F32, F16, BF16, Q8, Q6 | "Full precision" / "Near-lossless" |
| 🟣 High quality  | Q5, IQ4               | "High quality"                 |
| 🟡 Balanced      | Q4, IQ3               | "Balanced"                     |
| 🔴 Compressed    | Q3, IQ2               | "Compressed"                   |
| 🔴 Very compressed | Q2, IQ1             | "Very compressed"              |

---

## Local and API backend support

In addition to local GGUFs, NativeLab can use running Ollama models, optional Hugging Face Transformers models, and two API formats. All of these appear in the same model selector and use the shared engine/status surface.

### Local backend refs

| Backend | Ref format | Notes |
| --- | --- | --- |
| llama.cpp GGUF | filesystem path | Existing `.gguf` behavior is unchanged. |
| Ollama | `ollama:<model-name>` | Requires an already running Ollama service at `http://127.0.0.1:11434`. |
| HF Transformers | `hf:<repo-id-or-local-dir>` | Install with `pip install -e ".[hf]"`; supports safetensors model dirs/repos with config/tokenizer metadata. |

The Models tab includes buttons for adding installed Ollama models and HF repo/local-directory refs. HF vision models use the Transformers image-text-to-text path when the model architecture is supported.

### Download tab support

The Download tab now has three model/runtime paths:

- **GGUF HuggingFace search** downloads one `.gguf` file into `localllm/` for llama.cpp.
- **HF Transformers snapshot** inspects a repo revision, selects runtime files, preserves repo subdirectories, downloads into `localllm/hf_transformers/<namespace>/<repo>/`, resumes with `.part` files, and registers the result as `hf:<local-folder>`.
- **Ollama model pull** connects to the configured Ollama host, lists installed models from `/api/tags`, streams `/api/pull` progress, and registers completed pulls as `ollama:<model>`.

The built-in **Popular** selectors are defined in `POPULAR_MODEL_PRESETS` in `nativelab/Model/templates.py`, grouped as `gguf`, `hf_transformers`, and `ollama`. They fill the relevant repo/model field but do not block custom IDs.

For private or gated repos, sign in from **Accounts > Hugging Face** with **Login with Hugging Face**. NativeLab uses its built-in public OAuth client ID, stores credentials locally at `localllm/cred/huggingface.json`, masks tokens in the UI/logs, and uses that token for GGUF search/download, HF snapshot downloads, and `hf:` model loading. Manual `hf_...` token paste remains available as an advanced fallback, and the older App Configuration `hf_token` field remains as a lower-priority fallback when no saved login exists. Some gated repos, including Meta Llama repos, also require accepting terms or waiting for approval on the Hugging Face repo page; a signed-in token will still receive HTTP 403 until that repo access is granted.

### HF and Ollama settings

Open the top-right Settings button, then **App Configuration**, to edit backend load behavior:

| Setting group | Fields |
| --- | --- |
| HF Transformers | download/cache directory, token, default revision, trust remote code, local-files-only, safetensors policy, torch dtype, device map, low CPU memory loading, attention implementation, max-memory map, quantization mode |
| Ollama | host URL and default `keep_alive` |

`LlamaEngine` reads these settings before loading `hf:` or `ollama:` refs, so chat, Labs, pipelines, integrations, and CLI-visible engine state stay consistent. HF 8-bit and 4-bit quantization modes require a compatible `bitsandbytes` install; leave quantization as `none` otherwise.

Ollama refs and pulls require the Ollama daemon to already be running. If NativeLab shows a connection-refused message, start the Ollama desktop app or run `ollama serve`, then retry with the configured host.

### Formats

| Format       | Endpoint                          | Auth header                |
| ------------ | --------------------------------- | -------------------------- |
| `openai`     | `/chat/completions`               | `Authorization: Bearer …`  |
| `anthropic`  | `/v1/messages`                    | `x-api-key: …`             |

Any self-hosted server that exposes an OpenAI-compatible `/chat/completions` (LM Studio, Ollama, vLLM, llama-cpp-python) works with the `openai` format.

### `ApiConfig` fields

```python
name             # display name in the UI
provider         # "OpenAI", "Anthropic", or your custom name
model_id         # e.g. "gpt-4o-mini", "claude-3-5-sonnet-20241022"
api_key          # bearer or x-api-key
base_url         # e.g. "https://api.openai.com/v1"
api_format       # "openai" | "anthropic"
max_tokens       # cap for the API
temperature      # 0.0–2.0
use_custom_prompt, system_prompt, user_prefix, user_suffix, assistant_prefix
```

Stored in `api_models.json` and managed via `ApiRegistry`.

### `ApiEngine`

`ApiEngine` mirrors `LlamaEngine`'s public surface (`load`, `create_worker`, `is_loaded`, `status_text`, `shutdown`), so pipelines, summarisation, and reference injection work the same regardless of which engine is active.

`LabEndpoints.active_engine()` returns the API engine when it's loaded, else the local one.

---

## Adding a new model family

Add an entry to `FAMILY_TEMPLATES` (`nativelab/Model/templates.py`) and a matching pattern in `detect_model_family()`. Patterns are checked in order - put more specific ones (e.g. `phi-3.5`) before more general ones (`phi`).

```python
FAMILY_TEMPLATES["myfamily"] = ModelFamily(
    name="MyFamily",
    family="myfamily",
    user_prefix="<|user|>\n",
    user_suffix="<|end|>\n<|assistant|>\n",
    assistant_prefix="",
    assistant_suffix="<|end|>\n",
    bos="",
    eos="<|end|>",
    stop_tokens=["<|end|>", "<|user|>"],
)
```

Then add to the patterns list:

```python
("myfamily", FAMILY_TEMPLATES["myfamily"]),
```

The detection picks up the new family on next launch.
