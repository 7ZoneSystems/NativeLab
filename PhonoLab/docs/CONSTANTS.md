# PhonoLab Constants & Limits

All hardcoded values, limits, paths, colors, and magic numbers.

---

## Runtime Defaults

| Parameter | Default | Min | Max | Notes |
|-----------|---------|-----|-----|-------|
| `ctx` | 2048 | 512 | 32768 | <6GB RAM→2048, ≥6GB→4096 |
| `threads` | min(4, cpu) | 1 | 8 | Mobile capped at 4 |
| `n_predict` / `maxTokens` | 384 | 32 | 4096 | |
| `temperature` | 0.7 | 0.0 | 2.0 | |
| `top_p` | 0.9 | 0.05 | 1.0 | |
| `top_k` | 40 | 0 | — | Android only |
| `min_p` | 0.0 | 0.0 | — | Android only |
| `repeat_penalty` | 1.1 | 0.8 | 2.0 | |
| `seed` | -1 | -1 | — | -1 = random |
| `max_prompt_chars` | 18000 | 1000 | 300000 | Python safety |
| `max_download_bytes` | 7 GB | 50 MB | 50 GB | |
| `stream_timeout_seconds` | 180 | 10 | 1200 | |
| `user_agent` | "PhonoLabMobile/1" | | | |
| `maxChars` (buildPrompt) | 8000 | | | ChatSession.buildMessages |

---

## Model Catalog

| Key | Label | Repo | Min RAM | Context | Tier |
|-----|-------|------|---------|---------|------|
| smollm2-360m | SmolLM2 360M Instruct | bartowski/SmolLM2-360M-Instruct-GGUF | 2048 MB | 2048 | tiny |
| qwen25-05b | Qwen2.5 0.5B Instruct | bartowski/Qwen2.5-0.5B-Instruct-GGUF | 3072 MB | 2048 | minimal |
| llama32-1b | Llama 3.2 1B Instruct | bartowski/Llama-3.2-1B-Instruct-GGUF | 4096 MB | 2048 | low |
| qwen25-15b | Qwen2.5 1.5B Instruct | bartowski/Qwen2.5-1.5B-Instruct-GGUF | 6144 MB | 4096 | balanced |
| tinyllama-11b | TinyLlama 1.1B Chat | TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF | 4096 MB | 2048 | fallback |

**Quant preference:** Q4_K_M > Q4_0 > Q5_K_M > Q3_K_M

---

## File Paths (Python)

| Path | Android | iOS | Desktop |
|------|---------|-----|---------|
| PHONOLAB_HOME | app_storage/PhonoLab | ~/Documents/PhonoLab | ./PhonoLab/data |
| CONFIG_DIR | {home}/config | | |
| MODELS_DIR | {home}/models | | |
| RUNTIME_DIR | {home}/runtime | | |
| DOWNLOADS_DIR | {home}/downloads | | |
| STATE_DIR | {home}/state | | |

**Override:** `PHONOLAB_HOME` env var.

### Config Files

| File | Path |
|------|------|
| App config | CONFIG_DIR/phonolab_config.json |
| Model registry | CONFIG_DIR/model_registry.json |
| Setup state | STATE_DIR/llama_cpp_setup.json |

---

## Android Storage

```
<PhonoLabBase>/
  llama-server/    ← binary + .so files
  models/          ← downloaded GGUF models
  sessions/        ← chat history JSON
  config/          ← app config (model_registry.json)
  downloads/       ← temp downloads
  state/           ← state files
```

**Base dir resolution:** SAF URI → real path → `PhonoLab/` subdir. Fallback: `context.filesDir/PhonoLab`.

**SharedPreferences:**
| Name | Owner | Keys |
|------|-------|------|
| phonolab_prefs | PhonoLabStore | has_launched, binary_installed |
| phonolab_storage | StorageManager | folder_uri, use_default |
| phonolab_theme | ThemeManager | theme ("dark"/"light") |
| phonolab_api_server | ApiConfig | host, port, protocol, local_api_key, lan_api_key |
| llama_cpp_manager | LlamaCppManager | — |

---

## Network & API

| Constant | Value | Location |
|----------|-------|----------|
| Default API port | 8787 | ApiConfig.kt |
| Default API host | 0.0.0.0 | ApiConfig.kt |
| Protocol | "both" | ApiConfig.kt |
| API key format | "nl-" + UUID(32) | ApiConfig.kt |
| Server connect timeout | 5,000 ms | LlamaRuntime.kt |
| Server read timeout | 300,000 ms | LlamaRuntime.kt |
| Health check timeout | 2,000 ms | LlamaRuntime.kt |
| Server start timeout | 60,000 ms | LlamaRuntime.kt |
| API max body size | 1 MB (1,048,576 bytes) | PhonoLabApiServer.kt |
| API thread pool | 4 (fixed) | PhonoLabApiServer.kt |
| HuggingFace API timeout | 45,000 ms | SafeDownloader.kt |
| Download max size | 7 GB | SafeDownloader.kt |
| Download chunk size | 8 KB | SafeDownloader.kt |
| Session log cap | 500 entries | ChatSession.kt |
| RAG chunk size | 1,500 chars | RagProcessor.kt |
| RAG chunk overlap | 200 chars | RagProcessor.kt |
| RAG top-K chunks | 3 | RagProcessor.kt |
| RAG max text extraction | 50,000 chars | RagProcessor.kt |
| Error banner duration | 5,000 ms | ErrorBannerView.kt |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health, /v1/health | Server status |
| GET | /runtime | Runtime info |
| GET | /models, /v1/models | List models (OpenAI format) |
| GET | /capabilities | Full capabilities |
| POST | /chat/completions, /v1/chat/completions | OpenAI chat (supports SSE streaming) |
| POST | /completions, /v1/completions | OpenAI completion |
| POST | /messages, /v1/messages | Anthropic messages |

### llama-server Args (Android)

```
llama-server -m <model> --port <port> -t <threads> --ctx-size <ctx> -n <nPredict> --host 127.0.0.1
```

### llama-cli Args (Python)

```
llama-cli -m <model> -t <threads> --ctx-size <ctx> -n <n_predict> --temp <temp> --top-p <top_p> --repeat-penalty <repeat_penalty> --no-display-prompt --no-escape -p <prompt>
```

---

## Chat Template / Prompt Format

### Android (ChatSession.buildRequestBody)
Sends OpenAI-compatible messages array to `/v1/chat/completions` with `stream: true`. Server applies model's chat template.

### API Server (messagesToPrompt)
```
ROLE: content

ROLE: content

ASSISTANT: 
```

---

## Theme Colors

### Dark (NativeLab Studio)

| Key | Hex | Usage |
|-----|-----|-------|
| bg0 | #09090d | Window background |
| bg1 | #0f0f15 | Deep background |
| bg2 | #141420 | Input background |
| surface | #1e1e2e | Card background |
| surface2 | #252538 | Button/input bg |
| accent | #55C2A4 | Primary accent (teal) |
| accentDim | #1a3d33 | Dimmed accent |
| txt | #ededf5 | Primary text |
| txt2 | #7a7a9a | Secondary text |
| txt3 | #48485e | Disabled/hint text |
| ok | #1cb88a | Success |
| warn | #e8971a | Warning |
| err | #e84848 | Error |
| bubbleUser | #0e0c26 | User chat bubble |
| bubbleAst | #0c0c14 | Assistant chat bubble |

### Light (Cream & Sage)

| Key | Hex |
|-----|-----|
| bg0 | #fdf6f0 |
| surface | #f2e2d4 |
| accent | #55C2A4 |
| txt | #1a0f0a |
| ok | #15803d |
| warn | #b45309 |
| err | #b91c1c |

---

## Android Build

| Setting | Value |
|---------|-------|
| compileSdk | 36 |
| minSdk | 21 |
| targetSdk | 35 |
| ndkVersion | 27.0.12077973 |
| ABIs | arm64-v8a, armeabi-v7a |
| Java/Kotlin | 17 |
| AGP | 8.13.2 |
| Kotlin | 2.0.21 |
| CMake | 3.22.1 |
| applicationId | org.nativelab.phonolab |
| versionName | 0.1.0 |

---

## Math Detection Patterns (MathHelper)

| Pattern | Matches |
|---------|---------|
| DELIMITED_MATH | `$$...$$`, `$...$`, `\(...\)`, `\[...\]` |
| ENV_BLOCK | `\begin{equation}...\end{equation}`, `\begin{align}...` |
| LATEX_CMD | `\frac`, `\sqrt`, `\sin`, `\int`, `\sum`, `\alpha`, etc. |
| EXPONENT | `x^2`, `a^b` |
| EXPONENT_BRACE | `x^{n+1}`, `a^{i\pi}` |
| SUBSCRIPT | `a_n`, `x_i` |
| SUBSCRIPT_BRACE | `x_{i}`, `a_{n+1}` |

**Replacement:** Uses lambda `{ m -> "$" + m.value + "$" }` (not backreferences) for Android ICU compatibility.

---

## Crash Handling

| Layer | Component | What it does |
|-------|-----------|--------------|
| Global | `PhonoLabApp.installCrashHandler()` | UncaughtExceptionHandler: log, kill processes, restart dialog |
| Global | `PhonoLabApp.showError(fatal=true)` | AlertDialog with "Restart" / "Exit" |
| Global | `PhonoLabApp.showError(fatal=false)` | Red ErrorBannerView, auto-dismiss 5s |
| Global | `PhonoLabApp.safeRunState()` | Catches ISE/IAE, shows banner, returns null |
| Runtime | `LlamaRuntime.load()` | Returns String? instead of throwing |
| Runtime | `LlamaRuntime.generate()` | Returns [ERROR]/[ABORTED] prefix instead of throwing |
| Runtime | `LlamaRuntime._abort` | Volatile flag checked each SSE line |
| Runtime | `LlamaRuntime.startServer()` | Returns String? error (no throw) |
| JNI | `LlamaCppManager.nativeLoaded` | Flag guards all JNI calls |
| JNI | `LlamaCppManager.startServer()` | Returns -1 on failure (no throw) |
| JNI | `LlamaCppManager.downloadAndInstall()` | Top-level try-catch, returns null |
| Download | `SafeDownloader` | renameTo() fallback to copy+delete |
| API | `PhonoLabApiServer` | 1MB body limit, socket write guards, fixed pool |
| Fragment | `ChatFragment.runOnUi{}` | isAdded lifecycle guard |
| Fragment | `ChatFragment.Host` | Interface decouples from MainActivity |
| Data | `ChatSession.fromJson()` | optJSONObject() for safety |
| Data | `ChatSession.addLog()` | MAX_LOGS=500 cap |

---

## URLs

| URL | Usage |
|-----|-------|
| https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.zip | llama.cpp source |
| https://huggingface.co/api/models/{repo} | HF model API |
| https://huggingface.co/{repo}/resolve/main/{file} | HF file download |
| http://127.0.0.1:{port}/v1/chat/completions | llama-server chat |
| http://127.0.0.1:{port}/health | llama-server health |
