# PhonoLab File Index

Complete dictionary of every source file in the PhonoLab project.
Each entry: **file path** → purpose, key classes/functions, key constants, dependencies.

---

## Python Core (PhonoLab/)

### `__init__.py`
- Package init, exports `__version__ = "0.1.0"`

### `__main__.py`
- Entry point: calls `main.main()` → `raise SystemExit(main())`

### `main.py`
- `main()` → imports `ui.kivy_app.run()`, prints error if Kivy missing
- GUI-only, no CLI mode

### `config.py`
- **`MobileConfig`** dataclass — all runtime settings
  - `ctx` = 2048, `threads` = min(4, cpu_count), `n_predict` = 384
  - `temperature` = 0.7, `top_p` = 0.9, `repeat_penalty` = 1.1
  - `max_prompt_chars` = 18000, `max_download_bytes` = 7 GB
  - `stream_timeout_seconds` = 180
  - `llama_cli_path`, `llama_server_path`, `active_model`, `hf_token`
  - `user_agent` = "PhonoLabMobile/1"
- Bounds: ctx [512, 32768], threads [1, 8], n_predict [32, 4096], temp [0.0, 2.0]
- `load_config()` / `save_config()` — JSON read/write

### `paths.py`
- `PHONOLAB_HOME` — platform-dependent root (Android/iOS/Desktop)
- Subdirs: CONFIG_DIR, MODELS_DIR, RUNTIME_DIR, DOWNLOADS_DIR, STATE_DIR, LOG_DIR
- Files: CONFIG_FILE, REGISTRY_FILE, SETUP_STATE_FILE, CHAT_HISTORY_FILE
- `mobile_target()` → "android" | "ios" | "desktop"
- `ensure_dirs()`, `atomic_write_text()`, `safe_child_path()`

### `registry.py`
- **`MobileModelConfig`** — per-model settings (path, name, repo, quant, ctx, threads, etc.)
- **`MobileModelRegistry`** — manages model_registry.json
  - `add()` — auto-sets ctx based on RAM (<6GB → 2048, else 4096)
  - `discover()` — scans MODELS_DIR for .gguf files
  - `remove()` — unregisters model

### `hardware.py`
- **`HardwareProfile`** — target, system, arch, cpu_threads, ram_total_mb, accelerator
- `profile_hardware()` — detect system profile
- `detect_accelerator()` — ios→metal, android→vulkan, darwin→metal, else→cpu
- `recommended_threads()` — mobile: max 4, desktop: max 8

### `safety.py`
- `SafetyError`, `ContextLimitError`, `UserNotice`
- `validate_repo_id()`, `validate_model_path()`, `guard_prompt()`
- `sampler_args()` — clamp all params to safe ranges
- `explain_error()` — convert raw error to UserNotice

### `small_models.py`
- **`SMALL_MODEL_CATALOG`** — 5 built-in models:
  1. smollm2-360m (2GB RAM, tier "tiny")
  2. qwen25-05b (3GB, "minimal")
  3. llama32-1b (4GB, "low")
  4. qwen25-15b (6GB, "balanced")
  5. tinyllama-11b (4GB, "fallback")
- Quant: Q4_K_M > Q4_0 > Q5_K_M > Q3_K_M
- `choose_candidate()` — pick best for device RAM

### `downloads.py`
- `fetch_hf_gguf_files()` — list .gguf from HuggingFace API
- `choose_gguf_file()` — score by quant, skip mmproj/projector/clip
- **`ResumableDownload`** — chunk=256KB, max_retries=3, pause/resume/abort
- `download_candidate_model()` — full pipeline, skip if same size exists, cleanup duplicates
- `_cleanup_duplicates()` — remove smaller .gguf in same directory

### `llama_cpp_setup.py`
- `LLAMA_CPP_SOURCE_URL` — GitHub master.zip
- `LlamaCppPlan` — target, source_dir, cli_path, ready, message
- `find_llama_cli()` — search config → runtime_bin_dir → PATH
- `pull_llama_cpp()` — download + extract source
- `register_llama_cli()` — register custom binary

---

## Python Engine (PhonoLab/engine/)

### `__init__.py`
- Exports `MobileLlamaCppEngine`

### `llama_cpp.py`
- **`MobileLlamaCppEngine`** — llama.cpp CLI subprocess wrapper
  - `load()` — validate model, find CLI, register in registry
  - `generate()` — run llama-cli, stream tokens via callback
  - `_command()` — build CLI args with sampler params
  - Timeout: `stream_timeout_seconds` (180s)

---

## Python Native (PhonoLab/native/)

### `mobile_core.py`
- Loads native C library (libphonolab_mobile_core.so/.dylib/.dll)
- `estimate_tokens()` — native or fallback (max(bytes/4, words))
- `prompt_fits_context()` — check if prompt + reserved <= ctx

### `mobile_core.c`
- `phonolab_estimate_tokens(text)` — count bytes and words
- `phonolab_context_fits(text, ctx, reserved)` — check fit
- Pure C, no dependencies

---

## Python UI (PhonoLab/ui/)

### `kivy_app.py`
- NativeLab-themed Kivy GUI
- Dark palette: bg0=#09090d, accent=#55C2A4, txt=#ededf5
- Sections: Status, MODEL (spinner), RUNTIME (Setup/Download/Load), CHAT
- All blocking work in background threads

---

## Python Tests (PhonoLab/tests/)

### `test_mobile_node.py`
- Path traversal guard, repo validation, context guard, quant selection, registry round-trip

### `test_android_project.py`
- Required files exist, manifest permissions, syncLlamaCpp task

---

## Android App (PhonoLab/android/)

### Application

#### `PhonoLabApp.kt`
- **Application subclass** — holds singletons that survive `recreate()`
- Singletons: `store`, `sessionManager`, `modelManager`, `runtime`
- `currentActivity` — tracked via ActivityLifecycleCallbacks
- `installCrashHandler()` — global UncaughtExceptionHandler
  - Distinguishes: IllegalStateException, IllegalArgumentException, NullPointerException, OutOfMemoryError
  - Fatal → AlertDialog with "Restart" / "Exit" buttons
- `showError(message, fatal)` — fatal=dialog, non-fatal=red ErrorBannerView
- `safeRunState(tag, operation, block)` — catches ISE/IAE, shows banner, returns null
- `safeRun(tag, block)` — catch-and-return-null helper

#### `MainActivity.kt`
- Entry point, drawer layout, toolbar, session sidebar
- **Implements `ChatFragment.Host`** — `onSessionChanged()` refreshes sidebar
- Uses `PhonoLabApp` singletons (not local instances)
- `currentFragmentTag` — tracks active fragment for restore after recreate
- `onSaveInstanceState` / `initializeApp(savedInstanceState)` — preserves navigation
- `onDestroy` — only kills runtime when `isFinishing` (not on recreate)
- Fragments: ChatFragment (default), ModelsFragment, DownloadsFragment, ApiFragment
- Sidebar: nav_models, nav_downloads, nav_api, nav_logs, nav_theme_toggle
- Session management: new, select, rename, delete, export
- StorageManager for SAF folder picker

### Runtime

#### `LlamaRuntime.kt`
- Runtime engine backed by llama-server (persistent HTTP)
- Constructor: `LlamaRuntime(context, store, storageManager?, modelManager?)`
- `_abort` flag (Volatile) — for graceful generation cancellation
- `load(model: File): String?` — returns null=success, non-null=error message
  - Looks up ModelConfig from ModelManager for per-model settings
  - Auto-unloads old model before loading new one
- `unload()` — sets abort, stops server, clears state
- `abort()` — sets abort flag without killing server
- `generate(prompt, onToken): String` — returns text or `[ERROR]`/`[ABORTED]` prefix
  - SSE streaming via `/v1/chat/completions`
  - Uses loadedConfig for temperature/maxTokens
  - Checks `_abort` flag each line
  - Catches EOFException separately
- `startServer()` — uses cfg.ctx, cfg.threads, cfg.maxTokens
- `cleanup()` — emergency reset all state
- `killAllLlamaProcesses()` — pkill/killall all llama processes

#### `LlamaCppManager.kt`
- Manages llama-server binary bundled in APK
- `System.loadLibrary("runner")` — loads JNI .so (guarded by try-catch, sets `nativeLoaded` flag)
- JNI methods: `nativeExec`, `nativeKill`, `nativeKillForcibly`, `nativeWaitPid`
- All JNI calls guarded by `if (!nativeLoaded) return` checks
- `findServer()` — look in nativeLibraryDir for libllama_server.so
- `isValidElf()` — check 0x7F ELF magic bytes
- `detectArch()` — primary ABI, isArm64, isEmulator
- `startServer()` — builds args/env, calls nativeExec. Returns -1 on failure (no throw)
- `killServer()`, `isRunning()`, `checkProcess()` — all wrapped in try-catch
- `downloadAndInstall()` — top-level try-catch wrapper, returns null on any failure
- `renameTo()` fallback — copy+delete if rename fails

#### `PhonoLabStore.kt`
- App storage backed by StorageManager
- Directories: root, runtimeDir, sourceDir, modelsDir, downloadsDir, stateDir, sessionsDir, configDir
- `safeChild()` — path traversal guard
- `modelFiles()` — list all .gguf files
- SharedPreferences: "phonolab_prefs"

#### `StorageManager.kt`
- SAF folder management
- `isUsingDefault()` / `markUsingDefault()` — persists "use default" choice
- `getBaseDir()` — never returns null, always resolves to valid path
- Subdir getters: `getLlamaServerDir()`, `getModelsDir()`, `getSessionsDir()`, etc.
- `loadPersistedUri()` — checks both SAF URI and default flag
- SharedPreferences: "phonolab_storage"

#### `SafeDownloader.kt`
- `maxDownloadBytes` = 7 GB
- `fetchGgufFiles(repo)` — list .gguf from HuggingFace API
- `chooseFile()` — score by quant preference
- `downloadModel()` — skip if same size exists, cleanup duplicates after download
- `cleanupDuplicates()` — remove smaller .gguf in same directory

#### `ModelCatalog.kt`
- 5 built-in models (same as Python SMALL_MODEL_CATALOG)
- `chooseForDevice(totalRamMb)` — pick best eligible

### Data

#### `data/ChatSession.kt`
- **`ChatMessage`** — role, content, timestamp (HH:mm), optional `imageBase64`
  - `toJson()` / `fromJson()` — includes imageBase64 field
- **`ChatSession`** — id, title, created, messages, logs
  - `addMessage(role, content, imageBase64?)` — auto-title from first user message (40 chars)
  - `addLog(entry)` — adds log with **MAX_LOGS=500** cap (trims oldest)
  - `buildMessages(systemPrompt, maxChars)` — OpenAI-compatible messages array
  - `buildRequestBody(systemPrompt, temperature, maxTokens, maxChars)` — full /v1/chat/completions body
  - `toJson()` / `fromJson()` — uses `optJSONObject()` for safety
  - Vision support: if `imageBase64` present, content becomes array with text + image_url

#### `data/SessionManager.kt`
- CRUD for chat sessions as JSON files in sessionsDir
- `loadAll()`, `save()`, `delete()`, `rename()`, `exportMarkdown()`

#### `data/ModelManager.kt`
- **`ModelConfig`** — path, name, repo, quant, ctx, threads, temperature, topP, topK, minP, repeatPenalty, maxTokens, seed
- Defaults: ctx=2048, threads=4, temp=0.7, topP=0.9, topK=40, repeatPenalty=1.1, maxTokens=384
- Registry stored in configDir/model_registry.json
- `syncDiscovery()` — scans modelsDir, deduplicates by canonicalPath
- `remove()` — deletes .gguf from disk + empty parent dirs

### Theme

#### `theme/ThemeManager.kt`
- Dark palette (NativeLab Studio) and Light palette (Cream & Sage)
- `color(context, resId)` — resolve XML color as int
- `hexColor(key)` — get palette color as hex string
- `palette()` — get active palette as Map
- `toggleTheme()` — switch dark↔light, calls `recreate()`
- SharedPreferences: "phonolab_theme"

### API Server

#### `api/ApiConfig.kt`
- host: "0.0.0.0", port: 8787, protocol: "both"
- `localApiKey` / `lanApiKey` — auto-generated UUID-based ("nl-" prefix)
- `detectLanIp()` — scan network interfaces

#### `api/PhonoLabApiServer.kt`
- HTTP server on configurable port
- **Endpoints**: GET /health, /runtime, /models, /capabilities
  - POST /chat/completions (OpenAI), /completions, /messages (Anthropic)
- `handleOpenAiChatStreaming()` — SSE streaming with `streamGenerateFn`
  - Socket writes guarded for client disconnect (IOException caught, `clientDisconnected` flag)
- Auth: Bearer token, separate keys for local vs LAN
- CORS: Access-Control-Allow-Origin: *
- **Body size limit**: MAX_BODY_BYTES = 1MB (returns 413 if exceeded)
- **Thread pool**: `newFixedThreadPool(4)` (bounded, not cached)
- **Executor shutdown**: `stop()` calls `executor.shutdownNow()`

### Math

#### `util/MathHelper.kt`
- Detects math in text for KaTeX rendering
- Patterns: DELIMITED_MATH, ENV_BLOCK, LATEX_CMD, EXPONENT, EXPONENT_BRACE, SUBSCRIPT, SUBSCRIPT_BRACE
- `hasMath(text)` — check if text contains math
- `wrapAutoMath(text)` — wrap detected math in `$...$` for KaTeX
- Uses lambda replacements (not backreferences) for Android ICU compatibility

### UI Fragments

#### `ui/ChatFragment.kt`
- Chat UI with RecyclerView, model picker, prompt input
- **`Host` interface** — `onSessionChanged()` decouples from MainActivity
- All singletons from `PhonoLabApp` (no sharedStore/sharedRuntime fields)
- `runOnUi {}` helper — `isAdded` lifecycle guard on all UI updates
- `timeFmt` — `ThreadLocal<SimpleDateFormat>` for thread safety
- `showModelPicker()` — AlertDialog with ✓/⬇ icons, themed layout
- `downloadAndLoad()` — green progress bar in chat area
- `loadModelFromPath()` — uses runtime.load() which returns error string
- `sendPrompt()` — auto-loads model via `findAutoLoadModel()`, handles [ABORTED]/[ERROR]
  - RAG context prepended if `pendingRagResult` exists
  - Image attachment noted in message text
- `showError()` — displays error as assistant message with ⚠️
- `showLogs()` — dialog with session logs (capped at 500)
- `stopGeneration()` — calls `runtime.abort()` (doesn't kill server)
- **Attachments**: `showAttachmentSheet()` → Image/Document picker
- **RAG**: `processDocumentWithRag()` — shows progress, stores `pendingRagResult`
- No auto-scroll during generation

#### `ui/ModelsFragment.kt`
- Model list with parameter editor (ctx, threads, temp, topP, topK, rep, maxTokens)
- Warnings: ctx > 16384, temp > 1.5
- **API Models section**: add/edit/delete API models (provider, name, baseUrl, apiKey)
- Uses `PhonoLabApp` singletons (not private instances)

#### `ui/DownloadsFragment.kt`
- Runtime install + model catalog download
- Shows device arch info
- Uses `PhonoLabApp` singletons (not private instances)
- `runOnUi {}` helper with `isAdded` lifecycle guard
- `onDestroyView()` → `worker.shutdown()` (no leak)

#### `ui/ApiFragment.kt`
- API server control panel with start/stop, protocol selector
- Shows local/LAN URLs and API keys
- Uses `PhonoLabApp` singletons (not private instances)
- Server lambdas capture `runtime`/`modelManager` at creation time (no stale refs)
- `runOnUi {}` helper, `updateLog()` checks `isAdded`

### Adapters

#### `adapter/ChatAdapter.kt`
- 2 view types: TYPE_USER, TYPE_AST
- `appendToLast()` — returns updated ChatMessage for session sync
- Math detection: checks `MathHelper.hasMath()`, renders in WebView with KaTeX
- `wrapAutoMath()` applied before WebView rendering
- Theme colors passed to WebView HTML

#### `adapter/SessionAdapter.kt`
- Sessions grouped by date header, search/filter, active highlighting

#### `adapter/ModelAdapter.kt`
- Model list with name, info, size, status icon

#### `adapter/CatalogAdapter.kt`
- Model catalog with download buttons

#### `adapter/ApiModelAdapter.kt`
- API model list with provider, model name, edit/delete buttons

### UI Components

#### `ui/ErrorBannerView.kt`
- Red exclamation banner for non-fatal errors
- Auto-dismisses after 5 seconds
- Close button, fade animation
- Added to activity decor view programmatically by `PhonoLabApp.showBanner()`

#### `ui/AttachmentBottomSheet.kt`
- BottomSheetDialogFragment with Image/Document options
- `AttachmentCallback` interface: `onImageSelected()`, `onDocumentSelected()`
- Dismisses after selection

### Utilities

#### `util/UiHelpers.kt`
- `calcPercent(done, total): Int` — download progress percentage helper
- Used by ChatFragment and DownloadsFragment

#### `RagProcessor.kt`
- Document RAG processing for attachments
- `RagResult` — chunks, totalChars, filename, mimeType
- `processDocument(uri, onProgress)` — extracts text from PDF/text/docx, chunks with sliding window (1500 chars, 200 overlap)
- `retrieveChunks(query, chunks, topK)` — keyword-based relevance scoring
- PDF: uses `PdfRenderer` (page count only, text extraction needs pdfbox-android)
- DOCX: reads `word/document.xml` from zip, strips XML tags
- Text: direct read

#### `ModelFamily.kt`
- Chat template detection for GGUF models
- `detectVisionModel(filename)` — returns `VisionModelInfo` (isVision, family, hasThink)
- `detectMmprojForModel(modelPath)` — finds matching mmproj file in same directory
- Template families: ChatML, Llama3, Gemma, Phi, Zephyr, Mistral, etc.
- Vision patterns: llava, pixtral, moondream, qwen-vl, llama-vision, mllama

### JNI

#### `app/src/main/cpp/runner.cpp`
- `nativeExec(binary, args, env)` — fork()+execve(), returns PID
- `nativeKill(pid)` — SIGTERM
- `nativeKillForcibly(pid)` — SIGKILL
- `nativeWaitPid(pid)` — waitpid with WNOHANG
- Env MUST contain LD_LIBRARY_PATH=nativeLibraryDir

#### `app/src/main/cpp/CMakeLists.txt`
- cmake_minimum_required 3.22.1, library "runner", links log + android

### Layouts

#### `res/layout/fragment_chat.xml`
- RecyclerView (chat) + download_bar (green progress) + generating_banner
- **rag_bar** — progress bar + status text for document processing
- **attachment_chip** — preview with clear button for attached files
- Input bar: btn_attach (paperclip) + btn_model_picker (TextView "Model ▾") + prompt_input + btn_send

#### `res/layout/bottom_sheet_attachment.xml`
- Two side-by-side buttons: Image (gallery icon) + Document (edit icon)

#### `res/layout/item_api_model.xml`
- API model list item: provider, model name, edit/delete buttons

#### `res/layout/dialog_api_model.xml`
- Add/edit API model dialog: provider, model name, base URL, API key fields

#### `res/layout/activity_main.xml`
- DrawerLayout with toolbar, sidebar (models, downloads, api, logs, theme toggle, new chat, search, sessions)

#### `res/layout/item_message_ast.xml`
- Assistant bubble with TextView + WebView (for math) + timestamp

#### `res/layout/ph_spinner_item.xml` / `ph_spinner_dropdown_item.xml`
- Themed spinner layouts with ph_txt/ph_surface colors

#### `res/layout/ph_nav_header.xml`
- NativeLab icon + "NativeLab" title + "PhonoLab · Local AI on mobile" subtitle

### Resources

#### `res/values/colors.xml` / `values-night/colors.xml`
- Full dark/light palette: ph_bg0..ph_bg3, ph_surface, ph_accent, ph_txt, ph_ok, ph_warn, ph_err, etc.

#### `res/values/themes.xml` / `values-night/themes.xml`
- Theme.PhonoLab (Material3), Theme.PhonoLab.Splash

#### `assets/katex_template.html`
- KaTeX HTML template for math rendering (loads from CDN)

### Build

#### `app/build.gradle.kts`
- namespace: org.nativelab.phonolab, compileSdk: 36, minSdk: 21, targetSdk: 35
- ndkVersion: 27.0.12077973, ABIs: arm64-v8a, armeabi-v7a
- CMake: src/main/cpp/CMakeLists.txt
- Dependencies: AndroidX, Material 3, JSON, DocumentFile, SplashScreen

#### `build.gradle.kts` (root)
- AGP 8.13.2, Kotlin 2.0.21, syncLlamaCpp task

#### `setup_binaries.sh`
- Downloads latest llama.cpp Android arm64 release
- Copies .so to jniLibs/arm64-v8a/, renames llama-server → libllama_server.so

#### `AndroidManifest.xml`
- `android:name=".PhonoLabApp"` — Application class
- Permissions: INTERNET, storage, notifications, foreground service
- `extractNativeLibs="true"` — required for JNI binary
- `networkSecurityConfig` — allows localhost HTTP
